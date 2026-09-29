# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step config contracts -- what the compiler puts in each step's ``config``.

This module is the interface freeze. The compiler writes these; the three step binaries read
them. Freezing it here means the step binaries can be written without waiting for the submit
path, which is the single highest-leverage thing in the plan: those two workstreams share a data
structure and nothing else.

**How a step actually receives this.** The jobs controller serialises ``HelixJobStepSpec.config``
to a ConfigMap, mounts it, and points ``NEMO_JOB_STEP_CONFIG_FILE_PATH`` at the file. In the pod,
``nhx.common.jobs.config.get_task_config()`` reads that path and returns the dict. So a step does
``FetchStepConfig.model_validate(get_task_config())`` and nothing else.

**Read these by what each one does not carry.**

- ``fetch`` is told what to download. It is not told any registry, tag, or credential.
- ``supervise`` is told how to build. It is **not told where anything will be pushed** -- no
  registry, no tags, no secret name. The sandbox does not push, so the step orchestrating it has
  no reason to know where the trusted step intends to write, and not telling it means a
  compromise of it learns nothing about the destination.
- ``push`` is told where to publish, and where the credential broker is. It is not told anything
  about a fileset, a context, or a Dockerfile -- it publishes bytes it did not produce and cannot
  reproduce.

**No config carries a path.** Configs name things -- a fileset, an image -- and every step
derives where those things live from one :class:`WorkLayout`.

**No step config carries a credential, or the name of one.** ``push`` is told where the
credential broker is, and gets a token from it for this job's destinations alone, minutes before
it uses it. No step can read a standing credential or a key through its identity (Requirement 7):
a Secret a step's ServiceAccount could read would be readable by anyone able to submit a raw Jobs
job under that step's profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator


class ContextSource(BaseModel):
    """One build context: a whole fileset, or a subtree of one.

    ``fileset`` is always **qualified**, ``<workspace>/<name>``: the compiler qualifies a
    request's own-workspace names. One spelling per fileset is what keeps two filesets from
    landing one inside the other on the work volume -- ``foo`` and ``foo/bar`` would, and then
    one context would hold the other's files.
    """

    fileset: str
    context_path: str | None = None

    @field_validator("fileset")
    @classmethod
    def _fileset_is_qualified(cls, value: str) -> str:
        workspace, _, name = value.partition("/")
        if not workspace or not name or "/" in name:
            raise ValueError(f"fileset must be qualified as <workspace>/<name>, got {value!r}")
        return value


# ---------------------------------------------------------------------------
# The work volume
# ---------------------------------------------------------------------------


def _relative(part: str) -> PurePosixPath:
    """A caller-supplied path component, refused if it could leave the directory it is joined to.

    ``PurePosixPath("a") / "/etc"`` is ``/etc``: an absolute component does not nest, it
    replaces. So a fileset name or ``context_path`` that is absolute, or climbs with ``..``,
    would point a step outside this job's directory.
    """
    path = PurePosixPath(part)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{part!r} is not a relative path inside the work volume")
    return path


@dataclass(frozen=True, slots=True)
class WorkLayout:
    """Where everything lives in one job's slice of the work volume. The only definition of it.

    Rooted wherever the caller sees that slice:

    - ``fetch`` and ``push`` mount the slice, and root it at their mount path.
    - ``supervise`` mounts nothing, but writes the sandbox's ``subPath``\\ s, so it roots it at
      the slice's path *within* the volume.
    - The sandbox sees the same layout again under its own root, with only two parts of it
      mounted: its own context, read-only, and the output directory.

    So what ``fetch`` writes, what the sandbox mounts and builds, and what ``push`` reads are one
    function of the same names, evaluated under different roots. They cannot disagree.
    """

    root: PurePosixPath

    def fileset(self, name: str) -> PurePosixPath:
        """Where ``fetch`` copies a fileset, with its entry paths unchanged.

        ``context/<workspace>/<name>``: every fileset directory is exactly two levels down, so
        no fileset's directory can be inside another's.
        """
        path = _relative(name)
        if len(path.parts) != 2:
            raise ValueError(f"{name!r} is not a qualified fileset reference, <workspace>/<name>")
        return self.root / "context" / path

    def context(self, source: ContextSource) -> PurePosixPath:
        """A build context: its fileset's directory, or a subtree inside it.

        Inside, not beside: that is what lets one whole-fileset download serve every subtree of
        the same fileset.
        """
        path = self.fileset(source.fileset)
        return path / _relative(source.context_path) if source.context_path else path

    def context_hash_file(self, source: ContextSource) -> PurePosixPath:
        """Outside ``context/`` altogether, where no build context can contain it.

        Beside the context is not far enough: a fileset's subtree ``app`` has its hash at
        ``app.nhx-context-hash``, and that file would sit inside the context of the whole
        fileset -- or of a sibling spec that builds from the fileset root.
        """
        path = self.root / "hashes" / _relative(source.fileset)
        if source.context_path:
            path = path / _relative(source.context_path)
        return path.with_name(f"{path.name}.nhx-context-hash")

    @property
    def outputs(self) -> PurePosixPath:
        """Where the sandbox writes OCI layouts and ``push`` reads them."""
        return self.root / "out"

    def output(self, image: str) -> PurePosixPath:
        """One image's OCI layout. ``image`` is the ``ContainerImage`` row name."""
        return self.outputs / _relative(image)


# ---------------------------------------------------------------------------
# 1. fetch
# ---------------------------------------------------------------------------


class FetchStepConfig(BaseModel):
    """Trusted. Holds a Files client. Holds no registry credential. Runs no caller code."""

    sources: list[ContextSource] = Field(
        min_length=1,
        description=(
            "Deduplicated across the set: two specs sharing a source cause one download, and a "
            "whole fileset absorbs requests for its own subtrees."
        ),
    )


# ---------------------------------------------------------------------------
# 2. supervise  (and the sandbox it creates, which is not a step)
# ---------------------------------------------------------------------------


class SandboxSpec(BaseModel):
    """How to compose the untrusted pod.

    Every field here comes from operator config and **none of it from the request**. A caller
    picks a Dockerfile and a fileset; it does not pick egress, the executor image, the runtime
    class, or the service account. That is Requirement 2 expressed as a type.
    """

    image: str = Field(description="The kaniko executor image.")
    namespace: str
    work_pvc: str = Field(description="Claim the sandbox mounts for context and output.")
    service_account: str = Field(
        default="",
        description="Empty, always. The sandbox runs with automountServiceAccountToken: false.",
    )
    runtime_class: str | None = Field(
        default=None,
        description=(
            "`gvisor` on a cluster that offers it, unset where none exists. Sandboxed isolation "
            "is a cluster fact, so the operator states it rather than the design assuming it."
        ),
    )
    registry_mirror: str | None = Field(
        default=None,
        description=(
            "Pull-through mirror for every FROM. Unset in the PoC -- the mirror is `M2-1` and "
            "not built. Wired here so the flag path exists and the day it lands is config, not "
            "code. While it is unset, base images resolve upstream and Requirement 9 is unmet."
        ),
    )
    node_selector: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "PoC-only in practice: while the work volume is ReadWriteOnce, every build pod must "
            "land on one node. Production wants RWX and an empty selector."
        ),
    )
    dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "The sandbox uses `dnsPolicy: None` with these, NOT cluster DNS. Measured reason: "
            "GKE's NodeLocal DNSCache answers on a LINK-LOCAL address, so the egress denial that "
            "closes the metadata server closes cluster DNS with it -- and allowing kube-dns back "
            "reopens the Pod CIDR, because that allow is by construction an exception to the "
            "policy's `except` list. A sandbox needs to resolve pypi.org, not "
            "kubernetes.default.svc, so pointing it at public resolvers is strictly more closed."
        ),
    )
    cpu: str = "2"
    memory: str = "8Gi"


class SandboxImage(BaseModel):
    """One kaniko invocation inside a group.

    No tags and no registry: see the module docstring.
    """

    image: str = Field(description="`ContainerImage.name` -- the row this invocation is for.")
    platform: str
    dockerfile: str = Field(description="Dockerfile path, relative to the group's context.")


class SandboxGroup(BaseModel):
    """One sandbox pod. All images here share one source, so they can share one context mount.

    Where that context and the outputs are mounted is ``supervise``'s to decide, from the
    :class:`WorkLayout`: it owns the pod. The compiler could not write the ``subPath``\\ s anyway
    -- they include the job id, and the job does not exist yet when the compiler runs.
    """

    source: ContextSource
    images: list[SandboxImage] = Field(min_length=1)


class SuperviseStepConfig(BaseModel):
    """Trusted control plane for an untrusted pod. Holds NO credential of any kind.

    It does not mount the work volume, so it cannot read a context even by mistake.

    Images inside a group are built **sequentially**, one kaniko invocation each with
    ``--cleanup``. Not a throughput choice: they share one container root and measurably cannot
    overlap. Groups are separate pods and so COULD run in parallel, but they are built one after
    another, each under a one-hour cap -- a known gap, not a property.
    """

    sandbox: SandboxSpec
    groups: list[SandboxGroup] = Field(min_length=1)


# ---------------------------------------------------------------------------
# 3. push
# ---------------------------------------------------------------------------


class PushImage(BaseModel):
    """One OCI layout to publish. Its path is :meth:`WorkLayout.output` of ``image``."""

    image: str = Field(
        description=(
            "`ContainerImage.name` -- the row this will satisfy. It is also all the step names when "
            "it asks the broker to sign: the broker reads the row and resolves the system tag itself."
        )
    )
    system_ref: str = Field(
        description=(
            "`<registry>/<repository>:<system tag>`. Always pushed: it is what the broker resolves "
            "and signs, so a build that did not push it could never complete."
        )
    )
    caller_ref: str | None = Field(
        default=None,
        description="`<registry>/<repository>:<output.tag>`, when the spec named an output. Pushed first.",
    )

    @property
    def refs(self) -> list[str]:
        """Every reference to push, the system tag last: the one the broker signs goes up last."""
        return [*([self.caller_ref] if self.caller_ref else []), self.system_ref]


class PushStepConfig(BaseModel):
    """Trusted. Holds no credential and no key of its own. Runs no caller code.

    Everything it publishes with, it gets from the credential broker (``broker``) for this job
    alone: a registry token scoped to this job's destinations, and a signature it asks for by
    naming a row. It proves which job it belongs to with its pod's own ServiceAccount token, which
    opens nothing by itself -- ``nhx-build-push`` holds no RBAC.

    It publishes bytes it did not produce. That is the point, and it is also why this step must
    treat the OCI layout as hostile input: it is the only thing standing between a sandbox's
    output and the registry.
    """

    images: list[PushImage] = Field(min_length=1)
    registry: str = Field(description="The deployment's registry host. Every reference in `images` is on it.")
    broker: str = Field(description="URL of the credential broker: where this step gets credentials and signatures.")
    insecure: bool = Field(
        default=False,
        description=(
            "Push over plain HTTP. Dev registries only -- it means the bytes and the token cross the "
            "network unprotected, and the digest the registry reports is unauthenticated."
        ),
    )

    @model_validator(mode="after")
    def _every_ref_is_on_the_registry(self) -> PushStepConfig:
        """The broker grants tokens for ``registry`` only, so a reference elsewhere would push anonymously."""
        for image in self.images:
            for ref in image.refs:
                if not ref.startswith(f"{self.registry}/"):
                    raise ValueError(f"{ref} is not on {self.registry}")
        return self

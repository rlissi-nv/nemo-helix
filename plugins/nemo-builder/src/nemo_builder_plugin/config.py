# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder.

Everything about *how* a build runs lives here rather than in the request. A caller names a
fileset, a Dockerfile and a destination; it does not name the executor image, the namespace, the
service accounts, the egress posture, or whether its image gets signed. That split is
Requirement 2, and putting it in a config class rather than in the request schema is how it is
enforced instead of merely documented.

**Unconfigured means the submit fails, not that the build fails.** A deployment missing its
registry, its credential broker or its trust root should be rejected at submit with a clear
error, not produce a job that dies in a pod twenty minutes later. The fields that cannot have a
safe default are therefore ``None`` here and checked by the backend before anything is written.

**The credential broker reads this same section** (``builder:``), so the two cannot disagree
about the registry, the prefix or the push step's identity -- a prefix that differed between them
would silently mis-bound every grant.

Env prefix is ``NEMO_BUILDER_`` -- note that plugin ``NemoConfig`` uses ``NEMO_``, while core
services use ``NHX_``. Some prose in the repo says ``NHX_`` for plugins; the code is the
authority and it computes ``NEMO_<SAFE_NAME>_``.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from nemo_builder_plugin.identity import ImageIdentityError, validate_registry_host, validate_repository
from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, field_validator

#: A runtime name, as `BuildSet.runtime` accepts it.
_RUNTIME_NAME = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
#: A Dockerfile instruction that starts a new stage. See `RuntimeLayer`.
_FROM_LINE = re.compile(r"^\s*FROM\s", re.IGNORECASE | re.MULTILINE)


class RuntimeLayer(BaseModel):
    """One runtime's contract, as Dockerfile text appended to every image built for it.

    A runtime -- the evaluator's Harbor sandbox, a deployed agent -- runs images under rules an
    image built for Docker Desktop does not meet: a non-root user, a read-only root filesystem,
    tools its harness expects. This closes the gap in the build, once per runtime, instead of in
    every Dockerfile. It belongs to a runtime rather than a cluster: one cluster can host several,
    and Astra already does.
    """

    version: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Bumped whenever `dockerfile` changes. Recorded on every row as `<name>@<version>`.",
    )
    dockerfile: str = Field(
        min_length=1,
        description=(
            "Instructions appended after the image's own. No `FROM`: a new stage would discard "
            "the image it is meant to adapt, so the result would not be derived from it at all.\n\n"
            "Prefer pinned versions, or tools copied from a pinned image, over installing from the "
            "network: an unpinned install makes a derived image a function of the day it was built."
        ),
    )

    @field_validator("dockerfile")
    @classmethod
    def _no_new_stage(cls, value: str) -> str:
        if _FROM_LINE.search(value):
            raise ValueError("a runtime layer must not contain FROM; it would start a new stage")
        return value


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Helix."

    # --- The backend -----------------------------------------------------

    backend: Literal["execution"] = Field(
        default="execution",
        description=(
            "The deployment's one backend (Requirement 4), chosen when the platform is deployed. "
            "No request selects it, and none can change it: a platform that must produce images "
            "for two trust domains runs two deployments.\n\n"
            "`execution` builds in this cluster, in a sandbox that holds no credential, and is the "
            "only one implemented. Managed backends -- Cloud Build, ACES -- are values this field "
            "will gain."
        ),
    )

    # --- The kill switch -------------------------------------------------

    execution_enabled: bool = Field(
        default=True,
        description=(
            "When False, POST /builds refuses with a clear error and no job is created. The "
            "REQUEST SURFACE stays up -- reads still work, rows are still readable -- so "
            "disabling builds does not look like an outage to a consumer that is only resolving "
            "digests. An operator turns this off to stop running untrusted code without "
            "rolling back a deployment."
        ),
    )

    # --- Where builds run ------------------------------------------------

    namespace: str = Field(
        default="nhx-builds",
        description="The `baseline`-enforcing namespace. See plugins/nemo-builder/deploy/.",
    )
    work_pvc: str = Field(
        default="nhx-build-work",
        description="Work volume claim. fetch writes it, the sandbox reads and writes it, push reads it.",
    )
    sandbox_image: str = Field(
        default="gcr.io/kaniko-project/executor:debug",
        description=(
            "The kaniko executor. `:debug` rather than `:latest` because it ships a shell, and "
            "the shell is what sequences a group's builds inside one sandbox -- the plain image "
            "has no way to run two invocations.\n\n"
            "RFC 001 wants a pinned first-party fork here (`M2-3`): "
            "upstream kaniko was archived by Google on 2025-06-03, so this is a standing "
            "third-party dependency whose advisory obligation currently has no owner."
        ),
    )
    runtime_class: str | None = Field(
        default=None,
        description=(
            "`gvisor` where the cluster offers a sandboxed runtime. Unset by default: where gVisor "
            "nodes are tainted and in separate pools, as on GKE, that fights the single-node "
            "pinning the ReadWriteOnce work volume requires. Measured to work with kaniko, so "
            "this is a scheduling constraint rather than a capability one."
        ),
    )
    registry_mirror: str | None = Field(
        default=None,
        description=(
            "Pull-through mirror every FROM resolves against. Unset: the mirror is `M2-1` and "
            "not built. While it is unset Requirement 9 is unmet -- base images resolve upstream, "
            "and nothing at the network bounds what a Dockerfile may pull FROM."
        ),
    )
    node_selector: dict[str, str] = Field(
        default_factory=lambda: {"nhx.nvidia.com/build-node": "true"},
        description=(
            "Pins every build pod to one node. Required only while the work volume is "
            "ReadWriteOnce; see deploy/40-work-volume.yaml. Empty it once the work volume is RWX."
        ),
    )
    sandbox_cpu: str = Field(
        default="2",
        description=(
            "CPU request for the sandbox. Sizing is a property of the DEPLOYMENT, not of the "
            "request -- a caller cannot make its build bigger by asking.\n\n"
            "Worth knowing why this is settable at all: `CPUKubernetesJobBackend.schedule` "
            "forwards only a step's `container`, so `executor.resources` reaches nothing and a "
            "Jobs step is sized by its profile. The sandbox escapes that because the build step "
            "writes its pod spec directly -- a small unplanned benefit of this architecture."
        ),
    )
    sandbox_memory: str = Field(default="8Gi", description="Memory request for the sandbox. See `sandbox_cpu`.")
    sandbox_dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "The sandbox resolves against these, not cluster DNS. See deploy/30-networkpolicy.yaml: "
            "where cluster DNS is link-local, re-allowing it reopens the Pod CIDR."
        ),
    )

    # --- Execution profiles ----------------------------------------------

    fetch_profile: str = Field(default="build-fetch")
    control_profile: str = Field(default="build-control")
    push_profile: str = Field(default="build-push")
    push_service_account: str = Field(
        default="nhx-build-push",
        description=(
            "The ServiceAccount `push_profile` runs as, which must match that profile's definition "
            "in Jobs (deploy/10-serviceaccounts.yaml). The credential broker serves a push pod only "
            "under exactly this identity. It holds no RBAC at all: everything it uses, it gets "
            "from the broker."
        ),
    )

    # --- Publishing -------------------------------------------------------

    registry: str | None = Field(
        default=None,
        description=(
            "The registry HOST every image is published to -- `us-central1-docker.pkg.dev`, not "
            "`us-central1-docker.pkg.dev/project/repo`. One per deployment, and a caller cannot "
            "name another: whatever runs a built image has to pull it, and one registry means one "
            "pull credential for every workload that does.\n\n"
            "A host and a repository path are different things, and conflating them produces a "
            "reference that looks right in a log and builds the URL "
            "`https://host/project/repo/v2/...` when anything tries to resolve it. Validated below "
            "rather than trusted.\n\n"
            "No safe default; unset fails the submit."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path every image is published under, ahead of the submitting workspace: images land "
            "at `<repository_prefix>/<workspace>/<output.repository>`, or "
            "`<repository_prefix>/<workspace>/<set>/<spec>` for a spec that names no output. This "
            "is where a registry's project/repo path belongs -- for GAR, "
            "`<project>/<artifact-repo>`. Kept separate from `registry` so "
            "`ContainerImage.registry` stays a host that a registry client can actually connect to."
        ),
    )

    @field_validator("registry")
    @classmethod
    def _registry_is_a_host(cls, value: str | None) -> str | None:
        """A registry is a host, optionally with a port. It is not a path.

        Checked here because the failure is otherwise silent until the broker or a push step
        builds a URL, and at that point the error reads as a registry problem rather than a
        config one.
        """
        if value is None:
            return None
        if "/" in value:
            raise ValueError(
                f"registry must be a host without a path, got {value!r}. "
                "Put the project/repository path in `repository_prefix` instead."
            )
        try:
            # Lowercased here because every reference compared against it is normalized, and a
            # host is case-insensitive: `Registry.example.com` would otherwise never match.
            return validate_registry_host(value)
        except ImageIdentityError as exc:
            raise ValueError(f"registry must be a host, optionally with a port, got {value!r}") from exc

    @field_validator("repository_prefix")
    @classmethod
    def _prefix_is_a_repository_path(cls, value: str) -> str:
        """Normalized once here, so the composed path has exactly one `/` between its parts."""
        value = value.strip("/")
        return validate_repository(value) if value else value

    registry_insecure: bool = Field(
        default=False,
        description=(
            "Reach the registry over plain HTTP, for an in-cluster dev registry. Off by default "
            "and never a fallback: a registry reached over HTTP cannot establish that the digest "
            "it reports is the digest anyone else would see -- and the credential broker signs the "
            "digest it reads."
        ),
    )
    registry_narrows_scope: bool = Field(
        default=False,
        description=(
            "Whether the registry's token service narrows a token to the scope it was asked for, "
            "as the Distribution token specification requires. The broker asks for exactly a "
            "job's destinations either way; this says whether the registry honors that, and every "
            "row records it as `guarantees.push_scope` (Requirement 9).\n\n"
            "Unverified for GAR's and Artifactory's token services, so off by default: a registry "
            "is treated as not narrowing until it is measured to. Set, the broker also refuses any "
            "token whose scope it can read and finds wider than it asked for."
        ),
    )

    # --- The credential broker, and the trust root ------------------------

    credential_broker: str | None = Field(
        default=None,
        description=(
            "URL of the credential broker, e.g. `http://nhx-build-broker.nhx-build-broker.svc:8080`. "
            "The push step gets every registry credential it uses from here -- for its own job's "
            "destinations, for as long as the registry's token service allows -- and asks it to "
            "sign. No step holds a standing credential or a key (Requirement 7): a Secret a step's "
            "ServiceAccount could read would be readable by anyone able to submit a raw Jobs job "
            "under that step's profile.\n\n"
            "The broker speaks plain HTTP. With platform auth on, a step sends it a token the "
            "platform will exchange for the submitter's, so put TLS in front of it, as a service "
            "mesh does.\n\n"
            "No safe default; unset fails the submit."
        ),
    )
    signing_public_key: str | None = Field(
        default=None,
        description=(
            "PEM of the public half of the key the broker signs with. The trust root signatures "
            "are verified against: a row goes `ready` only on a signature over its digest "
            "that this key verifies (Requirements 6 and 8). The private half never leaves the "
            "broker, and nothing in the platform needs it.\n\n"
            "No safe default; unset fails the submit -- a build whose signature nothing could "
            "verify could never complete."
        ),
    )

    @field_validator("credential_broker")
    @classmethod
    def _broker_is_a_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
            raise ValueError(f"credential_broker must be an http(s) URL with no query, got {value!r}")
        return value.rstrip("/")

    @field_validator("signing_public_key")
    @classmethod
    def _public_key_is_a_public_key(cls, value: str | None) -> str | None:
        """Parsed here, so a truncated or private key fails the platform's start, not every build."""
        if value is None:
            return None
        try:
            key = serialization.load_pem_public_key(value.encode())
        except ValueError as exc:
            raise ValueError("signing_public_key must be a PEM public key (cosign.pub)") from exc
        if not isinstance(key, ec.EllipticCurvePublicKey | rsa.RSAPublicKey):
            raise ValueError("signing_public_key must be an EC or RSA key, which is what cosign signs with")
        return value

    # --- Imports and runtimes -------------------------------------------

    import_registries: list[str] = Field(
        default_factory=list,
        description=(
            "Registry hosts an `ImageSource` may name, e.g. `docker.io`. Checked at submit, before "
            "any row exists, against the NORMALIZED reference -- so `alpine@sha256:...` is "
            "`docker.io`. Empty, the default, refuses every import.\n\n"
            "This is what bounds what the signing key covers on the import path: every import is "
            "published and signed, so the list is the operator's statement of whose images it is "
            "willing to sign for."
        ),
    )
    runtime_layers: dict[str, RuntimeLayer] = Field(
        default_factory=dict,
        description=(
            "Runtime name -> the layer appended to every image built for it. A `BuildSet` names "
            "one in `runtime`, or none. With none, an import is a byte-for-byte copy; with one, it "
            "is `FROM <upstream>@<digest>` plus the layer. Supply as JSON through "
            "`NEMO_BUILDER_RUNTIME_LAYERS`, or in the config file."
        ),
    )

    @field_validator("import_registries")
    @classmethod
    def _import_registries_are_hosts(cls, value: list[str]) -> list[str]:
        """Each a host on the reference grammar, lowercased, and Docker Hub's aliases folded in.

        Normalized the way references are, because they are compared with normalized references:
        an entry that never matches anything -- `" docker.io"`, `Index.Docker.io` -- would refuse
        imports it was written to allow, with nothing saying why.
        """
        hosts: list[str] = []
        for host in value:
            try:
                normalized = validate_registry_host(host)
            except ImageIdentityError as exc:
                raise ValueError(f"import_registries entries must be registry hosts, got {host!r}") from exc
            hosts.append("docker.io" if normalized in ("index.docker.io", "registry-1.docker.io") else normalized)
        return hosts

    @field_validator("runtime_layers")
    @classmethod
    def _runtime_names(cls, value: dict[str, RuntimeLayer]) -> dict[str, RuntimeLayer]:
        for name in value:
            if not _RUNTIME_NAME.fullmatch(name):
                raise ValueError(f"invalid runtime name {name!r}; use lowercase letters, digits and '-'")
        return value

    # --- The failure sweep -----------------------------------------------

    sweep_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "How often the failure sweep looks at `pending` rows. It only ever fails rows -- a "
            "row goes `ready` the moment its signature is delivered -- so this bounds how long a "
            "dead build's rows stay pending, not how long a good one takes."
        ),
    )
    job_creation_grace_seconds: float = Field(
        default=300.0,
        gt=0,
        description=(
            "How long a row's job may be missing before the sweep fails the row. Rows are written "
            "before the job, so for a moment after every submit the job is legitimately not there "
            "yet; after this, a submit that died between the two has left rows nothing will build."
        ),
    )

    signature_storage: Literal["tag", "referrers"] = Field(
        default="tag",
        description=(
            "Pin to the oldest verifier served, not to the registry. Measured: GAR implements "
            "referrers fully AND accepts the legacy `.sig` tag, so the registry is not the "
            "constraint -- the consumer is. A v2 verifier reading a v3 referrers-stored "
            "signature reports 'no signatures found', i.e. UNSIGNED, not 'format mismatch'.\n\n"
            "Only `tag` is implemented: signatures are made with cosign v2, which writes the legacy tag. "
            "`referrers` is refused rather than accepted and ignored -- recorded on every row while "
            "the signature sat somewhere else, it would make each row misreport where its signature "
            "is."
        ),
    )

    @field_validator("signature_storage")
    @classmethod
    def _only_tag_storage_is_implemented(cls, value: str) -> str:
        if value != "tag":
            raise ValueError(
                "signature_storage 'referrers' is not implemented: the broker writes cosign's legacy "
                "`.sig` tag. Use 'tag'."
            )
        return value

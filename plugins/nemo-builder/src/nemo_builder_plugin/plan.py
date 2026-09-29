# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request resolved against this deployment, once.

Both halves of a submit read from the same :class:`BuildPlan`: the backend compiles it into a job,
and the submit path turns it into ``ContainerImage`` rows. Neither re-derives a name, a
destination or a tag, so the two cannot disagree about which image is which -- the job pushes the
tag each row's signature will cover because both read it from the same record.

**Two passes, because only one of them depends on the backend.** :meth:`BuildPlan.resolve` settles
what is true on every deployment: the job's name, each row's name, each spec's source. Then
:meth:`BuildPlan.with_destinations` asks the deployment's backend where each image goes -- the one
decision that differs between backends -- and checks what only a destination can show.

**Every check that can reject a request without asking another service runs here, in the
backend's ``check``, or in the request schema before either.** What comes out is facts, not
input: the compiler and the row builder have nothing left to validate. What only Files can answer
-- does this fileset exist, may this caller read it -- is answered by ``fetch``, as the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from nemo_builder_plugin.schema import BuildSet, BuildSpec
from nemo_builder_plugin.steps import ContextSource


def qualify_fileset(fileset: str, workspace: str) -> str:
    """``<workspace>/<name>``, whichever way the request spelled it.

    One spelling per fileset: ``fs`` and ``default/fs`` are one download, and no fileset's
    directory on the work volume can nest inside another's.
    """
    return fileset if "/" in fileset else f"{workspace}/{fileset}"


@dataclass(frozen=True, slots=True)
class Destination:
    """Where one image is published: a registry host, a repository path, and the system tag.

    ``registry`` is a HOST -- what a registry client connects to -- and ``repository`` a path within
    it. They are kept apart because ``https://host/project/repo/v2/...`` is what a joined string
    produces the moment anything resolves it.
    """

    registry: str
    repository: str
    #: The tag this system pushes and the broker signs; None where the backend chooses its own
    #: tag, and the signature names it instead.
    system_tag: str | None

    @property
    def system_ref(self) -> str | None:
        if self.system_tag is None:
            return None
        return f"{self.registry}/{self.repository}:{self.system_tag}"


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """One image, with everything about it decided at submit."""

    index: int
    spec: BuildSpec
    #: The submitting workspace, which an unqualified fileset name belongs to.
    workspace: str
    build_set: str
    revision: int
    #: The ``ContainerImage`` row name, ``<job>-<index>``.
    name: str
    #: Filled in by :meth:`BuildPlan.with_destinations`, from the backend.
    destination: Destination | None = None

    @property
    def source(self) -> ContextSource:
        return ContextSource(
            fileset=qualify_fileset(self.spec.source.fileset, self.workspace),
            context_path=self.spec.source.context_path or None,
        )

    @property
    def placed(self) -> Destination:
        """The destination, which every use after planning requires."""
        if self.destination is None:
            raise ValueError(f"{self.name} has no destination yet; plan.with_destinations() comes first")
        return self.destination

    @property
    def caller_ref(self) -> str | None:
        """``<registry>/<repository>:<output.tag>``, or None when the spec named no output."""
        if self.spec.output is None:
            return None
        return f"{self.placed.registry}/{self.placed.repository}:{self.spec.output.tag}"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """What will be built, where it goes, and what it will be called. Built by :meth:`resolve`."""

    build_set: BuildSet
    workspace: str
    #: ``<set>-<revision>``. Deterministic from the request, which is what lets concurrent
    #: submitters settle on create-or-get against the unique name index.
    job_name: str
    images: tuple[PlannedImage, ...]

    @classmethod
    def resolve(cls, build_set: BuildSet, *, workspace: str) -> BuildPlan:
        """What is true of this request on any deployment. Nothing here can fail but a name."""
        job_name = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            PlannedImage(
                index=index,
                spec=spec,
                workspace=workspace,
                build_set=build_set.name,
                revision=build_set.revision,
                # A tracking handle, not a label: stable within a revision, meaningless across
                # them. Not derived from `output.repository`, which is arbitrary, caller-owned, and
                # may be longer than a name may be.
                name=f"{job_name}-{index}",
            )
            for index, spec in enumerate(build_set.build_specs)
        )
        return cls(build_set=build_set, workspace=workspace, job_name=job_name, images=images)

    def with_destinations(self, destination: Callable[[PlannedImage], Destination]) -> BuildPlan:
        """The plan with each image's destination decided by ``destination`` -- the backend's rule.

        Refuses two specs publishing the same caller reference: the tag could point at only one.
        A plain ``ValueError`` -- a malformed request (400), not a deployment that cannot satisfy
        a well-formed one (409).
        """
        placed = tuple(replace(image, destination=destination(image)) for image in self.images)
        seen: dict[str, str] = {}
        for image in placed:
            ref = image.caller_ref
            if ref is None:
                continue
            if ref in seen:
                raise ValueError(
                    f"build specs {seen[ref]!r} and {image.spec.name!r} both publish {ref}; "
                    "the tag could point at only one of them"
                )
            seen[ref] = image.spec.name
        return replace(self, images=placed)

    def groups(self) -> list[tuple[ContextSource, tuple[PlannedImage, ...]]]:
        """What the sandbox builds, grouped, in first-appearance order. One sandbox per group.

        One sandbox per distinct ``(fileset, context_path)``, so a Dockerfile can never read the
        context of a different source in the same set. Images sharing a source share a sandbox,
        which is the common shape and also the cheap one.
        """
        grouped: dict[tuple[str, str | None], list[PlannedImage]] = {}
        for image in self.images:
            source = image.source
            grouped.setdefault((source.fileset, source.context_path), []).append(image)
        return [
            (ContextSource(fileset=fileset, context_path=context_path), tuple(images))
            for (fileset, context_path), images in grouped.items()
        ]

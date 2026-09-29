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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from nemo_builder_plugin.config import RuntimeLayer
from nemo_builder_plugin.identity import ImageReference, normalize_pinned_reference
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import ContextSource, RuntimeLayerSpec


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
    #: For an import: the upstream reference the caller named, normalized. None for a build.
    upstream: ImageReference | None = None
    #: For an import: the platform manifest the submit path resolved. Filled in by
    #: :meth:`BuildPlan.with_upstream_manifests`, the one step of a submit that reads a registry.
    upstream_manifest_digest: str | None = None

    @property
    def source(self) -> ContextSource | None:
        """The fileset context this image builds from, or None for an import."""
        source = self.spec.source
        if isinstance(source, FileSetSource):
            return ContextSource(
                fileset=qualify_fileset(source.fileset, self.workspace),
                context_path=source.context_path or None,
            )
        return None

    @property
    def pinned_upstream(self) -> str:
        """``<registry>/<repository>@<platform manifest digest>`` -- what an import pulls."""
        if self.upstream is None or self.upstream_manifest_digest is None:
            raise ValueError(f"{self.name} is not a resolved import")
        return f"{self.upstream.repository_ref}@{self.upstream_manifest_digest}"

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
    #: The layer of the runtime the set names, or None. It is what makes an import a derived
    #: build rather than a copy, so it is decided here, once, for every image in the set. None as
    #: well when the set names a runtime the deployment does not have -- which the backend
    #: refuses before anything is written.
    runtime_layer: RuntimeLayerSpec | None = None

    @classmethod
    def resolve(
        cls, build_set: BuildSet, *, workspace: str, runtime_layers: Mapping[str, RuntimeLayer] | None = None
    ) -> BuildPlan:
        """What is true of this request on any deployment, given the operator's runtime layers."""
        job_name = f"{build_set.name}-{build_set.revision}"
        layer = (runtime_layers or {}).get(build_set.runtime) if build_set.runtime else None
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
                upstream=_upstream(spec),
            )
            for index, spec in enumerate(build_set.build_specs)
        )
        runtime_layer = (
            RuntimeLayerSpec(name=build_set.runtime, version=layer.version, dockerfile=layer.dockerfile)
            if layer is not None and build_set.runtime is not None
            else None
        )
        return cls(
            build_set=build_set, workspace=workspace, job_name=job_name, images=images, runtime_layer=runtime_layer
        )

    def is_copy(self, image: PlannedImage) -> bool:
        """An import published byte for byte. Copies never reach the sandbox: nothing executes."""
        return image.upstream is not None and self.runtime_layer is None

    @property
    def imports(self) -> tuple[PlannedImage, ...]:
        return tuple(image for image in self.images if image.upstream is not None)

    def with_upstream_manifests(self, manifests: Mapping[str, str]) -> BuildPlan:
        """The plan with every import's platform manifest filled in, keyed by image name.

        The one piece of a plan that needs I/O -- a registry read -- so it is applied to a
        resolved plan rather than done inside :meth:`resolve`, which stays pure.
        """
        missing = [image.name for image in self.imports if image.name not in manifests]
        if missing:
            raise ValueError(f"no upstream manifest resolved for {', '.join(missing)}")
        images = tuple(
            replace(image, upstream_manifest_digest=manifests[image.name]) if image.upstream else image
            for image in self.images
        )
        return replace(self, images=images)

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

    def groups(self) -> list[tuple[ContextSource | None, tuple[PlannedImage, ...]]]:
        """What the sandbox builds, grouped, in first-appearance order. One sandbox per group.

        One sandbox per distinct ``(fileset, context_path)``, so a Dockerfile can never read the
        context of a different source in the same set. Images sharing a source share a sandbox,
        which is the common shape and also the cheap one.

        Derived imports come last, **one group each** (``None``). Each context is a Dockerfile
        ``fetch`` wrote, holding nothing secret -- but each build runs its upstream publisher's
        binaries, and in a shared sandbox one publisher's could rewrite another's image before it
        is signed. Copies are in no group at all -- nothing about them runs.
        """
        grouped: dict[tuple[str, str | None], list[PlannedImage]] = {}
        derived: list[PlannedImage] = []
        for image in self.images:
            source = image.source
            if source is not None:
                grouped.setdefault((source.fileset, source.context_path), []).append(image)
            elif not self.is_copy(image):
                derived.append(image)
        groups: list[tuple[ContextSource | None, tuple[PlannedImage, ...]]] = [
            (ContextSource(fileset=fileset, context_path=context_path), tuple(images))
            for (fileset, context_path), images in grouped.items()
        ]
        groups.extend((None, (image,)) for image in derived)
        return groups


def _upstream(spec: BuildSpec) -> ImageReference | None:
    """An import's upstream reference, parsed; None for a build from a fileset.

    The schema already normalized it, so this cannot fail on anything the schema accepted. Whether
    this deployment will import from its registry at all is the backend's to decide.
    """
    if not isinstance(spec.source, ImageSource):
        return None
    return normalize_pinned_reference(spec.source.image)

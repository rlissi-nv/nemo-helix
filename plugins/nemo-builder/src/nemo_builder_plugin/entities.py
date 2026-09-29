# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage`` -- the only entity in this design.

Created **at submit, in ``pending``, before anything is built.** That inversion is the whole
simplification: there is no orphan to detect. An earlier design wrote the row *after* the push,
so a pod dying in between left a pushed image with nothing pointing at it and needed a sweep over
terminal jobs to find it. Creating the row first means the pointer always exists and the only
question is what state it is in.

**Each transition has exactly one ground, and one writer** (Requirement 6). The submit route
writes the desired fields -- ``registry``, ``repository``, ``provenance``, ``platform`` -- and
never touches them again. Then:

- **``pending -> ready``** is written by the signature route, and only on a signature it
  verified: a signature over the image's digest, checked against the backend's trust root and
  tied to this row. The digest recorded is the one the signature covers.
- **``pending -> failed``** is written by the failure sweep, and only when the row's job has
  ended -- or never appeared -- while the row is still ``pending``.

Both writes are conditional on the version the row was read at, so neither can overwrite the
other. **The build plane never writes this entity.** Its last step carries the signature to the
control plane, and a courier can withhold it but cannot forge it: the signature is made by
the credential broker, or by a backend's own signer, over a digest *they* read from the registry.

**Write-once on ``ready``.** ``digest``, ``manifest_digest`` and ``tag`` are set exactly once, at
the ``pending -> ready`` transition. A rebuild creates a *new* row rather than mutating one a
consumer may have pinned. ``ready`` is terminal; so is ``failed``.

No migration is required to add this type. The entity store is one generic table keyed on
``entity_type`` with the payload in a JSON column, so a new ``NemoEntity`` subclass is storable
and filterable the moment it is declared. There is also no secondary-index mechanism, which
settles a question RFC 001's plan left open: the digest index it wanted is not merely expensive,
it has nowhere to live yet.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field, model_validator

#: `Provenance.backend` on a row no backend produced.
REGISTERED = "registered"


class RegisteredOrigin(BaseModel):
    """Registered through ``POST /container-images``. Nothing here built it.

    Not reachable in the PoC -- the registration route is `M1-10` and out of scope -- but the arm
    exists so the discriminator is stable, and so the failure sweep and the signature route can
    already refuse to touch such a row, which is the behaviour that makes the route safe to add
    later.
    """

    type: Literal["registered"] = "registered"
    image_ref: str = Field(
        description=(
            "The reference the caller supplied, recorded AS SUPPLIED. Redundant against `digest` "
            "on a good day and the only useful field on a bad one: when the two disagree later, "
            "this says what was asked for and the digest says what was got, so a tag that has "
            "since moved is visible as the difference."
        )
    )


class UpstreamImage(BaseModel):
    """The published image an import copied, or a derived build started ``FROM``.

    Written by the submit route from a registry read it makes before the row exists, so it is
    desired state: the broker and the signature route read it, and nothing writes it again.
    """

    image_ref: str = Field(description="The `ImageSource.image`, normalized: `<registry>/<repository>@<digest>`.")
    manifest_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "The platform manifest the import takes: the reference's own digest when it names a "
            "plain manifest, and the child matching the spec's platform when it names an index. For "
            "a copy, the broker signs only this digest, and the signature route accepts only it."
        ),
    )


class JobOrigin(BaseModel):
    """A build job produces it."""

    type: Literal["job"] = "job"
    build_set: str = Field(
        description=(
            "The `BuildSet.name` this row came from. Stored rather than parsed back out of "
            "`ContainerImage.name`, which is `<set>-<revision>-<index>` and whose last two "
            "components are both numeric."
        )
    )
    revision: int = Field(
        ge=1,
        description=(
            "The caller's ordinal for `build_set`. The only place a resubmission precondition "
            "can read that outlives the job: a HelixJob disappears on explicit delete and "
            "goes invisible to the non-terminal query the moment it completes, so a precondition "
            "read against jobs is blind in two ways that a read against this row is not."
        ),
    )
    job: str = Field(
        pattern=ENTITY_REF_PATTERN,
        description=(
            "`workspace/name` of the build job, on the platform's own reference pattern. The "
            "credential broker grants a step what its job's `pending` rows name, and finds them by "
            "this; the failure sweep asks Jobs about it."
        ),
    )
    system_tag: str | None = Field(
        description=(
            "The tag this system pushes for this image, and the one the broker resolves and signs. "
            "None on a backend that chooses its own tag, where the signature names it instead.\n\n"
            "Stored rather than re-derived. The redundancy is deliberate: this records the tag "
            "this build ACTUALLY PUSHES, so a later change to how the tag is composed leaves "
            "already-written rows signing what was pushed instead of re-deriving a string nobody "
            "pushed."
        )
    )
    request_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "sha256 of the canonical JSON of the `BuildSet` that wrote this row. What lets a "
            "resubmission of the same `(name, revision)` tell a retry of the same request, which "
            "adopts this row, from a different request under a reused revision, which is refused: "
            "the row name alone cannot, so without this an adopted row would describe a request "
            "the job never built. Every signature this system makes carries it, so this row's "
            "signature cannot pass for a different request's under the same name."
        ),
    )
    upstream: UpstreamImage | None = Field(
        default=None,
        description=(
            "Set when the spec was an `ImageSource`: what the import was of. With `runtime_layer` "
            "unset the row is a COPY, and its digest is a claim about the publisher's image rather "
            "than about anything built here -- the broker and the signature route both check it. "
            "With `runtime_layer` set it is a DERIVED image, and this names its base. None for a "
            "build from a fileset."
        ),
    )
    runtime_layer: str | None = Field(
        default=None,
        description=(
            "`<name>@<version>` of the runtime layer this job appended, or None where the set named "
            "no runtime. An image adapted to one runtime's contract -- or to one version of it -- "
            "is not an image adapted to another, and this is the only place that says which."
        ),
    )

    @property
    def is_copy(self) -> bool:
        """An import published byte for byte: the one row whose digest must equal upstream's."""
        return self.upstream is not None and self.runtime_layer is None


Origin = Annotated[RegisteredOrigin | JobOrigin, Field(discriminator="type")]


class Guarantees(BaseModel):
    """What every row a backend produces can promise. Written at submit, from the backend.

    A consumer checks the property it needs -- could the caller's ``RUN`` reach a credential; can
    this tag move -- and never reads it off a backend's name. RFC 001 made ``backend`` a closed
    list precisely so a consumer could learn *which* guarantees hold; a typed field says that
    directly, and does not need a new list entry for every implementation.
    """

    run_can_reach: Literal["nothing", "build-identity", "registry-credential"] = Field(
        description=(
            "What the caller's `RUN` could reach (Requirement 7). `nothing`: it ran in a sandbox "
            "holding no credential and no identity. `build-identity`: a provider's worker, under the "
            "build identity it runs every build as. `registry-credential`: a push credential sat "
            "beside it."
        )
    )
    egress: Literal["cluster-denied", "unrestricted"] = Field(
        description="Whether the build could reach the cluster's own networks, cloud metadata included."
    )
    signer: str = Field(
        description=(
            "The trust root this backend's signatures are verified against. Every `ready` "
            "row's `signature.verified_against` names the same one."
        )
    )
    provenance: Literal["none", "slsa", "builder-attested"] = Field(
        description="What build provenance accompanies the image, beyond the signature."
    )
    destination: Literal["per-workspace", "shared"] = Field(
        description=(
            "`per-workspace`: the image lives under its workspace's own path, which no other "
            "workspace's build can write. `shared`: one repository holds every tenant's images."
        )
    )
    tag: Literal["write-once", "may-move"] = Field(
        description=(
            "Whether the tag the row records keeps naming its digest. Only `write-once` makes a tag "
            "safe to hand to something that admits only tags."
        )
    )
    push_scope: Literal["per-repository", "registry-wide", "backend-held"] = Field(
        description=(
            "How far the credential that published the image reached (Requirement 9). "
            "`per-repository`: the registry narrowed it to this job's destinations. "
            "`registry-wide`: the registry does not narrow a token's scope, or it is not verified "
            "that it does, so the credential was as wide as the broker's own. `backend-held`: the "
            "backend published with a credential this system never held."
        )
    )


class Provenance(BaseModel):
    """Where an image came from. Written by one writer at one moment, and never touched again."""

    backend: str = Field(
        description=(
            f"The id of the backend that produced this -- the deployment's one backend, such as "
            f"`execution` -- or `{REGISTERED}` for a row no backend produced. Not recoverable from "
            "`built_by`, which only distinguishes a job from a registration.\n\n"
            "An id, not a promise: read `guarantees` to learn what holds. It is NOT a quality "
            "ordering either -- a managed backend emits SLSA provenance the execution mode cannot, "
            "and runs caller-authored RUN under a cloud identity the execution mode denies it."
        )
    )
    guarantees: Guarantees | None = Field(
        default=None,
        description="What the backend promises for this row. None only for a registered row, which no backend produced.",
    )
    built_by: Origin

    @model_validator(mode="after")
    def _guarantees_come_from_a_backend(self) -> Self:
        """A job row always says what its backend guarantees; a registered row never claims to."""
        if isinstance(self.built_by, RegisteredOrigin):
            if self.backend != REGISTERED or self.guarantees is not None:
                raise ValueError(f"a registered row has backend {REGISTERED!r} and no guarantees")
        elif self.backend == REGISTERED or self.guarantees is None:
            raise ValueError("a row a job produces names its backend and what that backend guarantees")
        return self


class Signature(BaseModel):
    """The signature the row was made ``ready`` on, as the control plane verified it.

    Written once, at the ``pending -> ready`` transition, in the same write as ``digest``. The
    signature covers that digest, was checked against a trust root the operator configured, and
    was tied to this row by the backend's signature policy -- so a row that says it is signed means
    the signature was *verified*, not merely found. The signature itself lives in the registry,
    beside the image, where the cluster's own verifier finds it at admission.
    """

    storage: Literal["tag", "referrers"] = Field(
        description=(
            "Where the signature is stored. A cosign signature lives either at the legacy "
            "`sha256-<hex>.sig` tag or as an OCI 1.1 referrer, and cosign v3 defaults to "
            "referrers. A verifier that reads only the legacy tag then finds nothing and reports "
            "'unsigned image' rather than 'format mismatch' -- so the storage mode is pinned to "
            "the oldest verifier we serve, not to what the registry supports."
        )
    )
    verified_against: str | None = Field(
        default=None,
        description=(
            "The trust root the signature was verified against. Set on every row a job produced, "
            "since such a row reaches `ready` only by a verified signature. None is reserved "
            "for a signature observed rather than verified -- a registered row's, once "
            "registration is built."
        ),
    )
    signer: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The signed claims the signature policy checked against this row: for this deployment's "
            "key, the workspace, image, job and request digest the signature names."
        ),
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image: desired state at submit, observed state once its signature is verified.

    ``name`` is ``<job-name>-<index>``, where the index is the spec's position in
    ``build_specs``. That makes the name deterministic from the request -- which is what lets
    concurrent submitters settle on create-or-get against the unique name index rather than a
    read-then-write that races across replicas. The index is positional: stable within a
    revision, meaningless across revisions.

    Not consulted at pull time. A workload names an image as a string and the kubelet pulls it.
    This row is what a dispatcher resolves *before* it writes that string, and what an auditor
    reads afterwards.
    """

    # --- Desired. Written once by the submit route; never touched again. ---

    status: Literal["pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    registry: str = Field(description="Registry host the image lives on.")
    repository: str = Field(
        description=(
            "Repository path within the registry. Where this deployment publishes: "
            "`<repository_prefix>/<workspace>/<output.repository>`, or "
            "`<repository_prefix>/<workspace>/<set>/<spec>` for a spec that names no output."
        )
    )
    provenance: Provenance
    platform: str = Field(
        default="linux/amd64",
        pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$",
    )

    # --- Observed. Absent until `status == "ready"`, then immutable. ---

    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description=(
            "The digest the verified signature covers: read from the registry by the "
            "party that signed it, never reported by a build."
        ),
    )
    manifest_digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description=(
            "The per-platform manifest inside an index. Equal to `digest` when the image is a plain "
            "manifest, which is all the broker signs today."
        ),
    )
    tag: str | None = Field(
        default=None,
        description=(
            "The tag the signer resolved to reach `digest`: the system tag where this deployment "
            "publishes, or the backend's own where it chooses one -- recorded as last seen."
        ),
    )
    source_digest: str | None = Field(
        default=None,
        description=(
            "Hash of the context the fetch step downloaded. ADVISORY, never identity: the build "
            "plane computes it, which is why it can never be the thing a consumer pins. Not yet "
            "carried to the row."
        ),
    )
    signature: Signature | None = Field(
        default=None,
        description=(
            "The verified signature the row was made `ready` on. None until then, and None on a "
            "registered row, always -- the broker signs only rows its own jobs produced. A "
            "consumer on a verification-enforcing cluster reads that None to know the row will be "
            "refused at admission before it schedules anything."
        ),
    )

    @property
    def image_ref(self) -> str | None:
        """``<registry>/<repository>@<digest>`` -- a projection, not a stored field.

        Stored, it would be a second place the digest lives, and two places disagree eventually.
        """
        if self.digest is None:
            return None
        return f"{self.registry}/{self.repository}@{self.digest}"

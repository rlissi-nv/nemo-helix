# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend interface: the one place a backend's differences live.

A deployment runs **exactly one backend**, chosen when the platform is deployed (Requirement 4).
The submit route, the signature route and the credential broker load the same backend from the same
configuration, and everything that differs between one backend and another sits behind
:class:`Backend`:

- what it can honor, refused at submit when it can't (``check``);
- the whole job (``compile``). Every backend runs as a Jobs job, including one that only calls an
  external builder: a call that blocks for most of an hour, a source read as the submitter, and
  retries that survive a restart all belong in a pod;
- where each image goes (``destination``);
- how its signatures are verified and tied to one row (``signature_policy``);
- what every row it produces can promise (``guarantees``);
- per step role, what the credential broker may hand out (``entitlements``). The broker uses
  only this.

**Who publishes and who signs follows the destination's trust domain.** Where an image lands
under this deployment's key, the broker signs it. Where it must be admitted under another
domain's key -- ACES's, on Astra -- the backend publishes and signs it there, and this system
*verifies*: completion is a verified signature, whoever carries it.

**What is not here is as deliberate.** No method writes a row, reads a registry, or holds a
credential. The control plane owns requests and records and never builds, pushes or signs; a
backend declares, and the control plane and the broker act on the declaration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol

from nemo_builder_plugin.entities import ContainerImage, Guarantees
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_helix_plugin.jobs.spec import HelixJobSpec


class BackendRefused(ValueError):
    """The deployment's backend cannot honor this request, or cannot build at all (409).

    A ``ValueError`` so the submit path's handling of malformed requests would catch it -- which
    is why that path catches this first: the request is well-formed, and the deployment is what
    stands in the way.
    """


@dataclass(frozen=True, slots=True)
class StepEntitlement:
    """What the credential broker may hand one step role, and how it recognises that step's pod.

    Recognition is from what Jobs wrote on the pod -- the step name and the execution profile in
    its labels, the ServiceAccount in its spec -- none of which the step can choose. The grant is
    never a scope the step asks for: it is a function of this entry and the job's ``pending``
    rows, bounded by the job's workspace. A role not listed gets nothing.
    """

    #: The Jobs step name, as the pod's ``job_step_name`` label records it.
    step: str
    #: The execution profile, as the pod's ``job_execution_profile`` label records it.
    profile: str
    #: The ServiceAccount the profile runs as.
    service_account: str
    #: Registry actions on the job's ``pending`` destinations. Never ``delete``: nothing this
    #: system does needs it, and a step that could delete could erase what it published.
    destinations: tuple[Literal["pull", "push"], ...] = ()
    #: Whether the step may ask the broker to sign its job's ``pending`` rows.
    sign: bool = False


@dataclass(frozen=True, slots=True)
class SignedPayload:
    """What a completion signature says, read out of its payload only AFTER the signature verified.

    Nothing the courier sends outside the signed payload is trusted, so this is everything the
    control plane learns from a signature: which repository, which digest, and the annotations the
    signer attached.
    """

    #: ``<registry>/<repository>``: where the signer found the image.
    reference: str
    #: The manifest digest the signature covers.
    digest: str
    annotations: Mapping[str, str] = field(default_factory=dict)


class SignatureRefused(Exception):
    """A signature that does not verify, or does not belong to the row it was delivered for."""


class SignaturePolicy(Protocol):
    """How a backend's signatures are checked (Requirement 6).

    Two halves: the signature must verify under ``trust_root``, and the verified claims must tie
    it to one row. The second is what makes a signature from a shared signer worth anything -- a
    signer that signs for many callers proves only that *someone* asked, until a policy says who
    and for what.
    """

    @property
    def trust_root(self) -> str:
        """An id for the key signatures are verified against, recorded on every ``ready`` row."""
        ...

    def verify(self, payload: bytes, signature: bytes) -> None:
        """Raise :class:`SignatureRefused` unless ``signature`` is the trust root's, over ``payload``."""
        ...

    def bind(self, row: ContainerImage, payload: SignedPayload) -> dict[str, str]:
        """Tie a verified payload to ``row``, or raise :class:`SignatureRefused`.

        Returns the claims it checked, which the row records as ``signature.signer``.
        """
        ...


class Backend(Protocol):
    """One per deployment (Requirement 4). The only place a backend's differences live."""

    #: ``provenance.backend`` on every row this backend produces: ``execution``, ``cloud-build``,
    #: ``aces``, ...
    id: ClassVar[str]

    def check(self, plan: BuildPlan) -> None:
        """Refuse at submit what this backend cannot honor, or refuse everything if it cannot build.

        Raises :class:`BackendRefused`: a platform it does not build, a context over its limit, an
        ``output`` where it chooses the destination -- or a deployment missing a setting nothing
        can default.
        """
        ...

    def destination(self, image: PlannedImage) -> Destination:
        """Where ``image`` will be published.

        Where this system decides, that is the registry, the repository and the system tag. A
        backend that chooses its own tag returns ``system_tag=None``, and its signature names the
        tag. Raises ``ValueError`` for a request whose destination cannot be composed.
        """
        ...

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        """The whole job, every step of it, from a checked and placed plan. Accepts nothing else."""
        ...

    def signature_policy(self) -> SignaturePolicy:
        """The trust root a completion signature is checked against, and how it binds to one row."""
        ...

    def guarantees(self) -> Guarantees:
        """What every row this backend produces can promise. Written on the row at submit."""
        ...

    def entitlements(self) -> Mapping[str, StepEntitlement]:
        """Per step role, what the broker may hand out. A role not listed gets nothing."""
        ...

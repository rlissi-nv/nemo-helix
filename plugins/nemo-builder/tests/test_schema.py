# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema rules, and the entity's write-once shape."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.entities import (
    REGISTERED,
    ContainerImage,
    Guarantees,
    JobOrigin,
    Provenance,
    RegisteredOrigin,
    Signature,
)
from nemo_builder_plugin.schema import MAX_BUILD_SPECS, BuildOutput, BuildSet, BuildSpec, FileSetSource
from pydantic import ValidationError

_DIGEST = "sha256:" + "b" * 64


def _spec(
    name: str, *, fileset: str = "fs-a", context_path: str | None = None, dockerfile: str = "Dockerfile"
) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(repository=f"team/{name}", tag="v1"),
        dockerfile=dockerfile,
    )


class TestDockerfileStaysWithinContext:
    """Caller-supplied paths resolved inside the sandbox, rejected at submit."""

    @pytest.mark.parametrize("good", ["Dockerfile", "event_feed/Dockerfile", "a/b/c.dockerfile"])
    def test_descends(self, good: str) -> None:
        assert _spec("x", dockerfile=good).dockerfile == good

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../Dockerfile", "a/../../b/Dockerfile", "", "."])
    def test_does_not_escape_and_names_a_file(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="dockerfile"):
            _spec("x", dockerfile=bad)


class TestSourcePaths:
    """Checked at submit, where a bad value is the caller's 4xx rather than a failed fetch pod."""

    @pytest.mark.parametrize("good", ["fs-a", "other-ws/fs-a"])
    def test_a_fileset_is_a_name_or_a_workspace_qualified_name(self, good: str) -> None:
        assert FileSetSource(fileset=good).fileset == good

    @pytest.mark.parametrize("bad", ["../x", "/etc", "a/b/c", "Fs", "fs--a", ""])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="fileset"):
            FileSetSource(fileset=bad)

    @pytest.mark.parametrize("bad", ["../fs-b", "/etc", "a/../../b", "."])
    def test_a_context_path_stays_inside_the_fileset(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="context_path"):
            FileSetSource(fileset="fs-a", context_path=bad)


class TestUnknownFieldsAreRefused:
    """A misspelt field, silently dropped, builds something other than what was asked for."""

    def test_on_the_spec(self) -> None:
        body = _spec("x").model_dump() | {"dockerFile": "build/Dockerfile"}
        with pytest.raises(ValidationError, match="dockerFile"):
            BuildSpec.model_validate(body)

    def test_on_the_source(self) -> None:
        with pytest.raises(ValidationError, match="contextPath"):
            FileSetSource.model_validate({"fileset": "fs-a", "contextPath": "env"})

    def test_on_the_set(self) -> None:
        body = {"name": "demo", "revision": 1, "build_specs": [_spec("a").model_dump()], "registry": "x"}
        with pytest.raises(ValidationError, match="registry"):
            BuildSet.model_validate(body)


class TestBuildOutput:
    """The compiler composes the path under the workspace, so the caller's part must stay a part."""

    def test_a_registry_is_refused_rather_than_ignored(self) -> None:
        """Ignored, it would publish somewhere other than where the caller asked."""
        with pytest.raises(ValidationError, match="registry"):
            BuildOutput.model_validate({"registry": "elsewhere.example.com", "repository": "team/app", "tag": "v1"})

    @pytest.mark.parametrize(
        "bad", ["../team-b/app", "team/../../app", "/team/app", "team//app", "team/app/", "Team/App"]
    )
    def test_a_repository_cannot_leave_the_workspaces_path(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="repository"):
            BuildOutput(repository=bad, tag="v1")

    @pytest.mark.parametrize("bad", ["", "-v1", "v1@sha256", "a" * 129, "café"])
    def test_a_tag_must_be_an_oci_tag(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="tag"):
            BuildOutput(repository="team/app", tag=bad)

    @pytest.mark.parametrize("reserved", ["default--demo-2-0", "v1--rc", "sha256-" + "a" * 64 + ".sig"])
    def test_the_shapes_of_system_and_signature_tags_are_reserved(self, reserved: str) -> None:
        """Pushable by a caller, either would let one image stand in for another's identity."""
        with pytest.raises(ValidationError, match="reserved"):
            BuildOutput(repository="team/app", tag=reserved)


class TestSpecNames:
    """A spec's name is a repository path component, because an output-less spec is published there."""

    @pytest.mark.parametrize("good", ["a", "app", "event-feed", "event_feed", "v1.2", "a__b", "9lives"])
    def test_a_repository_component_is_a_name(self, good: str) -> None:
        assert BuildSpec(name=good, source=FileSetSource(fileset="fs-a")).name == good

    @pytest.mark.parametrize("bad", ["", "App", "my app", "a/b", "-a", "a-", "a..b", "café", "a" * 64])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="name"):
            BuildSpec(name=bad, source=FileSetSource(fileset="fs-a"))


class TestOutputIsOptional:
    """A spec that names no destination runs unchanged on every deployment (Requirement 4)."""

    def test_a_spec_may_name_no_output(self) -> None:
        spec = BuildSpec(name="app", source=FileSetSource(fileset="fs-a"))
        assert spec.output is None

    def test_a_named_output_is_still_checked(self) -> None:
        with pytest.raises(ValidationError, match="repository"):
            BuildSpec.model_validate(
                {"name": "app", "source": {"fileset": "fs-a"}, "output": {"repository": "../x", "tag": "v1"}}
            )


class TestBuildSet:
    def test_revision_is_required_and_one_based(self) -> None:
        with pytest.raises(ValidationError, match="revision"):
            BuildSet(name="demo", revision=0, build_specs=[_spec("a")])
        with pytest.raises(ValidationError, match="revision"):
            BuildSet.model_validate({"name": "demo", "build_specs": [_spec("a").model_dump()]})

    def test_spec_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            BuildSet(name="demo", revision=1, build_specs=[_spec("a"), _spec("a")])

    def test_a_set_needs_at_least_one_spec(self) -> None:
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=[])

    def test_a_set_has_at_most_max_build_specs(self) -> None:
        specs = [_spec(f"s{i}") for i in range(MAX_BUILD_SPECS + 1)]
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=specs)

    @pytest.mark.parametrize("bad", ["Demo", "d", "demo-", "-demo", "de--mo", "de_mo", "de.mo", "de@mo", "9demo"])
    def test_a_name_must_fit_every_name_it_becomes(self, bad: str) -> None:
        """Row, job, tag and pod names each forbid something NAME_PATTERN allows."""
        with pytest.raises(ValidationError, match="name"):
            BuildSet(name=bad, revision=1, build_specs=[_spec("a")])

    def test_a_name_too_long_for_its_derived_names_is_refused_before_anything_is_written(self) -> None:
        # `job-fileset-<name>-<rev>`, the name Jobs gives the job's fileset, is the longest name a
        # set composes while it has at most MAX_BUILD_SPECS specs: 12 + 50 + 2 = 64, one over.
        with pytest.raises(ValidationError, match="too long"):
            BuildSet(name="a" * 50, revision=1, build_specs=[_spec("a")])
        assert BuildSet(name="a" * 49, revision=1, build_specs=[_spec("a")])


_GUARANTEES = Guarantees(
    run_can_reach="nothing",
    egress="cluster-denied",
    signer="sha256:" + "c" * 64,
    provenance="none",
    destination="per-workspace",
    tag="write-once",
    push_scope="per-repository",
)


def _origin(**overrides: object) -> JobOrigin:
    fields: dict[str, object] = {
        "build_set": "s",
        "revision": 1,
        "job": "default/s-1",
        "system_tag": "default--s-1-0",
        "request_digest": "sha256:" + "d" * 64,
    }
    return JobOrigin.model_validate(fields | overrides)


class TestContainerImage:
    def _pending(self) -> ContainerImage:
        return ContainerImage(
            name="s-1-0",
            workspace="default",
            registry="reg.example.com",
            repository="team/main",
            provenance=Provenance(backend="execution", guarantees=_GUARANTEES, built_by=_origin()),
        )

    def test_is_created_pending_with_nothing_observed(self) -> None:
        """The inversion: the row exists before the image does, so there is no orphan to detect."""
        row = self._pending()
        assert row.status == "pending"
        assert row.digest is None and row.manifest_digest is None and row.tag is None
        assert row.signature is None
        assert row.image_ref is None

    def test_image_ref_is_a_projection_of_the_observed_digest(self) -> None:
        row = self._pending()
        row.digest = _DIGEST
        assert row.image_ref == f"reg.example.com/team/main@{_DIGEST}"

    def test_digest_fields_reject_a_non_sha256_value(self) -> None:
        with pytest.raises(ValidationError):
            ContainerImage(
                name="s-1-0",
                workspace="default",
                registry="r.example.com",
                repository="team/main",
                digest="not-a-digest",
                provenance=Provenance(backend="execution", guarantees=_GUARANTEES, built_by=_origin()),
            )

    def test_a_signature_names_the_trust_root_it_was_verified_against(self) -> None:
        """A `ready` row is ready only by a verified signature, so the row's `signature` says what verified it."""
        signature = Signature(
            storage="tag", verified_against=_GUARANTEES.signer, signer={"nhx.nvidia.com/job": "default/s-1"}
        )
        assert signature.verified_against == _GUARANTEES.signer
        assert Signature(storage="tag").signer == {}

    def test_provenance_discriminates_on_origin_type(self) -> None:
        row = self._pending()
        assert isinstance(row.provenance.built_by, JobOrigin)
        assert row.provenance.built_by.system_tag == "default--s-1-0"

    def test_a_backend_that_tags_itself_leaves_no_system_tag(self) -> None:
        assert _origin(system_tag=None).system_tag is None

    def test_a_job_row_always_names_its_request(self) -> None:
        """Every signature carries the request digest, so a job row cannot be without one."""
        with pytest.raises(ValidationError, match="request_digest"):
            JobOrigin(build_set="s", revision=1, job="default/s-1", system_tag="t")  # ty: ignore[missing-argument]


class TestProvenance:
    """Which guarantees hold is a typed field, and only a backend can supply one."""

    def test_the_backend_is_an_open_id(self) -> None:
        assert Provenance(backend="cloud-build", guarantees=_GUARANTEES, built_by=_origin()).backend == "cloud-build"

    def test_a_job_row_says_what_its_backend_guarantees(self) -> None:
        with pytest.raises(ValidationError, match="guarantees"):
            Provenance(backend="execution", built_by=_origin())

    def test_a_job_row_cannot_claim_to_be_registered(self) -> None:
        with pytest.raises(ValidationError, match="guarantees"):
            Provenance(backend=REGISTERED, guarantees=_GUARANTEES, built_by=_origin())

    def test_a_registered_row_claims_no_guarantees(self) -> None:
        origin = RegisteredOrigin(image_ref="reg.example.com/team/app:v1")
        assert Provenance(backend=REGISTERED, built_by=origin).guarantees is None
        with pytest.raises(ValidationError, match="registered"):
            Provenance(backend=REGISTERED, guarantees=_GUARANTEES, built_by=origin)
        with pytest.raises(ValidationError, match="registered"):
            Provenance(backend="execution", built_by=origin)

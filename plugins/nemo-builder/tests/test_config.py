# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration: what is refused when the platform starts, rather than at every build."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from nemo_builder_plugin.config import BuilderConfig
from pydantic import ValidationError


def _pem(key: ec.EllipticCurvePrivateKey | ed25519.Ed25519PrivateKey) -> str:
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


class TestRegistry:
    def test_a_registry_with_a_path_is_rejected(self) -> None:
        """A host and a repository path are different things. Conflating them reads fine in a log
        and produces `https://host/project/repo/v2/...` the moment a registry client resolves it."""
        with pytest.raises(ValidationError, match="without a path"):
            BuilderConfig(registry="us-central1-docker.pkg.dev/proj/repo")

    def test_a_registry_host_is_lowercased(self) -> None:
        assert BuilderConfig(registry="Reg.Example.com:5000").registry == "reg.example.com:5000"

    def test_the_prefix_is_normalized_and_checked(self) -> None:
        assert BuilderConfig(repository_prefix="/proj/artifacts/").repository_prefix == "proj/artifacts"
        with pytest.raises(ValidationError, match="repository"):
            BuilderConfig(repository_prefix="proj/../elsewhere")

    def test_a_registry_is_not_assumed_to_narrow_scope(self) -> None:
        """Unverified for GAR and Artifactory, so rows say `registry-wide` until it is measured."""
        assert BuilderConfig().registry_narrows_scope is False


class TestTheBackend:
    def test_one_backend_per_deployment_and_execution_by_default(self) -> None:
        assert BuilderConfig().backend == "execution"

    def test_an_unimplemented_backend_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            BuilderConfig.model_validate({"backend": "cloud-build"})


class TestTheBrokerAndTheTrustRoot:
    def test_the_broker_is_an_http_url(self) -> None:
        config = BuilderConfig(credential_broker="http://nhx-build-broker.nhx-build-broker.svc:8080/")
        assert config.credential_broker == "http://nhx-build-broker.nhx-build-broker.svc:8080"

    @pytest.mark.parametrize("bad", ["nhx-build-broker:8080", "ftp://broker", "http://", "http://b?x=1"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="credential_broker"):
            BuilderConfig(credential_broker=bad)

    def test_the_public_key_is_parsed_when_the_platform_starts(self) -> None:
        pem = _pem(ec.generate_private_key(ec.SECP256R1()))
        assert BuilderConfig(signing_public_key=pem).signing_public_key == pem

    def test_a_key_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="PEM public key"):
            BuilderConfig(signing_public_key="-----BEGIN PUBLIC KEY-----\ntruncated\n-----END PUBLIC KEY-----\n")

    def test_a_key_cosign_does_not_sign_with_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="EC or RSA"):
            BuilderConfig(signing_public_key=_pem(ed25519.Ed25519PrivateKey.generate()))


def test_referrers_signature_storage_is_refused_not_ignored() -> None:
    """The broker writes the legacy `.sig` tag whatever this says, so accepting `referrers` would
    make every row misreport where its signature is."""
    with pytest.raises(ValidationError, match="not implemented"):
        BuilderConfig(signature_storage="referrers")

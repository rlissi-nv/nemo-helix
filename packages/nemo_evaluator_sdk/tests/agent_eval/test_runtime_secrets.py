# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving ``env_secrets`` to values for harnesses handed values, such as Gym."""

from __future__ import annotations

import os

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.secrets import env_secret_values, resolve_env_secrets
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef

REF = SecretRef("my-workspace/probe-api-key")


class _ValueOnlyResolver:
    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        return None


class _EnvOnlyResolver:
    def find_env_name(self, secret_ref: SecretRef, env_name: str) -> str | None:
        return env_name if os.environ.get(env_name) else None

    def missing_secret_message(self, secret_ref: SecretRef, env_name: str) -> str:
        return f"missing {secret_ref.root}"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MY_WORKSPACE_PROBE_API_KEY", "PROBE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_values_resolve_from_the_found_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_API_KEY", "b")
    assert await resolve_env_secrets({"LLM_API_KEY": REF}, LocalSecretResolver()) == {"LLM_API_KEY": "b"}


@pytest.mark.asyncio
async def test_a_missing_value_uses_the_resolver_message() -> None:
    with pytest.raises(ValueError, match=r"env_secrets\['LLM_API_KEY'\] -> secret .* is not found"):
        await resolve_env_secrets({"LLM_API_KEY": REF}, LocalSecretResolver())


@pytest.mark.asyncio
async def test_value_resolution_names_the_resolver_when_it_has_no_message() -> None:
    with pytest.raises(ValueError, match="could not be resolved by _ValueOnlyResolver"):
        await resolve_env_secrets({"LLM_API_KEY": REF}, _ValueOnlyResolver())


def test_env_secret_values_reads_named_variable_and_refuses_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "key")
    assert env_secret_values({"LLM_API_KEY": REF}, _EnvOnlyResolver()) == {"LLM_API_KEY": "key"}
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(ValueError, match="missing my-workspace/probe-api-key"):
        env_secret_values({"LLM_API_KEY": REF}, _EnvOnlyResolver())


@pytest.mark.asyncio
async def test_env_only_resolver_uses_named_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "job-key")
    assert await resolve_env_secrets({"LLM_API_KEY": REF}, _EnvOnlyResolver()) == {"LLM_API_KEY": "job-key"}


@pytest.mark.asyncio
async def test_custom_value_resolver_keeps_empty_value() -> None:
    class EmptyResolver:
        async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
            return ""

    assert await resolve_env_secrets({"LLM_API_KEY": REF}, EmptyResolver()) == {"LLM_API_KEY": ""}


@pytest.mark.asyncio
async def test_value_override_takes_precedence_over_env_source() -> None:
    class OverrideResolver(LocalSecretResolver):
        async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
            return "custom"

    assert await resolve_env_secrets({"LLM_API_KEY": REF}, OverrideResolver()) == {"LLM_API_KEY": "custom"}

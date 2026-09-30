# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``env_secrets`` to values for harnesses that are handed values, such as Gym."""

from __future__ import annotations

import os
from collections.abc import Mapping

from nemo_evaluator_sdk.resolver_protocols import EnvSecretSource, SecretResolver
from nemo_evaluator_sdk.values.common import SecretRef


def _missing_secret_message(resolver: SecretResolver | EnvSecretSource, secret_ref: SecretRef, env_name: str) -> str:
    if isinstance(resolver, EnvSecretSource):
        return resolver.missing_secret_message(secret_ref, env_name)
    return f"secret {secret_ref.root!r} could not be resolved by {type(resolver).__name__}."


def env_secret_values(env_secrets: Mapping[str, SecretRef], source: EnvSecretSource) -> dict[str, str]:
    """Read each secret from the environment variable named by ``source``."""
    values: dict[str, str] = {}
    for env_name, secret_ref in env_secrets.items():
        found = source.find_env_name(secret_ref, env_name)
        if found is None:
            raise ValueError(f"env_secrets[{env_name!r}] -> {source.missing_secret_message(secret_ref, env_name)}")
        values[env_name] = os.environ[found]
    return values


async def resolve_env_secrets(
    env_secrets: Mapping[str, SecretRef], resolver: SecretResolver | EnvSecretSource
) -> dict[str, str]:
    """Resolve ``env_secrets`` to values, keyed by the env var each one is handed over as."""
    if not isinstance(resolver, SecretResolver):
        return env_secret_values(env_secrets, resolver)
    resolved: dict[str, str] = {}
    for env_name, secret_ref in env_secrets.items():
        value = await resolver.resolve_secret(secret_ref)
        if value is None:
            raise ValueError(f"env_secrets[{env_name!r}] -> {_missing_secret_message(resolver, secret_ref, env_name)}")
        resolved[env_name] = value
    return resolved

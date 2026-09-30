# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared task-environment construction for evaluator plugin job compilers.

Both ``compile_evaluate_job`` and ``compile_agent_eval_job`` turn a spec's metric/endpoint secret
references into ``from_secret`` task environment variables. The per-job code differs only in *how*
it walks its own schema to collect ``(env_name, secret_name)`` pairs; the assembly — the
persistent-storage path, the reserved-name guard, and conflict detection — is identical and lives
here.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from nemo_evaluator_sdk.resolver_protocols import EnvSecretSource
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable, EnvironmentVariableFromSecret
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR

#: Step variable carrying the operator's sandbox decision to the job.
GYM_SANDBOX_PLAN_ENVVAR = "NEMO_EVALUATOR_GYM_SANDBOX_PLAN"

#: Env names a job sets itself, so they cannot be sourced from a secret ref.
RESERVED_SECRET_ENV_NAMES = frozenset({PERSISTENT_JOB_STORAGE_PATH_ENVVAR, GYM_SANDBOX_PLAN_ENVVAR})


class JobEnvSecretResolver(EnvSecretSource):
    """Name the environment variable holding each ``env_secrets`` entry in a platform job.

    The service injects every secret under its ``env_secrets`` key. ``find_env_name`` returns
    that variable's name when its value is non-empty; no other variable is consulted.
    This source names an existing variable rather than implementing ``SecretResolver``'s
    value-returning ``resolve_secret`` method.
    """

    def __init__(self, *, workspace: str) -> None:
        self._workspace = workspace

    def find_env_name(self, secret_ref: SecretRef, env_name: str) -> str | None:
        """``env_name`` when the service injected a non-empty value under it, else ``None``."""
        return env_name if os.environ.get(env_name) else None

    def missing_secret_message(self, secret_ref: SecretRef, env_name: str) -> str:
        """Point at the secret in its workspace: the ref's own, or the job's for a bare ref."""
        workspace, sep, name = secret_ref.root.rpartition("/")
        workspace = workspace if sep else self._workspace
        return (
            f"secret {secret_ref.root!r} was not injected into this job's environment. Check the secret exists "
            f"in workspace {workspace!r}: nemo secrets get {name} --workspace {workspace}"
        )


def build_task_environment(secret_refs: Iterable[tuple[str, str]]) -> list[EnvironmentVariable]:
    """Build a task's environment: the persistent-storage path plus ``from_secret`` variables.

    ``secret_refs`` yields ``(env_name, secret_name)`` pairs. Raises if a pair targets a reserved
    env name, or if two pairs map the same env name to different secrets.
    """
    environment = [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]
    resolved: dict[str, str] = {}
    for env_name, secret_name in secret_refs:
        if env_name in RESERVED_SECRET_ENV_NAMES:
            raise ValueError(f"{env_name!r} is reserved and cannot be sourced from secret refs")
        existing = resolved.get(env_name)
        if existing is not None and existing != secret_name:
            raise ValueError(
                f"conflicting secret references for environment variable {env_name!r}: {existing!r} and {secret_name!r}"
            )
        resolved[env_name] = secret_name

    environment.extend(
        EnvironmentVariable(name=env_name, from_secret=EnvironmentVariableFromSecret(name=secret_name))
        for env_name, secret_name in sorted(resolved.items())
    )
    return environment

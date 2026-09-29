# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment's one backend (Requirement 4), from its configuration.

Everything that acts on a backend's declarations -- the builder's routes, and the credential
broker -- loads it here, from the same ``builder:`` section, so they all act on one backend.
Adding a backend is an entry here and a value of ``builder.backend``; nothing that calls this
changes.
"""

from __future__ import annotations

from collections.abc import Callable

from nemo_builder_plugin.backend import Backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend

_BACKENDS: dict[str, Callable[[BuilderConfig], Backend]] = {ExecutionBackend.id: ExecutionBackend}


def load_backend(config: BuilderConfig) -> Backend:
    """The backend ``config.backend`` names."""
    return _BACKENDS[config.backend](config)

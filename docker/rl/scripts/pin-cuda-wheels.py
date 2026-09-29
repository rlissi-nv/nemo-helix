#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Point the image copy of NeMo-RL at the wheels mounted on /mnt/wheels.

Rewrites /opt/nemo-rl/pyproject.toml, then runs `uv lock` with no --upgrade
so only the packages whose source changed are relocked. The NeMo-RL fork and
NEMO_RL_REF are left alone; a checkout outside this image still builds from git.

The lock records each wheel filename and its sha256. It does not store a glob,
so /mnt/wheels has to be mounted again for every later `uv sync` of this lock.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

WHEELS = Path(os.environ.get("RL_CUDA_WHEELS", "/mnt/wheels"))

# (source key written into [tool.uv.sources], lock names, wheel filename prefixes)
PACKAGES = (
    ("causal-conv1d", ("causal-conv1d",), ("causal_conv1d-",)),
    ("mamba-ssm", ("mamba-ssm",), ("mamba_ssm-",)),
    ("nv-grouped-gemm", ("nv-grouped-gemm",), ("nv_grouped_gemm-",)),
    ("fast-hadamard-transform", ("fast-hadamard-transform",), ("fast_hadamard_transform-",)),
    ("flash_mla", ("flash-mla", "flash_mla"), ("flash_mla-",)),
    ("deep_ep", ("deep-ep", "deep_ep"), ("deep_ep-",)),
    ("deep_gemm", ("deep-gemm", "deep_gemm"), ("deep_gemm-",)),
    ("flash-attn", ("flash-attn",), ("flash_attn-",)),
    ("transformer-engine-torch", ("transformer-engine-torch",), ("transformer_engine_torch-",)),
)

DEEP_EP_GIT = "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e"
DEEP_GEMM_GIT = "deep_gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@67fc64863d43521080bf2005e6528d0fceee9510"
FAST_HADAMARD_GIT = (
    "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
    "@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"
)

NO_BUILD_OLD = """no-build-isolation-package = [
  "transformer-engine-torch",
  "flash-attn",
  "mamba-ssm",
  "causal-conv1d",
  "fast-hadamard-transform",
  "deep_gemm",
  "deep_ep",
  "flash_mla",
  "nv-grouped-gemm",          # from mlm (added here to make sure it's built no isolation since mlm workspace uses setup.py)
  "tensorrt-llm",             # custom PEP 517 backend calls build-custom-trtllm.sh; needs torch/CUDA from main venv
]"""

NO_BUILD_NEW = """no-build-isolation-package = [
  "tensorrt-llm",             # custom PEP 517 backend calls build-custom-trtllm.sh; needs torch/CUDA from main venv
]"""

EXTRA_BUILD_OLD = """# Augment build dependencies for packages that need torch at build time
[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]
# Git-sourced packages CAN use match-runtime = true if we provide dependency-metadata
deep_ep = [{ requirement = "torch", match-runtime = true }]
deep_gemm = [{ requirement = "torch", match-runtime = true }]
transformer-engine-torch = [{ requirement = "torch", match-runtime = true }]
mamba-ssm = [{ requirement = "torch", match-runtime = true }]
causal-conv1d = [{ requirement = "torch", match-runtime = true }]
fast-hadamard-transform = [{ requirement = "torch", match-runtime = true }]
nv-grouped-gemm = [{ requirement = "torch", match-runtime = true }]
flash_mla = [{ requirement = "torch", match-runtime = true }]
"""

EXTRA_BUILD_NEW = """# causal-conv1d, mamba-ssm, nv-grouped-gemm, fast-hadamard-transform, flash_mla,
# deep_ep, deep_gemm, flash-attn, and transformer-engine-torch install from
# /mnt/wheels, so they no longer need torch injected at build time.
"""

FLASH_ATTN_JOBS_OLD = """# flash-attn 2.8.1 has no prebuilt wheel for torch >= 2.11, so it compiles from the sdist.
# Its setup.py sizes MAX_JOBS from psutil's *host* free memory (not the container cgroup)
# and cpu_count()//2, then runs each nvcc with --threads 4; on nemo-ci's build runners that
# fan-out gets the CUDA compiler OOM-killed (`ResourceExhausted`, dozens of
# `Killed "$CICC_PATH/cicc"`) on both x86 and aarch64. 4 jobs x 4 threads built cleanly on
# both architectures (2h47m aarch64, ~2h x86 alongside TE/flash-mla).
flash-attn = { MAX_JOBS = "4" }
"""

ENVIRONMENTS_OLD = """environments = [
  "python_version == '3.13' and sys_platform == 'linux' and platform_machine == 'x86_64'",
  "python_version == '3.13' and sys_platform == 'linux' and platform_machine == 'aarch64'",
]"""

# Comments may still name these repos. Only a source assignment or a
# git+ requirement is a pin that would make uv compile again.
GONE_PINS = (
    'git = "https://github.com/Dao-AILab/causal-conv1d',
    'git = "https://github.com/state-spaces/mamba',
    'git = "https://github.com/fanshiqing/grouped_gemm',
    'git = "https://github.com/deepseek-ai/FlashMLA',
    "git+https://github.com/deepseek-ai/DeepEP",
    "git+https://github.com/deepseek-ai/DeepGEMM",
    "git+https://github.com/Dao-AILab/fast-hadamard-transform",
)


def must_replace(text: str, old: str, new: str, count: int | None = 1) -> str:
    found = text.count(old)
    if count is None:
        if found < 1:
            raise SystemExit(f"expected at least one occurrence of:\n{old}")
    elif found != count:
        raise SystemExit(f"expected {count} occurrence(s), found {found}, of:\n{old}")
    return text.replace(old, new)


def rewrite_pyproject(text: str, machine: str) -> str:
    if machine not in ("x86_64", "aarch64"):
        raise SystemExit(f"refusing to lock for platform_machine={machine!r}")

    environments = f"""environments = [
  "python_version == '3.13' and sys_platform == 'linux' and platform_machine == '{machine}'",
]
find-links = ["/mnt/wheels"]"""

    text = must_replace(text, ENVIRONMENTS_OLD, environments)
    text = must_replace(text, NO_BUILD_OLD, NO_BUILD_NEW)
    text = must_replace(text, EXTRA_BUILD_OLD, EXTRA_BUILD_NEW)
    text = must_replace(text, FLASH_ATTN_JOBS_OLD, "")

    # Bare names would resolve to whatever PyPI currently ships. Pin the
    # revisions this wheel image built. Git sources only attach to direct deps,
    # which is why mcore lists these next to megatron-bridge[ssm].
    text = must_replace(text, '  "mamba-ssm",', '  "mamba-ssm==2.2.6.post3",', 2)
    text = must_replace(text, '  "causal-conv1d",', '  "causal-conv1d==1.5.4",', 2)
    text = must_replace(text, '  "nv-grouped-gemm",', '  "nv-grouped-gemm==1.1.4.post7",', 1)
    text = must_replace(
        text,
        "  \"flash_mla ; sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')\",",
        "  \"flash_mla==1.0.0+b7643bd ; sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')\",",
        1,
    )

    text = must_replace(text, DEEP_EP_GIT, "deep_ep==1.2.1+17cfb81", 4)
    text = must_replace(text, DEEP_GEMM_GIT, "deep_gemm==2.5.0", 1)
    text = must_replace(text, FAST_HADAMARD_GIT, "fast-hadamard-transform==1.1.0", 1)

    text = must_replace(
        text,
        'causal-conv1d = { git = "https://github.com/Dao-AILab/causal-conv1d", rev = "4f6ae4e26ae5fe8af9372f8d312ab25cc4595223" }\n',
        "",
    )
    text = must_replace(
        text,
        'mamba-ssm = { git = "https://github.com/state-spaces/mamba.git", rev = "a14b1dff0454a3bc27d9eb31355dc01e4b2490ec" }\n',
        "",
    )
    text = must_replace(
        text,
        'nv-grouped-gemm = { git = "https://github.com/fanshiqing/grouped_gemm", tag = "v1.1.4.post7" }\n',
        "",
    )
    text = must_replace(
        text,
        "# FlashMLA derives its version from the Git revision in setup.py. Keep this pin\n"
        "# synchronized with the static dependency metadata below.\n"
        'flash_mla = { git = "https://github.com/deepseek-ai/FlashMLA", rev = "b7643bd54521f563b839b98289b5cd048c062ba2" }\n',
        "",
    )

    for pin in GONE_PINS:
        if pin in text:
            raise SystemExit(f"git pin still present after rewrite: {pin}")
    if "find-links" not in text or "/mnt/wheels" not in text:
        raise SystemExit("find-links = [/mnt/wheels] was not written")
    return text


def find_wheel(prefixes: tuple[str, ...]) -> Path:
    matches = [
        path
        for path in sorted(WHEELS.glob("*.whl"))
        if "cp313" in path.name and any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "(none)"
        raise SystemExit(f"expected one cp313 wheel in {WHEELS} matching {prefixes}, found: {names}")
    return matches[0]


def add_path_sources(text: str, source_keys: list[str]) -> str:
    header = "[tool.uv.sources]\n"
    if header not in text:
        raise SystemExit("pyproject.toml has no [tool.uv.sources] table")
    lines = []
    for source_key, _names, prefixes in PACKAGES:
        if source_key not in source_keys:
            continue
        if f"{source_key} = {{ path =" in text:
            continue
        wheel = find_wheel(prefixes)
        lines.append(f'{source_key} = {{ path = "{wheel}" }}')
    if not lines:
        return text
    return text.replace(header, header + "\n".join(lines) + "\n", 1)


def package_sections(lock_text: str) -> list[tuple[str, str]]:
    sections = []
    for part in lock_text.split("[[package]]")[1:]:
        name = ""
        for line in part.splitlines():
            if line.startswith("name = "):
                name = line.split("=", 1)[1].strip().strip('"')
                break
        sections.append((name, part))
    return sections


def wheels_missing_from_lock(lock_text: str) -> list[str]:
    sections = package_sections(lock_text)
    missing = []
    for source_key, names, _prefixes in PACKAGES:
        body = "\n".join(part for name, part in sections if name in names)
        if "/mnt/wheels/" not in body:
            missing.append(source_key)
    return missing


def run_lock(root: Path) -> subprocess.CompletedProcess[str]:
    print("+ uv lock", flush=True)
    proc = subprocess.run(
        ["uv", "lock"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.write(proc.stdout)
    return proc


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/nemo-rl")
    pyproject_path = root / "pyproject.toml"
    text = pyproject_path.read_text()
    machine = platform.machine()
    text = rewrite_pyproject(text, machine)
    pyproject_path.write_text(text)

    if os.environ.get("PIN_CUDA_WHEELS_REWRITE_ONLY"):
        return

    if not WHEELS.is_dir():
        raise SystemExit(f"{WHEELS} is not a directory")
    for _source_key, _names, prefixes in PACKAGES:
        find_wheel(prefixes)

    proc = run_lock(root)
    lock_path = root / "uv.lock"
    if proc.returncode != 0:
        conflict = any(
            marker in proc.stdout.lower() for marker in ("conflicting", "multiple source", "two source", "included as")
        )
        if not conflict:
            raise SystemExit(proc.returncode)
        print("uv lock reported conflicting sources; pinning the local wheels", flush=True)
        text = add_path_sources(pyproject_path.read_text(), [key for key, _, _ in PACKAGES])
        pyproject_path.write_text(text)
        proc = run_lock(root)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    missing = wheels_missing_from_lock(lock_path.read_text())
    if missing:
        print(
            "lock did not select /mnt/wheels for: " + ", ".join(missing) + "; pinning those paths",
            flush=True,
        )
        text = add_path_sources(pyproject_path.read_text(), missing)
        pyproject_path.write_text(text)
        proc = run_lock(root)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        missing = wheels_missing_from_lock(lock_path.read_text())
    if missing:
        raise SystemExit("uv.lock still does not install these from /mnt/wheels: " + ", ".join(missing))


if __name__ == "__main__":
    main()

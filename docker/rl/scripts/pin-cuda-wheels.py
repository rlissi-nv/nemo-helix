#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Point the image copy of NeMo-RL at the wheels mounted on /mnt/wheels.

Rewrites /opt/nemo-rl/pyproject.toml, then runs `uv lock` with no --upgrade
so only the packages whose source changed are relocked. The NeMo-RL fork and
NEMO_RL_REF are left alone; a checkout outside this image still builds from git.

The lock records each wheel filename and its sha256. It does not store a glob,
so /mnt/wheels has to be mounted again for every later `uv sync` of this lock.

Versions and git revisions come from docker/rl/cuda-wheel-versions.toml.
sync-cuda-wheel-versions.py rewrites the versions from a built wheel image.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tomllib
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


def deep_ep_git(rev: str) -> str:
    return f"deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@{rev}"


def deep_gemm_git(rev: str) -> str:
    return f"deep_gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@{rev}"


def fast_hadamard_git(rev: str) -> str:
    return f"fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@{rev}"


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


def retarget_metadata(text: str, metadata_name: str, version: str) -> str:
    """Point NeMo-RL's static requires-dist at the wheel version.

    The wheel's own metadata is the published one. mamba-ssm 2.3.2.post1
    requires tilelang==0.1.8, which conflicts with NeMo-RL's tilelang>=0.1.9.
    The static block is what the git pin used to avoid that, but it was keyed
    to the older version string, so uv ignored it.
    """
    pattern = re.compile(
        rf'(\[\[tool\.uv\.dependency-metadata\]\]\nname = "{re.escape(metadata_name)}"\n(?:#[^\n]*\n)*)version = "[^"]+"'
    )
    if pattern.search(text) is None:
        return text
    return pattern.sub(rf'\g<1>version = "{version}"', text, count=1)


def replace_exact_pin(text: str, name: str, version: str, count: int) -> str:
    pattern = re.compile(rf'"{re.escape(name)}==[^"]+"')
    found = pattern.findall(text)
    if len(found) != count:
        raise SystemExit(f"expected {count} {name}== pin(s), found {len(found)}: {found}")
    return pattern.sub(f'"{name}=={version}"', text)


def must_replace(text: str, old: str, new: str, count: int | None = 1) -> str:
    found = text.count(old)
    if count is None:
        if found < 1:
            raise SystemExit(f"expected at least one occurrence of:\n{old}")
    elif found != count:
        raise SystemExit(f"expected {count} occurrence(s), found {found}, of:\n{old}")
    return text.replace(old, new)


def versions_path() -> Path:
    override = os.environ.get("RL_CUDA_WHEEL_VERSIONS")
    if override:
        return Path(override)
    beside = Path(__file__).with_name("cuda-wheel-versions.toml")
    if beside.is_file():
        return beside
    return Path(__file__).resolve().parent.parent / "cuda-wheel-versions.toml"


def load_expected() -> dict[str, dict[str, str]]:
    path = versions_path()
    packages = tomllib.loads(path.read_text())["package"]
    expected = {item["name"]: item for item in packages}
    missing = [name for name, _lock_names, _prefixes in PACKAGES if name not in expected]
    if missing:
        raise SystemExit(f"{path} is missing packages: {', '.join(missing)}")
    for name, _lock_names, prefixes in PACKAGES:
        recorded = expected[name]["wheel_prefix"]
        if prefixes != (recorded,):
            raise SystemExit(f"{name} wheel_prefix {recorded} does not match {prefixes}")
    return expected


def rewrite_pyproject(text: str, machine: str, expected: dict[str, dict[str, str]]) -> str:
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
    # versions stamped into the wheels. Git sources only attach to direct deps,
    # which is why mcore lists these next to megatron-bridge[ssm].
    text = must_replace(text, '  "mamba-ssm",', f'  "mamba-ssm=={expected["mamba-ssm"]["version"]}",', 2)
    text = must_replace(text, '  "causal-conv1d",', f'  "causal-conv1d=={expected["causal-conv1d"]["version"]}",', 2)
    text = must_replace(
        text, '  "nv-grouped-gemm",', f'  "nv-grouped-gemm=={expected["nv-grouped-gemm"]["version"]}",', 1
    )
    flash_mla = expected["flash_mla"]["version"]
    text = must_replace(
        text,
        "  \"flash_mla ; sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')\",",
        f"  \"flash_mla=={flash_mla} ; sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')\",",
        1,
    )

    text = must_replace(text, deep_ep_git(expected["deep_ep"]["rev"]), f"deep_ep=={expected['deep_ep']['version']}", 4)
    text = must_replace(
        text,
        deep_gemm_git(expected["deep_gemm"]["rev"]),
        f"deep_gemm=={expected['deep_gemm']['version']}",
        1,
    )
    text = must_replace(
        text,
        fast_hadamard_git(expected["fast-hadamard-transform"]["rev"]),
        f"fast-hadamard-transform=={expected['fast-hadamard-transform']['version']}",
        1,
    )
    text = replace_exact_pin(text, "flash-attn", expected["flash-attn"]["version"], 2)
    text = replace_exact_pin(
        text,
        "transformer-engine[pytorch,core_cu13]",
        expected["transformer-engine-torch"]["version"],
        1,
    )

    text = must_replace(
        text,
        'causal-conv1d = { git = "https://github.com/Dao-AILab/causal-conv1d", rev = "'
        + expected["causal-conv1d"]["rev"]
        + '" }\n',
        "",
    )
    text = must_replace(
        text,
        'mamba-ssm = { git = "https://github.com/state-spaces/mamba.git", rev = "'
        + expected["mamba-ssm"]["rev"]
        + '" }\n',
        "",
    )
    text = must_replace(
        text,
        'nv-grouped-gemm = { git = "https://github.com/fanshiqing/grouped_gemm", tag = "'
        + expected["nv-grouped-gemm"]["rev"]
        + '" }\n',
        "",
    )
    text = must_replace(
        text,
        "# FlashMLA derives its version from the Git revision in setup.py. Keep this pin\n"
        "# synchronized with the static dependency metadata below.\n"
        'flash_mla = { git = "https://github.com/deepseek-ai/FlashMLA", rev = "'
        + expected["flash_mla"]["rev"]
        + '" }\n',
        "",
    )

    for name, record in expected.items():
        metadata_name = record.get("metadata_name")
        if metadata_name:
            text = retarget_metadata(text, metadata_name, record["version"])

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
    text = rewrite_pyproject(text, machine, load_expected())
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

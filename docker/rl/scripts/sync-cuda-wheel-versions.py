#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare rl-cuda-ext-wheel filenames with NeMo-RL and the wheel Dockerfile.

Version drift is a wheel filename whose version differs from
docker/rl/cuda-wheel-versions.toml. --update rewrites that version and the
matching comment in Dockerfile.python-wheels.

Revision drift is a NeMo-RL git rev or sdist pin that differs from the
Dockerfile. --update leaves `rev` and the sdist spec alone and still exits 1.
NeMo-RL's dependency-metadata versions are printed when they disagree with the
wheel; they are not a failure and they are not edited.

    python docker/rl/scripts/sync-cuda-wheel-versions.py \
        --wheels-dir /mnt/wheels \
        --nemo-rl /path/to/NeMo-RL/pyproject.toml

    python docker/rl/scripts/sync-cuda-wheel-versions.py \
        --image my-registry/rl-cuda-ext-wheel:TAG \
        --nemo-rl /path/to/NeMo-RL/pyproject.toml \
        --update
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_VERSIONS = REPO / "docker" / "rl" / "cuda-wheel-versions.toml"
DEFAULT_DOCKERFILE = REPO / "docker" / "base" / "Dockerfile.python-wheels"

# How NeMo-RL records the revision the wheel Dockerfile checks out.
# Every match in the file must be the same string.
NEMO_REV = {
    "causal-conv1d": r'causal-conv1d = \{ git = "[^"]+", rev = "([0-9a-f]+)" \}',
    "mamba-ssm": r'mamba-ssm = \{ git = "[^"]+", rev = "([0-9a-f]+)" \}',
    "nv-grouped-gemm": r'nv-grouped-gemm = \{ git = "[^"]+", tag = "([^"]+)" \}',
    "fast-hadamard-transform": (
        r"fast-hadamard-transform @ git\+https://github.com/Dao-AILab/fast-hadamard-transform.git@([0-9a-f]+)"
    ),
    "flash_mla": r'flash_mla = \{ git = "[^"]+", rev = "([0-9a-f]+)" \}',
    "deep_ep": r"deep_ep @ git\+https://github.com/deepseek-ai/DeepEP.git@([0-9a-f]+)",
    "deep_gemm": r"deep_gemm @ git\+https://github.com/deepseek-ai/DeepGEMM.git@([0-9a-f]+)",
}


def load_packages(text: str) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line == "[[package]]":
            current = {}
            packages.append(current)
            continue
        if not line or current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = value.strip().strip('"')
    if not packages:
        raise SystemExit("cuda-wheel-versions.toml has no [[package]] entries")
    return packages


def render_packages(text: str, versions: dict[str, str]) -> str:
    parts = re.split(r"(?=^\[\[package\]\])", text, flags=re.M)
    rendered = []
    for part in parts:
        if not part.startswith("[[package]]"):
            rendered.append(part)
            continue
        name_match = re.search(r'^name = "([^"]+)"', part, flags=re.M)
        if name_match is None or name_match.group(1) not in versions:
            rendered.append(part)
            continue
        version = versions[name_match.group(1)]
        updated, count = re.subn(
            r'^(version = ")[^"]*(")\s*$',
            rf"\g<1>{version}\2",
            part,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise SystemExit(f"{name_match.group(1)} has no version field to rewrite")
        rendered.append(updated)
    return "".join(rendered)


def wheel_version(filename: str, prefix: str) -> str | None:
    if not filename.startswith(prefix) or not filename.endswith(".whl"):
        return None
    pieces = filename[len(prefix) : -len(".whl")].split("-")
    # {version}-cp313-cp313-{platform}. Local versions keep the '+' from the build.
    if len(pieces) != 4 or pieces[1] != "cp313" or pieces[2] != "cp313":
        return None
    return pieces[0]


def versions_by_prefix(filenames: list[str], packages: list[dict[str, str]]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        prefix = package["wheel_prefix"]
        matched = []
        for filename in filenames:
            version = wheel_version(filename, prefix)
            if version is not None:
                matched.append((filename, version))
        unique = {version for _name, version in matched}
        if len(unique) != 1:
            names = ", ".join(name for name, _version in matched) or "(none)"
            raise SystemExit(f"expected one cp313 version for {prefix}, found: {names}")
        versions[package["name"]] = unique.pop()
    return versions


def wheel_names_from_image(image: str) -> list[str]:
    dest = Path(tempfile.mkdtemp(prefix="cuda-wheel-versions-"))
    try:
        subprocess.run(
            ["docker", "buildx", "build", "--output", f"type=local,dest={dest}", "-"],
            input=f"FROM {image}\n",
            text=True,
            check=True,
        )
        return [path.name for path in sorted(dest.rglob("*.whl"))]
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def dockerfile_value(text: str, env_name: str) -> str:
    match = re.search(rf"^[ \t]*{re.escape(env_name)}=(\S+)", text, flags=re.M)
    if match is None:
        raise SystemExit(f"{env_name} is not set in the wheel Dockerfile")
    return match.group(1)


def dockerfile_sdist_version(text: str, distribution: str) -> str:
    match = re.search(
        rf'build-rl-cuda-ext-wheel.sh sdist "{re.escape(distribution)}==([^"]+)"',
        text,
    )
    if match is None:
        raise SystemExit(f"wheel Dockerfile has no sdist build for {distribution}")
    return match.group(1)


def nemo_revs(text: str, name: str) -> list[str]:
    pattern = NEMO_REV.get(name)
    if pattern is None:
        return []
    found = re.findall(pattern, text)
    if not found:
        raise SystemExit(f"NeMo-RL pyproject has no revision for {name}")
    return found


def nemo_pin_versions(text: str, pin: str) -> list[str]:
    """Versions of an exact requirement. The name may differ from the wheel name."""
    name = pin.split("==", 1)[0]
    found = re.findall(rf"{re.escape(name)}==([^\"'\s]+)", text)
    if not found:
        raise SystemExit(f"NeMo-RL pyproject has no requirement starting with {name}==")
    return found


def metadata_version(text: str, metadata_name: str) -> str | None:
    pattern = re.compile(
        rf'\[\[tool\.uv\.dependency-metadata\]\]\nname = "{re.escape(metadata_name)}"\n(?:#[^\n]*\n)*version = "([^"]+)"'
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1)


def same_release(metadata: str, wheel: str) -> bool:
    return metadata.removeprefix("v") == wheel.removeprefix("v")


def update_comment(dockerfile: str, comment: str, old: str, new: str) -> tuple[str, bool]:
    pattern = re.compile(rf"^(# {re.escape(comment)} ){re.escape(old)}(?=\.|\s|$)", flags=re.M)
    updated, count = pattern.subn(rf"\g<1>{new}", dockerfile, count=1)
    return updated, count == 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheels-dir", type=Path, help="Directory of built wheels, searched recursively")
    source.add_argument("--image", help="Docker image whose /wheels/*.whl names are read")
    parser.add_argument("--nemo-rl", type=Path, required=True, help="NeMo-RL pyproject.toml")
    parser.add_argument("--wheels-dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    parser.add_argument("--versions", type=Path, default=DEFAULT_VERSIONS)
    parser.add_argument("--update", action="store_true", help="Rewrite version drift. Rev fields stay put.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    versions_text = args.versions.read_text()
    packages = load_packages(versions_text)
    dockerfile_text = args.wheels_dockerfile.read_text()
    nemo_text = args.nemo_rl.read_text()

    if args.image:
        wheel_names = wheel_names_from_image(args.image)
    else:
        wheel_names = [path.name for path in sorted(args.wheels_dir.rglob("*.whl"))]
        if not wheel_names:
            raise SystemExit(f"no wheels under {args.wheels_dir}")
    built = versions_by_prefix(wheel_names, packages)

    version_updates: dict[str, str] = {}
    comment_problems: list[str] = []
    revision_problems: list[str] = []
    metadata_notes: list[str] = []

    for package in packages:
        name = package["name"]
        wheel_version_found = built[name]
        expected = package["version"]
        if "rev" in package:
            dockerfile_rev = dockerfile_value(dockerfile_text, package["rev_env"])
            nemo = nemo_revs(nemo_text, name)
            if len(set(nemo)) != 1 or nemo[0] != dockerfile_rev:
                revision_problems.append(
                    f"rev  {name}: Dockerfile {package['rev_env']}={dockerfile_rev}  NeMo-RL {sorted(set(nemo))}"
                )
            elif package["rev"] != dockerfile_rev:
                revision_problems.append(
                    f"rev  {name}: cuda-wheel-versions.toml has {package['rev']}, Dockerfile has {dockerfile_rev}"
                )
            elif wheel_version_found != expected:
                version_updates[name] = wheel_version_found
                print(f"version  {name}: toml {expected}  wheel {wheel_version_found}")
        else:
            distribution = package["sdist"].split("==", 1)[0]
            sdist_version = dockerfile_sdist_version(dockerfile_text, distribution)
            recorded = package["sdist"].split("==", 1)[1]
            nemo_versions = nemo_pin_versions(nemo_text, package["nemo_pin"])
            nemo_version = package["nemo_pin"].split("==", 1)[1]
            pins_disagree = recorded != sdist_version or nemo_version != sdist_version
            if pins_disagree or any(item != sdist_version for item in nemo_versions):
                revision_problems.append(
                    f"pin  {name}: Dockerfile sdist {distribution}=={sdist_version}  "
                    f"NeMo-RL {sorted(set(nemo_versions))}  toml {recorded}"
                )
            elif wheel_version_found != sdist_version:
                revision_problems.append(
                    f"pin  {name}: wheel {wheel_version_found} does not match Dockerfile sdist {distribution}=={sdist_version}"
                )
            elif wheel_version_found != expected:
                version_updates[name] = wheel_version_found
                print(f"version  {name}: toml {expected}  wheel {wheel_version_found}")

        metadata = metadata_version(nemo_text, package.get("metadata_name", name))
        if metadata is not None and not same_release(metadata, wheel_version_found):
            metadata_notes.append(f"metadata  {name}: NeMo-RL {metadata}  wheel {wheel_version_found}")

    for note in metadata_notes:
        print(note)
    for problem in revision_problems:
        print(problem)

    if version_updates and args.update:
        args.versions.write_text(render_packages(versions_text, version_updates))
        print(f"updated {args.versions}")
        for package in packages:
            name = package["name"]
            if name not in version_updates:
                continue
            dockerfile_text, ok = update_comment(
                dockerfile_text, package["comment"], package["version"], version_updates[name]
            )
            if not ok:
                comment_problems.append(f"comment  {package['comment']} still says {package['version']}")
        if comment_problems:
            for problem in comment_problems:
                print(problem)
        else:
            args.wheels_dockerfile.write_text(dockerfile_text)
            print(f"updated {args.wheels_dockerfile}")

    if revision_problems or comment_problems:
        return 1
    if version_updates and not args.update:
        print("rerun with --update to write the wheel versions")
        return 1
    if not version_updates and not metadata_notes:
        print("wheel versions match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

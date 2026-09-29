# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the submit path asks an import's upstream registry: which manifest a pinned reference names.

One protocol, and it is not vendor-specific: Docker Hub, GHCR, GAR, ECR, ACR, Harbor, Quay and the
``distribution`` reference implementation all implement the **OCI Distribution Specification**.
So this is a client for that spec rather than for any registry.

**Anonymous, and read-only.** An import's source is a public registry the operator allowed
(``builder.import_registries``), and nothing about reading it needs a credential -- so the platform
holds none. It is the only registry read the control plane makes: a built image's digest reaches
a row through a verified signature, never through a read here.

Two notes that cost time to rediscover:

- The auth flow is a challenge. An unauthenticated request returns ``401`` with
  ``WWW-Authenticate: Bearer realm=...,service=...,scope=...``, and the realm issues an anonymous
  token for a public repository. Docker Hub does exactly this for every pull.
- Docker Hub's references say ``docker.io``, but its API is served at ``registry-1.docker.io``:
  ``https://docker.io/v2/`` redirects to the marketing site.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from nemo_builder_plugin.registry_auth import parse_challenges

logger = logging.getLogger(__name__)

#: Ask for every manifest media type, so an index and a plain manifest both resolve.
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

_INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: Registries whose API is not served at the hostname references use.
_API_HOSTS = {"docker.io": "registry-1.docker.io", "index.docker.io": "registry-1.docker.io"}


def _api_host(registry: str) -> str:
    return _API_HOSTS.get(registry, registry)


def _refuse_plain_http(request: httpx.Request) -> None:
    if request.url.scheme != "https":
        raise RegistryError(f"refusing a plain-HTTP request to {request.url.host}")


def _platform_matches(candidate: dict, platform: str) -> bool:
    """Whether an image's ``os``/``architecture``/``variant`` is ``os/arch[/variant]``.

    ``arm64`` with no variant IS ``arm64/v8``: buildx lists arm64 without one, and a caller who
    writes ``linux/arm64/v8`` means the same image.
    """
    os_name, _, rest = platform.partition("/")
    architecture, _, variant = rest.partition("/")
    if candidate.get("os") != os_name or candidate.get("architecture") != architecture:
        return False
    if not variant:
        return True
    have = candidate.get("variant") or ("v8" if architecture == "arm64" else "")
    return have == variant


def _select_manifest(manifests: list[dict], platform: str, index_digest: str) -> str:
    """The child of an index that matches ``os/arch[/variant]``.

    Attestation manifests carry ``unknown/unknown`` and so never match. No match is not a
    fallback to the first entry: the first entry of a multi-platform index is whatever the
    publisher listed first, and recording it would record an image for the wrong platform.
    """
    for manifest in manifests:
        if _platform_matches(manifest.get("platform") or {}, platform):
            return manifest["digest"]
    raise NoMatchingImage(f"index {index_digest} has no {platform} manifest")


class RegistryError(Exception):
    """The registry could not be reached, or answered in a way we do not accept."""


class ReferenceNotFound(RegistryError):
    """The reference does not resolve."""


class NoMatchingImage(ReferenceNotFound):
    """The reference resolves, but not to an image for the platform asked for."""


class RegistryDenied(RegistryError):
    """The registry refused the request: 401 or 403, after any challenge was answered."""


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """What one registry read observed."""

    digest: str
    manifest_digest: str
    media_type: str


class RegistryClient:
    """A small, anonymous OCI Distribution client: it resolves pinned references, and nothing else."""

    def __init__(self, *, timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        # `transport` is a test seam: an httpx.MockTransport stands in for a registry. Redirects
        # are followed -- Docker Hub serves blobs from a CDN that way -- but none may leave HTTPS:
        # the registry names the redirect, and cloud metadata servers answer only plain HTTP.
        hooks: dict[str, list[Callable[..., Any]]] = {"request": [_refuse_plain_http]}
        self._client = httpx.Client(timeout=timeout, follow_redirects=True, transport=transport, event_hooks=hooks)
        #: The bearer token that last answered each repository's challenge.
        self._authorization: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RegistryClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- auth ---------------------------------------------------------------

    def _anonymous_token(self, registry: str, repository: str, header: str) -> str | None:
        """An anonymous pull token for ``repository``, from the realm the challenge names.

        **The realm is the registry's to name, so it is checked before anything is sent to it:**
        HTTPS only. It is not echoed in errors, which surface in API responses.
        """
        params = dict(parse_challenges(header)).get("bearer") or {}
        realm = params.get("realm")
        if not realm:
            return None
        if urlsplit(realm).scheme != "https":
            raise RegistryError(f"{registry} named a token endpoint that is not HTTPS")
        query = {"service": params.get("service", registry), "scope": f"repository:{repository}:pull"}
        response = self._client.get(realm, params=query)
        if response.status_code != 200:
            raise RegistryError(f"{registry}'s token endpoint returned {response.status_code}")
        payload = response.json()
        token = payload.get("token") or payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise RegistryError(f"{registry}'s token endpoint returned no token")
        return token

    def _get(self, registry: str, repository: str, path: str, *, accept: str) -> httpx.Response:
        """GET, answering an auth challenge once; every failure to get an answer a RegistryError."""
        try:
            return self._get_answering_challenge(registry, repository, path, accept=accept)
        except httpx.HTTPError as exc:
            raise RegistryError(f"{registry}: {type(exc).__name__}: {exc}") from exc
        except ValueError as exc:  # `response.json()` on a body that is not JSON
            raise RegistryError(f"{registry} answered with something that is not JSON") from exc

    def _get_answering_challenge(self, registry: str, repository: str, path: str, *, accept: str) -> httpx.Response:
        """GET, answering a challenge once and remembering the token.

        **A 401 invalidates the remembered token**, and triggers exactly one fresh answer; a
        second 401 is returned to the caller rather than retried. The remembered token is also sent
        on the FIRST request, which saves the round trip every call would make to be challenged.
        """
        url = f"https://{_api_host(registry)}/v2/{repository}/{path}"
        cache_key = f"{registry}/{repository}"
        headers = {"Accept": accept}

        cached = self._authorization.get(cache_key)
        if cached:
            headers["Authorization"] = cached
        response = self._client.get(url, headers=headers)
        if response.status_code != 401:
            return response

        self._authorization.pop(cache_key, None)
        token = self._anonymous_token(registry, repository, response.headers.get("WWW-Authenticate", ""))
        if not token:
            return response
        self._authorization[cache_key] = headers["Authorization"] = f"Bearer {token}"
        return self._client.get(url, headers=headers)

    # -- reads --------------------------------------------------------------

    def resolve(self, registry: str, repository: str, reference: str, *, platform: str | None = None) -> ResolvedImage:
        """Resolve a tag (or digest) to the two digests that name the image.

        ``digest`` is whatever the reference resolved to. If that document is an **index**, it
        descends exactly one level for ``manifest_digest`` -- to the child matching ``platform``
        when one is given. If it is already a plain manifest, the two are equal.

        **The digest is computed from the bytes served, not read off a header.** A content
        address is only worth something if it addresses the content: taking the registry's
        ``Docker-Content-Digest`` at its word would let the body -- and the index child chosen
        from it -- be anything. The header, where a registry sends one, must agree.
        """
        response = self._get(registry, repository, f"manifests/{reference}", accept=MANIFEST_ACCEPT)
        if response.status_code == 404:
            raise ReferenceNotFound(f"{registry}/{repository}:{reference} does not resolve")
        if response.status_code in (401, 403):
            raise RegistryDenied(f"{registry}/{repository}:{reference} returned {response.status_code}")
        if response.status_code != 200:
            raise RegistryError(f"{registry}/{repository}:{reference} returned {response.status_code}")

        digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
        stated = response.headers.get("Docker-Content-Digest")
        if stated is not None and stated != digest:
            raise RegistryError(f"{registry}/{repository}:{reference} says it is {stated} but hashes to {digest}")
        if _DIGEST.fullmatch(reference) and digest != reference:
            # Asked for a digest, served something else. Content-addressed storage makes this
            # impossible for an honest registry, so it is refused rather than recorded.
            raise RegistryError(f"{registry}/{repository}@{reference} was served as {digest}")

        media_type = response.headers.get("Content-Type", "").split(";")[0]
        manifest_digest = digest
        if media_type in _INDEX_TYPES:
            manifests = self._index_entries(response, digest)
            manifest_digest = _select_manifest(manifests, platform, digest) if platform else manifests[0]["digest"]

        return ResolvedImage(digest=digest, manifest_digest=manifest_digest, media_type=media_type)

    @staticmethod
    def _index_entries(response: httpx.Response, digest: str) -> list[dict]:
        """An index's entries, each with a well-formed digest, or a RegistryError."""
        try:
            document = response.json()
        except ValueError as exc:
            raise RegistryError(f"index {digest} is not JSON") from exc
        manifests = document.get("manifests") if isinstance(document, dict) else None
        if not isinstance(manifests, list) or not manifests:
            raise RegistryError(f"index {digest} lists no manifests")
        for entry in manifests:
            if not isinstance(entry, dict) or not _DIGEST.fullmatch(str(entry.get("digest", ""))):
                raise RegistryError(f"index {digest} lists an entry without a sha256 digest")
        return manifests

    def resolve_import(self, registry: str, repository: str, digest: str, *, platform: str) -> ResolvedImage:
        """Resolve a caller's pinned import to the manifest for ``platform``, or refuse it.

        Stricter than :meth:`resolve` in two ways, both about a registry this deployment does not
        run:

        - **A single-platform manifest must BE for ``platform``.** Only an index lists platforms,
          so a plain manifest's is read from its config. Accepted unchecked, an arm64-only
          benchmark imported for ``linux/amd64`` would be published, signed, and recorded as
          amd64 -- and fail where it runs.
        - **Refused anonymous access is "not found".** A public registry answers a repository
          that does not exist, or is private, with 401 rather than 404. Either way the caller's
          reference does not resolve for this deployment.
        """
        try:
            resolved = self.resolve(registry, repository, digest, platform=platform)
        except RegistryDenied as exc:
            raise ReferenceNotFound(f"{registry}/{repository}@{digest} does not exist, or is not public") from exc
        if resolved.media_type not in _INDEX_TYPES:
            self._require_platform(registry, repository, resolved.digest, platform)
        return resolved

    def _require_platform(self, registry: str, repository: str, digest: str, platform: str) -> None:
        manifest = self._json(registry, repository, f"manifests/{digest}", digest, accept=MANIFEST_ACCEPT)
        config = manifest.get("config") if isinstance(manifest, dict) else None
        config_digest = config.get("digest") if isinstance(config, dict) else None
        if not isinstance(config_digest, str) or not _DIGEST.fullmatch(config_digest):
            raise RegistryError(f"manifest {digest} names no config")
        image_config = self._json(registry, repository, f"blobs/{config_digest}", config_digest, accept="*/*")
        if not isinstance(image_config, dict) or not _platform_matches(image_config, platform):
            found = (
                "/".join(
                    str(image_config.get(key)) for key in ("os", "architecture", "variant") if image_config.get(key)
                )
                if isinstance(image_config, dict)
                else "unknown"
            )
            raise NoMatchingImage(f"{registry}/{repository}@{digest} is a single {found} image, not {platform}")

    def _json(self, registry: str, repository: str, path: str, digest: str, *, accept: str) -> object:
        """A content-addressed document, verified against its digest before it is parsed."""
        response = self._get(registry, repository, path, accept=accept)
        if response.status_code != 200:
            raise RegistryError(f"{registry}/{repository} {path} returned {response.status_code}")
        if "sha256:" + hashlib.sha256(response.content).hexdigest() != digest:
            raise RegistryError(f"{registry}/{repository} served {path} with the wrong content")
        try:
            return response.json()
        except ValueError as exc:
            raise RegistryError(f"{registry}/{repository} {path} is not JSON") from exc

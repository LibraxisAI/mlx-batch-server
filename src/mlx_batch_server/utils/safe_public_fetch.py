"""Target-owned public HTTP(S) fetch boundary.

This module is the only network fetch surface for client-supplied public URLs.
It permits ordinary public ``http``/``https`` without an origin allowlist, and
it fails closed on credentials, non-HTTP schemes, localhost, private,
link-local, reserved, multicast, unspecified, documentation, cloud-metadata,
CGNAT and Tailscale targets. Every DNS answer and every redirect hop is
revalidated before the body is read.

Connect-time pinning: after DNS, the request URL host is rewritten to one
validated address and ``Host`` plus TLS ``sni_hostname`` keep the original
name. httpcore then ``connect_tcp``s to ``origin.host`` (the pinned IP), so a
later rebinding of the name cannot steer the socket. Do not treat DNS-only
prechecks as SSRF-safe.

Error messages are audit-safe: they never include resolved addresses,
sockaddrs, or redirect internals.

Vibecrafted. with AI Agents by Vetcoders (c)2024-2026 LibraxisAI
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx


@runtime_checkable
class FetchCancelCheck(Protocol):
    """Cooperative cancellation surface checked between hops and body chunks."""

    @property
    def cancelled(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6_NETWORK = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_METADATA_V4 = ipaddress.ip_address("169.254.169.254")
_METADATA_AWS_V6 = ipaddress.ip_address("fd00:ec2::254")
_BLOCKED_NETWORKS = (_CGNAT_NETWORK, _TAILSCALE_V6_NETWORK)
_METADATA_ADDRESSES = frozenset({_METADATA_V4, _METADATA_AWS_V6})
_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
    }
)
_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".invalid",
)
_USER_AGENT = "mlx-batch-server-media/1"


class SafePublicFetchError(ValueError):
    """Structured fail-closed error from the public fetch boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SafePublicFetchLimits:
    """Hard transport limits applied independently to every fetch hop."""

    max_bytes: int = 32 * 1024 * 1024
    timeout: float = 20.0
    connect_timeout: float = 5.0
    write_timeout: float = 5.0
    pool_timeout: float = 5.0
    max_redirects: int = 3
    chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if (
            min(
                self.timeout,
                self.connect_timeout,
                self.write_timeout,
                self.pool_timeout,
            )
            <= 0
        ):
            raise ValueError("timeouts must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")


@dataclass(frozen=True, slots=True)
class FetchedResource:
    """Sealed bytes plus audit-safe receipt metadata."""

    content: bytes
    media_type: str
    final_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("fetched content must be non-empty bytes")
        object.__setattr__(self, "media_type", _normalize_media_type(self.media_type))
        if not self.final_url:
            raise ValueError("final_url must be non-empty")


@dataclass(frozen=True, slots=True)
class _Hop:
    logical_url: str
    scheme: str
    hostname: str
    port: int
    origin: str


class SafePublicFetch:
    """Fetch a public HTTP(S) URL with DNS, redirect, and connect-time checks."""

    def __init__(
        self,
        *,
        limits: SafePublicFetchLimits = SafePublicFetchLimits(),
        allowed_origins: Iterable[str] = (),
        transport: httpx.AsyncBaseTransport | None = None,
        getaddrinfo: Callable[..., object] | None = None,
    ) -> None:
        self._limits = limits
        self._allowed_origins = frozenset(allowed_origins)
        self._transport = transport
        self._getaddrinfo = getaddrinfo or socket.getaddrinfo

    async def fetch(
        self,
        url: str,
        *,
        accepted_media_types: Sequence[str],
        max_bytes: int | None = None,
        cancel: FetchCancelCheck | None = None,
        deadline_s: float | None = None,
    ) -> FetchedResource:
        budget = self._limits.max_bytes if max_bytes is None else max_bytes
        if budget < 1:
            raise SafePublicFetchError(
                "invalid_fetch_budget",
                "URL fetch byte budget must be positive",
            )
        if deadline_s is not None and deadline_s <= 0:
            raise SafePublicFetchError(
                "url_fetch_timeout",
                "URL fetch exceeded its transport deadline",
            )
        accepted = _accepted_media_types(accepted_media_types)
        timeout = httpx.Timeout(
            connect=self._limits.connect_timeout,
            read=self._limits.timeout,
            write=self._limits.write_timeout,
            pool=self._limits.pool_timeout,
        )
        logical_url = url
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                if deadline_s is None:
                    return await self._fetch_hops(
                        client=client,
                        logical_url=logical_url,
                        accepted=accepted,
                        budget=budget,
                        cancel=cancel,
                    )
                async with asyncio.timeout(deadline_s):
                    return await self._fetch_hops(
                        client=client,
                        logical_url=logical_url,
                        accepted=accepted,
                        budget=budget,
                        cancel=cancel,
                    )
        except SafePublicFetchError:
            raise
        except TimeoutError as exc:
            raise SafePublicFetchError(
                "url_fetch_timeout",
                "URL fetch exceeded its transport deadline",
            ) from exc
        except httpx.TimeoutException as exc:
            raise SafePublicFetchError(
                "url_fetch_timeout",
                "URL fetch exceeded its transport deadline",
            ) from exc
        except httpx.HTTPError as exc:
            raise SafePublicFetchError(
                "url_fetch_failed",
                "URL fetch failed",
            ) from exc

    async def _fetch_hops(
        self,
        *,
        client: httpx.AsyncClient,
        logical_url: str,
        accepted: tuple[str, ...],
        budget: int,
        cancel: FetchCancelCheck | None = None,
    ) -> FetchedResource:
        current = logical_url
        for redirects in range(self._limits.max_redirects + 1):
            _raise_if_cancelled(cancel)
            hop = self._prepare_hop(current, redirect_hop=redirects > 0)
            pinned_ip = self._resolve_public_ip(hop)
            async with client.stream(
                "GET",
                _pinned_url(hop, pinned_ip),
                headers=_hop_headers(hop, accepted),
                extensions=_hop_extensions(hop),
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirects >= self._limits.max_redirects:
                        raise SafePublicFetchError(
                            "redirect_limit_exceeded",
                            "URL fetch exceeded its redirect limit",
                        )
                    location = response.headers.get("location")
                    if not location:
                        raise SafePublicFetchError(
                            "invalid_redirect",
                            "URL redirect is missing a location",
                        )
                    current = urljoin(hop.logical_url, location)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise SafePublicFetchError(
                        "url_fetch_status",
                        "URL fetch returned an unsuccessful HTTP status",
                    )
                media_type = _response_media_type(response)
                if media_type not in accepted:
                    raise SafePublicFetchError(
                        "unsupported_media_type",
                        "URL returned an unsupported media type",
                    )
                _validate_content_length(response, budget)
                content = await self._read_bounded(response, budget, cancel=cancel)
                return FetchedResource(
                    content=content,
                    media_type=media_type,
                    final_url=hop.logical_url,
                )
        raise SafePublicFetchError(
            "redirect_limit_exceeded",
            "URL fetch exceeded its redirect limit",
        )

    def _prepare_hop(self, url: str, *, redirect_hop: bool) -> _Hop:
        hop = _parse_http_url(url)
        if self._allowed_origins and hop.origin not in self._allowed_origins:
            if redirect_hop:
                raise SafePublicFetchError(
                    "redirect_not_allowed",
                    "URL redirect crossed the configured origin boundary",
                )
            raise SafePublicFetchError(
                "url_not_allowed",
                "URL origin is outside the configured lockdown",
            )
        return hop

    def _resolve_public_ip(
        self,
        hop: _Hop,
    ) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        try:
            literal = ipaddress.ip_address(hop.hostname)
        except ValueError:
            if _hostname_is_blocked(hop.hostname):
                raise SafePublicFetchError(
                    "url_target_blocked",
                    "URL target is not a public address",
                ) from None
            results = self._lookup(hop.hostname, hop.port)
            addresses = _addresses_from_addrinfo(results)
        else:
            addresses = (literal,)
        _require_public_addresses(addresses)
        return addresses[0]

    def _lookup(self, host: str, port: int) -> object:
        try:
            return self._getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise SafePublicFetchError(
                "dns_resolution_failed",
                "URL hostname could not be resolved",
            ) from exc

    async def _read_bounded(
        self,
        response: httpx.Response,
        max_bytes: int,
        *,
        cancel: FetchCancelCheck | None = None,
    ) -> bytes:
        content = bytearray()
        async for chunk in response.aiter_bytes(self._limits.chunk_bytes):
            _raise_if_cancelled(cancel)
            content.extend(chunk)
            if len(content) > max_bytes:
                raise SafePublicFetchError(
                    "source_bytes_exceeded",
                    "URL response exceeds the remaining source byte budget",
                )
        if not content:
            raise SafePublicFetchError(
                "empty_source",
                "URL response body must not be empty",
            )
        return bytes(content)


def _raise_if_cancelled(cancel: FetchCancelCheck | None) -> None:
    if cancel is not None and cancel.cancelled:
        raise asyncio.CancelledError(cancel.reason or "URL fetch cancelled")


def _parse_http_url(url: str) -> _Hop:
    if not isinstance(url, str) or not url.strip():
        raise SafePublicFetchError("invalid_url", "URL must be a non-empty string")
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SafePublicFetchError(
            "invalid_url_scheme",
            "URL scheme must be http or https",
        )
    if parsed.username is not None or parsed.password is not None:
        raise SafePublicFetchError(
            "url_credentials_forbidden",
            "URL must not include credentials",
        )
    if parsed.fragment:
        raise SafePublicFetchError("invalid_url", "URL must not include a fragment")
    hostname = parsed.hostname
    if not hostname:
        raise SafePublicFetchError("invalid_url", "URL must include a hostname")
    hostname = hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise SafePublicFetchError("invalid_url", "URL port is invalid") from exc
    if port is None:
        port = 80 if scheme == "http" else 443
    origin = _origin(scheme, hostname, port)
    logical = urlunsplit(
        (scheme, _netloc(hostname, port, scheme), parsed.path, parsed.query, "")
    )
    return _Hop(
        logical_url=logical,
        scheme=scheme,
        hostname=hostname,
        port=port,
        origin=origin,
    )


def _origin(scheme: str, hostname: str, port: int) -> str:
    default_port = 80 if scheme == "http" else 443
    suffix = "" if port == default_port else f":{port}"
    return f"{scheme}://{hostname}{suffix}"


def _netloc(hostname: str, port: int, scheme: str) -> str:
    default_port = 80 if scheme == "http" else 443
    host = _display_host(hostname)
    if port == default_port:
        return host
    return f"{host}:{port}"


def _display_host(hostname: str) -> str:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return hostname
    if address.version == 6:
        return f"[{address}]"
    return str(address)


def _pinned_url(
    hop: _Hop,
    pinned_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str:
    parsed = urlsplit(hop.logical_url)
    return urlunsplit(
        (
            hop.scheme,
            _netloc(str(pinned_ip), hop.port, hop.scheme),
            parsed.path,
            parsed.query,
            "",
        )
    )


def _hop_headers(hop: _Hop, accepted: tuple[str, ...]) -> dict[str, str]:
    default_port = 80 if hop.scheme == "http" else 443
    host = _display_host(hop.hostname)
    if hop.port != default_port:
        host = f"{host}:{hop.port}"
    return {
        "Host": host,
        "Accept": ", ".join(accepted),
        "User-Agent": _USER_AGENT,
    }


def _hop_extensions(hop: _Hop) -> dict[str, str]:
    try:
        ipaddress.ip_address(hop.hostname)
    except ValueError:
        if hop.scheme != "https":
            return {}
        return {"sni_hostname": hop.hostname.encode("idna").decode("ascii")}
    return {}


def _hostname_is_blocked(hostname: str) -> bool:
    if hostname in _BLOCKED_HOSTS:
        return True
    return any(hostname.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES)


def _addresses_from_addrinfo(
    results: object,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    if not isinstance(results, Sequence) or isinstance(results, str | bytes):
        raise SafePublicFetchError(
            "dns_resolution_failed",
            "URL hostname could not be resolved",
        )
    if not results:
        raise SafePublicFetchError(
            "dns_resolution_failed",
            "URL hostname could not be resolved",
        )
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    try:
        for item in results:
            sockaddr = item[4]
            raw = sockaddr[0]
            address = ipaddress.ip_address(raw)
            mapped = _canonical_ip(address)
            key = mapped.compressed
            if key in seen:
                continue
            seen.add(key)
            addresses.append(mapped)
    except (IndexError, TypeError, ValueError) as exc:
        raise SafePublicFetchError(
            "dns_resolution_failed",
            "URL hostname could not be resolved",
        ) from exc
    if not addresses:
        raise SafePublicFetchError(
            "dns_resolution_failed",
            "URL hostname could not be resolved",
        )
    return tuple(addresses)


def _canonical_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _require_public_addresses(
    addresses: Sequence[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> None:
    if any(_address_is_blocked(address) for address in addresses):
        raise SafePublicFetchError(
            "url_target_blocked",
            "URL target is not a public address",
        )


def _address_is_blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    address = _canonical_ip(address)
    if address in _METADATA_ADDRESSES:
        return True
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    ):
        return True
    return any(address in network for network in _BLOCKED_NETWORKS)


def _accepted_media_types(values: Sequence[str]) -> tuple[str, ...]:
    accepted = tuple(sorted({_normalize_media_type(value) for value in values}))
    if not accepted:
        raise SafePublicFetchError(
            "invalid_fetch_media_types",
            "URL fetch requires accepted media types",
        )
    return accepted


def _normalize_media_type(value: str) -> str:
    if not isinstance(value, str):
        raise SafePublicFetchError("invalid_media_type", "media type must be explicit")
    normalized = value.split(";", 1)[0].strip().lower()
    if not normalized or "/" not in normalized:
        raise SafePublicFetchError("invalid_media_type", "media type must be explicit")
    return normalized


def _response_media_type(response: httpx.Response) -> str:
    value = response.headers.get("content-type")
    if value is None:
        raise SafePublicFetchError(
            "missing_media_type",
            "URL response requires an explicit Content-Type",
        )
    return _normalize_media_type(value)


def _validate_content_length(response: httpx.Response, max_bytes: int) -> None:
    value = response.headers.get("content-length")
    if value is None:
        return
    try:
        content_length = int(value)
    except ValueError as exc:
        raise SafePublicFetchError(
            "invalid_content_length",
            "URL response Content-Length must be an integer",
        ) from exc
    if content_length < 0:
        raise SafePublicFetchError(
            "invalid_content_length",
            "URL response Content-Length must not be negative",
        )
    if content_length > max_bytes:
        raise SafePublicFetchError(
            "source_bytes_exceeded",
            "URL response exceeds the remaining source byte budget",
        )


__all__ = [
    "FetchCancelCheck",
    "FetchedResource",
    "SafePublicFetch",
    "SafePublicFetchError",
    "SafePublicFetchLimits",
]

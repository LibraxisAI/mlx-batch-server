"""Fail-closed public model alias resolution against the signed role directory."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..runtime.roles import RoleDirectory, UnknownRoleError
from .runtime_mapper import ResolvedRuntime, ResponsesMappingError

if TYPE_CHECKING:
    from ..runtime.contracts import RoleName, RuntimeKey


class ManifestRuntimeResolver:
    """Resolve only explicitly configured aliases into manifest-owned runtimes."""

    def __init__(
        self,
        roles: RoleDirectory,
        aliases: Mapping[str, RoleName | str],
    ) -> None:
        if not isinstance(roles, RoleDirectory):
            raise TypeError("roles must be a RoleDirectory")
        if not isinstance(aliases, Mapping) or not aliases:
            raise ValueError("aliases must be a non-empty mapping")

        resolved: dict[str, RoleName] = {}
        for raw_alias, raw_role in aliases.items():
            if not isinstance(raw_alias, str) or not raw_alias.strip():
                raise ValueError("runtime aliases must be non-empty strings")
            alias = raw_alias.strip()
            lookup = alias.casefold()
            if lookup in resolved:
                raise ValueError(
                    f"duplicate runtime alias after normalization: {alias!r}"
                )
            try:
                spec = roles.resolve(raw_role)
            except UnknownRoleError as exc:
                raise ValueError(
                    f"runtime alias {alias!r} references an unknown role"
                ) from exc
            resolved[lookup] = spec.name

        self._roles = roles
        self._lock = threading.RLock()
        self._alias_entries = resolved
        self._aliases = MappingProxyType(self._alias_entries)

    @property
    def aliases(self) -> Mapping[str, RoleName]:
        """Return a read-only live view of the process-local alias map."""

        return self._aliases

    def register_alias(self, alias: str, target: str) -> RoleName:
        """Bind an alias to a manifest-owned model without changing its role.

        The signed manifest remains authoritative for model identity. Runtime
        aliases may only point at an exact configured model or an existing
        alias, so this operation cannot redirect a process to another runtime.
        """

        normalized_alias = _required(alias, "alias")
        normalized_target = _required(target, "model")
        target_lookup = normalized_target.casefold()
        with self._lock:
            role = self._alias_entries.get(target_lookup)
            if role is None:
                matching_roles = tuple(
                    spec.name
                    for spec in self._roles.specs()
                    if spec.requested_model == normalized_target
                )
                if len(matching_roles) != 1:
                    raise ResponsesMappingError(
                        "runtime aliases may target only the manifest-owned model",
                        code="runtime_alias_target_forbidden",
                        param="model",
                    )
                role = matching_roles[0]

            alias_lookup = normalized_alias.casefold()
            current = self._alias_entries.get(alias_lookup)
            if current is not None and current is not role:
                raise ResponsesMappingError(
                    f"runtime alias {normalized_alias!r} already targets another role",
                    code="runtime_alias_conflict",
                    param="alias",
                )
            self._alias_entries[alias_lookup] = role
            return role

    def __call__(
        self,
        *,
        model: str,
        role: str | None,
        revision: str | None,
        adapter_path: str | None,
        draft_model_id: str | None,
        backend: str | None,
    ) -> ResolvedRuntime:
        requested_model = _required(model, "model")
        lookup = requested_model.casefold()
        with self._lock:
            resolved_role = self._alias_entries.get(lookup)
        if resolved_role is None:
            raise ResponsesMappingError(
                f"model alias {requested_model!r} is not configured",
                code="unknown_model_alias",
                param="model",
            )

        if role is not None and _required(role, "runtime_role") != resolved_role.value:
            raise ResponsesMappingError(
                "runtime_role cannot redirect the configured model alias",
                code="runtime_role_mismatch",
                param="runtime_role",
            )

        overrides = (
            ("revision", revision),
            ("adapter_path", adapter_path),
            ("draft_model_id", draft_model_id),
            ("backend", backend),
        )
        for field, value in overrides:
            if value is not None:
                raise ResponsesMappingError(
                    f"{field} is controlled by the signed runtime role manifest",
                    code="runtime_override_forbidden",
                    param=field,
                )

        runtime: RuntimeKey = self._roles.runtime_key(resolved_role)
        return ResolvedRuntime(
            runtime=runtime,
            requested_model=requested_model,
            role=resolved_role.value,
        )


def _required(value: object, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResponsesMappingError(
            f"{param} must be a non-empty string",
            code="invalid_responses_request",
            param=param,
        )
    return value.strip()


__all__ = ["ManifestRuntimeResolver"]

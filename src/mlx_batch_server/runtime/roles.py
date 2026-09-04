"""Role identity and deterministic model-runtime resolution."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .contracts import BackendKind, RoleName, RoleSnapshot, RoleSpec, RuntimeKey


class UnknownRoleError(KeyError):
    """Raised when a caller requests a role absent from the frozen directory."""


class RoleDirectory:
    """Immutable role definitions owned by the target control plane."""

    def __init__(self, specs: Iterable[RoleSpec]) -> None:
        by_name: dict[RoleName, RoleSpec] = {}
        by_port: dict[int, RoleSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate role {spec.name.value!r}")
            if spec.port < 1 or spec.port > 65535:
                raise ValueError(f"invalid port for role {spec.name.value!r}")
            if spec.port in by_port:
                other = by_port[spec.port]
                raise ValueError(
                    f"port {spec.port} belongs to both "
                    f"{other.name.value!r} and {spec.name.value!r}"
                )
            if not spec.requested_model.strip():
                raise ValueError(f"role {spec.name.value!r} has an empty model")
            if spec.revision is not None and (
                not spec.revision.strip() or spec.revision != spec.revision.strip()
            ):
                raise ValueError(
                    f"role {spec.name.value!r} has a non-canonical revision"
                )
            if spec.model_dir is not None:
                model_dir = Path(spec.model_dir)
                if not model_dir.is_absolute() or str(model_dir) != spec.model_dir:
                    raise ValueError(
                        f"role {spec.name.value!r} has a non-canonical model_dir"
                    )
            if (spec.revision is None) != (spec.model_dir is None):
                raise ValueError(
                    f"role {spec.name.value!r} must bind revision and model_dir together"
                )
            by_name[spec.name] = spec
            by_port[spec.port] = spec
        self._by_name = by_name
        self._by_port = by_port

    @staticmethod
    def _name(role: RoleName | str) -> RoleName:
        if isinstance(role, RoleName):
            return role
        try:
            return RoleName(str(role).strip().lower())
        except ValueError as exc:
            raise UnknownRoleError(str(role)) from exc

    def resolve(self, role: RoleName | str) -> RoleSpec:
        name = self._name(role)
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise UnknownRoleError(name.value) from exc

    def resolve_port(self, port: int) -> RoleSpec:
        try:
            return self._by_port[port]
        except KeyError as exc:
            raise UnknownRoleError(f"port:{port}") from exc

    def runtime_key(
        self,
        role: RoleName | str,
        *,
        revision: str | None = None,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        backend: BackendKind | None = None,
    ) -> RuntimeKey:
        spec = self.resolve(role)
        return RuntimeKey(
            model_id=spec.requested_model,
            revision=spec.revision if revision is None else revision,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            backend=backend or spec.backend,
        )

    def specs(self) -> tuple[RoleSpec, ...]:
        return tuple(self._by_name.values())

    def __contains__(self, role: object) -> bool:
        try:
            name = self._name(role)  # type: ignore[arg-type]
        except (UnknownRoleError, TypeError):
            return False
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)


__all__ = [
    "RoleDirectory",
    "RoleName",
    "RoleSnapshot",
    "RoleSpec",
    "UnknownRoleError",
]

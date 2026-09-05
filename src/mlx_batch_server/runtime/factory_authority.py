"""Closed capability binding for factories allowed to publish fused READY."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from .contracts import BackendFactory, TensorMaterializationReceipt

if TYPE_CHECKING:
    from .contracts import (
        BackendHandle,
        CapabilityReport,
        LoadConfig,
        ModelSpec,
        RuntimeKey,
    )

_FactoryBinder = Callable[[BackendFactory, object], BackendFactory]
_FactoryUnwrapper = Callable[
    [BackendFactory],
    tuple[BackendFactory, object | None],
]


def _build_factory_authority_domain() -> tuple[_FactoryBinder, _FactoryUnwrapper]:
    bindings: WeakKeyDictionary[
        object,
        tuple[BackendFactory, object],
    ] = WeakKeyDictionary()

    class _TrustedFusedBackendFactory:
        __slots__ = ("__weakref__",)

        def probe(self, model: ModelSpec) -> CapabilityReport:
            factory, _ = bindings[self]
            return factory.probe(model)

        async def load(
            self,
            runtime: RuntimeKey,
            config: LoadConfig,
        ) -> BackendHandle:
            factory, _ = bindings[self]
            return await factory.load(runtime, config)

    def bind(factory: BackendFactory, receipt_authority: object) -> BackendFactory:
        """Bind a concrete factory to its private materialization issuer."""

        if receipt_authority is None:
            raise ValueError("trusted fused factory requires receipt authority")
        binding = _TrustedFusedBackendFactory()
        bindings[binding] = (factory, receipt_authority)
        return binding

    def unwrap(
        factory: BackendFactory,
    ) -> tuple[BackendFactory, object | None]:
        """Return the delegate and seal only for a binding minted in this domain."""

        if type(factory) is not _TrustedFusedBackendFactory:
            return factory, None
        try:
            return bindings[factory]
        except KeyError:
            return factory, None

    return bind, unwrap


(
    _bind_trusted_fused_backend_factory,
    _unwrap_trusted_fused_backend_factory,
) = _build_factory_authority_domain()
del _build_factory_authority_domain


def _receipt_matches_factory_authority(
    receipt: TensorMaterializationReceipt,
    authority: object | None,
) -> bool:
    """Authenticate a receipt against the authority captured at composition."""

    if authority is None:
        return False
    authenticate = getattr(authority, "_authenticates", None)
    if callable(authenticate):
        return bool(authenticate(receipt))
    # Explicitly private binding support for protocol-isolated test factories.
    return receipt._issued_by(authority)

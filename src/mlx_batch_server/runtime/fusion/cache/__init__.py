"""Target-owned cache seams for the fused scheduler backend."""

from .contracts import (
    CacheBindingReceipt,
    CacheCleanupReceipt,
    CacheInvalidationReceipt,
    CacheLayout,
    CacheLeaseState,
    CacheNamespace,
    CacheReleaseReason,
    CacheTier,
    PagedCachePort,
    PrefixCachePort,
    SSDCachePort,
)
from .identity import CACHE_SIGNATURE_SCHEMA, build_cache_namespace
from .lifecycle import FusionCacheCoordinator, FusionCacheLease

__all__ = [
    "CACHE_SIGNATURE_SCHEMA",
    "CacheBindingReceipt",
    "CacheCleanupReceipt",
    "CacheInvalidationReceipt",
    "CacheLayout",
    "CacheLeaseState",
    "CacheNamespace",
    "CacheReleaseReason",
    "CacheTier",
    "FusionCacheCoordinator",
    "FusionCacheLease",
    "PagedCachePort",
    "PrefixCachePort",
    "SSDCachePort",
    "build_cache_namespace",
]

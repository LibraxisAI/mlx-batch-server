"""Source-only per-request Qwen4Exp vision contracts."""

from .mrope import MropePlan, build_mrope_plan, mrope_plan_digest
from .processing import (
    MAX_REQUEST_IMAGES,
    ImageGridReceipt,
    OpaqueRows,
    ProcessedVisionBatch,
    VisionContractError,
    VisionPreprocessorPort,
    VisionProcessingRequest,
    VisionRequestIdentity,
    validate_preprocessing_output,
)
from .splice import (
    VisionImageSpan,
    VisionSpliceCursor,
    VisionSplicePlan,
    VisionSpliceWindow,
    build_content_key_surrogates,
    build_image_spans,
    build_vision_splice_plan,
)
from .tower import (
    DeepstackFeatureReceipt,
    VisionEmbeddingSlice,
    VisionTowerOutput,
    VisionTowerPort,
    VisionTowerRequest,
    validate_tower_output,
)

__all__ = (
    "MAX_REQUEST_IMAGES",
    "DeepstackFeatureReceipt",
    "ImageGridReceipt",
    "MropePlan",
    "OpaqueRows",
    "ProcessedVisionBatch",
    "VisionContractError",
    "VisionEmbeddingSlice",
    "VisionImageSpan",
    "VisionPreprocessorPort",
    "VisionProcessingRequest",
    "VisionRequestIdentity",
    "VisionSpliceCursor",
    "VisionSplicePlan",
    "VisionSpliceWindow",
    "VisionTowerOutput",
    "VisionTowerPort",
    "VisionTowerRequest",
    "build_content_key_surrogates",
    "build_image_spans",
    "build_mrope_plan",
    "build_vision_splice_plan",
    "mrope_plan_digest",
    "validate_preprocessing_output",
    "validate_tower_output",
)

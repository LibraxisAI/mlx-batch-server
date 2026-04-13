from typing import Any

from pydantic import BaseModel, Field, model_serializer


class Model(BaseModel):
    """Model information as per OpenAI API specification"""

    id: str = Field(..., description="The model identifier")
    object: str = Field(default="model", description="The object type (always 'model')")
    created: int = Field(
        ..., description="Unix timestamp of when the model was created"
    )
    owned_by: str = Field(..., description="Organization that owns the model")
    details: dict[str, Any] | None = Field(
        default=None, description="Full model configuration (if details are requested)"
    )

    @model_serializer
    def serialize_model(self):
        """Custom serializer to exclude None details field"""
        data = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }
        if self.details is not None:
            data["details"] = self.details
        return data


class ModelList(BaseModel):
    """Response format for list of models"""

    object: str = Field(default="list", description="The object type (always 'list')")
    data: list[Model] = Field(..., description="List of model objects")


class ModelDeletion(BaseModel):
    """Response format for model deletion"""

    id: str = Field(..., description="The ID of the deleted model")
    object: str = Field(default="model", description="The object type (always 'model')")
    deleted: bool = Field(..., description="Whether the model was deleted")


# === Model Load/Unload Schemas ===


class ModelLoadRequest(BaseModel):
    """Request body for loading a model into memory."""

    model: str = Field(
        ..., description="Model ID to load (HuggingFace ID or local path)"
    )
    task: str | None = Field(
        default=None,
        description=(
            "Optional task hint (llm, embeddings, visual, images, stt, tts). "
            "If omitted, the server will attempt to auto-detect."
        ),
    )
    adapter_path: str | None = Field(
        default=None, description="Optional path to LoRA adapter"
    )
    draft_model_id: str | None = Field(
        default=None, description="Optional draft model for speculative decoding"
    )
    alias: str | None = Field(
        default=None,
        description="Optional runtime alias to register for this model in the current server process",
    )


class ModelLoadResponse(BaseModel):
    """Response for model load operation."""

    id: str = Field(..., description="The loaded model ID")
    object: str = Field(default="model", description="Object type")
    task: str | None = Field(default=None, description="Resolved task for the model")
    status: str = Field(..., description="Load status: 'loaded' or 'already_loaded'")
    message: str = Field(..., description="Human-readable status message")
    cache_info: dict[str, Any] | None = Field(
        default=None, description="Current cache state"
    )


class ModelUnloadRequest(BaseModel):
    """Request body for unloading a model from memory."""

    model: str | None = Field(
        default=None,
        description="Model ID to unload. If not provided, unloads all models.",
    )
    task: str | None = Field(
        default=None,
        description=(
            "Optional task hint (llm, embeddings, visual, images, stt, tts). "
            "If omitted, the server will attempt to auto-detect."
        ),
    )


class ModelUnloadResponse(BaseModel):
    """Response for model unload operation."""

    task: str | None = Field(default=None, description="Resolved task for unload")
    status: str = Field(..., description="Unload status")
    message: str = Field(..., description="Human-readable status message")
    unloaded_models: list[str] = Field(
        default_factory=list, description="List of unloaded model IDs"
    )
    cache_info: dict[str, Any] | None = Field(
        default=None, description="Current cache state after unload"
    )

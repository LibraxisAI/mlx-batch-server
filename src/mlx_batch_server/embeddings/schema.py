from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelType(str, Enum):
    BERT = "bert"


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    input: str | list[str] = Field(..., description="Input text to get embeddings for")
    encoding_format: str | None = Field(
        "float", description="The format of the embeddings"
    )
    user: str | None = None
    dimensions: int | None = None

    # Allow any additional fields
    model_config = ConfigDict(extra="allow")

    def get_extra_params(self) -> dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = {"model", "input", "encoding_format", "user", "dimensions"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage

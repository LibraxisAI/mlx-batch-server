"""RED contracts for lossless Responses multimodal input normalization."""

from mlx_batch_server.responses.normalizer import (
    has_media_content,
    normalise_responses_payload,
    parts_to_plaintext,
)
from mlx_batch_server.responses.schema import ContentPart, ContentPartType


def test_input_file_reference_is_preserved_as_media() -> None:
    body = normalise_responses_payload(
        {
            "model": "buddy",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": "file_lab_result",
                            "filename": "lab-result.pdf",
                            "detail": "high",
                        }
                    ],
                }
            ],
        }
    )

    assert body["input"][0]["content"] == [
        {
            "type": "input_file",
            "file_id": "file_lab_result",
            "filename": "lab-result.pdf",
            "detail": "high",
        }
    ]
    assert has_media_content(body) is True


def test_inline_file_data_is_not_stringified_or_truncated() -> None:
    file_data = "data:application/pdf;base64,JVBERi0xLjQK"
    body = normalise_responses_payload(
        {
            "model": "buddy",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_data": file_data,
                            "filename": "record.pdf",
                        }
                    ],
                }
            ],
        }
    )

    assert body["input"][0]["content"][0]["file_data"] == file_data
    assert body["input"][0]["content"][0]["type"] == "input_file"


def test_content_part_schema_accepts_file_inputs() -> None:
    part = ContentPart(
        type=ContentPartType.INPUT_FILE,
        file_url="https://example.test/lab-result.pdf",
        filename="lab-result.pdf",
    )

    assert part.type is ContentPartType.INPUT_FILE
    assert part.file_url == "https://example.test/lab-result.pdf"


def test_file_plaintext_fallback_discloses_reference_without_payload_dump() -> None:
    data = "data:application/pdf;base64," + ("A" * 100)

    rendered = parts_to_plaintext(
        [{"type": "input_file", "file_data": data, "filename": "record.pdf"}]
    )

    assert rendered == "[File: record.pdf]"
    assert data not in rendered

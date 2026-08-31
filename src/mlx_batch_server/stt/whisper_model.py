import os
import tempfile
from pathlib import Path

from mlx_whisper import transcribe
from mlx_whisper.writers import WriteSRT, WriteVTT

from ..aux_runtime import (
    forget_aux_runtime,
    forget_aux_runtime_lane,
    replace_aux_runtime,
)
from .schema import (
    ResponseFormat,
    STTRequestForm,
    TranscriptionResponse,
    TranscriptionWord,
)


class WhisperModel:
    async def _save_upload_file(self, file) -> str:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            return tmp.name

    def generate(self, audio_path: str, request: STTRequestForm):
        word_timestamps = False
        if request.timestamp_granularities:
            word_timestamps = "word" in [
                g.value for g in request.timestamp_granularities
            ]

        print(f"word_timestamps: {word_timestamps}")
        result = transcribe(
            audio=audio_path,
            path_or_hf_repo=request.model,
            temperature=request.temperature,
            initial_prompt=request.prompt,
            language=request.language,
            word_timestamps=word_timestamps,
            verbose=False,
            condition_on_previous_text=True,
        )
        return result

    def _generate_subtitle_file(self, result: dict, format: str) -> str:
        temp_dir = None
        temp_file = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, f"temp.{format}")

            if format == "srt":
                writer = WriteSRT(temp_dir)
            else:  # vtt
                writer = WriteVTT(temp_dir)

            writer(result, temp_file)

            with open(temp_file, encoding="utf-8") as f:
                return f.read()

        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def _format_response(
        self, result: dict, request: STTRequestForm
    ) -> dict | str | TranscriptionResponse:
        if request.response_format == ResponseFormat.TEXT:
            return result["text"]

        elif request.response_format == ResponseFormat.SRT:
            return self._generate_subtitle_file(result, "srt")

        elif request.response_format == ResponseFormat.VTT:
            return self._generate_subtitle_file(result, "vtt")

        elif request.response_format == ResponseFormat.VERBOSE_JSON:
            return result

        elif request.response_format == ResponseFormat.JSON:
            return {"text": result["text"]}

        else:
            text = result.get("text", "")
            language = result.get("language", "en")

            duration = 0
            if "segments" in result:
                for segment in result["segments"]:
                    if "end" in segment:
                        duration = max(duration, segment["end"])

            words = []
            if request.timestamp_granularities and "word" in [
                g.value for g in request.timestamp_granularities
            ]:
                for segment in result.get("segments", []):
                    for word_data in segment.get("words", []):
                        word = TranscriptionWord(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                        )
                        words.append(word)

            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                text=text,
                words=words if words else None,
            )


class STTService:
    def __init__(self):
        self.model = WhisperModel()

    async def transcribe(
        self,
        request: STTRequestForm,
    ) -> dict | str | TranscriptionResponse:
        try:
            preload_whisper_model(request.model)
            audio_path = await self.model._save_upload_file(request.file)
            result = self.model.generate(audio_path=audio_path, request=request)
            response = self.model._format_response(result, request)
            Path(audio_path).unlink(missing_ok=True)
            return response

        except Exception as e:
            if "audio_path" in locals():
                Path(audio_path).unlink(missing_ok=True)
            raise e


def preload_whisper_model(model_id: str) -> bool:
    """Preload a Whisper model into the global cache. Returns True if already loaded."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    previous_model_id = (
        ModelHolder.model_path if ModelHolder.model is not None else None
    )
    already_loaded = previous_model_id == model_id
    ModelHolder.get_model(model_id, mx.float16)
    if previous_model_id and previous_model_id != model_id:
        forget_aux_runtime("stt", previous_model_id)
    replace_aux_runtime("stt", model_id, lambda: _retire_whisper_model(model_id))
    return already_loaded


def _retire_whisper_model(model_id: str | None = None) -> list[str]:
    """Drop Whisper weights without mutating shared lifecycle metadata."""
    from mlx_whisper.transcribe import ModelHolder

    if ModelHolder.model is None:
        return []

    if model_id is None or ModelHolder.model_path == model_id:
        unloaded = [ModelHolder.model_path] if ModelHolder.model_path else []
        ModelHolder.model = None
        ModelHolder.model_path = None
        return unloaded

    return []


def unload_whisper_model(model_id: str | None = None) -> list[str]:
    """Unload Whisper model(s). Returns unloaded model IDs."""
    if model_id is None:
        forget_aux_runtime_lane("stt")
    else:
        forget_aux_runtime("stt", model_id)
    return _retire_whisper_model(model_id)


def get_loaded_whisper_models() -> list[str]:
    """Return the actual native Whisper holder residency."""
    from mlx_whisper.transcribe import ModelHolder

    if ModelHolder.model is None or not ModelHolder.model_path:
        return []
    return [ModelHolder.model_path]

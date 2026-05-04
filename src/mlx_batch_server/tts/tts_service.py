from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

from f5_tts_mlx.generate import generate
from mlx_audio.tts.generate import generate_audio
from pydantic import BaseModel, Field  # , PrivateAttr
from typing_extensions import override

from .schema import TTSRequest


class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""

    path_or_hf_repo: str | Path = Field(
        None, title="The path or the huggingface repository to load the model from."
    )

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        """
        Generate audio from input text.

        Args:
            request (TTSRequest): The request object containing the input text and other parameters.
            output_path (str | Path): The path to save the generated audio file.

        Returns:
            bool: True if the audio was generated successfully, False otherwise.
        """
        pass

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str) -> "TTSModelAdapter":
        if path_or_hf_repo == "lucasnewman/f5-tts-mlx":
            return F5Model(path_or_hf_repo=path_or_hf_repo)
        else:
            return MlxAudioModel(path_or_hf_repo=path_or_hf_repo)


class F5Model(TTSModelAdapter):
    _model_cache: ClassVar[dict[str, Any]] = {}
    _cache_lock: ClassVar[Lock] = Lock()

    @classmethod
    def _get_cached_model(cls, model_id: str) -> Any | None:
        with cls._cache_lock:
            return cls._model_cache.get(model_id)

    @classmethod
    def preload_model(cls, model_id: str) -> bool:
        """Preload an F5 TTS model. Returns True if it was already loaded."""
        with cls._cache_lock:
            if model_id in cls._model_cache:
                return True
        from f5_tts_mlx.cfm import F5TTS

        model = F5TTS.from_pretrained(model_id)
        with cls._cache_lock:
            cls._model_cache[model_id] = model
        return False

    @classmethod
    def unload_model(cls, model_id: str | None = None) -> list[str]:
        """Unload F5 models from cache and return unloaded model IDs."""
        with cls._cache_lock:
            if model_id:
                if model_id in cls._model_cache:
                    cls._model_cache.pop(model_id, None)
                    return [model_id]
                return []
            unloaded = list(cls._model_cache.keys())
            cls._model_cache.clear()
            return unloaded

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        cached = self._get_cached_model(request.model)
        if cached is None:
            generate(
                model_name=request.model,
                generation_text=request.input,
                speed=request.speed,
                output_path=str(output_path),
                **(request.get_extra_params() or {}),
            )
            return Path(output_path).exists()

        from f5_tts_mlx import cfm as f5_cfm

        with self._cache_lock:
            original = f5_cfm.F5TTS.from_pretrained

            def _from_pretrained(cls, *args, **kwargs):
                return cached

            f5_cfm.F5TTS.from_pretrained = classmethod(_from_pretrained)
            try:
                generate(
                    model_name=request.model,
                    generation_text=request.input,
                    speed=request.speed,
                    output_path=str(output_path),
                    **(request.get_extra_params() or {}),
                )
            finally:
                f5_cfm.F5TTS.from_pretrained = original
        return Path(output_path).exists()


class MlxAudioModel(TTSModelAdapter):
    path_or_hf_repo: str = Field("mlx-community/Kokoro-82M-4bit")
    _model_cache: ClassVar[dict[str, Any]] = {}
    _cache_lock: ClassVar[Lock] = Lock()

    @classmethod
    def _get_cached_model(cls, model_id: str) -> Any | None:
        with cls._cache_lock:
            return cls._model_cache.get(model_id)

    @classmethod
    def preload_model(cls, model_id: str) -> bool:
        """Preload an mlx-audio TTS model. Returns True if it was already loaded."""
        with cls._cache_lock:
            if model_id in cls._model_cache:
                return True

        from mlx_audio.tts.utils import load_model

        model = load_model(model_path=model_id)
        with cls._cache_lock:
            cls._model_cache[model_id] = model
        return False

    @classmethod
    def unload_model(cls, model_id: str | None = None) -> list[str]:
        """Unload mlx-audio models from cache and return unloaded model IDs."""
        with cls._cache_lock:
            if model_id:
                if model_id in cls._model_cache:
                    cls._model_cache.pop(model_id, None)
                    return [model_id]
                return []
            unloaded = list(cls._model_cache.keys())
            cls._model_cache.clear()
            return unloaded

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        voice = request.voice if hasattr(request, "voice") else "af_sky"
        lang_code = voice[:1]

        extra_params = request.get_extra_params() or {}
        cached = self._get_cached_model(self.path_or_hf_repo)
        model_arg = cached if cached is not None else self.path_or_hf_repo

        generate_audio(
            text=request.input,
            model=model_arg,
            voice=voice,
            speed=request.speed,
            lang_code=lang_code,
            file_prefix=str(output_path).rsplit(".", 1)[0],
            audio_format=request.response_format.value,
            sample_rate=24000,
            join_audio=True,
            verbose=False,
            **extra_params,
        )

        return Path(output_path).exists()


class TTSService:
    model: TTSModelAdapter

    @classmethod
    def preload_model(cls, model_id: str) -> bool:
        """Preload a TTS model. Returns True if it was already loaded."""
        adapter = TTSModelAdapter.from_path_or_hf_repo(model_id)
        if isinstance(adapter, F5Model):
            return F5Model.preload_model(model_id)
        return MlxAudioModel.preload_model(model_id)

    @classmethod
    def unload_model(cls, model_id: str | None = None) -> list[str]:
        """Unload cached TTS model(s) and return unloaded model IDs."""
        unloaded = []
        unloaded.extend(F5Model.unload_model(model_id))
        unloaded.extend(MlxAudioModel.unload_model(model_id))
        return list(dict.fromkeys(unloaded))

    def __init__(self, path_or_hf_repo: str | Path | None = None):
        self.model = TTSModelAdapter.from_path_or_hf_repo(path_or_hf_repo)
        self.sample_audio_path = Path("sample.wav")

    async def generate_speech(
        self,
        request: TTSRequest,
    ) -> bytes:
        try:
            self.model.generate_audio(
                request=request, output_path=self.sample_audio_path
            )
            with open(self.sample_audio_path, "rb") as audio_file:
                audio_content = audio_file.read()
            self.sample_audio_path.unlink(missing_ok=True)
            return audio_content
        except Exception as e:
            raise Exception(f"Error reading audio file: {e!s}") from e

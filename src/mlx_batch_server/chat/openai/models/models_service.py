import importlib
import json

from huggingface_hub import CachedRepoInfo, scan_cache_dir

from ....utils.logger import logger
from .schema import Model, ModelDeletion, ModelList

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
}


class ModelCacheScanner:
    """Scanner for finding and managing mlx-lm compatible models in the local cache."""

    def __init__(self):
        self._cache_info = None

    @property
    def cache_info(self):
        """Lazy load and cache the scan_cache_dir result"""
        if self._cache_info is None:
            self._cache_info = scan_cache_dir()
        return self._cache_info

    def _refresh_cache_info(self):
        """Force refresh the cache info"""
        self._cache_info = scan_cache_dir()

    def _get_model_classes(self, config: dict) -> tuple[type, type] | None:
        """
        Try to retrieve the model and model args classes based on the configuration.
        https://github.com/ml-explore/mlx-examples/blob/1e0766018494c46bc6078769278b8e2a360503dc/llms/mlx_lm/utils.py#L81

        Args:
            config (dict): The model configuration

        Returns:
            Optional tuple of (Model class, ModelArgs class) if model type is supported
        """
        try:
            model_type = config.get("model_type")
            model_type = MODEL_REMAPPING.get(model_type, model_type)
            if not model_type:
                return None

            # Try to import the model architecture module
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
            return arch.Model, arch.ModelArgs

        except ImportError:
            logger.debug(f"Model type {model_type} not supported by mlx-lm")
            return None
        except Exception as e:
            logger.warning(f"Error checking model compatibility: {e!s}")
            return None

    def is_model_supported(self, config_data: dict) -> bool:
        return self._get_model_classes(config_data) is not None

    def find_models_in_cache(self) -> list[tuple[CachedRepoInfo, dict]]:
        """
        Scan local cache for available models that are compatible with mlx-lm.

        Returns:
            List of tuples containing (CachedRepoInfo, config_dict)
        """
        supported_models = []

        for repo_info in self.cache_info.repos:
            if repo_info.repo_type != "model":
                continue

            first_revision = next(iter(repo_info.revisions), None)
            if not first_revision:
                continue

            config_file = next(
                (f for f in first_revision.files if f.file_name == "config.json"), None
            )
            if not config_file:
                continue

            try:
                with open(config_file.file_path) as f:
                    config_data = json.load(f)
                if self.is_model_supported(config_data):
                    supported_models.append((repo_info, config_data))
            except Exception as e:
                logger.error(
                    f"Error reading config.json for {repo_info.repo_id}: {e!s}"
                )

        return supported_models

    def get_model_info(self, model_id: str) -> tuple[CachedRepoInfo, dict] | None:
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id and repo_info.repo_type == "model":
                first_revision = next(iter(repo_info.revisions), None)
                if not first_revision:
                    continue

                config_file = next(
                    (f for f in first_revision.files if f.file_name == "config.json"),
                    None,
                )
                if not config_file:
                    continue

                try:
                    with open(config_file.file_path) as f:
                        config_data = json.load(f)
                    if self.is_model_supported(config_data):
                        return (repo_info, config_data)
                    else:
                        logger.warning(
                            f"Model {model_id} found but not compatible with mlx-lm"
                        )
                except Exception as e:
                    logger.error(
                        f"Error reading config.json for {repo_info.repo_id}: {e!s}"
                    )

        return None

    def delete_model(self, model_id: str) -> bool:
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id:
                revision_hashes = [rev.commit_hash for rev in repo_info.revisions]
                if not revision_hashes:
                    return False

                try:
                    delete_strategy = self.cache_info.delete_revisions(*revision_hashes)
                    logger.info(
                        f"Model '{model_id}': Will free {delete_strategy.expected_freed_size_str}"
                    )
                    delete_strategy.execute()
                    logger.info(f"Model '{model_id}': Cache deletion completed")
                    self._refresh_cache_info()
                    return True
                except Exception as e:
                    logger.error(f"Error deleting model '{model_id}': {e!s}")
                    raise

        return False


class ModelsService:
    def __init__(self):
        self.scanner = ModelCacheScanner()
        self.available_models = self._scan_models()

    def _scan_models(self) -> list[tuple[CachedRepoInfo, dict]]:
        """Scan local cache for available CausalLM models"""
        try:
            return self.scanner.find_models_in_cache()
        except Exception as e:
            print(f"Error scanning cache: {e!s}")
            return []

    @staticmethod
    def _get_model_owner(model_id: str) -> str:
        """Extract owner from model ID (part before the /)"""
        return model_id.split("/")[0] if "/" in model_id else model_id

    def list_models(self, include_details: bool = False) -> ModelList:
        """List all available models"""
        models = []
        for repo_info, config_data in self.available_models:
            model_kwargs = {
                "id": repo_info.repo_id,
                "created": int(repo_info.last_modified),
                "owned_by": self._get_model_owner(repo_info.repo_id),
            }
            if include_details:
                model_kwargs["details"] = config_data
            model_instance = Model(**model_kwargs)
            models.append(model_instance)
        return ModelList(data=models)

    def get_model(self, model_id: str, include_details: bool = False) -> Model | None:
        """Get information about a specific model"""
        model_info = self.scanner.get_model_info(model_id)
        if model_info:
            repo_info, config_data = model_info
            model_kwargs = {
                "id": model_id,
                "created": int(repo_info.last_modified),
                "owned_by": self._get_model_owner(model_id),
            }
            if include_details:
                model_kwargs["details"] = config_data
            return Model(**model_kwargs)
        return None

    def delete_model(self, model_id: str) -> ModelDeletion:
        """Delete a model from local cache"""
        if not self.scanner.delete_model(model_id):
            raise ValueError(f"Model '{model_id}' not found in cache")

        self.available_models = self._scan_models()
        return ModelDeletion(id=model_id, deleted=True)

    def load_model(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> dict:
        """Load a model into memory (MLX cache).

        Args:
            model_id: Model ID to load (HuggingFace ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model for speculative decoding

        Returns:
            Dict with load status and cache info
        """
        # Lazy import to avoid circular dependencies and startup overhead
        from ...mlx.wrapper_cache import wrapper_cache

        # Check if already loaded
        already_loaded = wrapper_cache.is_model_loaded(model_id)

        # Load model (get_wrapper handles caching)
        wrapper_cache.get_wrapper(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )

        return {
            "id": model_id,
            "status": "already_loaded" if already_loaded else "loaded",
            "message": (
                f"Model {model_id} was already loaded"
                if already_loaded
                else f"Model {model_id} loaded successfully"
            ),
            "cache_info": wrapper_cache.get_cache_info(),
        }

    def unload_model(self, model_id: str | None = None) -> dict:
        """Unload a model from memory.

        Args:
            model_id: Model ID to unload. If None, unloads all models.

        Returns:
            Dict with unload status and cache info
        """
        # Lazy import to avoid circular dependencies and startup overhead
        from ...mlx.wrapper_cache import wrapper_cache

        unloaded_models = []

        if model_id:
            # Unload specific model
            if wrapper_cache.unload_model(model_id):
                unloaded_models.append(model_id)
                status = "unloaded"
                message = f"Model {model_id} unloaded successfully"
            else:
                status = "not_found"
                message = f"Model {model_id} was not loaded"
        else:
            # Unload all models
            unloaded_models = wrapper_cache.get_loaded_models()
            wrapper_cache.clear_cache()
            status = "cleared"
            message = f"Cleared {len(unloaded_models)} model(s) from cache"

        return {
            "status": status,
            "message": message,
            "unloaded_models": unloaded_models,
            "cache_info": wrapper_cache.get_cache_info(),
        }

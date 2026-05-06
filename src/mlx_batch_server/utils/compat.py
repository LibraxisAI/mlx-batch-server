"""
Compatibility patches for deprecated APIs in dependencies.

Patches pkg_resources usage in jieba to use importlib.resources instead.
Must be imported BEFORE any module that imports jieba (e.g., f5_tts_mlx).

Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path


def _create_jieba_compat() -> types.ModuleType:
    """
    Create a patched jieba._compat module using importlib.resources.

    Replaces deprecated pkg_resources.resource_stream with modern API.
    This is a full reimplementation of jieba/_compat.py without pkg_resources.
    """
    compat = types.ModuleType("jieba._compat")
    compat.__file__ = "<patched by mlx_batch_server>"

    # jieba/__init__.py does "from ._compat import *" and expects these modules
    compat.os = os
    compat.sys = sys
    compat.logging = logging

    # Logging setup (same as original)
    log_console = logging.StreamHandler(sys.stderr)
    default_logger = logging.getLogger("jieba._compat")
    default_logger.setLevel(logging.DEBUG)

    def setLogLevel(log_level: int) -> None:
        default_logger.setLevel(log_level)

    compat.log_console = log_console
    compat.default_logger = default_logger
    compat.setLogLevel = setLogLevel

    # Paddle integration (same as original)
    compat.check_paddle_install = {"is_paddle_installed": False}

    def enable_paddle() -> None:
        try:
            import paddle
        except ImportError:
            default_logger.debug("Installing paddle-tiny, please wait a minute......")
            os.system("pip install paddlepaddle-tiny")
            try:
                import paddle
            except ImportError:
                default_logger.debug(
                    "Import paddle error, please use command to install: "
                    "pip install paddlepaddle-tiny==1.6.1. "
                    "Now, back to jieba basic cut......"
                )
                return

        if paddle.__version__ < "1.6.1":  # type: ignore[attr-defined]
            default_logger.debug(
                "Find your own paddle version doesn't satisfy the minimum "
                "requirement (1.6.1), please install paddle tiny by "
                "'pip install --upgrade paddlepaddle-tiny', or upgrade paddle "
                "full version by 'pip install --upgrade paddlepaddle "
                "(-gpu for GPU version)' "
            )
        else:
            try:
                from importlib.util import find_spec

                if find_spec("jieba.lac_small.predict"):
                    default_logger.debug("Paddle enabled successfully......")
                    compat.check_paddle_install["is_paddle_installed"] = True
            except ImportError:
                default_logger.debug(
                    "Import error, cannot find paddle.fluid and "
                    "jieba.lac_small.predict module. "
                    "Now, back to jieba basic cut......"
                )

    compat.enable_paddle = enable_paddle

    # Python 2/3 compatibility (jieba still supports Python 2)
    compat.PY2 = sys.version_info[0] == 2
    compat.default_encoding = sys.getfilesystemencoding()

    # Python 3 only (we don't support Python 2)
    compat.text_type = str
    compat.string_types = (str,)
    compat.xrange = range
    compat.iterkeys = lambda d: iter(d.keys())
    compat.itervalues = lambda d: iter(d.values())
    compat.iteritems = lambda d: iter(d.items())

    def strdecode(sentence: str | bytes) -> str:
        if not isinstance(sentence, str):
            try:
                sentence = sentence.decode("utf-8")
            except UnicodeDecodeError:
                sentence = sentence.decode("gbk", "ignore")
        return sentence

    compat.strdecode = strdecode

    def resolve_filename(f) -> str:
        try:
            return f.name
        except AttributeError:
            return repr(f)

    compat.resolve_filename = resolve_filename

    # Modern resource loading - NO pkg_resources, NO deprecation warning
    def get_module_res(*res: str):
        """Load jieba resources using importlib.resources (Python 3.9+)."""
        try:
            # Use importlib.resources.files() - the modern API
            resource_path = importlib.resources.files("jieba").joinpath(*res)
            return resource_path.open("rb")
        except Exception:
            # Fallback: direct file access (when jieba is installed as editable)
            try:
                import jieba as jieba_module

                base_path = os.path.dirname(jieba_module.__file__ or "")
                return open(os.path.join(base_path, *res), "rb")  # noqa: PTH118, PTH123
            except Exception:
                raise FileNotFoundError(f"Cannot find jieba resource: {res}")

    compat.get_module_res = get_module_res

    # __all__ for "from ._compat import *" to work correctly
    compat.__all__ = [
        "os",
        "sys",
        "logging",
        "log_console",
        "default_logger",
        "setLogLevel",
        "check_paddle_install",
        "enable_paddle",
        "get_module_res",
        "PY2",
        "default_encoding",
        "text_type",
        "string_types",
        "xrange",
        "iterkeys",
        "itervalues",
        "iteritems",
        "strdecode",
        "resolve_filename",
    ]

    return compat


def patch_jieba() -> bool:
    """
    Patch jieba._compat to avoid pkg_resources deprecation warning.

    Returns:
        True if patched, False if jieba was already imported.
    """
    if "jieba" in sys.modules:
        # Too late - jieba already imported
        return False

    if "jieba._compat" in sys.modules:
        # Already patched or imported
        return False

    # Install our patched _compat before jieba imports it
    sys.modules["jieba._compat"] = _create_jieba_compat()
    return True


def _patch_jieba_regex_file(path: Path, replacements: dict[str, str]) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    updated = text
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)

    if updated == text:
        return False

    try:
        path.write_text(updated, encoding="utf-8")
    except OSError:
        return False

    return True


def patch_jieba_regex_warnings() -> bool:
    """
    Patch jieba regex literals to avoid Python 3.12 SyntaxWarning.

    This updates the installed jieba source files in-place, replacing a few
    regex string literals with raw strings to avoid invalid escape warnings.
    """
    try:
        spec = importlib.util.find_spec("jieba")
    except Exception:
        return False

    if not spec or not spec.origin:
        return False

    base_path = Path(spec.origin).parent
    patched = False

    patched |= _patch_jieba_regex_file(
        base_path / "__init__.py",
        {
            're_han_default = re.compile("([\\u4E00-\\u9FD5a-zA-Z0-9+#&\\._%\\-]+)", re.U)': 're_han_default = re.compile(r"([\\u4E00-\\u9FD5a-zA-Z0-9+#&\\._%\\-]+)", re.U)',
            're_skip_default = re.compile("(\\r\\n|\\s)", re.U)': 're_skip_default = re.compile(r"(\\r\\n|\\s)", re.U)',
        },
    )

    patched |= _patch_jieba_regex_file(
        base_path / "finalseg" / "__init__.py",
        {
            're_skip = re.compile("([a-zA-Z0-9]+(?:\\.\\d+)?%?)")': 're_skip = re.compile(r"([a-zA-Z0-9]+(?:\\.\\d+)?%?)")',
        },
    )

    return patched


def patch_mlx_lm_logging() -> bool:
    """
    Patch mlx_lm.utils to add missing 'logging' import.

    mlx-lm has a bug where it uses 'logging.error()' without importing logging.
    This patch imports mlx_lm.utils and injects the logging module.

    Returns:
        True if patched, False if patch failed.
    """
    try:
        # Import mlx_lm.utils to patch it
        # This is safe - mlx_lm is a required dependency
        import mlx_lm.utils as mlx_utils

        # Check if logging is already in the module's namespace
        if hasattr(mlx_utils, "logging"):
            return False

        # Inject logging into mlx_lm.utils namespace
        mlx_utils.logging = logging
        return True
    except ImportError:
        # mlx-lm not installed - skip patching
        return False
    except Exception as e:
        # Log but don't crash - this is just a compatibility fix
        print(f"Warning: Failed to patch mlx_lm.utils: {e}")
        return False


def patch_transformers_video_processor_mapping() -> bool:
    """
    Patch transformers video processor mapping to avoid NoneType iteration.

    Transformers 5.0.0rc1 can set VIDEO_PROCESSOR_MAPPING_NAMES values to None
    when torchvision is unavailable, which breaks AutoVideoProcessor resolution.
    """
    try:
        import importlib

        module = importlib.import_module(
            "transformers.models.auto.video_processing_auto"
        )
    except Exception:
        return False

    mapping = getattr(module, "VIDEO_PROCESSOR_MAPPING_NAMES", None)
    patched = False

    if mapping is None:
        module.VIDEO_PROCESSOR_MAPPING_NAMES = {}
        patched = True
    else:
        for key, value in list(mapping.items()):
            if value is None:
                mapping[key] = ()
                patched = True

    if (
        patched
        and hasattr(module, "_LazyAutoMapping")
        and hasattr(module, "CONFIG_MAPPING_NAMES")
    ):
        module.VIDEO_PROCESSOR_MAPPING = module._LazyAutoMapping(
            module.CONFIG_MAPPING_NAMES,
            module.VIDEO_PROCESSOR_MAPPING_NAMES,
        )

    return patched


def patch_transformers_video_processor_loader() -> bool:
    """
    Allow AutoVideoProcessor to fail gracefully when torchvision is missing.

    For image-only VLM requests we can skip the video processor entirely.
    """
    try:
        import importlib

        auto_module = importlib.import_module(
            "transformers.models.auto.video_processing_auto"
        )
        processing_utils = importlib.import_module("transformers.processing_utils")
        AutoVideoProcessor = auto_module.AutoVideoProcessor
        ProcessorMixin = processing_utils.ProcessorMixin
    except Exception:
        return False

    patched = False

    if not hasattr(AutoVideoProcessor, "_mlx_batch_server_patched"):
        original = AutoVideoProcessor.from_pretrained

        def _from_pretrained(cls, *args, **kwargs):
            try:
                return original.__func__(cls, *args, **kwargs)
            except ImportError as exc:
                if "torchvision" in str(exc).lower():
                    return None
                raise

        AutoVideoProcessor.from_pretrained = classmethod(_from_pretrained)
        AutoVideoProcessor._mlx_batch_server_patched = True
        patched = True

    if not hasattr(ProcessorMixin, "_mlx_batch_server_video_none_ok"):
        original_check = ProcessorMixin.check_argument_for_proper_class

        def _check_argument_for_proper_class(self, argument_name, argument):
            if argument_name == "video_processor" and argument is None:
                return None
            return original_check(self, argument_name, argument)

        ProcessorMixin.check_argument_for_proper_class = (
            _check_argument_for_proper_class
        )
        ProcessorMixin._mlx_batch_server_video_none_ok = True
        patched = True

    return patched


# Auto-patch when this module is imported
_patched_jieba_regex = patch_jieba_regex_warnings()
_patched_jieba = patch_jieba()
# Patch mlx-lm missing logging import (bug in mlx-lm)
_patched_mlx_lm = patch_mlx_lm_logging()
# Patch transformers AutoVideoProcessor None mapping
_patched_transformers_video = patch_transformers_video_processor_mapping()
_patched_transformers_video_loader = patch_transformers_video_processor_loader()

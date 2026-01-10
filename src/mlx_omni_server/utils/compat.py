"""
Compatibility patches for deprecated APIs in dependencies.

Patches pkg_resources usage in jieba to use importlib.resources instead.
Must be imported BEFORE any module that imports jieba (e.g., f5_tts_mlx).

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import sys
import types


def _create_jieba_compat() -> types.ModuleType:
    """
    Create a patched jieba._compat module using importlib.resources.

    Replaces deprecated pkg_resources.resource_stream with modern API.
    This is a full reimplementation of jieba/_compat.py without pkg_resources.
    """
    compat = types.ModuleType("jieba._compat")
    compat.__file__ = "<patched by mlx_omni_server>"

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


# Auto-patch when this module is imported
_patched = patch_jieba()

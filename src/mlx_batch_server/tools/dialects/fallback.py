"""Target-original fallback for models without a tool-output protocol."""

from __future__ import annotations

from ..parser import DialectParse


class PlainTextDialect:
    """Expose all model output as text and intentionally recognize no tools."""

    def parse(self, text: str, *, final: bool) -> DialectParse:
        del final
        return DialectParse(visible_text=text)

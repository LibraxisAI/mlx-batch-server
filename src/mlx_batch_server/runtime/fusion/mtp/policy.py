"""Fail-closed lane selection for exact MTP.

MTPLX currently falls back for ordinary multi-row decode, while oMLX carries
an opt-in row-wise path. The fused runtime therefore guarantees singleton MTP
first and admits a multi-row cohort only after an explicit live-proof toggle
and conservative alignment checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import MtpAlignment, MtpDecision, MtpDisableReason, MtpMode


@dataclass(frozen=True, slots=True)
class MtpPolicy:
    enabled: bool = True
    draft_depth: int = 3
    allow_proven_multirow: bool = False
    max_proven_rows: int = 1

    def decide(
        self,
        *,
        alignment: MtpAlignment,
        model_supported: bool,
        head_attached: bool,
        decode_enabled: bool,
        verifier_available: bool,
        grammar_constrained: bool = False,
    ) -> MtpDecision:
        rows = alignment.row_count
        reason = self._disable_reason(
            alignment=alignment,
            model_supported=model_supported,
            head_attached=head_attached,
            decode_enabled=self.enabled and decode_enabled,
            verifier_available=verifier_available,
            grammar_constrained=grammar_constrained,
        )
        if reason is not None:
            return MtpDecision(
                enabled=False,
                mode=MtpMode.DISABLED,
                exact=False,
                row_count=rows,
                disable_reason=reason,
            )

        mode = MtpMode.EXACT_SINGLETON if rows == 1 else MtpMode.EXACT_ALIGNED_COHORT
        return MtpDecision(
            enabled=True,
            mode=mode,
            exact=True,
            row_count=rows,
            facts={
                "alignment_required": rows > 1,
                "draft_depth": self.draft_depth,
            },
        )

    def _disable_reason(
        self,
        *,
        alignment: MtpAlignment,
        model_supported: bool,
        head_attached: bool,
        decode_enabled: bool,
        verifier_available: bool,
        grammar_constrained: bool,
    ) -> MtpDisableReason | None:
        if alignment.row_count == 0:
            return MtpDisableReason.EMPTY_COHORT
        if not model_supported:
            return MtpDisableReason.MODEL_UNSUPPORTED
        if not head_attached:
            return MtpDisableReason.HEAD_MISSING
        if not decode_enabled:
            return MtpDisableReason.DECODE_DISABLED
        if not verifier_available:
            return MtpDisableReason.VERIFIER_MISSING
        if alignment.pending_prompt_work:
            return MtpDisableReason.PROMPT_MERGE_PENDING
        if grammar_constrained:
            return MtpDisableReason.GRAMMAR_PROCESSOR_UNSUPPORTED
        if alignment.row_count == 1:
            return None
        if not self.allow_proven_multirow or self.max_proven_rows < alignment.row_count:
            return MtpDisableReason.MULTIROW_NOT_PROVEN
        if not alignment.is_aligned:
            return MtpDisableReason.ROWS_UNALIGNED
        return None

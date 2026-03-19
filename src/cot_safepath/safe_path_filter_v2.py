"""
SafePathFilter v2: Redacts or flags dangerous reasoning patterns in CoT traces.

Operates on the output of CoTAnalyzer and applies configurable actions:
  - FLAG   : mark dangerous segments in-place with [FLAGGED: reason]
  - REDACT : replace dangerous segments with [REDACTED]
  - BLOCK  : refuse to pass the trace at all (returns empty + metadata)

Design principle: every transformation is reversible audit-wise — the original
text and the specific redaction positions are always preserved in the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .cot_analyzer import CoTAnalyzer, CoTAnalysis, RedFlag, RedFlagType


class FilterAction(str, Enum):
    FLAG = "flag"       # annotate in-place
    REDACT = "redact"   # replace with placeholder
    BLOCK = "block"     # reject entire trace


@dataclass
class FilterConfig:
    """Configuration for SafePathFilter v2."""
    action: FilterAction = FilterAction.FLAG
    # Risk threshold above which BLOCK is applied (overrides action if exceeded)
    block_threshold: float = 0.75
    # Risk threshold above which REDACT is applied (if action=FLAG and risk high)
    redact_threshold: float = 0.55
    # Whether to append a risk summary at the end of the filtered trace
    append_summary: bool = True
    # Per-type severity threshold below which flags are suppressed
    min_severity: float = 0.50


@dataclass
class FilteredTrace:
    """Result of SafePathFilter processing."""
    original: str
    filtered: str
    action_taken: FilterAction
    analysis: CoTAnalysis
    redactions: List[Tuple[int, int, str]] = field(default_factory=list)
    # (start, end, reason)
    blocked: bool = False

    @property
    def is_clean(self) -> bool:
        return not self.blocked and not self.redactions and self.analysis.is_safe


class SafePathFilterV2:
    """
    Real-time filter for dangerous chain-of-thought reasoning.

    Usage::

        f = SafePathFilterV2()
        result = f.filter(trace)
        if result.blocked:
            print("Trace blocked:", result.analysis.summary)
        else:
            print(result.filtered)
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._analyzer = CoTAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, trace: str) -> FilteredTrace:
        """Analyze and filter a CoT trace according to configured policy."""
        analysis = self._analyzer.analyze(trace)
        action = self._choose_action(analysis)

        if action == FilterAction.BLOCK:
            return FilteredTrace(
                original=trace,
                filtered="",
                action_taken=FilterAction.BLOCK,
                analysis=analysis,
                blocked=True,
            )

        filtered_text, redactions = self._apply_filter(trace, analysis, action)

        if self.config.append_summary:
            filtered_text = self._append_summary(filtered_text, analysis)

        return FilteredTrace(
            original=trace,
            filtered=filtered_text,
            action_taken=action,
            analysis=analysis,
            redactions=redactions,
            blocked=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_action(self, analysis: CoTAnalysis) -> FilterAction:
        """Pick the effective action based on risk score."""
        if analysis.overall_risk >= self.config.block_threshold:
            return FilterAction.BLOCK
        if (
            self.config.action == FilterAction.FLAG
            and analysis.overall_risk >= self.config.redact_threshold
        ):
            return FilterAction.REDACT
        return self.config.action

    def _apply_filter(
        self, trace: str, analysis: CoTAnalysis, action: FilterAction
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Apply FLAG or REDACT transformations to the trace."""
        if not analysis.red_flags:
            return trace, []

        relevant_flags = [
            f for f in analysis.red_flags
            if f.severity >= self.config.min_severity
        ]
        if not relevant_flags:
            return trace, []

        if action == FilterAction.FLAG:
            return self._flag_text(trace, relevant_flags)
        else:  # REDACT
            return self._redact_text(trace, relevant_flags)

    def _flag_text(
        self, trace: str, flags: List[RedFlag]
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Annotate flagged segments inline."""
        result = trace
        offset = 0
        redactions: List[Tuple[int, int, str]] = []

        for flag in flags:
            # Find the evidence text in the (possibly shifted) result
            for ev in flag.evidence:
                ev_clean = ev.strip()
                if not ev_clean:
                    continue
                pattern = re.compile(re.escape(ev_clean), re.IGNORECASE)
                m = pattern.search(result, offset)
                if m:
                    tag = f"[FLAGGED:{flag.flag_type.value}]"
                    replacement = f"{result[m.start():m.end()]}{tag}"
                    redactions.append((m.start(), m.end(), flag.flag_type.value))
                    result = result[: m.start()] + replacement + result[m.end():]
                    offset = m.start() + len(replacement)
                    break  # one annotation per flag instance

        return result, redactions

    def _redact_text(
        self, trace: str, flags: List[RedFlag]
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Replace flagged segments with redaction placeholders."""
        result = trace
        offset = 0
        redactions: List[Tuple[int, int, str]] = []

        for flag in flags:
            for ev in flag.evidence:
                ev_clean = ev.strip()
                if not ev_clean:
                    continue
                pattern = re.compile(re.escape(ev_clean), re.IGNORECASE)
                m = pattern.search(result, offset)
                if m:
                    placeholder = f"[REDACTED:{flag.flag_type.value}]"
                    redactions.append((m.start(), m.end(), flag.flag_type.value))
                    result = result[: m.start()] + placeholder + result[m.end():]
                    offset = m.start() + len(placeholder)
                    break

        return result, redactions

    @staticmethod
    def _append_summary(text: str, analysis: CoTAnalysis) -> str:
        if analysis.is_safe:
            return text
        return text + f"\n\n<!-- SafePath: {analysis.summary} -->"

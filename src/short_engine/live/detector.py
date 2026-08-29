"""Incremental highlight state machine for delayed live clipping."""

from short_engine.core.models import TimeRange
from short_engine.live.models import LiveHighlight, LiveHighlightState, LiveSignal


class LiveHighlightDetector:
    def __init__(self, hook_threshold: float = 0.7, payoff_threshold: float = 0.78) -> None:
        self.hook_threshold = hook_threshold
        self.payoff_threshold = payoff_threshold
        self._start: float | None = None
        self._state = LiveHighlightState.WATCHING

    def observe(self, signal: LiveSignal) -> LiveHighlight:
        hook_score = (signal.speech_energy + signal.visual_novelty + signal.semantic_stakes) / 3
        payoff_score = (signal.visual_novelty + signal.semantic_stakes) / 2
        if self._state is LiveHighlightState.WATCHING and hook_score >= self.hook_threshold:
            self._start = max(0, signal.at_seconds - 3)
            self._state = LiveHighlightState.HOOK_DETECTED
        elif self._state is LiveHighlightState.HOOK_DETECTED:
            self._state = LiveHighlightState.ESCALATING
        elif self._state is LiveHighlightState.ESCALATING and payoff_score >= self.payoff_threshold:
            self._state = LiveHighlightState.READY
        source = (
            TimeRange(start_seconds=self._start, end_seconds=signal.at_seconds + 2)
            if self._state is LiveHighlightState.READY and self._start is not None
            else None
        )
        return LiveHighlight(state=self._state, source=source, confidence=hook_score)

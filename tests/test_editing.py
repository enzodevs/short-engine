from short_engine.core.models import TimeRange
from short_engine.editing.jumpcuts import JumpCutConfig, JumpCutPlanner
from short_engine.transcription.models import TimedWord


def word(start: float, end: float, text: str) -> TimedWord:
    return TimedWord(start_seconds=start, end_seconds=end, text=text)


def test_jumpcut_planner_removes_only_meaningful_silence_with_handles() -> None:
    words = [word(0.2, 1, "one"), word(1.3, 2, "two"), word(4, 5, "three")]
    plan = JumpCutPlanner(JumpCutConfig(min_silence_seconds=0.8)).plan(
        TimeRange(start_seconds=0, end_seconds=6), words
    )

    assert len(plan.segments) == 2
    assert plan.segments[0].source.end_seconds == 2.08
    assert plan.segments[1].source.start_seconds == 3.94
    assert plan.removed_seconds == 1.86


def test_jumpcut_plan_remaps_word_timestamps_to_output_timeline() -> None:
    words = [word(0.2, 1, "one"), word(4, 5, "three")]
    plan = JumpCutPlanner(JumpCutConfig(min_silence_seconds=0.8)).plan(
        TimeRange(start_seconds=0, end_seconds=6), words
    )

    remapped = plan.remap_words(words)

    assert remapped[0].start_seconds == 0.2
    assert remapped[1].start_seconds < 2.3
    assert remapped[1].text == "three"


def test_short_pauses_do_not_create_choppy_cuts() -> None:
    words = [word(0.2, 1, "one"), word(1.4, 2, "two")]
    plan = JumpCutPlanner().plan(TimeRange(start_seconds=0, end_seconds=3), words)

    assert len(plan.segments) == 1
    assert plan.removed_seconds == 0


def test_jumpcut_planner_composes_multiple_source_ranges() -> None:
    words = [word(0.2, 1, "hook"), word(10.2, 11, "payoff")]

    plan = JumpCutPlanner().plan_many(
        [
            TimeRange(start_seconds=10, end_seconds=12),
            TimeRange(start_seconds=0, end_seconds=2),
        ],
        words,
    )

    assert [segment.source.start_seconds for segment in plan.segments] == [10, 0]
    assert plan.segments[1].output.start_seconds == 2


def test_repeated_payoff_teaser_keeps_captions_in_both_playback_positions() -> None:
    words = [word(0.2, 1, "setup"), word(10.2, 11, "payoff")]
    plan = JumpCutPlanner().plan_many(
        [
            TimeRange(start_seconds=10, end_seconds=12),
            TimeRange(start_seconds=0, end_seconds=12),
        ],
        words,
    )

    remapped = plan.remap_words(words)

    assert [item.text for item in remapped] == ["payoff", "setup", "payoff"]
    assert remapped[2].start_seconds > remapped[1].start_seconds

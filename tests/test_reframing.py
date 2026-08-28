from short_engine.core.models import AspectRatio, TimeRange
from short_engine.reframing.models import SubjectObservation, SubjectTrack
from short_engine.reframing.planner import CropPlanner


def test_crop_plan_is_bounded_and_smoothed() -> None:
    track = SubjectTrack(
        observations=[
            SubjectObservation(time_seconds=i, center_x=x, center_y=500, confidence=0.9)
            for i, x in enumerate([100, 900, 120, 880])
        ]
    )
    plan = CropPlanner(smoothing=0.8).plan(
        TimeRange(start_seconds=0, end_seconds=4), track, 1920, 1080, AspectRatio.VERTICAL
    )
    assert all(0 <= sample.x <= 1920 - plan.crop_width for sample in plan.samples)
    assert max(abs(b.x - a.x) for a, b in zip(plan.samples, plan.samples[1:], strict=False)) < 400
    assert not plan.used_fallback


def test_crop_plan_has_observable_center_fallback() -> None:
    plan = CropPlanner().plan(
        TimeRange(start_seconds=0, end_seconds=2),
        SubjectTrack(observations=[]),
        1920,
        1080,
        AspectRatio.VERTICAL,
    )
    assert plan.used_fallback
    assert plan.samples[0].x == (1920 - plan.crop_width) / 2

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


def test_comfort_camera_locks_small_stationary_jitter() -> None:
    track = SubjectTrack(
        observations=[
            SubjectObservation(time_seconds=i, center_x=x, center_y=500, confidence=0.9)
            for i, x in enumerate([900, 925, 890, 915, 905])
        ]
    )
    plan = CropPlanner().plan(
        TimeRange(start_seconds=0, end_seconds=5), track, 1920, 1080, AspectRatio.VERTICAL
    )

    assert len({sample.x for sample in plan.samples}) == 1


def test_comfort_camera_records_hard_cut_and_resets_between_scenes() -> None:
    track = SubjectTrack(
        observations=[
            SubjectObservation(
                time_seconds=i,
                center_x=x,
                center_y=500,
                confidence=0.9,
                scene_id=scene,
            )
            for i, (x, scene) in enumerate(
                [(300, 0), (800, 0), (1400, 0), (1700, 1), (1650, 1), (1720, 1)]
            )
        ]
    )
    plan = CropPlanner().plan(
        TimeRange(start_seconds=0, end_seconds=6),
        track,
        1920,
        1080,
        AspectRatio.VERTICAL,
        hard_cuts_seconds=[3],
    )

    assert plan.hard_cuts_seconds == [3]
    assert abs(plan.samples[3].x - plan.samples[2].x) > 100


def test_jump_cut_starts_new_camera_take_without_anticipation() -> None:
    track = SubjectTrack(
        observations=[
            SubjectObservation(time_seconds=time, center_x=x, center_y=500, confidence=0.9)
            for time, x in [
                (0, 350),
                (1, 350),
                (2, 350),
                (3, 1550),
                (4, 1550),
                (5, 1550),
            ]
        ]
    )
    plan = CropPlanner().plan(
        TimeRange(start_seconds=0, end_seconds=5),
        track,
        1920,
        1080,
        AspectRatio.VERTICAL,
        takes=[
            TimeRange(start_seconds=0, end_seconds=2),
            TimeRange(start_seconds=3, end_seconds=5),
        ],
    )

    first_take = [sample.x for sample in plan.samples if sample.time_seconds <= 2]
    second_take = [sample.x for sample in plan.samples if sample.time_seconds >= 3]
    assert len(set(first_take)) == 1
    assert len(set(second_take)) == 1
    assert second_take[0] - first_take[-1] > 800

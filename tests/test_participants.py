from app.pipeline.participants import ParticipantMap


def test_reordering_and_disappearance_never_relabels_the_remaining_person():
    participants = ParticipantMap()
    assert participants.select([10, 20], [[0, 0, 20, 40], [80, 0, 100, 40]]) == (0, 1)
    assert participants.select([20, 10], [[75, 0, 95, 40], [5, 0, 25, 40]]) == (1, 0)
    assert participants.select([20], [[75, 0, 95, 40]]) == (None, 0)
    assert participants.select([30, 20], [[5, 0, 25, 40], [75, 0, 95, 40]]) == (None, 1)


def test_waits_for_exactly_two_unique_tracked_people_and_initializes_left_to_right():
    participants = ParticipantMap()
    assert participants.select([1, 2, 3], [[0, 0, 10, 20]] * 3) == (None, None)
    assert participants.select([1, 1], [[0, 0, 10, 20]] * 2) == (None, None)
    assert participants.select([], []) == (None, None)
    assert participants.select([20, 10], [[80, 0, 100, 40], [0, 0, 20, 40]]) == (1, 0)


def test_new_session_does_not_inherit_the_previous_mapping():
    first = ParticipantMap()
    first.select([10, 20], [[0, 0, 20, 40], [80, 0, 100, 40]])
    assert ParticipantMap().select([20, 10], [[0, 0, 20, 40], [80, 0, 100, 40]]) == (0, 1)

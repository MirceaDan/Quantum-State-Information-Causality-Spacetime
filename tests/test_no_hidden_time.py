from relational_dynamics.history import RelationalEvent, assert_no_hidden_external_time


def test_event_labels_do_not_create_physical_time():
    event = RelationalEvent(computational_index=3, clock_reading=1.5, external_time=None)
    assert_no_hidden_external_time(event)

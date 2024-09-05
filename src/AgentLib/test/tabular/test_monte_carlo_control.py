# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)


def test_decay_schedule():
    """
    Test decay_schedule method
    """

    from ...tabular.monte_carlo_control.decay_schedule import decay_schedule

    values = decay_schedule(
        init_value=5, min_value=0.25, decay_ratio=0.25, max_steps=100
    )

    # Check init_value
    assert values[0] == 5

    # Check min_value
    assert values[-1] == 0.25

    # Check decay_ratio
    assert values[23] > 0.25
    assert values[24] == 0.25

    # Check max_steps
    assert values.size == 100


def test_generate_trajectory():
    """
    Test generate_trajectory method
    """

    from ...tabular.monte_carlo_control.generate_trajectory import generate_trajectory

    pass

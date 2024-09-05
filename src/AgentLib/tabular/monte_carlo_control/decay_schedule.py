# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import numpy as np


def decay_schedule(
    init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10
):
    """
    Compute decaying values
    """

    # Index where the decaying of values terminates, min_value continues until max_steps
    decay_steps = int(max_steps * decay_ratio)

    # The difference
    rem_steps = max_steps - decay_steps

    # Compute `decay_steps`` number of values spaced evenly on a log scale, then reverse them so the resulting curve is descending
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]

    # Normalize values to ensure they are exactly between 1 (starting value) and 0 (ending value)
    values = (values - values.min()) / (values.max() - values.min())

    # Apply a linear transformation to get points between init_value (starting value) and min_value (ending value)
    values = (init_value - min_value) * values + min_value

    # Repeat the right most value (min_value) rem_step number of times
    values = np.pad(values, (0, rem_steps), "edge")

    return values

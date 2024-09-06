# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)


def test_double_q_learning():
    """
    Test double_q_learning method
    """

    import gymnasium as gym

    from ...tabular.double_q_learning.double_q_learning import double_q_learning

    env = gym.make("FrozenLake-v1")

    Q, V, pi, Q_track, pi_track = double_q_learning(env)

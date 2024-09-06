# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)


def test_q_learning():
    """
    Test q_learning method
    """

    import gymnasium as gym

    from ...tabular.q_learning.q_learning import q_learning

    env = gym.make("FrozenLake-v1")

    Q, V, pi, Q_track, pi_track = q_learning(env)

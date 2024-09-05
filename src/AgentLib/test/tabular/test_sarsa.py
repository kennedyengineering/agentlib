# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)


def test_sarsa():
    """
    Test sarsa method
    """

    import gymnasium as gym

    from ...tabular.sarsa.sarsa import sarsa

    env = gym.make("FrozenLake-v1")

    Q, V, pi, Q_track, pi_track = sarsa(env)

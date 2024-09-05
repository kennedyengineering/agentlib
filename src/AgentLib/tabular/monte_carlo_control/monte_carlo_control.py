# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import numpy as np

from tqdm import tqdm

from .decay_schedule import decay_schedule
from .generate_trajectory import generate_trajectory


def monte_carlo_control(
    env,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay_ratio=0.9,
    n_episodes=3000,
    max_steps=200,
    first_visit=True,
):
    """
    Complete implementation of the Monte Carlo control method
    """

    # Find the number of discrete states and actions
    nS, nA = env.observation_space.n, env.action_space.n

    # Calculate values for the discount factors in advance
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    # Calculate values for the alphas in advance
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    # Calculate values for the epsilons in advance
    epsilons = decay_schedule(
        init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
    )

    # Set up variables
    Q = np.zeros((nS, nA), dtype=np.float64)

    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    pi_track = []

    # Define epsilon-greedy strategy, decaying epsilon each episode
    select_action = lambda state, Q, epsilon: (
        np.argmax(Q[state])
        if np.random.random() > epsilon
        else np.random.randint(len(Q(state)))
    )

    # Run the episode loop for `n_episodes` episodes
    for e in tqdm(range(n_episodes), leave=False):

        # Generate a new trajectory with the select_action policy, limit length to `max_steps` steps
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)

        # Keep track of visits to state-action pairs
        visited = np.zeros((nS, nA), dtype=bool)

        # Process trajectories offline
        for t, (state, action, reward, _, _) in enumerate(trajectory):

            # Check for state-action pair visits
            if visited[state][action] and first_visit:
                continue

            visited[state][action] = True

            # Calculate the return
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        # Save values for post analysis
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Extract state-value function
    V = np.max(Q, axis=1)

    # Extract greedy policy
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track

# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import numpy as np

from tqdm import tqdm

from ..monte_carlo_control.decay_schedule import decay_schedule


def q_learning(
    env,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay_ratio=0.9,
    n_episodes=3000,
):
    """
    Complete implementation of Q-Learning
    """

    # Find the number of discrete states and actions
    nS, nA = env.observation_space.n, env.action_space.n

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
        else np.random.randint(len(Q[state]))
    )

    # Run the episode loop for `n_episodes` episodes
    for e in tqdm(range(n_episodes), leave=False):

        # Start each episode by resetting the environment and the done flag
        state, _ = env.reset()
        done = False

        # Repeat until terminal state is hit
        while not done:

            # Select action for the current state
            action = select_action(state, Q, epsilons[e])

            # Step the environment ang get the experience
            next_state, reward, done, _, _ = env.step(action)

            # Calculate the `td_target`, multiply expression by (not done) to zero out the future on terminal states
            td_target = reward + gamma * Q[next_state].max() * (not done)

            # Calculate the `td_error`
            td_error = td_target - Q[state][action]

            # Update the Q-function
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            # Update state for next step
            state = next_state

        # Save values for post analysis
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Extract state-value function
    V = np.max(Q, axis=1)

    # Extract greedy policy
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track

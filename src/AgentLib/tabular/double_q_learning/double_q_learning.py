# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import numpy as np

from tqdm import tqdm

from ..monte_carlo_control.decay_schedule import decay_schedule


def double_q_learning(
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
    Complete implementation of Double Q-Learning
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
    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)

    Q1_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    Q2_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
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

            # Select action for the current state using the mean of the two Q-functions
            action = select_action(state, (Q1 + Q2) / 2.0, epsilons[e])

            # Step the environment and get the experience
            next_state, reward, done, _, _ = env.step(action)

            # Flip a coin to determine if Q1 or Q2 is updated
            if np.random.randint(2):

                # Use the action Q1 thinks is best
                argmax_Q1 = np.argmax(Q1[next_state])

                # Get the value from Q2 to calculate the TD target
                td_target = reward + gamma * Q2[next_state][argmax_Q1] * (not done)

                # Calculate the TD error from the Q1 estimate
                td_error = td_target - Q1[state][action]

                # Update Q1 to the target using the error
                Q1[state][action] = Q1[state][action] + alphas[e] * td_error
            else:

                # Use the action Q2 thinks is best
                argmax_Q2 = np.argmax(Q2[next_state])

                # Get the value from Q1 to calculate the TD target
                td_target = reward + gamma * Q1[next_state][argmax_Q2] * (not done)

                # Calculate the TD error from the Q2 estimate
                td_error = td_target - Q2[state][action]

                # Update Q2 to the target using the error
                Q2[state][action] = Q2[state][action] + alphas[e] * td_error

            # Update state for next step
            state = next_state

        # Save values for post analysis
        Q1_track[e] = Q1
        Q2_track[e] = Q2
        pi_track.append(np.argmax((Q1 + Q2) / 2.0, axis=1))

    # Extract the final Q-function
    Q = (Q1 + Q2) / 2.0

    # Extract state-value function
    V = np.max(Q, axis=1)

    # Extract greedy policy
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, (Q1_track + Q2_track) / 2.0, pi_track

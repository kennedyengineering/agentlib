# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import numpy as np

from itertools import count


def generate_trajectory(select_action, Q, epsilon, env, max_steps=200):
    """
    Roll out the policy in the environment for a full episode
    """

    # Initialize the done flag and a list of experiences named `trajectory`
    done, trajectory = False, []

    # Loop through until the done flag is set to True
    while not done:

        # Reset the environment to interact with a new episode
        state, _ = env.reset()

        # Start counting steps `t`
        for t in count():

            # Use the `select_action` function to pick an action
            action = select_action(state, Q, epsilon)

            # Step the environment using that action and obtain the full experience tuple
            next_state, reward, done, _, _ = env.step(action)

            experience = (state, action, reward, next_state, done)

            # Append the experience to the trajectory list
            trajectory.append(experience)

            # Break if a terminal state is encountered and the `done` flag is raised
            if done:
                break

            # If the count of steps `t` in the current trajectory hits the maximum, clear the trajectory, break, and try to obtain another trajectory
            if t >= max_steps - 1:
                trajectory = []
                break

            # Update the state
            state = next_state

    # Return a NumPy version of the trajectory for easy data manipulation
    return np.array(trajectory, object)

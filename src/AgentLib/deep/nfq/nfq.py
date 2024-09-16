# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import time


class NFQ:
    def __init__(
        self,
        value_model_fn,
        value_optimizer_fn,
        value_optimizer_lr,
        training_strategy_fn,
        evaluation_strategy_fn,
        batch_size,
        epochs,
    ):
        """
        Initialization method
        """

        # Store variables
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.batch_size = batch_size
        self.epochs = epochs

    def optimize_model(self, experiences):

        # Extract values from experience tuple
        states, actions, rewards, next_states, is_terminals = experiences

        # Determine batch size
        batch_size = len(is_terminals)

        # Get the values of the Q-function at `next_state`
        # `detach()` to avoid propagating values
        q_sp = self.online_model(next_states).detach()

        # Get the max value of the next state
        max_a_q_sp = q_sp.max(1)[0].unsqueeze(1)

        # Ensure terminal states are grounded to zero
        max_a_q_sp *= 1 - is_terminals

        # Calculate the target
        target_q_s = rewards + self.gamma * max_a_q_sp

        # Get the current estimate of Q(s, a)
        q_sa = self.online_model(states).gather(1, actions)

        # Compute error
        td_errors = q_sa - target_q_s

        # Compute value loss (MSE with a 0.5 scaling factor)
        value_loss = td_errors.pow(2).mul(0.5).mean()

        # Step the optimizer
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):

        # Select action
        action = self.training_strategy.select_action(self.online_model, state)

        # Step the environment and collect an experience tuple
        new_state, reward, is_terminal, is_truncated, info = env.step(action)

        # Define failure condition
        is_failure = is_terminal and not is_truncated

        # Define new experience tuple
        experience = (state, action, reward, new_state, float(is_failure))

        pass

    def train(
        self,
        make_env_fn,
        make_env_kargs,
        seed,
        gamma,
        max_minutes,
        max_episodes,
        goal_mean_100_reward,
    ):

        # Define training variables
        training_start = time.time()
        last_debug_time = float("-inf")

        self.checkpoint_dir = ""  # TODO: figure out filesystem
        self.make_env_fc = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        # FIXME: does seed also need to be set here?
        # torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        # TODO: cut out logging/debugging code? or keep it?

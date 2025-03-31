import abc

class BaseAlgorithm(abc.ABC):
    def __init__(self, env, seed=0, logger_kwargs=None):
        """
        Base class for all reinforcement learning algorithms.

        Args:
            env: The environment to interact with.
            seed: Random seed for reproducibility.
            logger_kwargs: Additional arguments for logging.
        """
        self.env = env
        self.seed = seed
        self.logger_kwargs = logger_kwargs or {}

    @abc.abstractmethod
    def train(self):
        """Train the algorithm."""
        pass

    @abc.abstractmethod
    def update(self):
        """Update the algorithm's parameters."""
        pass



class OnPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, env, seed=0, logger_kwargs=None):
        """
        Base class for on-policy reinforcement learning algorithms.

        Args:
            env: The environment to interact with.
            seed: Random seed for reproducibility.
            logger_kwargs: Additional arguments for logging.
        """
        super().__init__(env, seed, logger_kwargs)

    @abc.abstractmethod
    def compute_pi_loss(self, data, actor_critic_policy):
        """Compute the policy loss."""
        pass

    @abc.abstractmethod
    def compute_v_loss(self, data, actor_critic_policy):
        """Compute the value function loss."""
        pass
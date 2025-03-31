import jax.numpy as jnp

class RollOutBuffer:
    def __init__(self, obs_dim, act_dim, size):
        """
        Base class for a rollout buffer.

        Args:
            obs_dim: Dimension of the observation space.
            act_dim: Dimension of the action space.
            size: Maximum number of timesteps to store in the buffer.
        """
        self.obs_buf = jnp.zeros((size, obs_dim), dtype=jnp.float32)
        self.act_buf = jnp.zeros((size, act_dim), dtype=jnp.float32)
        self.ptr = 0  # Pointer to the current position in the buffer
        self.max_size = size  # Maximum size of the buffer

    def store(self, obs, act):
        """
        Store a single timestep of interaction.

        Args:
            obs: Observation from the environment.
            act: Action taken by the agent.
        """
        assert self.ptr < self.max_size, "Buffer is full. Cannot store more data."
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr += 1

    def reset(self):
        """Reset the buffer pointer to start storing from the beginning."""
        self.ptr = 0

    def get(self):
        """
        Retrieve the stored observations and actions.

        Returns:
            A dictionary containing observations and actions.
        """
        return {
            'obs': self.obs_buf[:self.ptr],
            'act': self.act_buf[:self.ptr]
        }
        pass
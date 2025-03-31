import jax
import jax.numpy as jnp
from gymnasium.spaces import Box, Discrete
import jax.nn as nn
import numpy as np
import distrax
# from distrax.distributions.normal import Normal
# from distrax.distributions.categorical import Categorical
from flax import linen as nn
from jax import random
from jax.scipy.stats import norm, multinomial
    


class MLPCategoricalActor(nn.Module):
    obs_dim: int
    act_dim: int
    hidden_sizes: list
    activation: nn.Module

    def setup(self):
        self.logits_net = self._mlp([self.obs_dim] + self.hidden_sizes + [self.act_dim], self.activation)

    def _mlp(self, sizes, activation):
        layers = []
        for j in range(len(sizes) - 1):
            layers.append(nn.Dense(sizes[j + 1]))
            if j < len(sizes) - 2:
                layers.append(activation())
        return nn.Sequential(layers)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return multinomial.Multinomial(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(nn.Module):
    obs_dim: int
    act_dim: int
    hidden_sizes: list
    activation: nn.Module

    def setup(self):
        log_std = -0.5 * jnp.ones(self.act_dim, dtype=jnp.float32)
        self.log_std = jax.device_put(log_std)  # Store log_std on the device
        self.mu_net = self._mlp([self.obs_dim] + self.hidden_sizes + [self.act_dim], self.activation)

    def _mlp(self, sizes, activation):
        layers = []
        for j in range(len(sizes) - 1):
            layers.append(nn.Dense(sizes[j + 1]))
            if j < len(sizes) - 2:
                layers.append(activation())
        return nn.Sequential(layers)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = jnp.exp(self.log_std)
        return norm(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Sum over the last axis

class MLPCritic(nn.Module):
    obs_dim: int
    hidden_sizes: list
    activation: nn.Module

    def setup(self):
        # Create the value network (v_net) using the MLP structure
        self.v_net = self._mlp([self.obs_dim] + self.hidden_sizes + [1], self.activation)

    def _mlp(self, sizes, activation):
        layers = []
        for j in range(len(sizes) - 1):
            layers.append(nn.Dense(sizes[j + 1]))  # Create a Dense layer
            if j < len(sizes) - 2:
                layers.append(activation())  # Add activation function
        return nn.Sequential(layers)

    def __call__(self, obs):
        # Forward pass through the value network
        return jnp.squeeze(self.v_net(obs), axis=-1)  # Ensure the output has the right shape
    

class MLPActorCritic(nn.Module):
    observation_space: any
    action_space: any
    hidden_sizes: tuple = (64, 64)
    activation: nn.Module = nn.tanh

    def setup(self):
        obs_dim = self.observation_space.shape[0]

        # Policy builder depends on action space
        if isinstance(self.action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, self.action_space.shape[0], self.hidden_sizes, self.activation)
        elif isinstance(self.action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, self.action_space.n, self.hidden_sizes, self.activation)

        # Build value function
        self.v = MLPCritic(obs_dim, self.hidden_sizes, self.activation)

    def step(self, obs):
        pi = self.pi._distribution(obs)
        a = pi.sample()  # Sample an action
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
    

# #EXAMPLE CODE ##########################
# # Example usage
# actor_critic_model = MLPActorCritic(observation_space=env.observation_space, action_space=env.action_space)
# obs = jnp.ones((1, env.observation_space.shape[0]))  # Example input
# action, value, log_prob = actor_critic_model.step(obs)  # Forward pass
# print(action, value, log_prob)

# # Example usage
# critic_model = MLPCritic(obs_dim=4, hidden_sizes=[64, 64], activation=nn.relu)
# obs = jnp.ones((1, 4))  # Example input
# value = critic_model(obs)  # Forward pass
# print(value)
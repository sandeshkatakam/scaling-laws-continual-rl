import jax
import jax.numpy as jnp
import optax
import gymnasium
import time
import scipy
from networks import *
import sys
import os
# ppo.py
import sys
import os

# Add the parent directory to the Python path
sys.path.append("/home/ai24mtech02002/scaling-laws-continual-rl")
# Add the path to the rl_scaling directory
from utils.logs import EpochLogger
from utils.run_utils import setup_logger_kwargs
from utils.tools import scalar_statistics
from rl_algorithms.base_algorithm import BaseAlgorithm, OnPolicyAlgorithm
from rl_algorithms.buffers.rollout_buffer import RollOutBuffer
from rl_algorithms.ppo import networks as networks


class PPOBuffer(RollOutBuffer):
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95):
        super().__init__(obs_dim, act_dim, size)  # Initialize the base class
        self.adv_buf = jnp.zeros(size, dtype=jnp.float32)
        self.rew_buf = jnp.zeros(size, dtype=jnp.float32)
        self.ret_buf = jnp.zeros(size, dtype=jnp.float32)
        self.val_buf = jnp.zeros(size, dtype=jnp.float32)
        self.logp_buf = jnp.zeros(size, dtype=jnp.float32)
        self.gamma, self.lam = gamma, lam
        self.path_start_idx = 0

    
    def store(self, obs, act, rew, val, logp):
        "Append on timestep of agent-environment interaction to the buffer"

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val = 0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = jnp.append(self.rew_buf[path_slice], last_val)
        vals = jnp.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1, float(- self.gamma * self.lam)], rews[::-1], axis=0)[::-1]
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = scipy.signal.lfilter([1], [1, float(- self.gamma * self.lam)], rews[::-1], axis=0)[::-1][:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all the data from the buffer,
        with advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer
        """

        assert self.ptr == self.max_size
        self.ptr , self.path_start_idx = 0, 0
        # Advantage normalization trick
        adv_mean, adv_std = scalar_statistics(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: jnp.array(v, dtype=jnp.float32) for k, v in data.items()}



class PPO(OnPolicyAlgorithm):
    def __init__(self,
                env,
                policy = networks.MLPActorCritic,
                seed = 0,
                steps_per_epoch = 4000,
                epochs = 50,
                gamma = 0.99,
                clip_ratio = 0.2,
                pi_lr = 3e-4,
                vf_lr = 1e-3,
                train_pi_iters = 80,
                train_vf_iters = 80,
                lam = 0.97,
                max_ep_len = 1000,
                target_kl = 0.01,
                save_freq = 10,
                logger_kwargs=dict(),
        ):
            
            self.env = env
            self.policy = policy
            self.seed = seed
            self.steps_per_epoch  = steps_per_epoch
            self.epochs = epochs
            self.gamma = gamma
            self.lam = lam
            self.clip_ratio = 0.2,
            self.pi_lr = pi_lr
            self.vf_lr = vf_lr
            self.train_pi_iters = train_pi_iters
            self.train_vf_iters = train_vf_iters
            self.max_ep_len = max_ep_len
            self.target_kl = target_kl
            self.save_freq = save_freq
            self.logger_kwargs = logger_kwargs

    def train(self):
        self.seed += 1000 
        key = jax.random.PRNGKey(self.seed)
        self.logger = EpochLogger(**logger_kwargs)
        env = self.env()
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        actor_critic_policy = self.policy(env.observation_space, env.action_space)
        # actor_critic_policy_params = actor_critic_policy.init(key, jnp.ones((1, env.observation_space.shape[0])), jnp.ones((1, env.action_space.n)))
        actor_critic_policy_params = actor_critic_policy.init(key, jnp.ones(obs_dim), jnp.ones(act_dim))  # Initialize parameters
        var_counts = (jnp.prod(p.shape) for p in actor_critic_policy_params['params']['pi']) + (jnp.prod(p.shape) for p in actor_critic_policy_params['params']['v'])

        
        self.logger.log('\n Number of Parameters: \t pi: %d, \t v: %d \n' %var_counts)
        local_steps_per_epoch = int(self.steps_per_epoch)
        buffer = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, self.gamma, self.lam)
        pi_optimizer = optax.adam(learning_rate = self.pi_lr)
        vf_optimizer = optax.adam(learning_rate = self.vf_lr)
        actor_critic_policy_pi_params = actor_critic_policy_params['params']['pi']
        actor_critic_policy_v_params = actor_critic_policy_params['params']['v']
        self.pi_opt_state = self.pi_optimizer.init(actor_critic_policy_params['params']['pi'])  # Initialize policy optimizer state
        self.vf_opt_state = self.vf_optimizer.init(actor_critic_policy_params['params']['v'])  # Initialize value function optimizer state
        # Initialize it before the training

        start_time = time.time()
        (obs, info), ep_ret, ep_len = env.reset(), 0, 0

        # Training Loop
        for epoch in range(self.epochs):
            for t in range(local_steps_per_epoch):
                actions, values, logp = actor_critic_policy.step(jax.Array(obs, dtype = jnp.float32))
                
                next_obs, ret, terminated, truncated , _ = env.step(actions)
                d = terminated or truncated
                ep_ret += ret
                ep_len += 1

                buffer.store(obs, actions, ret, values, logp)
                self.logger.store(V_Values = values)
                obs = next_obs
                timeout = ep_len = self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print("warning: Trajectory cut off by epoch at %d steps" %ep_len, flush = True)

                    if timeout or epoch_ended:
                        _, values, _ = actor_critic_policy.step(jax.Array(obs, dtype = jnp.float32))
                    else:
                        values = 0

                    buffer.finish_path(values)

                    if terminal:
                        self.logger.store(Ep_Returns = ep_ret, Ep_Length= ep_len)

                    obs, _ = env.reset()

                    ep_ret, ep_len = 0, 0


            if (epoch % self.save_freq == 0) or (epoch == self.epochs -1):
                self.logger.save_state({'env': env}, None)
            
            self.update(env, buffer, pi_optimizer, vf_optimizer, actor_critic_policy, actor_critic_policy_params, actor_critic_policy_pi_params, actor_critic_policy_v_params)

            # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('Ep_Returns', with_min_and_max=True)
        self.logger.log_tabular('Ep_Length', average_only=True)
        self.logger.log_tabular('V_Values', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('ClipFrac', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()
    
    def compute_pi_loss(self, data, actor_critic_policy, actor_critic_policy_params):
        obs,act,adv,logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy Loss
        pi, logp = actor_critic_policy.pi.apply(actor_critic_policy_params, obs)  # Access the policy output
        ratio = jnp.exp(logp - logp_old)
        clip_adv = jax.lax.clamp(ratio, 1- self.clip_ratio, 1+ self.clip_ratio) * adv
        loss_pi = - (jnp.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent =  pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio)| ratio.lt(1-self.clip_ratio)
        clipfrac  = jax.Array(clipped, dtype = jnp.float32).mean().item()
        pi_info = dict(kl = approx_kl, ent= ent, cf = clipfrac)

        return loss_pi, pi_info
    
    def compute_v_loss(self, data, actor_critic_policy, actor_critic_policy_params):
        obs, ret = data['obs'] , data['ret']
        value = actor_critic_policy.v.apply(actor_critic_policy_params, obs)  # Access the value function
        return ((value - ret)**2).mean()
        
    

    def update(self, env, buffer, pi_optimizer, vf_optimizer, actor_critic_policy, actor_critic_policy_params, actor_critic_policy_pi_params, actor_critic_policy_v_params):
        data = buffer.get()

        # Policy Updates
        pi_l_old , pi_info_old = self.compute_pi_loss(data, actor_critic_policy, actor_critic_policy_params)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_v_loss(data, actor_critic_policy, actor_critic_policy_params).item()
        # Compute gradients
        
        
        for i in range(self.train_pi_iters):
            # Compute Loss for Policy
            loss_pi, pi_info = self.compute_pi_loss(data, actor_critic_policy)
            kl = pi_info['kl']
            # Compute Gradients

            pi_grad = jax.grad(loss_pi)(actor_critic_policy_pi_params)

            
            
            # Early Stopping After reaching MAX KL 
            if kl > 1.5 * self.target_kl:
                self.logger.log("Early Stopping at step %d due to reaching max kl" %i)

                break

            # Update Policy Parameters
            pi_updates, self.pi_opt_state = self.pi_optimizer.update(pi_grad, self.pi_opt_state, actor_critic_policy_pi_params)
            actor_critic_policy_pi_params = optax.apply_updates(actor_critic_policy_pi_params, pi_updates)
        
        self.logger.store(StopIter = i)
        
        # Value Function Learning
        for i in range(self.train_v_iters):
            loss_v = self.compute_loss_v(data, actor_critic_policy)
            vf_grad = jax.grad(loss_v)(actor_critic_policy_v_params)

            # Update value function parameters
            vf_updates, self.vf_opt_state = self.vf_optimizer.update(vf_grad, self.vf_opt_state, actor_critic_policy_v_params)
            actor_critic_policy.v.parameters = optax.apply_updates(actor_critic_policy_v_params, vf_updates)
            
        
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
                            DeltaLossPi = (loss_pi.item() - pi_l_old),
                            DeltaLossV = (loss_v.item() - v_l_old))


if __name__ == '__main__':
    import argparse
    import rl_algorithms.ppo.networks as networks
    import ale_py

    gymnasium.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v5')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo = PPO(lambda : gymnasium.make(args.env), policy=networks.MLPActorCritic, gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
    
    ppo.train()



import torch
import torch.nn as nn
import torch.jit as jit
import torch.optim as optim
import numpy as np
import logging
import json
import pickle
import gzip
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
from tianshou.data import Batch, to_numpy, to_torch, to_torch_as
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy.base import _gae_return

from src.environments import get_environment
from src.environments.concat_action_wrapper import ConcatAFAWrapper
from src.policies.concat_action_ppo import *

class Agent(object):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps

    def _setup(self, env):
        # environment specific hyperparameters
        obs_high = env.observation_space.high
        num_embeddings = list(map(int, obs_high + 2))
        num_actions = env.num_measurable_features + env.action_space.n

        self.policy = build_policy(self.hps.policy, num_embeddings, num_actions)
        self.policy.to(self.hps.running.device)

        logging.info(f'\npolicy:\n{self.policy}\n')

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hps.running.lr)

    def set_training_status(self, policy):
        self.policy.train(policy)

    def set_update_status(self, policy):
        self.update_pi = policy

    def load(self, fname='agent', with_optim=False):
        load_dict = torch.load(f'{self.hps.running.exp_dir}/{fname}.pth')
        self.policy.load_state_dict(load_dict['policy'])
        if with_optim:
            self.optimizer.load_state_dict(load_dict['optim'])

    def save(self, fname='agent', with_optim=False):
        save_dict = {
            'policy': self.policy.state_dict()
        }
        if with_optim:
            save_dict['optim'] = self.optimizer.state_dict()
        torch.save(save_dict, f'{self.hps.running.exp_dir}/{fname}.pth')

    def _prepare_inputs(self, batch):
        obs = to_torch(batch.obs, device=self.hps.running.device)
        return obs

    def __call__(self, batch: Batch) -> Batch:
        inputs = self._prepare_inputs(batch)
        return self.policy(inputs)

    def map_action(self, act):
        return act

    def map_action_inverse(self, act):
        return act

    def update_policy(self, batch):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            inputs = self._prepare_inputs(minibatch)
            forward = self.policy(inputs)
            # calculate loss for actor
            act = to_torch_as(minibatch.act, forward.policy.vpred)
            ratio = (forward.dist.log_prob(act) - minibatch.policy.logp).exp().float()
            surr1 = ratio * minibatch.adv
            surr2 = ratio.clamp(1.0 - self.hps.agent.ratio_clip, 1.0 + self.hps.agent.ratio_clip) * minibatch.adv
            clip_loss = -torch.min(surr1, surr2).mean()
            # calculate loss for critic
            value = forward.policy.vpred
            vf_loss = (minibatch.returns - value).pow(2).mean()
            # calculate regularization and overall loss
            ent_loss = forward.dist.entropy().mean()
            loss = clip_loss + self.hps.agent.vf_weight * vf_loss - self.hps.agent.ent_weight * ent_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.hps.running.grad_norm:  # clip large gradient
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.hps.running.grad_norm)
            self.optimizer.step()
            clip_losses.append(clip_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent_loss.item())
            losses.append(loss.item())

        return {
            "policy_loss": np.mean(losses),
            "policy_loss_clip": np.mean(clip_losses),
            "policy_loss_vf": np.mean(vf_losses),
            "policy_loss_ent": np.mean(ent_losses),
        }

    def learn(self, batch):
        metrics = defaultdict(list)

        for _ in range(self.hps.running.repeat_per_collect):
            if self.update_pi:
                metric = self.update_policy(batch)
                for k, v in metric.items():
                    metrics[k].append(v)

        return {k: np.mean(v) for k, v in metrics.items()}

class Runner(object):
    def __init__(self, hps):
        self.hps = hps
        env = get_environment(hps.environment)
        self.agent = Agent(hps)
        self.agent._setup(env)

    def _get_env_fn(self):
        
        def _build_env():
            env = get_environment(self.hps.environment)
            env = ConcatAFAWrapper(env, self.hps.environment.cost)
            return env
        
        return _build_env

    def preprocess_fn(self, collector):
        buffer = collector.buffer
        batch, indices = buffer.sample(0)
        # for the end of trajectory (either done or not), the next_indices keeps the same
        next_indices = buffer.next(indices)
        v_s = batch.policy.vpred
        v_s = to_numpy(v_s.flatten())
        v_s_ = buffer[next_indices].policy.vpred
        v_s_ = to_numpy(v_s_.flatten())
        if buffer.unfinished_index().size:
            unfinished_index = buffer.unfinished_index()
            obs_next = buffer[unfinished_index].obs_next
            values = self.agent(obs_next).policy.vpred
            v_idx = [list(indices).index(ind) for ind in unfinished_index]
            v_s_[v_idx] = values.data.cpu().numpy()
        # masking
        value_mask = ~buffer.done[indices]
        v_s_ = v_s_ * value_mask
        end_flag = batch.done.copy()
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(
            v_s, 
            v_s_, 
            batch.rew,
            end_flag, 
            self.hps.agent.gamma, 
            self.hps.agent.gae_lambda
        )
        returns = advantage + v_s
        batch.returns = to_torch_as(returns, batch.policy.vpred)
        batch.adv = to_torch_as(advantage, batch.policy.vpred)
        return batch

    def train(self):
        # environment
        envs = SubprocVectorEnv([self._get_env_fn() for _ in range(self.hps.running.train_env_num)])
        # seed
        np.random.seed(self.hps.running.seed)
        torch.manual_seed(self.hps.running.seed)
        envs.seed(self.hps.running.seed)
        # agent setup
        self.agent.setup_optimizer()
        self.agent.set_training_status(True)
        # buffer
        buffer = VectorReplayBuffer(self.hps.running.buffer_size, len(envs))
        collector = Collector(self.agent, envs, buffer)
        collector.reset_stat()
        # train
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        best_reward = -np.inf

        self.agent.set_update_status(True)
        for step in range(self.hps.running.iterations):
            results = collector.collect(n_episode=self.hps.running.num_train_episodes_per_collect)
            batch = self.preprocess_fn(collector)
            for k, v in results.items():
                writer.add_scalar(f'collect/{k}', v, step)
            losses = self.agent.learn(batch)
            for k, v in losses.items():
                writer.add_scalar(f'losses/{k}', v, step)

            # validation
            if step % self.hps.running.validation_freq == 0:
                metrics = self.valid()
                for k, v in metrics.items():
                    writer.add_scalar(f'valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save()

    def valid(self):
        envs = SubprocVectorEnv([self._get_env_fn()])
        np.random.seed(self.hps.running.seed + 1587)
        torch.manual_seed(self.hps.running.seed + 1587)
        envs.seed(self.hps.running.seed + 1587)
        self.agent.set_training_status(False)
        # buffer
        buffer = VectorReplayBuffer(self.hps.running.buffer_size, 1)
        collector = Collector(self.agent, envs, buffer)

        metrics = Batch()
        for _ in range(self.hps.running.num_valid_episodes):
            collector.reset()
            collector.collect(n_episode=1)
            traj = _gather_traj(buffer)
            metric = _gather_metrics(traj)
            metrics.cat_(metric)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n{json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(True)

        return avg_metrics

    def test(self):
        envs = SubprocVectorEnv([self._get_env_fn()])
        np.random.seed(self.hps.running.seed + 3691)
        torch.manual_seed(self.hps.running.seed + 3691)
        envs.seed(self.hps.running.seed + 3691)
        self.agent.load()
        self.agent.set_training_status(False)
        # buffer
        buffer = VectorReplayBuffer(self.hps.running.buffer_size, 1)
        collector = Collector(self.agent, envs, buffer)

        trajectory = []
        metrics = Batch()
        for _ in range(self.hps.running.num_test_episodes):
            collector.reset()
            collector.collect(n_episode=1)
            traj = _gather_traj(buffer)
            trajectory.append(traj)
            metric = _gather_metrics(traj)
            metrics.cat_(metric)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nTest:\n{json.dumps(avg_metrics, indent=4)}')

        with open(f'{self.hps.running.exp_dir}/trajectory.pkl', 'wb') as f:
            pickle.dump(trajectory, f)

        
def _gather_traj(buffer):
    batch, _ = buffer.sample(0)
    keys = ['obs', 'act', 'rew', 'done', 'info']
    assert np.all(k in batch.keys() for k in keys)
    traj = Batch()
    traj.obs = Batch(observed=batch.obs.observed, mask=batch.obs.mask)
    traj.act = batch.act
    traj.rew = batch.rew
    traj.done = batch.done
    traj.info = batch.info
    return traj

def _gather_metrics(traj):
    episode_length = len(traj)
    episode_reward = sum(traj[i].rew for i in range(len(traj)))
    task_reward = sum(traj[i].info.task_reward for i in range(len(traj)))
    num_acquisitions = sum(traj[i].info.num_acquisitions for i in range(len(traj)))
    return Batch(
        episode_length=np.array([episode_length]),
        episode_reward=np.array([episode_reward]),
        task_reward=np.array([task_reward]),
        num_acquisitions=np.array([num_acquisitions]),
    )
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

from tianshou.data import Batch, to_numpy, to_torch, to_torch_as

from src.environments import get_environment
from src.policies.fully_observed_ppo import *
from src.utils.visualizer import plot_dict

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        # environment specific hyperparameters
        obs_high = env.observation_space.high
        num_embeddings = list(map(int, obs_high + 1))
        num_actions = env.action_space.n
        
        self.policy = build_policy(self.hps.policy, num_embeddings, num_actions)
        self.policy.to(self.hps.running.device)

        logging.info(f'\npolicy:\n{self.policy}\n')

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hps.running.lr)

    def set_training_status(self, pi):
        self.policy.train(pi)

    def set_update_status(self, pi):
        self.update_pi = pi

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

    @torch.no_grad()
    def _prepare_inputs(self, state):
        state = np.expand_dims(state, axis=0)
        return to_torch(state, dtype=torch.float32, device=self.hps.running.device)

    @torch.no_grad()
    def rollout(self, env):
        metrics = defaultdict(float)
        batches = []

        state, done = env.reset(), False
        traj = []
        while not done:
            inputs = self._prepare_inputs(state)
            res = self.agent.policy(inputs)
            act = to_numpy(res.act)[0]
            next_state, reward, done, info = env.step(act)
            data = Batch(
                obs=state, act=res.act[0], rew=reward, done=done,
                policy=Batch(logp=res.policy.logp[0], vpred=res.policy.vpred[0])
            )
            traj.append(data)
            metrics['task_reward'] += reward
            metrics['episode_reward'] += reward
            metrics['episode_length'] += 1
            metrics['num_actions'] += 1
            state = next_state
        batches.append(traj)

        return batches, metrics

    @torch.no_grad()
    def _process_traj(self, traj):
        batch = Batch.stack(traj)
        vpreds = to_numpy(batch.policy.vpred)
        rewards = batch.rew
        td_errors = [rewards[t] + self.hps.agent.gamma * vpreds[t+1] - vpreds[t] for t in range(len(rewards)-1)]
        td_errors += [rewards[-1] + self.hps.agent.gamma * 0.0 - vpreds[-1]]
        advs = []
        adv_so_far = 0.0
        for delta in td_errors[::-1]:
            adv_so_far = delta + self.hps.agent.gamma * self.hps.agent.gae_lambda * adv_so_far
            advs.append(adv_so_far)
        advs = np.array(advs[::-1])
        returns = advs + vpreds
        batch.returns = to_torch_as(returns, batch.policy.vpred)
        batch.adv = to_torch_as(advs, batch.policy.vpred)
        return batch

    @torch.no_grad()
    def collect(self, env):
        batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.train_env_num):
            batch, metric = self.rollout(env)
            batches.extend([self._process_traj(traj) for traj in batch])
            for k, v in metric.items():
                metrics[k].append(v)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        batches = Batch.cat(batches)

        return batches, avg_metrics

    def train(self):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed)
        self.agent.setup_optimizer()
        self.agent.set_training_status(True)
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        reward_history = []
        best_reward = -np.inf

        logging.info('=====Stage 1=====')
        self.agent.set_update_status(True)
        for step in range(self.hps.running.iterations):
            batch, metrics = self.collect(env)
            for k, v in metrics.items():
                writer.add_scalar(f'collect/{k}', v, step)
            losses = self.agent.learn(batch)
            for k, v in losses.items():
                writer.add_scalar(f'losses/{k}', v, step)

            # validation
            if step % self.hps.running.validation_freq == 0:
                logging.info(f'Step: {step}')
                metrics = self.valid()
                for k, v in metrics.items():
                    writer.add_scalar(f'valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save()
                # plot
                reward_history.append(metrics['task_reward'])
                plot_dict(f'{self.hps.running.exp_dir}/reward.png', {'reward': reward_history})

    def valid(self):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed+1)
        self.agent.set_training_status(False)

        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_valid_episodes):
            _, metric = self.rollout(env)
            for k, v in metric.items():
                metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n {json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(True)

        return avg_metrics

    def _record_traj(self, traj):
        batch = Batch.stack(traj)
        new_batch = Batch()
        new_batch.obs = batch.obs
        new_batch.act = batch.act
        new_batch.rew = batch.rew

        return new_batch

    def test(self):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed+2)
        self.agent.load()
        self.agent.set_training_status(False)

        batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_test_episodes):
            batch, metric = self.rollout(env)
            for k, v in metric.items():
                metrics[k].append(v)
            batches.extend([self._record_traj(traj) for traj in batch])
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nTest:\n {json.dumps(avg_metrics, indent=4)}')

        with gzip.open(f'{self.hps.running.exp_dir}/trajectory.pgz', 'wb') as f:
            pickle.dump({'full_batches': batches}, f)

        return avg_metrics
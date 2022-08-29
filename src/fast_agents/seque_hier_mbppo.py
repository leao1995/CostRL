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
from typing import Optional, Union, Any
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
from tianshou.data import Batch, to_numpy, to_torch, to_torch_as
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy.base import _gae_return

from src.environments import get_environment
from src.environments.seque_acquire_wrapper import SequeAFAWrapper
from src.environments.history_augment_wrapper import HistAugWrapper
from src.models import get_model
from src.policies.seque_hier_mbppo import *

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        # environment specific hyperparameters
        obs_high = env.observation_space.high
        num_embeddings = list(map(int, obs_high + 2))
        num_afa_actions = env.num_measurable_features + 1
        num_tsk_actions = env.action_space.n

        self.model = get_model(self.hps.model, env.observation_space, env.action_space)
        self.model.to(self.hps.running.device)
        belief_dim = self.model.belief_dim
        
        self.afa_policy = build_afa_policy(self.hps.policy, belief_dim, num_afa_actions)
        self.afa_policy.to(self.hps.running.device)
        self.tsk_policy = build_tsk_policy(self.hps.policy, belief_dim, num_tsk_actions)
        self.tsk_policy.to(self.hps.running.device)

        logging.info(f'\nmodel:\n{self.model}\n')
        logging.info(f'\nafa_policy:\n{self.afa_policy}\n')
        logging.info(f'\ntsk_policy:\n{self.tsk_policy}\n')

    def setup_optimizer(self):
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.hps.running.lr_model)
        self.afa_optimizer = optim.Adam(self.afa_policy.parameters(), lr=self.hps.running.lr_afa)
        self.tsk_optimizer = optim.Adam(self.tsk_policy.parameters(), lr=self.hps.running.lr_tsk)

    def set_training_status(self, model, afa, tsk):
        self.model.train(model)
        self.afa_policy.train(afa)
        self.tsk_policy.train(tsk)

    def set_update_status(self, model, afa, tsk):
        self.update_mod = model
        self.update_afa = afa
        self.update_tsk = tsk

    def load(self, fname='agent', with_optim=False):
        load_dict = torch.load(f'{self.hps.running.exp_dir}/{fname}.pth')
        self.model.load_state_dict(load_dict['model'])
        self.afa_policy.load_state_dict(load_dict['afa'])
        self.tsk_policy.load_state_dict(load_dict['tsk'])
        if with_optim:
            self.model_optimizer.load_state_dict(load_dict['model_optim'])
            self.afa_optimizer.load_state_dict(load_dict['afa_optim'])
            self.tsk_optimizer.load_state_dict(load_dict['tsk_optim'])

    def save(self, fname='agent', with_optim=False):
        save_dict = {
            'model': self.model.state_dict(),
            'afa': self.afa_policy.state_dict(),
            'tsk': self.tsk_policy.state_dict()
        }
        if with_optim:
            save_dict['model_optim'] = self.model_optimizer.state_dict()
            save_dict['afa_optim'] = self.afa_optimizer.state_dict()
            save_dict['tsk_optim'] = self.tsk_optimizer.state_dict()
        torch.save(save_dict, f'{self.hps.running.exp_dir}/{fname}.pth')

    def _prepare_inputs(self, batch):
        full = np.concatenate([batch.obs.hist.full, np.expand_dims(batch.obs.full, axis=1)], axis=1)
        observed = np.concatenate([batch.obs.hist.observed, np.expand_dims(batch.obs.observed, axis=1)], axis=1)
        mask = np.concatenate([batch.obs.hist.mask, np.expand_dims(batch.obs.mask, axis=1)], axis=1)
        action = batch.obs.hist.action

        full = to_torch(full, dtype=torch.float32, device=self.hps.running.device)
        observed = to_torch(observed, dtype=torch.float32, device=self.hps.running.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.hps.running.device)
        action = to_torch(action, dtype=torch.long, device=self.hps.running.device)

        with torch.no_grad():
            belief = self.model.belief(observed, mask, action, 
                self.hps.agent.num_belief_samples, keep_last=True)

        obs = to_torch(batch.obs, device=self.hps.running.device)
        obs.belief = belief
        obs.hist = Batch(full=full, observed=observed, mask=mask, action=action)

        return obs

    def __call__(
        self, 
        batch: Batch, 
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
        ) -> Batch:
        inputs = self._prepare_inputs(batch)
        afa_res = self.afa_policy(inputs)
        tsk_res = self.tsk_policy(inputs)

        return Batch(
            dist=Batch(afa_dist=afa_res.dist, tsk_dist=tsk_res.dist),
            act=Batch(afa_action=afa_res.act, tsk_action=tsk_res.act),
            policy=Batch(afa_policy=afa_res.policy, tsk_policy=tsk_res.policy)
        )

    def map_action(self, act):
        return act

    def map_action_inverse(self, act):
        return act

    def _update_afa_policy(self, minibatch):
        inputs = self._prepare_inputs(minibatch)
        forward = self.afa_policy(inputs)
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
        self.afa_optimizer.zero_grad()
        loss.backward()
        if self.hps.running.grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(self.afa_policy.parameters(), max_norm=self.hps.running.grad_norm)
        self.afa_optimizer.step()

        return {
            'afa_loss': loss.item(),
            'afa_clip_loss': clip_loss.item(),
            'afa_vf_loss': vf_loss.item(),
            'afa_ent_loss': ent_loss.item()
        }

    def _update_tsk_policy(self, minibatch):
        inputs = self._prepare_inputs(minibatch)
        forward = self.tsk_policy(inputs)
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
        self.tsk_optimizer.zero_grad()
        loss.backward()
        if self.hps.running.grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(self.tsk_policy.parameters(), max_norm=self.hps.running.grad_norm)
        self.tsk_optimizer.step()

        return {
            'tsk_loss': loss.item(),
            'tsk_clip_loss': clip_loss.item(),
            'tsk_vf_loss': vf_loss.item(),
            'tsk_ent_loss': ent_loss.item()
        }

    def _update_model(self, minibatch):
        inputs = self._prepare_inputs(minibatch)
        losses = self.model.loss(inputs.hist.full, inputs.hist.mask, inputs.hist.action)
        self.model_optimizer.zero_grad()
        losses['model_loss'].backward()
        if self.hps.running.grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hps.running.grad_norm)
        self.model_optimizer.step()

        acc = self.model.accuracy(inputs.hist.full, inputs.hist.mask, inputs.hist.action, 10)
        losses['acc'] = acc

        return losses

    def update_afa_policy(self, batch):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            metric = self._update_afa_policy(minibatch)
            clip_losses.append(metric['afa_clip_loss'])
            vf_losses.append(metric['afa_vf_loss'])
            ent_losses.append(metric['afa_ent_loss'])
            losses.append(metric['afa_loss'])

        return {
            "afa_loss": np.mean(losses),
            "afa_loss_clip": np.mean(clip_losses),
            "afa_loss_vf": np.mean(vf_losses),
            "afa_loss_ent": np.mean(ent_losses),
        }

    def update_tsk_policy(self, batch):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            metric = self._update_tsk_policy(minibatch)
            clip_losses.append(metric['tsk_clip_loss'])
            vf_losses.append(metric['tsk_vf_loss'])
            ent_losses.append(metric['tsk_ent_loss'])
            losses.append(metric['tsk_loss'])

        return {
            "tsk_loss": np.mean(losses),
            "tsk_loss_clip": np.mean(clip_losses),
            "tsk_loss_vf": np.mean(vf_losses),
            "tsk_loss_ent": np.mean(ent_losses),
        }

    def update_model(self, batch):
        metrics = defaultdict(list)
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            losses = self._update_model(minibatch)
            for k, v in losses.items():
                metrics[k].append(v.item())
        
        return {k: np.mean(v) for k, v in metrics.items()}

    def learn(self, afa_batch, tsk_batch):
        metrics = defaultdict(list)

        for _ in range(self.hps.running.steps_per_collect):
            if self.update_afa:
                afa_indices = np.random.choice(len(afa_batch), self.hps.running.batch_size)
                afa_minibatch = afa_batch[afa_indices]
            
            if self.update_mod or self.update_tsk:
                tsk_indices = np.random.choice(len(tsk_batch), self.hps.running.batch_size)
                tsk_minibatch = tsk_batch[tsk_indices]

            if self.update_afa:
                metric = self._update_afa_policy(afa_minibatch)
                for k, v in metric.items():
                    metrics[k].append(v)

            if self.update_mod:
                metric = self._update_model(tsk_minibatch)
                for k, v in metric.items():
                    metrics[k].append(v.item())

            if self.update_tsk:
                metric = self._update_tsk_policy(tsk_minibatch)
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
            env = SequeAFAWrapper(env, self.hps.environment.cost)
            env = HistAugWrapper(env, self.hos.agent.max_history_length)
            return env
        
        return _build_env

    def preprocess_fn(self, collector):
        buffer = collector.buffer
        assert buffer.unfinished_index().size == 0
        batch, indices = buffer.sample(0)
        afa_indices = np.where(batch.info.is_acquisition)[0]
        term_indices = np.where(batch.info.is_terminal)[0]
        tsk_indices = np.where(~batch.info.is_acquisition)[0]
        # afa
        value = batch.policy.afa_policy.vpred
        value = to_numpy(value.flatten())
        v_s = value[afa_indices]
        next_indices = np.roll(afa_indices, -1)
        v_s_ = value[next_indices]
        value_mask = ~batch.info.is_terminal[afa_indices]
        v_s_ = v_s_ * value_mask
        if self.hps.agent.terminal_reward_weight > 0:
            inputs = batch.obs_next[term_indices]
            inputs = to_torch(inputs, device=self.hps.running.device)
            term_rew = self._terminal_reward(inputs)
            rew = batch.rew.copy()
            rew[term_indices] += term_rew * self.hps.agent.terminal_reward_weight
            rew = rew[afa_indices]
        else:
            rew = batch.rew[afa_indices]
        end_flag = batch.done.copy()
        end_flag[term_indices] = True
        end_flag = end_flag[afa_indices]
        afa_advantage = _gae_return(
            v_s, 
            v_s_, 
            rew,
            end_flag, 
            self.hps.agent.gamma, 
            self.hps.agent.gae_lambda
        )
        afa_obs = batch.obs[afa_indices]
        afa_returns = to_torch_as(afa_advantage + v_s, batch.policy.afa_policy.vpred)
        afa_advantage = to_torch_as(afa_advantage, batch.policy.afa_policy.vpred)
        afa_logp = batch.policy.afa_policy.logp[afa_indices]
        afa_action = batch.act.afa_action[afa_indices]
        afa_batch = Batch(
            obs=afa_obs, act=afa_action, rew=rew, done=end_flag,
            policy=Batch(logp=afa_logp, vpred=v_s),
            returns=afa_returns, adv=afa_advantage
        )
        # tsk
        value = batch.policy.tsk_policy.vpred
        value = to_numpy(value.flatten())
        v_s = value[tsk_indices]
        next_indices = np.roll(tsk_indices, -1)
        v_s_ = value[next_indices]
        value_mask = ~batch.done[tsk_indices]
        v_s_ = v_s_ * value_mask
        rew = batch.rew[tsk_indices]
        end_flag = batch.done[tsk_indices]
        tsk_advantage = _gae_return(
            v_s, 
            v_s_, 
            rew,
            end_flag, 
            self.hps.agent.gamma, 
            self.hps.agent.gae_lambda
        )
        tsk_obs = batch.obs[tsk_indices]
        tsk_returns = to_torch_as(tsk_advantage + v_s, batch.policy.tsk_policy.vpred)
        tsk_advantage = to_torch_as(tsk_advantage, batch.policy.tsk_policy.vpred)
        tsk_logp = batch.policy.tsk_policy.logp[tsk_indices]
        tsk_action = batch.act.tsk_action[tsk_indices]
        tsk_batch = Batch(
            obs=tsk_obs, act=tsk_action, rew=rew, done=end_flag,
            policy=Batch(logp=tsk_logp, vpred=v_s),
            returns=tsk_returns, adv=tsk_advantage
        )

        return afa_batch, tsk_batch

    def _terminal_reward(self, inputs):
        if self.hps.agent.terminal_reward_type == 'value':
            rew = to_numpy(self.agent.tsk_policy.critic(inputs))
        elif self.hps.agent.terminal_reward_type == 'entropy':
            rew = - to_numpy(self.agent.tsk_policy.actor(inputs).entropy())
        elif self.hps.agent.terminal_reward_type == 'impute':
            rew = to_numpy(self.agent.model.reward(inputs.hist.full, inputs.hist.mask, inputs.hist.action, 10))
        elif self.hps.agent.terminal_reward_type == 'hybrid':
            rew1 = to_numpy(self.agent.tsk_policy.critic(inputs))
            rew2 = - to_numpy(self.agent.tsk_policy.actor(inputs).entropy())
            rew3 = to_numpy(self.agent.model.reward(inputs.hist.full, inputs.hist.mask, inputs.hist.action, 10))
            rew = rew1 + rew2 + rew3
        else:
            raise NotImplementedError()

        return rew

    def train(self):
        # environment
        envs = SubprocVectorEnv([self._get_env_fn() for _ in range(self.hps.running.train_env_num)])
        # seed
        np.random.seed(self.hps.running.seed)
        torch.manual_seed(self.hps.running.seed)
        envs.seed(self.hps.running.seed)
        # agent setup
        self.agent.setup_optimizer()
        self.agent.set_training_status(model=True, afa=True, tsk=True)
        # buffer
        buffer = VectorReplayBuffer(self.hps.running.buffer_size, len(envs))
        collector = Collector(self.agent, envs, buffer)
        collector.reset_stat()
        # train
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        best_reward = -np.inf
        best_loss = np.inf

        # stage1: train model  rand_afa=True  rand_tsk=True
        self.agent.set_update_status(model=True, afa=False, tsk=False)
        self.agent.afa_policy.set_temperature(5.0)
        self.agent.tsk_policy.set_temperature(5.0)
        for step in range(self.hps.running.stage1_iterations):
            results = collector.collect(n_episode=self.hps.running.num_train_episodes_per_collect)
            metrics = _gather_results(results)
            afa_batch, tsk_batch = self.preprocess_fn(collector)
            for k, v in metrics.items():
                writer.add_scalar(f'stage1_collect/{k}', v, step)
            losses = self.agent.learn(afa_batch, tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage1_losses/{k}', v, step)

            # save
            if losses['model_loss'] <= best_loss:
                best_loss = losses['model_loss']
                self.agent.save('stage1_best', with_optim=True)

        # save last
        self.agent.save('stage1_last', with_optim=True)

        # stage2: train tsk_policy  rand_afa=True rand_tsk=False
        self.agent.set_update_status(model=False, afa=False, tsk=True)
        self.agent.afa_policy.set_temperature(5.0)
        self.agent.tsk_policy.set_temperature(1.0)
        for step in range(self.hps.running.stage2_iterations):
            results = collector.collect(n_episode=self.hps.running.num_train_episodes_per_collect)
            metrics = _gather_results(results)
            afa_batch, tsk_batch = self.preprocess_fn(collector)
            for k, v in metrics.items():
                writer.add_scalar(f'stage2_collect/{k}', v, step)
            losses = self.agent.learn(afa_batch, tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage2_losses/{k}', v, step)

            # validation
            if step % self.hps.running.validation_freq == 0:
                metrics = self.valid()
                for k, v in metrics.items():
                    writer.add_scalar(f'stage2_valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save('stage2_best', with_optim=True)
        
        # save last
        self.agent.save('stage2_last', with_optim=True)

        # stage3: joint training
        self.agent.set_update_status(model=True, afa=True, tsk=True)
        self.agent.afa_policy.set_temperature(1.0)
        self.agent.tsk_policy.set_temperature(1.0)
        for step in range(self.hps.running.stage3_iterations):
            results = collector.collect(n_episode=self.hps.running.num_train_episodes_per_collect)
            metrics = _gather_results(results)
            afa_batch, tsk_batch = self.preprocess_fn(collector)
            for k, v in metrics.items():
                writer.add_scalar(f'stage3_collect/{k}', v, step)
            losses = self.agent.learn(afa_batch, tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage3_losses/{k}', v, step)
            
            # validation
            if step % self.hps.running.validation_freq == 0:
                metrics = self.valid()
                for k, v in metrics.items():
                    writer.add_scalar(f'stage3_valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save(with_optim=False)

    def valid(self):
        envs = SubprocVectorEnv([self._get_env_fn()])
        np.random.seed(self.hps.running.seed + 1587)
        torch.manual_seed(self.hps.running.seed + 1587)
        envs.seed(self.hps.running.seed + 1587)
        self.agent.set_training_status(model=False, afa=False, tsk=False)
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

        self.agent.set_training_status(model=True, afa=True, tsk=True)

        return avg_metrics

    def test(self):
        envs = SubprocVectorEnv([self._get_env_fn()])
        np.random.seed(self.hps.running.seed + 3691)
        torch.manual_seed(self.hps.running.seed + 3691)
        envs.seed(self.hps.running.seed + 3691)
        self.agent.load()
        self.agent.set_training_status(model=False, afa=False, tsk=False)
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


def _gather_results(results):
    metrics = {
        "num_episodes": results["n/ep"],
        "num_transitions": results["n/st"],
        "episode_reward": results["rew"],
        "episode_length": results["len"],
        "episode_reward_std": results["rew_std"],
        "episode_length_std": results["len_std"]
    }
    return metrics

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
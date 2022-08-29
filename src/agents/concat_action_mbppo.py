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
from src.environments.concat_action_wrapper import ConcatAFAWrapper
from src.models import get_model
from src.policies.concat_action_mbppo import *

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        # environment specific hyperparameters
        obs_high = env.observation_space.high
        num_embeddings = list(map(int, obs_high + 2))
        num_actions = env.num_measurable_features + env.action_space.n

        self.model = get_model(self.hps.model, env.observation_space, env.action_space)
        self.model.to(self.hps.running.device)
        belief_dim = self.model.belief_dim
        
        self.policy = build_policy(self.hps.policy, belief_dim, num_actions)
        self.policy.to(self.hps.running.device)

        logging.info(f'\nmodel:\n{self.model}\n')
        logging.info(f'\npolicy:\n{self.policy}\n')

    def setup_optimizer(self):
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.hps.running.lr_model)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.hps.running.lr_policy)

    def set_training_status(self, model, policy):
        self.model.train(model)
        self.policy.train(policy)

    def set_update_status(self, model, policy):
        self.update_mod = model
        self.update_pi = policy

    def load(self, fname='agent', with_optim=False):
        load_dict = torch.load(f'{self.hps.running.exp_dir}/{fname}.pth')
        self.model.load_state_dict(load_dict['model'])
        self.policy.load_state_dict(load_dict['policy'])
        if with_optim:
            self.model_optimizer.load_state_dict(load_dict['model_optim'])
            self.policy_optimizer.load_state_dict(load_dict['policy_optim'])

    def save(self, fname='agent', with_optim=False):
        save_dict = {
            'model': self.model.state_dict(),
            'policy': self.policy.state_dict()
        }
        if with_optim:
            save_dict['model_optim'] = self.model_optimizer.state_dict()
            save_dict['policy_optim'] = self.policy_optimizer.state_dict()
        torch.save(save_dict, f'{self.hps.running.exp_dir}/{fname}.pth')

    def _prepare_inputs(self, batch):
        full = np.concatenate([batch.hist.full, np.expand_dims(batch.full, axis=1)], axis=1)
        observed = np.concatenate([batch.hist.observed, np.expand_dims(batch.obs.observed, axis=1)], axis=1)
        mask = np.concatenate([batch.hist.mask, np.expand_dims(batch.obs.mask, axis=1)], axis=1)
        action = batch.hist.action

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

    def _update_policy(self, minibatch):
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
        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.hps.running.grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.hps.running.grad_norm)
        self.policy_optimizer.step()

        return {
            'policy_loss': loss.item(),
            'policy_clip_loss': clip_loss.item(),
            'policy_vf_loss': vf_loss.item(),
            'policy_ent_loss': ent_loss.item()
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

    def update_policy(self, batch):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            metric = self._update_policy(minibatch)
            clip_losses.append(metric['policy_clip_loss'])
            vf_losses.append(metric['policy_vf_loss'])
            ent_losses.append(metric['policy_ent_loss'])
            losses.append(metric['policy_loss'])

        return {
            "policy_loss": np.mean(losses),
            "policy_loss_clip": np.mean(clip_losses),
            "policy_loss_vf": np.mean(vf_losses),
            "policy_loss_ent": np.mean(ent_losses),
        }

    def update_model(self, batch):
        metrics = defaultdict(list)
        for minibatch in batch.split(self.hps.running.batch_size, merge_last=True):
            losses = self._update_model(minibatch)
            for k, v in losses.items():
                metrics[k].append(v.item())
        
        return {k: np.mean(v) for k, v in metrics.items()}

    def learn(self, batch):
        metrics = defaultdict(list)

        for _ in range(self.hps.running.steps_per_collect):
            indices = np.random.choice(len(batch), self.hps.running.batch_size)
            minibatch = batch[indices]
            
            if self.update_mod:
                metric = self._update_model(minibatch)
                for k, v in metric.items():
                    metrics[k].append(v.item())
            
            if self.update_pi:
                metric = self._update_policy(minibatch)
                for k, v in metric.items():
                    metrics[k].append(v)

        return {k: np.mean(v) for k, v in metrics.items()}

class History(object):
    def __init__(self, obs_shape, max_history_length):
        self.max_history_length = max_history_length
        self.full = deque(maxlen=max_history_length)
        self.observed = deque(maxlen=max_history_length)
        self.mask = deque(maxlen=max_history_length)
        self.action = deque(maxlen=max_history_length)

        for _ in range(max_history_length):
            self.full.append(np.zeros(obs_shape))
            self.observed.append(np.zeros(obs_shape))
            self.mask.append(np.zeros(obs_shape))
            self.action.append(-1)

    def append(self, full, observed, mask, action):
        self.full.append(full)
        self.observed.append(observed)
        self.mask.append(mask)
        self.action.append(action)

    def get(self):
        return Batch(
            full=np.array(self.full),
            observed=np.array(self.observed),
            mask=np.array(self.mask),
            action=np.array(self.action)
        )

class Runner(object):
    def __init__(self, hps):
        self.hps = hps
        env = get_environment(hps.environment)
        self.agent = Agent(hps)
        self.agent._setup(env)

    @torch.no_grad()
    def _prepare_inputs(self, full, obs, history):
        full = np.expand_dims(np.vstack([history.full, full]), axis=0)
        observed = np.expand_dims(np.vstack([history.observed, obs.observed]), axis=0)
        mask = np.expand_dims(np.vstack([history.mask, obs.mask]), axis=0)
        action = np.expand_dims(history.action, axis=0)

        full = to_torch(full, dtype=torch.float32, device=self.hps.running.device)
        observed = to_torch(observed, dtype=torch.float32, device=self.hps.running.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.hps.running.device)
        action = to_torch(action, dtype=torch.long, device=self.hps.running.device)
        belief = self.agent.model.belief(observed, mask, action, 
            self.hps.agent.num_belief_samples, keep_last=True)

        new_obs = Batch()
        for k, v in obs.items():
            new_obs[k] = np.expand_dims(v, axis=0)
        obs = to_torch(new_obs, device=self.hps.running.device)
        obs.belief = belief
        obs.hist = Batch(full=full, observed=observed, mask=mask, action=action)

        return obs

    @torch.no_grad()
    def rollout(self, env):
        metrics = defaultdict(float)
        batches = []

        obs, done = env.reset(), False
        history = History(env.observation_space['observed'].shape, self.hps.agent.max_history_length)
        while not done:
            state = env.state
            obs = Batch(obs)
            inputs = self._prepare_inputs(state, obs, history.get())
            res = self.agent.policy(inputs)
            act = to_numpy(res.act)[0]
            next_obs, reward, done, info = env.step(act)
            data = Batch(
                full=state, obs=obs, hist=history.get(), act=res.act[0], rew=reward, done=done,
                policy=Batch(logp=res.policy.logp[0], vpred=res.policy.vpred[0])
            )
            batches.append(data)
            metrics['episode_reward'] += reward
            metrics['episode_length'] += 1
            metrics['task_reward'] += info['task_reward']
            metrics['num_acquisitions'] += info['num_acquisitions']
            metrics['num_tsk_actions'] += 0 if info['is_acquisition'] else 1
            if not info['is_acquisition']:
                history.append(state, obs.observed, obs.mask, info['tsk_action'])
            obs = next_obs
        
        metrics['num_acquisitions_per_action'] = metrics['num_acquisitions'] / metrics['num_tsk_actions']

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
            batches.append(self._process_traj(batch))
            for k, v in metric.items():
                metrics[k].append(v)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        batches = Batch.cat(batches)

        return batches, avg_metrics

    def train(self):
        env = get_environment(self.hps.environment)
        env = ConcatAFAWrapper(env, self.hps.environment.cost)
        env.seed(self.hps.running.seed)
        self.agent.setup_optimizer()
        self.agent.set_training_status(model=True, policy=True)
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        best_reward = -np.inf

        self.agent.set_update_status(model=True, policy=True)
        for step in range(self.hps.running.iterations):
            batch, metrics = self.collect(env)
            for k, v in metrics.items():
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
        env = get_environment(self.hps.environment)
        env = ConcatAFAWrapper(env, self.hps.environment.cost)
        env.seed(self.hps.running.seed+1)
        self.agent.set_training_status(model=False, policy=False)

        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_valid_episodes):
            _, metric = self.rollout(env)
            for k, v in metric.items():
                metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n{json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(model=True, policy=True)

        return avg_metrics
        
    def _record_traj(self, traj):
        batch = Batch.stack(traj)
        new_batch = Batch()
        new_batch.full = batch.full
        new_batch.obs = batch.obs
        new_batch.act = batch.act
        new_batch.rew = batch.rew

        return new_batch

    def test(self):
        env = get_environment(self.hps.environment)
        env = ConcatAFAWrapper(env, self.hps.environment.cost)
        env.seed(self.hps.running.seed+2)
        self.agent.load()
        self.agent.set_training_status(model=False, policy=False)

        batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_test_episodes):
            batch, metric = self.rollout(env)
            for k, v in metric.items():
                metrics[k].append(v)
            batches.append(self._record_traj(batch))
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nTest:\n{json.dumps(avg_metrics, indent=4)}')

        with gzip.open(f'{self.hps.running.exp_dir}/trajectory.pgz', 'wb') as f:
            pickle.dump({"cat_batches": batches}, f)

        return avg_metrics
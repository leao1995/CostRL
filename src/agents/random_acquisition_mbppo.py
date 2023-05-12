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
from src.environments.batch_acquire_env import AcquireEnv
from src.models import get_model
from src.policies.random_acquisition_mbppo import PolicyBuilder
from src.utils.visualizer import plot_dict

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        self.model = get_model(self.hps.model, env.observation_space, env.action_space)
        self.model.to(self.hps.running.device)
        self.hps.policy.belief_dim = self.model.belief_dim

        policy_builder = PolicyBuilder(env, self.hps.policy)
        
        self.tsk_policy = policy_builder.build_tsk_policy()
        self.tsk_policy.to(self.hps.running.device)

        logging.info(f'\nmodel:\n{self.model}\n')
        logging.info(f'\ntsk_policy:\n{self.tsk_policy}\n')

    def setup_optimizer(self):
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.hps.running.lr_model)
        self.tsk_optimizer = optim.Adam(self.tsk_policy.parameters(), lr=self.hps.running.lr_tsk)

    def set_training_status(self, model, tsk):
        self.model.train(model)
        self.tsk_policy.train(tsk)

    def set_update_status(self, model, tsk):
        self.update_mod = model
        self.update_tsk = tsk

    def load(self, fname='agent', with_optim=False):
        load_dict = torch.load(f'{self.hps.running.exp_dir}/{fname}.pth')
        self.model.load_state_dict(load_dict['model'])
        self.tsk_policy.load_state_dict(load_dict['tsk'])
        if with_optim:
            self.model_optimizer.load_state_dict(load_dict['model_optim'])
            self.tsk_optimizer.load_state_dict(load_dict['tsk_optim'])

    def save(self, fname='agent', with_optim=False):
        save_dict = {
            'model': self.model.state_dict(),
            'tsk': self.tsk_policy.state_dict()
        }
        if with_optim:
            save_dict['model_optim'] = self.model_optimizer.state_dict()
            save_dict['tsk_optim'] = self.tsk_optimizer.state_dict()
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

    def learn(self, tsk_batch):
        metrics = defaultdict(list)

        for _ in range(self.hps.running.steps_per_collect):
            tsk_indices = np.random.choice(len(tsk_batch), self.hps.running.batch_size)
            tsk_minibatch = tsk_batch[tsk_indices]

            if self.update_mod:
                metric = self._update_model(tsk_minibatch)
                for k, v in metric.items():
                    metrics[k].append(v.item())

            if self.update_tsk:
                metric = self._update_tsk_policy(tsk_minibatch)
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

    def _random_acquisition(self, env, state):
        N = self.hps.agent.num_acquisitions_per_step # number of acquired features
        idx = np.random.choice(env.measurable_feature_ids, N, replace=False)
        mask = [i in idx or i not in env.measurable_feature_ids for i in range(env.num_observable_features)]
        mask = np.array(mask, dtype=np.float32)
        observed = state * mask

        return Batch(observed=observed, mask=mask)

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
    def rollout(self, env, rand_tsk):
        metrics = defaultdict(float)
        tsk_batches = []

        state, done = env.reset(), False
        tsk_traj = []
        history = History(env.observation_space.shape, self.hps.agent.max_history_length)
        while not done:
            obs = self._random_acquisition(env, state)
            tsk_data = Batch(full=state, obs=obs, hist=history.get())
            if rand_tsk:
                act = env.action_space.sample()
                tsk_data.update(act=act)
            else:
                inputs = self._prepare_inputs(state, obs, history.get())
                tsk_res = self.agent.tsk_policy(inputs)
                act = to_numpy(tsk_res.act)[0]
                tsk_data.update(act=tsk_res.act[0])
                tsk_data.update(policy=Batch(logp=tsk_res.policy.logp[0], vpred=tsk_res.policy.vpred[0]))
            next_state, reward, done, info = env.step(act)
            tsk_data.update(rew=reward, done=done)
            tsk_traj.append(tsk_data)
            metrics['task_reward'] += reward
            metrics['episode_reward'] += reward
            metrics['episode_length'] += 1
            metrics['num_tsk_actions'] += 1
            history.append(state, obs.observed, obs.mask, act)
            state = next_state
        tsk_batches.append(tsk_traj)

        metrics['num_acquisitions'] = self.hps.agent.num_acquisitions_per_step * metrics['num_tsk_actions']
        metrics['num_acquisitions_per_action'] = self.hps.agent.num_acquisitions_per_step

        return tsk_batches, metrics

    @torch.no_grad()
    def _process_traj(self, traj):
        batch = Batch.stack(traj)
        if not hasattr(batch, 'policy'): return batch # random acquired
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
    def collect(self, env, rand_tsk):
        tsk_batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.train_env_num):
            tsk_batch, metric = self.rollout(env, rand_tsk)
            tsk_batches.extend([self._process_traj(traj) for traj in tsk_batch])
            for k, v in metric.items():
                metrics[k].append(v)
            
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        tsk_batches = Batch.cat(tsk_batches)

        return tsk_batches, avg_metrics

    def train(self):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed)
        self.agent.setup_optimizer()
        self.agent.set_training_status(model=True, tsk=True)
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        reward_history = []
        best_reward = -np.inf
        best_loss = np.inf

        # stage1: train model  rand_afa=True  rand_tsk=True
        logging.info('=====Stage 1=====')
        self.agent.set_update_status(model=True, tsk=False)
        for step in range(self.hps.running.stage1_iterations):
            tsk_batch, metrics = self.collect(env, rand_tsk=True)
            for k, v in metrics.items():
                writer.add_scalar(f'stage1_collect/{k}', v, step)
            losses = self.agent.learn(tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage1_losses/{k}', v, step)

            # save
            if losses['model_loss'] <= best_loss:
                best_loss = losses['model_loss']
                self.agent.save('stage1_best', with_optim=True)
        
        # save last
        self.agent.save('stage1_last', with_optim=True)

        # stage2: joint training
        logging.info('=====Stage 2=====')
        self.agent.set_update_status(model=True, tsk=True)
        for step in range(self.hps.running.stage2_iterations):
            tsk_batch, metrics = self.collect(env, rand_tsk=False)
            for k, v in metrics.items():
                writer.add_scalar(f'stage2_collect/{k}', v, step)
            losses = self.agent.learn(tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage2_losses/{k}', v, step)
            
            # validation
            if step % self.hps.running.validation_freq == 0:
                logging.info(f'Step: {step}')
                metrics = self.valid(rand_tsk=False)
                for k, v in metrics.items():
                    writer.add_scalar(f'stage2_valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save(with_optim=False)
                # plot
                reward_history.append(metrics['task_reward'])
                plot_dict(f'{self.hps.running.exp_dir}/reward.png', {'reward': reward_history})

    def valid(self, rand_tsk):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed+1)
        self.agent.set_training_status(model=False, tsk=False)

        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_valid_episodes):
            _, metric = self.rollout(env, rand_tsk)
            for k, v in metric.items():
                metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n{json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(model=True, tsk=True)

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
        env.seed(self.hps.running.seed+2)
        self.agent.load()
        self.agent.set_training_status(model=False, tsk=False)

        tsk_batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_test_episodes):
            tsk_batch, metric = self.rollout(env, False)
            for k, v in metric.items():
                metrics[k].append(v)
            tsk_batches.append([self._record_traj(traj) for traj in tsk_batch])
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nTest:\n{json.dumps(avg_metrics, indent=4)}')

        with gzip.open(f'{self.hps.running.exp_dir}/trajectory.pgz', 'wb') as f:
            pickle.dump({'tsk_batches': tsk_batches}, f)

        return avg_metrics
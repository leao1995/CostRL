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
from src.environments.seque_acquire_env import AcquireEnv
from src.policies.seque_hier_hbppo import *
from src.utils.visualizer import plot_dict

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        # environment specific hyperparameters
        obs_high = env.observation_space.high
        num_embeddings = list(map(int, obs_high + 2))
        num_afa_actions = env.num_measurable_features + 1 # termination action
        num_tsk_actions = env.action_space.n

        self.afa_policy = build_afa_policy(self.hps.policy, num_embeddings, num_afa_actions)
        self.afa_policy.to(self.hps.running.device)
        self.tsk_policy = build_tsk_policy(self.hps.policy, num_embeddings, num_tsk_actions)
        self.tsk_policy.to(self.hps.running.device)

        logging.info(f'\nafa_policy:\n{self.afa_policy}\n')
        logging.info(f'\ntsk_policy:\n{self.tsk_policy}\n')

    def setup_optimizer(self):
        self.afa_optimizer = optim.Adam(self.afa_policy.parameters(), lr=self.hps.running.lr_afa)
        self.tsk_optimizer = optim.Adam(self.tsk_policy.parameters(), lr=self.hps.running.lr_tsk)

    def set_training_status(self, afa, tsk):
        self.afa_policy.train(afa)
        self.tsk_policy.train(tsk)

    def set_update_status(self, afa, tsk):
        self.update_afa = afa
        self.update_tsk = tsk

    def load(self, fname='agent', with_optim=False):
        load_dict = torch.load(f'{self.hps.running.exp_dir}/{fname}.pth')
        self.afa_policy.load_state_dict(load_dict['afa'])
        self.tsk_policy.load_state_dict(load_dict['tsk'])
        if with_optim:
            self.afa_optimizer.load_state_dict(load_dict['afa_optim'])
            self.tsk_optimizer.load_state_dict(load_dict['tsk_optim'])

    def save(self, fname='agent', with_optim=False):
        save_dict = {
            'afa': self.afa_policy.state_dict(),
            'tsk': self.tsk_policy.state_dict()
        }
        if with_optim:
            save_dict['afa_optim'] = self.afa_optimizer.state_dict()
            save_dict['tsk_optim'] = self.tsk_optimizer.state_dict()
        torch.save(save_dict, f'{self.hps.running.exp_dir}/{fname}.pth')

    def _prepare_inputs(self, batch):
        observed = np.concatenate([batch.hist.observed, np.expand_dims(batch.obs.observed, axis=1)], axis=1)
        mask = np.concatenate([batch.hist.mask, np.expand_dims(batch.obs.mask, axis=1)], axis=1)
        action = batch.hist.action

        observed = to_torch(observed, dtype=torch.float32, device=self.hps.running.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.hps.running.device)
        action = to_torch(action, dtype=torch.long, device=self.hps.running.device)

        obs = to_torch(batch.obs, device=self.hps.running.device)
        obs.hist = Batch(observed=observed, mask=mask, action=action)
        return obs

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

    def learn(self, afa_batch, tsk_batch):
        metrics = defaultdict(list)

        for _ in range(self.hps.running.steps_per_collect):
            if self.update_afa:
                afa_indices = np.random.choice(len(afa_batch), self.hps.running.batch_size)
                afa_minibatch = afa_batch[afa_indices]
            
            if self.update_tsk:
                tsk_indices = np.random.choice(len(tsk_batch), self.hps.running.batch_size)
                tsk_minibatch = tsk_batch[tsk_indices]

            if self.update_afa:
                metric = self._update_afa_policy(afa_minibatch)
                for k, v in metric.items():
                    metrics[k].append(v)

            if self.update_tsk:
                metric = self._update_tsk_policy(tsk_minibatch)
                for k, v in metric.items():
                    metrics[k].append(v)

        return {k: np.mean(v) for k, v in metrics.items()}

class History(object):
    def __init__(self, obs_shape, max_history_length):
        self.max_history_length = max_history_length
        self.observed = deque(maxlen=max_history_length)
        self.mask = deque(maxlen=max_history_length)
        self.action = deque(maxlen=max_history_length)

        for _ in range(max_history_length):
            self.observed.append(np.zeros(obs_shape))
            self.mask.append(np.zeros(obs_shape))
            self.action.append(-1)

    def append(self, observed, mask, action):
        self.observed.append(observed)
        self.mask.append(mask)
        self.action.append(action)

    def get(self):
        return Batch(
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
        N = np.random.randint(0, env.num_measurable_features+1) # number of acquired features
        idx = np.random.choice(env.measurable_feature_ids, N, replace=False)
        mask = [i in idx or i not in env.measurable_feature_ids for i in range(env.num_observable_features)]
        mask = np.array(mask, dtype=np.float32)
        observed = state * mask

        return Batch(observed=observed, mask=mask)

    @torch.no_grad()
    def _prepare_inputs(self, obs, history):
        observed = np.expand_dims(np.vstack([history.observed, obs.observed]), axis=0)
        mask = np.expand_dims(np.vstack([history.mask, obs.mask]), axis=0)
        action = np.expand_dims(history.action, axis=0)

        observed = to_torch(observed, dtype=torch.float32, device=self.hps.running.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.hps.running.device)
        action = to_torch(action, dtype=torch.long, device=self.hps.running.device)

        new_obs = Batch()
        for k, v in obs.items():
            new_obs[k] = np.expand_dims(v, axis=0)
        obs = to_torch(new_obs, device=self.hps.running.device)
        obs.hist = Batch(observed=observed, mask=mask, action=action)

        return obs

    def _terminal_reward(self, inputs, metrics):
        if self.hps.agent.terminal_reward_type == 'value':
            rew = to_numpy(self.agent.tsk_policy.critic(inputs))[0]
        elif self.hps.agent.terminal_reward_type == 'entropy':
            rew = - to_numpy(self.agent.tsk_policy.actor(inputs).entropy())[0]
        elif self.hps.agent.terminal_reward_type == 'hybrid':
            rew1 = to_numpy(self.agent.tsk_policy.critic(inputs))[0]
            rew2 = - to_numpy(self.agent.tsk_policy.actor(inputs).entropy())[0]
            rew = rew1 + rew2
            metrics['tsk_value_reward'] = float(rew1)
            metrics['tsk_entropy_reward'] = float(rew2)
        else:
            raise NotImplementedError()

        return rew

    @torch.no_grad()
    def rollout(self, env, rand_afa, rand_tsk):
        metrics = defaultdict(float)
        afa_batches = []
        tsk_batches = []

        state, done = env.reset(), False
        tsk_traj = []
        history = History(env.observation_space.shape, self.hps.agent.max_history_length)
        while not done:
            # afa
            if rand_afa:
                obs = self._random_acquisition(env, state)
            else:
                afa_env = AcquireEnv(env, state, self.hps.environment.cost)
                obs, terminate = afa_env.reset(), False
                afa_traj = []
                while not terminate:
                    inputs = self._prepare_inputs(obs, history.get())
                    afa_res = self.agent.afa_policy(inputs)
                    obs_next, reward, terminate, info = afa_env.step(to_numpy(afa_res.act)[0])
                    if terminate and self.hps.agent.terminal_reward_weight > 0:
                        term_rew = self._terminal_reward(inputs, metrics) # final afa action does not change obs
                        reward += term_rew * self.hps.agent.terminal_reward_weight
                        metrics['afa_term_reward'] += term_rew
                    afa_data = Batch(
                        full=state, obs=obs, hist=history.get(), act=afa_res.act[0], rew=reward, done=terminate, 
                        policy=Batch(logp=afa_res.policy.logp[0], vpred=afa_res.policy.vpred[0])
                    )
                    afa_traj.append(afa_data)
                    metrics['episode_reward'] += reward
                    metrics['episode_length'] += 1
                    metrics['num_afa_actions'] += 1
                    metrics['num_acquisitions'] += 0 if terminate else 1
                    obs = obs_next
                afa_batches.append(afa_traj)
            # tsk
            tsk_data = Batch(full=state, obs=obs, hist=history.get())
            if rand_tsk:
                act = env.action_space.sample()
                tsk_data.update(act=act)
            else:
                inputs = self._prepare_inputs(obs, history.get())
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
            history.append(obs.observed, obs.mask, act)
            state = next_state
        tsk_batches.append(tsk_traj)

        metrics['num_acquisitions_per_action'] = metrics['num_acquisitions'] / metrics['num_tsk_actions']
        metrics['average_term_reward'] = metrics['afa_term_reward'] / metrics['num_tsk_actions']

        return afa_batches, tsk_batches, metrics

    @torch.no_grad()
    def _process_traj(self, traj):
        batch = Batch.stack(traj)
        if not hasattr(batch, 'policy'): return batch
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
    def collect(self, env, rand_afa, rand_tsk):
        afa_batches = []
        tsk_batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.train_env_num):
            afa_batch, tsk_batch, metric = self.rollout(env, rand_afa, rand_tsk)
            afa_batches.extend([self._process_traj(traj) for traj in afa_batch])
            tsk_batches.extend([self._process_traj(traj) for traj in tsk_batch])
            for k, v in metric.items():
                metrics[k].append(v)
            
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        afa_batches = Batch.cat(afa_batches)
        tsk_batches = Batch.cat(tsk_batches)

        return afa_batches, tsk_batches, avg_metrics

    def train(self):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed)
        self.agent.setup_optimizer()
        self.agent.set_training_status(afa=True, tsk=True)
        writer = SummaryWriter(f'{self.hps.running.exp_dir}/summary')

        reward_history = []
        best_reward = -np.inf

        # stage1: train tsk_policy  rand_afa=True  rand_tsk=False
        logging.info('=====Stage 1=====')
        self.agent.set_update_status(afa=False, tsk=True)
        for step in range(self.hps.running.stage1_iterations):
            afa_batch, tsk_batch, metrics = self.collect(env, rand_afa=True, rand_tsk=False)
            for k, v in metrics.items():
                writer.add_scalar(f'stage1_collect/{k}', v, step)
            losses = self.agent.learn(afa_batch, tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage1_losses/{k}', v, step)

            # validation
            if step % self.hps.running.validation_freq == 0:
                logging.info(f'Step: {step}')
                metrics = self.valid(rand_afa=True, rand_tsk=False)
                for k, v in metrics.items():
                    writer.add_scalar(f'stage1_valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save('stage1_best', with_optim=True)
        
        # save last
        self.agent.save('stage1_last', with_optim=True)

        # stage2: joint training
        logging.info('=====Stage 2=====')
        self.agent.set_update_status(afa=True, tsk=True)
        for step in range(self.hps.running.stage2_iterations):
            afa_batch, tsk_batch, metrics = self.collect(env, rand_afa=False, rand_tsk=False)
            for k, v in metrics.items():
                writer.add_scalar(f'stage2_collect/{k}', v, step)
            losses = self.agent.learn(afa_batch, tsk_batch)
            for k, v in losses.items():
                writer.add_scalar(f'stage2_losses/{k}', v, step)
            
            # validation
            if step % self.hps.running.validation_freq == 0:
                logging.info(f'Step: {step}')
                metrics = self.valid(rand_afa=False, rand_tsk=False)
                for k, v in metrics.items():
                    writer.add_scalar(f'stage2_valid/{k}', v, step)
                # save
                if metrics['task_reward'] >= best_reward:
                    best_reward = metrics['task_reward']
                    self.agent.save()
                # plot
                reward_history.append(metrics['task_reward'])
                plot_dict(f'{self.hps.running.exp_dir}/reward.png', {'reward': reward_history})

    def valid(self, rand_afa, rand_tsk):
        env = get_environment(self.hps.environment)
        env.seed(self.hps.running.seed+1)
        self.agent.set_training_status(afa=False, tsk=False)

        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_valid_episodes):
            _, _, metric = self.rollout(env, rand_afa, rand_tsk)
            for k, v in metric.items():
                metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n{json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(afa=True, tsk=True)

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
        self.agent.set_training_status(afa=False, tsk=False)

        afa_batches = []
        tsk_batches = []
        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_test_episodes):
            afa_batch, tsk_batch, metric = self.rollout(env, False, False)
            for k, v in metric.items():
                metrics[k].append(v)
            afa_batches.append([self._record_traj(traj) for traj in afa_batch])
            tsk_batches.append([self._record_traj(traj) for traj in tsk_batch])
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nTest:\n{json.dumps(avg_metrics, indent=4)}')

        with gzip.open(f'{self.hps.running.exp_dir}/trajectory.pgz', 'wb') as f:
            pickle.dump({'afa_batches': afa_batches, 'tsk_batches': tsk_batches}, f)

        return avg_metrics
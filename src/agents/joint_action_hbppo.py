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
from src.environments.joint_action_wrapper import JointAFAWrapper
from src.policies.joint_action_hbppo import PolicyBuilder
from src.utils.visualizer import plot_dict

class Agent(object):
    def __init__(self, hps):
        self.hps = hps

    def _setup(self, env):
        policy_builder = PolicyBuilder(env, self.hps.policy)

        self.policy = policy_builder.build_policy()
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
        observed = np.concatenate([batch.hist.observed, np.expand_dims(batch.obs.observed, axis=1)], axis=1)
        mask = np.concatenate([batch.hist.mask, np.expand_dims(batch.obs.mask, axis=1)], axis=1)
        action = batch.hist.action

        observed = to_torch(observed, dtype=torch.float32, device=self.hps.running.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.hps.running.device)
        action = to_torch(action, dtype=torch.long, device=self.hps.running.device)

        obs = to_torch(batch.obs, device=self.hps.running.device)
        obs.hist = Batch(observed=observed, mask=mask, action=action)
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

    @torch.no_grad()
    def rollout(self, env):
        metrics = defaultdict(float)
        batches = []

        obs, done = env.reset(), False
        history = History(env.observation_space['observed'].shape, self.hps.agent.max_history_length)
        while not done:
            state = env.state
            obs = Batch(obs)
            inputs = self._prepare_inputs(obs, history.get())
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
            metrics['num_tsk_actions'] += 1  
            history.append(obs.observed, obs.mask, info['tsk_action'])
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
        env = JointAFAWrapper(env, self.hps.environment.cost)
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
        env = JointAFAWrapper(env, self.hps.environment.cost)
        env.seed(self.hps.running.seed+1)
        self.agent.set_training_status(False)

        metrics = defaultdict(list)
        for _ in range(self.hps.running.num_valid_episodes):
            _, metric = self.rollout(env)
            for k, v in metric.items():
                metrics[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        logging.info(f'\nValidation:\n{json.dumps(avg_metrics, indent=4)}')

        self.agent.set_training_status(True)

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
        env = JointAFAWrapper(env, self.hps.environment.cost)
        env.seed(self.hps.running.seed+2)
        self.agent.load()
        self.agent.set_training_status(False)

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
            pickle.dump({"joint_batches": batches}, f)

        return avg_metrics
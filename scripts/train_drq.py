import copy
import gym
import hydra
import math
import numpy as np
import os
import pickle as pkl
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TransformObservation
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

# Hack to import stuff from drq submodule.
sys.path.append('./drq')
import meta_drq as drq
import meta_utils as utils
from meta_logger import Logger
from meta_replay_buffer import ReplayBuffer
from meta_video import VideoRecorder

torch.backends.cudnn.benchmark = True


def make_env(cfg):
    """Helper function to create environment"""
    assert cfg.env in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()
    env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[cfg.env]
    env = env_constructor(view=cfg.view, render_img_size=cfg.image_size)
    env = utils.FrameStack(env, k=cfg.frame_stack)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def update_log_dir(log_dir: str):
    """
    Updates the log directory by appending a new experiment number.
    Example: 'logs/bc/' -> 'logs/bc/1' if 'logs/bc/0' exists and
             is the only experiment done thus far.
    Also adds a trailing '/' if it's missing.

    Args:
        log_dir: The old log directory string.

    Returns:
        The new log directory string.
    """

    # Add '/' to the end of log_dir if it's missing.
    if log_dir[-1] != '/':
        log_dir += '/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sub_dirs = [sub_dir for sub_dir in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, sub_dir)) and sub_dir.isdigit()]
    if len(sub_dirs) == 0:
        return log_dir + '0/'
    sub_dirs_as_ints = [int(s) for s in sub_dirs]
    last_sub_dir = max(sub_dirs_as_ints)
    return log_dir + str(last_sub_dir + 1) + '/'


class Workspace(object):
    def __init__(self, cfg):
        log_dir = os.getcwd() # e.g., 'logs/drq/'
        self.log_dir = update_log_dir(log_dir) # increment experiment number

        print(f'workspace: {self.log_dir}')

        self.cfg = cfg

        self.logger = Logger(self.log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          cfg.frame_stack)

        self.video_recorder = VideoRecorder(
            self.log_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        num_success = 0
        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            slack = 20 # num steps to run after success (to make video endings look better)
            succeeded = False
            while not done and not self.env.max_path_length_reached():
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                if episode < 5: # only record 5 videos total
                    self.video_recorder.record(self.env)
                # Terminate upon success, with some slack steps so that videos don't end too abruptly.
                if info['success']:
                    succeeded = True
                    slack -= 1
                    if slack <= 0:
                        done = True
                else: # Only accumulate episodes when there is no success yet.
                    episode_reward += reward
                    episode_step += 1

            if succeeded:
                num_success += 1
            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.gif')
        average_episode_reward /= self.cfg.num_eval_episodes
        success_rate = num_success / self.cfg.num_eval_episodes
        self.logger.eval_log('eval/episode_reward', average_episode_reward,
                        self.step, log_frequency=1)
        self.logger.eval_log('eval/success_rate', success_rate,
                        self.step, log_frequency=1)
        self.logger.dump(self.step)

    def save_checkpoint(self):
        self.agent.save_checkpoint(self.log_dir, self.step)

    def run(self):
        start_eval_logs = False # whether we should start logging eval metrics
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            # Evaluate the agent periodically and save checkpoint.
            if self.step % self.cfg.eval_frequency == 0 and start_eval_logs:
                self.logger.eval_log('eval/episode', episode, self.step)
                self.evaluate()
                self.save_checkpoint()
            # Check whether the training episode is complete or the max number of
            # episode steps is reached.
            if done or self.env.max_path_length_reached():
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                    # We need `start_eval_logs` to start eval logs a bit later because
                    # if we start eval logs immediately upon starting training, then we
                    # get an error where the _dump_to_csv() function in the logger hasn't
                    # yet saved the CSV headers seen during training.
                    if self.step > self.cfg.num_seed_steps:
                        start_eval_logs = True

                if self.step > 0:
                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                if self.step > 0:
                    self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)
            done = info['success']

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

@hydra.main(config_path='../drq/meta_config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

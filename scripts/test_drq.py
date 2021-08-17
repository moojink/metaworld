import copy
import datetime
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
from tqdm import tqdm

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
    env = env_constructor(train=False, view=cfg.view, random_init_obj_pos=True, render_img_size=cfg.image_size)
    env = utils.FrameStack(cfg.view, env, k=cfg.frame_stack)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env

class Workspace(object):
    def __init__(self, cfg):
        self.checkpoint_dir = os.getcwd() # e.g., 'logs/drq/0/'
        self.checkpoint_step = cfg.checkpoint_step
        print(f'checkpoint dir: {self.checkpoint_dir}')
        self.cfg = cfg
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
        
        test_vids_save_dir = self.checkpoint_dir + '/test/'
        if not os.path.exists(test_vids_save_dir):
            os.makedirs(test_vids_save_dir)
        self.video_recorder = VideoRecorder(
            cfg.view,
            test_vids_save_dir if cfg.save_video else None
        )

    def evaluate(self):
        average_episode_reward = 0
        average_num_steps_until_success = 0
        num_success = 0
        self.video_recorder.init(enabled=True)
        for episode in tqdm(range(self.cfg.num_eval_episodes)):
            # Reinitialize the environment because just resetting it doesn't change the
            # location(s) of the target object(s).
            self.env = make_env(self.cfg)

            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            slack = 10 # num steps to record after success (to make video endings less abrupt)
            succeeded = False
            record_num_steps_until_success = True
            while not done and not self.env.max_path_length_reached():
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                # Only record 3 episodes. Record at most X steps per episode. Stop recording
                # after success (with some slack steps so that the vids don't end too abruptly).
                if episode < 3 and episode_step < 100 and slack > 0:
                    self.video_recorder.record(self.env)
                if info['success']:
                    succeeded = True
                if succeeded:
                    slack -= 1
                    if record_num_steps_until_success:
                        average_num_steps_until_success += episode_step # only want to do this once per successful episode
                        record_num_steps_until_success = False

                episode_reward += reward
                episode_step += 1

            if succeeded:
                num_success += 1
            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.cfg.env}-view_{self.cfg.view}.gif')
        average_episode_reward /= self.cfg.num_eval_episodes
        if num_success > 0:
            average_num_steps_until_success /= num_success
        else:
            average_num_steps_until_success = self.env.max_path_length
        success_rate = num_success / self.cfg.num_eval_episodes
        f = open("drq-tests.txt", "a")
        f.write('Success rate: {}\n\tAvg episode reward: {}\n\tAvg num steps until success: {}\n'.format(success_rate, average_episode_reward, average_num_steps_until_success))
        print('Success rate: {}\n\tAvg episode reward: {}\n\tAvg num steps until success: {}\n\n'.format(success_rate, average_episode_reward, average_num_steps_until_success))
        f.close()

    def run(self):
        # Load checkpoint.
        self.agent.load_checkpoint(self.checkpoint_dir, self.checkpoint_step)
        self.evaluate()


@hydra.main(config_path='../drq/meta_config_test.yaml', strict=True)
def main(cfg):
    # Open log file and append start message.
    f = open("drq-tests.txt", "a")
    f.write('\n=====================================\n')
    f.write(str(datetime.datetime.now()) + '\n\n')
    f.write('Environment: {}\n'.format(cfg.env))
    f.write('View {}\n'.format(cfg.view))
    f.write('Checkpoint dir: {}\n'.format(cfg.checkpoint_dir))
    f.write('Checkpoint step: {}\n\n'.format(cfg.checkpoint_step))
    f.close()

    start_time = time.time()

    workspace = Workspace(cfg)
    workspace.run()

    end_time = time.time()

    print('Elapsed time (sec): {}'.format(end_time - start_time))
    f = open("drq-tests.txt", "a")
    f.write('Elapsed time (sec): {}'.format(end_time - start_time) + '\n')
    f.close()


if __name__ == '__main__':
    main()



# Source (before modifications): https://github.com/facebookresearch/drqv2/blob/main/train.py
import hydra
import numpy as np
import os
import sys
import torch
import warnings
from dm_env import specs
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor
from pathlib import Path

# Hack to import stuff from drqv2 submodule.
sys.path.append('./drqv2')
import meta_drqv2 as drqv2
import meta_utils as utils
from meta_logger import Logger
from meta_replay_buffer import ReplayBufferStorage, make_replay_loader
from meta_video import TrainVideoRecorder, VideoRecorder

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
torch.backends.cudnn.benchmark = True


def make_env(cfg, train):
    """Helper function to create environment.
    
    Args:
        cfg: Hydra config.
        train: True if creating training env. False if creating eval env.
    
    Returns:
        env: Environment object.
    """
    assert cfg.env in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()
    env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[cfg.env]
    env = env_constructor(view=cfg.view, train=train, random_init_obj_pos=cfg.random_init_obj_pos, render_img_size=cfg.image_size)
    # Put env through wrappers.
    env = utils.ActionRepeatWrapper(env, cfg.action_repeat, cfg.discount)
    env = utils.FrameStackWrapper(cfg.view, env, k=cfg.frame_stack)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def make_agent(img_obs_spec, action_spec, cfg):
    cfg.img_obs_shape = img_obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


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


class Workspace:
    def __init__(self, cfg):
        log_dir = os.getcwd() # e.g., 'logs/drq/'
        log_dir = update_log_dir(log_dir) # increment experiment number
        self.log_dir = Path(log_dir) # convert to Path object, which is more convenient to use
        print(f'workspace: {self.log_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.metaworld_stacked_img_obs_spec,
                                self.metaworld_action_spec, cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.log_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = make_env(self.cfg, train=True)
        self.eval_env = make_env(self.cfg, train=False)
        # create env specs
        self.metaworld_stacked_img_obs_spec = specs.BoundedArray(
            shape=(3 * self.cfg.frame_stack, self.cfg.image_size, self.cfg.image_size),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='img_obs'
        )
        self.metaworld_stacked_proprio_obs_spec = specs.Array(
            shape=(self.cfg.proprio_obs_shape * self.cfg.frame_stack,),
            dtype=np.float32,
            name='proprio_obs'
        )
        self.metaworld_action_spec = specs.BoundedArray(
            shape=self.train_env.action_space.shape,
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
        # create replay buffer
        data_specs = (self.metaworld_stacked_img_obs_spec,
                      self.metaworld_stacked_proprio_obs_spec,
                      self.metaworld_action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.log_dir / 'replay_buffer')

        self.replay_loader = make_replay_loader(
            self.log_dir / 'replay_buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(self.log_dir if self.cfg.save_video else None, render_size=self.cfg.image_size)
        self.train_video_recorder = TrainVideoRecorder(self.log_dir if self.cfg.save_train_video else None, render_size=self.cfg.image_size)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, num_success, total_num_steps_until_success = 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        self.video_recorder.init(self.eval_env, enabled=True)
        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            done = False
            slack = 10 # num steps to record after success (to make video endings less abrupt)
            succeeded = False
            record_num_steps_until_success = True
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, self.global_step, eval_mode=True)
                obs, reward, done, info = self.eval_env.step(action)
                # Only record 3 episodes. Record at most X steps per episode. Stop recording
                # after success (with some slack steps so that the vids don't end too abruptly).
                if episode < 3 and step < 100 and slack > 0:
                    self.video_recorder.record(self.eval_env)
                if info['success']:
                    succeeded = True
                if succeeded:
                    slack -= 1
                    if record_num_steps_until_success:
                        total_num_steps_until_success += step # only want to do this once per successful episode
                        record_num_steps_until_success = False
                total_reward += reward
                step += 1
                done = done or self.eval_env.max_path_length_reached()
            episode += 1
        if succeeded:
            num_success += 1
        self.video_recorder.save(f'{self.global_frame}.gif')
        if num_success > 0:
            average_num_steps_until_success = total_num_steps_until_success / num_success
        else:
            average_num_steps_until_success = self.eval_env.max_path_length
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('success_rate', num_success / episode)
            log('num_steps_until_success', average_num_steps_until_success)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        save_train_video_every_step = utils.Every(self.cfg.save_train_video_every_frames,
                                                  self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()
        img_obs, proprio_obs = obs
        action = np.zeros(self.metaworld_action_spec.shape) # zero action when resetting env
        reward = 0.0 # zero reward when resetting env
        discount = 1.0 # no discount when resetting env
        done = False
        self.replay_storage.add(img_obs, proprio_obs, action, reward, discount, done)
        metrics = None
        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                if save_train_video_every_step(self.global_step):
                    self.train_video_recorder.save(f'{self.global_frame}.gif')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                obs = self.train_env.reset()
                img_obs, proprio_obs = obs
                action = np.zeros(self.metaworld_action_spec.shape) # zero action when resetting env
                reward = 0.0 # zero reward when resetting env
                discount = 1.0 # no discount when resetting env
                done = False
                self.replay_storage.add(img_obs, proprio_obs, action, reward, discount, done)
                self.train_video_recorder.init(img_obs)
                episode_step = 0
                episode_reward = 0

            # try to evaluate and save snapshot
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                self.save_snapshot()


            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs, reward, done, info = self.train_env.step(action)
            img_obs, proprio_obs = obs
            episode_reward += reward
            done = done or self.train_env.max_path_length_reached()
            self.replay_storage.add(img_obs, proprio_obs, action, reward, self.cfg.discount, done)
            self.train_video_recorder.record(img_obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.log_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.log_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='../drqv2', config_name='meta_config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
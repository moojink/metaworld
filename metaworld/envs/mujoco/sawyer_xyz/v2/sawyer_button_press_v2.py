import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerButtonPressEnvV2(SawyerXYZEnv):
    def __init__(self, view, train, random_init_obj_pos):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)


        self.train = train
        self.train_positions = dict(
            obj_low = (-0.2, 0.85, 0.115),
            obj_high = (0, 0.9, 0.115),
        )
        self.test_positions = dict(
            obj_low = (-0.4, 0.75, 0.115),
            obj_high = (0.2, 0.9, 0.115),
        )
        if self.train:
            obj_low = self.train_positions['obj_low']
            obj_high = self.train_positions['obj_high']
        else:
            obj_low = self.test_positions['obj_low']
            obj_high = self.test_positions['obj_high']



        super().__init__(
            self.model_name,
            view=view,
            hand_low=hand_low,
            hand_high=hand_high,
            random_init_obj_pos=random_init_obj_pos,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.1, 0.875, 0.115], dtype=np.float32),
            'hand_init_pos': np.array([-0.1, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.goal = self.obj_init_pos + np.array([0, -0.12, 0.005])
        self.hand_init_pos = self.init_config['hand_init_pos']
        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high)) # this doesn't matter when we hide the goal

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_button_press.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('btnGeom')

    def _get_pos_objects(self):
        return self.get_body_com('button') + np.array([.0, -.193, .0])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('button')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self, seed=None):
        self._reset_hand()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if seed is not None:
            np.random.seed(seed=seed) # this ensures that every time we reset, we get the same initial obj positions

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()
            while self.should_resample_obj_pos(self.obj_init_pos):
                self.obj_init_pos = self._get_state_rand_vec()

        # self._target_pos = self.obj_init_pos + + np.array([0, -0.12, 0.005])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(0)
        self._target_pos = self._get_site_pos('hole')

        self._obj_to_target_init = abs(
            self._target_pos[1] - self._get_site_pos('buttonStart')[1]
        )

        return self._get_obs()


    def should_resample_obj_pos(self, obj_pos):
        """Returns True when the initial position of the object
        overlaps with the distribution of initial positions used during training.
        This is so that during test time we sample positions outside of the training
        distribution."""
        if self.train:
            return False # only possibly resample during testing
        obj_x, obj_y = obj_pos[0], obj_pos[1]
        obj_low_x, obj_low_y = self.train_positions['obj_low'][0], self.train_positions['obj_low'][1]
        obj_high_x, obj_high_y = self.train_positions['obj_high'][0], self.train_positions['obj_high'][1]
        # Only check x and y because z is the same during train and test.
        return obj_low_x <= obj_x and obj_x <= obj_high_x and obj_low_y <= obj_y and obj_y <= obj_high_y


    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )

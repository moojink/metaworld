import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDrawerCloseEnvV2(SawyerXYZEnv):
    _TARGET_RADIUS = 0.04
    def __init__(self, view, train, random_init_obj_pos):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        self.train = train
        self.train_positions = dict(
            obj_low = (-0.2, 0.9, 0.0),
            obj_high = (0, 0.9, 0.0),
        )
        self.test_positions = dict(
            obj_low = (-0.4, 0.85, 0.0),
            obj_high = (0.2, 0.9, 0.0),
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
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([-0.1, 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([-0.1, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward,
        tcp_to_obj,
        _,
        target_to_obj,
        object_grasped,
        in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS + 0.015),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com('drawer_link') + np.array([.0, -.16, .05])

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def reset_model(self, seed=None):
        self._reset_hand()

        if seed is not None:
            np.random.seed(seed=seed) # this ensures that every time we reset, we get the same initial obj positions

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()
            while self.should_resample_obj_pos(self.obj_init_pos):
                self.obj_init_pos = self._get_state_rand_vec()

        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id(
            'drawer'
        )] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed)
        self._target_pos = self.obj_init_pos + np.array([.0, -.16, .09])
        # Pull drawer out all the way and mark its starting position
        self._set_obj_xyz(-self.maxDist)
        self.obj_init_pos = self._get_pos_objects()

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
        obj = obs[4:7]

        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init-handle_reach_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.

        reward *= 10

        return (reward,
               tcp_to_obj,
               tcp_opened,
               target_to_obj,
               object_grasped,
               in_place)

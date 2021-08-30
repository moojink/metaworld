import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerNutAssemblyEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self, view, train, random_init_obj_pos):
        hand_low = (-0.5, 0.20, 0.05)
        hand_high = (0.5, 1, 0.5)


        self.train = train
        self.train_positions = dict(
            obj_low = (-0.155, 0.4, 0.02),
            obj_high = (-0.095, 0.44, 0.02),
            goal_low = (-0.155, 0.55, 0.1),
            goal_high = (-0.095, 0.6, 0.1),
        )
        self.test_positions = dict(
            obj_low = (-0.3, 0.4, 0.02),
            obj_high = (0.05, 0.44, 0.02),
            goal_low = (-0.3, 0.55, 0.1),
            goal_high = (0.05, 0.7, 0.1),
        )
        if self.train:
            obj_low = self.train_positions['obj_low']
            obj_high = self.train_positions['obj_high']
            goal_low = self.train_positions['goal_low']
            goal_high = self.train_positions['goal_high']
        else:
            obj_low = self.test_positions['obj_low']
            obj_high = self.test_positions['obj_high']
            goal_low = self.test_positions['goal_low']
            goal_high = self.test_positions['goal_high']

        super().__init__(
            self.model_name,
            view=view,
            hand_low=hand_low,
            hand_high=hand_high,
            random_init_obj_pos=random_init_obj_pos,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.125, 0.42, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((-0.125, 0.42, 0.2), dtype=np.float32),
        }
        self.goal = np.array([-0.125, 0.575, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_assembly_peg.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [('pegTop', self._target_pos)]

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('WrenchHandle')

    def _get_pos_objects(self):
        return self.data.site_xpos[self.model.site_name2id('RoundNut-8')]

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('RoundNut')

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('RoundNut')
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()

        if self.random_init:
            obj_pos, goal_pos = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.1 or self.should_resample_obj_pos(obj_pos, goal_pos):
                obj_pos, goal_pos = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = obj_pos
        self._target_pos = goal_pos

        peg_pos = self._target_pos - np.array([0., 0., 0.05])
        self._set_obj_xyz(self.obj_init_pos)
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos

        return self._get_obs()

    def should_resample_obj_pos(self, obj_pos, goal_pos):
        """Returns True when the initial position of either the object or the goal
        overlaps with the distribution of initial positions used during training.
        This is so that during test time we sample positions outside of the training
        distribution."""
        if self.train:
            return False # only possibly resample during testing
        obj_x, obj_y = obj_pos[0], obj_pos[1]
        goal_x, goal_y = goal_pos[0], goal_pos[1]
        obj_low_x, obj_low_y = self.train_positions['obj_low'][0], self.train_positions['obj_low'][1]
        obj_high_x, obj_high_y = self.train_positions['obj_high'][0], self.train_positions['obj_high'][1]
        goal_low_x, goal_low_y = self.train_positions['goal_low'][0], self.train_positions['goal_low'][1]
        goal_high_x, goal_high_y = self.train_positions['goal_high'][0], self.train_positions['goal_high'][1]
        # Return True when there is an overlap for either object, i.e. when either the object
        # or the goal lies inside its training-time bounding box.
        # Only check x and y because z is the same during train and test.
        return (obj_low_x <= obj_x and obj_x <= obj_high_x and obj_low_y <= obj_y and obj_y <= obj_high_y) or \
               (goal_low_x <= goal_x and goal_x <= goal_high_x and goal_low_y <= goal_y and goal_y <= goal_high_y)


    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = aligned and hooked

        # Target height is a 3D funnel centered on the peg.
        # use the success flag to widen the bottleneck once the agent
        # learns to place the wrench on the peg -- no reason to encourage
        # tons of alignment accuracy if task is already solved
        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1., 1., 3.])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * scale),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid='long_tail',
        )

        return in_place, success

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos('RoundNut')
        tcp = self.tcp_center
        wrench_handle = self.data.geom_xpos[self.unwrapped.model.geom_name2id('WrenchHandle')]
        tcp_to_wrench = np.linalg.norm(wrench_handle - tcp)
        tcp_to_wrench_init = np.linalg.norm(wrench_handle - self.init_tcp)
        gripper_near_wrench = reward_utils.tolerance(
            tcp_to_wrench,
            bounds=(0, 0.05),
            margin=tcp_to_wrench_init,
            sigmoid='long_tail',
        )
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = SawyerNutAssemblyEnvV2.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = SawyerNutAssemblyEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions, wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            medium_density=True,
        )
        reward_in_place, success = SawyerNutAssemblyEnvV2._reward_pos(
            wrench_center,
            self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat

        # Reward moving gripper closer to the wrench.
        reward += gripper_near_wrench

        # # Reward moving wrench closer to peg (only when gripper is holding wrench, which
        # # we detect by checking reward_grab, which is between 0 and 1).
        # peg = self._target_pos
        # wrench_init_pos = self.obj_init_pos
        # wrench_to_peg = np.linalg.norm(wrench_center - peg)
        # wrench_to_peg_init = np.linalg.norm(wrench_init_pos - peg)
        # wrench_near_peg = reward_utils.tolerance(
        #     wrench_to_peg,
        #     bounds=(0, 0.1),
        #     margin=wrench_to_peg_init,
        #     sigmoid='long_tail',
        # )
        # if reward_grab > 0.9:
        #     reward += wrench_near_peg * 0.2

        # Override reward on success
        if success:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )

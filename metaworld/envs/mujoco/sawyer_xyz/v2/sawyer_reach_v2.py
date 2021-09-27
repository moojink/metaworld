import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self, view, train, random_init_obj_pos):
        hand_low = (-0.6, 0.2, 0.05)
        hand_high = (0.2, 1, 0.05)

        # I don't know why there is an "object" in this task when it isn't used at all in the reward functin
        # and it seems like the "goal" is the only thing that is relevant. So I'll make the object transparent.

        # obj = (-0.1, 0.6, 0.02)
        # goal_low = (-0.1, 0.8, 0.05)
        # goal_high = (0.1, 0.9, 0.3)

        self.train = train
        # self.train_positions = dict(
        #     # target in front of gripper ('far' from robot base)
        #     goal_low_far = (-0.3, 0.9, 0.05),
        #     goal_high_far = (-0.1, 0.9, 0.05),
        #     # target behind gripper ('near' robot base)
        #     goal_low_near = (-0.3, 0.3, 0.05),
        #     goal_high_near = (-0.1, 0.3, 0.05),
        # )
        self.train_positions = dict(
            # target in front of gripper ('far' from robot base)
            goal_low_far = (-0.2, 0.85, 0.05),
            goal_high_far = (-0.2, 0.85, 0.05),
            # target behind gripper ('near' robot base)
            goal_low_near = (-0.2, 0.35, 0.05),
            goal_high_near = (-0.2, 0.35, 0.05),
        )
        self.test_positions = dict(
            # target in front of gripper ('far' from robot base)
            goal_low_far = (-0.5, 0.9, 0.05),
            goal_high_far = (0.1, 0.9, 0.05),
            # target behind gripper ('near' robot base)
            goal_low_near = (-0.5, 0.3, 0.05),
            goal_high_near = (0.1, 0.3, 0.05),
        )
        if self.train:
            goal_low_far = self.train_positions['goal_low_far']
            goal_high_far = self.train_positions['goal_high_far']
            goal_low_near = self.train_positions['goal_low_near']
            goal_high_near = self.train_positions['goal_high_near']
        else:
            goal_low_far = self.test_positions['goal_low_far']
            goal_high_far = self.test_positions['goal_high_far']
            goal_low_near = self.test_positions['goal_low_near']
            goal_high_near = self.test_positions['goal_high_near']


        super().__init__(
            self.model_name,
            view=view,
            hand_low=hand_low,
            hand_high=hand_high,
            random_init_obj_pos=random_init_obj_pos,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([-0.2, 0.6, 0.05]),
            'hand_init_pos': np.array([-0.2, 0.6, 0.05]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        # self._random_reset_space = Box(
        #     np.hstack((obj_low, goal_low)),
        #     np.hstack((obj_high, goal_high)),
        # )

        self._random_reset_space_far = Box(
            np.array(goal_low_far),
            np.array(goal_high_far),
        )
        self._random_reset_space_near = Box(
            np.array(goal_low_near),
            np.array(goal_high_near),
        )

        # self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_reach_v2.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):

        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= self._TARGET_RADIUS)

        info = {
            'success': success,
            'near_object': reach_dist,
            'grasp_success': 1.,
            'grasp_reward': reach_dist,
            'in_place_reward': in_place,
            'obj_to_target': reach_dist,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self, seed=None):
        self._reset_hand()

        # We don't care about the object, but leaving the code there anyway to not break things.
        # The target is the thing that matters.

        if seed is not None:
            np.random.seed(seed=seed) # this ensures that every time we reset, we get the same initial obj positions

        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            # Flip a coin to initialize the target either in front or behind the gripper.
            if np.random.rand() < 0.5: # 50-50 chance
                self._target_location = 'far'
                self._random_reset_space = self._random_reset_space_far
            else:
                self._target_location = 'near'
                self._random_reset_space = self._random_reset_space_near

            pos_target = self._get_state_rand_vec()
            while self.should_resample_target_pos(pos_target):
                pos_target = self._get_state_rand_vec()

        self._target_pos = pos_target
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos

        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1

        return self._get_obs()

    def should_resample_target_pos(self, target_pos):
        """Returns True when the initial position of the target
        overlaps with the distribution of initial positions used during training.
        This is so that during test time we sample positions outside of the training
        distribution."""
        if self.train:
            return False # only possibly resample during testing
        target_x, target_y = target_pos[0], target_pos[1]
        if self._target_location == 'far':
            target_low_key = 'goal_low_far'
            target_high_key = 'goal_high_far'
        else:
            target_low_key = 'goal_low_near'
            target_high_key = 'goal_high_near'
        target_low_x, target_low_y = self.train_positions[target_low_key][0], self.train_positions[target_low_key][1]
        target_high_x, target_high_y = self.train_positions[target_high_key][0], self.train_positions[target_high_key][1]
        # Only check x and y because z is the same during train and test.
        return target_low_x <= target_x and target_x <= target_high_x and target_low_y <= target_y and target_y <= target_high_y


    def compute_reward(self, actions, obs):
        self._TARGET_RADIUS = 0.06
        tcp = self.tcp_center
        target = self._target_pos

        tcp_to_target = np.linalg.norm(tcp - target)
        tcp_to_target_init = np.linalg.norm(self.init_tcp - target)

        in_place = reward_utils.tolerance(tcp_to_target,
                                    bounds=(0, self._TARGET_RADIUS),
                                    margin=tcp_to_target_init,
                                    sigmoid='long_tail',)
        reward = in_place
        # print("tcp_to_target:", tcp_to_target)
        if tcp_to_target <= self._TARGET_RADIUS:
            reward = 10.

        return [reward, tcp_to_target, in_place]

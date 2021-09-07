import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from scipy.spatial.transform import Rotation


class SawyerPegInsertionSideHardEnvV2(SawyerXYZEnv):
    TARGET_RADIUS = 0.07
    """
    This is a harder version of SawyerPegInsertionSideEnvV2. The target box location is fixed,
    but the peg location is randomly initialized such that it can start in front of or behind
    the gripper with a large enough distance from the gripper such that the peg is not initially
    visible to the wrist-view camera.
    """
    def __init__(self, view, train, random_init_obj_pos):
        hand_init_pos = (0, 0.55, 0.05)
        hand_low = (-0.5, 0.25, 0.045)
        hand_high = (0.5, 1, 0.5)

        self.train = train
        self.train_positions = dict(
            # peg in front of gripper ('far' from robot base)
            obj_low_far = (-0.025, 0.85, 0.02),
            obj_high_far = (0.025, 0.85, 0.02),
            # peg behind gripper ('near' robot base)
            obj_low_near = (-0.025, 0.35, 0.02),
            obj_high_near = (0.025, 0.35, 0.02),
            # box
            goal_low = (-0.525, 0.5, -0.001),
            goal_high = (-0.525, 0.6, 0.001),
        )
        self.test_positions = dict(
            # peg in front of gripper ('far' from robot base)
            obj_low_far = (-0.3, 0.85, 0.02),
            obj_high_far = (0.1, 0.85, 0.02),
            # peg behind gripper ('near' robot base)
            obj_low_near = (-0.3, 0.35, 0.02),
            obj_high_near = (0.1, 0.35, 0.02),
            # box
            goal_low = (-0.525, 0.3, -0.001),
            goal_high = (-0.525, 0.75, 0.001),
        )
        if self.train:
            obj_low_far = self.train_positions['obj_low_far']
            obj_high_far = self.train_positions['obj_high_far']
            obj_low_near = self.train_positions['obj_low_near']
            obj_high_near = self.train_positions['obj_high_near']
            goal_low = self.train_positions['goal_low']
            goal_high = self.train_positions['goal_high']
        else:
            obj_low_far = self.test_positions['obj_low_far']
            obj_high_far = self.test_positions['obj_high_far']
            obj_low_near = self.test_positions['obj_low_near']
            obj_high_near = self.test_positions['obj_high_near']
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
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([-0.3, 0.6, 0.0])

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.hand_init_pos = np.array(hand_init_pos)

        self._random_reset_space = None # will be overridden upon reset
        self._random_reset_space_far = Box(
            np.hstack((obj_low_far, goal_low)),
            np.hstack((obj_high_far, goal_high)),
        )
        self._random_reset_space_near = Box(
            np.hstack((obj_low_near, goal_low)),
            np.hstack((obj_high_near, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([.03, .0, .13]),
            np.array(goal_high) + np.array([.03, .0, .13])
        )

        self.object_grasped = False

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_peg_insertion_side.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward, collision_box_front, ip_orig= (
            self.compute_reward(action, obs))
        grasp_success = float(tcp_to_obj < 0.02 and (tcp_open > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]))
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self._get_site_pos('pegGrasp')

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_site_xmat('pegGrasp')).as_quat()

    def reset_model(self):
        self.object_grasped = False
        self._reset_hand()

        pos_peg = self.obj_init_pos
        pos_box = self.goal
        if self.random_init:
            # Flip a coin to initialize the peg either in front or behind the gripper.
            if np.random.rand() < 0.5: # 50-50 chance
                self._peg_location = 'far'
                self._random_reset_space = self._random_reset_space_far
            else:
                self._peg_location = 'near'
                self._random_reset_space = self._random_reset_space_near

            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1 or self.should_resample_obj_pos(pos_peg, pos_box):
                pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = pos_peg
        self._set_obj_xyz(self.obj_init_pos)
        self.peg_head_pos_init = self._get_site_pos('pegHead')

        self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box
        self._target_pos = pos_box + np.array([.03, .0, .13])

        return self._get_obs()

    def should_resample_obj_pos(self, pos_peg, pos_box):
        """Returns True when the initial position of either the peg or the box
        overlaps with the distribution of initial positions used during training.
        This is so that during test time we sample positions outside of the training
        distribution."""
        if self.train:
            return False # only possibly resample during testing
        peg_x, peg_y = pos_peg[0], pos_peg[1]
        box_x, box_y = pos_box[0], pos_box[1]
        if self._peg_location == 'far':
            obj_low_key = 'obj_low_far'
            obj_high_key = 'obj_high_far'
        else:
            obj_low_key = 'obj_low_near'
            obj_high_key = 'obj_high_near'
        peg_low_x, peg_low_y = self.train_positions[obj_low_key][0], self.train_positions[obj_low_key][1]
        peg_high_x, peg_high_y = self.train_positions[obj_high_key][0], self.train_positions[obj_high_key][1]
        box_low_x, box_low_y = self.train_positions['goal_low'][0], self.train_positions['goal_low'][1]
        box_high_x, box_high_y = self.train_positions['goal_high'][0], self.train_positions['goal_high'][1]
        # Return True when there is an overlap for either object, i.e. when either the peg
        # or the box lies inside its training-time bounding box.
        # Only check x and y because z is the same during train and test.
        return (peg_low_x <= peg_x and peg_x <= peg_high_x and peg_low_y <= peg_y and peg_y <= peg_high_y) or \
               (box_low_x <= box_x and box_x <= box_high_x and box_low_y <= box_y and box_y <= box_high_y)


    def compute_reward(self, action, obs):
        tcp = self.tcp_center
        obj = obs[4:7]
        obj_head = self._get_site_pos('pegHead')
        tcp_opened = obs[3]
        target = self._target_pos
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        near_peg = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.1),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        scale = np.array([1., 2., 2.])
        #  force agent to pick up object then insert
        obj_to_target = np.linalg.norm((obj_head - target) * scale)

        in_place_margin = np.linalg.norm((self.peg_head_pos_init - target) * scale)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, self.TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
        ip_orig = in_place
        brc_col_box_1 = self._get_site_pos('bottom_right_corner_collision_box_1')
        tlc_col_box_1 = self._get_site_pos('top_left_corner_collision_box_1')

        brc_col_box_2 = self._get_site_pos('bottom_right_corner_collision_box_2')
        tlc_col_box_2 = self._get_site_pos('top_left_corner_collision_box_2')
        collision_box_bottom_1 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                                   one=tlc_col_box_1,
                                                                   zero=brc_col_box_1)
        collision_box_bottom_2 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                                   one=tlc_col_box_2,
                                                                   zero=brc_col_box_2)
        collision_boxes = reward_utils.hamacher_product(collision_box_bottom_2,
                                                        collision_box_bottom_1)
        in_place = reward_utils.hamacher_product(in_place,
                                                 collision_boxes)

        pad_success_margin = 0.03
        object_reach_radius=0.01
        x_z_margin = 0.005
        obj_radius = 0.0075

        object_grasped = self._gripper_caging_reward(action,
                                                     obj,
                                                     object_reach_radius=object_reach_radius,
                                                     obj_radius=obj_radius,
                                                     pad_success_thresh=pad_success_margin,
                                                     xz_thresh=x_z_margin,
                                                     high_density=True)
        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            object_grasped = 1.

        if object_grasped == 1.:
            self.object_grasped = True
        else:
            self.object_grasped = False

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5 * in_place

        if obj_to_target <= 0.07:
            reward = 10.

        # Reward moving closer to the peg.
        reward += near_peg * 0.01

        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place, collision_boxes, ip_orig]


import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerNutAssemblyEnv(SawyerXYZEnv):

    def __init__(self, random_init=True):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-0.1, 0.75, 0.1)
        goal_high = (0.1, 0.85, 0.1)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.random_init = random_init

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.6, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low,)),
            np.hstack((self.hand_high, obj_high,)),
        )
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_assembly_peg.xml')

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward , _, reachDist, pickRew, _, placingDist, _, success = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1
        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': placingDist, 'success': float(success)}
        info['goal'] = self.goal
        return ob, reward, False, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        graspPos =  self.data.get_geom_xpos('RoundNut-8')
        flat_obs = np.concatenate((hand, graspPos))
        return np.concatenate([flat_obs,])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        graspPos =  self.data.get_geom_xpos('RoundNut-8')
        objPos = self.get_body_com('RoundNut')
        flat_obs = np.concatenate((hand, graspPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('pegTop')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('RoundNut-8')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
            self.obj_init_pos = goal_pos[:3]
            self._state_goal = goal_pos[-3:]

        peg_pos = self._state_goal - np.array([0., 0., 0.05])
        self._set_obj_xyz(self.obj_init_pos)
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._state_goal
        self._set_goal_marker(self._state_goal)
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def compute_reward(self, actions, obs):
        obs = obs['state_observation']

        graspPos = obs[3:6]
        objPos = self.get_body_com('RoundNut')

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(graspPos - fingerCOM)

        placingDist = np.linalg.norm(objPos[:2] - placingGoal[:2])
        placingDistFinal = np.abs(objPos[-1] - self.objHeight)

        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.04:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance) and reachDist < 0.03:
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)

        def placeCompletionCriteria():
            if abs(objPos[0] - placingGoal[0]) < 0.03 and \
                abs(objPos[1] - placingGoal[1]) < 0.03:
                return True
            else:
                return False

        if placeCompletionCriteria():
            self.placeCompleted = True
        else:
            self.placeCompleted = False

        def orig_pickReward():
            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not(objDropped())):
                return hScale*heightTarget
            elif (reachDist < 0.04) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeRewardMove():
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            if self.placeCompleted:
                c4 = 2000; c5 = 0.003; c6 = 0.0003
                placeRew += 2000*(heightTarget - placingDistFinal) + c4*(np.exp(-(placingDistFinal**2)/c5) + np.exp(-(placingDistFinal**2)/c6))
            placeRew = max(placeRew,0)
            cond = self.placeCompleted or (self.pickCompleted and (reachDist < 0.04) and not(objDropped()))
            if cond:
                return [placeRew, placingDist, placingDistFinal]
            else:
                return [0, placingDist, placingDistFinal]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist, placingDistFinal = placeRewardMove()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        success = (abs(objPos[0] - placingGoal[0]) < 0.03 and abs(objPos[1] - placingGoal[1]) < 0.03 and placingDistFinal <= 0.04)
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, placingDistFinal, success]


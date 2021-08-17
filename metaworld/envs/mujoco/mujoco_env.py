import abc
import warnings

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self._view = None # will be set later

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(self, offscreen=False, camera_name="corner2", resolution=(640, 480)):
        # If `camera_name` is "configured_view", we render with whatever view that
        # self._view is configured to (set during initialization of the environment).
        assert_string = ("camera_name should be one of ",
                "corner3, corner, corner2, topview, gripperPOV, behindGripper, view_1, view_3", "configured_view")
        assert camera_name in {"corner3", "corner", "corner2", 
            "topview", "gripperPOV", "behindGripper", "view_1", "view_3", "configured_view"}, assert_string
        if not offscreen:
            self._get_viewer('human').render()
        else:
            # Use the configured camera view if applicable.
            if camera_name == "configured_view":
                if self._view is None:
                    print("Warning: self._view is None, but env.render() was called with arg "\
                            "camera_name='configured_view'. You might want to call env.set_camera_view() "\
                            "immediately after initializing the environment in order to configure the "\
                            "view. Defaulting to an arbitrary view (e.g., camera_name='view_3').")
                if self._view == 1:
                    camera_name = "view_1"
                else:
                    camera_name = "view_3"
            img = self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name
            )
            return np.transpose(img, (2, 0, 1)) # want (3, img_size, img_size), not (img_size, img_size, 3)

    def render_overwrite(self, offscreen, overwrite_view, resolution):
        """Used to overwrite self.view, e.g. when we want to render both views."""
        return self.render(offscreen, overwrite_view, resolution)


    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def _get_offscreen_viewer(self, sim):
        # This function is incomplete/garbage. Just leaving it for reference in case we ever need to mess with the views here.
        if self.viewer is None:
            self.viewer = mujoco_py.MjRenderContextOffscreen(sim, -1)
            self.viewer.cam.azimuth =  205
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 2.3
            self.viewer.cam.lookat[0] = 1.1
            self.viewer.cam.lookat[1] = 1.1 
            self.viewer.cam.lookat[2] = -0.1
        return self.viewer


    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def set_camera_view(self, view):
        assert view in (1, 3, 'both')
        self._view = view

    def set_render_img_size(self, render_img_size):
        self._render_img_size = render_img_size

    def max_path_length_reached(self) -> bool:
        return getattr(self, 'curr_path_length', 0) >= self.max_path_length

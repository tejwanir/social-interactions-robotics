import copy
import mujoco_py
import numpy as np
import os

from fetch_env import rotations, robot_env, mujoco_utils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pnp_three_objects.xml')

# Initial positions of the robot and the objects
INIT_POS = {
    # initial x,y,z of the robot
    'robot0:slide0': 0.405,
    'robot0:slide1': 0.48,
    'robot0:slide2': 0.0,
    'gripper_offset': [0., 0., 0.15],
    # initial pos and rot (in quat.) of the object
    'objectR:joint': [1.3, 0.75, 0.4, 1., 0., 0., 0.],
    'objectG:joint': [1.3, 0.75, 0.4, 1., 0., 0., 0.],
    'objectB:joint': [1.3, 0.75, 0.4, 1., 0., 0., 0.],
    #'objectG:joint': [1.3, 0.92, 0.4, 1., 0., 0., 0.],
    #'objectB:joint': [1.45, 0.7, 0.4, 1., 0., 0., 0.],
}

POS_HALF_RANGE = {
    'objectR:joint': [0.08, 0.16, 0.],
    'objectG:joint': [0.08, 0.16, 0.],
    'objectB:joint': [0.08, 0.16, 0.],
    'gripper_offset': [0.2, 0.3, 0.]
}


def in_tray(pos):
    """Check if a position is in the tray.
    """
    tray_pos = [0.67+0.8, 0.23+0.75, 0.401]
    tray_half_range = [0.08, 0.1, 0.025]  # z is the height of the block
    for i, p in enumerate(tray_pos):
        if pos[i] < p - tray_half_range[i] or \
                pos[i] > p + tray_half_range[i]:
            return False
    return True

def obj_overlap(pos_dict, target_name):
    for obj_name, obj_pos in pos_dict.items():
        if obj_name not in POS_HALF_RANGE.keys() or obj_name == target_name:
            continue
        d = np.linalg.norm(np.array(obj_pos[:3]) - np.array(pos_dict[target_name][:3]))
        if d < 0.08:
            return True
    return False

def get_obj_names():
    return ['objectR', 'objectG', 'objectB']


class FetchThreeObjEnv(robot_env.RobotEnv):
    """Superclass for all MuJoCo Fetch environments.
    """

    def __init__(self, initial_qpos, n_substeps, timelimit=50):
        """Initializes a new Fetch environment.

        Args:
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            n_substeps (int): number of substeps the simulation runs on every call to step
            timelimit (int): number of time steps before the agent fails due to timeout
        """
        model_path = MODEL_XML_PATH

        self.objects = get_obj_names()
        self.timelimit = timelimit
        self.current_time = 0

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
        self.initial_gripper_xpos = None

        super(FetchThreeObjEnv, self).__init__(
            model_path=model_path,
            initial_qpos=initial_qpos, n_substeps=n_substeps, n_actions=4)

        for k, v in self.sim.model._sensor_name2id.items():
            self._touch_sensor_id_site_id.append((v, self.sim.model._site_name2id[k]))
            self._touch_sensor_id.append(v)

    # Fetch methods
    # ----------------------------

    @staticmethod
    def sample_env():
        new_pos = {}
        init_copy = copy.deepcopy(INIT_POS)
        for k,v in init_copy.items():
            new_pos[k] = v
            if k not in POS_HALF_RANGE.keys():
                continue
            r = POS_HALF_RANGE[k]
            # sample object positions
            if 'joint' in k:
                new_pos[k] = v
                for i in range(3):
                    new_pos[k][i] = new_pos[k][i] + np.random.uniform(-r[i], r[i])
                # sample the position again if the object is in the tray
                while in_tray(new_pos[k][:3]) or obj_overlap(new_pos, k):
                    new_pos[k] = v
                    for i in range(3):
                        new_pos[k][i] = new_pos[k][i] + np.random.uniform(-r[i], r[i])
            elif 'gripper_offset' in k:
                new_pos[k] = v
                for i in range(3):
                    new_pos[k][i] = new_pos[k][i]  # TODO: randomize gripper position
            else:
                continue
        return new_pos


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.current_time += 1

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        mujoco_utils.ctrl_set_action(self.sim, action)
        mujoco_utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # get gripper positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(self.sim)
        
        # get object features
        object_features = []
        for obj_name in self.objects:
            object_pos = self.sim.data.get_site_xpos(obj_name)
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_name))
            # velocities
            object_velp = self.sim.data.get_site_xvelp(obj_name) * dt
            object_velr = self.sim.data.get_site_xvelr(obj_name) * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            object_features.extend([object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
                                    object_velp.ravel(), object_velr.ravel()])
        
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        features = [grip_pos, gripper_state, grip_velp, gripper_vel]
        features.extend(object_features)
        obs = np.concatenate(features)

        out = { 'observation': obs.copy() }
        # TODO: may add other features to the observation dict
        return out

    def _get_other_obs(self):
        # RGB-D
        im0, _ = self.sim.render(width=500, height=500, camera_name='external_camera_0', depth=True)
        im1, d1 = self.sim.render(width=500, height=500, camera_name='external_camera_1', depth=True)
        im2, d2 = self.sim.render(width=500, height=500, camera_name='external_camera_2', depth=True)

        # touch sensor data
        contact_data = self.sim.data.sensordata[self._touch_sensor_id]

        # removing the red target
        # TODO: Move this to render callback
        name = 'target0'
        target_geom_ids = [self.sim.model.geom_name2id(name)
                           for name in self.sim.model.geom_names if name.startswith('target')]
        target_mat_ids = [self.sim.model.geom_matid[gid] for gid in target_geom_ids]
        target_site_ids = [self.sim.model.site_name2id(name)
                           for name in self.sim.model.site_names if name.startswith('target')]

        self.sim.model.mat_rgba[target_mat_ids, -1] = 0
        self.sim.model.geom_rgba[target_geom_ids, -1] = 0
        self.sim.model.site_rgba[target_site_ids, -1] = 0

        return {
            'image0': im0[::-1, :, :].copy(),
            'image1': im1[::-1, :, :].copy(),
            'image2': im2[::-1, :, :].copy(),
            'depth1': d1[::-1].copy(),
            'depth2': d2[::-1].copy(),
            'contact': contact_data.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        # site is the the place to place the item
        #sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        #site_id = self.sim.model.site_name2id('target0')
        #self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        # Running simulation one step forward (do it after updating positions)
        self.sim.forward()

    def _reset_sim(self):
        # sample a new initial configuration
        initial_qpos = FetchThreeObjEnv.sample_env()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        # reset state
        #self.sim.set_state(self.initial_state)
        #self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            if 'gripper_offset' in name:  # skip the gripper offset, set it later
                continue
            self.sim.data.set_joint_qpos(name, value)
        mujoco_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        if self.initial_gripper_xpos is None:
            self.init_gripper = np.array([-0.498, 0.005, -0.431]) + \
                    self.sim.data.get_site_xpos('robot0:grip')
            gripper_target = np.array([-0.498 + initial_qpos['gripper_offset'][0], \
                    0.005 + initial_qpos['gripper_offset'][1], \
                    -0.431 + initial_qpos['gripper_offset'][2]]) + \
                    self.sim.data.get_site_xpos('robot0:grip')
            gripper_rotation = np.array([1., 0., 1., 0.])
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            for _ in range(10):
                self.sim.step()

            # Extract information for sampling goals.
            self.init_gripper_pos = gripper_target
            self.init_gripper_rot = gripper_rotation
            self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
            self.height_offset = self.sim.data.get_site_xpos('objectR')[2]
        else:  # just move to the stored gripper position if already have one
            self.sim.data.set_mocap_pos('robot0:mocap', self.init_gripper_pos)
            self.sim.data.set_mocap_quat('robot0:mocap', self.init_gripper_rot)
            for _ in range(10):
                self.sim.step()

    def _is_success(self, obs):
        for object_name in self.objects:
            object_qpos = self.sim.data.get_joint_qpos(object_name+':joint')
            object_xpos = object_qpos[:3]
            if in_tray(object_xpos):
                return True
        return False

    def _check_done(self, obs):
        if self.current_time >= self.timelimit:
            return True
        return False

    def _compute_reward(self, obs, info):
        if info['is_success']:
            return True
        reward = -np.inf
        tray_pos = np.array([0.68+0.8, 0.25+0.75, 0.401])
        for object_name in self.objects:
            object_qpos = self.sim.data.get_joint_qpos(object_name+':joint')
            object_xpos = object_qpos[:3]
            r = -np.linalg.norm(object_xpos - tray_pos)
            if r > reward:
                reward = r
        return reward

    def render(self, mode='human', width=500, height=500):
        return super(FetchThreeObjEnv, self).render(mode, width, height)


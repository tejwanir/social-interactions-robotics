import copy
import mujoco_py
import numpy as np
import os
import json
from collections import OrderedDict

from fetch_env import rotations, robot_env, mujoco_utils
from fetch_env.env_creator import load_configs_from_json

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'pnp_three_objects.xml')
MODEL_XML_PATH = os.path.join('full_env', 'env.xml')

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


def in_tray(obj_pos, obj_size, tray_pos, tray_size):
    """Check if a position is in the tray.
    """
    tray_half_range = [tray_size[0], tray_size[1], obj_size[2]]  # z is the height of the block
    for i, p in enumerate(tray_pos):
        if obj_pos[i] < p - tray_half_range[i] or \
                obj_pos[i] > p + tray_half_range[i]:
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

    def __init__(self, initial_qpos, robot_configs, table_configs, object_configs, tray_configs, n_substeps, initial_robot_pos=None, timelimit=50):
        """Initializes a new Fetch environment.

        Args:
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            n_substeps (int): number of substeps the simulation runs on every call to step
            initial_robot_pos (list): a list of tuples specifying the positions of the robots
            timelimit (int): number of time steps before the agent fails due to timeout
        """
        self.robot_configs = robot_configs
        self.table_configs = table_configs
        self.object_configs = object_configs
        self.tray_configs = tray_configs

        self.robot_dict = self.create_dict(robot_configs)
        self.table_dict = self.create_dict(table_configs)
        self.object_dict = self.create_dict(object_configs)
        self.tray_dict = self.create_dict(tray_configs)

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
    
    def create_dict(self, configs):
        d = {}
        for config in configs:
            d[config.name] = config
        return d

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
    
    @staticmethod
    def sample_env_from_json(path_to_json):
        robot_configs, table_configs, object_configs, tray_configs = load_configs_from_json(path_to_json)    
        new_pos = {}

        for r_config in robot_configs:
            r_name = r_config.name
            if (r_config.init_pos != None):
                new_pos[r_name + ':slide0'] = r_config.init_pos[0]
                new_pos[r_name + ':slide1'] = r_config.init_pos[1]
                new_pos[r_name + ':slide2'] = r_config.init_pos[2]
            else:
                table_config = list(filter(lambda x: x.name == r_config.table_name, table_configs))[0]
                pos, angle = FetchThreeObjEnv.get_position_relative_to_table(
                    r_config.side,
                    r_config.offset,
                    r_config.distance,
                    table_config.pos,
                    table_config.size
                )
                new_pos[r_name + ':slide0'] = pos[0]
                new_pos[r_name + ':slide1'] = pos[1]
                new_pos[r_name + ':slide2'] = pos[2]
                new_pos[r_name + ':bottom_hinge'] = angle

            new_pos[r_name + ':gripper_offset'] = [0., 0., 0.15] # TODO: Hardcoded

        for o_config in object_configs:
            o_name = o_config.name
            if (o_config.pos != None):
                new_pos[o_name + ':joint'] = o_config.pos + o_config.quat
            elif (o_config.table_name != None):
                pass # TODO: Implement this
            else:
                pos = [0., 0., 0.]
                quat = [1., 0., 0., 0.]
                pos[0] = np.random.uniform(o_config.x_range[0], o_config.x_range[1])
                pos[1] = np.random.uniform(o_config.y_range[0], o_config.y_range[1])
                pos[2] = np.random.uniform(o_config.z_range[0], o_config.z_range[1])
                new_pos['joint:' + o_name] = pos + quat
        
        for t_config in tray_configs:
            t_name = t_config.name
            if (t_config.pos != None):
                new_pos[t_name + ':joint'] = t_config.pos + t_config.quat
            else:
                table_config = list(filter(lambda x: x.name == r_config.table_name, table_configs))[0]
                pos = FetchThreeObjEnv.get_position_on_table(
                    t_config.table_pos,
                    t_config.size,
                    table_config.pos,
                    table_config.size
                )
                quat = t_config.quat
                new_pos[t_name + ':joint'] = pos + quat

        return new_pos, robot_configs, table_configs, object_configs, tray_configs
    
    @staticmethod
    def get_position_on_table(tray_pos, tray_size, table_pos, table_size):
        tray_pos = np.array(tray_pos)
        tray_size = np.array(tray_size)
        table_pos = np.array(table_pos)
        table_size = np.array(table_size)
        
        # This ensures the tray isn't hanging off the edge
        inner_size = table_size - np.array([tray_size[0], tray_size[1], 0.])
        mult_factor = np.array([tray_pos[0], tray_pos[1], 1])
        
        new_pos = mult_factor * inner_size + table_pos + np.array([0., 0., 0.001])
        return list(new_pos)

    @staticmethod
    def get_position_relative_to_table(side, offset, distance, table_pos=[0., 0.], table_size=[0., 0.]):
        '''
        side (int, int): the side of the table the robot should be on (x, z). Can be 
                         (0,1), (0,-1), (1,0), or (-1,0)
        offset (float): the offset from the center of side
        distance (float): the distance from the table
        table_pos (np.array): the position of the table
        table_size (np.array): the half sizes of the table
        '''

        center = None
        displacement = None
        angle = None
        if (side[0] > 0):
            center = np.array([table_size[0] + table_pos[0], table_pos[1], 0]) 
            displacement = np.array([distance, offset, 0])
            angle = np.pi
        elif (side[0] < 0):
            center = np.array([-table_size[0] + table_pos[0], table_pos[1], 0]) 
            displacement = np.array([-distance, offset, 0])
            angle = 0.0
        elif (side[1] > 0):
            center = np.array([table_pos[0], table_size[1] + table_pos[1], 0])
            displacement = np.array([offset, distance, 0])
            angle = 1.5 * np.pi
        else:
            center = np.array([table_pos[0], -table_size[1] + table_pos[1], 0])
            displacement = np.array([offset, -distance, 0])
            angle = 0.5 * np.pi

        new_pos = center + displacement
        new_pos[2] = 0 # TODO: Hardcoded
        return new_pos, angle
        
    # RobotEnv methods
    # ----------------------------
    def _get_table(self, table_name):
        table_id = self.sim.model.body_name2id(table_name)
        table_geom_id = self.sim.model.geom_name2id(table_name)
        
        table_pos = self.sim.data.body_xpos[table_id]
        table_size = self.sim.model.geom_size[table_geom_id]

        return table_pos, table_size
    
    def _get_tray(self, tray_name):
        tray_id = self.sim.model.body_name2id(tray_name)
        tray_geom_id = self.sim.model.geom_name2id(tray_name)
        
        tray_pos = self.sim.data.body_xpos[tray_id]
        tray_size = self.sim.model.geom_size[tray_geom_id]

        return tray_pos, tray_size
    
    def _get_obj(self, obj_name):
        obj_id = self.sim.model.body_name2id(obj_name)
        obj_geom_id = self.sim.model.geom_name2id(obj_name)
        
        obj_pos = self.sim.data.body_xpos[obj_id]
        obj_size = self.sim.model.geom_size[obj_geom_id]

        return obj_pos, obj_size

    def _get_floor_level(self):
        # TODO: Hardcoded
        return 0.

    def _step_callback(self):
        self.current_time += 1
    
    # TODO: Specify actions as a dictionary instead of array
    def _set_action(self, actions):
        assert actions.shape == (len(self.robot_configs), 4)
        
        actions = actions.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = actions[:, :3], actions[:, 3]
        
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = np.array(len(self.robot_configs) * [[1., 0., 1., 0.]])  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (len(self.robot_configs), 2)
        actions = np.concatenate((pos_ctrl, rot_ctrl, gripper_ctrl), axis=1)
        
        # Apply action to simulation.
        mujoco_utils.ctrl_set_action(self.sim, actions.T)
        mujoco_utils.mocap_set_action(self.sim, actions.T)

    def _get_obs(self):
        # get gripper positions
        out = {}
        for config in self.robot_configs:
            r_name = config.name
            grip_pos = self.sim.data.get_site_xpos(r_name + ':grip')

            dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            grip_velp = self.sim.data.get_site_xvelp(r_name + ':grip') * dt
            robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(self.sim)
        
            # get object features
            object_features = []
            for config in self.object_configs:
                obj_name = config.name
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

            out[r_name + '_observation'] = obs.copy()
        
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
        name0 = self.robot_configs[0].name
        body_id = self.sim.model.body_name2id(name0 + ':gripper_link')
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

    def _reset_sim(self): # TODO: Implement this
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
            for config in self.robot_configs:
                r_name = config.name
                self.init_gripper = np.array([-0.498, 0.005, -0.431]) + \
                        self.sim.data.get_site_xpos(r_name + ':grip')
                gripper_target = np.array([-0.498 + initial_qpos[r_name + ':gripper_offset'][0], \
                        0.005 + initial_qpos[r_name + ':gripper_offset'][1], \
                        -0.431 + initial_qpos[r_name + ':gripper_offset'][2]]) + \
                        self.sim.data.get_site_xpos(r_name + ':grip')
                gripper_rotation = np.array([1., 0., 1., 0.])
                self.sim.data.set_mocap_pos(r_name + ':mocap', gripper_target)
                self.sim.data.set_mocap_quat(r_name + ':mocap', gripper_rotation)
            
            for _ in range(10):
                self.sim.step()

            # Extract information for sampling goals.
            '''
            self.init_gripper_pos = gripper_target
            self.init_gripper_rot = gripper_rotation
            self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
            self.height_offset = self.sim.data.get_site_xpos('Small_Square')[2] #TODO: Hardcoded
            '''
        else:  # just move to the stored gripper position if already have one #TODO: Implement this
            self.sim.data.set_mocap_pos('robot0:mocap', self.init_gripper_pos)
            self.sim.data.set_mocap_quat('robot0:mocap', self.init_gripper_rot)
            for _ in range(10):
                self.sim.step()

    def _is_success(self, obs):
        for config in self.object_configs:
            # NOTE: The joint position is ever so slightly different and idk why
            # object_qpos = self.sim.data.get_joint_qpos(config.name+':joint')
            # object_xpos = object_qpos[:3]

            obj_pos, obj_size = self._get_obj(config.name)
            tray_pos, tray_size = self._get_tray(config.target)
            if in_tray(obj_pos, obj_size, tray_pos, tray_size):
                return True
        return False

    def _check_done(self, obs):
        if self.current_time >= self.timelimit:
            return True
        return False

    def _compute_reward(self, obs, info):
        if info['is_success']:
            return True
        reward_dict = OrderedDict()
        
        # First find the "move_object" robot and calculate their rewards
        for config in self.robot_configs:
            if config.task != 'move_object':
                continue

            reward = 0.
            for obj_name in config.target_objects:
                # NOTE: The joint position is ever so slightly different and idk why
                # object_qpos = self.sim.data.get_joint_qpos(config.name+':joint')
                # object_xpos = object_qpos[:3]
                
                obj_pos, _ = self._get_obj(obj_name)
                tray_pos, _ = self._get_tray(self.object_dict[obj_name].target)
                r = -np.linalg.norm(obj_pos - tray_pos)
                if r > -np.inf:
                    reward += r
            reward_dict[config.name] = reward
        
        # Now calculate the rewards of all the robots trying to help and hinder
        for config in self.robot_configs:
            if config.task == 'move_object':
                continue
            
            # You can only help or hinder robots with the "move_object" task
            reward_dict[config.name] = 0
            for target_name in config.target_robots:
                if config.task == 'help':
                    reward_dict[config.name] += reward_dict[target_name]
                elif config.task == 'hinder':
                    reward_dict[config.name] += -reward_dict[target_name]

        return reward_dict

    def render(self, mode='human', width=500, height=500):
        return super(FetchThreeObjEnv, self).render(mode, width, height)


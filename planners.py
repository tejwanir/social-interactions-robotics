import numpy as np

class BasePlanner():
    
    def __init__(self, name, robot_configs, object_configs, tray_configs):
        self.name = name
        self.robot_configs = robot_configs
        self.object_configs = object_configs
        self.tray_configs = tray_configs

        self.config = [config for config in robot_configs if config.name == name][0]
    
    def convert_obs(self, obs):
        # First get the robot obs
        obs_dict = {}
        r_obs = {}

        r_obs['pos']  = obs[0:3]
        r_obs['state'] = obs[3:5]
        r_obs['velp'] = obs[5:8]
        r_obs['vel'] = obs[8:10]
        obs_dict[self.name] = r_obs
        
        # Get all of the object observations
        obs = obs[10:]
        for i, config in enumerate(self.object_configs):
            o_obs = {}
            o_name = config.name

            o_obs['pos'] = obs[i*15:i*15+3]
            o_obs['rel_pos'] = obs[i*15+3:i*15+6]
            o_obs['rot'] = obs[i*15+6:i*15+9]
            o_obs['velp'] = obs[i*15+9:i*15+12]
            o_obs['velr'] = obs[i*15+12:i*15+15]
            obs_dict[o_name] = o_obs
        
        # Get all of the tray observations
        obs = obs[len(self.object_configs)*15:]
        for i, config in enumerate(self.tray_configs):
            t_obs = {}
            t_name = config.name

            t_obs['pos'] = obs[i*3:i*3+3]
            obs_dict[t_name] = t_obs
        
        # Get all of the other robot observations
        # These observations are different than the itself because they are more limited ie less info
        obs = obs[len(self.tray_configs)*3:]
        for i, config in enumerate(self.robot_configs):
            or_name = config.name
            if or_name == self.name:
                i -= 1
                continue
     
            or_obs = {}
            or_obs['pos'] = obs[i*6:i*6+3]
            or_obs['velp'] = obs[i*6+3:i*6+6]
            obs_dict[or_name] = or_obs

        return obs_dict

class SimplePlanner(BasePlanner):
    
    def __init__(self, name, robot_config, object_configs, tray_configs):
        super().__init__(name, robot_config, object_configs, tray_configs)

        self.target = self.config.target_objects[0]
        self.target_tray = [config for config in self.object_configs if config.name == self.target][0].target

        self.state = 0
        self.close_count = 0
        self.up_count = 0

    def move(self, obs):
        robot_obs = obs[self.name + '_observation']
        obs_dict = self.convert_obs(robot_obs)

        if self.state == 0: # Go towards the cube
            obj_pos = obs_dict[self.target]['pos']
            obj_pos[2] += 0.02
            grip_pos = obs_dict[self.name]['pos']
            
            rel_pos = obj_pos - grip_pos
            abs_rel_pos = abs(rel_pos)
            if abs_rel_pos[0] < 0.01 and abs_rel_pos[1] < 0.01 and abs_rel_pos[2] < 0.01:
                self.state = 1
                self.close_count = 0
            return np.array(list(obj_pos - grip_pos) + [1.])
        elif self.state == 1: # Close the gripper
            self.close_count += 1
            if self.close_count == 15:
                self.state = 2
                self.up_count = 0
            return np.array([0., 0., 0., -0.1])
        elif self.state == 2: # Lift straight up
            self.up_count += 1
            if self.up_count == 10:
                self.state = 3
            return np.array([0., 0., 0.1, -0.1])
        elif self.state == 3: # Move towards the tray
            for config in self.tray_configs:
                if config.name == self.target_tray:
                    tray_pos = obs_dict[config.name]['pos']
                    tray_pos[2] += 0.1
            grip_pos = obs_dict[self.name]['pos']
            
            rel_pos = tray_pos - grip_pos
            abs_rel_pos = abs(rel_pos)
            if abs_rel_pos[0] < 0.01 and abs_rel_pos[1] < 0.01 and abs_rel_pos[2] < 0.01:
                self.state = 4
            return np.array(list(tray_pos - grip_pos) + [-0.1])
        else: # Release the object
            return np.array([0., 0., 0., 0.1])

class HinderPlanner(BasePlanner):
    
    def __init__(self, name, robot_config, object_configs, tray_configs):
        super().__init__(name, robot_config, object_configs, tray_configs)
        
        self.state = 0
        self.close_count = 0
        self.up_count = 0
        self.move_count = 0

    def move(self, obs):
        robot_obs = obs[self.name + '_observation']
        obs_dict = self.convert_obs(robot_obs)

        if self.state == 0: # Go towards the cube
            obj_pos = obs_dict[self.object_configs[0].name]['pos']
            obj_pos[2] += 0.02
            grip_pos = obs_dict[self.name]['pos']
            
            rel_pos = obj_pos - grip_pos
            abs_rel_pos = abs(rel_pos)
            if abs_rel_pos[0] < 0.01 and abs_rel_pos[1] < 0.01 and abs_rel_pos[2] < 0.01:
                self.state = 1
                self.close_count = 0
            return np.array(list(obj_pos - grip_pos) + [1.])
        elif self.state == 1: # Close the gripper
            self.close_count += 1
            if self.close_count == 15:
                self.state = 2
                self.up_count = 0
            return np.array([0., 0., 0., -0.1])
        elif self.state == 2: # Lift straight up
            self.up_count += 1
            if self.up_count == 10:
                self.state = 3
                self.move_count = 0
            return np.array([0., 0., 0.1, -0.1])
        elif self.state == 3: # Move it away from the other arm
            self.move_count += 1
            if self.move_count == 20:
                self.state = 4
            return np.array([-0.1, -0.1, 0.1, -0.1])
        else: # Release the object
            return np.array([0., 0., 0., 0.1])

class HelpPlanner(BasePlanner):
    
    def __init__(self, name, robot_config, object_configs, tray_configs):
        super().__init__(name, robot_config, object_configs, tray_configs)
        
        self.close_count = 0
        self.up_count = 0
        self.move_count = 0

    def move(self, obs):
        robot_obs = obs[self.name + '_observation']
        obs_dict = self.convert_obs(robot_obs)
        
        # Get the position of the two robot's grippers
        o_grip_pos = obs_dict[self.config.target_robots[0]]['pos']
        grip_pos = obs_dict[self.name]['pos']
        rel_pos = o_grip_pos - grip_pos
        
        squared_distance = sum(np.square(rel_pos))
        if squared_distance > 0.1:
            return np.array(list(rel_pos) + [1.])
        else:
            return np.array([0., 0., 0., 1.])
            

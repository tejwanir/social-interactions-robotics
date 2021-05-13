import numpy as np

class SimplePlanner():
    
    def __init__(self, name, object_configs):
        # 0 is going towards cube, 1 is moving towards object, and 2 is done
        self.state = 0
        self.name = name
        self.object_configs = object_configs

    def move(self, obs):
        robot_obs = obs[self.name + '_observation']
        obs_dict = self.convert_obs(robot_obs)

        obj_pos = obs_dict['Small_Square']['pos']
        # obj_pos[2] += 0.02
        grip_pos = obs_dict[self.name]['pos']

        print(obj_pos)
        
        if self.state == 0:
            return np.array(list(obj_pos - grip_pos) + [0])
    
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
        

        return obs_dict

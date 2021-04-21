import numpy as np
import os
import json

BASE_XML = '''
<?xml version="1.0" encoding="utf-8"?>
<mujoco>
	<compiler angle="radian" coordinate="local" meshdir="../stls/fetch" texturedir="../textures"></compiler>
	<option timestep="0.002">
		<flag warmstart="enable"></flag>
	</option>

	<include file="shared.xml"></include>

	<worldbody>
		<geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom>
		<body name="floor0" pos="0.8 0.75 0">
			<site name="target0" pos="0.67 0.23 0.401" size="0.08 0.1 0.001" rgba="1 0 0 1" type="box"></site>
		</body>

        {body}

		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
	</worldbody>

	<actuator>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
	</actuator>
</mujoco>
'''

TABLE_XML = '''
<body pos="{pos}" name="{name}">
	<geom size="{size}" type="box" mass="2000" material="table_mat"></geom>
</body>
'''

OBJECT_XML = '''
<body name="{name}" pos="{pos}">
	<joint name="{name}:joint" type="free" damping="0.01"></joint>
	<geom size="{size}" type="box" condim="3" name="{name}" material="blue_mat" mass="2"></geom>
	<site name="{name}" pos="0 0 0" size="0.02 0.02 0.02" rgba="{color} 1" type="sphere"></site>
</body>
'''

class RobotConfig():
    
    def __init__(
        self,
        name,
        init_pos=None,
        table_name=None,
        side=None,
        offset=None,
        distance=None,
    ):
        assert init_pos != None or table_name != None
        
        self.name = name
        self.init_pos = init_pos
        self.table_name = table_name
        self.side = side
        self.offset = offset
        self.distance = distance
 
class TableConfig():
    
    def __init__(
        self,
        name,
        pos,
        size
    ):
        self.name = name
        self.pos = pos
        self.size = size

class ObjectConfig():
    
    def __init__(
        self,
        name,
        size,
        color,
        pos=None,
        quat=None,
        x_range=(0,0),
        y_range=(0,0),
        z_range=(0,0),
        table_name=None
    ):
        self.name = name
        self.size = size
        self.color = color
        self.pos = pos
        self.quat = quat
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

def load_configs_from_json(path_to_json):
    json_text = open(path_to_json, 'r').read()
    env_config = json.loads(json_text)
    
    assert "Robots" in env_config.keys()
    assert "Tables" in env_config.keys()
    assert "Objects" in env_config.keys()

    robot_configs = []
    for r_kwargs in env_config["Robots"]:
        robot_configs.append(RobotConfig(**r_kwargs))
    
    table_configs = []
    for t_kwargs in env_config["Tables"]:
        table_configs.append(TableConfig(**t_kwargs))

    object_configs = []
    for o_kwargs in env_config["Objects"]:
        object_configs.append(ObjectConfig(**o_kwargs))

    return robot_configs, table_configs, object_configs
    
# A class that enables you to specify the number of robots, tables, and objects in an environment.
# It takes these specs and compiles them into an xml file that can be fed into Mujoco
class EnvCreator():
    
    def __init__(self, path_to_json):
        robot_configs, table_configs, object_configs = load_configs_from_json(path_to_json)    
        self.robot_configs = robot_configs
        self.table_configs = table_configs
        self.object_configs = object_configs
    
        self.create_xml()

    def create_xml(self, path_to_xml=None):
        body_xml = ''
        for i, config in enumerate(self.robot_configs):   
            robot_xml = open(os.path.join('fetch_env', 'assets', 'fetch', 'robot.xml'), 'r').read()
            
            # Remove the first and last lines to get rid of the <mujoco> tags
            robot_xml = '\n'.join(robot_xml.split('\n')[1:-1])
            
            robot_xml = robot_xml.replace('robot0', config.name)
            body_xml += robot_xml + '\n\n'

        for i, config in enumerate(self.table_configs):
            pos_str = str(config.pos[0]) + ' ' + str(config.pos[1]) + ' ' + str(config.pos[2])
            size_str = str(config.size[0]) + ' ' + str(config.size[1]) + ' ' + str(config.size[2])
            table_xml = TABLE_XML.format(name='table' + str(i), pos=pos_str, size=size_str)
            body_xml += table_xml + '\n\n'
        
        for i, config in enumerate(self.object_configs):
            pos_str = str(config.pos[0]) + ' ' + str(config.pos[1]) + ' ' + str(config.pos[2])
            size_str = str(config.size[0]) + ' ' + str(config.size[1]) + ' ' + str(config.size[2])
            color_str = str(config.color[0]) + ' ' + str(config.color[1]) + ' ' + str(config.color[2])
            object_xml = OBJECT_XML.format(name=config.name, pos=pos_str, size=size_str, color=color_str)
            body_xml += object_xml + '\n\n'

        full_xml = BASE_XML.format(body=body_xml)
        if path_to_xml:
            f = open(path_to_xml, "w")
            f.write(full_xml)
            f.close()
        else:
            return full_xml

if __name__  == '__main__':
    env_creator = EnvCreator(os.path.join('fetch_env', 'test_env.json'))
    env_creator.create_xml(os.path.join('fetch_env', 'env.xml'))

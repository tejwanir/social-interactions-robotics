import mujoco_py
import numpy as np
import os

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
        init_x=None,
        init_y=None,
        init_z=None,
        side=None,
        offset=None,
        distance=None
    ):
        assert init_x != None or side != None

        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.side = side
        self.offset = offset
        self.distance = distance
 
class TableConfig():
    
    def __init__(
        self,
        x,
        y,
        z,
        size_x,
        size_y,
        size_z
    ):
        self.x = x
        self.y = y
        self.z = z
        self.size_x = x
        self.size_y = y
        self.size_z = z

class ObjectConfig():
    
    def __init__(
        self,
        name,
        size_x,
        size_y,
        size_z,
        color_r,
        color_g,
        color_b,
        x=0,
        y=0,
        z=0,
        randomize=False
    ):
        self.name = name
        self.size_x = x
        self.size_y = y
        self.size_z = z
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.x = x
        self.y = y
        self.z = z
        self.randomize = randomize

# A class that enables you to specify the number of robots, tables, and objects in an environment.
# It takes these specs and compiles them into an xml file that can be fed into Mujoco
class EnvConigs():
    
    def __init__(self, robot_configs, table_configs, object_configs):
       self.robot_configs = robot_configs
       self.table_configs = table_configs
       self.object_configs = object_configs

       self.create_xml()

    def create_xml():
        body_xml = ''
        for i, config in enumerate(self.robot_configs):   
            robot_xml = open(os.path.join('fetch', 'robot.xml'), 'r').read()
            robot_xml = robot_xml.replace('robot0', 'robot' + str(i))
            body_xml += robot_xml + '\n\n'

        for i, config in enumerate(self.table_configs):
            table_xml = TABLE_XML.format(name='table' + str(i))
            pos_str = str(config.x) + ' ' + str(config.y) + ' ' + str(config.z)
            size_str = str(config.size_x) + ' ' + str(config.size_y) + ' ' + str(config.size_z)
            table_xml = table_xml.format(pos=pos_str, size=size_str)
            body_xml += table_xml + '\n\n'
        
        for i, config in enumerate(self.object_configs):
            object_xml = OBJECT_XML.format(name=config.name)
            pos_str = str(config.x) + ' ' + str(config.y) + ' ' + str(config.z)
            size_str = str(config.size_x) + ' ' + str(config.size_y) + ' ' + str(config.size_z)
            color_str = str(config.color_r) + ' ' + str(config.color_g) + ' ' + str(config.color_b)
            object_xml = object_xml.format(pos=pos_str, size=size_str, color=color_str)
            body_xml += object_xml + '\n\n'

        return BASE_XML.format(body=body_xml)

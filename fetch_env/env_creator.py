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

SHARED_XML = '''
<mujoco>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.44 0.85 0.56" rgb2="0.46 0.87 0.58" width="32" height="32"></texture>
        <texture name="texture_block" file="block.png" gridsize="3 4" gridlayout=".U..LFRB.D.."></texture>

        <material name="floor_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.2 0.2 0.2 1"></material>
        <material name="table_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.93 0.93 0.93 1"></material>
        <material name="block_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.2 0.2 0.2 1"></material>
        <material name="red_mat" specular="0" shininess="0.5" reflectance="0" rgba="1 0 0 1"></material>
        <material name="green_mat" specular="0" shininess="0.5" reflectance="0" rgba="0 1 0 1"></material>
        <material name="blue_mat" specular="0" shininess="0.5" reflectance="0" rgba="0 0 1 1"></material>
        <material name="puck_mat" specular="0" shininess="0.5" reflectance="0" rgba="0.2 0.2 0.2 1"></material>
{robot_shared_1}
    </asset>

    <equality>
        <weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.02 1"></weld>
    </equality>

    <contact>
        <exclude body1="robot0:r_gripper_finger_link" body2="robot0:l_gripper_finger_link"></exclude>
        <exclude body1="robot0:torso_lift_link" body2="robot0:torso_fixed_link"></exclude>
        <exclude body1="robot0:torso_lift_link" body2="robot0:shoulder_pan_link"></exclude>
    </contact>
    
{robot_shared_2}

    <sensor>
        <touch name="robot0:l_gripper_touch" site="robot0:l_gripper_touch"></touch>
        <touch name="robot0:r_gripper_touch" site="robot0:r_gripper_touch"></touch>
    </sensor>
</mujoco>
'''

ROBOT_SHARED_XML_1 = '''
        <material name="{name}:geomMat" shininess="0.03" specular="0.4"></material>
        <material name="{name}:gripper_finger_mat" shininess="0.03" specular="0.4" reflectance="0"></material>
        <material name="{name}:gripper_mat" shininess="0.03" specular="0.4" reflectance="0"></material>
        <material name="{name}:arm_mat" shininess="0.03" specular="0.4" reflectance="0"></material>
        <material name="{name}:head_mat" shininess="0.03" specular="0.4" reflectance="0"></material>
        <material name="{name}:torso_mat" shininess="0.03" specular="0.4" reflectance="0"></material>
        <material name="{name}:base_mat" shininess="0.03" specular="0.4" reflectance="0"></material>

        <mesh file="base_link_collision.stl" name="{name}:base_link"></mesh>
        <mesh file="bellows_link_collision.stl" name="{name}:bellows_link"></mesh>
        <mesh file="elbow_flex_link_collision.stl" name="{name}:elbow_flex_link"></mesh>
        <mesh file="estop_link.stl" name="{name}:estop_link"></mesh>
        <mesh file="forearm_roll_link_collision.stl" name="{name}:forearm_roll_link"></mesh>
        <mesh file="gripper_link.stl" name="{name}:gripper_link"></mesh>
        <mesh file="head_pan_link_collision.stl" name="{name}:head_pan_link"></mesh>
        <mesh file="head_tilt_link_collision.stl" name="{name}:head_tilt_link"></mesh>
        <mesh file="l_wheel_link_collision.stl" name="{name}:l_wheel_link"></mesh>
        <mesh file="laser_link.stl" name="{name}:laser_link"></mesh>
        <mesh file="r_wheel_link_collision.stl" name="{name}:r_wheel_link"></mesh>
        <mesh file="torso_lift_link_collision.stl" name="{name}:torso_lift_link"></mesh>
        <mesh file="shoulder_pan_link_collision.stl" name="{name}:shoulder_pan_link"></mesh>
        <mesh file="shoulder_lift_link_collision.stl" name="{name}:shoulder_lift_link"></mesh>
        <mesh file="upperarm_roll_link_collision.stl" name="{name}:upperarm_roll_link"></mesh>
        <mesh file="wrist_flex_link_collision.stl" name="{name}:wrist_flex_link"></mesh>
        <mesh file="wrist_roll_link_collision.stl" name="{name}:wrist_roll_link"></mesh>
        <mesh file="torso_fixed_link.stl" name="{name}:torso_fixed_link"></mesh>
'''

ROBOT_SHARED_XML_2 = '''
    <default>
        <default class="{name}:fetch">
            <geom margin="0.001" material="{name}:geomMat" rgba="1 1 1 1" solimp="0.99 0.99 0.01" solref="0.01 1" type="mesh" user="0"></geom>
            <joint armature="1" damping="50" frictionloss="0" stiffness="0"></joint>

            <default class="{name}:fetchGripper">
                <geom condim="4" margin="0.001" type="box" user="0" rgba="0.356 0.361 0.376 1.0"></geom>
                <joint armature="100" damping="1000" limited="true" solimplimit="0.99 0.999 0.01" solreflimit="0.01 1" type="slide"></joint>
            </default>

            <default class="{name}:grey">
                <geom rgba="0.356 0.361 0.376 1.0"></geom>
            </default>
            <default class="{name}:blue">
                <geom rgba="0.086 0.506 0.767 1.0"></geom>
            </default>
        </default>
    </default>
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
        # Create the shared xml that defines all of the classes
        shared_1 = ''
        shared_2 = ''
        for i, config in enumerate(self.robot_configs):
            shared_1 += ROBOT_SHARED_XML_1.format(name=config.name)
            shared_2 += ROBOT_SHARED_XML_2.format(name=config.name)
        shared_xml = SHARED_XML.format(robot_shared_1=shared_1, robot_shared_2=shared_2)

        # Create the env xml
        body_xml = ''
        for i, config in enumerate(self.robot_configs):   
            robot_xml = open(os.path.join('fetch_env', 'assets', 'full_env', 'robot.xml'), 'r').read()
            
            # Remove the first and last lines to get rid of the <mujoco> tags
            robot_xml = '\n'.join(robot_xml.rstrip().split('\n')[1:-1])
            
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
            f = open(os.path.join(path_to_xml, 'shared.xml'), "w")
            f.write(shared_xml)
            f.close()
            
            f = open(os.path.join(path_to_xml, 'env.xml'), "w")
            f.write(full_xml)
            f.close()
        else:
            return shared_xml, full_xml

if __name__  == '__main__':
    env_creator = EnvCreator(os.path.join('fetch_env', 'test_env.json'))
    env_creator.create_xml(os.path.join('fetch_env', 'assets', 'full_env'))

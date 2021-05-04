import numpy as np
import os

from fetch_env.fetch_three_obj_env import FetchThreeObjEnv
from fetch_env.env_creator import EnvCreator
from mujoco_py import GlfwContext  # Requires GlfwContext so GLEW won't complain


def main(headless):
    if headless:
        GlfwContext(offscreen=True)
    
    env_creator = EnvCreator(os.path.join('fetch_env', 'test_env.json'))
    env_creator.create_xml(os.path.join('fetch_env', 'assets', 'full_env'))
    
    init_qpos, robot_configs, table_configs, object_configs, tray_configs = FetchThreeObjEnv.sample_env_from_json(
        os.path.join("fetch_env", "test_env.json")
    )
    print(init_qpos)
    env = FetchThreeObjEnv(
        robot_configs=robot_configs,
        table_configs=table_configs,
        object_configs=object_configs,
        tray_configs=tray_configs,
        n_substeps=20,
        initial_qpos=init_qpos
   )
    while True:
        action = np.random.uniform(low=-1., high=1., size=(len(env.robot_configs), env.action_space.shape[0]))
        # print(' action:', action)
        next_obs, reward, done, info = env.step(action)
        print('  done:', done, ', reward:', reward)
        done = False
        env.sim.step()
        if headless:
            env.render('rgb_array')
        else:
            env.render('human')
        '''
        if done:
            env.reset()
            break
        '''

if __name__ == '__main__':
    main(headless=False)


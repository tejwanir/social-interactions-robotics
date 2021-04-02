import numpy as np

from fetch_env.fetch_three_obj_env import FetchThreeObjEnv
from mujoco_py import GlfwContext  # Requires GlfwContext so GLEW won't complain


def main(headless):
    if headless:
        GlfwContext(offscreen=True)

    init_qpos = FetchThreeObjEnv.sample_env()
    env = FetchThreeObjEnv(n_substeps=20,
                           initial_qpos=init_qpos)
    while True:
        action = np.random.uniform(low=-1., high=1., size=(env.action_space.shape[0]))
        print(' action:', action)
        next_obs, reward, done, info = env.step(action)
        print('  done:', done, ', reward:', reward)
        if headless:
            env.render('rgb_array')
        else:
            env.render('human')
        if done:
            env.reset()
            break


if __name__ == '__main__':
    main(headless=True)


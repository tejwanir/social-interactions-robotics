# Social Interaction Robots

## Setting up the custom Fetch environment

- Create a new conda environment
  ```
  conda create mpi4py python=3.7 -n fetch
  ```

- Set up MuJoCo
  1. Download the MuJoCo version 2.0 binaries from the [MuJoCo website](https://www.roboti.us/license.html).
  2. Set the paths (`MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH`) if you put the binary and key in a different place.

- Install requirements
  ```
  pip install -r requirements.txt
  ```

- Demo environment: `FetchThreeObjEnv`
  - After installing the requirements, run `demo.py` to see the demo environment.
  - This environment contains a Fetch robot and three blocks on the table. The red rectangle area at the corner of the table is the tray area. This environment gives reward 1 if the robot successfully move any block to the tray; and the environment gives `done=True` when the robot succeeds or timeout (the default time limit is 50 time steps).
  - To add other robot or objects, there are at least two required changes:
    1. update `pnp_three_objects.xml` or add a new xml file to specify the environment configuration.
    2. update `fetch_three_obj_env.py` or add a new class that extends `RobotEnv` to specify the rules for `reset()` and `step()`


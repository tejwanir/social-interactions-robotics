# Social Interaction Robots

## Setting up the Fetch environment

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


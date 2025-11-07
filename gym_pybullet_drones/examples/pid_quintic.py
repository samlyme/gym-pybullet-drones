"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, -.3).

"""

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import numpy as np


def quintic_coeffs_tspan(p0, v0, a0, p1, v1, a1, t0, tf):
    """
    Quintic between states at absolute times t0 and tf.
    Accepts scalars or np arrays (e.g., shape (3,) for x,y,z).
    Returns (b, t0, tf) where b[...,6] are normalized-time coeffs for u=(t-t0)/(tf-t0).
    """
    T = float(tf) - float(t0)
    if T <= 0:
        raise ValueError("Require tf > t0")

    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a1 = np.asarray(a1, dtype=float)

    b0 = p0
    b1 = v0 * T
    b2 = 0.5 * a0 * T**2

    S1 = p1 - (b0 + b1 + b2)
    S2 = v1 * T - (b1 + 2 * b2)
    S3 = a1 * T**2 - (2 * b2)

    b3 = 10 * S1 - 4 * S2 + 0.5 * S3
    b4 = -15 * S1 + 7 * S2 - S3
    b5 = 6 * S1 - 3 * S2 + 0.5 * S3

    b = np.stack([b0, b1, b2, b3, b4, b5], axis=-1)
    return b, float(t0), float(tf)


def quintic_eval_abs(b, t, t0, tf, clip=True):
    """
    Evaluate pos, vel, acc at absolute time t.
    b: (...,6) coefficients from quintic_coeffs_tspan
    If clip=True, clamps t to [t0, tf] (u in [0,1]); set False to allow extrapolation.
    """
    T = tf - t0
    if T <= 0:
        raise ValueError("Require tf > t0")

    t = float(t)
    if clip:
        if t < t0:
            t = t0
        if t > tf:
            t = tf

    u = (t - t0) / T
    b0, b1, b2, b3, b4, b5 = [b[..., i] for i in range(6)]

    # Position via Horner
    x = ((((b5 * u + b4) * u + b3) * u + b2) * u + b1) * u + b0

    # Derivatives wrt u
    dx_du = (((5 * b5 * u + 4 * b4) * u + 3 * b3) * u + 2 * b2) * u + b1
    d2x_du2 = ((20 * b5 * u + 12 * b4) * u + 6 * b3) * u + 2 * b2

    v = dx_du / T
    a = d2x_du2 / (T**2)
    return x, v, a


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
):
    #### Initialize the simulation #############################
    H = 0.1
    H_STEP = 0.05
    R = 0.3
    INIT_XYZS = np.array(
        [
            [
                0+i,
                0,
                H + i * H_STEP,
            ]
            for i in range(num_drones)
        ]
    )
    INIT_RPYS = np.array(
        [[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)]
    )

    #### Initialize a Quintic trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    print(INIT_XYZS[0])
    q_arr = [
        quintic_coeffs_tspan(
            pos, [0, 0, 0], [0, 0, 0], 
            (pos + 1)*2, [0, 0, 0], [0, 0, 0], 
            0, PERIOD
        )[0]
        for pos in INIT_XYZS
    ]
    print("q_arr", q_arr)
    for i in range(NUM_WP):
        TARGET_POS[i, :] = (
            quintic_eval_abs(q_arr[0][0], PERIOD*i/NUM_WP, 0, PERIOD)[0],
            quintic_eval_abs(q_arr[0][1], PERIOD*i/NUM_WP, 0, PERIOD)[0],
            quintic_eval_abs(q_arr[0][2], PERIOD*i/NUM_WP, 0, PERIOD)[0],
            # 1,1,1
        )
    print("TARGET_POS", TARGET_POS[-1])
    wp_counters = np.array([int((i * NUM_WP) % NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        print("Naving to ", TARGET_POS[wp_counters[0]])

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=TARGET_POS[wp_counters[j]],
                # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                target_rpy=INIT_RPYS[j, :],
            )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack(
                    [
                        TARGET_POS[wp_counters[j], 0:2],
                        INIT_XYZS[j, 2],
                        INIT_RPYS[j, :],
                        np.zeros(6),
                    ]
                ),
                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
            )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid")  # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Helix flight script using CtrlAviary and DSLPIDControl"
    )
    parser.add_argument(
        "--drone",
        default=DEFAULT_DRONES,
        type=DroneModel,
        help="Drone model (default: CF2X)",
        metavar="",
        choices=DroneModel,
    )
    parser.add_argument(
        "--num_drones",
        default=DEFAULT_NUM_DRONES,
        type=int,
        help="Number of drones (default: 3)",
        metavar="",
    )
    parser.add_argument(
        "--physics",
        default=DEFAULT_PHYSICS,
        type=Physics,
        help="Physics updates (default: PYB)",
        metavar="",
        choices=Physics,
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VISION,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--obstacles",
        default=DEFAULT_OBSTACLES,
        type=str2bool,
        help="Whether to add obstacles to the environment (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="Simulation frequency in Hz (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
        help="Control frequency in Hz (default: 48)",
        metavar="",
    )
    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
        help="Duration of the simulation in seconds (default: 5)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))

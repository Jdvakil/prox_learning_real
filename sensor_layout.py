"""Proximity sensors on the arm.

Each link streams UDP multicast on its own port. A packet's first byte is the sensor id on
that link; the rest is 128 uint16 distances in mm, two row-major 8x8 blocks. Block 1 holds
only confident zones (4000 mm = no target); block 2 equals block 1 wherever block 1 is valid
and fills the other zones with low-confidence ranges. 41 sensors in total.
"""

LINKS = {
    "link1": 5004,
    "link2": 5005,
    "link3": 5006,
    "link5_small": 5007,
    "link5_large": 5008,
    "link6": 5009,
    "link4": 5010,
    "end_effector": 5011,
}

SENSORS_PER_LINK = {
    "link1": 7,
    "link2": 7,
    "link3": 5,
    "link5_small": 4,
    "link5_large": 6,
    "link6": 6,
    "link4": 5,
    "end_effector": 1,
}

NO_TARGET_MM = 4000

# Camera prefix of each link in the sim model (prox_learning
# assets/robots/franka_skin/model_hybrid_franka_hand.xml). Sensor id k on a link is taken to be
# <prefix>_sensor_k. The counts match the arm; the id order has not been checked on the arm.
SIM_PREFIX = {
    "link1": "link1",
    "link2": "link2",
    "link3": "link3",
    "link4": "link4",
    "link5_small": "link5_front",
    "link5_large": "link5_back",
    "link6": "link6",
    "end_effector": "link7_hand",
}


def sim_name(link, sensor_id):
    return f"{SIM_PREFIX[link]}_sensor_{sensor_id}"

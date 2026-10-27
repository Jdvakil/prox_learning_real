"""Proximity sensors on the arm.

A packet's first byte is the sensor id on that link. The rest is a zone image:
128 uint16 distances, laid out as 64 zones for the first target and 64 for the
second. 4000 mm means that zone has no target. The ids below are the 41 sensors.
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
ZONES_PER_SENSOR = 64


def zone_ranges(depth_mm):
    """One distance per zone. None means that zone did not return a target."""
    depths = list(depth_mm)
    if len(depths) >= ZONES_PER_SENSOR * 2:
        zones = []
        for index in range(ZONES_PER_SENSOR):
            candidates = (depths[index], depths[ZONES_PER_SENSOR + index])
            hits = [value for value in candidates if 0 < value < NO_TARGET_MM]
            zones.append(min(hits) if hits else None)
        return zones
    return [value if 0 < value < NO_TARGET_MM else None for value in depths]


def nearest_mm(depth_mm):
    hits = [value for value in zone_ranges(depth_mm) if value is not None]
    if not hits:
        return None
    return min(hits)

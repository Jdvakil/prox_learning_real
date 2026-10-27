import json
import sys

from sensor_layout import SENSORS_PER_LINK, ZONES_PER_SENSOR, nearest_mm, zone_ranges

PATH = sys.argv[1] if len(sys.argv) > 1 else "steps_dummy.json"


def load_steps(path):
    steps = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: skipped invalid JSON ({exc})")
    return steps


def packets_from(reading):
    if reading is None:
        return []
    if isinstance(reading, list):
        return [packet for packet in reading if isinstance(packet, dict)]
    if isinstance(reading, dict) and "depth_mm" in reading:
        return [reading]
    if isinstance(reading, dict):
        return [packet for packet in reading.values() if isinstance(packet, dict) and "depth_mm" in packet]
    return []


def describe(packets):
    nearest = []
    latest_zones = 0
    for depth in packets:
        zones = zone_ranges(depth)
        latest_zones = sum(zone is not None for zone in zones)
        hit = nearest_mm(depth)
        if hit is not None:
            nearest.append(hit)
    if not nearest:
        return "check", f"no target, 0/{ZONES_PER_SENSOR} zones, {len(packets)} packets"
    latest = nearest[-1]
    if len(nearest) == 1:
        span = f"{latest} mm nearest"
    else:
        span = f"{latest} mm nearest, range {min(nearest)}–{max(nearest)} mm over {len(nearest)} packets"
    return "ok", f"{span}, {latest_zones}/{ZONES_PER_SENSOR} zones"


def main():
    steps = load_steps(PATH)
    print(f"file: {PATH}")
    print(f"lines: {len(steps)}")
    if not steps:
        print("result: no sensor readings")
        return

    by_sensor = {name: {} for name in SENSORS_PER_LINK}
    addresses = {name: None for name in SENSORS_PER_LINK}
    for step in steps:
        readings = step.get("sensors", {})
        for name in SENSORS_PER_LINK:
            for packet in packets_from(readings.get(name)):
                sensor_id = int(packet.get("sensor", -1))
                by_sensor[name].setdefault(sensor_id, []).append(packet.get("depth_mm", []))
                addresses[name] = packet.get("address", addresses[name])

    expected = sum(SENSORS_PER_LINK.values())
    present = 0
    rows = []
    for name, count in SENSORS_PER_LINK.items():
        sensor_ids = list(range(count))
        sensor_ids.extend(sensor_id for sensor_id in sorted(by_sensor[name]) if sensor_id not in sensor_ids)
        link_rows = []
        for sensor_id in sensor_ids:
            packets = by_sensor[name].get(sensor_id, [])
            if not packets:
                link_rows.append((sensor_id, "check", "missing"))
                continue
            status, detail = describe(packets)
            if status == "ok":
                present += 1
            link_rows.append((sensor_id, status, detail))
        rows.append((name, link_rows))

    if present == expected:
        print(f"result: all {expected} sensors returned a range")
    else:
        print(f"result: {present}/{expected} sensors returned a range")
    print()

    for name, link_rows in rows:
        fine = sum(status == "ok" for _, status, _ in link_rows)
        address = addresses[name] or "unknown"
        print(f"{name}: {fine}/{len(link_rows)} sensors, address {address}")
        for sensor_id, status, detail in link_rows:
            print(f"  sensor {sensor_id}: {status:5}  {detail}")
        print()


if __name__ == "__main__":
    main()

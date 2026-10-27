import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "steps.json"

LINKS = (
    "link1",
    "link2",
    "link3",
    "link5_small",
    "link5_large",
    "link6",
    "link4",
    "end_effector",
)

# Arm proximity readings outside this band are usually a missed or saturated return.
MIN_MM = 1
MAX_MM = 8000


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


def plausible(value):
    return MIN_MM <= value <= MAX_MM


def sensor_status(values, expected):
    if not values:
        return "check", "no data"
    latest = values[-1]
    lo, hi = min(values), max(values)
    bad = [value for value in values if not plausible(value)]
    if len(values) == 1:
        reading = f"{latest} mm"
    else:
        reading = f"{latest} mm, range {lo}–{hi} mm over {len(values)} steps"
    notes = []
    if len(values) != expected:
        missed = expected - len(values)
        notes.append(f"missing from {missed} step" if missed == 1 else f"missing from {missed} steps")
    if len(bad) == len(values):
        if all(value == 0 for value in values):
            notes.append("all zeros")
        elif all(value == 65535 for value in values):
            notes.append("saturated")
        else:
            notes.append("out of range")
    elif bad:
        notes.append(f"{len(bad)}/{len(values)} out of range")
    if notes:
        return "check", f"{reading} ({'; '.join(notes)})"
    return "ok", reading


def main():
    steps = load_steps(PATH)
    print(f"file: {PATH}")
    print(f"steps: {len(steps)}")
    if not steps:
        print("result: no sensor readings")
        return

    per_link = {name: [] for name in LINKS}
    for step in steps:
        sensors = step.get("sensors", {})
        for name in LINKS:
            per_link[name].append(sensors.get(name))

    reports = []
    for name in LINKS:
        readings = [reading for reading in per_link[name] if reading is not None]
        missing = len(per_link[name]) - len(readings)
        if not readings:
            reports.append((name, None, missing, []))
            continue
        counts = [len(reading.get("depth_mm", [])) for reading in readings]
        width = max(counts)
        sensors = []
        for index in range(width):
            values = [
                reading["depth_mm"][index]
                for reading in readings
                if index < len(reading.get("depth_mm", []))
            ]
            sensors.append(sensor_status(values, len(readings)))
        reports.append((name, readings[-1], missing, sensors, len(set(counts)) > 1))

    ok_count = sum(status == "ok" for _, _, _, sensors, *_ in reports for status, _ in sensors)
    total = sum(len(sensors) for _, _, _, sensors, *_ in reports)
    missing_links = [name for name, reading, _, _, *_ in reports if reading is None]
    needs_look = total - ok_count + len(missing_links)
    if needs_look == 0:
        print(f"result: all {total} sensors look fine")
    else:
        noun = "sensor needs" if needs_look == 1 else "sensors need"
        print(f"result: {ok_count}/{total} sensors look fine, {needs_look} {noun} a look")
    print()

    for name, reading, missing, sensors, *rest in reports:
        if reading is None:
            print(f"{name}: missing from every step")
            continue
        fine = sum(status == "ok" for status, _ in sensors)
        varied = rest[0] if rest else False
        print(f"{name}: {fine}/{len(sensors)} fine, address {reading.get('address')}")
        if missing:
            print(f"  missing from {missing} steps")
        if varied:
            print("  sensor count changed between steps")
        for index, (status, detail) in enumerate(sensors):
            print(f"  sensor {index}: {status:5}  {detail}")
        print()


if __name__ == "__main__":
    main()

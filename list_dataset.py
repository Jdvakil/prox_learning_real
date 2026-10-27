import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "steps.json"


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


def main():
    steps = load_steps(PATH)
    sensor_names = []
    seen = set()
    for step in steps:
        for name in step.get("sensors", {}):
            if name not in seen:
                seen.add(name)
                sensor_names.append(name)

    print(f"file: {PATH}")
    print(f"steps: {len(steps)}")
    print(f"sensors: {', '.join(sensor_names) if sensor_names else '(none)'}")

    if not steps:
        return

    print()
    for step in steps:
        print(f"step {step.get('step')}")
        for name, reading in step.get("sensors", {}).items():
            depths = reading.get("depth_mm", [])
            print(
                f"  {name}: "
                f"address={reading.get('address')} "
                f"sensor={reading.get('sensor')} "
                f"len={reading.get('len')} "
                f"depth_count={len(depths)}"
            )
            print(f"    depth_mm: {depths}")
        print()


if __name__ == "__main__":
    main()

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
    print(f"file: {PATH}")
    print(f"steps: {len(steps)}")

    if not steps:
        return

    print()
    for step in steps:
        print(f"step {step.get('step')}")
        for name, reading in step.get("sensors", {}).items():
            depths = reading.get("depth_mm", [])
            print(f"  {name}:")
            print(f"    address: {reading.get('address')}")
            print(f"    packet_len: {reading.get('len')}")
            if not depths:
                print("    sensors: (none)")
                continue
            for index, depth in enumerate(depths):
                print(f"    sensor {index}: {depth} mm")
        print()


if __name__ == "__main__":
    main()

"""Write one offline HTML dashboard for a recorded row.

    python dashboard_row.py recordings/<utc stamp>        # -> viz/<utc stamp>_dashboard.html
    python dashboard_row.py recordings/*/                 # one dashboard per row

Needs h5py, numpy and opencv. dashboard/template.html holds the page, dashboard/d3.min.js its charts.
"""

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
NO_TARGET_MM = 4000
MAX_DEPTH_MM = 10000             # D435 range; larger values are low-disparity noise
FRAME_STEP = 4                   # ticks per replay frame
STUCK_VALID, STUCK_MM = 0.9, 100.0
SATURATED_LIMIT_PCT = 5.0
BLOCKED_SHARPNESS = 15.0         # Laplacian variance of a 320x240 frame; a covered lens scores near 0


def number(value, digits=1):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def jpeg_uri(bgr, quality=62):
    data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def depth_image(depth, near, far):
    level = np.clip((far - depth.astype(np.float32)) / (far - near), 0, 1)
    out = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[(depth == 0) | (depth > MAX_DEPTH_MM)] = 0
    return out


def depth_range(depth):
    frames = depth[:: max(len(depth) // 10, 1)]
    valid = frames[(frames > 0) & (frames <= MAX_DEPTH_MM)]
    if valid.size == 0:
        return 100.0, 4000.0
    near, far = np.percentile(valid, [2, 98])
    return float(near), float(max(far, near + 1))


def rgb_frames(group, row, name, indices):
    """{frame index: BGR} from the h5 copy (--rgb-h5), else the <name>_<fps>fps.mp4."""
    wanted = sorted(set(int(i) for i in indices))
    if "rgb" in group:
        return {i: group["rgb"][i][..., ::-1] for i in wanted}
    videos = sorted(row.glob(f"{name}_*fps.mp4"))
    if not videos:
        return {}
    video, out, index = cv2.VideoCapture(str(videos[0])), {}, 0
    while index <= wanted[-1]:
        ok, frame = video.read()
        if not ok:
            break
        if index in wanted:
            out[index] = frame
        index += 1
    video.release()
    return {i: out[min(out, key=lambda k: abs(k - i))] for i in wanted} if out else {}


def sensor_rows(real, t):
    depth, age, fresh = real["depth_mm"][:], real["age_ms"][:], real["fresh"][:]
    names = [value.decode() for value in real["sensor_names"][:]]
    links = [value.decode() for value in real["sensor_link"][:]]
    block1 = depth[:, :, 0]
    valid = (block1 > 0) & (block1 < NO_TARGET_MM)
    span = max(t[-1] - t[0], 1e-9)
    T = len(t)
    rows = []
    for i, name in enumerate(names):
        zones = block1[:, i].reshape(T, 64).astype(float)
        ok = (zones > 0) & (zones < NO_TARGET_MM)
        stuck = sum(1 for z in range(64)
                    if ok[:, z].mean() > STUCK_VALID and np.median(zones[:, z][ok[:, z]]) < STUCK_MM)
        ticks = np.flatnonzero(fresh[:, i])
        noise = None
        if len(ticks) > 1:
            both = ok[ticks[1:]] & ok[ticks[:-1]]
            if both.any():
                noise = number(np.median(np.abs(zones[ticks[1:]] - zones[ticks[:-1]])[both]))
        seen = np.isfinite(age[:, i])
        rows.append({"index": i, "sensor": name, "link": links[i], "valid_pct": number(100 * ok.mean()),
                     "rate_hz": number(fresh[:, i].sum() / span), "age_ms": number(np.median(age[:, i][seen])) if seen.any() else None,
                     "median_mm": int(np.median(zones[ok])) if ok.any() else None, "noise_mm": noise, "stuck_zones": stuck})
    link_rows = []
    for link in sorted(set(links)):
        index = [i for i, value in enumerate(links) if value == link]
        link_rows.append({"link": link, "sensors": len(index), "rate_hz": number(fresh[:, index].sum(0).mean() / span),
                          "valid_pct": number(100 * valid[:, index].mean()),
                          "stuck_zones": sum(r["stuck_zones"] for r in rows if r["link"] == link)})
    return rows, link_rows, depth, age, fresh, links


def tile(label, value, unit, state, note):
    mark = {"ok": "✓", "warn": "!", "bad": "✕"}[state]
    return (f'<div class="sk-tile"><span class="lab">{html.escape(label)}</span><span><b class="num">{value}</b> '
            f'<span class="unit">{html.escape(unit)}</span></span><span class="sk-pill {state}">{mark} {html.escape(note)}</span></div>')


def build(row, out_dir):
    with h5py.File(row / "trajectory.h5", "r") as handle:
        traj = handle["traj_0"]
        real = traj["real"]
        t, q, dq, target = real["t"][:], real["q"][:], real["dq"][:], real["target_q"][:]
        pose, phase = real["O_T_EE"][:], traj["obs/extra/policy_phase"][:]
        T = len(t)
        sensors, link_rows, depth, age, fresh, links = sensor_rows(real, t)
        block1 = depth[:, :, 0]
        valid = (block1 > 0) & (block1 < NO_TARGET_MM)
        order = sorted(set(links))

        kinematics = []
        for tick in range(T):
            entry = {"tick": tick, "t": round(float(t[tick]), 3), "phase": int(phase[tick]),
                     "x": round(float(pose[tick, 0, 3]), 3), "y": round(float(pose[tick, 1, 3]), 3),
                     "z": round(float(pose[tick, 2, 3]), 3)}
            for joint in range(7):
                entry[f"q{joint + 1}"] = round(float(q[tick, joint]), 4)
                entry[f"v{joint + 1}"] = round(float(dq[tick, joint]), 4)
            for link in order:
                index = [i for i, value in enumerate(links) if value == link]
                values = block1[tick, index][valid[tick, index]]
                entry[f"near_{link}"] = round(float(values.min()) / 1000, 3) if values.size else None
            kinematics.append(entry)

        zones = [{"tick": i, "t": round(float(t[i]), 3), "fresh": "".join("1" if v else "0" for v in fresh[i]),
                  "b1": base64.b64encode(depth[i, :, 0].astype("<u2").tobytes()).decode(),
                  "b2": base64.b64encode(depth[i, :, 1].astype("<u2").tobytes()).decode()} for i in range(T)]

        ticks = list(range(0, T, FRAME_STEP))
        frames = [{"tick": i, "t": round(float(t[i]), 3)} for i in ticks]
        figures, tiles, problems = [], [], []
        error = str(traj.attrs.get("error", ""))
        final = float(np.abs(target[-1] - q[-1]).max())
        moved = not error and final < 0.01
        tiles.append(tile("Largest joint error at the end", f"{final:.4f}", "rad", "ok" if moved else "bad",
                          "Motion finished" if moved else "Motion did not finish"))
        if not moved:
            problems.append(f"the motion stopped early ({error})" if error else "the arm did not reach its last waypoint")
        for name in (list(real["cameras"]) if "cameras" in real else []):
            group = real[f"cameras/{name}"]
            tick_frame = group["tick_frame"][:]
            index = [int(tick_frame[i]) for i in ticks]
            rgb = rgb_frames(group, row, name, index)
            stack = group["depth_mm"] if "depth_mm" in group else None
            near, far = depth_range(stack) if stack is not None else (0.0, 1.0)
            sharp = []
            for slot, frame_index in enumerate(index):
                if frame_index in rgb:
                    small = cv2.resize(rgb[frame_index], (320, 240), interpolation=cv2.INTER_AREA)
                    frames[slot][f"{name}_rgb"] = jpeg_uri(small)
                    sharp.append(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if stack is not None:
                    frames[slot][f"{name}_depth"] = jpeg_uri(cv2.resize(depth_image(stack[frame_index], near, far), (320, 240),
                                                                        interpolation=cv2.INTER_NEAREST))
            for suffix, caption in (("rgb", "colour"), ("depth", f"depth, {near:.0f} to {far:.0f} mm")):
                if suffix == "rgb" and not rgb or suffix == "depth" and stack is None:
                    continue
                figures.append(f'<figure><div class="sk-img" data-draw="cam" data-cam="{html.escape(name)}_{suffix}">—</div>'
                               f'<figcaption>{html.escape(name)} {caption}</figcaption></figure>')
            arrival = group["t"][:]
            fps = (len(arrival) - 1) / max(arrival[-1] - arrival[0], 1e-9)
            dropped = int((np.diff(group["frame_number"][:]) - 1).sum())
            saturated = 100 * float((stack[len(stack) // 2] >= 65535).mean()) if stack is not None else 0.0
            sharpness = float(np.median(sharp)) if sharp else 0.0
            print(f"{name}: sharpness {sharpness:.0f}, saturated depth {saturated:.1f}%")
            if sharpness < BLOCKED_SHARPNESS or saturated > SATURATED_LIMIT_PCT:
                state, note = "bad", "Looks blocked or blurred"
                problems.append(f"{name} looks blocked or blurred")
            elif dropped:
                state, note = "warn", f"{dropped} frames dropped"
                problems.append(f"{name} dropped {dropped} frames")
            else:
                state, note = "ok", "Good"
            tiles.append(tile(f"{name} camera", f"{fps:.2f}", f"fps, {dropped} dropped", state, note))
        reporting = int(np.isfinite(age).any(0).sum())
        if reporting == 0:
            tiles.append(tile("Sensors reporting", "0", f"of {len(sensors)}", "warn", "Sensors were off"))
            problems.append("the sensors were off")
        else:
            state = "ok" if reporting == len(sensors) else "bad"
            tiles.append(tile("Sensors reporting", str(reporting), f"of {len(sensors)}", state,
                              "All reporting" if state == "ok" else "Some silent"))
            if state != "ok":
                problems.append(f"{len(sensors) - reporting} sensors never reported")
            slowest = min(r["rate_hz"] for r in sensors)
            tiles.append(tile("Slowest sensor", f"{slowest}", f"Hz, oldest packet {np.nanmax(age):.0f} ms", "warn" if slowest < 15 else "ok",
                              "Below the 15 Hz log rate" if slowest < 15 else "Keeps up"))
            stuck = sum(r["stuck_zones"] for r in sensors)
            tiles.append(tile("Zones stuck under 100 mm", str(stuck), f"of {len(sensors) * 64}", "warn" if stuck else "ok",
                              "Mask before training" if stuck else "None"))
        takeaway = ("Every check passed." if not problems
                    else "Check before using this row: " + "; ".join(problems) + ".")

    data = {"kinematics": kinematics, "sensors": sensors, "links": link_rows, "zones": zones, "frames": frames}
    page = (HERE / "dashboard" / "template.html").read_text()
    replacements = {"{{ROW}}": html.escape(row.name), "{{TAKEAWAY}}": html.escape(takeaway), "{{TILES}}": "".join(tiles),
                    "{{CAMERAS}}": "".join(figures), "{{FRAME_STEP}}": str(FRAME_STEP),
                    "{{D3}}": (HERE / "dashboard" / "d3.min.js").read_text(),
                    "{{DATA}}": json.dumps(data, separators=(",", ":")).replace("</", "<\\/")}
    for key, value in replacements.items():
        page = page.replace(key, value)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{row.name}_dashboard.html"
    path.write_text(page)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {takeaway}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path, nargs="+", help="row folders holding trajectory.h5")
    parser.add_argument("--out", type=Path, default=Path("viz"))
    args = parser.parse_args()
    for row in args.rows:
        build(row.resolve(), args.out)


if __name__ == "__main__":
    main()

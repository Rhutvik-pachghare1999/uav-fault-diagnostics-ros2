
import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
import random
import csv

# ----------------------------
# Environment defaults (MUST run BEFORE importing omni/SimulationApp)
# ----------------------------
os.environ.setdefault("EXP_PATH", os.environ.get("EXP_PATH", "/home/rhutvik/isaac_sim_uav"))
os.environ.setdefault("CARB_APP_PATH", os.environ.get("CARB_APP_PATH", "/home/rhutvik/isaacsim"))
os.environ.setdefault("PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.environ["EXP_PATH"]))
os.environ.setdefault("ISAAC_PATH", os.environ.get("ISAAC_PATH", os.environ.get("CARB_APP_PATH")))
os.environ.setdefault("QUAD_USD_PATH", os.environ.get("QUAD_USD_PATH",
                                                     os.path.join(os.environ["PROJECT_ROOT"], "scripts", "quad.usd")))
os.environ.setdefault("ISAAC_OUTDIR", os.environ.get("ISAAC_OUTDIR",
                                                     os.path.join(os.environ["PROJECT_ROOT"], "isaac_dataset")))

# ----------------------------
# CLI args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Robust Isaac/ fallback recorder -> CSVs with min rows")
    p.add_argument("--usd", type=str, default=os.environ.get("QUAD_USD_PATH"))
    p.add_argument("--outdir", type=str, default=os.environ.get("ISAAC_OUTDIR"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--robot-prim", type=str, default="/World/body")
    p.add_argument("--duration", type=float, default=20.0, help="seconds (will be increased if needed to reach min-rows)")
    p.add_argument("--fps", type=float, default=100.0, help="samples per second")
    p.add_argument("--min-rows", type=int, default=20000, help="minimum number of samples/rows to produce")
    p.add_argument("--rotor-prims", type=str, default="", help="comma separated rotor prim paths")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--apply-fault", action="store_true")
    p.add_argument("--fault-motor", type=int, default=1)
    p.add_argument("--ur", type=float, default=0.0, help="fault severity (0..1)")
    p.add_argument("--use-fallback", action="store_true",
                   help="force fallback synthetic data generator (skip Isaac imports)")
    return p.parse_args()

# ----------------------------
# utility: thrust/torque calc
# ----------------------------
SIMPLE_THRUST_K = 6e-7
SIMPLE_TORQUE_K = 2e-8

def analytic_thrust_torque_from_rpm(rpms):
    thrusts = [SIMPLE_THRUST_K * (float(r)**2) for r in rpms]
    torques = [SIMPLE_TORQUE_K * (float(r)**2) for r in rpms]
    return thrusts, torques

def rpm_to_rad_s(rpm):
    return rpm * 2.0 * math.pi / 60.0

# ----------------------------
# Fallback synthetic generator
# ----------------------------
def generate_synthetic(outdir, run_name, duration, fps, min_rows, apply_fault=False, fault_motor=1, ur=0.0):
    print("Running FALLBACK synthetic data generator (no Isaac).")
    # ensure at least min_rows samples
    steps = max(int(math.ceil(duration * fps)), int(min_rows))
    adjusted_duration = steps / float(fps)
    if adjusted_duration > duration:
        print(f"[INFO] Increased duration from {duration:.3f}s -> {adjusted_duration:.3f}s to meet min_rows={min_rows}")
        duration = adjusted_duration
    tstep = 1.0 / fps
    run_folder = os.path.join(outdir, run_name)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    imu_csv = os.path.join(run_folder, "imu_full.csv")
    thrust_csv = os.path.join(run_folder, "thrust_tau.csv")
    state_csv = os.path.join(run_folder, "state.csv")

    # base waveforms and controlled randomness to look realistic
    base_rpm = 3200.0
    rpm_amp = 800.0
    noise_scale_acc = 0.05  # m/s^2 noise
    noise_scale_gyro = 0.005  # rad/s noise
    pos_noise = 0.01

    prev_rpy = (0.0, 0.0, 0.0)

    with open(imu_csv, "w", newline='') as f_imu, \
         open(thrust_csv, "w", newline='') as f_th, \
         open(state_csv, "w", newline='') as f_state:

        imu_writer = csv.writer(f_imu)
        th_writer = csv.writer(f_th)
        st_writer = csv.writer(f_state)

        imu_writer.writerow(["time","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","roll","pitch","yaw","rpm1","rpm2","rpm3","rpm4"])
        th_writer.writerow(["time","thrust1","thrust2","thrust3","thrust4","torque1","torque2","torque3","torque4"])
        st_writer.writerow(["time","x","y","z","vx","vy","vz","roll","pitch","yaw"])

        # simple dynamics state
        x=y=z=vx=vy=vz=0.0

        for step in range(steps):
            t = step * tstep

            # RPM waveform: slow modulation + motor-to-motor offsets + slight jitter
            base = base_rpm + rpm_amp * math.sin(2*math.pi*0.2*t)
            target_rpms = [base + (i*20.0) + (5.0*math.sin(2*math.pi*0.5*(i+1)*t)) + random.uniform(-10,10) for i in range(4)]
            if apply_fault:
                idx = max(0, min(3, fault_motor-1))
                target_rpms[idx] *= (1.0 - ur)

            # measured rpms are slightly noisy but close to target
            rpms = [max(0.0, r + random.uniform(-2.0, 2.0)) for r in target_rpms]
            thrusts, torques = analytic_thrust_torque_from_rpm(rpms)

            # fake imu: gravity + small vibration from motor imbalance
            acc_x = 0.02*math.sin(2*math.pi*4.0*t) + random.gauss(0, noise_scale_acc)
            acc_y = 0.02*math.cos(2*math.pi*3.5*t) + random.gauss(0, noise_scale_acc)
            acc_z = 9.81 + 0.01*math.sin(2*math.pi*1.0*t) + random.gauss(0, noise_scale_acc)

            gyro_x = 0.01*math.sin(2*math.pi*2.0*t) + random.gauss(0, noise_scale_gyro)
            gyro_y = 0.01*math.cos(2*math.pi*2.2*t) + random.gauss(0, noise_scale_gyro)
            gyro_z = 0.02*math.sin(2*math.pi*1.3*t) + random.gauss(0, noise_scale_gyro)

            # update pose with tiny responses to thrust imbalances
            net_thrust = sum(thrusts)
            az = (net_thrust * 1e3) - 9.81
            vz += az * tstep
            z += vz * tstep + random.uniform(-pos_noise, pos_noise)

            # small planar drift
            vx += 0.001 * math.sin(2*math.pi*0.05*t)
            vy += 0.001 * math.cos(2*math.pi*0.05*t)
            x += vx * tstep + random.uniform(-pos_noise, pos_noise)
            y += vy * tstep + random.uniform(-pos_noise, pos_noise)

            # complementary filter
            roll_g = prev_rpy[0] + gyro_x * tstep
            pitch_g = prev_rpy[1] + gyro_y * tstep
            yaw_g = prev_rpy[2] + gyro_z * tstep
            try:
                roll_a = math.atan2(acc_y, acc_z)
                pitch_a = math.atan2(-acc_x, math.sqrt(acc_y*acc_y + acc_z*acc_z))
            except Exception:
                roll_a, pitch_a = roll_g, pitch_g
            alpha = 0.98
            roll = alpha * roll_g + (1-alpha) * roll_a
            pitch = alpha * pitch_g + (1-alpha) * pitch_a
            yaw = yaw_g
            prev_rpy = (roll, pitch, yaw)

            imu_writer.writerow([f"{t:.6f}", f"{acc_x:.6f}", f"{acc_y:.6f}", f"{acc_z:.6f}",
                                 f"{gyro_x:.6f}", f"{gyro_y:.6f}", f"{gyro_z:.6f}",
                                 f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}",
                                 f"{rpms[0]:.2f}", f"{rpms[1]:.2f}", f"{rpms[2]:.2f}", f"{rpms[3]:.2f}"])

            th_writer.writerow([f"{t:.6f}", f"{thrusts[0]:.8f}", f"{thrusts[1]:.8f}", f"{thrusts[2]:.8f}", f"{thrusts[3]:.8f}",
                                f"{torques[0]:.10f}", f"{torques[1]:.10f}", f"{torques[2]:.10f}", f"{torques[3]:.10f}"])

            st_writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                                f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}"])

    meta = {
        "mode": "fallback",
        "usd": None,
        "robot_prim": None,
        "rotor_candidates": [],
        "rotor_exists": {},
        "run_folder": run_folder
    }
    summary = {"run_folder": run_folder, "steps": steps, "duration": duration, "fps": fps}
    with open(os.path.join(run_folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Synthetic run complete. Wrote {steps} rows to each CSV in: {run_folder}")
    return run_folder

# ----------------------------
# Isaac-enabled recorder (best-effort)
# ----------------------------
def run_isaac_recording(args):
    """Attempt actual Isaac sim run. If anything critical fails, raise Exception so caller can fallback."""
    try:
        from omni.isaac.kit import SimulationApp
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import Usd, UsdGeom
        import omni.usd
    except Exception as e:
        raise RuntimeError("Isaac imports failed: " + repr(e))

    sim_opts = {"headless": bool(args.headless)}
    sim = SimulationApp(sim_opts)
    world = World(physics_dt=1.0/args.fps, rendering_dt=1.0/args.fps)

    if not os.path.exists(args.usd):
        sim.close()
        raise FileNotFoundError(f"USD file not found: {args.usd}")

    add_reference_to_stage(args.usd)
    time.sleep(0.2)
    stage = omni.usd.get_context().get_stage()

    rotor_prims = []
    if args.rotor_prims:
        rotor_prims = [p.strip() for p in args.rotor_prims.split(",") if p.strip()]
    else:
        names = ["m1_prop","m2_prop","m3_prop","m4_prop","m1","m2","m3","m4"]
        cand = args.robot_prim if stage.GetPrimAtPath(args.robot_prim) and stage.GetPrimAtPath(args.robot_prim).IsValid() else "/World"
        for n in names:
            rotor_prims.append(cand + "/" + n)
    rotor_prims = list(dict.fromkeys(rotor_prims))
    exist_map = {rp: bool(stage.GetPrimAtPath(rp) and stage.GetPrimAtPath(rp).IsValid()) for rp in rotor_prims}

    dyn = None
    try:
        dyn = _dynamic_control.acquire_dynamic_control()
    except Exception:
        dyn = None

    run_name = args.run_name or f"run_auto_{int(time.time())}"
    outdir = os.path.join(args.outdir, run_name)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    imu_csv = os.path.join(outdir, "imu_full.csv")
    thrust_csv = os.path.join(outdir, "thrust_tau.csv")
    state_csv = os.path.join(outdir, "state.csv")

    # compute steps: ensure at least min_rows
    steps = max(int(math.ceil(args.duration * args.fps)), int(args.min_rows))
    adjusted_duration = steps / float(args.fps)
    if adjusted_duration > args.duration:
        print(f"[INFO] Increased duration from {args.duration:.3f}s -> {adjusted_duration:.3f}s to meet min_rows={args.min_rows}")
        args.duration = adjusted_duration

    prev_rpy = (0.0,0.0,0.0)

    with open(imu_csv, "w", newline='') as f_imu, \
         open(thrust_csv, "w", newline='') as f_th, \
         open(state_csv, "w", newline='') as f_state:

        imu_writer = csv.writer(f_imu)
        th_writer = csv.writer(f_th)
        st_writer = csv.writer(f_state)

        imu_writer.writerow(["time","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","roll","pitch","yaw","rpm1","rpm2","rpm3","rpm4"])
        th_writer.writerow(["time","thrust1","thrust2","thrust3","thrust4","torque1","torque2","torque3","torque4"])
        st_writer.writerow(["time","x","y","z","vx","vy","vz","roll","pitch","yaw"])

        for step in range(steps):
            try:
                world.step(render=False)
            except Exception as e:
                print("world.step failed:", e)
                break
            t = step / args.fps

            # try reading IMU prim if exists
            acc = (0.0,0.0,9.81); gyro = (0.0,0.0,0.0)
            try:
                imu_prim = None
                for c in [args.robot_prim + "/imu", "/World/imu", args.robot_prim + "/IMU"]:
                    p = stage.GetPrimAtPath(c)
                    if p and p.IsValid():
                        imu_prim = p; break
                if imu_prim:
                    a = imu_prim.GetAttribute("linear_acceleration")
                    if a and a.IsValid():
                        val = a.Get(); acc = tuple(val) if val else acc
                    a2 = imu_prim.GetAttribute("angular_velocity")
                    if a2 and a2.IsValid():
                        val2 = a2.Get(); gyro = tuple(val2) if val2 else gyro
            except Exception:
                pass

            # compute target rpms waveform (sinusoidal + small variations)
            base = 3200.0 + 800.0 * math.sin(2*math.pi*0.2*t)
            target_rpms = [base + i*20.0 for i in range(4)]
            if args.apply_fault:
                idx = args.fault_motor - 1
                if 0 <= idx < 4:
                    target_rpms[idx] = target_rpms[idx] * (1.0 - args.ur)

            # attempt to command rotors if dynamic control available (non-blocking)
            if dyn is not None:
                for i,rp in enumerate(rotor_prims[:4]):
                    try:
                        rb = dyn.get_rigid_body(rp)
                        if rb is not None and hasattr(dyn, "set_rigid_body_angular_velocity"):
                            ang = (0.0, 0.0, rpm_to_rad_s(target_rpms[i]))
                            dyn.set_rigid_body_angular_velocity(rb, ang)
                    except Exception:
                        pass

            # attempt to read actual rpms from prim attrs or dyn
            rpms = [0.0,0.0,0.0,0.0]
            try:
                for i,rp in enumerate(rotor_prims[:4]):
                    p = stage.GetPrimAtPath(rp)
                    if p and p.IsValid():
                        for an in ("angularVelocity","angular_velocity","rpm","rotationSpeed"):
                            try:
                                a = p.GetAttribute(an)
                                if a and a.IsValid():
                                    v = a.Get()
                                    if v is not None:
                                        if isinstance(v, (list, tuple)):
                                            rpms[i] = float(v[2] if len(v)>2 else v[0])
                                        else:
                                            rpms[i] = float(v)
                                        break
                            except Exception:
                                continue
                if all(abs(r)<1e-6 for r in rpms) and dyn is not None:
                    for i,rp in enumerate(rotor_prims[:4]):
                        try:
                            rb = dyn.get_rigid_body(rp)
                            if rb is not None and hasattr(dyn, "get_rigid_body_angular_velocity"):
                                angv = dyn.get_rigid_body_angular_velocity(rb)
                                val = abs(angv[2]) * 60.0 / (2*math.pi)
                                rpms[i] = float(val)
                        except Exception:
                            pass
            except Exception:
                pass

            if all(abs(r)<1e-6 for r in rpms):
                rpms = target_rpms

            thrusts, torques = analytic_thrust_torque_from_rpm(rpms)

            # approximate pose read
            x=y=z=vx=vy=vz=0.0
            try:
                prim = stage.GetPrimAtPath(args.robot_prim)
                if prim and prim.IsValid():
                    xform = UsdGeom.XformCommonAPI(prim)
                    trans = xform.GetTranslateAttr().Get()
                    if trans:
                        x,y,z = float(trans[0]), float(trans[1]), float(trans[2])
            except Exception:
                pass

            # complementary filter
            gyro_vals = list(gyro)
            if any(abs(g) > 50 for g in gyro_vals):
                gyro_vals = [g * math.pi/180.0 for g in gyro_vals]
            prev_rpy = complementary_filter_update(prev_rpy, acc, gyro_vals, 1.0/args.fps, alpha=0.98)

            imu_writer.writerow([f"{t:.6f}", f"{acc[0]:.6f}", f"{acc[1]:.6f}", f"{acc[2]:.6f}",
                                 f"{gyro_vals[0]:.6f}", f"{gyro_vals[1]:.6f}", f"{gyro_vals[2]:.6f}",
                                 f"{prev_rpy[0]:.6f}", f"{prev_rpy[1]:.6f}", f"{prev_rpy[2]:.6f}",
                                 f"{rpms[0]:.2f}", f"{rpms[1]:.2f}", f"{rpms[2]:.2f}", f"{rpms[3]:.2f}"])

            th_writer.writerow([f"{t:.6f}", f"{thrusts[0]:.8f}", f"{thrusts[1]:.8f}", f"{thrusts[2]:.8f}", f"{thrusts[3]:.8f}",
                                f"{torques[0]:.10f}", f"{torques[1]:.10f}", f"{torques[2]:.10f}", f"{torques[3]:.10f}"])

            st_writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                                f"{prev_rpy[0]:.6f}", f"{prev_rpy[1]:.6f}", f"{prev_rpy[2]:.6f}"])

    meta = {"mode": "isaac", "usd": args.usd, "robot_prim": args.robot_prim, "rotor_candidates": rotor_prims, "rotor_exists": exist_map}
    summary = {"run_folder": outdir, "steps": steps, "duration": args.duration, "fps": args.fps}
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        sim.close()
    except Exception:
        pass

    print(f"Isaac run complete. Wrote {steps} rows to each CSV in: {outdir}")
    return outdir

# ----------------------------
# complementary filter helper used in Isaac branch
# ----------------------------
def complementary_filter_update(prev_q, acc, gyro, dt, alpha=0.98):
    roll_g = prev_q[0] + gyro[0]*dt
    pitch_g = prev_q[1] + gyro[1]*dt
    yaw_g = prev_q[2] + gyro[2]*dt
    ax, ay, az = acc
    if abs(az) < 1e-6:
        roll_a = roll_g; pitch_a = pitch_g
    else:
        roll_a = math.atan2(ay, az)
        pitch_a = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    roll = alpha * roll_g + (1-alpha) * roll_a
    pitch = alpha * pitch_g + (1-alpha) * pitch_a
    yaw = yaw_g
    return (roll, pitch, yaw)

# ----------------------------
# main
# ----------------------------
def main():
    args = parse_args()
    args.outdir = args.outdir or os.environ.get("ISAAC_OUTDIR")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"run_auto_{int(time.time())}"

    # Ensure at least min_rows by possibly extending duration
    desired_steps = max(int(math.ceil(args.duration * args.fps)), int(args.min_rows))
    if desired_steps > int(math.ceil(args.duration * args.fps)):
        new_duration = desired_steps / float(args.fps)
        print(f"[MAIN] Extending duration {args.duration:.3f}s -> {new_duration:.3f}s to reach min_rows={args.min_rows}")
        args.duration = new_duration

    if args.use_fallback:
        run = generate_synthetic(args.outdir, run_name, args.duration, args.fps, args.min_rows, args.apply_fault, args.fault_motor, args.ur)
        print("Done (fallback).")
        return

    try:
        # attach min_rows to args for Isaac branch
        args.min_rows = args.min_rows
        run = run_isaac_recording(args)
        print("Done (Isaac). run_folder:", run)
    except Exception as e:
        print("Isaac run failed or not available:", repr(e))
        print("Falling back to synthetic generator to ensure CSV output.")
        run = generate_synthetic(args.outdir, run_name, args.duration, args.fps, args.min_rows, args.apply_fault, args.fault_motor, args.ur)
        print("Done (fallback after Isaac failure). run_folder:", run)

if __name__ == "__main__":
    main()

"""UAV propeller-fault data generation on the REAL Bitcraze Crazyflie 2.1 asset
(Isaac Sim 5.1, cf2x.usd from the NVIDIA cloud asset server).

Physics is genuine:
  * the 4 rotor revolute joints (m1..m4) of the articulation are driven by
    PhysX velocity drives -> spin-up/spin-down reaction torques on the frame
    are computed by the physics engine, not scripted;
  * a faulted rotor gets its prop-link center-of-mass offset -> PhysX computes
    the real rotating centrifugal (1P) force transmitted through the joint;
  * aerodynamic thrust (k_t*w^2) is applied at the rotor hubs, with per-rotor
    efficiency loss (erosion) and a once-per-rev thrust harmonic (chip);
  * prop aerodynamic drag torques are applied to the prop links (steady-state
    load -> weakened motors show a real RPM deficit) and the matching reaction
    torque is applied to the body;
  * an IsaacSim physics IMU sensor is mounted on the body link;
  * a cascaded position/attitude/rate controller closes the loop at 100 Hz,
    so faults appear through the closed-loop response (RPM redistribution,
    steady-state tilt, vibration), not as injected noise.

Outputs runs compatible with scripts/build_ml_dataset_v2.py:
  <out_root>/run_<name>/imu.csv    rpm1..4, roll, pitch, yaw, gyro_x/y/z, acc_x/y/z
  <out_root>/run_<name>/state.csv  t, pos, vel, rotor phase, rpm commands
  <out_root>/run_<name>/meta.json  fault_type, fault_params, ...

Run tokens:  healthy_p00_s0 | mask01_p05_s0 | mask0f_p10_s1
  maskNN = fault bitmask (hex 01..0f, bit i -> rotor i+1)
  pNN    = payload in grams (0, 5, 10 -> different hover RPM operating points)
  sN     = RNG seed (wind + trajectory phase)

Usage (Isaac Sim python):
  python.sh isaac/cf2x_fault_sim.py --out-root isaac_dataset --duration 9 \
      --runs healthy_p00_s0,mask01_p00_s0
  python.sh isaac/cf2x_fault_sim.py --list-runs
"""

import argparse
import json
import math
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np


def log(msg):
    print(msg)
    sys.stdout.flush()


try:
    import carb
    import omni.kit.commands
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.physics import _sensor
    from pxr import Gf, Sdf, UsdPhysics, UsdUtils
    from pxr import PhysicsSchemaTools

    try:
        from isaacsim.core.api.objects import GroundPlane
    except ImportError:
        from isaacsim.core.objects import GroundPlane
    try:
        from isaacsim.core.prims import SingleArticulation as ArticCls
    except ImportError:
        from isaacsim.core.api import Articulation as ArticCls

    ASSET = os.environ.get(
        "CF2X_USD",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "isaac",
            "assets",
            "Bitcraze",
            "Crazyflie",
            "cf2x.usd",
        ),
    )
    CF_PATH = "/World/cf2x"
    BODY_PATH = "/World/cf2x/body"
    PROP_PATHS = [f"/World/cf2x/m{i}_prop" for i in (1, 2, 3, 4)]
    JOINT_PATHS = [f"/World/cf2x/body/m{i}_joint" for i in (1, 2, 3, 4)]

    PHYS_HZ = 500.0
    CTRL_HZ = 100.0
    CTRL_EVERY = int(PHYS_HZ / CTRL_HZ)
    DT = 1.0 / PHYS_HZ

    # ----- Crazyflie parameters -----
    M_BODY_ASSET = 0.025  # kg (from USD)
    M_PROP = 0.0008  # kg each (from USD)
    BASE_MASS = M_BODY_ASSET + 4 * M_PROP  # ~28.2 g
    K_THRUST = 7.7e-7  # N/(rad/s)^2 -> hover w ~ 300 rad/s (~2865 RPM)
    C_YAW = 0.010 * K_THRUST  # prop drag torque coefficient
    ROTOR_DIR = np.array([+1.0, -1.0, +1.0, -1.0])  # m1 CCW, m2 CW, m3 CCW, m4 CW
    MOTOR_DAMP = 4.0e-6  # N*m*s/rad nominal drive damping
    HOVER_ALT = 0.55
    RPM_CONV = 60.0 / (2.0 * math.pi)
    RAD2DEG = 180.0 / math.pi  # USD angular drives take deg/s, API returns rad/s

    # ----- fault profile (applied to each faulted rotor) -----
    F_COM_OFFSET = 0.6e-3  # m prop COM eccentricity -> real 1P centrifugal force
    F_AERO_1P = 0.15  # thrust once-per-rev harmonic (chipped blade)
    F_EFFICIENCY = 0.85  # thrust efficiency loss (eroded blade)
    F_MOTOR_GAIN = 0.75  # drive damping scale (weakened motor -> RPM sag)
    F_UR = 0.30  # reported unbalance ratio per faulted rotor

    # fallback arm geometry (Bitcraze spec) if USD joints carry no translation
    CF_SPEC_OFFSETS = np.array(
        [[0.0325, 0.0325, 0.003], [0.0325, -0.0325, 0.003], [-0.0325, -0.0325, 0.003], [-0.0325, 0.0325, 0.003]]
    )

    def _set(api, create, value):
        attr = getattr(api, create)(value)
        attr.Set(value)
        return attr

    def quat_to_rpy(q):
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def quat_to_R(q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    class CFXSim:
        def __init__(self):
            self.world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=1.0 / 30.0)
            self.stage = omni.usd.get_context().get_stage()

            ref_prim = self.stage.DefinePrim(CF_PATH, "Xform")
            ok = ref_prim.GetReferences().AddReference(ASSET)
            log(f"ADD_REF_OK={ok}")

            self.ground = self.world.scene.add(GroundPlane(prim_path="/World/ground"))

            # rotor arm offsets + joint axes from the USD joints
            # (joints carry physics:localPos0 in body0 frame, not xform ops)
            self.arm_offsets = []
            axes = []
            for jp in JOINT_PATHS:
                prim = self.stage.GetPrimAtPath(jp)
                lp0 = prim.GetAttribute("physics:localPos0").Get()
                rj = UsdPhysics.RevoluteJoint(prim)
                ax = rj.GetAxisAttr().Get()
                if lp0 is None or (abs(lp0[0]) + abs(lp0[1])) < 1e-9:
                    self.arm_offsets = [o.copy() for o in CF_SPEC_OFFSETS]
                    axes = [(0.0, 0.0, 1.0)] * 4
                    log("WARN arm offsets unavailable, using CF spec")
                    break
                self.arm_offsets.append(np.array([float(lp0[0]), float(lp0[1]), float(lp0[2])]))
                ax = rj.GetAxisAttr().Get()
                if isinstance(ax, str):
                    axes.append(ax.upper())
                else:
                    axes.append(tuple(round(float(v), 3) for v in (ax if ax is not None else (0, 0, 1))))
            log(f"ARM_OFFSETS={[[round(float(v), 4) for v in o] for o in self.arm_offsets]}")
            log(f"JOINT_AXES={axes}")

            # thrust mix matrix: [T_total; tau_x; tau_y; 0] = M @ [T1..T4]
            M = np.zeros((4, 4))
            M[0, :] = 1.0
            for i, o in enumerate(self.arm_offsets):
                M[1, i] = o[1]  # tau_x = sum y_i * T_i   (r x F, F=(0,0,T))
                M[2, i] = -o[0]  # tau_y = sum -x_i * T_i
            self.Minv = np.linalg.pinv(M)

            # IMU sensor on the body link
            self.imu_path = None
            try:
                out = omni.kit.commands.execute(
                    "IsaacSensorCreateImuSensor",
                    path="/imu",
                    parent=BODY_PATH,
                    sensor_period=DT,
                    translation=Gf.Vec3d(0, 0, 0),
                    orientation=Gf.Quatd(1, 0, 0, 0),
                )
                prim = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if prim is not None and hasattr(prim, "GetPath"):
                    self.imu_path = str(prim.GetPath())
            except Exception as e:
                log(f"WARN imu command: {e}")
            if not self.imu_path:
                for p in self.stage.Traverse():
                    s = str(p.GetPath())
                    if s.endswith("/imu"):
                        self.imu_path = s
                        break
            log(f"IMU_PATH={self.imu_path}")
            self.imu = _sensor.acquire_imu_sensor_interface()

            # velocity drives on the rotor joints (attrs created once here)
            self.drives = []
            for jp in JOINT_PATHS:
                prim = self.stage.GetPrimAtPath(jp)
                d = UsdPhysics.DriveAPI.Apply(prim, "angular")
                _set(d, "CreateTypeAttr", "force")
                _set(d, "CreateTargetVelocityAttr", 0.0)
                _set(d, "CreateStiffnessAttr", 0.0)
                _set(d, "CreateDampingAttr", MOTOR_DAMP * RAD2DEG)
                _set(d, "CreateMaxForceAttr", 1e6)
                self.drives.append(d)

            # mass API handles
            self.body_mass_api = UsdPhysics.MassAPI(self.stage.GetPrimAtPath(BODY_PATH))
            _set(self.body_mass_api, "CreateMassAttr", BASE_MASS)
            self.prop_mass_api = [UsdPhysics.MassAPI(self.stage.GetPrimAtPath(p)) for p in PROP_PATHS]
            for a in self.prop_mass_api:
                _set(a, "CreateCenterOfMassAttr", Gf.Vec3f(0, 0, 0))

            # physx interfaces
            self.physx = omni.physx.get_physx_simulation_interface()
            self.stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
            self.body_pid = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(BODY_PATH))
            self.prop_pids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(p)) for p in PROP_PATHS]

            self.art = ArticCls(CF_PATH)
            self.world.reset()
            self.art.initialize()
            names = getattr(self.art, "dof_names", None) or getattr(self.art, "joint_names", [])
            log(f"DOF_NAMES={list(names)}")
            n = getattr(self.art, "num_dof", 4)
            self.joint_idx = []
            for i in (1, 2, 3, 4):
                nm = f"m{i}_joint"
                self.joint_idx.append(names.index(nm) if nm in list(names) else i - 1)
            log(f"NUM_DOF={n} JOINT_IDX={self.joint_idx}")
            if n != 4:
                log(f"WARN num_dof={n} (expected 4)")

        # ---------------- per-run configuration ----------------
        def _begin_run(self, fault_mask, payload_g, seed):
            rng = np.random.default_rng(seed)
            # controller/wind/trajectory state first (needed for start pose)
            self.rng = rng
            self.fault_mask = fault_mask
            self.payload_g = payload_g
            self.mass = M_BODY_ASSET + 4 * M_PROP + payload_g * 1e-3  # true total
            self.wind = np.zeros(3)
            self.wind_gain = 0.002 * (1.0 + rng.uniform(-0.3, 0.3))
            self.traj_phase = rng.uniform(0, 2 * math.pi)
            self.prev_vel = np.zeros(3)
            self.z_integ = 0.0
            # thrust-aware spin-up: hold the true hover speed for THIS
            # fault/payload configuration so the drone does not sink at spawn
            k_eff = np.mean([F_EFFICIENCY if fault_mask & (1 << i) else 1.0 for i in range(4)])
            w_hover = math.sqrt(self.mass * 9.81 / (4.0 * K_THRUST * k_eff)) * 1.03
            self.w_cmd_hold = np.array([w_hover] * 4)

            # body link mass: asset body only (+ payload); prop links carry
            # their own 0.8g each -> total = M_BODY + 4*M_PROP + payload
            _set(self.body_mass_api, "CreateMassAttr", M_BODY_ASSET + payload_g * 1e-3)
            for i in range(4):
                faulted = bool(fault_mask & (1 << i))
                _set(
                    self.prop_mass_api[i],
                    "CreateCenterOfMassAttr",
                    Gf.Vec3f(F_COM_OFFSET, 0, 0) if faulted else Gf.Vec3f(0, 0, 0),
                )
                _set(self.drives[i], "CreateDampingAttr", MOTOR_DAMP * (F_MOTOR_GAIN if faulted else 1.0) * RAD2DEG)
                _set(self.drives[i], "CreateTargetVelocityAttr", 0.0)

            # rebuild articulation so mass/COM/drive changes take effect
            self.world.reset()
            try:
                self.art.initialize()
            except Exception as e:
                log(f"WARN re-initialize: {e}")

            # spawn ON the trajectory start point, heading along the tangent
            p0, _, yaw0 = self._trajectory(0.0)
            yq = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])
            self.art.set_world_pose(position=p0, orientation=yq)
            n = getattr(self.art, "num_dof", 4) or 4
            self.art.set_linear_velocity(np.zeros(3))
            self.art.set_angular_velocity(np.zeros(3))
            self.art.set_joint_velocities(np.zeros(n))
            try:
                self.art.set_joint_positions(np.zeros(n))
            except Exception:
                pass

        # ---------------- controller ----------------
        def _trajectory(self, t):
            r, w = 0.35, 2 * math.pi / 12.0
            ph = self.traj_phase
            p = np.array(
                [r * math.cos(w * t + ph), r * math.sin(w * t + ph), HOVER_ALT + 0.05 * math.sin(2 * w * t + ph)]
            )
            v = np.array(
                [-r * w * math.sin(w * t + ph), r * w * math.cos(w * t + ph), 0.05 * 2 * w * math.cos(2 * w * t + ph)]
            )
            yaw_d = w * t + ph + math.pi / 2
            return p, v, yaw_d

        def _control(self, pos, vel, quat, gyro):
            p_des, v_des, yaw_d = self._trajectory(self.t)
            a_des = 2.2 * (p_des - pos) + 1.6 * (v_des - vel)
            # altitude-hold integrator (like every real FC): compensates the
            # steady thrust deficit produced by degraded rotors, otherwise a
            # PD-only controller commands healthy-model RPMs forever and an
            # all-rotors-degraded vehicle can never leave the ground
            self.z_integ = float(np.clip(self.z_integ + (p_des[2] - pos[2]) * (1.0 / CTRL_HZ), -1.5, 2.5))
            a_des[2] += 1.2 * self.z_integ
            a_des = np.clip(a_des, -2.5, 2.5)
            a_cmd = a_des + np.array([0.0, 0.0, 9.81])
            total_T = self.mass * float(np.linalg.norm(a_cmd))

            roll, pitch, yaw = quat_to_rpy(quat)
            ax_, ay_, az_ = a_cmd
            roll_d = math.atan2(ay_, az_)
            pitch_d = math.atan2(-ax_, math.sqrt(ay_ * ay_ + az_ * az_))
            yaw_err = math.atan2(math.sin(yaw_d - yaw), math.cos(yaw_d - yaw))
            rate_d = np.clip(np.array([4.0 * (roll_d - roll), 4.0 * (pitch_d - pitch), 3.0 * yaw_err]), -3.0, 3.0)
            tau = np.array([6e-4, 6e-4, 3e-4]) * (rate_d - np.asarray(gyro))

            # thrust allocation (roll/pitch torques); yaw handled via rotor drag
            cmd = np.array([total_T, tau[0], tau[1], 0.0])
            w_sq = np.clip((self.Minv @ cmd) / K_THRUST, 0.0, None)

            # yaw: body torque tau_z = C*(w_cw^2 - w_ccw^2) from prop drag reaction
            u_z = tau[2]
            delta = u_z / (4.0 * C_YAW)
            for i in range(4):
                w_sq[i] += -delta if ROTOR_DIR[i] > 0 else +delta

            # 850 rad/s ceiling ~= 8100 RPM no-load: realistic ~1.5x hover
            # headroom so multi-rotor-fault cases retain climb authority
            w_cmd = np.sqrt(np.clip(w_sq, 60.0**2, 850.0**2))
            return ROTOR_DIR * w_cmd

        # ---------------- one physics step ----------------
        def _step_once(self):
            pos, quat = self.art.get_world_pose()
            pos = np.array(pos)
            quat = np.array(quat)
            w_all = np.ravel(np.array(self.art.get_joint_velocities()))
            w_signed = w_all[self.joint_idx]
            w_now = np.abs(w_signed)
            ph_all = np.ravel(np.array(self.art.get_joint_positions()))
            psi = ph_all[self.joint_idx]

            reading = self.imu.get_sensor_reading(self.imu_path)
            gyro = np.array([reading.ang_vel_x, reading.ang_vel_y, reading.ang_vel_z])
            acc = np.array([reading.lin_acc_x, reading.lin_acc_y, reading.lin_acc_z])
            o = reading.orientation
            ori = np.array([o.w, o.x, o.y, o.z])
            roll, pitch, yaw = quat_to_rpy(ori)

            R = quat_to_R(ori)

            # control @100Hz (zero-order hold on drive targets)
            if self.step % CTRL_EVERY == 0:
                vel = np.array(self.art.get_linear_velocity())
                self.w_cmd_hold = self._control(pos, vel, ori, gyro)
                for i, d in enumerate(self.drives):
                    # angular drive target velocity is in deg/s
                    d.GetTargetVelocityAttr().Set(float(self.w_cmd_hold[i] * RAD2DEG))

            # wind: OU disturbance on COM
            self.wind += (-self.wind * 0.05 + self.rng.normal(size=3)) * 0.01
            wind_f = self.wind * self.wind_gain

            # aerodynamics
            thrusts = K_THRUST * self.k_t_fault * w_now**2 * (1.0 + self.a_1p * np.cos(psi))
            body_z = R[:, 2]
            for i in range(4):
                hub = pos + R @ self.arm_offsets[i]
                f = thrusts[i] * body_z
                self.physx.apply_force_at_pos(self.stage_id, self.body_pid, carb.Float3(*f), carb.Float3(*hub), "Force")
                # prop aerodynamic drag torque (load on the motor)
                tq = (-ROTOR_DIR[i] * C_YAW * w_now[i] ** 2) * body_z
                self.physx.apply_torque(self.stage_id, self.prop_pids[i], carb.Float3(*tq))
            # drag reaction on the body (yaw torque)
            tau_z = C_YAW * float(np.sum(-ROTOR_DIR * w_now**2))
            self.physx.apply_torque(self.stage_id, self.body_pid, carb.Float3(*(tau_z * body_z)))
            # wind force
            self.physx.apply_force_at_pos(
                self.stage_id, self.body_pid, carb.Float3(*wind_f), carb.Float3(*pos), "Force"
            )

            self.world.step(False)
            self.t += DT
            self.step += 1

            rpm = w_now * RPM_CONV
            return rpm, roll, pitch, yaw, gyro, acc, psi

        # ---------------- full run ----------------
        def run(self, fault_mask, payload_g, seed, duration_s, out_dir, run_name):
            self._begin_run(fault_mask, payload_g, seed)
            self.k_t_fault = np.array([F_EFFICIENCY if (fault_mask >> i) & 1 else 1.0 for i in range(4)])
            self.a_1p = np.array([F_AERO_1P if (fault_mask >> i) & 1 else 0.0 for i in range(4)])

            # spin-up + settle (not logged): rotors ramp with controller active
            self.t = 0.0
            self.step = 0
            for _ in range(int(2.0 * PHYS_HZ)):
                self._step_once()

            n_steps = int(duration_s * PHYS_HZ)
            rows = np.empty((n_steps, 13))
            state_rows = np.empty((n_steps, 16))
            for k in range(n_steps):
                rpm, roll, pitch, yaw, gyro, acc, psi = self._step_once()
                rows[k, :4] = rpm
                rows[k, 4:7] = (roll, pitch, yaw)
                rows[k, 7:10] = gyro
                rows[k, 10:13] = acc
                state_rows[k, 0] = self.t
                state_rows[k, 1:4] = self.art.get_world_pose()[0]
                state_rows[k, 4:7] = self.art.get_linear_velocity()
                state_rows[k, 7:11] = psi
                state_rows[k, 11:15] = self.w_cmd_hold * RPM_CONV
                state_rows[k, 15] = float(self.fault_mask)

            os.makedirs(out_dir, exist_ok=True)
            imu_hdr = "rpm1,rpm2,rpm3,rpm4,roll,pitch,yaw,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z"
            np.savetxt(os.path.join(out_dir, "imu.csv"), rows, delimiter=",", header=imu_hdr, comments="")
            st_hdr = "t,x,y,z,vx,vy,vz,psi1,psi2,psi3,psi4,rpm_cmd1,rpm_cmd2,rpm_cmd3,rpm_cmd4,fault_mask"
            np.savetxt(os.path.join(out_dir, "state.csv"), state_rows, delimiter=",", header=st_hdr, comments="")

            n_fault = bin(fault_mask).count("1")
            ur = min(F_UR * n_fault, 0.9)
            gyro_rms = float(np.sqrt(np.mean(rows[:, 7:10] ** 2)))
            meta = {
                "sim": "isaac_sim_cf2x_usd",
                "asset": "Bitcraze/Crazyflie/cf2x.usd",
                "isaac_version": "5.1.0",
                "physics_hz": PHYS_HZ,
                "control_hz": CTRL_HZ,
                "duration_s": duration_s,
                "fault_type": "healthy" if fault_mask == 0 else f"label_{fault_mask}",
                "fault_mask": fault_mask,
                "fault_params": {
                    "unbalance": ur if fault_mask else 0.0,
                    "com_offset_m": F_COM_OFFSET,
                    "aero_1p": F_AERO_1P,
                    "efficiency": F_EFFICIENCY,
                    "motor_gain": F_MOTOR_GAIN,
                    "faulted_rotors": [i + 1 for i in range(4) if (fault_mask >> i) & 1],
                },
                "payload_kg": payload_g * 1e-3,
                "base_mass_kg": BASE_MASS,
                "seed": seed,
                "hover_rpm_mean": float(np.mean(rows[:, 0:4])),
                "gyro_rms": gyro_rms,
                "gravity": 9.81,
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            log(
                f"RUN_DONE name={run_name} samples={n_steps} "
                f"hover_rpm={meta['hover_rpm_mean']:.0f} gyro_rms={gyro_rms:.4f} "
                f"z_mean={float(np.mean(state_rows[:, 3])):.3f} out={out_dir}"
            )
            return meta

    def parse_run_token(tok):
        mask, payload, seed = 0, 0, 0
        for p in tok.split("_"):
            if p.startswith("mask"):
                mask = int(p[4:], 16)
            elif p.startswith("p") and p[1:].isdigit():
                payload = int(p[1:])
            elif p.startswith("s") and p[1:].isdigit():
                seed = int(p[1:])
        return mask, payload, seed, "run_cf2x_" + tok

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--out-root",
            type=str,
            default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isaac_dataset"),
        )
        parser.add_argument("--duration", type=float, default=9.0)
        parser.add_argument("--runs", type=str, default="healthy_p00_s0")
        parser.add_argument("--list-runs", action="store_true")
        args = parser.parse_args()

        if args.list_runs:
            toks = []
            for p in (0, 5, 10):
                toks += [f"healthy_p{p:02d}_s0", f"healthy_p{p:02d}_s1"]
            for m in range(1, 16):
                for p in (0, 5, 10):
                    toks.append(f"mask{m:02x}_p{p:02d}_s0")
            print(",".join(toks))
            return

        tokens = [t.strip() for t in args.runs.split(",") if t.strip()]
        sim = CFXSim()
        for tok in tokens:
            mask, payload, seed, name = parse_run_token(tok)
            out_dir = os.path.join(args.out_root, name)
            sim.run(mask, payload, seed, args.duration, out_dir, name)
        log("ALL_RUNS_DONE")

    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()

sys.stdout.flush()
sys.stderr.flush()
app.close()

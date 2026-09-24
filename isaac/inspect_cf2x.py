"""Inspect the Bitcraze Crazyflie 2.1 (cf2x.usd) drone asset: links, joints, drives, masses."""

import sys
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def log(msg):
    print(msg)
    sys.stdout.flush()


try:
    from pxr import Usd, UsdPhysics, PhysxSchema

    STAGE_PATH = "/home/rhutvik/uav-fault-diagnostics-ros2/isaac/assets/Bitcraze/Crazyflie/cf2x.usd"
    stage = Usd.Stage.Open(STAGE_PATH)
    log("STAGE_OPENED")

    log("=== TOP-LEVEL PRIMS ===")
    for child in stage.GetPseudoRoot().GetChildren():
        log(f"TOP: {child.GetPath().pathString} type={child.GetTypeName()}")

    log("=== ALL PHYSICS PRIMS (walk) ===")

    def walk(prim, depth=0):
        kind = prim.GetTypeName() or ""
        api = ""
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api += " [RigidBody]"
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api += " [ArticulationRoot]"
        if prim.HasAPI(UsdPhysics.MassAPI):
            mapi = UsdPhysics.MassAPI(prim)
            api += f" [Mass={mapi.GetMassAttr().Get()} Rho={mapi.GetDensityAttr().Get()} COM={mapi.GetCenterOfMassAttr().Get()}]"
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            api += f" [Joint b0={j.GetBody0Rel().GetTargets()} b1={j.GetBody1Rel().GetTargets()}]"
        if kind or api:
            log("  " * depth + f"{prim.GetPath().pathString}  ({kind}){api}")
        for c in prim.GetChildren():
            walk(c, depth + 1)

    for child in stage.GetPseudoRoot().GetChildren():
        walk(child, 1)

    log("=== JOINTS + DRIVES ===")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            log(f"JOINT: {prim.GetPath().pathString} type={prim.GetTypeName()}")
            j = UsdPhysics.Joint(prim)
            log(f"  body0={j.GetBody0Rel().GetTargets()} body1={j.GetBody1Rel().GetTargets()}")
            if prim.HasAPI(PhysxSchema.PhysxDriveAPI):
                d = PhysxSchema.PhysxDriveAPI(prim)
                log(f"  PhysxDrive: type={d.GetDriveTypeAttr().Get()} stiffness={d.GetDriveStiffnessAttr().Get()} "
                    f"damping={d.GetDriveDampingAttr().Get()} maxVel={d.GetMaxJointVelocityAttr().Get()}")
            if prim.HasAPI(UsdPhysics.DriveAPI):
                d = UsdPhysics.DriveAPI(prim)
                try:
                    log(f"  UsdDrive type={d.GetDriveTypeAttr().Get()}")
                except AttributeError:
                    pass

    log("=== RIGID BODIES ===")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            log(f"BODY: {prim.GetPath().pathString} mass={m.GetMassAttr().Get()} "
                f"density={m.GetDensityAttr().Get()} com={m.GetCenterOfMassAttr().Get()} "
                f"inertia_diag={m.GetDiagonalInertiaAttr().Get()}")
            rb = UsdPhysics.RigidBodyAPI(prim)
            log(f"      kinematic={rb.GetKinematicEnabledAttr().Get()} "
                f"disableGravity={rb.GetDisableGravityAttr().Get()}")

    log("=== REVOLUTE JOINT AXES ===")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            log(f"REVOLUTE: {prim.GetPath().pathString} axis={rj.GetAxisAttr().Get()} "
                f"limLo={rj.GetLowerLimitAttr().Get()} limHi={rj.GetUpperLimitAttr().Get()}")

    log("INSPECT_DONE")
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()

sys.stdout.flush()
sys.stderr.flush()
app.close()

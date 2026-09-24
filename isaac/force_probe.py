"""Probe: verify apply_force_at_pos / apply_torque call pattern on a cube and
on the cf2x body link. Measures whether forces actually move bodies."""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def log(m):
    print(m)
    sys.stdout.flush()


try:
    import carb
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import Sdf, UsdUtils
    from pxr import PhysicsSchemaTools

    import numpy as np

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 250.0)
    stage = omni.usd.get_context().get_stage()

    # --- test 1: cube ---
    cube = world.scene.add(DynamicCuboid(prim_path="/World/cube",
                                         scale=[0.1, 0.1, 0.1],
                                         mass=1.0, position=np.array([0, 0, 1.0])))

    ref_prim = stage.DefinePrim("/World/cf2x", "Xform")
    ref_prim.GetReferences().AddReference(
        "/home/rhutvik/uav-fault-diagnostics-ros2/isaac/assets/Bitcraze/Crazyflie/cf2x.usd")
    body = stage.GetPrimAtPath("/World/cf2x/body")

    from isaacsim.core.prims import SingleArticulation as ArticCls
    art = ArticCls("/World/cf2x")
    world.reset()
    art.initialize()
    art.set_world_pose(position=np.array([0.0, 0.0, 1.0]),
                       orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    physx = omni.physx.get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cube_pid = PhysicsSchemaTools.sdfPathToInt(Sdf.Path("/World/cube"))
    body_pid = PhysicsSchemaTools.sdfPathToInt(Sdf.Path("/World/cf2x/body"))
    log(f"STAGE_ID={stage_id} CUBE_PID={cube_pid} BODY_PID={body_pid}")

    log("API_METHODS=" + str([m for m in dir(physx) if "apply" in m.lower()]))

    # apply 5 N up to cube, 0.5 N up to cf2x body, step 1 s, report z
    z0_cube = float(cube.get_world_pose()[0][2])
    z0_body = float(np.array(art.get_world_pose()[0])[2])
    for i in range(250):
        physx.apply_force_at_pos(stage_id, cube_pid,
                                 carb.Float3(0, 0, 5.0), carb.Float3(0, 0, 0), "Force")
        physx.apply_force_at_pos(stage_id, body_pid,
                                 carb.Float3(0, 0, 0.5), carb.Float3(0, 0, 0), "Force")
        world.step(False)
    z1_cube = float(cube.get_world_pose()[0][2])
    z1_body = float(np.array(art.get_world_pose()[0])[2])
    log(f"CUBE z {z0_cube:.3f} -> {z1_cube:.3f} (expect ~+2.5 if force works)")
    log(f"BODY z {z0_body:.3f} -> {z1_body:.3f} (expect upward drift if force works)")
    log("PROBE_DONE")
except Exception:
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()

sys.stdout.flush()
sys.stderr.flush()
app.close()

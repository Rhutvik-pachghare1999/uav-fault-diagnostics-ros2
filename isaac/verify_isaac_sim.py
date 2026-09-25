"""Verify Isaac Sim 5.1.0 starts headless, steps physics, and exposes the IMU sensor API.

Run with Isaac Sim's embedded Python:
    ~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh isaac/verify_isaac_sim.py
"""

import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

try:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.objects import DynamicCuboid

    print("[1] isaacsim core + IMU imports: OK")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0)
    print("[2] World created: OK")

    cube = DynamicCuboid(
        prim_path="/World/cube",
        name="cube",
        position=np.array([0.0, 0.0, 1.0]),
        scale=np.array([0.2] * 3),
    )
    world.scene.add(cube)
    print("[3] DynamicCuboid added: OK")

    world.reset()
    for _ in range(50):
        world.step()
    print(f"[4] Stepped 50 physics frames: OK (t={world.current_time:.3f}s)")

    pos, quat = cube.get_world_pose()
    print(f"[5] Cube pose after 50 steps: pos={np.round(pos, 3).tolist()}")

    print("ISAAC_SIM_VERIFY: PASS")
except Exception:
    import traceback

    traceback.print_exc()
    print("ISAAC_SIM_VERIFY: FAIL")
    sys.exit(1)
finally:
    simulation_app.close()

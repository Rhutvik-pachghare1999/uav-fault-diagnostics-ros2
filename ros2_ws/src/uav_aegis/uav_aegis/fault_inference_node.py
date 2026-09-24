"""UAV Fault Inference ROS2 Node - package entry point.

Uses the build-time vendored copy of scripts/ros2_inference_node.py
(see setup.py vendor_scripts). No repo-relative sys.path hacks: the entry
point works from the colcon INSTALL tree on any machine.
"""

from uav_aegis.vendor.ros2_inference_node import main

if __name__ == "__main__":
    main()

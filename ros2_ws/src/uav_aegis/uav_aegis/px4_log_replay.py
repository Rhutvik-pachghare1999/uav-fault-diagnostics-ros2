"""PX4 Log Replay - package entry point.

Uses the build-time vendored copy of scripts/px4_log_replay.py
(see setup.py vendor_scripts). Works from the colcon INSTALL tree.
"""

from uav_aegis.vendor.px4_log_replay import main

if __name__ == "__main__":
    main()

"""PX4 Log Replay - Package Entry Point"""

import sys
from pathlib import Path

# Add the project scripts to path
project_root = Path(__file__).resolve().parents[3]
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from px4_log_replay import main

if __name__ == "__main__":
    main()
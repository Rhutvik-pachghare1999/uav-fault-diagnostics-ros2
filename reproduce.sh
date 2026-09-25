#!/usr/bin/env bash
# ============================================================================
# UAV Aegis - One-Command Reproduction Script
# ============================================================================
# This script reproduces the entire UAV fault diagnostics pipeline:
#   1. Environment setup
#   2. Real physics data generation (Isaac Sim)
#   3. Dataset building + SEALED split protocol (frozen manifest)
#   4. Model training (sealed 13-ch CNN + IMU-only 9-ch CNN)
#   5. Evaluation (sealed test split, TRUE LOSO retraining, uncertainty, benchmark)
#   6. Report figures
#   7. ROS2 package build + offline replay check
#   8. Tests
#
# Usage:
#   ./reproduce.sh [--skip-deps] [--skip-train] [--quick]
#
# Options:
#   --skip-deps    Skip dependency installation (assumes already installed)
#   --skip-train   Skip model training (use pre-trained weights if available)
#   --quick        Quick mode: fewer epochs, smaller dataset
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SKIP_DEPS=false
SKIP_TRAIN=false
QUICK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-deps) SKIP_DEPS=true ;;
        --skip-train) SKIP_TRAIN=true ;;
        --quick) QUICK_MODE=true ;;
        -h|--help)
            echo "Usage: $0 [--skip-deps] [--skip-train] [--quick]"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  UAV Aegis - End-to-End Reproduction Pipeline${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# ============================================================================
# Step 0: Environment Check
# ============================================================================
echo -e "${YELLOW}[0/9] Environment Check${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"

# Check for ROS2
if command -v ros2 &> /dev/null; then
    echo "ROS2: $(ros2 --version 2>&1 | head -1)"
else
    echo -e "${YELLOW}ROS2: Not found (will use jazzy from /opt/ros/jazzy)${NC}"
fi

# ============================================================================
# Step 1: Install Dependencies
# ============================================================================
if [ "$SKIP_DEPS" = false ]; then
    echo -e "\n${YELLOW}[1/9] Installing Python Dependencies${NC}"
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    # NOTE: rclpy and ROS message packages are provided by the ROS 2
    # installation itself (sourced from /opt/ros/jazzy/setup.sh below) and
    # must NOT be pip-installed - a pip copy can shadow the ROS one.
else
    echo -e "${YELLOW}[1/9] Skipping Dependency Installation${NC}"
fi

# ============================================================================
# Step 2: Generate Real Flight Data (Isaac Sim physics, Crazyflie 2.x asset)
# ============================================================================
echo -e "\n${YELLOW}[2/9] Generating Real Physics Flight Data${NC}"

# Full real-data generation requires Isaac Sim (natively-headless physics).
#   python.sh = Isaac Sim's bundled python launcher, e.g.:
#   ~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh
PYTHON_SH="${PYTHON_SH:-$HOME/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh}"

if command -v "$PYTHON_SH" >/dev/null 2>&1; then
    # A fresh clone contains run_* directories with meta.json only (CSVs are
    # gitignored), so checking for the DIRECTORIES would wrongly "skip
    # generation". Check for actual IMU data instead.
    N_RUNS_WITH_DATA=$(ls isaac_dataset/run_cf2x_*/imu.csv 2>/dev/null | wc -l)
    if [ "$N_RUNS_WITH_DATA" -ge 96 ]; then
        echo "Real runs already present ($N_RUNS_WITH_DATA runs with imu.csv), skipping generation..."
    elif ls -d isaac_dataset/run_cf2x_* >/dev/null 2>&1; then
        echo "WARNING: run directories exist but only $N_RUNS_WITH_DATA have imu.csv"
        echo "  (fresh clone: meta.json is tracked, CSVs are gitignored). Regenerating..."
        RUNS=$(python3 - <<'EOF'
toks = []
for p in (0, 5, 10):
    toks += [f'healthy_p{p:02d}_s0', f'healthy_p{p:02d}_s1']
for m in range(1, 16):
    for p in (0, 5, 10):
        toks += [f'mask{m:02x}_p{p:02d}_s0', f'mask{m:02x}_p{p:02d}_s1']
print(','.join(toks))
EOF
)
        PYTHONUNBUFFERED=1 "$PYTHON_SH" isaac/cf2x_fault_sim.py --duration 9 --runs "$RUNS"
    else
        echo "Generating 96 real physics runs (16 classes x 6 runs: 3 payloads x 2 seeds)..."
        RUNS=$(python3 - <<'EOF'
toks = []
for p in (0, 5, 10):
    toks += [f'healthy_p{p:02d}_s0', f'healthy_p{p:02d}_s1']
for m in range(1, 16):
    for p in (0, 5, 10):
        toks += [f'mask{m:02x}_p{p:02d}_s0', f'mask{m:02x}_p{p:02d}_s1']
print(','.join(toks))
EOF
)
        PYTHONUNBUFFERED=1 "$PYTHON_SH" isaac/cf2x_fault_sim.py --duration 9 --runs "$RUNS"
    fi
    N_RUNS_WITH_DATA=$(ls isaac_dataset/run_cf2x_*/imu.csv 2>/dev/null | wc -l)
    if [ "$N_RUNS_WITH_DATA" -lt 96 ]; then
        echo -e "${RED}ERROR: expected 96 runs with imu.csv, found $N_RUNS_WITH_DATA${NC}"
        exit 1
    fi
else
    echo "WARNING: Isaac Sim not found at $PYTHON_SH — cannot generate real data."
    echo "  Install Isaac Sim 5.1.0, or download the pre-generated dataset (see README)."
    exit 1
fi

# ============================================================================
# Step 3: Build ML Dataset (real runs only)
# ============================================================================
echo -e "\n${YELLOW}[3/9] Building ML Dataset from Real Runs${NC}"

if [ "$QUICK_MODE" = true ]; then
    WINDOW=100
    STEP=20
else
    WINDOW=100
    STEP=4
fi

python3 scripts/build_ml_dataset_v2.py --project-root . --out ml_dataset_v2.h5 --window $WINDOW --step $STEP \
    --vars "rpm1,rpm2,rpm3,rpm4,roll,pitch,yaw,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z"

# ============================================================================
# Step 4: Sealed Split Protocol (manifest + sealed dataset)
# ============================================================================
echo -e "\n${YELLOW}[4/9] Building Frozen Split Manifest + Sealed Dataset${NC}"

# The manifest is the single source of truth for the run-level split
# (48 train / 16 val / 32 test runs, stratified 3/1/2 per fault class,
# payload coverage enforced). The sealed dataset stores augmented copies
# of TRAIN runs only — val/test runs have zero augmented copies, so
# window-level leakage is structurally impossible.
python3 scripts/make_split_manifest.py --out splits/split_manifest.json

python3 scripts/build_sealed_dataset.py \
    --h5 ml_dataset_v2.h5 \
    --manifest splits/split_manifest.json \
    --out ml_sealed.h5 \
    --aug-copies 2 --aug-seed 42

# ============================================================================
# Step 5: Train Models (sealed protocol only)
# ============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n${YELLOW}[5/9] Training Models (sealed protocol)${NC}"

    if [ "$QUICK_MODE" = true ]; then
        EPOCHS=10
    else
        EPOCHS=50
    fi

    echo "Training CNN classifier, 13 channels (epochs=$EPOCHS)..."
    python3 scripts/train_cnn.py --h5 ml_sealed.h5 --manifest splits/split_manifest.json \
        --out models/cnn_sealed.pth --epochs $EPOCHS --history results/train_history_sealed.json

    echo "Training IMU-only CNN, 9 channels (no RPM telemetry needed)..."
    python3 scripts/train_cnn.py --h5 ml_sealed.h5 --manifest splits/split_manifest.json \
        --out models/cnn_imu9.pth --epochs $EPOCHS \
        --vars-subset "roll,pitch,yaw,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z" \
        --history results/train_history_imu9.json
else
    echo -e "${YELLOW}[5/9] Skipping Model Training${NC}"
fi

# ============================================================================
# Step 6: Evaluate Models (test split only — never training windows)
# ============================================================================
echo -e "\n${YELLOW}[6/9] Evaluating Models${NC}"

echo "Running classification evaluation (sealed test split, originals only)..."
python3 scripts/eval_classifier.py --h5 ml_sealed.h5 --model models/cnn_sealed.pth --out results/eval_sealed

echo "Running IMU-only model evaluation..."
python3 scripts/eval_classifier.py --h5 ml_sealed.h5 --model models/cnn_imu9.pth --out results/eval_imu9

if [ "$QUICK_MODE" = true ]; then
    echo "Skipping cross-speed LOSO retraining in --quick mode (it retrains one model per RPM bin)."
else
    echo "Running TRUE leave-one-RPM-bin-out retraining (fresh model per speed bin)..."
    python3 scripts/eval_cross_speed.py --dataset ml_sealed.h5 --retrain --output results/cross_speed_loso.json
fi

echo "Running uncertainty quantification (sealed test originals only)..."
python3 scripts/uncertainty_quantification.py --model models/cnn_sealed.pth --dataset ml_sealed.h5 \
    --n-samples 4000 --n-ood 2000 --mc-samples 10 --output results/uncertainty_sealed.json

echo "Running benchmark..."
python3 scripts/benchmarks/run_benchmark.py

# ============================================================================
# Step 6b: Report Figures
# ============================================================================
echo "Generating report figures..."
python3 scripts/make_report_plots.py --run-root isaac_dataset --results results --out-dir results/figures
echo "Generating architecture diagrams..."
python3 scripts/make_architecture_diagrams.py --out-dir results/figures

# ============================================================================
# Step 7: Test ROS2 Package
# ============================================================================
echo -e "\n${YELLOW}[7/9] Building ROS2 Package${NC}"

if [ -f /opt/ros/jazzy/setup.sh ] && [ "$QUICK_MODE" = false ]; then
    # Build ROS2 package (only in full mode)
    echo "Building ROS2 package (this may take a moment)..."
    cd ros2_ws
    source /opt/ros/jazzy/setup.sh 2>/dev/null || true
    colcon build --packages-select uav_aegis --event-handlers console_direct+ 2>&1 | tail -20
    cd ..
else
    echo -e "${YELLOW}Skipping ROS2 build (quick mode or ROS2 not available)${NC}"
fi

# Test offline replay mode (doesn't require ROS2 running) and ASSERT the
# modal prediction on the healthy run is the healthy class. Without this
# check a broken model/normalization path ships silently (the tracked
# results/replay_test.csv was once all label_14 on this healthy input).
echo "Testing offline replay mode..."
python3 scripts/ros2_inference_node.py --model models/cnn_imu9.pth --mode replay \
    --input isaac_dataset/run_cf2x_healthy_p00_s0/imu.csv --output results/replay_test.csv \
    --window 100 --step 10
REPLAY_MODAL=$(python3 -c "
import csv, collections
with open('results/replay_test.csv') as f:
    rows = list(csv.DictReader(f))
modal = collections.Counter(r['fault_id'] for r in rows).most_common(1)[0]
print(f'{modal[0]} {modal[1]}/{len(rows)}')
")
echo "Replay modal prediction: $REPLAY_MODAL"
if [ "$(echo "$REPLAY_MODAL" | cut -d' ' -f1)" != "0" ]; then
    echo -e "${RED}FAIL: modal prediction on the healthy replay run is not fault_id 0 (healthy).${NC}"
    exit 1
fi

# ============================================================================
# Step 8: Run Tests
# ============================================================================
echo -e "\n${YELLOW}[8/9] Running Tests${NC}"
pytest tests/ -v

# ============================================================================
# Step 9: Launch Dashboard (Optional)
# ============================================================================
echo -e "\n${YELLOW}[9/9] Pipeline Complete!${NC}"
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  Reproduction Pipeline Completed Successfully!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Results available in:"
echo "  - results/eval_sealed/           : Classification eval (sealed test split)"
echo "  - results/eval_imu9/             : IMU-only model eval (sealed test split)"
echo "  - results/cross_speed_loso.json  : TRUE cross-speed LOSO retraining"
echo "  - results/uncertainty_sealed.json: Uncertainty + OOD (sealed test only)"
echo "  - results/benchmark_*.csv       : Latency benchmarks"
echo "  - models/cnn_sealed.pth          : Trained 13-channel CNN"
echo "  - models/cnn_imu9.pth            : Trained IMU-only 9-channel CNN"
echo ""
echo "To launch the dashboard:"
echo "  streamlit run scripts/dashboard.py"
echo ""
echo "To run ROS2 inference node (IMU-only model needs no RPM telemetry):"
echo "  source /opt/ros/jazzy/setup.sh"
echo "  ros2 run uav_aegis fault_inference_node --model models/cnn_imu9.pth --mode live"
echo ""
echo "  13-channel model (needs RPM telemetry):"
echo "  ros2 run uav_aegis fault_inference_node --model models/cnn_sealed.pth --mode live --rpm-topic /rotor_rpms"
echo ""
echo "To replay a PX4 flight log (.ulg) or an IMU CSV:"
echo "  python3 scripts/px4_log_replay.py --mode ros2 <log.ulg> --model models/cnn_imu9.pth"
echo "  python3 scripts/px4_log_replay.py --mode offline <imu.csv> --model models/cnn_imu9.pth --output results/px4_replay.csv"
echo ""
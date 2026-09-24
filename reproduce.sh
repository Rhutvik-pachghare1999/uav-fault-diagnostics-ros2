#!/usr/bin/env bash
# ============================================================================
# UAV Aegis - One-Command Reproduction Script
# ============================================================================
# This script reproduces the entire UAV fault diagnostics pipeline:
#   1. Environment setup
#   2. Synthetic data generation
#   3. Dataset building & augmentation
#   4. Model training (CNN + LSTM)
#   5. Evaluation (cross-speed, uncertainty, benchmark)
#   6. Dashboard launch
#   7. ROS2 package build
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
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Install ROS2 Python packages if ROS2 is available
    if [ -f /opt/ros/jazzy/setup.sh ]; then
        source /opt/ros/jazzy/setup.sh
        pip install rclpy rosidl-runtime-py sensor-msgs std-msgs 2>/dev/null || true
    fi
else
    echo -e "\n${YELLOW}[1/9] Skipping Dependency Installation${NC}"
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
    if ls isaac_dataset/run_cf2x_* >/dev/null 2>&1; then
        echo "Real runs already present, skipping generation..."
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
# Step 4: Augment Dataset
# ============================================================================
echo -e "\n${YELLOW}[4/9] Augmenting Dataset${NC}"

if [ "$QUICK_MODE" = true ]; then
    AUG_TIMES=2
else
    AUG_TIMES=3
fi

python3 scripts/augment_dataset.py --in ml_dataset_v2.h5 --out ml_dataset_v2_aug.h5 --times $AUG_TIMES

# ============================================================================
# Step 5: Train Models
# ============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n${YELLOW}[5/9] Training Models${NC}"
    
    if [ "$QUICK_MODE" = true ]; then
        EPOCHS=10
        BATCH_SIZE=64
    else
        EPOCHS=50
        BATCH_SIZE=32
    fi
    
    echo "Training CNN classifier (epochs=$EPOCHS)..."
    python3 scripts/train_cnn.py --h5 ml_dataset_v2_aug.h5 --out models/cnn_multi.pth --epochs $EPOCHS --batch-size $BATCH_SIZE --lr 1e-3
    
    echo "Self-supervised pre-training on healthy data..."
    python3 scripts/self_supervised_pretrain.py --h5 ml_dataset_v2_aug.h5 --out models/encoder_pretrained.pth --epochs 20 --batch-size 128
else
    echo -e "\n${YELLOW}[5/9] Skipping Model Training${NC}"
fi

# ============================================================================
# Step 6: Evaluate Models
# ============================================================================
echo -e "\n${YELLOW}[6/9] Evaluating Models${NC}"

echo "Running classification evaluation..."
python3 scripts/eval_classifier.py --h5 ml_dataset_v2_aug.h5 --model models/cnn_multi.pth --out results/eval_final --max-samples 10000

echo "Running cross-speed robustness evaluation..."
python3 scripts/eval_cross_speed.py --model models/cnn_multi.pth --dataset ml_dataset_v2_aug.h5 --n-speed-bins 4

echo "Running uncertainty quantification..."
python3 scripts/uncertainty_quantification.py --model models/cnn_multi.pth --dataset ml_dataset_v2_aug.h5 --n-samples 500 --mc-samples 10

echo "Running benchmark..."
python3 scripts/benchmarks/run_benchmark.py

# ============================================================================
# Step 6b: Report Figures
# ============================================================================
echo "Generating report figures..."
python3 scripts/make_report_plots.py --run-root isaac_dataset --results results --out-dir results/figures

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

# Test offline replay mode (doesn't require ROS2 running)
echo "Testing offline replay mode..."
python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth --mode replay --input isaac_dataset/run_cf2x_healthy_p00_s0/imu.csv --output results/replay_test.csv --window 100 --step 10

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
echo "  - results/eval_final/           : Classification evaluation"
echo "  - results/cross_speed_eval.json : Cross-speed robustness"
echo "  - results/uncertainty_eval.json : Uncertainty quantification"
echo "  - results/benchmark_*.csv       : Latency benchmarks"
echo "  - models/cnn_multi.pth          : Trained CNN model"
echo "  - models/encoder_pretrained.pth : Pre-trained encoder"
echo ""
echo "To launch the dashboard:"
echo "  streamlit run scripts/dashboard.py"
echo ""
echo "To run ROS2 inference node:"
echo "  source /opt/ros/jazzy/setup.sh"
echo "  ros2 run uav_aegis fault_inference_node --model models/cnn_multi.pth --mode live"
echo ""
echo "To replay a PX4 log file:"
echo "  python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth --mode replay --input <log.csv>"
echo ""
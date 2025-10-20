#!/bin/bash
# Master script to launch all three domain trainings in parallel
#
# This script:
# 1. Launches three training processes in background
# 2. Starts centralized TensorBoard for monitoring
# 3. Provides status monitoring and log tailing
#
# Usage:
#   bash scripts/launch_parallel_training.sh

set -e

echo "========================================"
echo "NSM-25: Launching Parallel Training"
echo "========================================"
echo ""

# Configuration
CAUSAL_DIR="/Users/preston/Projects/nsm-causal"
PLANNING_DIR="/Users/preston/Projects/nsm-planning"
KG_DIR="/Users/preston/Projects/nsm-kg"

# Check directories exist
for dir in "$CAUSAL_DIR" "$PLANNING_DIR" "$KG_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Directory not found: $dir"
        exit 1
    fi
done

echo "✓ All worktree directories found"
echo ""

# Create master logs directory
MASTER_LOG_DIR="logs/parallel_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

echo "Master logs: $MASTER_LOG_DIR"
echo ""

# Function to launch training in background
launch_training() {
    local domain=$1
    local dir=$2
    local log_file="$MASTER_LOG_DIR/${domain}_training.log"

    echo "Launching ${domain} training..."

    (
        cd "$dir"
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate nsm
        bash experiments/run_full_training.sh --use-tensorboard 2>&1 | tee "$log_file"
        echo "COMPLETE" > "$MASTER_LOG_DIR/${domain}_status.txt"
    ) &

    local pid=$!
    echo "$pid" > "$MASTER_LOG_DIR/${domain}_pid.txt"
    echo "  → PID: $pid"
    echo "  → Log: $log_file"
    echo ""
}

# Launch all three trainings
echo "========================================"
echo "Launching Training Processes"
echo "========================================"
echo ""

launch_training "causal" "$CAUSAL_DIR"
launch_training "planning" "$PLANNING_DIR"
launch_training "kg" "$KG_DIR"

# Wait for TensorBoard logs to be created
echo "Waiting for TensorBoard logs to initialize (10 seconds)..."
sleep 10

# Launch centralized TensorBoard
echo ""
echo "========================================"
echo "Launching Centralized TensorBoard"
echo "========================================"
echo ""

TENSORBOARD_LOG="$MASTER_LOG_DIR/tensorboard.log"

tensorboard \
    --logdir_spec=causal:${CAUSAL_DIR}/checkpoints/causal_full,planning:${PLANNING_DIR}/checkpoints/planning_full,kg:${KG_DIR}/checkpoints/kg_full \
    --port 6006 \
    --bind_all \
    2>&1 | tee "$TENSORBOARD_LOG" &

TENSORBOARD_PID=$!
echo "$TENSORBOARD_PID" > "$MASTER_LOG_DIR/tensorboard_pid.txt"

echo "TensorBoard launched!"
echo "  → PID: $TENSORBOARD_PID"
echo "  → URL: http://localhost:6006"
echo "  → Log: $TENSORBOARD_LOG"
echo ""

# Create monitoring script
cat > "$MASTER_LOG_DIR/monitor.sh" << 'EOF'
#!/bin/bash
# Monitor training progress

MASTER_LOG_DIR=$(dirname "$0")

echo "========================================"
echo "NSM-25 Training Status"
echo "========================================"
echo ""

# Check each domain
for domain in causal planning kg; do
    pid_file="$MASTER_LOG_DIR/${domain}_pid.txt"
    status_file="$MASTER_LOG_DIR/${domain}_status.txt"

    if [ -f "$status_file" ] && [ "$(cat $status_file)" = "COMPLETE" ]; then
        echo "✅ ${domain}: COMPLETE"
    elif [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "🔄 ${domain}: RUNNING (PID: $pid)"
            # Show last line of log
            log_file="$MASTER_LOG_DIR/${domain}_training.log"
            if [ -f "$log_file" ]; then
                echo "   Latest: $(tail -n 1 $log_file | cut -c 1-80)"
            fi
        else
            echo "❌ ${domain}: STOPPED (check logs)"
        fi
    else
        echo "⏳ ${domain}: NOT STARTED"
    fi
    echo ""
done

echo "========================================"
echo "TensorBoard: http://localhost:6006"
echo "========================================"
EOF

chmod +x "$MASTER_LOG_DIR/monitor.sh"

# Create shutdown script
cat > "$MASTER_LOG_DIR/shutdown.sh" << 'EOF'
#!/bin/bash
# Gracefully shutdown all training processes

MASTER_LOG_DIR=$(dirname "$0")

echo "Shutting down all training processes..."
echo ""

# Stop training processes
for domain in causal planning kg; do
    pid_file="$MASTER_LOG_DIR/${domain}_pid.txt"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping ${domain} (PID: $pid)..."
            kill $pid
        fi
    fi
done

# Stop TensorBoard
tensorboard_pid_file="$MASTER_LOG_DIR/tensorboard_pid.txt"
if [ -f "$tensorboard_pid_file" ]; then
    pid=$(cat "$tensorboard_pid_file")
    if ps -p $pid > /dev/null 2>&1; then
        echo "Stopping TensorBoard (PID: $pid)..."
        kill $pid
    fi
fi

echo ""
echo "All processes stopped."
EOF

chmod +x "$MASTER_LOG_DIR/shutdown.sh"

# Print summary
echo ""
echo "========================================"
echo "Parallel Training Launched Successfully!"
echo "========================================"
echo ""
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 Logs: $MASTER_LOG_DIR"
echo ""
echo "Monitoring Commands:"
echo "  → Watch status: bash $MASTER_LOG_DIR/monitor.sh"
echo "  → Tail causal: tail -f $MASTER_LOG_DIR/causal_training.log"
echo "  → Tail planning: tail -f $MASTER_LOG_DIR/planning_training.log"
echo "  → Tail kg: tail -f $MASTER_LOG_DIR/kg_training.log"
echo ""
echo "Control Commands:"
echo "  → Check status: bash $MASTER_LOG_DIR/monitor.sh"
echo "  → Shutdown all: bash $MASTER_LOG_DIR/shutdown.sh"
echo ""
echo "⏱️  Expected runtime: 4-6 hours (CPU) or 1-2 hours (GPU)"
echo ""
echo "========================================"

# Keep monitoring in foreground
echo ""
echo "Press Ctrl+C to detach (training continues in background)"
echo "Or wait here for status updates every 5 minutes..."
echo ""

# Monitor loop
while true; do
    sleep 300  # 5 minutes
    echo ""
    echo "=== Status Update: $(date) ==="
    bash "$MASTER_LOG_DIR/monitor.sh"
done

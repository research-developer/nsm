#!/bin/bash
# Launch NSM-25 parallel training in persistent tmux sessions
#
# Creates 4 tmux sessions:
#   - nsm-causal: Causal domain training
#   - nsm-planning: Planning domain training
#   - nsm-kg: Knowledge graph training
#   - nsm-tensorboard: TensorBoard server
#
# Usage:
#   bash scripts/launch_tmux_training.sh
#
# Reconnect to sessions:
#   tmux attach -t nsm-causal
#   tmux attach -t nsm-planning
#   tmux attach -t nsm-kg
#   tmux attach -t nsm-tensorboard
#
# List all sessions:
#   tmux ls
#
# Kill all sessions:
#   tmux kill-session -t nsm-causal
#   tmux kill-session -t nsm-planning
#   tmux kill-session -t nsm-kg
#   tmux kill-session -t nsm-tensorboard

set -e

echo "========================================"
echo "NSM-25: Launching Training in tmux"
echo "========================================"
echo ""

# Configuration
CAUSAL_DIR="/Users/preston/Projects/nsm-causal"
PLANNING_DIR="/Users/preston/Projects/nsm-planning"
KG_DIR="/Users/preston/Projects/nsm-kg"

# Create master logs directory
MASTER_LOG_DIR="logs/parallel_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

echo "Master logs: $MASTER_LOG_DIR"
echo ""

# Function to launch training in tmux session
launch_tmux_training() {
    local session_name=$1
    local domain=$2
    local dir=$3

    echo "Creating tmux session: $session_name"

    # Create tmux session in detached mode with zsh
    tmux new-session -d -s "$session_name" -c "$dir"

    # Set default shell to zsh if available
    if command -v zsh &> /dev/null; then
        tmux send-keys -t "$session_name" "export SHELL=/bin/zsh" C-m
    fi

    # Activate conda and run training
    tmux send-keys -t "$session_name" "source ~/miniconda3/etc/profile.d/conda.sh" C-m
    tmux send-keys -t "$session_name" "conda activate nsm" C-m
    tmux send-keys -t "$session_name" "echo '========================================'" C-m
    tmux send-keys -t "$session_name" "echo 'NSM-25: ${domain} Training'" C-m
    tmux send-keys -t "$session_name" "echo '========================================'" C-m
    tmux send-keys -t "$session_name" "echo ''" C-m
    tmux send-keys -t "$session_name" "echo 'Session: $session_name'" C-m
    tmux send-keys -t "$session_name" "echo 'Working directory: $dir'" C-m
    tmux send-keys -t "$session_name" "echo 'Log directory: $MASTER_LOG_DIR'" C-m
    tmux send-keys -t "$session_name" "echo ''" C-m
    tmux send-keys -t "$session_name" "echo 'To reconnect: tmux attach -t $session_name'" C-m
    tmux send-keys -t "$session_name" "echo 'To detach: Ctrl+B then D'" C-m
    tmux send-keys -t "$session_name" "echo '========================================'" C-m
    tmux send-keys -t "$session_name" "echo ''" C-m
    tmux send-keys -t "$session_name" "bash experiments/run_full_training.sh --use-tensorboard 2>&1 | tee $MASTER_LOG_DIR/${domain}_training.log" C-m

    echo "  ✓ Session created: $session_name"
    echo "  → Reconnect: tmux attach -t $session_name"
    echo ""
}

# Kill any existing sessions
echo "Checking for existing sessions..."
for session in nsm-causal nsm-planning nsm-kg nsm-tensorboard; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "  Killing existing session: $session"
        tmux kill-session -t "$session"
    fi
done
echo ""

# Launch training sessions
echo "========================================"
echo "Launching Training Sessions"
echo "========================================"
echo ""

launch_tmux_training "nsm-causal" "causal" "$CAUSAL_DIR"
launch_tmux_training "nsm-planning" "planning" "$PLANNING_DIR"
launch_tmux_training "nsm-kg" "kg" "$KG_DIR"

# Wait for TensorBoard directories to be created
echo "Waiting for TensorBoard directories (15 seconds)..."
sleep 15

# Launch TensorBoard in tmux
echo "Creating TensorBoard session: nsm-tensorboard"

tmux new-session -d -s "nsm-tensorboard"

if command -v zsh &> /dev/null; then
    tmux send-keys -t "nsm-tensorboard" "export SHELL=/bin/zsh" C-m
fi

tmux send-keys -t "nsm-tensorboard" "cd /Users/preston/Projects/NSM" C-m
tmux send-keys -t "nsm-tensorboard" "echo '========================================'" C-m
tmux send-keys -t "nsm-tensorboard" "echo 'NSM-25: TensorBoard Dashboard'" C-m
tmux send-keys -t "nsm-tensorboard" "echo '========================================'" C-m
tmux send-keys -t "nsm-tensorboard" "echo ''" C-m
tmux send-keys -t "nsm-tensorboard" "echo 'URL: http://localhost:6006'" C-m
tmux send-keys -t "nsm-tensorboard" "echo 'Session: nsm-tensorboard'" C-m
tmux send-keys -t "nsm-tensorboard" "echo ''" C-m
tmux send-keys -t "nsm-tensorboard" "echo 'To reconnect: tmux attach -t nsm-tensorboard'" C-m
tmux send-keys -t "nsm-tensorboard" "echo 'To detach: Ctrl+B then D'" C-m
tmux send-keys -t "nsm-tensorboard" "echo '========================================'" C-m
tmux send-keys -t "nsm-tensorboard" "echo ''" C-m
tmux send-keys -t "nsm-tensorboard" "tensorboard --logdir_spec=causal:${CAUSAL_DIR}/checkpoints/causal_full,planning:${PLANNING_DIR}/checkpoints/planning_full,kg:${KG_DIR}/checkpoints/kg_full --port 6006 --bind_all 2>&1 | tee $MASTER_LOG_DIR/tensorboard.log" C-m

echo "  ✓ Session created: nsm-tensorboard"
echo "  → Reconnect: tmux attach -t nsm-tensorboard"
echo "  → URL: http://localhost:6006"
echo ""

# Create monitoring script
cat > "$MASTER_LOG_DIR/reconnect_info.txt" << EOF
NSM-25 Parallel Training - tmux Session Information
=====================================================

TMUX SESSIONS:
  - nsm-causal      : Causal domain training
  - nsm-planning    : Planning domain training
  - nsm-kg          : Knowledge graph training
  - nsm-tensorboard : TensorBoard dashboard

RECONNECT COMMANDS:
  tmux attach -t nsm-causal
  tmux attach -t nsm-planning
  tmux attach -t nsm-kg
  tmux attach -t nsm-tensorboard

LIST ALL SESSIONS:
  tmux ls

SWITCH BETWEEN SESSIONS:
  Ctrl+B then S (shows session list)
  Use arrow keys to select, Enter to attach

DETACH FROM SESSION:
  Ctrl+B then D (detaches without killing)

KILL SPECIFIC SESSION:
  tmux kill-session -t nsm-causal
  tmux kill-session -t nsm-planning
  tmux kill-session -t nsm-kg
  tmux kill-session -t nsm-tensorboard

KILL ALL NSM SESSIONS:
  tmux kill-session -t nsm-causal && \\
  tmux kill-session -t nsm-planning && \\
  tmux kill-session -t nsm-kg && \\
  tmux kill-session -t nsm-tensorboard

TENSORBOARD:
  URL: http://localhost:6006
  Session: nsm-tensorboard

LOG FILES:
  Causal: $MASTER_LOG_DIR/causal_training.log
  Planning: $MASTER_LOG_DIR/planning_training.log
  KG: $MASTER_LOG_DIR/kg_training.log
  TensorBoard: $MASTER_LOG_DIR/tensorboard.log

MONITORING:
  # Watch live logs
  tail -f $MASTER_LOG_DIR/causal_training.log
  tail -f $MASTER_LOG_DIR/planning_training.log
  tail -f $MASTER_LOG_DIR/kg_training.log

  # Check session status
  tmux ls

  # See running sessions
  ps aux | grep -E "train_(causal|planning|kg)"

EXPECTED RUNTIME:
  4-6 hours (CPU) or 1-2 hours (GPU)

NOTES:
  - Sessions persist even if you disconnect from SSH
  - Training continues in background
  - Safe to close terminal - reconnect anytime
  - Use Ctrl+B then D to detach (NOT Ctrl+C which kills)
EOF

cat > "$MASTER_LOG_DIR/quick_status.sh" << 'EOFSCRIPT'
#!/bin/bash
# Quick status check for all training sessions

MASTER_LOG_DIR=$(dirname "$0")

echo "========================================"
echo "NSM-25 tmux Session Status"
echo "========================================"
echo ""

# Check tmux sessions
echo "Active tmux sessions:"
if tmux ls 2>/dev/null | grep -E "nsm-(causal|planning|kg|tensorboard)"; then
    echo ""
else
    echo "  No NSM sessions found"
    exit 0
fi

echo ""
echo "========================================"
echo "Latest Progress"
echo "========================================"
echo ""

# Check each log for latest line
for domain in causal planning kg; do
    log_file="$MASTER_LOG_DIR/${domain}_training.log"
    if [ -f "$log_file" ]; then
        echo "[$domain]"
        tail -n 3 "$log_file" | head -n 3
        echo ""
    fi
done

echo "========================================"
echo "Reconnect Commands"
echo "========================================"
echo "  tmux attach -t nsm-causal"
echo "  tmux attach -t nsm-planning"
echo "  tmux attach -t nsm-kg"
echo "  tmux attach -t nsm-tensorboard"
echo ""
echo "TensorBoard: http://localhost:6006"
echo "========================================"
EOFSCRIPT

chmod +x "$MASTER_LOG_DIR/quick_status.sh"

# Save connection info for easy access
echo "$MASTER_LOG_DIR" > /tmp/nsm_latest_run.txt

# Print final summary
echo ""
echo "========================================"
echo "✅ All Sessions Launched Successfully!"
echo "========================================"
echo ""
echo "📊 TENSORBOARD:"
echo "   URL: http://localhost:6006"
echo "   Session: tmux attach -t nsm-tensorboard"
echo ""
echo "🔧 TRAINING SESSIONS:"
echo "   Causal:   tmux attach -t nsm-causal"
echo "   Planning: tmux attach -t nsm-planning"
echo "   KG:       tmux attach -t nsm-kg"
echo ""
echo "📋 QUICK COMMANDS:"
echo "   List all:     tmux ls"
echo "   Status:       bash $MASTER_LOG_DIR/quick_status.sh"
echo "   Reconnect:    tmux attach -t <session-name>"
echo "   Detach:       Ctrl+B then D"
echo ""
echo "📁 LOGS & INFO:"
echo "   Directory: $MASTER_LOG_DIR"
echo "   Info file: $MASTER_LOG_DIR/reconnect_info.txt"
echo ""
echo "⏱️  EXPECTED RUNTIME:"
echo "   4-6 hours (CPU) or 1-2 hours (GPU)"
echo ""
echo "💡 TIP: Sessions persist even if you disconnect!"
echo "   Your training will continue running in background."
echo "   Reconnect anytime with: tmux attach -t <session>"
echo ""
echo "========================================"
echo ""
echo "Press Enter to continue (or Ctrl+C to exit)..."
echo ""

# Show current sessions
echo "Current tmux sessions:"
tmux ls 2>/dev/null | grep -E "nsm-" || echo "Error listing sessions"
echo ""

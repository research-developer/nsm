# NSM Cross-Domain Testing Makefile
# Orchestrates testing across parallel worktree branches

.PHONY: help test-all test-causal test-kg test-planning clean-all push-all setup-env

# Worktree paths
CAUSAL_DIR := ../nsm-causal
KG_DIR := ../nsm-kg
PLANNING_DIR := ../nsm-planning

# Python & pytest
PYTHON := python
PYTEST := pytest
PYTEST_FLAGS := -v --tb=short

# Conda environment
CONDA_ENV := nsm

help:
	@echo "NSM Cross-Domain Testing Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all       - Run tests across all three domains"
	@echo "  make test-causal    - Run tests in Causal branch"
	@echo "  make test-kg        - Run tests in KG branch"
	@echo "  make test-planning  - Run tests in Planning branch"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean-all      - Clean generated files in all branches"
	@echo "  make push-all       - Push all branches to remote"
	@echo "  make status-all     - Show git status for all branches"
	@echo ""
	@echo "Environment:"
	@echo "  make setup-env      - Check conda environment setup"
	@echo ""
	@echo "Worktree paths:"
	@echo "  Causal:   $(CAUSAL_DIR)"
	@echo "  KG:       $(KG_DIR)"
	@echo "  Planning: $(PLANNING_DIR)"

# Test all domains in sequence
test-all:
	@echo "========================================="
	@echo "Testing Causal Domain"
	@echo "========================================="
	@cd $(CAUSAL_DIR) && $(PYTEST) tests/data/test_causal_dataset.py $(PYTEST_FLAGS)
	@echo ""
	@echo "========================================="
	@echo "Testing Knowledge Graph Domain"
	@echo "========================================="
	@cd $(KG_DIR) && $(PYTEST) tests/data/test_kg_dataset.py $(PYTEST_FLAGS)
	@echo ""
	@echo "========================================="
	@echo "Testing Planning Domain"
	@echo "========================================="
	@cd $(PLANNING_DIR) && $(PYTEST) tests/data/test_planning_dataset.py $(PYTEST_FLAGS)
	@echo ""
	@echo "========================================="
	@echo "✅ All domain tests complete!"
	@echo "========================================="

# Individual domain tests
test-causal:
	@echo "Testing Causal domain..."
	@cd $(CAUSAL_DIR) && $(PYTEST) tests/data/test_causal_dataset.py $(PYTEST_FLAGS)

test-kg:
	@echo "Testing Knowledge Graph domain..."
	@cd $(KG_DIR) && $(PYTEST) tests/data/test_kg_dataset.py $(PYTEST_FLAGS)

test-planning:
	@echo "Testing Planning domain..."
	@cd $(PLANNING_DIR) && $(PYTEST) tests/data/test_planning_dataset.py $(PYTEST_FLAGS)

# Clean generated files
clean-all:
	@echo "Cleaning Causal branch..."
	@cd $(CAUSAL_DIR) && rm -rf logs/*.log checkpoints/*/ results/*/ data/causal/processed/ || true
	@echo "Cleaning KG branch..."
	@cd $(KG_DIR) && rm -rf logs/*.log checkpoints/*/ results/*/ data/kg/processed/ || true
	@echo "Cleaning Planning branch..."
	@cd $(PLANNING_DIR) && rm -rf logs/*.log checkpoints/*/ results/*/ data/planning/processed/ || true
	@echo "✅ All branches cleaned!"

# Push all branches
push-all:
	@echo "Pushing Causal branch..."
	@cd $(CAUSAL_DIR) && git push origin dataset-causal
	@echo "Pushing KG branch..."
	@cd $(KG_DIR) && git push origin dataset-knowledge-graph
	@echo "Pushing Planning branch..."
	@cd $(PLANNING_DIR) && git push origin dataset-planning
	@echo "✅ All branches pushed!"

# Git status for all branches
status-all:
	@echo "========================================="
	@echo "Causal Branch Status"
	@echo "========================================="
	@cd $(CAUSAL_DIR) && git status --short
	@echo ""
	@echo "========================================="
	@echo "Knowledge Graph Branch Status"
	@echo "========================================="
	@cd $(KG_DIR) && git status --short
	@echo ""
	@echo "========================================="
	@echo "Planning Branch Status"
	@echo "========================================="
	@cd $(PLANNING_DIR) && git status --short

# Check environment setup
setup-env:
	@echo "Checking conda environment..."
	@conda env list | grep $(CONDA_ENV) || echo "❌ Conda environment '$(CONDA_ENV)' not found!"
	@echo ""
	@echo "Checking PyTorch Geometric installation..."
	@conda run -n $(CONDA_ENV) python -c "import torch; import torch_geometric; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ PyG: {torch_geometric.__version__}')" || echo "❌ PyG not installed!"
	@echo ""
	@echo "Checking worktree directories..."
	@test -d $(CAUSAL_DIR) && echo "✅ Causal worktree exists" || echo "❌ Causal worktree missing"
	@test -d $(KG_DIR) && echo "✅ KG worktree exists" || echo "❌ KG worktree missing"
	@test -d $(PLANNING_DIR) && echo "✅ Planning worktree exists" || echo "❌ Planning worktree missing"

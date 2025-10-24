#!/bin/bash
pass_count=0
fail_count=0

for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "Run $i:"
    result=$(pytest tests/data/test_planning_dataset.py::TestPlanningTripleDataset::test_split_independence --tb=no -q 2>&1)
    if echo "$result" | grep -q "1 passed"; then
        echo "  PASSED"
        pass_count=$((pass_count+1))
    else
        echo "  FAILED"
        echo "$result" | grep -E "AssertionError|assert"
        fail_count=$((fail_count+1))
    fi
done

echo ""
echo "Summary: $pass_count passed, $fail_count failed out of 10 runs"

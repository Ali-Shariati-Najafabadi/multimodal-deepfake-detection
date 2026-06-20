#!/bin/bash

echo "======================================================================"
echo "Phase 4: Testing All Fusion Strategies"
echo "======================================================================"

BASE_CONFIG="configs/phase4_gated_config.yaml"
OUTPUT_DIR="/data/shared/ali/deepfake-detection/results/phase4_fusion_comparison"
mkdir -p $OUTPUT_DIR

FUSION_STRATEGIES=("concat" "add" "multiply" "avg" "weighted" "cross_attention" "gated" "film")

for strategy in "${FUSION_STRATEGIES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Testing Fusion Strategy: $strategy"
    echo "======================================================================"
    
    CONFIG_FILE="configs/phase4_${strategy}_config.yaml"
    
    cp $BASE_CONFIG $CONFIG_FILE
    
    sed -i "s/type: \"gated\"/type: \"$strategy\"/" $CONFIG_FILE
    sed -i "s/phase4_gated/phase4_$strategy/" $CONFIG_FILE
    sed -i "s/Gated Fusion/$strategy Fusion/" $CONFIG_FILE
    
    python training/train.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR/$strategy
    
    echo "Completed: $strategy"
    echo "Results saved to: $OUTPUT_DIR/$strategy"
    echo ""
done

echo ""
echo "======================================================================"
echo "All Fusion Strategies Tested!"
echo "======================================================================"
echo "Results Directory: $OUTPUT_DIR"
echo ""
echo "Summary of Results:"
python scripts/compare_fusion_results.py --results_dir $OUTPUT_DIR

#!/bin/bash

echo "======================================================================"
echo "Phase 6: Testing All Audio Models"
echo "======================================================================"

BASE_CONFIG="configs/phase6_wavlm_config.yaml"
OUTPUT_DIR="/data/shared/ali/deepfake-detection/results/phase6_audio_comparison"
mkdir -p $OUTPUT_DIR

declare -A AUDIO_MODELS
AUDIO_MODELS=(
    ["hubert_base"]="hubert:facebook/hubert-base-ls960"
    ["hubert_large"]="hubert:facebook/hubert-large-ls960-ft"
    ["wavlm_base"]="wavlm:microsoft/wavlm-base"
    ["wavlm_large"]="wavlm:microsoft/wavlm-large"
    ["wav2vec2_base"]="wav2vec2:facebook/wav2vec2-base-960h"
    ["wav2vec2_large"]="wav2vec2:facebook/wav2vec2-large-960h"
)

for model_name in "${!AUDIO_MODELS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Testing Audio Model: $model_name"
    echo "======================================================================"
    
    IFS=':' read -r arch pretrained <<< "${AUDIO_MODELS[$model_name]}"
    
    CONFIG_FILE="configs/phase6_${model_name}_config.yaml"
    
    cp $BASE_CONFIG $CONFIG_FILE
    
    sed -i "s/architecture: \"wavlm\"/architecture: \"$arch\"/" $CONFIG_FILE
    sed -i "s|pretrained_model: \"microsoft/wavlm-base\"|pretrained_model: \"$pretrained\"|" $CONFIG_FILE
    sed -i "s/phase6_wavlm/phase6_$model_name/" $CONFIG_FILE
    sed -i "s/WavLM/$model_name/" $CONFIG_FILE
    
    python training/train.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR/$model_name
    
    echo "Completed: $model_name"
    echo "Results saved to: $OUTPUT_DIR/$model_name"
    echo ""
done

echo ""
echo "======================================================================"
echo "All Audio Models Tested!"
echo "======================================================================"
echo "Results Directory: $OUTPUT_DIR"
echo ""
echo "Summary of Results:"
python scripts/compare_audio_results.py --results_dir $OUTPUT_DIR

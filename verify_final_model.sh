#!/bin/bash
# Run the best model configuration 3 times and collect results

echo "======================================================================"
echo "VERIFYING BEST MODEL CONFIGURATION (3 runs)"
echo "======================================================================"
echo "Config: unified_best_final.yaml"
echo "Architecture: [384, 256, 128, 64, 32]"
echo "Dropout: 0.15, Weight Decay: 0.001, LR: 0.001, Epochs: 50"
echo "----------------------------------------------------------------------"

for i in 1 2 3; do
    echo -e "\nRun $i/3:"
    echo "----------------------------------------------------------------------"

    # Run model and extract key metrics
    output=$(python train.py --config-name=unified_best_final 2>&1)

    # Extract metrics
    avg_acc=$(echo "$output" | grep "Average Accuracy:" | awk '{print $3}')
    avg_loss=$(echo "$output" | grep "Average CE Loss:" | awk '{print $4}')

    attr_acc=$(echo "$output" | grep -A2 "ATTRACTIVE:" | grep "Accuracy:" | awk '{print $2}')
    smart_acc=$(echo "$output" | grep -A2 "SMART:" | grep "Accuracy:" | awk '{print $2}')
    trust_acc=$(echo "$output" | grep -A2 "TRUSTWORTHY:" | grep "Accuracy:" | awk '{print $2}')

    if [ ! -z "$avg_acc" ]; then
        echo "✓ Average Accuracy: $avg_acc"
        echo "  Average CE Loss: $avg_loss"
        echo "  Per-target accuracy:"
        echo "    Attractive: $attr_acc"
        echo "    Smart: $smart_acc"
        echo "    Trustworthy: $trust_acc"

        # Store for summary
        echo "$avg_acc" >> /tmp/accuracies.txt
        echo "$avg_loss" >> /tmp/losses.txt
    else
        echo "✗ Failed to get results"
    fi
done

echo -e "\n======================================================================"
echo "SUMMARY"
echo "======================================================================"

if [ -f /tmp/accuracies.txt ]; then
    # Calculate mean
    mean_acc=$(awk '{ sum += $1 } END { printf "%.4f", sum/NR }' /tmp/accuracies.txt)
    mean_loss=$(awk '{ sum += $1 } END { printf "%.4f", sum/NR }' /tmp/losses.txt)

    echo "Mean Accuracy across runs: $mean_acc"
    echo "Mean CE Loss across runs: $mean_loss"

    # Check if meets expected performance
    expected=0.73
    if (( $(echo "$mean_acc > $expected" | bc -l) )); then
        echo -e "\n✓ VERIFICATION SUCCESSFUL"
        echo "  Model consistently achieves ~73% accuracy"
    else
        echo -e "\n⚠ Performance below expected"
    fi

    # Clean up temp files
    rm -f /tmp/accuracies.txt /tmp/losses.txt
else
    echo "✗ No successful runs"
fi

echo "======================================================================
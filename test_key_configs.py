#!/usr/bin/env python3
"""
Test key promising configurations on actual data using train.py
"""

import subprocess
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key configurations to test
configs = [
    # 1. Baseline (current best)
    {
        'name': 'Baseline',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50'
    },

    # 2. No dropout with higher weight decay
    {
        'name': 'NoDropout_HighWD',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.dropout=0.0'
    },

    # 3. Lower dropout
    {
        'name': 'LowDropout',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.dropout=0.05'
    },

    # 4. Higher dropout
    {
        'name': 'HighDropout',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.dropout=0.2'
    },

    # 5. Layer normalization instead of batch
    {
        'name': 'LayerNorm',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.use_batchnorm=false model.use_layernorm=true'
    },

    # 6. No normalization
    {
        'name': 'NoNorm',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.use_batchnorm=false model.use_layernorm=false model.dropout=0.2'
    },

    # 7. Narrower architecture
    {
        'name': 'Narrow',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.hidden_dims=[256,128,64]'
    },

    # 8. Bottleneck architecture
    {
        'name': 'Bottleneck',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.hidden_dims=[512,128,32]'
    },

    # 9. Shallow network
    {
        'name': 'Shallow',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.hidden_dims=[512,128]'
    },

    # 10. Very deep and narrow
    {
        'name': 'DeepNarrow',
        'args': '--config-name=unified_best task_type=comparison training.epochs=50 model.hidden_dims=[384,256,128,64,32]'
    },
]

results = []

logger.info("="*70)
logger.info("TESTING KEY CONFIGURATIONS ON ACTUAL DATA")
logger.info("="*70)

for i, config in enumerate(configs):
    logger.info(f"\n[{i+1}/{len(configs)}] Testing: {config['name']}")
    logger.info(f"Args: {config['args']}")

    try:
        # Run train.py with configuration
        cmd = f"python train.py {config['args']}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # Parse output for accuracy
        output_lines = result.stdout.split('\n')
        avg_acc = None
        avg_loss = None

        for line in output_lines:
            if 'Average Accuracy:' in line:
                avg_acc = float(line.split(':')[1].strip())
            elif 'Average CE Loss:' in line:
                avg_loss = float(line.split(':')[1].strip())

        if avg_acc:
            results.append({
                'name': config['name'],
                'accuracy': avg_acc,
                'loss': avg_loss,
                'config': config['args']
            })
            logger.info(f"  ✓ Accuracy: {avg_acc:.3f}, Loss: {avg_loss:.3f}")
        else:
            logger.info(f"  ✗ Failed to parse results")

    except subprocess.TimeoutExpired:
        logger.info(f"  ✗ Timeout")
    except Exception as e:
        logger.info(f"  ✗ Error: {e}")

# Sort by accuracy
results.sort(key=lambda x: x['accuracy'], reverse=True)

# Report results
logger.info("\n" + "="*70)
logger.info("RESULTS SUMMARY")
logger.info("="*70)

for i, result in enumerate(results):
    logger.info(f"{i+1}. {result['name']}: {result['accuracy']:.3f} (loss: {result['loss']:.3f})")

if results:
    logger.info("\n" + "="*70)
    logger.info(f"BEST CONFIGURATION: {results[0]['name']}")
    logger.info(f"Accuracy: {results[0]['accuracy']:.3f}")
    logger.info(f"Config: {results[0]['config']}")
    logger.info("="*70)

# Save results
with open('key_config_results.json', 'w') as f:
    json.dump(results, f, indent=2)
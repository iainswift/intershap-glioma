"""
Master Script: Cross-Attention Synergy Experiment

This script runs the complete experiment to test whether explicit interaction
mechanisms can capture more cross-modal synergy than the standard MLP.

EXPERIMENT DESIGN:
1. Train 3 attention-based models (Cross-Attention, Bilinear, Gated)
2. Run InterSHAP analysis on all models
3. Compare to original MLP baseline (InterSHAP = 4.82%)

EXPECTED OUTCOMES:
A) InterSHAP jumps to 15-25% → "Synergy was hidden by architecture"
B) InterSHAP stays ~5% → "Biology is fundamentally additive"

Usage:
    python run_attention_experiment.py
    
    # Or with custom settings:
    python run_attention_experiment.py --epochs 100 --lr 5e-5
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Complete Attention Synergy Experiment')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs per model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only run analysis')
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
    
    print("\nCross-Attention Synergy Experiment")
    print("=" * 40)
    print(f"  Baseline MLP InterSHAP: 4.82%")
    print(f"  Models: cross_attention, bilinear, gated")
    
    models = ['cross_attention', 'bilinear', 'gated']
    
    if not args.skip_training:
        # Train each model
        for model in models:
            success = run_command(
                [python, os.path.join(script_dir, 'train_attention_models.py'),
                 '--model', model,
                 '--epochs', str(args.epochs),
                 '--lr', str(args.lr)],
                f"Training {model.upper()} model"
            )
            if not success:
                print(f"Stopping due to training failure for {model}")
                return
    else:
        print("\nSkipping training (--skip_training flag set)")
    
    # Run InterSHAP analysis
    run_command(
        [python, os.path.join(script_dir, 'analyze_attention_intershap.py')],
        "Running InterSHAP Analysis"
    )
    
    print("\nExperiment complete. Results saved to Intershap/ directory.")


if __name__ == '__main__':
    main()

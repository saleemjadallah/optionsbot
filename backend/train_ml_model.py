"""
Script to train the ML pricing model with proper configuration
"""

import sys
sys.path.append('.')

from models.ml_pricing import MLOptionsPricer, ModelConfig
import os

def train_ml_pricing_model():
    """Train the ML pricing model with production settings"""

    print("=" * 60)
    print("ML Options Pricing Model Training")
    print("=" * 60)

    # Check if we want to resume or start fresh
    model_path = "models/ml_pricer"
    checkpoint_exists = os.path.exists(f"{model_path}/checkpoint.pth")

    if checkpoint_exists:
        print(f"\n⚠️  Found existing checkpoint at {model_path}/checkpoint.pth")
        response = input("Do you want to start fresh training? (y/n): ").lower()
        if response == 'y':
            print("Starting fresh training...")
            # Clear existing files
            import shutil
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.makedirs(model_path, exist_ok=True)

    # Configuration for production training
    config = ModelConfig(
        epochs=50,  # Full training
        batch_size=512,
        learning_rate=0.001,
        early_stopping_patience=15,
        validation_split=0.2
    )

    print(f"\nTraining Configuration:")
    print(f"- Epochs: {config.epochs}")
    print(f"- Batch Size: {config.batch_size}")
    print(f"- Learning Rate: {config.learning_rate}")
    print(f"- Device: {config.device}")
    print(f"- Model Path: {model_path}")

    # Initialize pricer
    print("\nInitializing ML Options Pricer...")
    pricer = MLOptionsPricer(config, model_path=model_path)

    # Train model (will generate synthetic data automatically)
    print("\nStarting training (this may take 10-15 minutes)...")
    print("Generating 300,000 synthetic training samples...")

    try:
        training_stats = pricer.train(save_model=True)

        print("\n" + "=" * 60)
        print("✅ Training Completed Successfully!")
        print("=" * 60)

        print(f"\nTraining Results:")
        print(f"- Epochs Trained: {training_stats['epochs_trained']}")
        print(f"- Training MAE: ${training_stats['train_mae']:.6f}")
        print(f"- Validation MAE: ${training_stats['val_mae']:.6f}")
        print(f"- Best Validation Loss: {training_stats['best_val_loss']:.6f}")
        print(f"- Total Parameters: {training_stats['total_parameters']:,}")

        # Run benchmark
        print("\n" + "-" * 40)
        print("Running Benchmark vs Black-Scholes...")
        print("-" * 40)

        benchmark = pricer.benchmark_against_black_scholes(test_size=10000)

        print(f"\nBenchmark Results:")
        print(f"- Mean Absolute Error: ${benchmark['mae']:.6f}")
        print(f"- Root Mean Square Error: ${benchmark['rmse']:.6f}")
        print(f"- Relative Error: {benchmark['relative_error']:.4%}")
        print(f"- Speed Improvement: {benchmark['speed_improvement']:.1f}x faster")
        print(f"- ML Inference Time: {benchmark['ml_time']:.3f}s for {benchmark['test_samples']} samples")
        print(f"- Black-Scholes Time: {benchmark['bs_time']:.3f}s for {benchmark['test_samples']} samples")

        # Test predictions
        print("\n" + "-" * 40)
        print("Sample Predictions")
        print("-" * 40)

        # ATM Call Option
        call_price = pricer.predict_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
        print(f"\nATM Call (S=100, K=100, T=0.25, σ=0.2): ${call_price:.4f}")

        # ATM Put Option
        put_price = pricer.predict_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put')
        print(f"ATM Put  (S=100, K=100, T=0.25, σ=0.2): ${put_price:.4f}")

        print("\n" + "=" * 60)
        print("📊 Model saved successfully to:", model_path)
        print("=" * 60)

        return True

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial model saved as checkpoint.pth")
        return False

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_ml_pricing_model()
    if success:
        print("\n✅ Training completed! Model is ready for production use.")
    else:
        print("\n⚠️  Training incomplete. Please run again to complete training.")
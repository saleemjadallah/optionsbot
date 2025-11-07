"""Quick test script for ML pricing model"""

import sys
sys.path.append('.')

from models.ml_pricing import (
    MLOptionsPricer,
    ModelConfig,
    DataGenerator
)

def test_ml_pricing():
    """Quick test of the ML pricing model with reduced epochs"""

    # Configuration for quick testing
    config = ModelConfig(
        epochs=5,  # Reduced from 50 for quick testing
        batch_size=512,
        learning_rate=0.001
    )

    # Initialize pricer
    print("Initializing ML Options Pricer...")
    pricer = MLOptionsPricer(config, model_path="models/test_ml_pricer")

    # Generate smaller dataset for quick testing
    print("\nGenerating test data...")
    test_df = DataGenerator.generate_black_scholes_data(10000)  # Reduced from 300000

    # Train model
    print("\nTraining ML Options Pricer (5 epochs for quick test)...")
    training_stats = pricer.train(df=test_df, save_model=False)

    print(f"\nTraining completed!")
    print(f"- Epochs trained: {training_stats['epochs_trained']}")
    print(f"- Training MAE: {training_stats['train_mae']:.6f}")
    print(f"- Validation MAE: {training_stats['val_mae']:.6f}")
    print(f"- Model Parameters: {training_stats['total_parameters']:,}")

    # Test single prediction
    print("\n--- Testing Single Prediction ---")
    price = pricer.predict_price(
        S=100,     # Stock price
        K=100,     # Strike price
        T=0.25,    # Time to expiry (3 months)
        r=0.05,    # Risk-free rate
        sigma=0.2, # Volatility
        option_type='call'
    )
    print(f"Predicted call option price: ${price:.4f}")

    # Test batch prediction
    print("\n--- Testing Batch Prediction ---")
    import pandas as pd
    test_batch = pd.DataFrame({
        'S': [100, 100, 100],
        'K': [95, 100, 105],    # ITM, ATM, OTM
        'T': [0.25, 0.25, 0.25],
        'r': [0.05, 0.05, 0.05],
        'sigma': [0.2, 0.2, 0.2],
        'option_type': [1, 1, 1]  # All calls
    })

    batch_prices = pricer.predict_batch(test_batch)
    print("Strike prices: [95, 100, 105]")
    print(f"Predicted prices: {batch_prices}")
    print(f"- ITM (K=95): ${batch_prices[0]:.4f}")
    print(f"- ATM (K=100): ${batch_prices[1]:.4f}")
    print(f"- OTM (K=105): ${batch_prices[2]:.4f}")

    # Quick benchmark
    print("\n--- Quick Benchmark ---")
    benchmark = pricer.benchmark_against_black_scholes(test_size=1000)
    print(f"Mean Absolute Error: {benchmark['mae']:.6f}")
    print(f"Relative Error: {benchmark['relative_error']:.4%}")
    print(f"Speed Improvement: {benchmark['speed_improvement']:.1f}x")

    print("\n✅ ML Pricing Model test completed successfully!")

if __name__ == "__main__":
    test_ml_pricing()
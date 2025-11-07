"""
Merton Jump Diffusion Options Pricing Model
==========================================

A comprehensive implementation of the Merton Jump Diffusion model for options pricing
that captures sudden market movements through compound Poisson processes. This model
is particularly effective during earnings announcements and market crashes.

Features:
- Semi-closed form solution with 40 iteration truncation for 1% accuracy
- Automatic calibration using market data
- Parameter optimization with multiple algorithms
- Monte Carlo validation
- Batch pricing for entire options chains
- Performance benchmarking against Black-Scholes
- Production-ready inference with error handling
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize, differential_evolution, least_squares
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import time
import warnings
import joblib
from concurrent.futures import ThreadPoolExecutor
import math

warnings.filterwarnings('ignore')

@dataclass
class MertonParameters:
    """Parameters for Merton Jump Diffusion model"""
    sigma: float = 0.15          # Diffusion volatility
    lam: float = 1.0             # Jump intensity (jumps per year)
    mu_j: float = -0.1           # Mean jump size
    sigma_j: float = 0.2         # Jump volatility

    def __post_init__(self):
        """Validate parameter ranges"""
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.lam < 0:
            raise ValueError("lambda must be non-negative")
        if self.sigma_j <= 0:
            raise ValueError("sigma_j must be positive")

@dataclass
class CalibrationConfig:
    """Configuration for model calibration"""
    max_iterations: int = 1000
    tolerance: float = 1e-8
    method: str = 'differential_evolution'  # 'lm', 'differential_evolution', 'minimize'
    bounds: Dict = None

    def __post_init__(self):
        if self.bounds is None:
            self.bounds = {
                'sigma': (0.01, 2.0),
                'lam': (0.0, 10.0),
                'mu_j': (-1.0, 1.0),
                'sigma_j': (0.01, 1.0)
            }

class MertonJumpDiffusion:
    """
    Merton Jump Diffusion options pricing model

    The model assumes stock price follows:
    dS/S = (r - λk)dt + σdW + (e^J - 1)dN

    Where:
    - σ: volatility of continuous diffusion
    - λ: intensity of Poisson jumps
    - J ~ N(μ_j, σ_j²): jump size distribution
    - k = E[e^J - 1] = e^(μ_j + 0.5σ_j²) - 1
    """

    def __init__(self, max_iterations: int = 40, tolerance: float = 1e-10):
        """
        Initialize Merton Jump Diffusion model

        Parameters:
        - max_iterations: Maximum terms in infinite series (40 gives 1% accuracy)
        - tolerance: Convergence tolerance for series
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.parameters = None
        self.calibration_results = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MertonJumpDiffusion')

    def _jump_compensator(self, mu_j: float, sigma_j: float) -> float:
        """Calculate jump compensator k = E[e^J - 1]"""
        return np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def _compensated_drift(self, r: float, lam: float, mu_j: float, sigma_j: float) -> float:
        """Calculate compensated drift rate"""
        k = self._jump_compensator(mu_j, sigma_j)
        return r - lam * k

    def _adjusted_parameters(self, n_jumps: int, T: float, sigma: float,
                            mu_j: float, sigma_j: float, drift: float) -> Tuple[float, float]:
        """
        Calculate adjusted BS parameters conditional on n jumps

        Returns:
        - r_n: Adjusted risk-free rate
        - sigma_n: Adjusted volatility
        """
        r_n = drift + n_jumps * mu_j / T
        sigma_n = np.sqrt(sigma**2 + n_jumps * sigma_j**2 / T)
        return r_n, sigma_n

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str = 'call') -> float:
        """Standard Black-Scholes formula"""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        if sigma <= 0:
            return self._black_scholes_price(S, K, T, r, 1e-10, option_type)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0)  # Ensure non-negative

    def price_option(self, S: float, K: float, T: float, r: float,
                    params: MertonParameters, option_type: str = 'call') -> float:
        """
        Price option using Merton Jump Diffusion model

        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity (years)
        - r: Risk-free rate
        - params: Merton model parameters
        - option_type: 'call' or 'put'

        Returns:
        - Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        # Calculate compensated drift
        drift = self._compensated_drift(r, params.lam, params.mu_j, params.sigma_j)

        # Sum over possible number of jumps
        total_price = 0.0
        lambda_t = params.lam * T

        for n in range(self.max_iterations):
            # Poisson probability of n jumps
            poisson_prob = poisson.pmf(n, lambda_t)

            if poisson_prob < self.tolerance:
                break

            # Adjusted BS parameters for n jumps
            r_n, sigma_n = self._adjusted_parameters(
                n, T, params.sigma, params.mu_j, params.sigma_j, drift
            )

            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes_price(S, K, T, r_n, sigma_n, option_type)

            # Add weighted contribution
            contribution = poisson_prob * bs_price
            total_price += contribution

            # Early termination if contribution is negligible
            if contribution < self.tolerance and n > 5:
                break

        return total_price

    def price_batch(self, options_data: pd.DataFrame, params: MertonParameters) -> np.ndarray:
        """
        Price multiple options efficiently using vectorization where possible

        Parameters:
        - options_data: DataFrame with columns ['S', 'K', 'T', 'r', 'option_type']
        - params: Merton model parameters

        Returns:
        - Array of option prices
        """
        prices = np.zeros(len(options_data))

        # Use threading for CPU-bound computation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, row in options_data.iterrows():
                future = executor.submit(
                    self.price_option,
                    row['S'], row['K'], row['T'], row['r'],
                    params, row.get('option_type', 'call')
                )
                futures.append((i, future))

            # Collect results
            for i, future in futures:
                prices[i] = future.result()

        return prices

    def calibrate_to_market(self, market_data: pd.DataFrame,
                          config: CalibrationConfig = None) -> MertonParameters:
        """
        Calibrate model parameters to market prices

        Parameters:
        - market_data: DataFrame with columns ['S', 'K', 'T', 'r', 'market_price', 'option_type']
        - config: Calibration configuration

        Returns:
        - Calibrated parameters
        """
        if config is None:
            config = CalibrationConfig()

        self.logger.info(f"Starting calibration with {len(market_data)} market prices...")

        # Initial parameter guess
        initial_params = [0.15, 1.0, -0.1, 0.2]  # sigma, lambda, mu_j, sigma_j

        # Objective function
        def objective(x):
            try:
                params = MertonParameters(sigma=x[0], lam=x[1], mu_j=x[2], sigma_j=x[3])
                model_prices = self.price_batch(market_data, params)
                market_prices = market_data['market_price'].values

                # Mean squared error
                mse = np.mean((model_prices - market_prices)**2)
                return mse
            except Exception as e:
                self.logger.warning(f"Error in objective function: {e}")
                return 1e6  # Large penalty for invalid parameters

        # Parameter bounds
        bounds = [(config.bounds['sigma'][0], config.bounds['sigma'][1]),
                 (config.bounds['lam'][0], config.bounds['lam'][1]),
                 (config.bounds['mu_j'][0], config.bounds['mu_j'][1]),
                 (config.bounds['sigma_j'][0], config.bounds['sigma_j'][1])]

        # Optimization
        start_time = time.time()

        if config.method == 'differential_evolution':
            result = differential_evolution(
                objective, bounds, maxiter=config.max_iterations,
                tol=config.tolerance, seed=42, workers=1
            )
        elif config.method == 'minimize':
            result = minimize(
                objective, initial_params, bounds=bounds,
                method='L-BFGS-B', options={'maxiter': config.max_iterations}
            )
        elif config.method == 'lm':
            # Levenberg-Marquardt for least squares
            def residuals(x):
                try:
                    params = MertonParameters(sigma=x[0], lam=x[1], mu_j=x[2], sigma_j=x[3])
                    model_prices = self.price_batch(market_data, params)
                    market_prices = market_data['market_price'].values
                    return model_prices - market_prices
                except:
                    return np.ones(len(market_data)) * 1000

            result = least_squares(
                residuals, initial_params, bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                max_nfev=config.max_iterations
            )
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")

        calibration_time = time.time() - start_time

        # Extract optimal parameters
        optimal_params = MertonParameters(
            sigma=result.x[0], lam=result.x[1],
            mu_j=result.x[2], sigma_j=result.x[3]
        )

        # Calculate final metrics
        final_prices = self.price_batch(market_data, optimal_params)
        market_prices = market_data['market_price'].values

        rmse = np.sqrt(np.mean((final_prices - market_prices)**2))
        mae = np.mean(np.abs(final_prices - market_prices))
        relative_error = np.mean(np.abs((final_prices - market_prices) / market_prices))

        # Store calibration results
        self.calibration_results = {
            'parameters': asdict(optimal_params),
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'calibration_time': calibration_time,
            'optimization_success': result.success if hasattr(result, 'success') else True,
            'function_evaluations': result.nfev if hasattr(result, 'nfev') else result.get('nfev', 'N/A'),
            'method': config.method
        }

        self.parameters = optimal_params

        self.logger.info(f"Calibration completed in {calibration_time:.2f}s")
        self.logger.info(f"  RMSE: {rmse:.6f}")
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  Relative Error: {relative_error:.4%}")
        self.logger.info(f"  Parameters: σ={optimal_params.sigma:.4f}, λ={optimal_params.lam:.4f}, "
                        f"μ_j={optimal_params.mu_j:.4f}, σ_j={optimal_params.sigma_j:.4f}")

        return optimal_params

    def monte_carlo_validation(self, S: float, K: float, T: float, r: float,
                             params: MertonParameters, n_simulations: int = 100000,
                             option_type: str = 'call') -> Dict:
        """
        Validate analytical pricing against Monte Carlo simulation

        Returns:
        - Dictionary with validation results
        """
        self.logger.info(f"Running Monte Carlo validation with {n_simulations:,} simulations...")

        # Analytical price
        analytical_price = self.price_option(S, K, T, r, params, option_type)

        # Monte Carlo simulation
        dt = T / 252  # Daily steps
        n_steps = int(T / dt)

        # Pre-calculate jump parameters
        drift = self._compensated_drift(r, params.lam, params.mu_j, params.sigma_j)

        payoffs = np.zeros(n_simulations)

        for i in range(n_simulations):
            St = S

            for _ in range(n_steps):
                # Brownian motion component
                dW = np.random.normal(0, np.sqrt(dt))

                # Jump component
                n_jumps = np.random.poisson(params.lam * dt)
                jump_size = 0
                if n_jumps > 0:
                    jumps = np.random.normal(params.mu_j, params.sigma_j, n_jumps)
                    jump_size = np.sum(np.exp(jumps) - 1)

                # Update stock price
                St = St * np.exp((drift - 0.5 * params.sigma**2) * dt +
                               params.sigma * dW) * (1 + jump_size)

            # Calculate payoff
            if option_type.lower() == 'call':
                payoffs[i] = max(St - K, 0)
            else:
                payoffs[i] = max(K - St, 0)

        # Discount payoffs
        mc_price = np.exp(-r * T) * np.mean(payoffs)
        mc_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

        # Calculate validation metrics
        absolute_error = abs(analytical_price - mc_price)
        relative_error = absolute_error / analytical_price if analytical_price > 0 else np.inf

        validation_results = {
            'analytical_price': analytical_price,
            'monte_carlo_price': mc_price,
            'monte_carlo_std': mc_std,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'n_simulations': n_simulations,
            'within_2_std': abs(analytical_price - mc_price) < 2 * mc_std
        }

        self.logger.info(f"Validation Results:")
        self.logger.info(f"  Analytical: {analytical_price:.6f}")
        self.logger.info(f"  Monte Carlo: {mc_price:.6f} ± {mc_std:.6f}")
        self.logger.info(f"  Absolute Error: {absolute_error:.6f}")
        self.logger.info(f"  Relative Error: {relative_error:.4%}")

        return validation_results

    def benchmark_against_black_scholes(self, test_cases: pd.DataFrame) -> Dict:
        """
        Benchmark Merton model against Black-Scholes

        Parameters:
        - test_cases: DataFrame with test option specifications

        Returns:
        - Benchmark results
        """
        if self.parameters is None:
            # Use default parameters for benchmarking
            self.parameters = MertonParameters()

        # Time Merton pricing
        start_time = time.time()
        merton_prices = self.price_batch(test_cases, self.parameters)
        merton_time = time.time() - start_time

        # Time Black-Scholes pricing
        start_time = time.time()
        bs_prices = []
        for _, row in test_cases.iterrows():
            bs_price = self._black_scholes_price(
                row['S'], row['K'], row['T'], row['r'],
                self.parameters.sigma, row.get('option_type', 'call')
            )
            bs_prices.append(bs_price)
        bs_time = time.time() - start_time

        bs_prices = np.array(bs_prices)

        # Calculate differences
        price_differences = merton_prices - bs_prices
        relative_differences = price_differences / bs_prices

        benchmark_results = {
            'n_options': len(test_cases),
            'merton_time': merton_time,
            'black_scholes_time': bs_time,
            'time_ratio': merton_time / bs_time,
            'avg_price_difference': np.mean(price_differences),
            'max_price_difference': np.max(np.abs(price_differences)),
            'avg_relative_difference': np.mean(relative_differences),
            'max_relative_difference': np.max(np.abs(relative_differences)),
            'parameters_used': asdict(self.parameters)
        }

        self.logger.info(f"Benchmark Results ({len(test_cases)} options):")
        self.logger.info(f"  Merton time: {merton_time:.4f}s")
        self.logger.info(f"  Black-Scholes time: {bs_time:.4f}s")
        self.logger.info(f"  Time ratio: {merton_time/bs_time:.2f}x")
        self.logger.info(f"  Avg price difference: {np.mean(price_differences):.4f}")
        self.logger.info(f"  Max relative difference: {np.max(np.abs(relative_differences)):.2%}")

        return benchmark_results

    def save_model(self, filepath: str):
        """Save calibrated model parameters"""
        if self.parameters is None:
            raise ValueError("No parameters to save. Run calibration first.")

        model_data = {
            'parameters': asdict(self.parameters),
            'calibration_results': self.calibration_results,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load calibrated model parameters"""
        model_data = joblib.load(filepath)

        self.parameters = MertonParameters(**model_data['parameters'])
        self.calibration_results = model_data.get('calibration_results', {})
        self.max_iterations = model_data.get('max_iterations', 40)
        self.tolerance = model_data.get('tolerance', 1e-10)

        self.logger.info(f"Model loaded from {filepath}")

# Utility functions for easy usage
def create_sample_market_data(n_options: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)

    data = {
        'S': np.random.uniform(80, 120, n_options),
        'K': np.random.uniform(85, 115, n_options),
        'T': np.random.uniform(0.1, 1.0, n_options),
        'r': np.random.uniform(0.02, 0.08, n_options),
        'option_type': np.random.choice(['call', 'put'], n_options)
    }

    df = pd.DataFrame(data)

    # Generate "market prices" using Merton model with known parameters
    merton = MertonJumpDiffusion()
    true_params = MertonParameters(sigma=0.2, lam=2.0, mu_j=-0.05, sigma_j=0.15)
    df['market_price'] = merton.price_batch(df, true_params)

    # Add some noise to simulate market imperfections
    noise = np.random.normal(0, 0.01, n_options)
    df['market_price'] += noise
    df['market_price'] = np.maximum(df['market_price'], 0.01)  # Ensure positive prices

    return df

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    merton = MertonJumpDiffusion(max_iterations=40)

    # Example 1: Price single option
    print("=== Single Option Pricing ===")
    params = MertonParameters(sigma=0.2, lam=2.0, mu_j=-0.1, sigma_j=0.3)

    price = merton.price_option(S=100, K=100, T=0.25, r=0.05, params=params, option_type='call')
    print(f"Call option price: ${price:.4f}")

    # Compare with Black-Scholes
    bs_price = merton._black_scholes_price(100, 100, 0.25, 0.05, params.sigma, 'call')
    print(f"Black-Scholes price: ${bs_price:.4f}")
    print(f"Difference: ${price - bs_price:.4f}")

    # Example 2: Model calibration
    print("\n=== Model Calibration ===")
    market_data = create_sample_market_data(50)

    calibrated_params = merton.calibrate_to_market(market_data)
    print(f"Calibrated parameters: {calibrated_params}")

    # Example 3: Monte Carlo validation
    print("\n=== Monte Carlo Validation ===")
    validation = merton.monte_carlo_validation(
        S=100, K=100, T=0.25, r=0.05, params=calibrated_params,
        n_simulations=50000, option_type='call'
    )

    # Example 4: Batch pricing
    print("\n=== Batch Pricing ===")
    test_options = pd.DataFrame({
        'S': [100, 100, 100],
        'K': [95, 100, 105],
        'T': [0.25, 0.25, 0.25],
        'r': [0.05, 0.05, 0.05],
        'option_type': ['call', 'call', 'call']
    })

    batch_prices = merton.price_batch(test_options, calibrated_params)
    print(f"Batch prices: {batch_prices}")

    # Example 5: Benchmarking
    print("\n=== Performance Benchmark ===")
    large_test_set = create_sample_market_data(1000)
    benchmark_results = merton.benchmark_against_black_scholes(large_test_set)
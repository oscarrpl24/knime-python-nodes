# =============================================================================
# Test Data Generator for Model Performance Monitor
# =============================================================================
# 
# This script generates realistic synthetic loan data for testing the
# Model Performance Monitor node. It creates both production and baseline
# datasets with configurable levels of drift and degradation.
#
# Usage in KNIME:
# 1. Run this as a Python Script node (no inputs required)
# 2. Outputs 2 tables: Baseline Data and Production Data
# 3. Connect these to the Model Performance Monitor inputs
#
# Or run standalone:
# python test_data_generator.py
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data generation parameters
BASELINE_SIZE = 5000      # Number of loans in baseline/training data
PRODUCTION_SIZE = 2000    # Number of loans in production data

# Scorecard parameters (matching typical credit scoring setup)
SCORECARD_POINTS = 600    # Base score at target odds
SCORECARD_ODDS = 20       # Target odds ratio (1:19)
SCORECARD_PDO = 50        # Points to Double the Odds

# Population drift parameters (for production data)
POPULATION_DRIFT = "moderate"  # Options: "none", "slight", "moderate", "significant"
MODEL_DEGRADATION = "slight"   # Options: "none", "slight", "moderate", "severe"

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_score_from_probability(prob: float, points: float = 600, 
                                    odds: float = 20, pdo: float = 50) -> int:
    """
    Convert probability to scorecard score using the standard formula.
    
    Formula:
        b = PDO / log(2)
        a = Points + b * log(odds0)
        odds0 = 1 / (Odds - 1)
        log_odds = log(prob / (1 - prob))
        score = a - b * log_odds
    """
    # Handle edge cases
    prob = np.clip(prob, 0.0001, 0.9999)
    
    # Calculate scaling parameters
    b = pdo / np.log(2)
    odds0 = 1 / (odds - 1)
    a = points + b * np.log(odds0)
    
    # Convert probability to log-odds
    log_odds = np.log(prob / (1 - prob))
    
    # Calculate score
    score = a - b * log_odds
    
    return int(round(score))


def generate_baseline_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate baseline/training data with realistic credit characteristics.
    
    This simulates a healthy, well-calibrated model on the training population.
    """
    print(f"\nGenerating baseline data ({n_samples} samples)...")
    
    # Generate underlying "true" risk scores (latent variable)
    # Normal distribution, mean=0, std=1
    true_risk = np.random.normal(0, 1, n_samples)
    
    # Convert to probability using logistic function
    # This gives a realistic distribution of default probabilities
    true_prob = 1 / (1 + np.exp(-true_risk))
    
    # Generate actual outcomes (0/1) based on true probability
    actual_outcome = (np.random.random(n_samples) < true_prob).astype(int)
    
    # Calculate scores from probabilities
    scores = np.array([calculate_score_from_probability(p, SCORECARD_POINTS, 
                                                         SCORECARD_ODDS, SCORECARD_PDO) 
                      for p in true_prob])
    
    # Add some noise to scores (model isn't perfect)
    score_noise = np.random.normal(0, 15, n_samples)
    scores = np.clip(scores + score_noise, 300, 850).astype(int)
    
    # Generate approval decisions based on score thresholds
    # Higher scores = higher approval probability
    approval_threshold = 550  # Base threshold
    approval_randomness = 30   # Some randomness in decisions
    
    approval_prob = 1 / (1 + np.exp(-(scores - approval_threshold) / approval_randomness))
    is_approved = (np.random.random(n_samples) < approval_prob).astype(int)
    
    # Generate funding decisions (not all approved loans get funded)
    # Funding rate is high for high scores, lower for borderline cases
    funding_prob = np.where(is_approved == 1,
                            1 / (1 + np.exp(-(scores - 500) / 50)),
                            0)
    is_funded = (np.random.random(n_samples) < funding_prob).astype(int)
    
    # Generate ROI for funded loans
    # ROI depends on whether loan defaulted and the score
    # Good loans (outcome=0): ROI > 1 (profit)
    # Bad loans (outcome=1): ROI < 1 (loss)
    
    roi = np.full(n_samples, np.nan)
    
    # For funded loans only
    funded_mask = is_funded == 1
    
    # Good loans (no default): ROI between 1.0 and 1.3
    good_funded = funded_mask & (actual_outcome == 0)
    roi[good_funded] = np.random.uniform(1.0, 1.3, good_funded.sum())
    
    # Bad loans (default): ROI between 0.0 and 0.9
    # Higher scores recover more even in default
    bad_funded = funded_mask & (actual_outcome == 1)
    recovery_rate = (scores[bad_funded] - 300) / 550  # Normalize to 0-1
    roi[bad_funded] = np.clip(recovery_rate * 0.9, 0.0, 0.9)
    
    # Add some noise to ROI
    roi[funded_mask] += np.random.normal(0, 0.05, funded_mask.sum())
    roi = np.clip(roi, 0.0, 1.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': [f'BASELINE_{i:05d}' for i in range(n_samples)],
        'score': scores,
        'isApproved': is_approved,
        'isFunded': is_funded,
        'isBad': actual_outcome,
        'ROI': roi,
        'true_probability': true_prob,  # Hidden - for validation only
        'dataset': 'baseline'
    })
    
    print(f"  Generated {n_samples} loans")
    print(f"  Approval rate: {is_approved.mean():.1%}")
    print(f"  Funding rate: {is_funded.mean():.1%}")
    print(f"  Bad rate (all): {actual_outcome.mean():.1%}")
    print(f"  Bad rate (funded): {actual_outcome[funded_mask].mean():.1%}")
    print(f"  Avg ROI (funded): {roi[funded_mask].mean():.3f}")
    
    return df


def generate_production_data(n_samples: int = 2000,
                             drift_level: str = "moderate",
                             degradation_level: str = "slight") -> pd.DataFrame:
    """
    Generate production data with configurable drift and model degradation.
    
    Parameters:
        n_samples: Number of loans to generate
        drift_level: Population drift - "none", "slight", "moderate", "significant"
        degradation_level: Model degradation - "none", "slight", "moderate", "severe"
    """
    print(f"\nGenerating production data ({n_samples} samples)...")
    print(f"  Drift level: {drift_level}")
    print(f"  Degradation level: {degradation_level}")
    
    # Population shift parameters
    drift_shifts = {
        "none": 0.0,
        "slight": 0.15,
        "moderate": 0.35,
        "significant": 0.60
    }
    mean_shift = drift_shifts.get(drift_level, 0.0)
    
    # Model degradation parameters (how much model loses predictive power)
    degradation_factors = {
        "none": 0.0,
        "slight": 0.15,
        "moderate": 0.35,
        "severe": 0.60
    }
    degradation = degradation_factors.get(degradation_level, 0.0)
    
    # Generate underlying risk - SHIFTED distribution (population drift)
    true_risk = np.random.normal(mean_shift, 1.0, n_samples)
    true_prob = 1 / (1 + np.exp(-true_risk))
    
    # Generate actual outcomes
    actual_outcome = (np.random.random(n_samples) < true_prob).astype(int)
    
    # Model's view of risk - DEGRADED (less accurate)
    # Add noise to model's predictions
    model_risk = true_risk + np.random.normal(0, degradation, n_samples)
    model_prob = 1 / (1 + np.exp(-model_risk))
    
    # Calculate scores from MODEL'S (degraded) probabilities
    scores = np.array([calculate_score_from_probability(p, SCORECARD_POINTS, 
                                                         SCORECARD_ODDS, SCORECARD_PDO) 
                      for p in model_prob])
    
    # Add noise to scores
    score_noise = np.random.normal(0, 15, n_samples)
    scores = np.clip(scores + score_noise, 300, 850).astype(int)
    
    # Generate approval decisions
    approval_threshold = 550
    approval_randomness = 30
    
    approval_prob = 1 / (1 + np.exp(-(scores - approval_threshold) / approval_randomness))
    is_approved = (np.random.random(n_samples) < approval_prob).astype(int)
    
    # Generate funding decisions
    funding_prob = np.where(is_approved == 1,
                            1 / (1 + np.exp(-(scores - 500) / 50)),
                            0)
    is_funded = (np.random.random(n_samples) < funding_prob).astype(int)
    
    # Generate ROI - based on TRUE outcomes (not model predictions)
    roi = np.full(n_samples, np.nan)
    funded_mask = is_funded == 1
    
    good_funded = funded_mask & (actual_outcome == 0)
    roi[good_funded] = np.random.uniform(1.0, 1.3, good_funded.sum())
    
    bad_funded = funded_mask & (actual_outcome == 1)
    recovery_rate = (scores[bad_funded] - 300) / 550
    roi[bad_funded] = np.clip(recovery_rate * 0.9, 0.0, 0.9)
    
    roi[funded_mask] += np.random.normal(0, 0.05, funded_mask.sum())
    roi = np.clip(roi, 0.0, 1.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': [f'PROD_{i:05d}' for i in range(n_samples)],
        'score': scores,
        'isApproved': is_approved,
        'isFunded': is_funded,
        'isBad': actual_outcome,
        'ROI': roi,
        'true_probability': true_prob,
        'model_probability': model_prob,  # For validation
        'dataset': 'production'
    })
    
    print(f"  Generated {n_samples} loans")
    print(f"  Approval rate: {is_approved.mean():.1%}")
    print(f"  Funding rate: {is_funded.mean():.1%}")
    print(f"  Bad rate (all): {actual_outcome.mean():.1%}")
    print(f"  Bad rate (funded): {actual_outcome[funded_mask].mean():.1%}")
    print(f"  Avg ROI (funded): {roi[funded_mask].mean():.3f}")
    
    return df


def calculate_expected_metrics(baseline_df: pd.DataFrame, 
                               production_df: pd.DataFrame) -> None:
    """
    Calculate and display expected performance metrics.
    
    This helps you understand what the Model Performance Monitor should find.
    """
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE METRICS")
    print("="*70)
    
    # Filter to funded loans
    baseline_funded = baseline_df[baseline_df['isFunded'] == 1]
    prod_funded = production_df[production_df['isFunded'] == 1]
    
    print(f"\nSample Sizes:")
    print(f"  Baseline funded: {len(baseline_funded)}")
    print(f"  Production funded: {len(prod_funded)}")
    
    # Calculate score statistics
    print(f"\nScore Distribution:")
    print(f"  Baseline mean: {baseline_df['score'].mean():.0f}")
    print(f"  Production mean: {production_df['score'].mean():.0f}")
    print(f"  Difference: {production_df['score'].mean() - baseline_df['score'].mean():.0f}")
    
    # Calculate bad rates
    baseline_bad_rate = baseline_funded['isBad'].mean()
    prod_bad_rate = prod_funded['isBad'].mean()
    
    print(f"\nBad Rates (Funded):")
    print(f"  Baseline: {baseline_bad_rate:.1%}")
    print(f"  Production: {prod_bad_rate:.1%}")
    print(f"  Change: {(prod_bad_rate - baseline_bad_rate):.1%}")
    
    # Calculate ROI
    baseline_roi = baseline_funded['ROI'].mean()
    prod_roi = prod_funded['ROI'].mean()
    
    print(f"\nAverage ROI (Funded):")
    print(f"  Baseline: {baseline_roi:.3f}")
    print(f"  Production: {prod_roi:.3f}")
    print(f"  Change: {(prod_roi - baseline_roi):.3f}")
    
    # Rough PSI estimate (simplified)
    baseline_scores = baseline_df['score'].values
    prod_scores = production_df['score'].values
    
    # Create deciles
    bins = np.percentile(baseline_scores, np.linspace(0, 100, 11))
    baseline_counts = np.histogram(baseline_scores, bins=bins)[0]
    prod_counts = np.histogram(prod_scores, bins=bins)[0]
    
    baseline_pcts = baseline_counts / len(baseline_scores) + 0.0001
    prod_pcts = prod_counts / len(prod_scores) + 0.0001
    
    psi = np.sum((prod_pcts - baseline_pcts) * np.log(prod_pcts / baseline_pcts))
    
    print(f"\nEstimated PSI: {psi:.3f}")
    
    # Expected recommendation
    print(f"\nExpected Recommendation:")
    if psi >= 0.25:
        rec = "RETRAIN"
    elif psi >= 0.1:
        rec = "MONITOR"
    else:
        rec = "OK"
    
    print(f"  Based on PSI: {rec}")
    
    if abs(prod_bad_rate - baseline_bad_rate) >= 0.02:
        print(f"  Bad rate increase: WARNING (≥2%)")
    
    print("\n" + "="*70)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate test datasets and save to CSV files."""
    
    print("="*70)
    print("MODEL PERFORMANCE MONITOR - TEST DATA GENERATOR")
    print("="*70)
    
    # Generate baseline data
    baseline_df = generate_baseline_data(BASELINE_SIZE)
    
    # Generate production data with configured drift/degradation
    production_df = generate_production_data(
        PRODUCTION_SIZE, 
        POPULATION_DRIFT, 
        MODEL_DEGRADATION
    )
    
    # Calculate expected metrics
    calculate_expected_metrics(baseline_df, production_df)
    
    # Remove validation columns before output
    baseline_output = baseline_df.drop(columns=['true_probability', 'dataset'])
    production_output = production_df.drop(columns=['true_probability', 'model_probability', 'dataset'])
    
    # If running in KNIME, output to KNIME tables
    try:
        import knime.scripting.io as knio
        knio.output_tables[0] = knio.Table.from_pandas(baseline_output)
        knio.output_tables[1] = knio.Table.from_pandas(production_output)
        print("\n✓ Data exported to KNIME output tables")
    except:
        # If not in KNIME, save to CSV files
        baseline_output.to_csv('test_data_baseline.csv', index=False)
        production_output.to_csv('test_data_production.csv', index=False)
        print("\n✓ Data saved to CSV files:")
        print("  - test_data_baseline.csv")
        print("  - test_data_production.csv")
    
    print("\n" + "="*70)
    print("Test data generation complete!")
    print("="*70)
    
    # Generate sample data preview
    print("\nSample data (first 5 rows of production):")
    print(production_output.head())
    
    return baseline_output, production_output


if __name__ == "__main__":
    # When run as standalone script
    baseline_df, production_df = main()
    
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("\n1. In KNIME:")
    print("   - Add this script as a Python Script node (no inputs)")
    print("   - Configure for 2 output ports")
    print("   - Connect Output 1 (baseline) to Monitor Input 2")
    print("   - Connect Output 2 (production) to Monitor Input 1")
    print("\n2. Or use generated CSV files:")
    print("   - Load test_data_baseline.csv")
    print("   - Load test_data_production.csv")
    print("   - Connect to Model Performance Monitor")
    print("\n3. Configure Monitor flow variables:")
    print("   DependentVariable: 'isBad'")
    print("   ScoreColumn: 'score'")
    print("   ApprovalColumn: 'isApproved'")
    print("   FundedColumn: 'isFunded'")
    print("   ROIColumn: 'ROI'")
    print("   Points: 600")
    print("   Odds: 20")
    print("   PDO: 50")
    print("="*70)


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================
# 
# Try different combinations by changing the parameters at the top:
#
# Scenario 1: Healthy Model (should get "OK")
#   POPULATION_DRIFT = "none"
#   MODEL_DEGRADATION = "none"
#
# Scenario 2: Slight Population Shift (should get "OK" or "MONITOR")
#   POPULATION_DRIFT = "slight"
#   MODEL_DEGRADATION = "none"
#
# Scenario 3: Moderate Drift (should get "MONITOR")
#   POPULATION_DRIFT = "moderate"
#   MODEL_DEGRADATION = "slight"
#
# Scenario 4: Significant Issues (should get "RETRAIN")
#   POPULATION_DRIFT = "significant"
#   MODEL_DEGRADATION = "moderate"
#
# Scenario 5: Complete Model Failure (should get "RETRAIN")
#   POPULATION_DRIFT = "significant"
#   MODEL_DEGRADATION = "severe"
#
# =============================================================================

# Execute if run directly
if __name__ == "__main__":
    main()

# =============================================================================
# Model Performance Monitor for KNIME Python Script Node
# =============================================================================
# 
# PURPOSE:
# Monitor production model performance to detect when retraining is needed.
# Tracks discrimination, calibration, population stability, and business metrics.
#
# INPUTS:
# 1. Production Data (required) - Current period loan data
#    Required columns:
#    - score: Scorecard points (integer)
#    - isApproved: 1=approved, 0=declined (integer)
#    - isFunded: 1=funded, 0=not funded (integer)  
#    - <DependentVariable>: Actual outcome 0/1 (only for funded loans)
#    - ROI: Return on investment (only for funded loans, <1 = loss)
#
# 2. Baseline Data (optional) - Training data or stable historical period
#    Same structure as Production Data
#    Used for PSI calculation and performance comparison
#
# OUTPUTS:
# 1. Performance Summary Table - Key metrics with alerts
# 2. Decile Analysis Table - Performance by score decile
# 3. Calibration Table - Expected vs observed bad rates by score band
# 4. Production Data with Diagnostics - Original data + diagnostic columns
#
# FLOW VARIABLES (for headless mode):
# - DependentVariable (str, required): Name of actual outcome column
# - ScoreColumn (str, default: "score"): Name of score column
# - ApprovalColumn (str, default: "isApproved"): Name of approval column
# - FundedColumn (str, default: "isFunded"): Name of funded column
# - ROIColumn (str, default: "ROI"): Name of ROI column
# 
# Scorecard Parameters (for probability calculation):
# - Points (int, default: 600): Base score at target odds
# - Odds (int, default: 20): Target odds ratio (1:X)
# - PDO (int, default: 50): Points to Double the Odds
#
# Alert Thresholds:
# - PSI_Warning (float, default: 0.1): PSI threshold for "MONITOR"
# - PSI_Critical (float, default: 0.25): PSI threshold for "RETRAIN"
# - AUC_Decline_Warning (float, default: 0.03): AUC decline for "MONITOR"
# - AUC_Decline_Critical (float, default: 0.05): AUC decline for "RETRAIN"
# - KS_Decline_Warning (float, default: 0.05): K-S decline for "MONITOR"
# - KS_Decline_Critical (float, default: 0.10): K-S decline for "RETRAIN"
# - BadRate_Increase_Warning (float, default: 0.02): Bad rate increase threshold
# - CalibrationError_Warning (float, default: 0.05): Calibration error threshold
# - MinSampleSize (int, default: 500): Minimum funded loans for reliable metrics
#
# Slack Notifications (Phase 4 - not implemented yet):
# - SlackWebhookURL (str, optional): Slack webhook for notifications
# - SendSlackAlerts (bool, default: False): Enable/disable Slack alerts
# - AlertLevel (str, default: "WARNING"): "WARNING", "CRITICAL", or "ALL"
#
# Release Date: 2026-01-28
# Version: 1.0
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# =============================================================================
# Install/Import Dependencies
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install a Python package if not already available."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package])

install_if_missing('scikit-learn', 'sklearn')
install_if_missing('scipy')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('plotly')

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# Try to import Shiny
try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    SHINY_AVAILABLE = True
except ImportError:
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False

# =============================================================================
# Utility Functions
# =============================================================================

def score_to_probability(score: float, points: float = 600, odds: float = 20, pdo: float = 50) -> float:
    """
    Convert scorecard score to probability using the inverse scorecard formula.
    
    The scorecard formula is:
        score = a - b * log_odds
    
    Where:
        b = PDO / log(2)
        a = Points + b * log(odds0)
        odds0 = 1 / (odds - 1)
    
    Therefore:
        log_odds = (a - score) / b
        probability = 1 / (1 + exp(-log_odds))
    
    Parameters:
        score: Scorecard score (e.g., 650)
        points: Base score at target odds (default 600)
        odds: Target odds ratio 1:X (default 20, meaning 1:19)
        pdo: Points to Double the Odds (default 50)
    
    Returns:
        Probability of bad outcome (0-1)
    """
    # Calculate scaling parameters
    b = pdo / np.log(2)
    odds0 = 1 / (odds - 1)
    a = points + b * np.log(odds0)
    
    # Convert score to log-odds
    log_odds = (a - score) / b
    
    # Convert log-odds to probability
    probability = 1 / (1 + np.exp(-log_odds))
    
    return probability


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted from a baseline.
    Formula: PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    
    Interpretation:
    - PSI < 0.1: Insignificant change (stable)
    - 0.1 <= PSI < 0.25: Moderate change (monitor)
    - PSI >= 0.25: Significant change (investigate/retrain)
    
    Parameters:
        expected: Baseline distribution (e.g., training data scores)
        actual: Current distribution (e.g., production data scores)
        bins: Number of bins to use for distribution comparison
    
    Returns:
        PSI value (float)
    """
    # Create quantile bins based on expected distribution
    try:
        # Get quantile boundaries from expected distribution
        quantiles = [i/bins for i in range(bins+1)]
        breakpoints = expected.quantile(quantiles).values
        
        # Ensure breakpoints are unique and increasing
        breakpoints = np.unique(breakpoints)
        
        # Add -inf and +inf to capture all values
        breakpoints = np.concatenate([[-np.inf], breakpoints[1:-1], [np.inf]])
        
    except Exception as e:
        print(f"Warning: Could not create quantile bins, using equal-width bins instead: {e}")
        # Fallback to equal-width bins
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, bins+1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
    
    # Bin the data
    expected_binned = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_binned = pd.cut(actual, bins=breakpoints, include_lowest=True)
    
    # Calculate percentages in each bin
    expected_percents = expected_binned.value_counts(normalize=True, sort=False)
    actual_percents = actual_binned.value_counts(normalize=True, sort=False)
    
    # Align indices (ensure both have same bins)
    expected_percents, actual_percents = expected_percents.align(actual_percents, fill_value=0)
    
    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 0.0001
    expected_percents = expected_percents + epsilon
    actual_percents = actual_percents + epsilon
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi


def calculate_discrimination_metrics(y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
    """
    Calculate discrimination metrics: AUC, Gini, K-S.
    
    Parameters:
        y_true: Actual outcomes (0/1)
        y_score: Predicted probabilities or scores
    
    Returns:
        Dictionary with metrics: AUC, Gini, KS
    """
    # Handle missing values
    mask = ~(y_true.isna() | y_score.isna())
    y_true_clean = y_true[mask]
    y_score_clean = y_score[mask]
    
    if len(y_true_clean) < 10:
        return {'AUC': np.nan, 'Gini': np.nan, 'KS': np.nan}
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y_true_clean, y_score_clean)
    except:
        auc = np.nan
    
    # Calculate Gini (2*AUC - 1)
    gini = 2 * auc - 1 if not np.isnan(auc) else np.nan
    
    # Calculate K-S statistic
    try:
        # Get cumulative distributions
        fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
        ks = np.max(tpr - fpr)
    except:
        ks = np.nan
    
    return {
        'AUC': round(auc, 4) if not np.isnan(auc) else np.nan,
        'Gini': round(gini, 4) if not np.isnan(gini) else np.nan,
        'KS': round(ks, 4) if not np.isnan(ks) else np.nan
    }


def calculate_calibration(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate calibration metrics: expected vs observed rates by score band.
    
    Parameters:
        y_true: Actual outcomes (0/1)
        y_prob: Predicted probabilities
        n_bins: Number of probability bins
    
    Returns:
        DataFrame with calibration metrics per bin
    """
    # Create dataframe
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # Remove missing values
    df = df.dropna()
    
    if len(df) < 10:
        return pd.DataFrame()
    
    # Create probability bins
    try:
        df['prob_bin'] = pd.qcut(df['y_prob'], q=n_bins, duplicates='drop')
    except:
        # If quantiles fail, use equal-width bins
        df['prob_bin'] = pd.cut(df['y_prob'], bins=n_bins)
    
    # Calculate metrics per bin
    calibration = df.groupby('prob_bin', observed=True).agg({
        'y_true': ['count', 'sum', 'mean'],
        'y_prob': 'mean'
    }).reset_index()
    
    # Flatten column names
    calibration.columns = ['bin', 'count', 'actual_bads', 'observed_rate', 'predicted_rate']
    
    # Calculate difference
    calibration['difference'] = calibration['observed_rate'] - calibration['predicted_rate']
    
    # Mark as calibrated if difference < 0.05
    calibration['calibrated'] = calibration['difference'].abs() < 0.05
    
    # Format bin labels
    calibration['bin'] = calibration['bin'].astype(str)
    
    return calibration


def create_decile_analysis(df: pd.DataFrame, score_col: str, dv_col: str, 
                          approval_col: str = None, roi_col: str = None) -> pd.DataFrame:
    """
    Create decile analysis table.
    
    Parameters:
        df: DataFrame with score, outcome, and optional approval/ROI columns
        score_col: Name of score column
        dv_col: Name of dependent variable (outcome) column
        approval_col: Optional approval column
        roi_col: Optional ROI column
    
    Returns:
        DataFrame with decile-level metrics
    """
    # Create working copy
    analysis_df = df.copy()
    
    # Create deciles based on score (higher score = better, so decile 1 = highest scores)
    analysis_df['decile'] = pd.qcut(analysis_df[score_col], q=10, labels=False, duplicates='drop') + 1
    analysis_df['decile'] = 11 - analysis_df['decile']  # Reverse so decile 1 = highest scores
    
    # Group by decile
    decile_groups = analysis_df.groupby('decile')
    
    # Calculate metrics
    decile_metrics = pd.DataFrame({
        'decile': decile_groups.size().index,
        'count': decile_groups.size().values,
        'score_min': decile_groups[score_col].min().values,
        'score_max': decile_groups[score_col].max().values,
        'score_avg': decile_groups[score_col].mean().round(0).values,
    })
    
    # Add bad rate (for funded loans only)
    funded_mask = analysis_df['decile'].notna()  # All loans have decile
    decile_metrics['bad_rate'] = decile_groups[dv_col].mean().round(4).values
    decile_metrics['bad_count'] = decile_groups[dv_col].sum().values
    
    # Add approval rate if available
    if approval_col and approval_col in analysis_df.columns:
        decile_metrics['approval_rate'] = decile_groups[approval_col].mean().round(4).values
    
    # Add average ROI if available (for funded loans)
    if roi_col and roi_col in analysis_df.columns:
        decile_metrics['avg_roi'] = decile_groups[roi_col].mean().round(4).values
    
    # Calculate cumulative bad rate
    decile_metrics['cumulative_bads'] = decile_metrics['bad_count'].cumsum()
    decile_metrics['cumulative_count'] = decile_metrics['count'].cumsum()
    decile_metrics['cumulative_bad_rate'] = (decile_metrics['cumulative_bads'] / 
                                             decile_metrics['cumulative_count']).round(4)
    
    return decile_metrics


def calculate_business_metrics(df: pd.DataFrame, dv_col: str, approval_col: str, 
                               funded_col: str, roi_col: str) -> Dict[str, float]:
    """
    Calculate business performance metrics.
    
    Parameters:
        df: DataFrame with loan data
        dv_col: Dependent variable column
        approval_col: Approval column
        funded_col: Funded column
        roi_col: ROI column
    
    Returns:
        Dictionary of business metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['total_applications'] = len(df)
    
    # Approval metrics
    if approval_col in df.columns:
        metrics['approvals'] = df[approval_col].sum()
        metrics['approval_rate'] = df[approval_col].mean()
    
    # Funded metrics
    if funded_col in df.columns:
        metrics['funded'] = df[funded_col].sum()
        metrics['funding_rate'] = df[funded_col].mean()
    
    # Performance metrics (funded loans only)
    funded_mask = df[funded_col] == 1
    funded_df = df[funded_mask]
    
    if len(funded_df) > 0:
        metrics['funded_count'] = len(funded_df)
        
        # Bad rate in funded population
        if dv_col in funded_df.columns:
            metrics['bad_rate_funded'] = funded_df[dv_col].mean()
            metrics['bads_funded'] = funded_df[dv_col].sum()
        
        # ROI metrics
        if roi_col in funded_df.columns:
            metrics['avg_roi'] = funded_df[roi_col].mean()
            metrics['median_roi'] = funded_df[roi_col].median()
            metrics['roi_std'] = funded_df[roi_col].std()
            
            # Count losses (ROI < 1.0)
            metrics['losses_count'] = (funded_df[roi_col] < 1.0).sum()
            metrics['loss_rate'] = (funded_df[roi_col] < 1.0).mean()
    
    return metrics


def determine_recommendation(metrics: Dict[str, float], thresholds: Dict[str, float]) -> str:
    """
    Determine recommendation based on metrics and thresholds.
    
    Parameters:
        metrics: Dictionary of calculated metrics
        thresholds: Dictionary of threshold values
    
    Returns:
        "OK", "MONITOR", or "RETRAIN"
    """
    recommendation = "OK"
    
    # Check PSI
    if 'PSI' in metrics and not np.isnan(metrics['PSI']):
        if metrics['PSI'] >= thresholds.get('PSI_Critical', 0.25):
            recommendation = "RETRAIN"
        elif metrics['PSI'] >= thresholds.get('PSI_Warning', 0.1):
            recommendation = "MONITOR"
    
    # Check AUC decline
    if 'AUC_Delta' in metrics and not np.isnan(metrics['AUC_Delta']):
        if metrics['AUC_Delta'] <= -thresholds.get('AUC_Decline_Critical', 0.05):
            recommendation = "RETRAIN"
        elif metrics['AUC_Delta'] <= -thresholds.get('AUC_Decline_Warning', 0.03):
            if recommendation == "OK":
                recommendation = "MONITOR"
    
    # Check K-S decline
    if 'KS_Delta' in metrics and not np.isnan(metrics['KS_Delta']):
        if metrics['KS_Delta'] <= -thresholds.get('KS_Decline_Critical', 0.10):
            recommendation = "RETRAIN"
        elif metrics['KS_Delta'] <= -thresholds.get('KS_Decline_Warning', 0.05):
            if recommendation == "OK":
                recommendation = "MONITOR"
    
    # Check bad rate increase
    if 'BadRate_Delta' in metrics and not np.isnan(metrics['BadRate_Delta']):
        if metrics['BadRate_Delta'] >= thresholds.get('BadRate_Increase_Warning', 0.02):
            if recommendation == "OK":
                recommendation = "MONITOR"
    
    # Check calibration error
    if 'Calibration_Error' in metrics and not np.isnan(metrics['Calibration_Error']):
        if metrics['Calibration_Error'] >= thresholds.get('CalibrationError_Warning', 0.05):
            if recommendation == "OK":
                recommendation = "MONITOR"
    
    return recommendation


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_model_performance(production_df: pd.DataFrame,
                             baseline_df: Optional[pd.DataFrame],
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to analyze model performance.
    
    Parameters:
        production_df: Current production data
        baseline_df: Baseline data (optional)
        config: Configuration dictionary with column names and thresholds
    
    Returns:
        Tuple of (summary_df, decile_df, calibration_df, diagnostics_df)
    """
    print("\n" + "="*70)
    print("Starting Model Performance Analysis")
    print("="*70)
    
    # Extract configuration
    score_col = config.get('score_col', 'score')
    dv_col = config.get('dv_col')
    approval_col = config.get('approval_col', 'isApproved')
    funded_col = config.get('funded_col', 'isFunded')
    roi_col = config.get('roi_col', 'ROI')
    
    # Scorecard parameters
    points = config.get('points', 600)
    odds = config.get('odds', 20)
    pdo = config.get('pdo', 50)
    
    # Thresholds
    thresholds = {
        'PSI_Warning': config.get('psi_warning', 0.1),
        'PSI_Critical': config.get('psi_critical', 0.25),
        'AUC_Decline_Warning': config.get('auc_decline_warning', 0.03),
        'AUC_Decline_Critical': config.get('auc_decline_critical', 0.05),
        'KS_Decline_Warning': config.get('ks_decline_warning', 0.05),
        'KS_Decline_Critical': config.get('ks_decline_critical', 0.10),
        'BadRate_Increase_Warning': config.get('badrate_increase_warning', 0.02),
        'CalibrationError_Warning': config.get('calibration_error_warning', 0.05),
        'MinSampleSize': config.get('min_sample_size', 500)
    }
    
    # Calculate probabilities from scores
    print(f"\nConverting scores to probabilities (Points={points}, Odds={odds}, PDO={pdo})...")
    production_df['probability'] = production_df[score_col].apply(
        lambda x: score_to_probability(x, points, odds, pdo)
    )
    
    if baseline_df is not None:
        baseline_df['probability'] = baseline_df[score_col].apply(
            lambda x: score_to_probability(x, points, odds, pdo)
        )
    
    # Filter to funded loans for performance analysis
    prod_funded = production_df[production_df[funded_col] == 1].copy()
    
    # Check sample size
    funded_count = len(prod_funded)
    min_sample = thresholds['MinSampleSize']
    
    if funded_count < min_sample:
        print(f"\n⚠️  WARNING: Only {funded_count} funded loans (minimum recommended: {min_sample})")
        print("    Metrics may be unreliable with small sample sizes.")
    else:
        print(f"\n✓ Sample size: {funded_count} funded loans (sufficient)")
    
    # Initialize results
    summary_metrics = {}
    
    # =========================================================================
    # Business Metrics
    # =========================================================================
    print("\nCalculating business metrics...")
    business_metrics = calculate_business_metrics(
        production_df, dv_col, approval_col, funded_col, roi_col
    )
    summary_metrics.update(business_metrics)
    
    # =========================================================================
    # Discrimination Metrics (Production)
    # =========================================================================
    print("Calculating discrimination metrics (AUC, Gini, K-S)...")
    prod_discrimination = calculate_discrimination_metrics(
        prod_funded[dv_col], prod_funded['probability']
    )
    
    summary_metrics['Current_AUC'] = prod_discrimination['AUC']
    summary_metrics['Current_Gini'] = prod_discrimination['Gini']
    summary_metrics['Current_KS'] = prod_discrimination['KS']
    
    # =========================================================================
    # Baseline Comparison (if provided)
    # =========================================================================
    if baseline_df is not None:
        print("\nBaseline data provided - calculating comparative metrics...")
        
        # Filter baseline to funded
        base_funded = baseline_df[baseline_df[funded_col] == 1].copy()
        
        print(f"  Baseline sample size: {len(base_funded)} funded loans")
        
        # PSI on score distribution
        print("  Calculating PSI (Population Stability Index)...")
        psi = calculate_psi(baseline_df[score_col], production_df[score_col])
        summary_metrics['PSI'] = round(psi, 4)
        
        # Baseline discrimination
        base_discrimination = calculate_discrimination_metrics(
            base_funded[dv_col], base_funded['probability']
        )
        
        summary_metrics['Baseline_AUC'] = base_discrimination['AUC']
        summary_metrics['Baseline_Gini'] = base_discrimination['Gini']
        summary_metrics['Baseline_KS'] = base_discrimination['KS']
        
        # Calculate deltas
        summary_metrics['AUC_Delta'] = summary_metrics['Current_AUC'] - summary_metrics['Baseline_AUC']
        summary_metrics['Gini_Delta'] = summary_metrics['Current_Gini'] - summary_metrics['Baseline_Gini']
        summary_metrics['KS_Delta'] = summary_metrics['Current_KS'] - summary_metrics['Baseline_KS']
        
        # Bad rate comparison
        if 'bad_rate_funded' in business_metrics:
            baseline_bad_rate = base_funded[dv_col].mean()
            summary_metrics['Baseline_BadRate'] = baseline_bad_rate
            summary_metrics['BadRate_Delta'] = business_metrics['bad_rate_funded'] - baseline_bad_rate
    else:
        print("\nNo baseline data provided - skipping PSI and delta calculations")
    
    # =========================================================================
    # Calibration Analysis
    # =========================================================================
    print("\nCalculating calibration metrics...")
    calibration_df = calculate_calibration(prod_funded[dv_col], prod_funded['probability'])
    
    if not calibration_df.empty:
        calibration_error = calibration_df['difference'].abs().mean()
        summary_metrics['Calibration_Error'] = round(calibration_error, 4)
        summary_metrics['Calibrated_Bins'] = calibration_df['calibrated'].sum()
        summary_metrics['Total_Bins'] = len(calibration_df)
    
    # =========================================================================
    # Decile Analysis
    # =========================================================================
    print("Creating decile analysis...")
    decile_df = create_decile_analysis(
        production_df, score_col, dv_col, approval_col, roi_col
    )
    
    # =========================================================================
    # Determine Recommendation
    # =========================================================================
    print("\nDetermining recommendation...")
    recommendation = determine_recommendation(summary_metrics, thresholds)
    summary_metrics['Recommendation'] = recommendation
    
    # =========================================================================
    # Create Production Diagnostics
    # =========================================================================
    print("Adding diagnostic columns to production data...")
    diagnostics_df = production_df.copy()
    
    # Add prediction correctness for funded loans
    diagnostics_df['prediction_correct'] = np.nan
    funded_mask = diagnostics_df[funded_col] == 1
    if funded_mask.sum() > 0:
        predicted = (diagnostics_df.loc[funded_mask, 'probability'] > 0.5).astype(int)
        diagnostics_df.loc[funded_mask, 'prediction_correct'] = (
            predicted == diagnostics_df.loc[funded_mask, dv_col]
        ).astype(int)
    
    # Add score decile
    diagnostics_df['score_decile'] = pd.qcut(
        diagnostics_df[score_col], q=10, labels=False, duplicates='drop'
    ) + 1
    diagnostics_df['score_decile'] = 11 - diagnostics_df['score_decile']  # Reverse
    
    # =========================================================================
    # Format Summary Table
    # =========================================================================
    print("\nFormatting summary table...")
    
    # Create summary rows
    summary_rows = []
    
    # Sample size check
    sample_status = "OK" if funded_count >= min_sample else "WARNING"
    summary_rows.append({
        'Metric': 'Sample Size (Funded)',
        'Current': funded_count,
        'Baseline': len(baseline_df[baseline_df[funded_col] == 1]) if baseline_df is not None else np.nan,
        'Delta': np.nan,
        'Status': sample_status,
        'Alert': '✓' if sample_status == "OK" else '⚠️'
    })
    
    # AUC
    auc_status = "OK"
    if 'AUC_Delta' in summary_metrics:
        if summary_metrics['AUC_Delta'] <= -thresholds['AUC_Decline_Critical']:
            auc_status = "CRITICAL"
        elif summary_metrics['AUC_Delta'] <= -thresholds['AUC_Decline_Warning']:
            auc_status = "WARNING"
    
    summary_rows.append({
        'Metric': 'AUC',
        'Current': summary_metrics.get('Current_AUC', np.nan),
        'Baseline': summary_metrics.get('Baseline_AUC', np.nan),
        'Delta': summary_metrics.get('AUC_Delta', np.nan),
        'Status': auc_status,
        'Alert': '✓' if auc_status == "OK" else ('⚠️' if auc_status == "WARNING" else '🔴')
    })
    
    # Gini
    summary_rows.append({
        'Metric': 'Gini',
        'Current': summary_metrics.get('Current_Gini', np.nan),
        'Baseline': summary_metrics.get('Baseline_Gini', np.nan),
        'Delta': summary_metrics.get('Gini_Delta', np.nan),
        'Status': auc_status,  # Same as AUC
        'Alert': '✓' if auc_status == "OK" else ('⚠️' if auc_status == "WARNING" else '🔴')
    })
    
    # K-S
    ks_status = "OK"
    if 'KS_Delta' in summary_metrics:
        if summary_metrics['KS_Delta'] <= -thresholds['KS_Decline_Critical']:
            ks_status = "CRITICAL"
        elif summary_metrics['KS_Delta'] <= -thresholds['KS_Decline_Warning']:
            ks_status = "WARNING"
    
    summary_rows.append({
        'Metric': 'K-S Statistic',
        'Current': summary_metrics.get('Current_KS', np.nan),
        'Baseline': summary_metrics.get('Baseline_KS', np.nan),
        'Delta': summary_metrics.get('KS_Delta', np.nan),
        'Status': ks_status,
        'Alert': '✓' if ks_status == "OK" else ('⚠️' if ks_status == "WARNING" else '🔴')
    })
    
    # PSI
    psi_status = "N/A"
    psi_alert = "N/A"
    if 'PSI' in summary_metrics:
        if summary_metrics['PSI'] >= thresholds['PSI_Critical']:
            psi_status = "CRITICAL"
            psi_alert = '🔴'
        elif summary_metrics['PSI'] >= thresholds['PSI_Warning']:
            psi_status = "WARNING"
            psi_alert = '⚠️'
        else:
            psi_status = "OK"
            psi_alert = '✓'
    
    summary_rows.append({
        'Metric': 'PSI (Score)',
        'Current': summary_metrics.get('PSI', np.nan),
        'Baseline': np.nan,
        'Delta': np.nan,
        'Status': psi_status,
        'Alert': psi_alert
    })
    
    # Approval Rate
    summary_rows.append({
        'Metric': 'Approval Rate',
        'Current': summary_metrics.get('approval_rate', np.nan),
        'Baseline': np.nan,
        'Delta': np.nan,
        'Status': "INFO",
        'Alert': 'ℹ️'
    })
    
    # Bad Rate
    badrate_status = "OK"
    if 'BadRate_Delta' in summary_metrics:
        if summary_metrics['BadRate_Delta'] >= thresholds['BadRate_Increase_Warning']:
            badrate_status = "WARNING"
    
    summary_rows.append({
        'Metric': 'Bad Rate (Funded)',
        'Current': summary_metrics.get('bad_rate_funded', np.nan),
        'Baseline': summary_metrics.get('Baseline_BadRate', np.nan),
        'Delta': summary_metrics.get('BadRate_Delta', np.nan),
        'Status': badrate_status,
        'Alert': '✓' if badrate_status == "OK" else '⚠️'
    })
    
    # Average ROI
    summary_rows.append({
        'Metric': 'Avg ROI (Funded)',
        'Current': summary_metrics.get('avg_roi', np.nan),
        'Baseline': np.nan,
        'Delta': np.nan,
        'Status': "INFO",
        'Alert': 'ℹ️'
    })
    
    # Calibration Error
    calib_status = "OK"
    if 'Calibration_Error' in summary_metrics:
        if summary_metrics['Calibration_Error'] >= thresholds['CalibrationError_Warning']:
            calib_status = "WARNING"
    
    summary_rows.append({
        'Metric': 'Calibration Error',
        'Current': summary_metrics.get('Calibration_Error', np.nan),
        'Baseline': np.nan,
        'Delta': np.nan,
        'Status': calib_status,
        'Alert': '✓' if calib_status == "OK" else '⚠️'
    })
    
    # Recommendation
    rec_alert = '✓' if recommendation == "OK" else ('⚠️' if recommendation == "MONITOR" else '🔴')
    summary_rows.append({
        'Metric': 'Recommendation',
        'Current': recommendation,
        'Baseline': np.nan,
        'Delta': np.nan,
        'Status': recommendation,
        'Alert': rec_alert
    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    print("\n" + "="*70)
    print(f"Analysis Complete - Recommendation: {recommendation}")
    print("="*70)
    
    return summary_df, decile_df, calibration_df, diagnostics_df


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_monitor_app(production_df: pd.DataFrame, 
                      baseline_df: Optional[pd.DataFrame],
                      default_config: Dict[str, Any]):
    """Create the interactive Shiny application for model monitoring."""
    
    # Results storage
    app_results = {
        'summary': None,
        'deciles': None,
        'calibration': None,
        'diagnostics': None,
        'completed': False
    }
    
    # UI Definition
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
                
                body { 
                    font-family: 'Inter', sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #fff;
                }
                
                .card { 
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 12px; 
                    padding: 24px; 
                    margin: 16px 0; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    color: #2d3748;
                }
                
                .card-header {
                    color: #2d3748;
                    font-weight: 700;
                    font-size: 1.25rem;
                    margin-bottom: 20px;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 12px;
                }
                
                h2 { 
                    color: #fff; 
                    text-align: center; 
                    font-weight: 700;
                    margin: 32px 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                }
                
                .threshold-input {
                    background: #f7fafc;
                    border: 2px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                }
                
                .btn-primary { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 12px 32px;
                    border-radius: 8px;
                    font-size: 1.1rem;
                }
                .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                }
                
                .btn-success { 
                    background: #48bb78;
                    border: none;
                    color: white;
                    font-weight: 700;
                    padding: 14px 40px;
                    border-radius: 8px;
                    font-size: 1.2rem;
                }
                .btn-success:hover {
                    background: #38a169;
                    transform: translateY(-2px);
                }
                
                .metric-box {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                .metric-value {
                    font-size: 2rem;
                    font-weight: 700;
                }
                .metric-label {
                    font-size: 0.9rem;
                    margin-top: 8px;
                    opacity: 0.9;
                }
            """)
        ),
        
        ui.h2("🎯 Model Performance Monitor"),
        
        # Configuration Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "⚙️ Alert Thresholds"),
            ui.row(
                ui.column(3,
                    ui.div({"class": "threshold-input"},
                        ui.input_numeric("psi_warning", "PSI Warning", 
                                       value=default_config.get('psi_warning', 0.1), 
                                       min=0, max=1, step=0.01)
                    )
                ),
                ui.column(3,
                    ui.div({"class": "threshold-input"},
                        ui.input_numeric("psi_critical", "PSI Critical", 
                                       value=default_config.get('psi_critical', 0.25), 
                                       min=0, max=1, step=0.01)
                    )
                ),
                ui.column(3,
                    ui.div({"class": "threshold-input"},
                        ui.input_numeric("auc_decline", "AUC Decline Warning", 
                                       value=default_config.get('auc_decline_warning', 0.03), 
                                       min=0, max=1, step=0.01)
                    )
                ),
                ui.column(3,
                    ui.div({"class": "threshold-input"},
                        ui.input_numeric("badrate_increase", "Bad Rate Increase Warning", 
                                       value=default_config.get('badrate_increase_warning', 0.02), 
                                       min=0, max=1, step=0.01)
                    )
                )
            ),
            ui.row(
                ui.column(12, 
                    ui.div({"style": "text-align: center; margin-top: 16px;"},
                        ui.input_action_button("analyze", "🔍 Analyze Performance", 
                                             class_="btn btn-primary btn-lg")
                    )
                )
            )
        ),
        
        # Summary Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "📊 Performance Summary"),
            ui.output_ui("summary_display")
        ),
        
        # Decile Analysis Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "📈 Decile Analysis"),
            ui.output_data_frame("decile_table")
        ),
        
        # Calibration Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "🎯 Calibration Analysis"),
            ui.output_data_frame("calibration_table")
        ),
        
        # Action Buttons
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run", "✅ Save Results & Close", class_="btn btn-success btn-lg"),
            ui.tags.br(),
            ui.tags.br(),
            ui.input_action_button("close", "❌ Close Without Saving", class_="btn btn-secondary btn-lg")
        )
    )
    
    # Server Logic
    def server(input: Inputs, output: Outputs, session: Session):
        results_rv = reactive.Value(None)
        
        @reactive.Effect
        @reactive.event(input.analyze)
        def run_analysis():
            """Run analysis with current threshold settings."""
            # Update config with current threshold values
            config = default_config.copy()
            config['psi_warning'] = input.psi_warning() or 0.1
            config['psi_critical'] = input.psi_critical() or 0.25
            config['auc_decline_warning'] = input.auc_decline() or 0.03
            config['badrate_increase_warning'] = input.badrate_increase() or 0.02
            
            try:
                summary, deciles, calibration, diagnostics = analyze_model_performance(
                    production_df, baseline_df, config
                )
                
                results_rv.set({
                    'summary': summary,
                    'deciles': deciles,
                    'calibration': calibration,
                    'diagnostics': diagnostics
                })
            except Exception as e:
                print(f"Error in analysis: {e}")
                import traceback
                traceback.print_exc()
        
        @output
        @render.ui
        def summary_display():
            """Display summary table."""
            results = results_rv.get()
            if results is None:
                return ui.p("Click 'Analyze Performance' to run analysis", 
                          style="text-align: center; color: #718096;")
            
            return render.DataGrid(results['summary'], height="400px", width="100%")
        
        @output
        @render.data_frame
        def decile_table():
            """Display decile analysis table."""
            results = results_rv.get()
            if results is None:
                return render.DataGrid(pd.DataFrame())
            
            return render.DataGrid(results['deciles'], height="400px", width="100%")
        
        @output
        @render.data_frame
        def calibration_table():
            """Display calibration table."""
            results = results_rv.get()
            if results is None:
                return render.DataGrid(pd.DataFrame())
            
            return render.DataGrid(results['calibration'], height="400px", width="100%")
        
        @reactive.Effect
        @reactive.event(input.run)
        async def save_and_close():
            """Save results and close."""
            results = results_rv.get()
            if results is not None:
                app_results.update(results)
                app_results['completed'] = True
            await session.close()
        
        @reactive.Effect
        @reactive.event(input.close)
        async def close_without_save():
            """Close without saving."""
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def run_monitor_ui(production_df: pd.DataFrame, 
                  baseline_df: Optional[pd.DataFrame],
                  config: Dict[str, Any],
                  port: int = 8054):
    """Run the monitoring UI."""
    import threading
    import time
    import socket
    
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except socket.error:
                return False
    
    if not is_port_available(port):
        print(f"WARNING: Port {port} is in use, trying {port+1}...")
        port = port + 1
    
    app = create_monitor_app(production_df, baseline_df, config)
    
    def run_server():
        try:
            print("="*70)
            print(f"Starting Model Monitor UI on http://127.0.0.1:{port}")
            print("="*70)
            app.run(port=port, launch_browser=True)
        except Exception as e:
            print(f"Server error: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    
    wait_count = 0
    while not app.results.get('completed', False):
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            print(f"Waiting... ({wait_count}s)")
    
    time.sleep(0.5)
    print("="*70)
    print("Analysis complete - returning results")
    print("="*70)
    
    return app.results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("Model Performance Monitor - Starting...")
print("="*70)

# Read input data
production_df = knio.input_tables[0].to_pandas()
print(f"Production Data: {len(production_df)} rows")

# Check for baseline data
baseline_df = None
if len(knio.input_tables) > 1:
    baseline_df = knio.input_tables[1].to_pandas()
    print(f"Baseline Data: {len(baseline_df)} rows")
else:
    print("No baseline data provided - PSI and delta metrics will not be calculated")

# Read flow variables
config = {}

# Required: Dependent Variable
try:
    dv_col = knio.flow_variables.get("DependentVariable", None)
    if dv_col is None:
        raise ValueError("DependentVariable flow variable is required")
    config['dv_col'] = dv_col
except Exception as e:
    print(f"ERROR: {e}")
    raise

# Column names
config['score_col'] = knio.flow_variables.get("ScoreColumn", "score")
config['approval_col'] = knio.flow_variables.get("ApprovalColumn", "isApproved")
config['funded_col'] = knio.flow_variables.get("FundedColumn", "isFunded")
config['roi_col'] = knio.flow_variables.get("ROIColumn", "ROI")

# Scorecard parameters
config['points'] = int(knio.flow_variables.get("Points", 600))
config['odds'] = int(knio.flow_variables.get("Odds", 20))
config['pdo'] = int(knio.flow_variables.get("PDO", 50))

# Thresholds
config['psi_warning'] = float(knio.flow_variables.get("PSI_Warning", 0.1))
config['psi_critical'] = float(knio.flow_variables.get("PSI_Critical", 0.25))
config['auc_decline_warning'] = float(knio.flow_variables.get("AUC_Decline_Warning", 0.03))
config['auc_decline_critical'] = float(knio.flow_variables.get("AUC_Decline_Critical", 0.05))
config['ks_decline_warning'] = float(knio.flow_variables.get("KS_Decline_Warning", 0.05))
config['ks_decline_critical'] = float(knio.flow_variables.get("KS_Decline_Critical", 0.10))
config['badrate_increase_warning'] = float(knio.flow_variables.get("BadRate_Increase_Warning", 0.02))
config['calibration_error_warning'] = float(knio.flow_variables.get("CalibrationError_Warning", 0.05))
config['min_sample_size'] = int(knio.flow_variables.get("MinSampleSize", 500))

print(f"\nConfiguration:")
print(f"  DV Column: {config['dv_col']}")
print(f"  Score Column: {config['score_col']}")
print(f"  Scorecard: Points={config['points']}, Odds={config['odds']}, PDO={config['pdo']}")

# Check for interactive mode
force_headless = knio.flow_variables.get("ForceHeadless", False)

# Run analysis
if force_headless or not SHINY_AVAILABLE:
    # Headless mode
    print("\nRunning in HEADLESS mode")
    summary_df, decile_df, calibration_df, diagnostics_df = analyze_model_performance(
        production_df, baseline_df, config
    )
else:
    # Interactive mode
    print("\nRunning in INTERACTIVE mode - launching UI...")
    results = run_monitor_ui(production_df, baseline_df, config)
    
    if results['completed']:
        summary_df = results['summary']
        decile_df = results['deciles']
        calibration_df = results['calibration']
        diagnostics_df = results['diagnostics']
    else:
        print("Interactive session cancelled - returning empty results")
        summary_df = pd.DataFrame()
        decile_df = pd.DataFrame()
        calibration_df = pd.DataFrame()
        diagnostics_df = pd.DataFrame()

# Output results
knio.output_tables[0] = knio.Table.from_pandas(summary_df)
knio.output_tables[1] = knio.Table.from_pandas(decile_df)
knio.output_tables[2] = knio.Table.from_pandas(calibration_df)
knio.output_tables[3] = knio.Table.from_pandas(diagnostics_df)

print("\n" + "="*70)
print("Model Performance Monitor completed successfully")
print("="*70)
print(f"Output 1 (Summary): {len(summary_df)} metrics")
print(f"Output 2 (Deciles): {len(decile_df)} rows")
print(f"Output 3 (Calibration): {len(calibration_df)} bins")
print(f"Output 4 (Diagnostics): {len(diagnostics_df)} rows")
print("="*70)

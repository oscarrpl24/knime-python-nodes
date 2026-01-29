# =============================================================================
# Model Analyzer V2 for KNIME Python Script Node
# =============================================================================
# Enhanced Model Analyzer with comprehensive evaluation metrics beyond AUC
#
# NEW METRICS IN V2:
# 1. Calibration Metrics: Brier Score, Log Loss, Expected Calibration Error (ECE)
# 2. Imbalanced-Friendly: PR-AUC, Average Precision, F1/F2/Fbeta, MCC
# 3. Cost-Sensitive: H-Measure, Expected Maximum Profit (EMP)
# 4. Business Metrics: Precision at Top K, Cumulative Capture Rate
# 5. Monitoring: Population Stability Index (PSI), Characteristic Stability Index
#
# INPUT PORTS:
# Port 1 (Required) - Training Data with probabilities and actual outcomes
# Port 2 (Required) - Coefficients Table for test data predictions
# Port 3 (Optional) - Test Data for out-of-sample evaluation
#
# OUTPUT PORTS:
# Port 1 - Combined data with predictions
# Port 2 - Gains table (decile-based analysis)
# Port 3 - Comprehensive model metrics (all metrics)
#
# Version: 2.0
# Release Date: 2026-01-28
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS AND SETUP
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import os
import gc
import sys
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# Stability settings for multiple instance execution
BASE_PORT = 8052
RANDOM_PORT_RANGE = 1000
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Threading limits for stability
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


# =============================================================================
# SECTION 2: DEPENDENCY INSTALLATION
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install a Python package if not available."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package])

# Install required packages
install_if_missing('scikit-learn', 'sklearn')
install_if_missing('plotly')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('kaleido')
install_if_missing('scipy')

# Import scikit-learn metrics
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, f1_score, fbeta_score,
    matthews_corrcoef, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.special import betaln

# Try to import Shiny
try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    SHINY_AVAILABLE = True
except ImportError:
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# SECTION 3: DATA CLASSES
# =============================================================================

@dataclass
class GainsTable:
    """Container for gains table results."""
    table: pd.DataFrame
    total_obs: int
    total_events: int
    total_non_events: int


@dataclass
class ModelMetricsV2:
    """
    Comprehensive model performance metrics container.
    
    Includes traditional metrics plus new calibration, imbalanced-friendly,
    and cost-sensitive metrics.
    """
    # Traditional Discrimination Metrics
    auc: float                    # Area Under ROC Curve
    gini: float                   # Gini coefficient = 2*AUC - 1
    ks_statistic: float           # Kolmogorov-Smirnov statistic
    ks_decile: int               # Decile where max K-S occurs
    
    # Confusion Matrix Based (at threshold=0.5)
    accuracy: float
    sensitivity: float            # True Positive Rate / Recall
    specificity: float            # True Negative Rate
    precision: float              # Positive Predictive Value
    
    # Calibration Metrics
    brier_score: float            # Mean squared error of probabilities
    log_loss_value: float         # Cross-entropy loss
    ece: float                    # Expected Calibration Error
    
    # Imbalanced-Friendly Metrics
    pr_auc: float                 # Precision-Recall AUC
    average_precision: float      # Average Precision Score
    f1_score: float              # F1 Score (harmonic mean of precision/recall)
    f2_score: float              # F2 Score (recall weighted)
    mcc: float                   # Matthews Correlation Coefficient
    
    # Cost-Sensitive Metrics
    h_measure: float             # H-Measure (coherent alternative to AUC)
    
    # Business Metrics
    precision_top_10pct: float   # Precision when rejecting top 10%
    precision_top_20pct: float   # Precision when rejecting top 20%
    capture_rate_top_10pct: float  # % of bads caught in top 10%
    capture_rate_top_20pct: float  # % of bads caught in top 20%
    
    # Population info
    event_rate: float            # Base rate of events
    n_observations: int          # Total observations
    n_events: int               # Total events


# =============================================================================
# SECTION 4: CORE MATHEMATICAL FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert log-odds to probabilities using sigmoid function."""
    x = np.array(x, dtype=float)
    nan_mask = np.isnan(x)
    x = np.clip(x, -500, 500)
    result = 1 / (1 + np.exp(-x))
    result[nan_mask] = np.nan
    return result


def is_log_odds(values: np.ndarray) -> bool:
    """Detect whether values are log-odds or probabilities."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return False
    if np.any(values < 0) or np.any(values > 1):
        return True
    if np.all((values == 0) | (values == 1)):
        return False
    return False


def ensure_probabilities(values: np.ndarray, col_name: str = "values") -> np.ndarray:
    """Ensure values are probabilities (0-1), converting from log-odds if needed."""
    values = np.array(values, dtype=float)
    if is_log_odds(values):
        print(f"Converting '{col_name}' from log-odds to probabilities")
        return sigmoid(values)
    return values


def parse_coefficients_table(coef_df: pd.DataFrame) -> Dict[str, float]:
    """Parse coefficients table into dictionary."""
    coefficients = {}
    numeric_cols = coef_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in coefficients table")
    coef_col = numeric_cols[0]
    for idx, row in coef_df.iterrows():
        var_name = str(idx)
        coef_value = row[coef_col]
        coefficients[var_name] = float(coef_value)
    print(f"Loaded {len(coefficients)} coefficients")
    return coefficients


def predict_with_coefficients(
    df: pd.DataFrame,
    coefficients: Dict[str, float],
    return_log_odds: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply logistic regression coefficients to compute predictions."""
    n = len(df)
    intercept = coefficients.get('(Intercept)', 0.0)
    if pd.isna(intercept):
        intercept = 0.0
    log_odds = np.full(n, intercept, dtype=float)
    
    for var_name, coef in coefficients.items():
        if var_name == '(Intercept)' or pd.isna(coef):
            continue
        if var_name in df.columns:
            values = np.nan_to_num(df[var_name].values.astype(float), nan=0.0)
            log_odds += coef * values
    
    probabilities = sigmoid(log_odds)
    predicted_class = (probabilities >= 0.5).astype(int)
    
    if return_log_odds:
        return probabilities, predicted_class, log_odds
    return probabilities, predicted_class


# =============================================================================
# SECTION 5: GAINS TABLE CALCULATION
# =============================================================================

def calculate_gains_table(actual: np.ndarray, predicted: np.ndarray, n_deciles: int = 10) -> GainsTable:
    """Calculate gains table with decile-based metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
    
    total_obs = len(df)
    total_events = df['actual'].sum()
    total_non_events = total_obs - total_events
    
    df['decile'] = pd.qcut(range(len(df)), q=n_deciles, labels=False) + 1
    
    gains_data = []
    cumulative_events = 0
    cumulative_non_events = 0
    
    for decile in range(1, n_deciles + 1):
        decile_data = df[df['decile'] == decile]
        n_obs = len(decile_data)
        n_events = decile_data['actual'].sum()
        n_non_events = n_obs - n_events
        
        cumulative_events += n_events
        cumulative_non_events += n_non_events
        
        event_rate = n_events / n_obs if n_obs > 0 else 0
        pct_events = n_events / total_events if total_events > 0 else 0
        pct_non_events = n_non_events / total_non_events if total_non_events > 0 else 0
        cum_pct_events = cumulative_events / total_events if total_events > 0 else 0
        cum_pct_non_events = cumulative_non_events / total_non_events if total_non_events > 0 else 0
        ks = abs(cum_pct_events - cum_pct_non_events)
        decile_pct = decile / n_deciles
        lift = cum_pct_events / decile_pct if decile_pct > 0 else 0
        
        gains_data.append({
            'decile': decile, 'n': n_obs, 
            'events': int(n_events), 'non_events': int(n_non_events),
            'event_rate': round(event_rate, 4), 
            'pct_events': round(pct_events, 4),
            'pct_non_events': round(pct_non_events, 4),
            'cum_events': int(cumulative_events),
            'cum_non_events': int(cumulative_non_events),
            'cum_pct_events': round(cum_pct_events, 4),
            'cum_pct_non_events': round(cum_pct_non_events, 4),
            'ks': round(ks, 4), 'lift': round(lift, 4),
            'min_prob': round(decile_data['predicted'].min(), 4),
            'max_prob': round(decile_data['predicted'].max(), 4),
            'avg_prob': round(decile_data['predicted'].mean(), 4)
        })
    
    return GainsTable(
        table=pd.DataFrame(gains_data),
        total_obs=total_obs,
        total_events=int(total_events),
        total_non_events=int(total_non_events)
    )


# =============================================================================
# SECTION 6: TRADITIONAL METRICS (AUC, Gini, K-S)
# =============================================================================

def calculate_roc_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calculate ROC curve metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    fpr, tpr, _ = roc_curve(actual, predicted)
    auc_score = auc(fpr, tpr)
    gini_index = 2 * auc_score - 1
    return fpr, tpr, round(auc_score, 5), round(gini_index, 5)


def calculate_ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, int]:
    """Calculate K-S statistic."""
    gains = calculate_gains_table(actual, predicted)
    ks_values = gains.table['ks'].values
    ks_statistic = ks_values.max()
    ks_decile = int(np.argmax(ks_values) + 1)
    return round(ks_statistic, 4), ks_decile


# =============================================================================
# SECTION 7: CALIBRATION METRICS (Brier Score, Log Loss, ECE)
# =============================================================================

def calculate_brier_score(actual: np.ndarray, predicted_prob: np.ndarray) -> float:
    """
    Calculate Brier Score - mean squared error of probability predictions.
    
    Brier Score = (1/N) * sum((predicted - actual)^2)
    
    Range: 0 to 1 (lower is better)
    - 0 = perfect predictions
    - 0.25 = random predictions (predicting 0.5 for all)
    - 1 = perfectly wrong predictions
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan
    
    return round(brier_score_loss(actual, predicted_prob), 6)


def calculate_log_loss(actual: np.ndarray, predicted_prob: np.ndarray) -> float:
    """
    Calculate Log Loss (Cross-Entropy Loss).
    
    Log Loss = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
    
    Range: 0 to infinity (lower is better)
    - Penalizes confident wrong predictions heavily
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan
    
    # Clip probabilities to avoid log(0)
    predicted_prob = np.clip(predicted_prob, 1e-15, 1 - 1e-15)
    
    return round(log_loss(actual, predicted_prob), 6)


def calculate_ece(actual: np.ndarray, predicted_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual frequencies.
    Divides predictions into bins and computes weighted average of
    |accuracy - confidence| for each bin.
    
    Range: 0 to 1 (lower is better)
    - 0 = perfectly calibrated
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(actual)
    
    for i in range(n_bins):
        in_bin = (predicted_prob >= bin_edges[i]) & (predicted_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            in_bin = (predicted_prob >= bin_edges[i]) & (predicted_prob <= bin_edges[i + 1])
        
        n_in_bin = in_bin.sum()
        if n_in_bin > 0:
            bin_accuracy = actual[in_bin].mean()
            bin_confidence = predicted_prob[in_bin].mean()
            ece += (n_in_bin / total) * abs(bin_accuracy - bin_confidence)
    
    return round(ece, 6)


# =============================================================================
# SECTION 8: IMBALANCED-FRIENDLY METRICS (PR-AUC, F-scores, MCC)
# =============================================================================

def calculate_pr_auc(actual: np.ndarray, predicted_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Precision-Recall AUC and Average Precision.
    
    These metrics are more informative than ROC-AUC when dealing with
    highly imbalanced datasets (common in credit risk/fraud detection).
    
    Returns:
        Tuple of (PR-AUC, Average Precision)
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0 or actual.sum() == 0:
        return np.nan, np.nan
    
    precision, recall, _ = precision_recall_curve(actual, predicted_prob)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(actual, predicted_prob)
    
    return round(pr_auc, 5), round(avg_precision, 5)


def calculate_f_scores(actual: np.ndarray, predicted_class: np.ndarray) -> Tuple[float, float]:
    """
    Calculate F1 and F2 scores.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    F2 = 5 * (precision * recall) / (4 * precision + recall)
    
    F2 weights recall higher than precision - useful when catching all
    positives (bads) is more important than avoiding false alarms.
    """
    actual = np.array(actual)
    predicted_class = np.array(predicted_class)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_class))
    actual = actual[mask].astype(int)
    predicted_class = predicted_class[mask].astype(int)
    
    if len(actual) == 0:
        return np.nan, np.nan
    
    f1 = f1_score(actual, predicted_class, zero_division=0)
    f2 = fbeta_score(actual, predicted_class, beta=2, zero_division=0)
    
    return round(f1, 5), round(f2, 5)


def calculate_mcc(actual: np.ndarray, predicted_class: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).
    
    MCC is considered the most balanced metric for binary classification,
    especially on imbalanced datasets. It uses all four quadrants of
    the confusion matrix.
    
    Range: -1 to 1
    - 1 = perfect prediction
    - 0 = random prediction
    - -1 = inverse prediction
    """
    actual = np.array(actual)
    predicted_class = np.array(predicted_class)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_class))
    actual = actual[mask].astype(int)
    predicted_class = predicted_class[mask].astype(int)
    
    if len(actual) == 0:
        return np.nan
    
    return round(matthews_corrcoef(actual, predicted_class), 5)


# =============================================================================
# SECTION 9: COST-SENSITIVE METRICS (H-Measure)
# =============================================================================

def calculate_h_measure(actual: np.ndarray, predicted_prob: np.ndarray, 
                        severity_ratio: float = None) -> float:
    """
    Calculate the H-Measure - a coherent alternative to AUC.
    
    The H-Measure addresses a fundamental flaw in AUC: AUC implicitly uses
    different cost weightings when comparing different classifiers, making
    it incoherent as a comparison metric.
    
    H-Measure uses a Beta distribution prior over cost ratios, providing
    a consistent framework for classifier comparison.
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    predicted_prob : array-like  
        Predicted probabilities
    severity_ratio : float, optional
        Ratio of cost(FN) / cost(FP). If None, uses the class imbalance ratio.
        Higher values weight missing positives more heavily.
    
    Returns:
    --------
    float : H-Measure between 0 and 1 (higher is better)
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan
    
    n1 = actual.sum()  # Number of positives
    n0 = len(actual) - n1  # Number of negatives
    
    if n1 == 0 or n0 == 0:
        return np.nan
    
    pi1 = n1 / len(actual)  # Proportion of positives
    pi0 = 1 - pi1
    
    if severity_ratio is None:
        # Default: use class imbalance as severity ratio
        severity_ratio = pi0 / pi1 if pi1 > 0 else 1.0
    
    # Sort by predicted probability
    order = np.argsort(-predicted_prob)
    actual_sorted = actual[order]
    
    # Calculate cumulative true positives and false positives
    tp = np.cumsum(actual_sorted)
    fp = np.cumsum(1 - actual_sorted)
    
    # True positive rate and false positive rate
    tpr = tp / n1
    fpr = fp / n0
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # H-Measure calculation using Beta(2,2) prior by default
    # This gives moderate weight to both types of errors
    a, b = 2, 2  # Beta distribution parameters
    
    # Calculate the H-measure integral numerically
    # This is an approximation using trapezoidal integration
    n_points = len(tpr)
    h_integral = 0.0
    
    for i in range(1, n_points):
        # Cost at this threshold
        c = fpr[i] * pi0 / (fpr[i] * pi0 + (1 - tpr[i]) * pi1 + 1e-10)
        
        # Weight from Beta distribution
        if 0 < c < 1:
            weight = (c ** (a - 1)) * ((1 - c) ** (b - 1))
        else:
            weight = 0
        
        # Loss at this threshold
        loss = pi0 * fpr[i] + pi1 * (1 - tpr[i])
        
        # Trapezoidal integration
        dc = abs(fpr[i] - fpr[i-1]) / (pi0 + 1e-10)
        h_integral += weight * loss * dc
    
    # Normalize by maximum possible loss
    max_loss = min(pi0, pi1)
    
    # H-Measure
    h = 1 - h_integral / (max_loss + 1e-10)
    h = max(0, min(1, h))  # Clamp to [0, 1]
    
    return round(h, 5)


def calculate_emp(actual: np.ndarray, predicted_prob: np.ndarray,
                  revenue_per_good: float = 100,
                  loss_per_bad: float = 500,
                  cost_per_application: float = 0) -> Tuple[float, float]:
    """
    Calculate Expected Maximum Profit (EMP).
    
    EMP is a business-oriented metric that directly measures profitability
    by considering:
    - Revenue from approving good customers
    - Loss from approving bad customers  
    - Processing costs
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0=good, 1=bad)
    predicted_prob : array-like
        Predicted probability of being bad
    revenue_per_good : float
        Profit from approving a good customer
    loss_per_bad : float
        Loss from approving a bad customer (positive number)
    cost_per_application : float
        Cost to process each application
    
    Returns:
    --------
    Tuple of (max_profit, optimal_threshold)
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan, np.nan
    
    n = len(actual)
    n_bad = actual.sum()
    n_good = n - n_bad
    
    # Sort by predicted probability (descending - highest risk first)
    order = np.argsort(-predicted_prob)
    actual_sorted = actual[order]
    prob_sorted = predicted_prob[order]
    
    # Calculate profit at each possible threshold
    # Start with rejecting nobody (approve all)
    profits = []
    thresholds = []
    
    # Cumulative counts as we reject more
    cum_bad_rejected = 0
    cum_good_rejected = 0
    
    for i in range(n + 1):
        # Number approved
        n_approved = n - i
        
        # Of those approved, how many are good vs bad
        bad_approved = n_bad - cum_bad_rejected
        good_approved = n_good - cum_good_rejected
        
        # Calculate profit
        profit = (good_approved * revenue_per_good - 
                  bad_approved * loss_per_bad - 
                  n * cost_per_application)
        
        profits.append(profit)
        thresholds.append(prob_sorted[i-1] if i > 0 else 1.0)
        
        # Update cumulative counts for next iteration
        if i < n:
            if actual_sorted[i] == 1:
                cum_bad_rejected += 1
            else:
                cum_good_rejected += 1
    
    profits = np.array(profits)
    max_profit_idx = np.argmax(profits)
    max_profit = profits[max_profit_idx]
    optimal_threshold = thresholds[max_profit_idx]
    
    # Normalize profit to per-application basis
    max_profit_per_app = max_profit / n
    
    return round(max_profit_per_app, 4), round(optimal_threshold, 4)


# =============================================================================
# SECTION 9B: DYNAMIC PROFIT OPTIMIZATION WITH EXPOSURE
# =============================================================================
# These functions calculate profit using actual loan amounts/exposures,
# which provides much more realistic profit optimization for credit decisions.

@dataclass
class ProfitAnalysisResult:
    """
    Container for dynamic profit analysis results.
    
    Provides comprehensive profit metrics considering actual loan exposures.
    """
    # Optimal threshold and profit
    optimal_threshold: float          # Probability threshold for max profit
    max_total_profit: float           # Maximum achievable total profit
    max_profit_per_application: float # Max profit normalized per application
    
    # Profit at different thresholds
    profit_at_50pct: float           # Profit using 0.50 threshold
    profit_at_optimal: float         # Same as max_total_profit
    
    # Approval statistics at optimal threshold
    approval_rate: float             # % of applications approved
    n_approved: int                  # Count of approved applications
    n_rejected: int                  # Count of rejected applications
    
    # Loss statistics at optimal threshold
    expected_loss_approved: float    # Expected loss from approved bads
    expected_profit_approved: float  # Expected profit from approved goods
    
    # Portfolio characteristics
    total_exposure_approved: float   # Total loan amount approved
    avg_exposure_approved: float     # Average loan size approved
    
    # Risk metrics at optimal threshold
    bad_rate_approved: float         # Bad rate among approved
    bad_rate_rejected: float         # Bad rate among rejected
    
    # Full profit curve data (for charting)
    thresholds: np.ndarray
    profits: np.ndarray
    approval_rates: np.ndarray


def calculate_dynamic_profit(
    actual: np.ndarray,
    predicted_prob: np.ndarray,
    exposure: np.ndarray,
    lgd: Union[float, np.ndarray] = 0.45,
    interest_margin: Union[float, np.ndarray] = 0.05,
    term_years: Union[float, np.ndarray] = 3.0,
    origination_cost: Union[float, np.ndarray] = 50.0,
    servicing_cost_annual: Union[float, np.ndarray] = 0.0,
    recovery_cost_pct: float = 0.10
) -> ProfitAnalysisResult:
    """
    Calculate profit optimization using actual loan amounts/exposures.
    
    This function provides realistic profit calculations by considering:
    - Variable loan amounts (exposure) per customer
    - Loss Given Default (LGD) - what fraction of loan is lost
    - Interest margin - profit rate on performing loans
    - Loan term - duration affects total interest earned
    - Origination and servicing costs
    
    PROFIT FORMULA PER CUSTOMER:
    
    For GOOD customers (approved, doesn't default):
        Profit = exposure × interest_margin × term_years - origination_cost - servicing_costs
    
    For BAD customers (approved, defaults):
        Loss = exposure × LGD + recovery_cost - (partial_interest_earned)
             ≈ exposure × LGD × (1 + recovery_cost_pct)  # simplified
    
    Parameters:
    -----------
    actual : array-like
        Binary actual outcomes (0=good, 1=bad)
    
    predicted_prob : array-like
        Predicted probability of being bad (0 to 1)
    
    exposure : array-like
        Loan amount / Exposure at Default (EAD) for each customer
        This is the key differentiator from simple EMP!
    
    lgd : float or array-like, default=0.45
        Loss Given Default (0 to 1). Can be:
        - Single value (e.g., 0.45 = 45% of loan lost on default)
        - Array with different LGD per customer (e.g., secured vs unsecured)
        Typical values: 0.20-0.40 for secured, 0.60-0.80 for unsecured
    
    interest_margin : float or array-like, default=0.05
        Annual interest margin (net interest income / loan amount)
        e.g., 0.05 = 5% annual margin on performing loans
        Can vary by product (credit cards ~15%, mortgages ~2%)
    
    term_years : float or array-like, default=3.0
        Loan term in years. Longer terms = more interest earned
        Can be per-loan if different products have different terms
    
    origination_cost : float or array-like, default=50.0
        Cost to acquire and originate each loan
        Includes underwriting, documentation, sales commission
    
    servicing_cost_annual : float or array-like, default=0.0
        Annual cost to service each loan (statements, payments, etc.)
    
    recovery_cost_pct : float, default=0.10
        Additional cost as % of loss for collections/recovery
        e.g., 0.10 = 10% of recovered amount goes to collection costs
    
    Returns:
    --------
    ProfitAnalysisResult : Dataclass with comprehensive profit analysis
    
    Example:
    --------
    # Simple usage with loan amounts
    result = calculate_dynamic_profit(
        actual=df['is_bad'].values,
        predicted_prob=df['probability'].values,
        exposure=df['loan_amount'].values,
        lgd=0.45,
        interest_margin=0.06,
        term_years=3
    )
    print(f"Optimal threshold: {result.optimal_threshold}")
    print(f"Maximum profit: ${result.max_total_profit:,.0f}")
    print(f"Approval rate: {result.approval_rate:.1%}")
    
    # Advanced usage with product-specific parameters
    result = calculate_dynamic_profit(
        actual=df['is_bad'].values,
        predicted_prob=df['probability'].values,
        exposure=df['loan_amount'].values,
        lgd=df['product_lgd'].values,  # Different LGD per product
        interest_margin=df['product_margin'].values,  # Different margins
        term_years=df['loan_term_years'].values
    )
    """
    # Convert all inputs to numpy arrays
    actual = np.array(actual, dtype=float)
    predicted_prob = np.array(predicted_prob, dtype=float)
    exposure = np.array(exposure, dtype=float)
    
    # Handle scalar vs array parameters
    n = len(actual)
    lgd = np.full(n, lgd) if np.isscalar(lgd) else np.array(lgd, dtype=float)
    interest_margin = np.full(n, interest_margin) if np.isscalar(interest_margin) else np.array(interest_margin, dtype=float)
    term_years = np.full(n, term_years) if np.isscalar(term_years) else np.array(term_years, dtype=float)
    origination_cost = np.full(n, origination_cost) if np.isscalar(origination_cost) else np.array(origination_cost, dtype=float)
    servicing_cost_annual = np.full(n, servicing_cost_annual) if np.isscalar(servicing_cost_annual) else np.array(servicing_cost_annual, dtype=float)
    
    # Remove records with any NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob) | np.isnan(exposure))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    exposure = exposure[mask]
    lgd = lgd[mask]
    interest_margin = interest_margin[mask]
    term_years = term_years[mask]
    origination_cost = origination_cost[mask]
    servicing_cost_annual = servicing_cost_annual[mask]
    
    if len(actual) == 0:
        return None
    
    n = len(actual)
    
    # Calculate profit/loss for each customer if approved
    # GOOD customer profit: interest income minus costs
    total_servicing_cost = servicing_cost_annual * term_years
    profit_if_good = exposure * interest_margin * term_years - origination_cost - total_servicing_cost
    
    # BAD customer loss: LGD of exposure plus recovery costs
    # Assume default happens mid-way through term (partial interest earned)
    partial_interest = exposure * interest_margin * (term_years * 0.25)  # ~25% of expected interest
    loss_if_bad = exposure * lgd * (1 + recovery_cost_pct) - partial_interest + origination_cost
    
    # Sort by predicted probability (highest risk first = rejected first)
    order = np.argsort(-predicted_prob)
    actual_sorted = actual[order]
    prob_sorted = predicted_prob[order]
    exposure_sorted = exposure[order]
    profit_if_good_sorted = profit_if_good[order]
    loss_if_bad_sorted = loss_if_bad[order]
    
    # Calculate profit at each possible threshold
    # Start with rejecting everyone (approve nobody)
    profits = []
    thresholds = []
    approval_rates_list = []
    
    # Running totals as we approve more (from safest to riskiest)
    cum_profit = 0.0
    
    # Start from the END (lowest risk) and work backwards
    # This is approve-one-more approach from safest to riskiest
    profit_curve = np.zeros(n + 1)
    
    # Calculate cumulative profit from the low-risk end
    for i in range(n):
        idx = n - 1 - i  # Start from lowest probability
        if actual_sorted[idx] == 0:  # Good customer
            cum_profit += profit_if_good_sorted[idx]
        else:  # Bad customer
            cum_profit -= loss_if_bad_sorted[idx]
        profit_curve[i + 1] = cum_profit
    
    # Now profit_curve[k] = profit when approving the (n-k) lowest-risk customers
    # We need to flip the perspective: profit_curve[k] when we REJECT the top k
    
    # Recalculate with reject-from-top perspective
    profits = np.zeros(n + 1)
    total_profit_approve_all = 0.0
    
    # First calculate profit if we approve everyone
    for i in range(n):
        if actual_sorted[i] == 0:  # Good
            total_profit_approve_all += profit_if_good_sorted[i]
        else:  # Bad
            total_profit_approve_all -= loss_if_bad_sorted[i]
    
    profits[0] = total_profit_approve_all  # Approve all
    
    # Now progressively reject from the top (highest risk)
    cum_rejected_profit = 0.0
    for i in range(n):
        if actual_sorted[i] == 0:  # Rejecting a good customer
            cum_rejected_profit -= profit_if_good_sorted[i]  # Lost opportunity
        else:  # Rejecting a bad customer
            cum_rejected_profit += loss_if_bad_sorted[i]  # Avoided loss
        
        profits[i + 1] = total_profit_approve_all + cum_rejected_profit
    
    # Generate thresholds and approval rates
    thresholds = np.zeros(n + 1)
    thresholds[0] = 0.0  # Approve all (threshold = 0, anyone above 0 is rejected... but nobody is)
    for i in range(n):
        thresholds[i + 1] = prob_sorted[i]
    
    approval_rates = 1 - np.arange(n + 1) / n
    
    # Find optimal threshold
    max_profit_idx = np.argmax(profits)
    max_profit = profits[max_profit_idx]
    optimal_threshold = thresholds[max_profit_idx]
    optimal_approval_rate = approval_rates[max_profit_idx]
    
    # Calculate statistics at optimal threshold
    n_rejected = max_profit_idx
    n_approved = n - n_rejected
    
    if n_approved > 0:
        approved_mask = np.zeros(n, dtype=bool)
        approved_mask[n_rejected:] = True  # The ones we keep (lower risk)
        
        approved_actual = actual_sorted[approved_mask]
        approved_exposure = exposure_sorted[approved_mask]
        approved_profit_if_good = profit_if_good_sorted[approved_mask]
        approved_loss_if_bad = loss_if_bad_sorted[approved_mask]
        
        bad_rate_approved = approved_actual.mean() if len(approved_actual) > 0 else 0
        total_exposure_approved = approved_exposure.sum()
        avg_exposure_approved = approved_exposure.mean()
        
        # Expected profit from goods and loss from bads in approved
        expected_profit_goods = approved_profit_if_good[approved_actual == 0].sum()
        expected_loss_bads = approved_loss_if_bad[approved_actual == 1].sum()
    else:
        bad_rate_approved = 0
        total_exposure_approved = 0
        avg_exposure_approved = 0
        expected_profit_goods = 0
        expected_loss_bads = 0
    
    if n_rejected > 0:
        rejected_actual = actual_sorted[:n_rejected]
        bad_rate_rejected = rejected_actual.mean()
    else:
        bad_rate_rejected = 0
    
    # Calculate profit at standard 0.5 threshold
    idx_50 = np.searchsorted(thresholds, 0.5)
    profit_at_50 = profits[min(idx_50, len(profits) - 1)]
    
    return ProfitAnalysisResult(
        optimal_threshold=round(optimal_threshold, 4),
        max_total_profit=round(max_profit, 2),
        max_profit_per_application=round(max_profit / n, 2),
        profit_at_50pct=round(profit_at_50, 2),
        profit_at_optimal=round(max_profit, 2),
        approval_rate=round(optimal_approval_rate, 4),
        n_approved=n_approved,
        n_rejected=n_rejected,
        expected_loss_approved=round(expected_loss_bads, 2),
        expected_profit_approved=round(expected_profit_goods, 2),
        total_exposure_approved=round(total_exposure_approved, 2),
        avg_exposure_approved=round(avg_exposure_approved, 2),
        bad_rate_approved=round(bad_rate_approved, 4),
        bad_rate_rejected=round(bad_rate_rejected, 4),
        thresholds=thresholds,
        profits=profits,
        approval_rates=approval_rates
    )


def calculate_expected_loss(
    predicted_prob: np.ndarray,
    exposure: np.ndarray,
    lgd: Union[float, np.ndarray] = 0.45
) -> Tuple[float, np.ndarray]:
    """
    Calculate Expected Loss (EL) for a portfolio.
    
    Expected Loss = PD × EAD × LGD
    
    This is the fundamental formula for credit risk capital calculation
    under Basel II/III regulations.
    
    Parameters:
    -----------
    predicted_prob : array-like
        Probability of Default (PD) for each customer
    exposure : array-like  
        Exposure at Default (EAD) - loan amount for each customer
    lgd : float or array-like
        Loss Given Default (0 to 1)
    
    Returns:
    --------
    Tuple of:
        - Total portfolio expected loss (float)
        - Array of expected loss per customer
    """
    predicted_prob = np.array(predicted_prob, dtype=float)
    exposure = np.array(exposure, dtype=float)
    
    n = len(predicted_prob)
    lgd = np.full(n, lgd) if np.isscalar(lgd) else np.array(lgd, dtype=float)
    
    # Remove NaN
    mask = ~(np.isnan(predicted_prob) | np.isnan(exposure) | np.isnan(lgd))
    predicted_prob = predicted_prob[mask]
    exposure = exposure[mask]
    lgd = lgd[mask]
    
    # Calculate EL per customer
    el_per_customer = predicted_prob * exposure * lgd
    
    # Total portfolio EL
    total_el = el_per_customer.sum()
    
    return round(total_el, 2), el_per_customer


def calculate_risk_adjusted_return(
    actual: np.ndarray,
    predicted_prob: np.ndarray,
    exposure: np.ndarray,
    interest_margin: float = 0.05,
    lgd: float = 0.45,
    term_years: float = 3.0,
    capital_requirement_pct: float = 0.08,
    cost_of_capital: float = 0.10
) -> Dict[str, float]:
    """
    Calculate Risk-Adjusted Return on Capital (RAROC) metrics.
    
    RAROC = (Revenue - Expected Loss - Operating Costs) / Economic Capital
    
    This helps evaluate whether the return justifies the risk taken.
    
    Parameters:
    -----------
    capital_requirement_pct : float
        % of exposure held as capital (Basel requirement ~8%)
    cost_of_capital : float  
        Required return on capital (typically 10-15%)
    
    Returns:
    --------
    Dict with RAROC metrics
    """
    actual = np.array(actual, dtype=float)
    predicted_prob = np.array(predicted_prob, dtype=float)
    exposure = np.array(exposure, dtype=float)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob) | np.isnan(exposure))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    exposure = exposure[mask]
    
    if len(actual) == 0:
        return {}
    
    # Portfolio totals
    total_exposure = exposure.sum()
    
    # Expected revenue from interest
    expected_revenue = total_exposure * interest_margin * term_years
    
    # Expected loss
    total_el, _ = calculate_expected_loss(predicted_prob, exposure, lgd)
    
    # Actual loss (for comparison)
    actual_loss = (exposure * actual * lgd).sum()
    
    # Economic capital required
    economic_capital = total_exposure * capital_requirement_pct
    
    # Capital charge
    capital_charge = economic_capital * cost_of_capital * term_years
    
    # Net profit
    net_profit = expected_revenue - total_el - capital_charge
    
    # RAROC
    raroc = net_profit / economic_capital if economic_capital > 0 else 0
    
    # Return on Assets (ROA)
    roa = net_profit / total_exposure if total_exposure > 0 else 0
    
    return {
        'total_exposure': round(total_exposure, 2),
        'expected_revenue': round(expected_revenue, 2),
        'expected_loss': round(total_el, 2),
        'actual_loss': round(actual_loss, 2),
        'economic_capital': round(economic_capital, 2),
        'capital_charge': round(capital_charge, 2),
        'net_profit': round(net_profit, 2),
        'raroc': round(raroc, 4),
        'roa': round(roa, 4),
        'loss_ratio': round(total_el / expected_revenue if expected_revenue > 0 else 0, 4)
    }


def create_profit_curve_chart(profit_result: ProfitAnalysisResult, 
                              model_name: str = "Model") -> go.Figure:
    """
    Create a profit curve chart showing profit at different thresholds.
    
    This visualization helps identify the optimal cutoff point and
    understand the profit/approval trade-off.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Profit curve
    fig.add_trace(
        go.Scatter(
            x=profit_result.thresholds,
            y=profit_result.profits,
            mode='lines',
            name='Total Profit',
            line=dict(color='#2ECC71', width=2)
        ),
        secondary_y=False
    )
    
    # Approval rate curve
    fig.add_trace(
        go.Scatter(
            x=profit_result.thresholds,
            y=profit_result.approval_rates * 100,
            mode='lines',
            name='Approval Rate (%)',
            line=dict(color='#3498DB', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    # Mark optimal threshold
    fig.add_vline(
        x=profit_result.optimal_threshold,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text=f'Optimal: {profit_result.optimal_threshold:.3f}'
    )
    
    # Mark standard 0.5 threshold
    fig.add_vline(
        x=0.5,
        line=dict(color='gray', dash='dot', width=1),
        annotation_text='0.50'
    )
    
    # Add annotation for max profit
    fig.add_annotation(
        x=profit_result.optimal_threshold,
        y=profit_result.max_total_profit,
        text=f'Max Profit: ${profit_result.max_total_profit:,.0f}<br>Approval: {profit_result.approval_rate:.1%}',
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=-50
    )
    
    fig.update_layout(
        title=dict(text=f'{model_name} - Profit Optimization Curve', font=dict(size=18)),
        xaxis_title='Probability Threshold (Reject if P(Bad) > threshold)',
        template='plotly_white',
        width=800,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_yaxes(title_text="Total Profit ($)", secondary_y=False)
    fig.update_yaxes(title_text="Approval Rate (%)", secondary_y=True)
    
    return fig


def create_profit_vs_risk_chart(profit_result: ProfitAnalysisResult,
                                 gains_table: GainsTable,
                                 model_name: str = "Model") -> go.Figure:
    """
    Create a chart showing the trade-off between profit and risk.
    
    Combines profit optimization with bad rate analysis.
    """
    # Sample thresholds at decile boundaries
    decile_thresholds = np.percentile(profit_result.thresholds[1:], 
                                       np.linspace(0, 100, 11))
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Profit vs Threshold', 
                                       'Profit vs Approval Rate',
                                       'Bad Rate by Decision',
                                       'Profit per Approved App'])
    
    # 1. Profit vs Threshold
    fig.add_trace(
        go.Scatter(x=profit_result.thresholds, y=profit_result.profits,
                   mode='lines', name='Profit', line=dict(color='#2ECC71')),
        row=1, col=1
    )
    fig.add_vline(x=profit_result.optimal_threshold, 
                  line=dict(color='#E74C3C', dash='dash'),
                  row=1, col=1)
    
    # 2. Profit vs Approval Rate
    fig.add_trace(
        go.Scatter(x=profit_result.approval_rates * 100, y=profit_result.profits,
                   mode='lines', name='Profit vs Approval', 
                   line=dict(color='#3498DB')),
        row=1, col=2
    )
    
    # 3. Bad rate visualization
    deciles = gains_table.table['decile'].values
    event_rates = gains_table.table['event_rate'].values * 100
    fig.add_trace(
        go.Bar(x=deciles, y=event_rates, name='Bad Rate by Decile',
               marker_color='#E74C3C'),
        row=2, col=1
    )
    
    # 4. Profit per approved application
    n_per_threshold = len(profit_result.thresholds)
    approved_counts = profit_result.approval_rates * len(profit_result.profits)
    approved_counts[approved_counts == 0] = 1  # Avoid division by zero
    profit_per_approved = profit_result.profits / approved_counts
    
    fig.add_trace(
        go.Scatter(x=profit_result.thresholds, y=profit_per_approved,
                   mode='lines', name='Profit/Approved',
                   line=dict(color='#9B59B6')),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(text=f'{model_name} - Profit & Risk Analysis', font=dict(size=18)),
        template='plotly_white',
        width=900,
        height=700,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Approval Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="Decile", row=2, col=1)
    fig.update_xaxes(title_text="Threshold", row=2, col=2)
    
    fig.update_yaxes(title_text="Total Profit ($)", row=1, col=1)
    fig.update_yaxes(title_text="Total Profit ($)", row=1, col=2)
    fig.update_yaxes(title_text="Bad Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Profit per Approved ($)", row=2, col=2)
    
    return fig


def profit_analysis_to_dataframe(result: ProfitAnalysisResult, 
                                  dataset_name: str = "Dataset") -> pd.DataFrame:
    """Convert ProfitAnalysisResult to DataFrame for output."""
    data = {
        'dataset': [dataset_name],
        'optimal_threshold': [result.optimal_threshold],
        'max_total_profit': [result.max_total_profit],
        'max_profit_per_application': [result.max_profit_per_application],
        'profit_at_50pct_threshold': [result.profit_at_50pct],
        'approval_rate': [result.approval_rate],
        'n_approved': [result.n_approved],
        'n_rejected': [result.n_rejected],
        'expected_profit_from_goods': [result.expected_profit_approved],
        'expected_loss_from_bads': [result.expected_loss_approved],
        'total_exposure_approved': [result.total_exposure_approved],
        'avg_exposure_approved': [result.avg_exposure_approved],
        'bad_rate_approved': [result.bad_rate_approved],
        'bad_rate_rejected': [result.bad_rate_rejected]
    }
    return pd.DataFrame(data)


# =============================================================================
# SECTION 10: BUSINESS METRICS (Precision at Top K, Capture Rate)
# =============================================================================

def precision_at_k(actual: np.ndarray, predicted_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Calculate precision when rejecting top k% highest-risk applicants.
    
    This answers: "If I reject the top k% by predicted probability,
    what fraction of those rejected are actually bad?"
    
    Parameters:
    -----------
    k : float
        Fraction of population to reject (0.10 = top 10%)
    
    Returns:
    --------
    float : Precision in top k%
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        return np.nan
    
    n_reject = max(1, int(len(actual) * k))
    top_k_idx = np.argsort(-predicted_prob)[:n_reject]
    
    precision = actual[top_k_idx].mean()
    return round(precision, 4)


def capture_rate_at_k(actual: np.ndarray, predicted_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Calculate capture rate (recall) when looking at top k% by predicted probability.
    
    This answers: "If I only review the top k% of cases,
    what fraction of all bad cases will I catch?"
    
    Parameters:
    -----------
    k : float
        Fraction of population to review (0.10 = top 10%)
    
    Returns:
    --------
    float : Capture rate (recall) in top k%
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0 or actual.sum() == 0:
        return np.nan
    
    n_review = max(1, int(len(actual) * k))
    top_k_idx = np.argsort(-predicted_prob)[:n_review]
    
    captured = actual[top_k_idx].sum()
    total_bads = actual.sum()
    
    return round(captured / total_bads, 4)


# =============================================================================
# SECTION 11: MONITORING METRICS (PSI, CSI)
# =============================================================================

def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures how much the distribution of scores has shifted between
    two time periods (or between development and validation samples).
    
    Interpretation:
    - PSI < 0.10: No significant change
    - 0.10 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, model may need recalibration
    
    Parameters:
    -----------
    expected : array-like
        Scores from reference period (e.g., development sample)
    actual : array-like
        Scores from current period (e.g., recent population)
    n_bins : int
        Number of bins for bucketing scores
    
    Returns:
    --------
    float : PSI value (0 to infinity, lower is better)
    """
    expected = np.array(expected)
    actual = np.array(actual)
    
    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins based on expected distribution
    bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Calculate bin percentages
    expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual)
    
    # Add small constant to avoid division by zero and log(0)
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return round(psi, 4)


def calculate_csi(expected_df: pd.DataFrame, actual_df: pd.DataFrame, 
                  variables: List[str]) -> pd.DataFrame:
    """
    Calculate Characteristic Stability Index (CSI) for multiple variables.
    
    CSI is PSI calculated at the variable level, helping identify which
    specific input features have shifted over time.
    
    Returns DataFrame with CSI for each variable.
    """
    results = []
    
    for var in variables:
        if var in expected_df.columns and var in actual_df.columns:
            expected = expected_df[var].values
            actual = actual_df[var].values
            
            csi = calculate_psi(expected, actual)
            
            results.append({
                'variable': var,
                'csi': csi,
                'status': 'OK' if csi < 0.10 else ('INVESTIGATE' if csi < 0.25 else 'ALERT')
            })
    
    return pd.DataFrame(results)


# =============================================================================
# SECTION 12: COMPREHENSIVE METRICS CALCULATION
# =============================================================================

def calculate_all_metrics(actual: np.ndarray, predicted_prob: np.ndarray, 
                          threshold: float = 0.5,
                          revenue_per_good: float = 100,
                          loss_per_bad: float = 500) -> ModelMetricsV2:
    """
    Calculate all model performance metrics.
    
    This is the main function that computes all metrics for a model.
    """
    actual = np.array(actual)
    predicted_prob = np.array(predicted_prob)
    
    # Remove NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted_prob))
    actual = actual[mask]
    predicted_prob = predicted_prob[mask]
    
    if len(actual) == 0:
        raise ValueError("No valid observations after removing NaN")
    
    # Predicted class at threshold
    predicted_class = (predicted_prob >= threshold).astype(int)
    
    # Traditional metrics
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted_prob)
    ks_stat, ks_decile = calculate_ks_statistic(actual, predicted_prob)
    
    # Confusion matrix metrics
    if len(np.unique(actual)) >= 2:
        cm = confusion_matrix(actual, predicted_class)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = precision_score(actual, predicted_class, zero_division=0)
    else:
        accuracy = sensitivity = specificity = prec = 0.0
    
    # Calibration metrics
    brier = calculate_brier_score(actual, predicted_prob)
    log_loss_val = calculate_log_loss(actual, predicted_prob)
    ece = calculate_ece(actual, predicted_prob)
    
    # Imbalanced-friendly metrics
    pr_auc, avg_precision = calculate_pr_auc(actual, predicted_prob)
    f1, f2 = calculate_f_scores(actual, predicted_class)
    mcc = calculate_mcc(actual, predicted_class)
    
    # Cost-sensitive metrics
    h_measure = calculate_h_measure(actual, predicted_prob)
    
    # Business metrics
    prec_10 = precision_at_k(actual, predicted_prob, k=0.10)
    prec_20 = precision_at_k(actual, predicted_prob, k=0.20)
    capture_10 = capture_rate_at_k(actual, predicted_prob, k=0.10)
    capture_20 = capture_rate_at_k(actual, predicted_prob, k=0.20)
    
    # Population info
    event_rate = actual.mean()
    n_obs = len(actual)
    n_events = int(actual.sum())
    
    return ModelMetricsV2(
        # Traditional
        auc=auc_score,
        gini=gini,
        ks_statistic=ks_stat,
        ks_decile=ks_decile,
        
        # Confusion matrix based
        accuracy=round(accuracy, 5),
        sensitivity=round(sensitivity, 5),
        specificity=round(specificity, 5),
        precision=round(prec, 5),
        
        # Calibration
        brier_score=brier,
        log_loss_value=log_loss_val,
        ece=ece,
        
        # Imbalanced-friendly
        pr_auc=pr_auc if not np.isnan(pr_auc) else 0.0,
        average_precision=avg_precision if not np.isnan(avg_precision) else 0.0,
        f1_score=f1 if not np.isnan(f1) else 0.0,
        f2_score=f2 if not np.isnan(f2) else 0.0,
        mcc=mcc if not np.isnan(mcc) else 0.0,
        
        # Cost-sensitive
        h_measure=h_measure if not np.isnan(h_measure) else 0.0,
        
        # Business
        precision_top_10pct=prec_10 if not np.isnan(prec_10) else 0.0,
        precision_top_20pct=prec_20 if not np.isnan(prec_20) else 0.0,
        capture_rate_top_10pct=capture_10 if not np.isnan(capture_10) else 0.0,
        capture_rate_top_20pct=capture_20 if not np.isnan(capture_20) else 0.0,
        
        # Population
        event_rate=round(event_rate, 5),
        n_observations=n_obs,
        n_events=n_events
    )


def metrics_to_dataframe(metrics: ModelMetricsV2, dataset_name: str = "Dataset") -> pd.DataFrame:
    """Convert ModelMetricsV2 to DataFrame for output."""
    data = {
        'dataset': [dataset_name],
        
        # Traditional Discrimination
        'auc': [metrics.auc],
        'gini': [metrics.gini],
        'ks_statistic': [metrics.ks_statistic],
        'ks_decile': [metrics.ks_decile],
        
        # Confusion Matrix Based
        'accuracy': [metrics.accuracy],
        'sensitivity': [metrics.sensitivity],
        'specificity': [metrics.specificity],
        'precision': [metrics.precision],
        
        # Calibration
        'brier_score': [metrics.brier_score],
        'log_loss': [metrics.log_loss_value],
        'ece': [metrics.ece],
        
        # Imbalanced-Friendly
        'pr_auc': [metrics.pr_auc],
        'average_precision': [metrics.average_precision],
        'f1_score': [metrics.f1_score],
        'f2_score': [metrics.f2_score],
        'mcc': [metrics.mcc],
        
        # Cost-Sensitive
        'h_measure': [metrics.h_measure],
        
        # Business Metrics
        'precision_top_10pct': [metrics.precision_top_10pct],
        'precision_top_20pct': [metrics.precision_top_20pct],
        'capture_rate_top_10pct': [metrics.capture_rate_top_10pct],
        'capture_rate_top_20pct': [metrics.capture_rate_top_20pct],
        
        # Population Info
        'event_rate': [metrics.event_rate],
        'n_observations': [metrics.n_observations],
        'n_events': [metrics.n_events]
    }
    
    return pd.DataFrame(data)


# =============================================================================
# SECTION 13: CHART CREATION FUNCTIONS
# =============================================================================

def create_roc_curve(actual: np.ndarray, predicted: np.ndarray, 
                     model_name: str = "Model", color: str = "#E74C3C") -> go.Figure:
    """Create ROC curve chart."""
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
        name=f'{model_name} (AUC={auc_score:.4f}, Gini={gini:.4f})',
        line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
        name='Random (AUC=0.5)', line=dict(color='gray', dash='dash', width=1)))
    
    fig.update_layout(
        title=dict(text='ROC Curve', font=dict(size=18)),
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        template='plotly_white', width=600, height=500)
    return fig


def create_pr_curve(actual: np.ndarray, predicted: np.ndarray,
                    model_name: str = "Model", color: str = "#2ECC71") -> go.Figure:
    """Create Precision-Recall curve chart."""
    precision, recall, _ = precision_recall_curve(actual, predicted)
    pr_auc, avg_prec = calculate_pr_auc(actual, predicted)
    
    base_rate = actual.mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
        name=f'{model_name} (PR-AUC={pr_auc:.4f})',
        line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[base_rate, base_rate], mode='lines',
        name=f'Random (base rate={base_rate:.3f})', 
        line=dict(color='gray', dash='dash', width=1)))
    
    fig.update_layout(
        title=dict(text='Precision-Recall Curve', font=dict(size=18)),
        xaxis_title='Recall (Sensitivity)',
        yaxis_title='Precision',
        template='plotly_white', width=600, height=500)
    return fig


def create_calibration_curve(actual: np.ndarray, predicted: np.ndarray,
                             n_bins: int = 10, model_name: str = "Model") -> go.Figure:
    """Create calibration (reliability) curve chart."""
    prob_true, prob_pred = calibration_curve(actual, predicted, n_bins=n_bins)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers',
        name=f'{model_name}', line=dict(color='#3498DB', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
        name='Perfect Calibration', line=dict(color='gray', dash='dash', width=1)))
    
    # Add histogram of predictions
    fig.add_trace(go.Histogram(x=predicted, name='Prediction Distribution',
        yaxis='y2', opacity=0.3, nbinsx=20, marker_color='#9B59B6'))
    
    fig.update_layout(
        title=dict(text='Calibration Curve', font=dict(size=18)),
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        yaxis2=dict(title='Count', overlaying='y', side='right', showgrid=False),
        template='plotly_white', width=700, height=500)
    return fig


def create_ks_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create K-S chart."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    ks_max = df['ks'].max()
    ks_decile = df.loc[df['ks'].idxmax(), 'decile']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['decile'], y=df['cum_pct_events'], mode='lines+markers',
        name='Cumulative % Events', line=dict(color='#3498DB', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df['decile'], y=df['cum_pct_non_events'], mode='lines+markers',
        name='Cumulative % Non-Events', line=dict(color='#E74C3C', width=2), marker=dict(size=8)))
    
    fig.add_vline(x=ks_decile, line=dict(color='green', dash='dash', width=2))
    
    fig.update_layout(
        title=dict(text=f'K-S Chart (Max K-S={ks_max:.4f} at Decile {ks_decile})', font=dict(size=18)),
        xaxis_title='Decile', yaxis_title='Cumulative Percentage',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white', width=600, height=500)
    return fig


def create_lorenz_curve(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve."""
    sorted_idx = np.argsort(-predicted)
    actual_sorted = actual[sorted_idx]
    
    n = len(actual_sorted)
    total_events = actual_sorted.sum()
    
    cum_pct_pop = np.arange(1, n + 1) / n
    cum_pct_events = np.cumsum(actual_sorted) / total_events
    
    if n > 1000:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        cum_pct_pop = cum_pct_pop[idx]
        cum_pct_events = cum_pct_events[idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_pct_pop, y=cum_pct_events, mode='lines',
        name='Lorenz Curve', fill='tozeroy', line=dict(color='#3498DB', width=2),
        fillcolor='rgba(52, 152, 219, 0.3)'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
        name='Random', line=dict(color='#E74C3C', dash='dash', width=2)))
    
    fig.update_layout(
        title=dict(text='Lorenz Curve', font=dict(size=18)),
        xaxis_title='Cumulative % of Population', yaxis_title='Cumulative % of Events',
        template='plotly_white', width=600, height=500)
    return fig


def create_decile_lift_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Decile Lift chart."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['decile'], y=df['lift'], name='Lift',
        marker_color='#40E0D0', text=df['lift'].round(2), textposition='outside'))
    fig.add_hline(y=1, line=dict(color='#E74C3C', dash='dash', width=2))
    
    fig.update_layout(
        title=dict(text='Decile Lift Chart', font=dict(size=18)),
        xaxis_title='Decile', yaxis_title='Cumulative Lift',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white', width=600, height=500)
    return fig


def create_event_rate_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Event Rate by Decile chart."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    overall_rate = df['events'].sum() / df['n'].sum() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['decile'], y=df['event_rate'] * 100, name='Event Rate (%)',
        marker_color='#00CED1', text=(df['event_rate'] * 100).round(1), textposition='outside'))
    fig.add_hline(y=overall_rate, line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text=f'Overall: {overall_rate:.1f}%')
    
    fig.update_layout(
        title=dict(text='Event Rate by Decile', font=dict(size=18)),
        xaxis_title='Decile', yaxis_title='Event Rate (%)',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white', width=600, height=500)
    return fig


def create_capture_rate_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Capture Rate by Decile chart."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['decile'], y=df['pct_events'] * 100, name='Capture Rate (%)',
        marker_color='#27AE60', text=(df['pct_events'] * 100).round(1), textposition='outside'))
    fig.add_hline(y=10, line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text='Random: 10%')
    
    fig.update_layout(
        title=dict(text='Capture Rate by Decile', font=dict(size=18)),
        xaxis_title='Decile', yaxis_title='% of Total Events',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white', width=600, height=500)
    return fig


def create_metrics_summary_chart(metrics: ModelMetricsV2) -> go.Figure:
    """Create a summary chart showing key metrics as gauges/indicators."""
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{'type': 'indicator'}] * 4, [{'type': 'indicator'}] * 4],
        subplot_titles=['AUC', 'Gini', 'K-S', 'PR-AUC',
                       'Brier Score', 'MCC', 'H-Measure', 'F2 Score']
    )
    
    # Row 1 - Discrimination metrics (higher is better)
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.auc,
        gauge=dict(axis=dict(range=[0.5, 1]), bar=dict(color="#3498DB"),
        steps=[dict(range=[0.5, 0.7], color="#FADBD8"),
               dict(range=[0.7, 0.8], color="#FCF3CF"),
               dict(range=[0.8, 1.0], color="#D5F5E3")])), row=1, col=1)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.gini,
        gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#3498DB"))), row=1, col=2)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.ks_statistic,
        gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#3498DB"))), row=1, col=3)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.pr_auc,
        gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#2ECC71"))), row=1, col=4)
    
    # Row 2 - Other metrics
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.brier_score,
        gauge=dict(axis=dict(range=[0, 0.5]), bar=dict(color="#E74C3C"),
        steps=[dict(range=[0, 0.1], color="#D5F5E3"),
               dict(range=[0.1, 0.25], color="#FCF3CF"),
               dict(range=[0.25, 0.5], color="#FADBD8")])), row=2, col=1)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.mcc,
        gauge=dict(axis=dict(range=[-1, 1]), bar=dict(color="#9B59B6"))), row=2, col=2)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.h_measure,
        gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#F39C12"))), row=2, col=3)
    
    fig.add_trace(go.Indicator(mode="gauge+number", value=metrics.f2_score,
        gauge=dict(axis=dict(range=[0, 1]), bar=dict(color="#1ABC9C"))), row=2, col=4)
    
    fig.update_layout(
        title=dict(text='Model Performance Summary', font=dict(size=20)),
        height=600, width=1000,
        template='plotly_white'
    )
    
    return fig


def save_chart(fig: go.Figure, filepath: str) -> None:
    """Save Plotly figure as image."""
    try:
        fig.write_image(filepath, format='jpeg', width=800, height=600, scale=2)
        print(f"Saved: {filepath}")
    except Exception as e:
        try:
            png_path = filepath.replace('.jpeg', '.png').replace('.jpg', '.png')
            fig.write_image(png_path, format='png', width=800, height=600, scale=2)
            print(f"Saved as PNG: {png_path}")
        except Exception as e2:
            print(f"Failed to save chart: {e2}")


# =============================================================================
# SECTION 14: HEADLESS MODE PROCESSING
# =============================================================================

def run_headless_analysis(df, dv, prob_col, dataset_col, analyze_dataset, model_name, file_path,
                          save_roc=False, save_pr_curve=False, save_calibration=False,
                          save_capture_rate=False, save_ks=False, save_lorenz=False, 
                          save_decile_lift=False, save_event_rate=False, save_summary=False,
                          save_profit_curve=False,
                          # Profit optimization parameters
                          exposure_col=None, lgd=0.45, interest_margin=0.05, 
                          term_years=3.0, origination_cost=50.0):
    """
    Run comprehensive model analysis in headless mode.
    
    Parameters for Dynamic Profit Optimization:
    -------------------------------------------
    exposure_col : str, optional
        Column name containing loan amounts/exposure. If provided,
        enables dynamic profit optimization using actual amounts.
    
    lgd : float, default=0.45
        Loss Given Default (0 to 1). Fraction of exposure lost on default.
        
    interest_margin : float, default=0.05
        Annual interest margin rate (net interest income / exposure).
        
    term_years : float, default=3.0
        Average loan term in years.
        
    origination_cost : float, default=50.0
        Cost to originate each loan.
    """
    
    if file_path and not file_path.endswith(os.sep):
        file_path += os.sep
    
    if file_path:
        os.makedirs(file_path, exist_ok=True)
    
    all_gains = []
    all_metrics = []
    all_profit_analysis = []
    
    # Check if exposure column is available for profit optimization
    has_exposure = exposure_col is not None and exposure_col in df.columns
    if has_exposure:
        print(f"Dynamic profit optimization enabled using '{exposure_col}' column")
        print(f"  LGD: {lgd:.2%}, Interest Margin: {interest_margin:.2%}, Term: {term_years} years")
    
    has_dataset_col = dataset_col in df.columns
    
    if has_dataset_col:
        df_train = df[df[dataset_col].str.lower() == 'training'].copy()
        df_test = df[df[dataset_col].str.lower() == 'test'].copy()
    else:
        df_train = df.copy()
        df_test = pd.DataFrame()
    
    # Process Training data
    if analyze_dataset in ["Training", "Both"] and len(df_train) > 0:
        print(f"Processing Training data: {len(df_train)} observations")
        actual = df_train[dv].values
        predicted = df_train[prob_col].values
        
        # Calculate gains table
        gains = calculate_gains_table(actual, predicted)
        gains_df = gains.table.copy()
        gains_df['dataset'] = 'Training'
        all_gains.append(gains_df)
        
        # Calculate all metrics
        metrics = calculate_all_metrics(actual, predicted)
        metrics_df = metrics_to_dataframe(metrics, 'Training')
        
        # Dynamic Profit Analysis (if exposure available)
        if has_exposure:
            exposure = df_train[exposure_col].values
            profit_result = calculate_dynamic_profit(
                actual=actual,
                predicted_prob=predicted,
                exposure=exposure,
                lgd=lgd,
                interest_margin=interest_margin,
                term_years=term_years,
                origination_cost=origination_cost
            )
            
            if profit_result:
                # Add profit metrics to metrics DataFrame
                metrics_df['optimal_threshold'] = profit_result.optimal_threshold
                metrics_df['max_total_profit'] = profit_result.max_total_profit
                metrics_df['max_profit_per_app'] = profit_result.max_profit_per_application
                metrics_df['profit_approval_rate'] = profit_result.approval_rate
                metrics_df['total_exposure_approved'] = profit_result.total_exposure_approved
                metrics_df['bad_rate_at_optimal'] = profit_result.bad_rate_approved
                
                # Store profit analysis for separate output
                profit_df = profit_analysis_to_dataframe(profit_result, 'Training')
                all_profit_analysis.append(profit_df)
                
                print(f"  Profit Optimization Results:")
                print(f"    Optimal Threshold: {profit_result.optimal_threshold:.4f}")
                print(f"    Max Total Profit: ${profit_result.max_total_profit:,.0f}")
                print(f"    Approval Rate: {profit_result.approval_rate:.1%}")
                print(f"    Bad Rate (Approved): {profit_result.bad_rate_approved:.2%}")
                
                # Save profit curve chart
                if file_path and save_profit_curve:
                    save_chart(create_profit_curve_chart(profit_result, model_name),
                              f"{file_path}{model_name}_Training_ProfitCurve.jpeg")
        
        all_metrics.append(metrics_df)
        
        # Save charts for Training (if only Training or Both requested)
        if file_path and analyze_dataset in ["Training"]:
            if save_roc:
                save_chart(create_roc_curve(actual, predicted, model_name), 
                          f"{file_path}{model_name}_Training_ROC.jpeg")
            if save_pr_curve:
                save_chart(create_pr_curve(actual, predicted, model_name),
                          f"{file_path}{model_name}_Training_PR.jpeg")
            if save_calibration:
                save_chart(create_calibration_curve(actual, predicted, model_name=model_name),
                          f"{file_path}{model_name}_Training_Calibration.jpeg")
            if save_ks:
                save_chart(create_ks_chart(actual, predicted),
                          f"{file_path}{model_name}_Training_KS.jpeg")
            if save_lorenz:
                save_chart(create_lorenz_curve(actual, predicted),
                          f"{file_path}{model_name}_Training_Lorenz.jpeg")
            if save_decile_lift:
                save_chart(create_decile_lift_chart(actual, predicted),
                          f"{file_path}{model_name}_Training_Lift.jpeg")
            if save_event_rate:
                save_chart(create_event_rate_chart(actual, predicted),
                          f"{file_path}{model_name}_Training_EventRate.jpeg")
            if save_capture_rate:
                save_chart(create_capture_rate_chart(actual, predicted),
                          f"{file_path}{model_name}_Training_CaptureRate.jpeg")
            if save_summary:
                save_chart(create_metrics_summary_chart(metrics),
                          f"{file_path}{model_name}_Training_Summary.jpeg")
    
    # Process Test data
    if analyze_dataset in ["Test", "Both"] and len(df_test) > 0:
        print(f"Processing Test data: {len(df_test)} observations")
        actual = df_test[dv].values
        predicted = df_test[prob_col].values
        
        # Calculate gains table
        gains = calculate_gains_table(actual, predicted)
        gains_df = gains.table.copy()
        gains_df['dataset'] = 'Test'
        all_gains.append(gains_df)
        
        # Calculate all metrics
        metrics = calculate_all_metrics(actual, predicted)
        metrics_df = metrics_to_dataframe(metrics, 'Test')
        
        # Dynamic Profit Analysis (if exposure available)
        if has_exposure and exposure_col in df_test.columns:
            exposure = df_test[exposure_col].values
            profit_result = calculate_dynamic_profit(
                actual=actual,
                predicted_prob=predicted,
                exposure=exposure,
                lgd=lgd,
                interest_margin=interest_margin,
                term_years=term_years,
                origination_cost=origination_cost
            )
            
            if profit_result:
                # Add profit metrics to metrics DataFrame
                metrics_df['optimal_threshold'] = profit_result.optimal_threshold
                metrics_df['max_total_profit'] = profit_result.max_total_profit
                metrics_df['max_profit_per_app'] = profit_result.max_profit_per_application
                metrics_df['profit_approval_rate'] = profit_result.approval_rate
                metrics_df['total_exposure_approved'] = profit_result.total_exposure_approved
                metrics_df['bad_rate_at_optimal'] = profit_result.bad_rate_approved
                
                # Store profit analysis
                profit_df = profit_analysis_to_dataframe(profit_result, 'Test')
                all_profit_analysis.append(profit_df)
                
                print(f"  Profit Optimization Results (Test):")
                print(f"    Optimal Threshold: {profit_result.optimal_threshold:.4f}")
                print(f"    Max Total Profit: ${profit_result.max_total_profit:,.0f}")
                print(f"    Approval Rate: {profit_result.approval_rate:.1%}")
                
                # Save profit curve chart
                if file_path and save_profit_curve:
                    save_chart(create_profit_curve_chart(profit_result, model_name),
                              f"{file_path}{model_name}_Test_ProfitCurve.jpeg")
        
        all_metrics.append(metrics_df)
        
        # Save charts
        if file_path:
            if save_roc:
                save_chart(create_roc_curve(actual, predicted, model_name),
                          f"{file_path}{model_name}_Test_ROC.jpeg")
            if save_pr_curve:
                save_chart(create_pr_curve(actual, predicted, model_name),
                          f"{file_path}{model_name}_Test_PR.jpeg")
            if save_calibration:
                save_chart(create_calibration_curve(actual, predicted, model_name=model_name),
                          f"{file_path}{model_name}_Test_Calibration.jpeg")
            if save_ks:
                save_chart(create_ks_chart(actual, predicted),
                          f"{file_path}{model_name}_Test_KS.jpeg")
            if save_lorenz:
                save_chart(create_lorenz_curve(actual, predicted),
                          f"{file_path}{model_name}_Test_Lorenz.jpeg")
            if save_decile_lift:
                save_chart(create_decile_lift_chart(actual, predicted),
                          f"{file_path}{model_name}_Test_Lift.jpeg")
            if save_event_rate:
                save_chart(create_event_rate_chart(actual, predicted),
                          f"{file_path}{model_name}_Test_EventRate.jpeg")
            if save_capture_rate:
                save_chart(create_capture_rate_chart(actual, predicted),
                          f"{file_path}{model_name}_Test_CaptureRate.jpeg")
            if save_summary:
                save_chart(create_metrics_summary_chart(metrics),
                          f"{file_path}{model_name}_Test_Summary.jpeg")
    
    # Calculate PSI if both datasets present
    if analyze_dataset == "Both" and len(df_train) > 0 and len(df_test) > 0:
        train_probs = df_train[prob_col].values
        test_probs = df_test[prob_col].values
        psi = calculate_psi(train_probs, test_probs)
        print(f"Population Stability Index (PSI): {psi}")
        
        # Add PSI to metrics
        if all_metrics:
            for m in all_metrics:
                m['psi'] = psi
    
    combined_gains = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()
    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    
    return combined_gains, combined_metrics


# =============================================================================
# SECTION 15: SHINY UI APPLICATION
# =============================================================================

def create_model_analyzer_app(df, dv=None, prob_col="probability", pred_col="predicted", 
                              dataset_col="dataset"):
    """Create the Model Analyzer V2 Shiny application."""
    
    if not SHINY_AVAILABLE:
        raise RuntimeError("Shiny is not available. Run in headless mode instead.")
    
    # Detect available datasets
    available_datasets = ["All"]
    if dataset_col in df.columns:
        available_datasets = df[dataset_col].unique().tolist()
    
    # Detect DV if not specified
    if dv is None:
        binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
        if binary_cols:
            dv = binary_cols[0]
        else:
            raise ValueError("No binary column found for dependent variable")
    
    # Define UI
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
                       color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
                .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 15px; padding: 20px; margin: 10px; }
                h1, h2, h3 { color: #00d4ff; }
                .metric-box { background: linear-gradient(145deg, #1e3a5f, #0d2137);
                              border-radius: 10px; padding: 15px; margin: 5px;
                              text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #00d4ff; }
                .metric-label { font-size: 12px; color: #aaa; }
            """)
        ),
        
        ui.h1("Model Analyzer V2 - Comprehensive Model Evaluation"),
        ui.p("Enhanced model evaluation with calibration, imbalanced-friendly, and cost-sensitive metrics"),
        
        ui.row(
            ui.column(3,
                ui.div({"class": "card"},
                    ui.h3("Settings"),
                    ui.input_select("dataset_select", "Dataset:",
                                   choices=available_datasets,
                                   selected=available_datasets[0]),
                    ui.input_slider("threshold", "Classification Threshold:", 
                                   min=0.1, max=0.9, value=0.5, step=0.05),
                    ui.hr(),
                    ui.h4("Chart Selection"),
                    ui.input_checkbox("show_roc", "ROC Curve", True),
                    ui.input_checkbox("show_pr", "Precision-Recall", True),
                    ui.input_checkbox("show_calibration", "Calibration", True),
                    ui.input_checkbox("show_ks", "K-S Chart", True),
                    ui.input_checkbox("show_lorenz", "Lorenz Curve", False),
                    ui.input_checkbox("show_lift", "Lift Chart", False),
                )
            ),
            
            ui.column(9,
                # Metrics Summary
                ui.div({"class": "card"},
                    ui.h3("Key Metrics"),
                    ui.output_ui("metrics_summary")
                ),
                
                # Charts
                ui.row(
                    ui.column(6, 
                        ui.div({"class": "card"},
                            output_widget("roc_chart")
                        )
                    ),
                    ui.column(6,
                        ui.div({"class": "card"},
                            output_widget("pr_chart")
                        )
                    )
                ),
                ui.row(
                    ui.column(6,
                        ui.div({"class": "card"},
                            output_widget("calibration_chart")
                        )
                    ),
                    ui.column(6,
                        ui.div({"class": "card"},
                            output_widget("ks_chart")
                        )
                    )
                ),
                
                # Gains Table
                ui.div({"class": "card"},
                    ui.h3("Gains Table"),
                    ui.output_data_frame("gains_table")
                ),
                
                # Full Metrics Table
                ui.div({"class": "card"},
                    ui.h3("Complete Metrics"),
                    ui.output_data_frame("full_metrics_table")
                )
            )
        ),
        
        ui.hr(),
        ui.row(
            ui.column(12,
                ui.input_action_button("done_btn", "Complete Analysis & Close",
                                       class_="btn btn-success btn-lg")
            )
        )
    )
    
    # Define Server
    def server(input: Inputs, output: Outputs, session: Session):
        
        @reactive.Calc
        def filtered_data():
            dataset = input.dataset_select()
            if dataset == "All" or dataset_col not in df.columns:
                return df
            return df[df[dataset_col] == dataset]
        
        @reactive.Calc
        def current_metrics():
            data = filtered_data()
            if len(data) == 0:
                return None
            actual = data[dv].values
            predicted = data[prob_col].values
            threshold = input.threshold()
            return calculate_all_metrics(actual, predicted, threshold=threshold)
        
        @output
        @render.ui
        def metrics_summary():
            metrics = current_metrics()
            if metrics is None:
                return ui.p("No data available")
            
            def metric_box(label, value, fmt=".4f"):
                val_str = f"{value:{fmt}}" if isinstance(value, float) else str(value)
                return ui.div({"class": "metric-box"},
                    ui.div({"class": "metric-value"}, val_str),
                    ui.div({"class": "metric-label"}, label)
                )
            
            return ui.row(
                ui.column(2, metric_box("AUC", metrics.auc)),
                ui.column(2, metric_box("Gini", metrics.gini)),
                ui.column(2, metric_box("K-S", metrics.ks_statistic)),
                ui.column(2, metric_box("PR-AUC", metrics.pr_auc)),
                ui.column(2, metric_box("Brier", metrics.brier_score, ".5f")),
                ui.column(2, metric_box("MCC", metrics.mcc)),
            )
        
        @output
        @render_plotly
        def roc_chart():
            if not input.show_roc():
                return go.Figure()
            data = filtered_data()
            if len(data) == 0:
                return go.Figure()
            return create_roc_curve(data[dv].values, data[prob_col].values)
        
        @output
        @render_plotly
        def pr_chart():
            if not input.show_pr():
                return go.Figure()
            data = filtered_data()
            if len(data) == 0:
                return go.Figure()
            return create_pr_curve(data[dv].values, data[prob_col].values)
        
        @output
        @render_plotly
        def calibration_chart():
            if not input.show_calibration():
                return go.Figure()
            data = filtered_data()
            if len(data) == 0:
                return go.Figure()
            return create_calibration_curve(data[dv].values, data[prob_col].values)
        
        @output
        @render_plotly
        def ks_chart():
            if not input.show_ks():
                return go.Figure()
            data = filtered_data()
            if len(data) == 0:
                return go.Figure()
            return create_ks_chart(data[dv].values, data[prob_col].values)
        
        @output
        @render.data_frame
        def gains_table():
            data = filtered_data()
            if len(data) == 0:
                return pd.DataFrame()
            gains = calculate_gains_table(data[dv].values, data[prob_col].values)
            return gains.table
        
        @output
        @render.data_frame
        def full_metrics_table():
            metrics = current_metrics()
            if metrics is None:
                return pd.DataFrame()
            return metrics_to_dataframe(metrics, input.dataset_select()).T.reset_index()
        
        @reactive.Effect
        @reactive.event(input.done_btn)
        async def close_app():
            await session.close()
    
    return App(app_ui, server)


def find_free_port(start_port: int = 8052, max_attempts: int = 50) -> int:
    """Find an available port."""
    import socket
    for _ in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_model_analyzer(df, dv=None, prob_col="probability", pred_col="predicted",
                       dataset_col="dataset", port=None):
    """Run the Model Analyzer V2 application."""
    if port is None:
        port = find_free_port()
    
    print(f"Starting Model Analyzer V2 on port {port}")
    app = create_model_analyzer_app(df, dv, prob_col, pred_col, dataset_col)
    app.run(port=port, launch_browser=True)


# =============================================================================
# SECTION 16: MAIN EXECUTION - KNIME I/O
# =============================================================================

print("=" * 80)
print("MODEL ANALYZER V2 - Comprehensive Model Evaluation")
print("=" * 80)

# Read Input 1: Training data
print("\nReading Input 1 (Training data)...")
df_train = knio.input_tables[0].to_pandas()
print(f"  Shape: {df_train.shape}")
print(f"  Columns: {list(df_train.columns)[:10]}...")

# Read Input 2: Coefficients
print("\nReading Input 2 (Coefficients)...")
try:
    df_coef = knio.input_tables[1].to_pandas()
    print(f"  Shape: {df_coef.shape}")
    coefficients = parse_coefficients_table(df_coef)
    has_coefficients = True
except Exception as e:
    print(f"  Warning: Could not read coefficients: {e}")
    coefficients = {}
    has_coefficients = False

# Read Input 3: Test data (optional)
print("\nReading Input 3 (Test data)...")
try:
    df_test = knio.input_tables[2].to_pandas()
    if len(df_test) > 0:
        print(f"  Shape: {df_test.shape}")
        has_test_data = True
    else:
        df_test = None
        has_test_data = False
except:
    df_test = None
    has_test_data = False
    print("  Not available")

# Read flow variables
print("\nReading flow variables...")

dv = None
model_name = "Model"
analyze_dataset = "Both" if has_test_data else "Training"
file_path = None
prob_col = "probabilities"

save_roc = save_pr = save_calibration = save_capture_rate = False
save_ks = save_lorenz = save_decile_lift = save_event_rate = save_summary = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if dv in ["missing", ""]: dv = None
    print(f"  DependentVariable: {dv}")
except: pass

try:
    model_name = knio.flow_variables.get("ModelName", "Model")
    if model_name in ["missing", ""]: model_name = "Model"
except: pass

try:
    analyze_dataset = knio.flow_variables.get("AnalyzeDataset", analyze_dataset)
except: pass

try:
    file_path = knio.flow_variables.get("FilePath", None)
    if file_path in ["missing", ""]: file_path = None
    print(f"  FilePath: {file_path}")
except: pass

try:
    prob_col = knio.flow_variables.get("ProbabilitiesColumn", "probabilities")
    if prob_col in ["missing", ""]: prob_col = "probabilities"
except: pass

# Chart save flags
try: save_roc = knio.flow_variables.get("saveROC", 0) == 1
except: pass
try: save_pr = knio.flow_variables.get("savePRCurve", 0) == 1
except: pass
try: save_calibration = knio.flow_variables.get("saveCalibration", 0) == 1
except: pass
try: save_capture_rate = knio.flow_variables.get("saveCaptureRate", 0) == 1
except: pass
try: save_ks = knio.flow_variables.get("saveK-S", 0) == 1
except: pass
try: save_lorenz = knio.flow_variables.get("saveLorenzCurve", 0) == 1
except: pass
try: save_decile_lift = knio.flow_variables.get("saveDecileLift", 0) == 1
except: pass
try: save_event_rate = knio.flow_variables.get("saveEventRate", 0) == 1
except: pass
try: save_summary = knio.flow_variables.get("saveSummary", 0) == 1
except: pass
try: save_profit_curve = knio.flow_variables.get("saveProfitCurve", 0) == 1
except: save_profit_curve = False

# =============================================================================
# DYNAMIC PROFIT OPTIMIZATION PARAMETERS
# =============================================================================
# These flow variables enable profit-based model evaluation using actual loan amounts

exposure_col = None  # Column containing loan amount/exposure
lgd = 0.45          # Loss Given Default (default 45%)
interest_margin = 0.05  # Annual interest margin (default 5%)
term_years = 3.0    # Loan term in years
origination_cost = 50.0  # Cost per loan originated

try:
    exposure_col = knio.flow_variables.get("ExposureColumn", None)
    if exposure_col in ["missing", "", "none", "None"]: 
        exposure_col = None
    if exposure_col:
        print(f"  ExposureColumn: {exposure_col}")
except: pass

try:
    lgd = float(knio.flow_variables.get("LGD", 0.45))
    print(f"  LGD (Loss Given Default): {lgd:.2%}")
except: pass

try:
    interest_margin = float(knio.flow_variables.get("InterestMargin", 0.05))
    print(f"  Interest Margin: {interest_margin:.2%}")
except: pass

try:
    term_years = float(knio.flow_variables.get("TermYears", 3.0))
    print(f"  Term (Years): {term_years}")
except: pass

try:
    origination_cost = float(knio.flow_variables.get("OriginationCost", 50.0))
    print(f"  Origination Cost: ${origination_cost:.2f}")
except: pass

# Auto-detect exposure column if not specified
if exposure_col is None:
    for alt in ['loan_amount', 'LoanAmount', 'exposure', 'Exposure', 'amount', 'Amount', 
                'principal', 'Principal', 'balance', 'Balance', 'EAD']:
        if alt in df_train.columns:
            exposure_col = alt
            print(f"  Auto-detected exposure column: '{exposure_col}'")
            break

# Auto-detect probabilities column
if prob_col not in df_train.columns:
    for alt in ['probability', 'prob', 'probs', 'score', 'pred_prob', 'log_odds']:
        if alt in df_train.columns:
            prob_col = alt
            print(f"  Using '{prob_col}' as probabilities column")
            break

# Process data
print("\n" + "=" * 80)
print("PROCESSING DATA")
print("=" * 80)

df_train['dataset'] = 'Training'
df_train['probability'] = ensure_probabilities(df_train[prob_col].values, prob_col)

if has_test_data and has_coefficients:
    print("\nComputing test predictions from coefficients...")
    test_probs, test_preds, test_log_odds = predict_with_coefficients(df_test, coefficients, return_log_odds=True)
    df_test['probability'] = test_probs
    df_test['predicted'] = test_preds
    df_test['dataset'] = 'Test'
elif has_test_data:
    print("Warning: Test data provided but no coefficients - cannot compute predictions")
    df_test = None
    has_test_data = False

# Combine data
if has_test_data:
    common_cols = ['probability', 'predicted', 'dataset']
    if dv and dv in df_train.columns and dv in df_test.columns:
        common_cols.insert(0, dv)
    # Include exposure column if available
    if exposure_col and exposure_col in df_train.columns:
        if exposure_col not in common_cols:
            common_cols.append(exposure_col)
    df_combined = pd.concat([df_train[common_cols], df_test[common_cols]], ignore_index=True)
    df_combined = df_combined.dropna(subset=['probability'])
    print(f"\nCombined data: {len(df_combined)} observations")
else:
    df_combined = df_train.copy()

# Determine mode
contains_dv = dv is not None and dv in df_combined.columns
contains_file_path = file_path is not None and len(file_path) > 0

print(f"\nMode: {'HEADLESS' if contains_dv and contains_file_path else 'INTERACTIVE'}")

# Main processing
print("\n" + "=" * 80)
print("RUNNING ANALYSIS")
print("=" * 80)

if contains_dv and contains_file_path:
    # Headless mode
    gains_table, metrics_df = run_headless_analysis(
        df=df_combined, dv=dv, prob_col='probability', dataset_col='dataset',
        analyze_dataset=analyze_dataset, model_name=model_name, file_path=file_path,
        save_roc=save_roc, save_pr_curve=save_pr, save_calibration=save_calibration,
        save_capture_rate=save_capture_rate, save_ks=save_ks, save_lorenz=save_lorenz,
        save_decile_lift=save_decile_lift, save_event_rate=save_event_rate, 
        save_summary=save_summary, save_profit_curve=save_profit_curve,
        # Dynamic profit optimization parameters
        exposure_col=exposure_col,
        lgd=lgd,
        interest_margin=interest_margin,
        term_years=term_years,
        origination_cost=origination_cost
    )
elif contains_dv and SHINY_AVAILABLE:
    # Interactive mode
    print("\nLaunching interactive Model Analyzer V2...")
    run_model_analyzer(df_combined, dv=dv, prob_col='probability', 
                       pred_col='predicted', dataset_col='dataset')
    
    # Calculate metrics after UI closes
    gains_table, metrics_df = run_headless_analysis(
        df=df_combined, dv=dv, prob_col='probability', dataset_col='dataset',
        analyze_dataset=analyze_dataset, model_name=model_name, file_path=None,
        exposure_col=exposure_col, lgd=lgd, interest_margin=interest_margin,
        term_years=term_years, origination_cost=origination_cost
    )
else:
    print("Warning: DependentVariable not specified or Shiny not available")
    gains_table = pd.DataFrame()
    metrics_df = pd.DataFrame()

# Prepare outputs
print("\n" + "=" * 80)
print("PREPARING OUTPUTS")
print("=" * 80)

# Output 1: Combined data
if 'predicted' in df_combined.columns:
    df_combined['predicted'] = pd.to_numeric(df_combined['predicted'], errors='coerce').fillna(0).astype('Int32')
if 'probability' in df_combined.columns:
    df_combined['probability'] = pd.to_numeric(df_combined['probability'], errors='coerce').astype('Float64')
if 'dataset' in df_combined.columns:
    df_combined['dataset'] = df_combined['dataset'].astype(str)

print(f"  Output 1 (Combined data): {df_combined.shape}")
knio.output_tables[0] = knio.Table.from_pandas(df_combined)

# Output 2: Gains table
if isinstance(gains_table, pd.DataFrame) and len(gains_table) > 0:
    print(f"  Output 2 (Gains table): {gains_table.shape}")
    knio.output_tables[1] = knio.Table.from_pandas(gains_table)
else:
    knio.output_tables[1] = knio.Table.from_pandas(pd.DataFrame())

# Output 3: Metrics
if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
    print(f"  Output 3 (Metrics): {metrics_df.shape}")
    print(f"\n  Metrics columns: {list(metrics_df.columns)}")
    knio.output_tables[2] = knio.Table.from_pandas(metrics_df)
else:
    knio.output_tables[2] = knio.Table.from_pandas(pd.DataFrame())

print("\n" + "=" * 80)
print("MODEL ANALYZER V2 COMPLETE")
print("=" * 80)

# Cleanup
sys.stdout.flush()
gc.collect()

# =============================================================================
# Model Analyzer for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's Model Analyzer functionality
# Compatible with KNIME 5.9, Python 3.9
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When DependentVariable and FilePath flow variables are provided
#
# Inputs:
# 1. Training Data (required):
#    - All WOE feature columns (WOE_*)
#    - Dependent variable column (actual outcomes 0/1)
#    - "predicted" (Int): Predicted class (0 or 1) 
#    - "probabilities" (Float): Linear predictor / log-odds from regression
#      (Will be converted to probability using sigmoid function if needed)
#
# 2. Coefficients Table (required):
#    - Row ID: Variable names (including "(Intercept)")
#    - Column: Coefficient values (named "model$coefficients" or similar)
#
# 3. Test Data (optional):
#    - Same WOE feature columns as training data
#    - Dependent variable column (actual outcomes 0/1)
#    - Predictions will be computed using the coefficients table
#
# Outputs:
# 1. Combined data with predictions (training + test with probabilities)
# 2. Gains table DataFrame
# 3. Model performance metrics DataFrame
#
# Flow Variables (for headless mode):
# - DependentVariable (string): Binary target variable name
# - ModelName (string, optional): Name for saved chart files
# - AnalyzeDataset (string): "Training", "Test", or "Both" - which subset to analyze
# - FilePath (string): Directory path to save chart images
# - ProbabilitiesColumn (string, optional): Column for log-odds (default: "probabilities")
# - saveROC (int): 1 to save ROC curve
# - saveCaptureRate (int): 1 to save Capture Rate chart
# - saveK-S (int): 1 to save K-S chart
# - saveLorenzCurve (int): 1 to save Lorenz Curve
# - saveDecileLift (int): 1 to save Decile Lift chart
#
# Release Date: 2026-01-16
# Version: 1.2
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
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# Use random port to avoid conflicts when running multiple instances
BASE_PORT = 8051
RANDOM_PORT_RANGE = 1000  # Will pick random port between BASE_PORT and BASE_PORT + RANDOM_PORT_RANGE

# Process isolation: Set unique temp directories per instance
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading conflicts

# =============================================================================
# Install/Import Dependencies
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install package if not available."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package])

install_if_missing('scikit-learn', 'sklearn')
install_if_missing('plotly')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('kaleido')  # For saving plotly figures as images

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    import plotly.express as px
    SHINY_AVAILABLE = True
except ImportError:
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GainsTable:
    """Container for gains table results"""
    table: pd.DataFrame
    total_obs: int
    total_events: int
    total_non_events: int


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    auc: float
    gini: float
    ks_statistic: float
    ks_decile: int
    accuracy: float
    sensitivity: float
    specificity: float


# =============================================================================
# Logistic Regression Prediction Functions
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid function to convert log-odds to probabilities.
    
    Parameters:
    -----------
    x : array-like
        Log-odds (linear predictor) values
    
    Returns:
    --------
    Probabilities between 0 and 1
    """
    x = np.array(x, dtype=float)
    # Preserve NaN positions
    nan_mask = np.isnan(x)
    # Clip to avoid overflow (only non-NaN values)
    x = np.clip(x, -500, 500)
    result = 1 / (1 + np.exp(-x))
    # Restore NaN
    result[nan_mask] = np.nan
    return result


def is_log_odds(values: np.ndarray) -> bool:
    """
    Detect if values are log-odds (can be outside 0-1) or probabilities (0-1).
    
    Returns True if values appear to be log-odds (some values outside 0-1 range).
    """
    values = np.array(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return False
    
    # If any values are outside [0, 1], they must be log-odds
    if np.any(values < 0) or np.any(values > 1):
        return True
    
    # If all values are exactly 0 or 1, probably not log-odds
    if np.all((values == 0) | (values == 1)):
        return False
    
    # If range is very narrow around 0.5, might be log-odds near 0
    # But typically log-odds have more spread, so assume probability if in [0,1]
    return False


def parse_coefficients_table(coef_df: pd.DataFrame) -> Dict[str, float]:
    """
    Parse coefficients table from R model output.
    
    Expected format:
    - Row ID contains variable names (including "(Intercept)")
    - First numeric column contains coefficient values
    
    Parameters:
    -----------
    coef_df : DataFrame
        Coefficients table with Row ID as variable names
    
    Returns:
    --------
    Dictionary mapping variable names to coefficient values
    """
    coefficients = {}
    
    # Find the coefficient column (first numeric column)
    numeric_cols = coef_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in coefficients table")
    
    coef_col = numeric_cols[0]
    print(f"Using coefficient column: '{coef_col}'")
    
    # Use index (Row ID) as variable names
    for idx, row in coef_df.iterrows():
        var_name = str(idx)
        coef_value = row[coef_col]
        coefficients[var_name] = float(coef_value)
    
    print(f"Loaded {len(coefficients)} coefficients")
    if '(Intercept)' in coefficients:
        print(f"  Intercept: {coefficients['(Intercept)']:.6f}")
    
    return coefficients


def predict_with_coefficients(
    df: pd.DataFrame,
    coefficients: Dict[str, float],
    return_log_odds: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply logistic regression coefficients to compute predictions.
    
    Parameters:
    -----------
    df : DataFrame
        Data with feature columns matching coefficient names
    coefficients : dict
        Dictionary mapping variable names to coefficients
    return_log_odds : bool
        If True, also return raw log-odds values
    
    Returns:
    --------
    Tuple of (probabilities, predicted_class) arrays
    If return_log_odds=True, returns (probabilities, predicted_class, log_odds)
    """
    n = len(df)
    
    # Start with intercept
    intercept = coefficients.get('(Intercept)', 0.0)
    if pd.isna(intercept):
        intercept = 0.0
    log_odds = np.full(n, intercept, dtype=float)
    
    # Add contribution from each variable
    matched_vars = 0
    missing_vars = []
    nan_filled_vars = []
    
    for var_name, coef in coefficients.items():
        if var_name == '(Intercept)':
            continue
        
        # Skip if coefficient is NaN
        if pd.isna(coef):
            continue
        
        if var_name in df.columns:
            values = df[var_name].values.astype(float)
            # Count NaN values before filling
            nan_count = np.isnan(values).sum()
            if nan_count > 0:
                nan_filled_vars.append((var_name, nan_count))
            # Fill NaN with 0 (neutral for WOE variables)
            values = np.nan_to_num(values, nan=0.0)
            log_odds += coef * values
            matched_vars += 1
        else:
            missing_vars.append(var_name)
    
    if nan_filled_vars:
        print(f"Note: Filled NaN with 0 in {len(nan_filled_vars)} variables:")
        for var, count in nan_filled_vars[:3]:
            print(f"  - {var}: {count} NaN values")
        if len(nan_filled_vars) > 3:
            print(f"  ... and {len(nan_filled_vars) - 3} more")
    
    if missing_vars:
        print(f"Warning: {len(missing_vars)} coefficient variables not found in data:")
        for var in missing_vars[:5]:
            print(f"  - {var}")
        if len(missing_vars) > 5:
            print(f"  ... and {len(missing_vars) - 5} more")
    
    print(f"Matched {matched_vars} variables from coefficients")
    
    # Convert log-odds to probabilities
    probabilities = sigmoid(log_odds)
    
    # Predict class (0 or 1) based on 0.5 threshold
    predicted_class = (probabilities >= 0.5).astype(int)
    
    if return_log_odds:
        return probabilities, predicted_class, log_odds
    
    return probabilities, predicted_class


def ensure_probabilities(values: np.ndarray, col_name: str = "values") -> np.ndarray:
    """
    Ensure values are probabilities (0-1). Convert from log-odds if needed.
    
    Parameters:
    -----------
    values : array-like
        Either probabilities (0-1) or log-odds
    col_name : str
        Column name for logging
    
    Returns:
    --------
    Probabilities between 0 and 1
    """
    values = np.array(values, dtype=float)
    
    # Handle NaN values
    nan_count = np.isnan(values).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in '{col_name}'")
    
    if is_log_odds(values):
        print(f"Converting '{col_name}' from log-odds to probabilities (values outside 0-1 detected)")
        return sigmoid(values)
    else:
        print(f"'{col_name}' appears to already be probabilities (all values in 0-1 range)")
        return values


# =============================================================================
# Core Metric Calculation Functions
# =============================================================================

def calculate_gains_table(actual: np.ndarray, predicted: np.ndarray, n_deciles: int = 10) -> GainsTable:
    """
    Calculate gains table (equivalent to R's blorr::blr_gains_table).
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0/1)
    predicted : array-like
        Predicted probabilities
    n_deciles : int
        Number of deciles (default 10)
    
    Returns:
    --------
    GainsTable object with gains table DataFrame and totals
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Create DataFrame and sort by predicted probability descending
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    })
    df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
    
    total_obs = len(df)
    total_events = df['actual'].sum()
    total_non_events = total_obs - total_events
    
    # Create deciles
    df['decile'] = pd.qcut(range(len(df)), q=n_deciles, labels=False) + 1
    
    # Calculate metrics per decile
    gains_data = []
    cumulative_events = 0
    cumulative_non_events = 0
    cumulative_obs = 0
    
    for decile in range(1, n_deciles + 1):
        decile_data = df[df['decile'] == decile]
        
        n_obs = len(decile_data)
        n_events = decile_data['actual'].sum()
        n_non_events = n_obs - n_events
        
        cumulative_obs += n_obs
        cumulative_events += n_events
        cumulative_non_events += n_non_events
        
        event_rate = n_events / n_obs if n_obs > 0 else 0
        pct_events = n_events / total_events if total_events > 0 else 0
        pct_non_events = n_non_events / total_non_events if total_non_events > 0 else 0
        
        cum_pct_events = cumulative_events / total_events if total_events > 0 else 0
        cum_pct_non_events = cumulative_non_events / total_non_events if total_non_events > 0 else 0
        
        # K-S statistic for this decile
        ks = abs(cum_pct_events - cum_pct_non_events)
        
        # Lift
        decile_pct = decile / n_deciles
        lift = cum_pct_events / decile_pct if decile_pct > 0 else 0
        
        # Min/Max predicted probability in decile
        min_prob = decile_data['predicted'].min()
        max_prob = decile_data['predicted'].max()
        avg_prob = decile_data['predicted'].mean()
        
        gains_data.append({
            'decile': decile,
            'n': n_obs,
            'events': int(n_events),
            'non_events': int(n_non_events),
            'event_rate': round(event_rate, 4),
            'pct_events': round(pct_events, 4),
            'pct_non_events': round(pct_non_events, 4),
            'cum_events': int(cumulative_events),
            'cum_non_events': int(cumulative_non_events),
            'cum_pct_events': round(cum_pct_events, 4),
            'cum_pct_non_events': round(cum_pct_non_events, 4),
            'ks': round(ks, 4),
            'lift': round(lift, 4),
            'min_prob': round(min_prob, 4),
            'max_prob': round(max_prob, 4),
            'avg_prob': round(avg_prob, 4)
        })
    
    gains_df = pd.DataFrame(gains_data)
    
    return GainsTable(
        table=gains_df,
        total_obs=total_obs,
        total_events=int(total_events),
        total_non_events=int(total_non_events)
    )


def calculate_roc_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate ROC curve metrics.
    
    Returns:
    --------
    fpr, tpr, auc_score, gini_index
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    auc_score = auc(fpr, tpr)
    gini_index = 2 * auc_score - 1
    
    return fpr, tpr, round(auc_score, 5), round(gini_index, 5)


def calculate_ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, int]:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    Returns:
    --------
    ks_statistic, ks_decile (decile where max KS occurs)
    """
    gains = calculate_gains_table(actual, predicted)
    ks_values = gains.table['ks'].values
    ks_statistic = ks_values.max()
    ks_decile = int(np.argmax(ks_values) + 1)
    
    return round(ks_statistic, 4), ks_decile


def calculate_model_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.5) -> ModelMetrics:
    """
    Calculate comprehensive model performance metrics.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # ROC metrics
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
    
    # K-S statistic
    ks_stat, ks_decile = calculate_ks_statistic(actual, predicted)
    
    # Confusion matrix metrics
    predicted_class = (predicted >= threshold).astype(int)
    cm = confusion_matrix(actual, predicted_class)
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    return ModelMetrics(
        auc=auc_score,
        gini=gini,
        ks_statistic=ks_stat,
        ks_decile=ks_decile,
        accuracy=round(accuracy, 4),
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4)
    )


# =============================================================================
# Chart Creation Functions (using Plotly)
# =============================================================================

def create_roc_curve(actual: np.ndarray, predicted: np.ndarray, 
                     model_name: str = "Model", color: str = "#E74C3C") -> go.Figure:
    """Create ROC curve with AUC and Gini index."""
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.4f}, Gini = {gini:.4f})',
        line=dict(color=color, width=2)
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title=dict(text='ROC Curve', font=dict(size=18)),
        xaxis_title='1 - Specificity (False Positive Rate)',
        yaxis_title='Sensitivity (True Positive Rate)',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_roc_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                          test_actual: np.ndarray, test_predicted: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    """Create ROC curve comparing training and test datasets."""
    train_fpr, train_tpr, train_auc, train_gini = calculate_roc_metrics(train_actual, train_predicted)
    test_fpr, test_tpr, test_auc, test_gini = calculate_roc_metrics(test_actual, test_predicted)
    
    fig = go.Figure()
    
    # Training ROC curve
    fig.add_trace(go.Scatter(
        x=train_fpr, y=train_tpr,
        mode='lines',
        name=f'Training (AUC = {train_auc:.4f}, Gini = {train_gini:.4f})',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Test ROC curve
    fig.add_trace(go.Scatter(
        x=test_fpr, y=test_tpr,
        mode='lines',
        name=f'Test (AUC = {test_auc:.4f}, Gini = {test_gini:.4f})',
        line=dict(color='#E74C3C', width=2)
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title=dict(text='ROC Curves - Training vs Test', font=dict(size=18)),
        xaxis_title='1 - Specificity (False Positive Rate)',
        yaxis_title='Sensitivity (True Positive Rate)',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_ks_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create K-S (Kolmogorov-Smirnov) chart (equivalent to blorr::blr_ks_chart)."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    ks_max = df['ks'].max()
    ks_decile = df.loc[df['ks'].idxmax(), 'decile']
    
    fig = go.Figure()
    
    # Cumulative events line
    fig.add_trace(go.Scatter(
        x=df['decile'], y=df['cum_pct_events'],
        mode='lines+markers',
        name='Cumulative % Events (Sensitivity)',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=8)
    ))
    
    # Cumulative non-events line
    fig.add_trace(go.Scatter(
        x=df['decile'], y=df['cum_pct_non_events'],
        mode='lines+markers',
        name='Cumulative % Non-Events (1 - Specificity)',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=8)
    ))
    
    # Mark the maximum K-S point
    ks_row = df[df['decile'] == ks_decile].iloc[0]
    fig.add_annotation(
        x=ks_decile,
        y=(ks_row['cum_pct_events'] + ks_row['cum_pct_non_events']) / 2,
        text=f'Max K-S = {ks_max:.4f}',
        showarrow=True,
        arrowhead=2,
        font=dict(size=12, color='#2C3E50')
    )
    
    # Vertical line at max K-S
    fig.add_vline(
        x=ks_decile,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f'Decile {ks_decile}'
    )
    
    fig.update_layout(
        title=dict(text=f'K-S Chart (Max K-S = {ks_max:.4f} at Decile {ks_decile})', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Percentage',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_lorenz_curve(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve (equivalent to blorr::blr_lorenz_curve)."""
    # Sort by predicted probability descending
    sorted_idx = np.argsort(-predicted)
    actual_sorted = actual[sorted_idx]
    
    n = len(actual_sorted)
    total_events = actual_sorted.sum()
    
    # Calculate cumulative percentages
    cum_pct_pop = np.arange(1, n + 1) / n
    cum_pct_events = np.cumsum(actual_sorted) / total_events
    
    # Subsample for plotting if too many points
    if n > 1000:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        cum_pct_pop = cum_pct_pop[idx]
        cum_pct_events = cum_pct_events[idx]
    
    fig = go.Figure()
    
    # Lorenz curve
    fig.add_trace(go.Scatter(
        x=cum_pct_pop, y=cum_pct_events,
        mode='lines',
        name='Lorenz Curve',
        fill='tozeroy',
        line=dict(color='#3498DB', width=2),
        fillcolor='rgba(52, 152, 219, 0.3)'
    ))
    
    # Diagonal line (perfect equality)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Line of Equality',
        line=dict(color='#E74C3C', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=dict(text='Lorenz Curve', font=dict(size=18)),
        xaxis_title='Cumulative % of Population',
        yaxis_title='Cumulative % of Events',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_decile_lift_chart(actual: np.ndarray, predicted: np.ndarray, 
                             bar_color: str = '#40E0D0') -> go.Figure:
    """Create Decile Lift chart (equivalent to blorr::blr_decile_lift_chart)."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['lift'],
        name='Lift',
        marker_color=bar_color,
        text=df['lift'].round(2),
        textposition='outside'
    ))
    
    # Reference line at lift = 1
    fig.add_hline(
        y=1,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text='Baseline Lift = 1'
    )
    
    fig.update_layout(
        title=dict(text='Decile Lift Chart', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Lift',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_ks_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                         test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create K-S chart comparing training and test datasets."""
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    train_df = train_gains.table
    test_df = test_gains.table
    
    train_ks = train_df['ks'].max()
    test_ks = test_df['ks'].max()
    
    fig = go.Figure()
    
    # Training K-S (difference between cumulative events and non-events)
    fig.add_trace(go.Scatter(
        x=train_df['decile'], y=train_df['ks'],
        mode='lines+markers',
        name=f'Training (K-S = {train_ks:.4f})',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=8)
    ))
    
    # Test K-S
    fig.add_trace(go.Scatter(
        x=test_df['decile'], y=test_df['ks'],
        mode='lines+markers',
        name=f'Test (K-S = {test_ks:.4f})',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=dict(text=f'K-S Chart - Training vs Test', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='K-S Statistic',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_lorenz_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                             test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve comparing training and test datasets."""
    
    def get_lorenz_data(actual, predicted):
        sorted_idx = np.argsort(-predicted)
        actual_sorted = actual[sorted_idx]
        n = len(actual_sorted)
        total_events = actual_sorted.sum()
        if total_events == 0:
            return np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        cum_pct_pop = np.arange(1, n + 1) / n
        cum_pct_events = np.cumsum(actual_sorted) / total_events
        if n > 500:
            idx = np.linspace(0, n - 1, 500, dtype=int)
            cum_pct_pop = cum_pct_pop[idx]
            cum_pct_events = cum_pct_events[idx]
        return cum_pct_pop, cum_pct_events
    
    train_pop, train_events = get_lorenz_data(train_actual, train_predicted)
    test_pop, test_events = get_lorenz_data(test_actual, test_predicted)
    
    fig = go.Figure()
    
    # Training Lorenz
    fig.add_trace(go.Scatter(
        x=train_pop, y=train_events,
        mode='lines',
        name='Training',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Test Lorenz
    fig.add_trace(go.Scatter(
        x=test_pop, y=test_events,
        mode='lines',
        name='Test',
        line=dict(color='#E74C3C', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title=dict(text='Lorenz Curve - Training vs Test', font=dict(size=18)),
        xaxis_title='Cumulative % of Population',
        yaxis_title='Cumulative % of Events',
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_decile_lift_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                   test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Decile Lift chart comparing training and test."""
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    fig = go.Figure()
    
    # Training bars
    fig.add_trace(go.Bar(
        x=train_gains.table['decile'] - 0.2,
        y=train_gains.table['lift'],
        name='Training',
        marker_color='#3498DB',
        text=train_gains.table['lift'].round(2),
        textposition='outside',
        width=0.35
    ))
    
    # Test bars
    fig.add_trace(go.Bar(
        x=test_gains.table['decile'] + 0.2,
        y=test_gains.table['lift'],
        name='Test',
        marker_color='#E74C3C',
        text=test_gains.table['lift'].round(2),
        textposition='outside',
        width=0.35
    ))
    
    # Reference line
    fig.add_hline(y=1, line=dict(color='gray', dash='dash', width=1))
    
    fig.update_layout(
        title=dict(text='Decile Lift - Training vs Test', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Lift',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_event_rate_chart(actual: np.ndarray, predicted: np.ndarray,
                            bar_color: str = '#00CED1') -> go.Figure:
    """Create Event Rate by Decile chart - shows the event rate within each decile."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    
    # Event rate = events / n for each decile (already calculated as 'event_rate')
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['event_rate'] * 100,
        name='Event Rate (%)',
        marker_color=bar_color,
        text=(df['event_rate'] * 100).round(1),
        textposition='outside'
    ))
    
    # Add overall event rate reference line
    overall_rate = df['events'].sum() / df['n'].sum() * 100
    fig.add_hline(
        y=overall_rate,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text=f'Overall Rate: {overall_rate:.1f}%'
    )
    
    fig.update_layout(
        title=dict(text='Event Rate by Decile', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Event Rate (%)',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_event_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                  test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Event Rate by Decile chart comparing training and test."""
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    fig = go.Figure()
    
    # Training bars
    fig.add_trace(go.Bar(
        x=train_gains.table['decile'] - 0.2,
        y=train_gains.table['event_rate'] * 100,
        name='Training',
        marker_color='#3498DB',
        text=(train_gains.table['event_rate'] * 100).round(1),
        textposition='outside',
        width=0.35
    ))
    
    # Test bars
    fig.add_trace(go.Bar(
        x=test_gains.table['decile'] + 0.2,
        y=test_gains.table['event_rate'] * 100,
        name='Test',
        marker_color='#E74C3C',
        text=(test_gains.table['event_rate'] * 100).round(1),
        textposition='outside',
        width=0.35
    ))
    
    fig.update_layout(
        title=dict(text='Event Rate by Decile - Training vs Test', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Event Rate (%)',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        barmode='group',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=600,
        height=500
    )
    
    return fig


def create_capture_rate_chart(actual: np.ndarray, predicted: np.ndarray,
                              bar_color: str = '#27AE60') -> go.Figure:
    """Create Capture Rate by Decile chart - shows % of total events captured in each decile."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    
    # pct_events = what % of all events are in this decile
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['pct_events'] * 100,
        name='Capture Rate (%)',
        marker_color=bar_color,
        text=(df['pct_events'] * 100).round(1),
        textposition='outside'
    ))
    
    # Reference line at 10% (expected if random)
    fig.add_hline(
        y=10,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text='Random: 10%'
    )
    
    fig.update_layout(
        title=dict(text='Capture Rate by Decile', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='% of Total Events Captured',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_capture_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                   test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Capture Rate by Decile chart comparing training and test."""
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    fig = go.Figure()
    
    # Training bars
    fig.add_trace(go.Bar(
        x=train_gains.table['decile'] - 0.2,
        y=train_gains.table['pct_events'] * 100,
        name='Training',
        marker_color='#3498DB',
        text=(train_gains.table['pct_events'] * 100).round(1),
        textposition='outside',
        width=0.35
    ))
    
    # Test bars
    fig.add_trace(go.Bar(
        x=test_gains.table['decile'] + 0.2,
        y=test_gains.table['pct_events'] * 100,
        name='Test',
        marker_color='#E74C3C',
        text=(test_gains.table['pct_events'] * 100).round(1),
        textposition='outside',
        width=0.35
    ))
    
    fig.update_layout(
        title=dict(text='Capture Rate by Decile - Training vs Test', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='% of Total Events Captured',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        barmode='group',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=600,
        height=500
    )
    
    return fig


def create_cumulative_capture_chart(actual: np.ndarray, predicted: np.ndarray,
                                    bar_color: str = '#9B59B6') -> go.Figure:
    """Create Cumulative Capture Rate chart."""
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['cum_pct_events'] * 100,
        name='Cumulative Capture Rate (%)',
        marker_color=bar_color,
        text=(df['cum_pct_events'] * 100).round(1),
        textposition='outside'
    ))
    
    # Reference line (perfect random model)
    fig.add_trace(go.Scatter(
        x=df['decile'],
        y=df['decile'] * 10,
        mode='lines+markers',
        name='Random Model',
        line=dict(color='#E74C3C', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=dict(text='Cumulative Capture Rate by Decile', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative % of Events Captured',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def save_chart(fig: go.Figure, filepath: str) -> None:
    """Save Plotly figure as JPEG image."""
    try:
        fig.write_image(filepath, format='jpeg', width=800, height=600, scale=2)
        print(f"Saved chart to: {filepath}")
    except Exception as e:
        print(f"Error saving chart to {filepath}: {e}")
        # Try PNG fallback
        try:
            png_path = filepath.replace('.jpeg', '.png').replace('.jpg', '.png')
            fig.write_image(png_path, format='png', width=800, height=600, scale=2)
            print(f"Saved chart as PNG to: {png_path}")
        except Exception as e2:
            print(f"Could not save chart: {e2}")


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_model_analyzer_app(
    df: pd.DataFrame,
    dv: Optional[str] = None,
    prob_col: str = "probabilities",
    pred_col: str = "predicted",
    dataset_col: str = "dataset"
):
    """
    Create the Model Analyzer Shiny application.
    
    Parameters:
    -----------
    df : DataFrame
        Input data with predictions and probabilities columns
    dv : str, optional
        Dependent variable (actual values) column name
    prob_col : str
        Column name for predicted probabilities (default: "probabilities")
    pred_col : str
        Column name for predicted class (default: "predicted")
    dataset_col : str
        Column name for dataset indicator (default: "dataset")
    """
    
    app_results = {
        'gains_table': None,
        'metrics': None,
        'completed': False
    }
    
    # Get column options
    columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if dataset column exists and split data
    has_dataset_col = dataset_col in df.columns
    if has_dataset_col:
        unique_datasets = df[dataset_col].dropna().unique().tolist()
        has_training = any(d.lower() == 'training' for d in unique_datasets if isinstance(d, str))
        has_test = any(d.lower() == 'test' for d in unique_datasets if isinstance(d, str))
        
        if has_training and has_test:
            dataset_choices = ["Training", "Test", "Both"]
        elif has_training:
            dataset_choices = ["Training"]
        elif has_test:
            dataset_choices = ["Test"]
        else:
            dataset_choices = ["All Data"]
    else:
        dataset_choices = ["All Data"]
        has_test = False
    
    # Default prob_col if not found
    if prob_col not in df.columns:
        prob_col = numeric_cols[-1] if numeric_cols else columns[0]
    
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                body { 
                    font-family: 'Source Sans Pro', sans-serif; 
                    background: #f5f7fa;
                    min-height: 100vh;
                    color: #2c3e50;
                }
                .card { 
                    background: #ffffff;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                .card-header {
                    color: #2c3e50;
                    font-weight: 700;
                    font-size: 1.1rem;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }
                h3 { 
                    color: #2c3e50; 
                    text-align: center; 
                    font-weight: 700;
                    margin-bottom: 24px;
                }
                .btn-primary { 
                    background: #3498db;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 10px 24px;
                    border-radius: 6px;
                }
                .btn-primary:hover {
                    background: #2980b9;
                }
                .btn-success { 
                    background: #27ae60;
                    border: none;
                    color: white;
                    font-weight: 700;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-success:hover {
                    background: #219a52;
                }
                .form-control, .form-select {
                    background: #ffffff;
                    border: 1px solid #ced4da;
                    color: #2c3e50;
                    border-radius: 6px;
                }
                .form-control:focus, .form-select:focus {
                    background: #ffffff;
                    border-color: #3498db;
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
                    color: #2c3e50;
                }
                .form-label {
                    color: #2c3e50;
                    font-weight: 600;
                }
                .metric-value {
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #2c3e50;
                    text-align: center;
                }
                .metric-label {
                    color: #7f8c8d;
                    text-align: center;
                    font-size: 0.85rem;
                    margin-top: 4px;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                    margin-top: 12px;
                }
                .metric-box {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 14px;
                    text-align: center;
                    border: 1px solid #e1e8ed;
                }
                .metric-box-train {
                    background: #ebf5fb;
                    border: 1px solid #3498db;
                }
                .metric-box-test {
                    background: #fdedec;
                    border: 1px solid #e74c3c;
                }
                .section-title {
                    color: #2c3e50;
                    font-weight: 600;
                    margin-bottom: 8px;
                    padding-bottom: 4px;
                    border-bottom: 2px solid;
                }
                .section-title-train {
                    border-color: #3498db;
                }
                .section-title-test {
                    border-color: #e74c3c;
                }
            """)
        ),
        
        ui.h3("Model Analyzer"),
        
        # Configuration Panel
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Configuration"),
            ui.row(
                ui.column(3,
                    ui.input_text("analyzer_name", "Analyzer Name", value="Logistic Regression"),
                ),
                ui.column(3,
                    ui.input_select("dataset", "Dataset", choices=dataset_choices),
                ),
                ui.column(3,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=columns,
                                   selected=dv if dv and dv in columns else columns[0]),
                ),
                ui.column(3,
                    ui.input_select("prob_col", "Probabilities Column", 
                                   choices=numeric_cols,
                                   selected=prob_col if prob_col in numeric_cols else (numeric_cols[-1] if numeric_cols else None)),
                ),
            ),
        ),
        
        # Metrics Display
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Performance Metrics"),
            ui.output_ui("metrics_display"),
        ),
        
        # Charts Row 1
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 550px;"},
                    ui.div({"class": "card-header"}, "ROC Curve"),
                    output_widget("roc_chart", height="480px")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 550px;"},
                    ui.div({"class": "card-header"}, "K-S Chart"),
                    output_widget("ks_chart", height="480px")
                )
            ),
        ),
        
        # Charts Row 2
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 550px;"},
                    ui.div({"class": "card-header"}, "Lorenz Curve"),
                    output_widget("lorenz_chart", height="480px")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 550px;"},
                    ui.div({"class": "card-header"}, "Other Charts"),
                    ui.input_select("other_chart", "Select Chart", 
                                   choices=["Event Rate by Decile", "Capture Rate by Decile",
                                           "Decile Lift Chart", "Cumulative Capture Rate"]),
                    output_widget("other_chart_display", height="420px")
                )
            ),
        ),
        
        # Gains Table
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Gains Table"),
            ui.output_data_frame("gains_table"),
        ),
        
        # Close Button
        ui.div(
            {"class": "card", "style": "text-align: center; margin-top: 20px;"},
            ui.input_action_button("close_btn", "✓ Complete Analysis", class_="btn btn-success btn-lg"),
        ),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        
        @reactive.Calc
        def get_data():
            """Get actual and predicted values based on selected dataset."""
            dv_col = input.dv()
            selected_prob_col = input.prob_col()
            dataset_choice = input.dataset()
            
            # Filter data by dataset column if it exists
            if has_dataset_col and dataset_col in df.columns:
                if dataset_choice == "Training":
                    subset = df[df[dataset_col].str.lower() == 'training']
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                elif dataset_choice == "Test":
                    subset = df[df[dataset_col].str.lower() == 'test']
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                elif dataset_choice == "Both":
                    train_subset = df[df[dataset_col].str.lower() == 'training']
                    test_subset = df[df[dataset_col].str.lower() == 'test']
                    train_actual = train_subset[dv_col].values
                    train_predicted = train_subset[selected_prob_col].values
                    test_actual = test_subset[dv_col].values
                    test_predicted = test_subset[selected_prob_col].values
                    return train_actual, train_predicted, test_actual, test_predicted
                else:
                    # All Data
                    actual = df[dv_col].values
                    predicted = df[selected_prob_col].values
                    return actual, predicted, None, None
            else:
                # No dataset column - use all data
                actual = df[dv_col].values
                predicted = df[selected_prob_col].values
                return actual, predicted, None, None
        
        @reactive.Calc
        def get_metrics():
            """Calculate model metrics."""
            data = get_data()
            if input.dataset() == "Both":
                train_metrics = calculate_model_metrics(data[0], data[1])
                test_metrics = calculate_model_metrics(data[2], data[3])
                return train_metrics, test_metrics
            else:
                metrics = calculate_model_metrics(data[0], data[1])
                return metrics, None
        
        @output
        @render.ui
        def metrics_display():
            metrics_data = get_metrics()
            
            if input.dataset() == "Both":
                train_m, test_m = metrics_data
                return ui.div(
                    ui.div({"class": "section-title section-title-train"}, "Training"),
                    ui.div(
                        {"class": "metrics-grid"},
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.auc:.4f}"),
                               ui.div({"class": "metric-label"}, "AUC")),
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.gini:.4f}"),
                               ui.div({"class": "metric-label"}, "Gini")),
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.ks_statistic:.4f}"),
                               ui.div({"class": "metric-label"}, f"K-S (Decile {train_m.ks_decile})")),
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.accuracy:.1%}"),
                               ui.div({"class": "metric-label"}, "Accuracy")),
                    ),
                    ui.div({"class": "section-title section-title-test", "style": "margin-top: 16px;"}, "Test"),
                    ui.div(
                        {"class": "metrics-grid"},
                        ui.div({"class": "metric-box metric-box-test"},
                               ui.div({"class": "metric-value"}, f"{test_m.auc:.4f}"),
                               ui.div({"class": "metric-label"}, "AUC")),
                        ui.div({"class": "metric-box metric-box-test"},
                               ui.div({"class": "metric-value"}, f"{test_m.gini:.4f}"),
                               ui.div({"class": "metric-label"}, "Gini")),
                        ui.div({"class": "metric-box metric-box-test"},
                               ui.div({"class": "metric-value"}, f"{test_m.ks_statistic:.4f}"),
                               ui.div({"class": "metric-label"}, f"K-S (Decile {test_m.ks_decile})")),
                        ui.div({"class": "metric-box metric-box-test"},
                               ui.div({"class": "metric-value"}, f"{test_m.accuracy:.1%}"),
                               ui.div({"class": "metric-label"}, "Accuracy")),
                    ),
                )
            else:
                m = metrics_data[0]
                return ui.div(
                    {"class": "metrics-grid"},
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.auc:.4f}"),
                           ui.div({"class": "metric-label"}, "AUC")),
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.gini:.4f}"),
                           ui.div({"class": "metric-label"}, "Gini Index")),
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.ks_statistic:.4f}"),
                           ui.div({"class": "metric-label"}, f"K-S Statistic (Decile {m.ks_decile})")),
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.accuracy:.1%}"),
                           ui.div({"class": "metric-label"}, "Accuracy")),
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.sensitivity:.1%}"),
                           ui.div({"class": "metric-label"}, "Sensitivity")),
                    ui.div({"class": "metric-box"},
                           ui.div({"class": "metric-value"}, f"{m.specificity:.1%}"),
                           ui.div({"class": "metric-label"}, "Specificity")),
                )
        
        @output
        @render_plotly
        def roc_chart():
            data = get_data()
            if input.dataset() == "Both":
                return create_roc_curve_both(data[0], data[1], data[2], data[3], input.analyzer_name())
            else:
                return create_roc_curve(data[0], data[1], input.analyzer_name())
        
        @output
        @render_plotly
        def ks_chart():
            data = get_data()
            if input.dataset() == "Both":
                # Show both training and test K-S curves
                return create_ks_chart_both(data[0], data[1], data[2], data[3])
            else:
                return create_ks_chart(data[0], data[1])
        
        @output
        @render_plotly
        def lorenz_chart():
            data = get_data()
            if input.dataset() == "Both":
                # Show both training and test Lorenz curves
                return create_lorenz_curve_both(data[0], data[1], data[2], data[3])
            else:
                return create_lorenz_curve(data[0], data[1])
        
        @output
        @render_plotly
        def other_chart_display():
            data = get_data()
            chart_type = input.other_chart()
            
            if input.dataset() == "Both":
                # Use "both" versions for comparison
                if chart_type == "Event Rate by Decile":
                    return create_event_rate_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Capture Rate by Decile":
                    return create_capture_rate_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Decile Lift Chart":
                    return create_decile_lift_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Cumulative Capture Rate":
                    return create_cumulative_capture_chart(data[2], data[3])
            else:
                actual, predicted = data[0], data[1]
                if chart_type == "Event Rate by Decile":
                    return create_event_rate_chart(actual, predicted)
                elif chart_type == "Capture Rate by Decile":
                    return create_capture_rate_chart(actual, predicted)
                elif chart_type == "Decile Lift Chart":
                    return create_decile_lift_chart(actual, predicted)
                elif chart_type == "Cumulative Capture Rate":
                    return create_cumulative_capture_chart(actual, predicted)
        
        @output
        @render.data_frame
        def gains_table():
            data = get_data()
            if input.dataset() == "Both":
                # Show both training and test gains tables combined
                train_gains = calculate_gains_table(data[0], data[1])
                test_gains = calculate_gains_table(data[2], data[3])
                train_df = train_gains.table.copy()
                test_df = test_gains.table.copy()
                train_df['dataset'] = 'Training'
                test_df['dataset'] = 'Test'
                combined = pd.concat([train_df, test_df], ignore_index=True)
                # Reorder columns to put dataset first
                cols = ['dataset'] + [c for c in combined.columns if c != 'dataset']
                return combined[cols]
            else:
                gains = calculate_gains_table(data[0], data[1])
                return gains.table
        
        @reactive.Effect
        @reactive.event(input.close_btn)
        async def handle_close():
            data = get_data()
            if input.dataset() == "Both":
                gains = calculate_gains_table(data[2], data[3])
                metrics = calculate_model_metrics(data[2], data[3])
            else:
                gains = calculate_gains_table(data[0], data[1])
                metrics = calculate_model_metrics(data[0], data[1])
            
            app_results['gains_table'] = gains.table
            app_results['metrics'] = {
                'auc': metrics.auc,
                'gini': metrics.gini,
                'ks_statistic': metrics.ks_statistic,
                'ks_decile': metrics.ks_decile,
                'accuracy': metrics.accuracy,
                'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity
            }
            app_results['completed'] = True
            
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def find_free_port(start_port: int = 8051, max_attempts: int = 50) -> int:
    """Find an available port starting from start_port."""
    import socket
    
    for offset in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    # Fallback: let OS assign a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_model_analyzer(
    df: pd.DataFrame,
    dv: Optional[str] = None,
    prob_col: str = "probabilities",
    pred_col: str = "predicted",
    dataset_col: str = "dataset",
    port: int = None
):
    """Run the Model Analyzer application and return results."""
    import threading
    import time
    
    # Find a free port to avoid conflicts with multiple instances
    if port is None:
        port = find_free_port(BASE_PORT)
    
    print(f"Starting Shiny app on port {port}")
    sys.stdout.flush()
    
    app = create_model_analyzer_app(df, dv, prob_col, pred_col, dataset_col)
    
    # Run app in a separate thread so we can monitor completion
    server_thread = None
    server_exception = [None]  # Use list to allow modification in nested function
    
    def run_server():
        try:
            app.run(port=port, launch_browser=True)
        except Exception as e:
            server_exception[0] = e
            print(f"Server stopped: {e}")
            sys.stdout.flush()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for the app to complete (user clicks close button) with timeout check
    timeout_counter = 0
    max_timeout = 7200  # 2 hours max
    
    while not app.results.get('completed', False):
        time.sleep(0.5)
        timeout_counter += 0.5
        
        # Check if server crashed
        if server_exception[0] is not None:
            print(f"Server encountered error: {server_exception[0]}")
            break
            
        if timeout_counter >= max_timeout:
            print("Session timed out after 2 hours")
            break
    
    # Give a moment for cleanup
    time.sleep(0.5)
    print("Analysis complete - returning results")
    sys.stdout.flush()
    
    # Force garbage collection
    gc.collect()
    
    return app.results


# =============================================================================
# Headless Mode Processing
# =============================================================================

def run_headless_analysis(
    df: pd.DataFrame,
    dv: str,
    prob_col: str,
    dataset_col: str,
    analyze_dataset: str,
    model_name: str,
    file_path: str,
    save_roc: bool = False,
    save_capture_rate: bool = False,
    save_ks: bool = False,
    save_lorenz: bool = False,
    save_decile_lift: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run model analysis in headless mode and save charts to files.
    
    Parameters:
    -----------
    df : DataFrame
        Input data with predictions and probabilities
    dv : str
        Dependent variable column name
    prob_col : str
        Probabilities column name
    dataset_col : str
        Dataset indicator column name (values: "Training", "Test")
    analyze_dataset : str
        Which dataset to analyze: "Training", "Test", or "Both"
    model_name : str
        Name for saved chart files
    file_path : str
        Directory path to save charts
    
    Returns:
    --------
    Tuple of (gains_table DataFrame, metrics DataFrame)
    """
    # Ensure file path ends with separator
    if not file_path.endswith(os.sep):
        file_path += os.sep
    
    # Ensure directory exists
    os.makedirs(file_path, exist_ok=True)
    
    all_gains = []
    all_metrics = []
    
    # Check if dataset column exists
    has_dataset_col = dataset_col in df.columns
    
    # Split data by dataset if column exists
    if has_dataset_col:
        df_train = df[df[dataset_col].str.lower() == 'training'].copy()
        df_test = df[df[dataset_col].str.lower() == 'test'].copy()
    else:
        df_train = df.copy()
        df_test = pd.DataFrame()
    
    if analyze_dataset == "Training" or analyze_dataset == "Both":
        if len(df_train) > 0:
            actual = df_train[dv].values
            predicted = df_train[prob_col].values
        
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Training'
            all_gains.append(gains_df)
            
            metrics = calculate_model_metrics(actual, predicted)
            all_metrics.append({
                'dataset': 'Training',
                'auc': metrics.auc,
                'gini': metrics.gini,
                'ks_statistic': metrics.ks_statistic,
                'ks_decile': metrics.ks_decile,
                'accuracy': metrics.accuracy,
                'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity
            })
            
            if analyze_dataset == "Training":
                if save_roc:
                    fig = create_roc_curve(actual, predicted, model_name)
                    save_chart(fig, f"{file_path}{model_name}_Training_ROC.jpeg")
                
                if save_capture_rate:
                    fig = create_event_rate_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Training_CaptureRate.jpeg")
                
                if save_ks:
                    fig = create_ks_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Training_KS.jpeg")
                
                if save_lorenz:
                    fig = create_lorenz_curve(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Training_Lorenz.jpeg")
                
                if save_decile_lift:
                    fig = create_decile_lift_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Training_Lift.jpeg")
    
    if analyze_dataset == "Test" or analyze_dataset == "Both":
        if len(df_test) == 0:
            print("Warning: No 'Test' rows found in dataset column but 'Test' or 'Both' was selected")
        else:
            actual = df_test[dv].values
            predicted = df_test[prob_col].values
            
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Test'
            all_gains.append(gains_df)
            
            metrics = calculate_model_metrics(actual, predicted)
            all_metrics.append({
                'dataset': 'Test',
                'auc': metrics.auc,
                'gini': metrics.gini,
                'ks_statistic': metrics.ks_statistic,
                'ks_decile': metrics.ks_decile,
                'accuracy': metrics.accuracy,
                'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity
            })
            
            if analyze_dataset == "Test":
                if save_roc:
                    fig = create_roc_curve(actual, predicted, model_name)
                    save_chart(fig, f"{file_path}{model_name}_Test_ROC.jpeg")
                
                if save_capture_rate:
                    fig = create_event_rate_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Test_CaptureRate.jpeg")
                
                if save_ks:
                    fig = create_ks_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Test_KS.jpeg")
                
                if save_lorenz:
                    fig = create_lorenz_curve(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Test_Lorenz.jpeg")
                
                if save_decile_lift:
                    fig = create_decile_lift_chart(actual, predicted)
                    save_chart(fig, f"{file_path}{model_name}_Test_Lift.jpeg")
    
    if analyze_dataset == "Both" and len(df_test) > 0 and len(df_train) > 0:
        train_actual = df_train[dv].values
        train_predicted = df_train[prob_col].values
        test_actual = df_test[dv].values
        test_predicted = df_test[prob_col].values
        
        if save_roc:
            fig = create_roc_curve_both(train_actual, train_predicted, 
                                        test_actual, test_predicted, model_name)
            save_chart(fig, f"{file_path}{model_name}_Both_ROC.jpeg")
        
        if save_capture_rate:
            fig = create_event_rate_chart(test_actual, test_predicted)
            save_chart(fig, f"{file_path}{model_name}_Both_CaptureRate.jpeg")
        
        if save_ks:
            fig = create_ks_chart(test_actual, test_predicted)
            save_chart(fig, f"{file_path}{model_name}_Both_KS.jpeg")
        
        if save_lorenz:
            fig = create_lorenz_curve(test_actual, test_predicted)
            save_chart(fig, f"{file_path}{model_name}_Both_Lorenz.jpeg")
        
        if save_decile_lift:
            fig = create_decile_lift_chart(test_actual, test_predicted)
            save_chart(fig, f"{file_path}{model_name}_Both_Lift.jpeg")
    
    # Combine results
    combined_gains = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    
    return combined_gains, metrics_df


# =============================================================================
# Read Input Data
# =============================================================================

# Input 1: Training data (required)
# - All WOE feature columns
# - Dependent variable (actual outcomes 0/1)
# - "predicted" column (predicted class 0/1)
# - "probabilities" column (log-odds / linear predictor from R)
df_train = knio.input_tables[0].to_pandas()
print(f"Input 1 (Training data): {len(df_train)} rows, {len(df_train.columns)} columns")

# Input 2: Coefficients table (required)
# - Row ID = variable names (including "(Intercept)")
# - Column = coefficient values
try:
    df_coef = knio.input_tables[1].to_pandas()
    print(f"Input 2 (Coefficients): {len(df_coef)} rows")
    coefficients = parse_coefficients_table(df_coef)
    has_coefficients = True
except Exception as e:
    print(f"Warning: Could not read coefficients table: {e}")
    coefficients = {}
    has_coefficients = False

# Input 3: Test data (optional)
# - Same WOE feature columns as training
# - Dependent variable (actual outcomes 0/1)
# - Predictions will be computed using coefficients
try:
    df_test = knio.input_tables[2].to_pandas()
    if len(df_test) > 0:
        print(f"Input 3 (Test data): {len(df_test)} rows, {len(df_test.columns)} columns")
        has_test_data = True
    else:
        df_test = None
        has_test_data = False
except:
    df_test = None
    has_test_data = False
    print("No test data provided (Input 3)")

# Check for expected columns in training data
if 'probabilities' in df_train.columns:
    print("Found 'probabilities' column in training data")
elif 'probability' in df_train.columns:
    print("Found 'probability' column (will use as probabilities)")
else:
    print("Warning: No 'probabilities' column found in training data")

if 'predicted' in df_train.columns:
    print("Found 'predicted' column in training data")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================

dv = None
model_name = "Model"
analyze_dataset = "Both" if has_test_data else "Training"
file_path = None

# Column names (with defaults)
prob_col = "probabilities"
pred_col = "predicted"

save_roc = False
save_capture_rate = False
save_ks = False
save_lorenz = False
save_decile_lift = False

# Read flow variables
try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if dv == "missing" or dv == "":
        dv = None
except:
    pass

try:
    model_name = knio.flow_variables.get("ModelName", "Model")
    if model_name == "missing" or model_name == "":
        model_name = "Model"
except:
    pass

try:
    analyze_dataset = knio.flow_variables.get("AnalyzeDataset", analyze_dataset)
except:
    pass

try:
    # Also check for old flow variable name for compatibility
    analyze_dataset = knio.flow_variables.get("Dataset", analyze_dataset)
except:
    pass

try:
    file_path = knio.flow_variables.get("FilePath", None)
    if file_path == "missing" or file_path == "":
        file_path = None
except:
    pass

try:
    prob_col = knio.flow_variables.get("ProbabilitiesColumn", "probabilities")
    if prob_col == "missing" or prob_col == "":
        prob_col = "probabilities"
except:
    pass

try:
    pred_col = knio.flow_variables.get("PredictedColumn", "predicted")
    if pred_col == "missing" or pred_col == "":
        pred_col = "predicted"
except:
    pass

try:
    save_roc = knio.flow_variables.get("saveROC", 0) == 1
except:
    pass

try:
    save_capture_rate = knio.flow_variables.get("saveCaptureRate", 0) == 1
except:
    pass

try:
    save_ks = knio.flow_variables.get("saveK-S", 0) == 1
except:
    pass

try:
    save_lorenz = knio.flow_variables.get("saveLorenzCurve", 0) == 1
except:
    pass

try:
    save_decile_lift = knio.flow_variables.get("saveDecileLift", 0) == 1
except:
    pass

# Auto-detect probabilities column in training data if not found
if prob_col not in df_train.columns:
    for alt_name in ['probability', 'prob', 'probs', 'score', 'pred_prob', 'log_odds']:
        if alt_name in df_train.columns:
            prob_col = alt_name
            print(f"Using '{prob_col}' as probabilities column")
            break
    else:
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            prob_col = numeric_cols[-1]
            print(f"Warning: Using '{prob_col}' as probabilities column (last numeric column)")

# =============================================================================
# Process Training Data - Convert log-odds to probabilities if needed
# =============================================================================

print("\n--- Processing Training Data ---")
df_train['dataset'] = 'Training'

# Convert log-odds to probabilities if needed
if prob_col in df_train.columns:
    raw_values = df_train[prob_col].values
    df_train['probability'] = ensure_probabilities(raw_values, prob_col)
else:
    raise ValueError(f"Probabilities column '{prob_col}' not found in training data")

# =============================================================================
# Process Test Data - Compute predictions using coefficients
# =============================================================================

if has_test_data and has_coefficients:
    print("\n--- Processing Test Data (computing predictions from coefficients) ---")
    
    # Compute predictions for test data using coefficients
    test_probs, test_preds, test_log_odds = predict_with_coefficients(
        df_test, coefficients, return_log_odds=True
    )
    
    # Check for NaN in predictions
    nan_probs = np.isnan(test_probs).sum()
    if nan_probs > 0:
        print(f"Warning: {nan_probs} NaN values in predicted probabilities - filling with 0.5")
        test_probs = np.nan_to_num(test_probs, nan=0.5)
        test_preds = (test_probs >= 0.5).astype(int)
        test_log_odds = np.nan_to_num(test_log_odds, nan=0.0)
    
    df_test['probability'] = test_probs
    df_test['predicted'] = test_preds
    df_test['log_odds'] = test_log_odds
    df_test['dataset'] = 'Test'
    
    print(f"Test predictions computed: {len(df_test)} rows")
    print(f"  Probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    print(f"  Predicted class distribution: 0={np.sum(test_preds==0)}, 1={np.sum(test_preds==1)}")
    
    # Check for NaN in test DV
    if dv and dv in df_test.columns:
        test_dv_nan = df_test[dv].isna().sum()
        if test_dv_nan > 0:
            print(f"Warning: {test_dv_nan} NaN values in test DV column '{dv}'")
            
elif has_test_data and not has_coefficients:
    print("\nWarning: Test data provided but no coefficients table - cannot compute predictions")
    df_test = None
    has_test_data = False

# =============================================================================
# Combine Data for Analysis
# =============================================================================

# Create combined dataframe with 'dataset' column
if has_test_data:
    # Ensure both dataframes have the same columns for analysis
    common_cols = ['probability', 'predicted', 'dataset']
    if dv and dv in df_train.columns and dv in df_test.columns:
        common_cols.insert(0, dv)
    
    df_combined = pd.concat([
        df_train[common_cols], 
        df_test[common_cols]
    ], ignore_index=True)
    
    # Report and handle NaN values
    for col in common_cols:
        nan_count = df_combined[col].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in combined '{col}' column")
    
    # Drop rows with NaN in critical columns for analysis
    critical_cols = ['probability']
    if dv and dv in df_combined.columns:
        critical_cols.append(dv)
    
    before_len = len(df_combined)
    df_combined = df_combined.dropna(subset=critical_cols)
    dropped = before_len - len(df_combined)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN in {critical_cols}")
    
    print(f"\nCombined data: {len(df_combined)} rows (Training: {len(df_train)}, Test: {len(df_test)})")
else:
    df_combined = df_train.copy()
    df_combined['probability'] = df_train['probability']
    
    # Handle NaN in training data
    critical_cols = ['probability']
    if dv and dv in df_combined.columns:
        critical_cols.append(dv)
    
    before_len = len(df_combined)
    df_combined = df_combined.dropna(subset=critical_cols)
    dropped = before_len - len(df_combined)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN in {critical_cols}")
    
    print(f"\nUsing training data only: {len(df_combined)} rows")

# Determine mode
contains_dv = dv is not None and dv in df_combined.columns
contains_file_path = file_path is not None and len(file_path) > 0

print(f"\nDV: {dv}, Analyze Dataset: {analyze_dataset}")
print(f"Mode: {'Headless' if contains_dv and contains_file_path else 'Interactive'}")

# =============================================================================
# Main Processing Logic
# =============================================================================

# Use 'probability' column (already converted from log-odds if needed)
analysis_prob_col = 'probability'

if contains_dv and contains_file_path:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    print(f"\nRunning in headless mode...")
    print(f"Saving charts to: {file_path}")
    
    gains_table, metrics_df = run_headless_analysis(
        df=df_combined,
        dv=dv,
        prob_col=analysis_prob_col,
        dataset_col='dataset',
        analyze_dataset=analyze_dataset,
        model_name=model_name,
        file_path=file_path,
        save_roc=save_roc,
        save_capture_rate=save_capture_rate,
        save_ks=save_ks,
        save_lorenz=save_lorenz,
        save_decile_lift=save_decile_lift
    )
    
    print("Headless analysis completed successfully")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    if not SHINY_AVAILABLE:
        raise RuntimeError("Shiny is not available. Please install shiny and shinywidgets packages.")
    
    print("\nRunning in interactive mode - launching Shiny UI...")
    
    results = run_model_analyzer(
        df=df_combined,
        dv=dv,
        prob_col=analysis_prob_col,
        pred_col='predicted',
        dataset_col='dataset'
    )
    
    if results['completed']:
        gains_table = results['gains_table']
        metrics_dict = results['metrics']
        metrics_df = pd.DataFrame([metrics_dict]) if metrics_dict else pd.DataFrame()
        print("Interactive session completed successfully")
    else:
        print("Interactive session cancelled - returning empty results")
        gains_table = pd.DataFrame()
        metrics_df = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

# Fix column types before output to avoid Arrow conversion errors
print("\nPreparing output tables...")

# Ensure consistent column types in df_combined
if 'predicted' in df_combined.columns:
    df_combined['predicted'] = pd.to_numeric(df_combined['predicted'], errors='coerce').fillna(0).astype('Int32')
if 'probability' in df_combined.columns:
    df_combined['probability'] = pd.to_numeric(df_combined['probability'], errors='coerce').astype('Float64')
if 'dataset' in df_combined.columns:
    df_combined['dataset'] = df_combined['dataset'].astype(str)

# Output 1: Combined data with predictions (training + test if available)
# Includes: original columns + probability + predicted + dataset columns
knio.output_tables[0] = knio.Table.from_pandas(df_combined)

# Output 2: Gains table - ensure numeric types
if isinstance(gains_table, pd.DataFrame) and len(gains_table) > 0:
    # Convert numeric columns
    for col in gains_table.columns:
        if col not in ['dataset']:
            gains_table[col] = pd.to_numeric(gains_table[col], errors='coerce')
    knio.output_tables[1] = knio.Table.from_pandas(gains_table)
else:
    knio.output_tables[1] = knio.Table.from_pandas(pd.DataFrame())

# Output 3: Model performance metrics
if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
    knio.output_tables[2] = knio.Table.from_pandas(metrics_df)
else:
    knio.output_tables[2] = knio.Table.from_pandas(pd.DataFrame())

print("\n" + "="*60)
print("Model Analyzer completed successfully")
print("="*60)

# =============================================================================
# Cleanup for Stability
# =============================================================================
sys.stdout.flush()

# Delete large objects to free memory
try:
    del df_train
except:
    pass

try:
    del df_test
except:
    pass

try:
    del df_combined
except:
    pass

try:
    del df_coef
except:
    pass

# Force garbage collection
gc.collect()
# =============================================================================
# Model Analyzer for KNIME Python Script Node - DEBUG VERSION
# =============================================================================
# This is a DEBUG version with extensive logging on every function.
# Use this version to troubleshoot issues with the Model Analyzer node.
#
# DEBUG FEATURES:
# - Entry/exit logging for all functions
# - Parameter values logged at function entry
# - Return values logged at function exit
# - Key operation logging within functions
# - Exception logging with full stack traces
# - Data shape and content logging
#
# Original Release Date: 2026-01-16
# Debug Version Date: 2026-01-28
# Version: 1.2-DEBUG
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import os
import gc
import sys
import random
import traceback
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================

# Create a custom logger for this module
DEBUG_LOGGER = logging.getLogger('ModelAnalyzer_DEBUG')
DEBUG_LOGGER.setLevel(logging.DEBUG)

# Create console handler with detailed formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter with timestamp, level, and message
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to logger (avoid duplicate handlers)
if not DEBUG_LOGGER.handlers:
    DEBUG_LOGGER.addHandler(console_handler)

def debug_log(message: str, level: str = "DEBUG"):
    """Helper function to log debug messages."""
    if level == "DEBUG":
        DEBUG_LOGGER.debug(message)
    elif level == "INFO":
        DEBUG_LOGGER.info(message)
    elif level == "WARNING":
        DEBUG_LOGGER.warning(message)
    elif level == "ERROR":
        DEBUG_LOGGER.error(message)
    elif level == "CRITICAL":
        DEBUG_LOGGER.critical(message)
    # Also print to stdout for KNIME console visibility
    print(f"[DEBUG] {message}")
    sys.stdout.flush()

def log_function_entry(func_name: str, **kwargs):
    """Log function entry with parameters."""
    debug_log(f">>> ENTERING {func_name}")
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            debug_log(f"    {key}: DataFrame with shape {value.shape}, columns: {list(value.columns)[:10]}...")
        elif isinstance(value, np.ndarray):
            debug_log(f"    {key}: ndarray with shape {value.shape}, dtype: {value.dtype}")
        elif isinstance(value, dict):
            debug_log(f"    {key}: dict with {len(value)} keys: {list(value.keys())[:5]}...")
        elif isinstance(value, (list, tuple)):
            debug_log(f"    {key}: {type(value).__name__} with {len(value)} items")
        else:
            debug_log(f"    {key}: {type(value).__name__} = {value}")

def log_function_exit(func_name: str, result=None):
    """Log function exit with return value."""
    if result is not None:
        if isinstance(result, pd.DataFrame):
            debug_log(f"<<< EXITING {func_name} -> DataFrame with shape {result.shape}")
        elif isinstance(result, np.ndarray):
            debug_log(f"<<< EXITING {func_name} -> ndarray with shape {result.shape}")
        elif isinstance(result, tuple):
            debug_log(f"<<< EXITING {func_name} -> tuple with {len(result)} elements")
        elif isinstance(result, dict):
            debug_log(f"<<< EXITING {func_name} -> dict with {len(result)} keys")
        else:
            debug_log(f"<<< EXITING {func_name} -> {type(result).__name__}: {result}")
    else:
        debug_log(f"<<< EXITING {func_name}")

def log_exception(func_name: str, exception: Exception):
    """Log exception with full stack trace."""
    debug_log(f"!!! EXCEPTION in {func_name}: {type(exception).__name__}: {exception}", "ERROR")
    debug_log(f"    Stack trace:\n{traceback.format_exc()}", "ERROR")

debug_log("=" * 80)
debug_log("MODEL ANALYZER DEBUG VERSION INITIALIZED")
debug_log(f"Timestamp: {datetime.now().isoformat()}")
debug_log(f"Python version: {sys.version}")
debug_log(f"Process ID: {os.getpid()}")
debug_log("=" * 80)

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
BASE_PORT = 8051
RANDOM_PORT_RANGE = 1000

INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
debug_log(f"Instance ID: {INSTANCE_ID}")

os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
debug_log("Threading environment variables set to 1")

# =============================================================================
# Install/Import Dependencies
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install package if not available."""
    log_function_entry("install_if_missing", package=package, import_name=import_name)
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
        debug_log(f"Package '{import_name}' already installed")
        log_function_exit("install_if_missing", result="already_installed")
    except ImportError:
        debug_log(f"Package '{import_name}' not found, installing '{package}'...")
        import subprocess
        try:
            subprocess.check_call(['pip', 'install', package])
            debug_log(f"Successfully installed '{package}'")
            log_function_exit("install_if_missing", result="installed")
        except Exception as e:
            log_exception("install_if_missing", e)
            raise

debug_log("Checking/installing dependencies...")
install_if_missing('scikit-learn', 'sklearn')
install_if_missing('plotly')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('kaleido')

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
debug_log("sklearn imports successful")

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    import plotly.express as px
    SHINY_AVAILABLE = True
    debug_log("Shiny imports successful - Interactive mode AVAILABLE")
except ImportError as e:
    debug_log(f"Shiny import failed: {e}", "WARNING")
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
    """Apply sigmoid function to convert log-odds to probabilities."""
    log_function_entry("sigmoid", x_shape=x.shape if hasattr(x, 'shape') else len(x))
    try:
        x = np.array(x, dtype=float)
        debug_log(f"Input array shape: {x.shape}, dtype: {x.dtype}")
        
        nan_mask = np.isnan(x)
        nan_count = nan_mask.sum()
        debug_log(f"NaN values in input: {nan_count}")
        
        x = np.clip(x, -500, 500)
        debug_log(f"Clipped values range: [{x[~nan_mask].min():.4f}, {x[~nan_mask].max():.4f}]" if nan_count < len(x) else "All NaN")
        
        result = 1 / (1 + np.exp(-x))
        result[nan_mask] = np.nan
        
        debug_log(f"Output probabilities range: [{result[~nan_mask].min():.4f}, {result[~nan_mask].max():.4f}]" if nan_count < len(x) else "All NaN")
        log_function_exit("sigmoid", result=result)
        return result
    except Exception as e:
        log_exception("sigmoid", e)
        raise


def is_log_odds(values: np.ndarray) -> bool:
    """Detect if values are log-odds or probabilities."""
    log_function_entry("is_log_odds", values_len=len(values) if hasattr(values, '__len__') else 1)
    try:
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        debug_log(f"Non-NaN values count: {len(values)}")
        
        if len(values) == 0:
            debug_log("No valid values, returning False")
            log_function_exit("is_log_odds", result=False)
            return False
        
        min_val, max_val = values.min(), values.max()
        debug_log(f"Value range: [{min_val:.6f}, {max_val:.6f}]")
        
        if np.any(values < 0) or np.any(values > 1):
            debug_log("Values outside [0,1] detected - these are LOG-ODDS")
            log_function_exit("is_log_odds", result=True)
            return True
        
        if np.all((values == 0) | (values == 1)):
            debug_log("All values are binary (0 or 1) - these are PROBABILITIES")
            log_function_exit("is_log_odds", result=False)
            return False
        
        debug_log("All values in [0,1] - these are PROBABILITIES")
        log_function_exit("is_log_odds", result=False)
        return False
    except Exception as e:
        log_exception("is_log_odds", e)
        raise


def parse_coefficients_table(coef_df: pd.DataFrame) -> Dict[str, float]:
    """Parse coefficients table from R model output."""
    log_function_entry("parse_coefficients_table", coef_df=coef_df)
    try:
        coefficients = {}
        
        debug_log(f"Coefficients table shape: {coef_df.shape}")
        debug_log(f"Coefficients table columns: {list(coef_df.columns)}")
        debug_log(f"Coefficients table index: {list(coef_df.index)[:10]}...")
        
        numeric_cols = coef_df.select_dtypes(include=[np.number]).columns.tolist()
        debug_log(f"Numeric columns found: {numeric_cols}")
        
        if not numeric_cols:
            raise ValueError("No numeric columns found in coefficients table")
        
        coef_col = numeric_cols[0]
        debug_log(f"Using coefficient column: '{coef_col}'")
        
        for idx, row in coef_df.iterrows():
            var_name = str(idx)
            coef_value = row[coef_col]
            coefficients[var_name] = float(coef_value)
            debug_log(f"  Coefficient: {var_name} = {coef_value:.6f}")
        
        debug_log(f"Loaded {len(coefficients)} coefficients")
        if '(Intercept)' in coefficients:
            debug_log(f"Intercept value: {coefficients['(Intercept)']:.6f}")
        
        log_function_exit("parse_coefficients_table", result=coefficients)
        return coefficients
    except Exception as e:
        log_exception("parse_coefficients_table", e)
        raise


def predict_with_coefficients(
    df: pd.DataFrame,
    coefficients: Dict[str, float],
    return_log_odds: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply logistic regression coefficients to compute predictions."""
    log_function_entry("predict_with_coefficients", df=df, num_coefficients=len(coefficients), return_log_odds=return_log_odds)
    try:
        n = len(df)
        debug_log(f"Computing predictions for {n} observations")
        
        intercept = coefficients.get('(Intercept)', 0.0)
        if pd.isna(intercept):
            intercept = 0.0
        debug_log(f"Intercept: {intercept:.6f}")
        
        log_odds = np.full(n, intercept, dtype=float)
        
        matched_vars = 0
        missing_vars = []
        nan_filled_vars = []
        
        for var_name, coef in coefficients.items():
            if var_name == '(Intercept)':
                continue
            
            if pd.isna(coef):
                debug_log(f"Skipping NaN coefficient for '{var_name}'")
                continue
            
            if var_name in df.columns:
                values = df[var_name].values.astype(float)
                nan_count = np.isnan(values).sum()
                if nan_count > 0:
                    nan_filled_vars.append((var_name, nan_count))
                    debug_log(f"  Variable '{var_name}': {nan_count} NaN values filled with 0")
                values = np.nan_to_num(values, nan=0.0)
                log_odds += coef * values
                matched_vars += 1
                debug_log(f"  Applied coefficient for '{var_name}': coef={coef:.6f}")
            else:
                missing_vars.append(var_name)
        
        if nan_filled_vars:
            debug_log(f"Filled NaN with 0 in {len(nan_filled_vars)} variables")
        
        if missing_vars:
            debug_log(f"WARNING: {len(missing_vars)} coefficient variables not found in data: {missing_vars[:5]}", "WARNING")
        
        debug_log(f"Matched {matched_vars} variables from coefficients")
        debug_log(f"Log-odds range: [{log_odds.min():.4f}, {log_odds.max():.4f}]")
        
        probabilities = sigmoid(log_odds)
        debug_log(f"Probabilities range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        
        predicted_class = (probabilities >= 0.5).astype(int)
        debug_log(f"Predicted class distribution: 0={np.sum(predicted_class==0)}, 1={np.sum(predicted_class==1)}")
        
        if return_log_odds:
            log_function_exit("predict_with_coefficients", result=(probabilities, predicted_class, log_odds))
            return probabilities, predicted_class, log_odds
        
        log_function_exit("predict_with_coefficients", result=(probabilities, predicted_class))
        return probabilities, predicted_class
    except Exception as e:
        log_exception("predict_with_coefficients", e)
        raise


def ensure_probabilities(values: np.ndarray, col_name: str = "values") -> np.ndarray:
    """Ensure values are probabilities (0-1). Convert from log-odds if needed."""
    log_function_entry("ensure_probabilities", values_len=len(values), col_name=col_name)
    try:
        values = np.array(values, dtype=float)
        
        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            debug_log(f"WARNING: {nan_count} NaN values in '{col_name}'", "WARNING")
        
        if is_log_odds(values):
            debug_log(f"Converting '{col_name}' from log-odds to probabilities")
            result = sigmoid(values)
        else:
            debug_log(f"'{col_name}' appears to already be probabilities")
            result = values
        
        log_function_exit("ensure_probabilities", result=result)
        return result
    except Exception as e:
        log_exception("ensure_probabilities", e)
        raise


# =============================================================================
# Core Metric Calculation Functions
# =============================================================================

def calculate_gains_table(actual: np.ndarray, predicted: np.ndarray, n_deciles: int = 10) -> GainsTable:
    """Calculate gains table (equivalent to R's blorr::blr_gains_table)."""
    log_function_entry("calculate_gains_table", actual_len=len(actual), predicted_len=len(predicted), n_deciles=n_deciles)
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        debug_log(f"Actual values - unique: {np.unique(actual)}, NaN count: {np.isnan(actual).sum()}")
        debug_log(f"Predicted values - range: [{predicted[~np.isnan(predicted)].min():.4f}, {predicted[~np.isnan(predicted)].max():.4f}]")
        
        df = pd.DataFrame({
            'actual': actual,
            'predicted': predicted
        })
        df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
        debug_log(f"Sorted DataFrame shape: {df.shape}")
        
        total_obs = len(df)
        total_events = df['actual'].sum()
        total_non_events = total_obs - total_events
        debug_log(f"Totals - obs: {total_obs}, events: {total_events}, non_events: {total_non_events}")
        
        df['decile'] = pd.qcut(range(len(df)), q=n_deciles, labels=False) + 1
        
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
            
            ks = abs(cum_pct_events - cum_pct_non_events)
            
            decile_pct = decile / n_deciles
            lift = cum_pct_events / decile_pct if decile_pct > 0 else 0
            
            min_prob = decile_data['predicted'].min()
            max_prob = decile_data['predicted'].max()
            avg_prob = decile_data['predicted'].mean()
            
            debug_log(f"Decile {decile}: n={n_obs}, events={int(n_events)}, event_rate={event_rate:.4f}, ks={ks:.4f}, lift={lift:.2f}")
            
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
        
        result = GainsTable(
            table=gains_df,
            total_obs=total_obs,
            total_events=int(total_events),
            total_non_events=int(total_non_events)
        )
        
        log_function_exit("calculate_gains_table", result=f"GainsTable with {len(gains_df)} deciles")
        return result
    except Exception as e:
        log_exception("calculate_gains_table", e)
        raise


def calculate_roc_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calculate ROC curve metrics."""
    log_function_entry("calculate_roc_metrics", actual_len=len(actual), predicted_len=len(predicted))
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        debug_log(f"Calculating ROC metrics...")
        debug_log(f"Actual unique values: {np.unique(actual)}")
        debug_log(f"Predicted range: [{predicted.min():.4f}, {predicted.max():.4f}]")
        
        fpr, tpr, thresholds = roc_curve(actual, predicted)
        debug_log(f"ROC curve points: {len(fpr)}")
        
        auc_score = auc(fpr, tpr)
        gini_index = 2 * auc_score - 1
        
        debug_log(f"AUC: {auc_score:.5f}")
        debug_log(f"Gini: {gini_index:.5f}")
        
        log_function_exit("calculate_roc_metrics", result=(auc_score, gini_index))
        return fpr, tpr, round(auc_score, 5), round(gini_index, 5)
    except Exception as e:
        log_exception("calculate_roc_metrics", e)
        raise


def calculate_ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, int]:
    """Calculate Kolmogorov-Smirnov statistic."""
    log_function_entry("calculate_ks_statistic", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        ks_values = gains.table['ks'].values
        ks_statistic = ks_values.max()
        ks_decile = int(np.argmax(ks_values) + 1)
        
        debug_log(f"K-S Statistic: {ks_statistic:.4f} at decile {ks_decile}")
        
        log_function_exit("calculate_ks_statistic", result=(ks_statistic, ks_decile))
        return round(ks_statistic, 4), ks_decile
    except Exception as e:
        log_exception("calculate_ks_statistic", e)
        raise


def calculate_model_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.5) -> ModelMetrics:
    """Calculate comprehensive model performance metrics."""
    log_function_entry("calculate_model_metrics", actual_len=len(actual), predicted_len=len(predicted), threshold=threshold)
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
        ks_stat, ks_decile = calculate_ks_statistic(actual, predicted)
        
        predicted_class = (predicted >= threshold).astype(int)
        cm = confusion_matrix(actual, predicted_class)
        
        debug_log(f"Confusion matrix:\n{cm}")
        
        tn, fp, fn, tp = cm.ravel()
        debug_log(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        debug_log(f"Accuracy: {accuracy:.4f}")
        debug_log(f"Sensitivity: {sensitivity:.4f}")
        debug_log(f"Specificity: {specificity:.4f}")
        
        result = ModelMetrics(
            auc=auc_score,
            gini=gini,
            ks_statistic=ks_stat,
            ks_decile=ks_decile,
            accuracy=round(accuracy, 4),
            sensitivity=round(sensitivity, 4),
            specificity=round(specificity, 4)
        )
        
        log_function_exit("calculate_model_metrics", result=f"ModelMetrics(AUC={auc_score}, Gini={gini}, KS={ks_stat})")
        return result
    except Exception as e:
        log_exception("calculate_model_metrics", e)
        raise


# =============================================================================
# Chart Creation Functions (using Plotly)
# =============================================================================

def create_roc_curve(actual: np.ndarray, predicted: np.ndarray, 
                     model_name: str = "Model", color: str = "#E74C3C") -> go.Figure:
    """Create ROC curve with AUC and Gini index."""
    log_function_entry("create_roc_curve", actual_len=len(actual), predicted_len=len(predicted), model_name=model_name)
    try:
        fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.4f}, Gini = {gini:.4f})',
            line=dict(color=color, width=2)
        ))
        
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
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
            template='plotly_white',
            width=600,
            height=500
        )
        
        debug_log(f"ROC curve created successfully")
        log_function_exit("create_roc_curve", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_roc_curve", e)
        raise


def create_roc_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                          test_actual: np.ndarray, test_predicted: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    """Create ROC curve comparing training and test datasets."""
    log_function_entry("create_roc_curve_both", train_len=len(train_actual), test_len=len(test_actual), model_name=model_name)
    try:
        train_fpr, train_tpr, train_auc, train_gini = calculate_roc_metrics(train_actual, train_predicted)
        test_fpr, test_tpr, test_auc, test_gini = calculate_roc_metrics(test_actual, test_predicted)
        
        debug_log(f"Training - AUC: {train_auc:.4f}, Gini: {train_gini:.4f}")
        debug_log(f"Test - AUC: {test_auc:.4f}, Gini: {test_gini:.4f}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_fpr, y=train_tpr,
            mode='lines',
            name=f'Training (AUC = {train_auc:.4f}, Gini = {train_gini:.4f})',
            line=dict(color='#3498DB', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_fpr, y=test_tpr,
            mode='lines',
            name=f'Test (AUC = {test_auc:.4f}, Gini = {test_gini:.4f})',
            line=dict(color='#E74C3C', width=2)
        ))
        
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
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
            template='plotly_white',
            width=600,
            height=500
        )
        
        log_function_exit("create_roc_curve_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_roc_curve_both", e)
        raise


def create_ks_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create K-S (Kolmogorov-Smirnov) chart."""
    log_function_entry("create_ks_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        ks_max = df['ks'].max()
        ks_decile = df.loc[df['ks'].idxmax(), 'decile']
        debug_log(f"K-S max: {ks_max:.4f} at decile {ks_decile}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['decile'], y=df['cum_pct_events'],
            mode='lines+markers',
            name='Cumulative % Events (Sensitivity)',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['decile'], y=df['cum_pct_non_events'],
            mode='lines+markers',
            name='Cumulative % Non-Events (1 - Specificity)',
            line=dict(color='#E74C3C', width=2),
            marker=dict(size=8)
        ))
        
        ks_row = df[df['decile'] == ks_decile].iloc[0]
        fig.add_annotation(
            x=ks_decile,
            y=(ks_row['cum_pct_events'] + ks_row['cum_pct_non_events']) / 2,
            text=f'Max K-S = {ks_max:.4f}',
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color='#2C3E50')
        )
        
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
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
            template='plotly_white',
            width=600,
            height=500
        )
        
        log_function_exit("create_ks_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_ks_chart", e)
        raise


def create_lorenz_curve(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve."""
    log_function_entry("create_lorenz_curve", actual_len=len(actual), predicted_len=len(predicted))
    try:
        sorted_idx = np.argsort(-predicted)
        actual_sorted = actual[sorted_idx]
        
        n = len(actual_sorted)
        total_events = actual_sorted.sum()
        debug_log(f"Sorted data: n={n}, total_events={total_events}")
        
        cum_pct_pop = np.arange(1, n + 1) / n
        cum_pct_events = np.cumsum(actual_sorted) / total_events
        
        if n > 1000:
            idx = np.linspace(0, n - 1, 500, dtype=int)
            cum_pct_pop = cum_pct_pop[idx]
            cum_pct_events = cum_pct_events[idx]
            debug_log(f"Subsampled to 500 points for plotting")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cum_pct_pop, y=cum_pct_events,
            mode='lines',
            name='Lorenz Curve',
            fill='tozeroy',
            line=dict(color='#3498DB', width=2),
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        
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
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
            template='plotly_white',
            width=600,
            height=500
        )
        
        log_function_exit("create_lorenz_curve", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_lorenz_curve", e)
        raise


def create_decile_lift_chart(actual: np.ndarray, predicted: np.ndarray, 
                             bar_color: str = '#40E0D0') -> go.Figure:
    """Create Decile Lift chart."""
    log_function_entry("create_decile_lift_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        debug_log(f"Lift values by decile: {df['lift'].tolist()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['decile'],
            y=df['lift'],
            name='Lift',
            marker_color=bar_color,
            text=df['lift'].round(2),
            textposition='outside'
        ))
        
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
        
        log_function_exit("create_decile_lift_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_decile_lift_chart", e)
        raise


def create_ks_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                         test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create K-S chart comparing training and test datasets."""
    log_function_entry("create_ks_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        train_df = train_gains.table
        test_df = test_gains.table
        
        train_ks = train_df['ks'].max()
        test_ks = test_df['ks'].max()
        debug_log(f"Training K-S: {train_ks:.4f}, Test K-S: {test_ks:.4f}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_df['decile'], y=train_df['ks'],
            mode='lines+markers',
            name=f'Training (K-S = {train_ks:.4f})',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=8)
        ))
        
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
        
        log_function_exit("create_ks_chart_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_ks_chart_both", e)
        raise


def create_lorenz_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                             test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve comparing training and test datasets."""
    log_function_entry("create_lorenz_curve_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
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
        
        fig.add_trace(go.Scatter(
            x=train_pop, y=train_events,
            mode='lines',
            name='Training',
            line=dict(color='#3498DB', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_pop, y=test_events,
            mode='lines',
            name='Test',
            line=dict(color='#E74C3C', width=2)
        ))
        
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
        
        log_function_exit("create_lorenz_curve_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_lorenz_curve_both", e)
        raise


def create_decile_lift_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                   test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Decile Lift chart comparing training and test."""
    log_function_entry("create_decile_lift_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=train_gains.table['decile'] - 0.2,
            y=train_gains.table['lift'],
            name='Training',
            marker_color='#3498DB',
            text=train_gains.table['lift'].round(2),
            textposition='outside',
            width=0.35
        ))
        
        fig.add_trace(go.Bar(
            x=test_gains.table['decile'] + 0.2,
            y=test_gains.table['lift'],
            name='Test',
            marker_color='#E74C3C',
            text=test_gains.table['lift'].round(2),
            textposition='outside',
            width=0.35
        ))
        
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
        
        log_function_exit("create_decile_lift_chart_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_decile_lift_chart_both", e)
        raise


def create_event_rate_chart(actual: np.ndarray, predicted: np.ndarray,
                            bar_color: str = '#00CED1') -> go.Figure:
    """Create Event Rate by Decile chart."""
    log_function_entry("create_event_rate_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        debug_log(f"Event rates by decile: {df['event_rate'].tolist()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['decile'],
            y=df['event_rate'] * 100,
            name='Event Rate (%)',
            marker_color=bar_color,
            text=(df['event_rate'] * 100).round(1),
            textposition='outside'
        ))
        
        overall_rate = df['events'].sum() / df['n'].sum() * 100
        debug_log(f"Overall event rate: {overall_rate:.2f}%")
        
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
        
        log_function_exit("create_event_rate_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_event_rate_chart", e)
        raise


def create_event_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                  test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Event Rate by Decile chart comparing training and test."""
    log_function_entry("create_event_rate_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=train_gains.table['decile'] - 0.2,
            y=train_gains.table['event_rate'] * 100,
            name='Training',
            marker_color='#3498DB',
            text=(train_gains.table['event_rate'] * 100).round(1),
            textposition='outside',
            width=0.35
        ))
        
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
        
        log_function_exit("create_event_rate_chart_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_event_rate_chart_both", e)
        raise


def create_capture_rate_chart(actual: np.ndarray, predicted: np.ndarray,
                              bar_color: str = '#27AE60') -> go.Figure:
    """Create Capture Rate by Decile chart."""
    log_function_entry("create_capture_rate_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        debug_log(f"Capture rates by decile: {df['pct_events'].tolist()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['decile'],
            y=df['pct_events'] * 100,
            name='Capture Rate (%)',
            marker_color=bar_color,
            text=(df['pct_events'] * 100).round(1),
            textposition='outside'
        ))
        
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
        
        log_function_exit("create_capture_rate_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_capture_rate_chart", e)
        raise


def create_capture_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                   test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Capture Rate by Decile chart comparing training and test."""
    log_function_entry("create_capture_rate_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=train_gains.table['decile'] - 0.2,
            y=train_gains.table['pct_events'] * 100,
            name='Training',
            marker_color='#3498DB',
            text=(train_gains.table['pct_events'] * 100).round(1),
            textposition='outside',
            width=0.35
        ))
        
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
        
        log_function_exit("create_capture_rate_chart_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_capture_rate_chart_both", e)
        raise


def create_cumulative_capture_chart(actual: np.ndarray, predicted: np.ndarray,
                                    bar_color: str = '#9B59B6') -> go.Figure:
    """Create Cumulative Capture Rate chart."""
    log_function_entry("create_cumulative_capture_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        debug_log(f"Cumulative capture rates: {df['cum_pct_events'].tolist()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['decile'],
            y=df['cum_pct_events'] * 100,
            name='Cumulative Capture Rate (%)',
            marker_color=bar_color,
            text=(df['cum_pct_events'] * 100).round(1),
            textposition='outside'
        ))
        
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
        
        log_function_exit("create_cumulative_capture_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_cumulative_capture_chart", e)
        raise


def save_chart(fig: go.Figure, filepath: str) -> None:
    """Save Plotly figure as JPEG image."""
    log_function_entry("save_chart", filepath=filepath)
    try:
        debug_log(f"Attempting to save chart to: {filepath}")
        fig.write_image(filepath, format='jpeg', width=800, height=600, scale=2)
        debug_log(f"Successfully saved chart to: {filepath}")
        log_function_exit("save_chart")
    except Exception as e:
        debug_log(f"Error saving chart to {filepath}: {e}", "WARNING")
        try:
            png_path = filepath.replace('.jpeg', '.png').replace('.jpg', '.png')
            debug_log(f"Attempting PNG fallback: {png_path}")
            fig.write_image(png_path, format='png', width=800, height=600, scale=2)
            debug_log(f"Successfully saved chart as PNG to: {png_path}")
        except Exception as e2:
            log_exception("save_chart", e2)


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
    """Create the Model Analyzer Shiny application."""
    log_function_entry("create_model_analyzer_app", df=df, dv=dv, prob_col=prob_col, pred_col=pred_col, dataset_col=dataset_col)
    
    app_results = {
        'gains_table': None,
        'metrics': None,
        'completed': False
    }
    
    columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    debug_log(f"Available columns: {len(columns)}, Numeric columns: {len(numeric_cols)}")
    
    has_dataset_col = dataset_col in df.columns
    debug_log(f"Dataset column '{dataset_col}' present: {has_dataset_col}")
    
    if has_dataset_col:
        unique_datasets = df[dataset_col].dropna().unique().tolist()
        debug_log(f"Unique datasets: {unique_datasets}")
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
    
    debug_log(f"Dataset choices: {dataset_choices}")
    
    if prob_col not in df.columns:
        prob_col = numeric_cols[-1] if numeric_cols else columns[0]
        debug_log(f"Probabilities column not found, using: {prob_col}")
    
    # [UI code remains the same as original - abbreviated for space]
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                body { font-family: 'Source Sans Pro', sans-serif; background: #f5f7fa; min-height: 100vh; color: #2c3e50; }
                .card { background: #ffffff; border: 1px solid #e1e8ed; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
                .card-header { color: #2c3e50; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
                h3 { color: #2c3e50; text-align: center; font-weight: 700; margin-bottom: 24px; }
                .btn-primary { background: #3498db; border: none; color: white; font-weight: 600; padding: 10px 24px; border-radius: 6px; }
                .btn-success { background: #27ae60; border: none; color: white; font-weight: 700; padding: 12px 32px; border-radius: 6px; font-size: 1.1rem; }
                .form-control, .form-select { background: #ffffff; border: 1px solid #ced4da; color: #2c3e50; border-radius: 6px; }
                .form-label { color: #2c3e50; font-weight: 600; }
                .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; text-align: center; }
                .metric-label { color: #7f8c8d; text-align: center; font-size: 0.85rem; margin-top: 4px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
                .metric-box { background: #f8f9fa; border-radius: 8px; padding: 14px; text-align: center; border: 1px solid #e1e8ed; }
                .metric-box-train { background: #ebf5fb; border: 1px solid #3498db; }
                .metric-box-test { background: #fdedec; border: 1px solid #e74c3c; }
                .section-title { color: #2c3e50; font-weight: 600; margin-bottom: 8px; padding-bottom: 4px; border-bottom: 2px solid; }
                .section-title-train { border-color: #3498db; }
                .section-title-test { border-color: #e74c3c; }
            """)
        ),
        ui.h3("Model Analyzer - DEBUG VERSION"),
        ui.div({"class": "card"},
            ui.div({"class": "card-header"}, "Configuration"),
            ui.row(
                ui.column(3, ui.input_text("analyzer_name", "Analyzer Name", value="Logistic Regression")),
                ui.column(3, ui.input_select("dataset", "Dataset", choices=dataset_choices)),
                ui.column(3, ui.input_select("dv", "Dependent Variable", choices=columns, selected=dv if dv and dv in columns else columns[0])),
                ui.column(3, ui.input_select("prob_col", "Probabilities Column", choices=numeric_cols, selected=prob_col if prob_col in numeric_cols else (numeric_cols[-1] if numeric_cols else None))),
            ),
        ),
        ui.div({"class": "card"},
            ui.div({"class": "card-header"}, "Model Performance Metrics"),
            ui.output_ui("metrics_display"),
        ),
        ui.row(
            ui.column(6, ui.div({"class": "card", "style": "height: 550px;"}, ui.div({"class": "card-header"}, "ROC Curve"), output_widget("roc_chart", height="480px"))),
            ui.column(6, ui.div({"class": "card", "style": "height: 550px;"}, ui.div({"class": "card-header"}, "K-S Chart"), output_widget("ks_chart", height="480px"))),
        ),
        ui.row(
            ui.column(6, ui.div({"class": "card", "style": "height: 550px;"}, ui.div({"class": "card-header"}, "Lorenz Curve"), output_widget("lorenz_chart", height="480px"))),
            ui.column(6, ui.div({"class": "card", "style": "height: 550px;"}, ui.div({"class": "card-header"}, "Other Charts"),
                ui.input_select("other_chart", "Select Chart", choices=["Event Rate by Decile", "Capture Rate by Decile", "Decile Lift Chart", "Cumulative Capture Rate"]),
                output_widget("other_chart_display", height="420px"))),
        ),
        ui.div({"class": "card"}, ui.div({"class": "card-header"}, "Gains Table"), ui.output_data_frame("gains_table")),
        ui.div({"class": "card", "style": "text-align: center; margin-top: 20px;"},
            ui.input_action_button("close_btn", "✓ Complete Analysis", class_="btn btn-success btn-lg")),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        debug_log("Shiny server started")
        
        @reactive.Calc
        def get_data():
            debug_log(f"get_data() called - dataset: {input.dataset()}, dv: {input.dv()}, prob_col: {input.prob_col()}")
            dv_col = input.dv()
            selected_prob_col = input.prob_col()
            dataset_choice = input.dataset()
            
            if has_dataset_col and dataset_col in df.columns:
                if dataset_choice == "Training":
                    subset = df[df[dataset_col].str.lower() == 'training']
                    debug_log(f"Training subset: {len(subset)} rows")
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                elif dataset_choice == "Test":
                    subset = df[df[dataset_col].str.lower() == 'test']
                    debug_log(f"Test subset: {len(subset)} rows")
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                elif dataset_choice == "Both":
                    train_subset = df[df[dataset_col].str.lower() == 'training']
                    test_subset = df[df[dataset_col].str.lower() == 'test']
                    debug_log(f"Both - Training: {len(train_subset)} rows, Test: {len(test_subset)} rows")
                    return train_subset[dv_col].values, train_subset[selected_prob_col].values, test_subset[dv_col].values, test_subset[selected_prob_col].values
                else:
                    actual = df[dv_col].values
                    predicted = df[selected_prob_col].values
                    return actual, predicted, None, None
            else:
                actual = df[dv_col].values
                predicted = df[selected_prob_col].values
                return actual, predicted, None, None
        
        @reactive.Calc
        def get_metrics():
            debug_log("get_metrics() called")
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
            debug_log("Rendering metrics_display")
            metrics_data = get_metrics()
            
            if input.dataset() == "Both":
                train_m, test_m = metrics_data
                return ui.div(
                    ui.div({"class": "section-title section-title-train"}, "Training"),
                    ui.div({"class": "metrics-grid"},
                        ui.div({"class": "metric-box metric-box-train"}, ui.div({"class": "metric-value"}, f"{train_m.auc:.4f}"), ui.div({"class": "metric-label"}, "AUC")),
                        ui.div({"class": "metric-box metric-box-train"}, ui.div({"class": "metric-value"}, f"{train_m.gini:.4f}"), ui.div({"class": "metric-label"}, "Gini")),
                        ui.div({"class": "metric-box metric-box-train"}, ui.div({"class": "metric-value"}, f"{train_m.ks_statistic:.4f}"), ui.div({"class": "metric-label"}, f"K-S (Decile {train_m.ks_decile})")),
                        ui.div({"class": "metric-box metric-box-train"}, ui.div({"class": "metric-value"}, f"{train_m.accuracy:.1%}"), ui.div({"class": "metric-label"}, "Accuracy")),
                    ),
                    ui.div({"class": "section-title section-title-test", "style": "margin-top: 16px;"}, "Test"),
                    ui.div({"class": "metrics-grid"},
                        ui.div({"class": "metric-box metric-box-test"}, ui.div({"class": "metric-value"}, f"{test_m.auc:.4f}"), ui.div({"class": "metric-label"}, "AUC")),
                        ui.div({"class": "metric-box metric-box-test"}, ui.div({"class": "metric-value"}, f"{test_m.gini:.4f}"), ui.div({"class": "metric-label"}, "Gini")),
                        ui.div({"class": "metric-box metric-box-test"}, ui.div({"class": "metric-value"}, f"{test_m.ks_statistic:.4f}"), ui.div({"class": "metric-label"}, f"K-S (Decile {test_m.ks_decile})")),
                        ui.div({"class": "metric-box metric-box-test"}, ui.div({"class": "metric-value"}, f"{test_m.accuracy:.1%}"), ui.div({"class": "metric-label"}, "Accuracy")),
                    ),
                )
            else:
                m = metrics_data[0]
                return ui.div({"class": "metrics-grid"},
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.auc:.4f}"), ui.div({"class": "metric-label"}, "AUC")),
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.gini:.4f}"), ui.div({"class": "metric-label"}, "Gini Index")),
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.ks_statistic:.4f}"), ui.div({"class": "metric-label"}, f"K-S Statistic (Decile {m.ks_decile})")),
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.accuracy:.1%}"), ui.div({"class": "metric-label"}, "Accuracy")),
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.sensitivity:.1%}"), ui.div({"class": "metric-label"}, "Sensitivity")),
                    ui.div({"class": "metric-box"}, ui.div({"class": "metric-value"}, f"{m.specificity:.1%}"), ui.div({"class": "metric-label"}, "Specificity")),
                )
        
        @output
        @render_plotly
        def roc_chart():
            debug_log("Rendering roc_chart")
            data = get_data()
            if input.dataset() == "Both":
                return create_roc_curve_both(data[0], data[1], data[2], data[3], input.analyzer_name())
            else:
                return create_roc_curve(data[0], data[1], input.analyzer_name())
        
        @output
        @render_plotly
        def ks_chart():
            debug_log("Rendering ks_chart")
            data = get_data()
            if input.dataset() == "Both":
                return create_ks_chart_both(data[0], data[1], data[2], data[3])
            else:
                return create_ks_chart(data[0], data[1])
        
        @output
        @render_plotly
        def lorenz_chart():
            debug_log("Rendering lorenz_chart")
            data = get_data()
            if input.dataset() == "Both":
                return create_lorenz_curve_both(data[0], data[1], data[2], data[3])
            else:
                return create_lorenz_curve(data[0], data[1])
        
        @output
        @render_plotly
        def other_chart_display():
            debug_log(f"Rendering other_chart_display: {input.other_chart()}")
            data = get_data()
            chart_type = input.other_chart()
            
            if input.dataset() == "Both":
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
            debug_log("Rendering gains_table")
            data = get_data()
            if input.dataset() == "Both":
                train_gains = calculate_gains_table(data[0], data[1])
                test_gains = calculate_gains_table(data[2], data[3])
                train_df = train_gains.table.copy()
                test_df = test_gains.table.copy()
                train_df['dataset'] = 'Training'
                test_df['dataset'] = 'Test'
                combined = pd.concat([train_df, test_df], ignore_index=True)
                cols = ['dataset'] + [c for c in combined.columns if c != 'dataset']
                return combined[cols]
            else:
                gains = calculate_gains_table(data[0], data[1])
                return gains.table
        
        @reactive.Effect
        @reactive.event(input.close_btn)
        async def handle_close():
            debug_log("Close button clicked - completing analysis")
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
            debug_log("Analysis completed, closing session")
            
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    log_function_exit("create_model_analyzer_app", result="App object")
    return app


def find_free_port(start_port: int = 8051, max_attempts: int = 50) -> int:
    """Find an available port starting from start_port."""
    log_function_entry("find_free_port", start_port=start_port, max_attempts=max_attempts)
    import socket
    
    for offset in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                debug_log(f"Found free port: {port}")
                log_function_exit("find_free_port", result=port)
                return port
        except OSError:
            debug_log(f"Port {port} in use, trying next...")
            continue
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        debug_log(f"Using OS-assigned port: {port}")
        log_function_exit("find_free_port", result=port)
        return port


def run_model_analyzer(
    df: pd.DataFrame,
    dv: Optional[str] = None,
    prob_col: str = "probabilities",
    pred_col: str = "predicted",
    dataset_col: str = "dataset",
    port: int = None
):
    """Run the Model Analyzer application and return results."""
    log_function_entry("run_model_analyzer", df=df, dv=dv, prob_col=prob_col, pred_col=pred_col, dataset_col=dataset_col, port=port)
    import threading
    import time
    
    if port is None:
        port = find_free_port(BASE_PORT)
    
    debug_log(f"Starting Shiny app on port {port}")
    sys.stdout.flush()
    
    app = create_model_analyzer_app(df, dv, prob_col, pred_col, dataset_col)
    
    server_thread = None
    server_exception = [None]
    
    def run_server():
        try:
            debug_log("Server thread starting...")
            app.run(port=port, launch_browser=True)
        except Exception as e:
            server_exception[0] = e
            debug_log(f"Server stopped with exception: {e}", "ERROR")
            sys.stdout.flush()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    debug_log("Server thread started")
    
    timeout_counter = 0
    max_timeout = 7200
    
    while not app.results.get('completed', False):
        time.sleep(0.5)
        timeout_counter += 0.5
        
        if server_exception[0] is not None:
            debug_log(f"Server encountered error: {server_exception[0]}", "ERROR")
            break
            
        if timeout_counter >= max_timeout:
            debug_log("Session timed out after 2 hours", "WARNING")
            break
        
        if int(timeout_counter) % 60 == 0 and timeout_counter > 0:
            debug_log(f"Waiting for user interaction... ({int(timeout_counter)}s elapsed)")
    
    time.sleep(0.5)
    debug_log("Analysis complete - returning results")
    sys.stdout.flush()
    
    gc.collect()
    
    log_function_exit("run_model_analyzer", result=app.results)
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
    """Run model analysis in headless mode and save charts to files."""
    log_function_entry("run_headless_analysis", df=df, dv=dv, prob_col=prob_col, 
                       analyze_dataset=analyze_dataset, model_name=model_name, file_path=file_path,
                       save_roc=save_roc, save_capture_rate=save_capture_rate, save_ks=save_ks,
                       save_lorenz=save_lorenz, save_decile_lift=save_decile_lift)
    try:
        if not file_path.endswith(os.sep):
            file_path += os.sep
        
        debug_log(f"Creating output directory: {file_path}")
        os.makedirs(file_path, exist_ok=True)
        
        all_gains = []
        all_metrics = []
        
        has_dataset_col = dataset_col in df.columns
        debug_log(f"Dataset column present: {has_dataset_col}")
        
        if has_dataset_col:
            df_train = df[df[dataset_col].str.lower() == 'training'].copy()
            df_test = df[df[dataset_col].str.lower() == 'test'].copy()
            debug_log(f"Training rows: {len(df_train)}, Test rows: {len(df_test)}")
        else:
            df_train = df.copy()
            df_test = pd.DataFrame()
        
        if analyze_dataset == "Training" or analyze_dataset == "Both":
            if len(df_train) > 0:
                debug_log("Processing Training data...")
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
                debug_log("WARNING: No 'Test' rows found in dataset column", "WARNING")
            else:
                debug_log("Processing Test data...")
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
            debug_log("Creating comparison charts for Both datasets...")
            train_actual = df_train[dv].values
            train_predicted = df_train[prob_col].values
            test_actual = df_test[dv].values
            test_predicted = df_test[prob_col].values
            
            if save_roc:
                fig = create_roc_curve_both(train_actual, train_predicted, test_actual, test_predicted, model_name)
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
        
        combined_gains = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()
        metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
        
        debug_log(f"Headless analysis complete - gains table: {combined_gains.shape}, metrics: {metrics_df.shape}")
        log_function_exit("run_headless_analysis", result=(combined_gains.shape, metrics_df.shape))
        return combined_gains, metrics_df
    except Exception as e:
        log_exception("run_headless_analysis", e)
        raise


# =============================================================================
# Read Input Data
# =============================================================================

debug_log("=" * 80)
debug_log("READING INPUT DATA")
debug_log("=" * 80)

# Input 1: Training data (required)
debug_log("Reading Input 1 (Training data)...")
df_train = knio.input_tables[0].to_pandas()
debug_log(f"Input 1 shape: {df_train.shape}")
debug_log(f"Input 1 columns: {list(df_train.columns)}")
debug_log(f"Input 1 dtypes:\n{df_train.dtypes}")
debug_log(f"Input 1 first 5 rows:\n{df_train.head()}")

# Input 2: Coefficients table (required)
debug_log("Reading Input 2 (Coefficients)...")
try:
    df_coef = knio.input_tables[1].to_pandas()
    debug_log(f"Input 2 shape: {df_coef.shape}")
    debug_log(f"Input 2 index: {list(df_coef.index)[:10]}...")
    debug_log(f"Input 2 columns: {list(df_coef.columns)}")
    coefficients = parse_coefficients_table(df_coef)
    has_coefficients = True
except Exception as e:
    log_exception("Reading Input 2", e)
    coefficients = {}
    has_coefficients = False

# Input 3: Test data (optional)
debug_log("Reading Input 3 (Test data)...")
try:
    df_test = knio.input_tables[2].to_pandas()
    if len(df_test) > 0:
        debug_log(f"Input 3 shape: {df_test.shape}")
        debug_log(f"Input 3 columns: {list(df_test.columns)}")
        has_test_data = True
    else:
        debug_log("Input 3 is empty")
        df_test = None
        has_test_data = False
except Exception as e:
    debug_log(f"Input 3 not available: {e}")
    df_test = None
    has_test_data = False

# Check for expected columns
if 'probabilities' in df_train.columns:
    debug_log("Found 'probabilities' column in training data")
elif 'probability' in df_train.columns:
    debug_log("Found 'probability' column (will use as probabilities)")
else:
    debug_log("WARNING: No 'probabilities' column found in training data", "WARNING")

if 'predicted' in df_train.columns:
    debug_log("Found 'predicted' column in training data")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================

debug_log("=" * 80)
debug_log("READING FLOW VARIABLES")
debug_log("=" * 80)

dv = None
model_name = "Model"
analyze_dataset = "Both" if has_test_data else "Training"
file_path = None
prob_col = "probabilities"
pred_col = "predicted"
save_roc = False
save_capture_rate = False
save_ks = False
save_lorenz = False
save_decile_lift = False

# Read all flow variables with debug logging
try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if dv == "missing" or dv == "":
        dv = None
    debug_log(f"DependentVariable: {dv}")
except Exception as e:
    debug_log(f"Error reading DependentVariable: {e}")

try:
    model_name = knio.flow_variables.get("ModelName", "Model")
    if model_name == "missing" or model_name == "":
        model_name = "Model"
    debug_log(f"ModelName: {model_name}")
except Exception as e:
    debug_log(f"Error reading ModelName: {e}")

try:
    analyze_dataset = knio.flow_variables.get("AnalyzeDataset", analyze_dataset)
    debug_log(f"AnalyzeDataset: {analyze_dataset}")
except:
    pass

try:
    analyze_dataset = knio.flow_variables.get("Dataset", analyze_dataset)
except:
    pass

try:
    file_path = knio.flow_variables.get("FilePath", None)
    if file_path == "missing" or file_path == "":
        file_path = None
    debug_log(f"FilePath: {file_path}")
except Exception as e:
    debug_log(f"Error reading FilePath: {e}")

try:
    prob_col = knio.flow_variables.get("ProbabilitiesColumn", "probabilities")
    if prob_col == "missing" or prob_col == "":
        prob_col = "probabilities"
    debug_log(f"ProbabilitiesColumn: {prob_col}")
except:
    pass

try:
    pred_col = knio.flow_variables.get("PredictedColumn", "predicted")
    if pred_col == "missing" or pred_col == "":
        pred_col = "predicted"
    debug_log(f"PredictedColumn: {pred_col}")
except:
    pass

try:
    save_roc = knio.flow_variables.get("saveROC", 0) == 1
    debug_log(f"saveROC: {save_roc}")
except:
    pass

try:
    save_capture_rate = knio.flow_variables.get("saveCaptureRate", 0) == 1
    debug_log(f"saveCaptureRate: {save_capture_rate}")
except:
    pass

try:
    save_ks = knio.flow_variables.get("saveK-S", 0) == 1
    debug_log(f"saveK-S: {save_ks}")
except:
    pass

try:
    save_lorenz = knio.flow_variables.get("saveLorenzCurve", 0) == 1
    debug_log(f"saveLorenzCurve: {save_lorenz}")
except:
    pass

try:
    save_decile_lift = knio.flow_variables.get("saveDecileLift", 0) == 1
    debug_log(f"saveDecileLift: {save_decile_lift}")
except:
    pass

# Auto-detect probabilities column
if prob_col not in df_train.columns:
    debug_log(f"Probabilities column '{prob_col}' not found, searching for alternatives...")
    for alt_name in ['probability', 'prob', 'probs', 'score', 'pred_prob', 'log_odds']:
        if alt_name in df_train.columns:
            prob_col = alt_name
            debug_log(f"Using '{prob_col}' as probabilities column")
            break
    else:
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            prob_col = numeric_cols[-1]
            debug_log(f"WARNING: Using '{prob_col}' as probabilities column (last numeric column)", "WARNING")

# =============================================================================
# Process Training Data
# =============================================================================

debug_log("=" * 80)
debug_log("PROCESSING TRAINING DATA")
debug_log("=" * 80)

df_train['dataset'] = 'Training'

if prob_col in df_train.columns:
    raw_values = df_train[prob_col].values
    debug_log(f"Raw probability values - range: [{raw_values[~np.isnan(raw_values)].min():.4f}, {raw_values[~np.isnan(raw_values)].max():.4f}]")
    df_train['probability'] = ensure_probabilities(raw_values, prob_col)
else:
    raise ValueError(f"Probabilities column '{prob_col}' not found in training data")

# =============================================================================
# Process Test Data
# =============================================================================

if has_test_data and has_coefficients:
    debug_log("=" * 80)
    debug_log("PROCESSING TEST DATA")
    debug_log("=" * 80)
    
    test_probs, test_preds, test_log_odds = predict_with_coefficients(
        df_test, coefficients, return_log_odds=True
    )
    
    nan_probs = np.isnan(test_probs).sum()
    if nan_probs > 0:
        debug_log(f"WARNING: {nan_probs} NaN values in predicted probabilities - filling with 0.5", "WARNING")
        test_probs = np.nan_to_num(test_probs, nan=0.5)
        test_preds = (test_probs >= 0.5).astype(int)
        test_log_odds = np.nan_to_num(test_log_odds, nan=0.0)
    
    df_test['probability'] = test_probs
    df_test['predicted'] = test_preds
    df_test['log_odds'] = test_log_odds
    df_test['dataset'] = 'Test'
    
    debug_log(f"Test predictions computed: {len(df_test)} rows")
    debug_log(f"Probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    debug_log(f"Predicted class distribution: 0={np.sum(test_preds==0)}, 1={np.sum(test_preds==1)}")
    
    if dv and dv in df_test.columns:
        test_dv_nan = df_test[dv].isna().sum()
        if test_dv_nan > 0:
            debug_log(f"WARNING: {test_dv_nan} NaN values in test DV column '{dv}'", "WARNING")
            
elif has_test_data and not has_coefficients:
    debug_log("WARNING: Test data provided but no coefficients table - cannot compute predictions", "WARNING")
    df_test = None
    has_test_data = False

# =============================================================================
# Combine Data for Analysis
# =============================================================================

debug_log("=" * 80)
debug_log("COMBINING DATA")
debug_log("=" * 80)

if has_test_data:
    common_cols = ['probability', 'predicted', 'dataset']
    if dv and dv in df_train.columns and dv in df_test.columns:
        common_cols.insert(0, dv)
    
    debug_log(f"Common columns for merge: {common_cols}")
    
    df_combined = pd.concat([
        df_train[common_cols], 
        df_test[common_cols]
    ], ignore_index=True)
    
    for col in common_cols:
        nan_count = df_combined[col].isna().sum()
        if nan_count > 0:
            debug_log(f"WARNING: {nan_count} NaN values in combined '{col}' column", "WARNING")
    
    critical_cols = ['probability']
    if dv and dv in df_combined.columns:
        critical_cols.append(dv)
    
    before_len = len(df_combined)
    df_combined = df_combined.dropna(subset=critical_cols)
    dropped = before_len - len(df_combined)
    if dropped > 0:
        debug_log(f"Dropped {dropped} rows with NaN in {critical_cols}")
    
    debug_log(f"Combined data: {len(df_combined)} rows (Training: {len(df_train)}, Test: {len(df_test)})")
else:
    df_combined = df_train.copy()
    df_combined['probability'] = df_train['probability']
    
    critical_cols = ['probability']
    if dv and dv in df_combined.columns:
        critical_cols.append(dv)
    
    before_len = len(df_combined)
    df_combined = df_combined.dropna(subset=critical_cols)
    dropped = before_len - len(df_combined)
    if dropped > 0:
        debug_log(f"Dropped {dropped} rows with NaN in {critical_cols}")
    
    debug_log(f"Using training data only: {len(df_combined)} rows")

# Determine mode
contains_dv = dv is not None and dv in df_combined.columns
contains_file_path = file_path is not None and len(file_path) > 0

debug_log(f"DV column present: {contains_dv}")
debug_log(f"File path specified: {contains_file_path}")
debug_log(f"Mode: {'HEADLESS' if contains_dv and contains_file_path else 'INTERACTIVE'}")

# =============================================================================
# Main Processing Logic
# =============================================================================

debug_log("=" * 80)
debug_log("MAIN PROCESSING")
debug_log("=" * 80)

analysis_prob_col = 'probability'

if contains_dv and contains_file_path:
    debug_log("Running in HEADLESS mode...")
    debug_log(f"Saving charts to: {file_path}")
    
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
    
    debug_log("Headless analysis completed successfully")

else:
    if not SHINY_AVAILABLE:
        raise RuntimeError("Shiny is not available. Please install shiny and shinywidgets packages.")
    
    debug_log("Running in INTERACTIVE mode - launching Shiny UI...")
    
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
        debug_log("Interactive session completed successfully")
    else:
        debug_log("Interactive session cancelled - returning empty results", "WARNING")
        gains_table = pd.DataFrame()
        metrics_df = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

debug_log("=" * 80)
debug_log("PREPARING OUTPUT TABLES")
debug_log("=" * 80)

# Fix column types
if 'predicted' in df_combined.columns:
    df_combined['predicted'] = pd.to_numeric(df_combined['predicted'], errors='coerce').fillna(0).astype('Int32')
if 'probability' in df_combined.columns:
    df_combined['probability'] = pd.to_numeric(df_combined['probability'], errors='coerce').astype('Float64')
if 'dataset' in df_combined.columns:
    df_combined['dataset'] = df_combined['dataset'].astype(str)

debug_log(f"Output 1 (Combined data): {df_combined.shape}")
knio.output_tables[0] = knio.Table.from_pandas(df_combined)

if isinstance(gains_table, pd.DataFrame) and len(gains_table) > 0:
    for col in gains_table.columns:
        if col not in ['dataset']:
            gains_table[col] = pd.to_numeric(gains_table[col], errors='coerce')
    debug_log(f"Output 2 (Gains table): {gains_table.shape}")
    knio.output_tables[1] = knio.Table.from_pandas(gains_table)
else:
    debug_log("Output 2 (Gains table): Empty DataFrame")
    knio.output_tables[1] = knio.Table.from_pandas(pd.DataFrame())

if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
    debug_log(f"Output 3 (Metrics): {metrics_df.shape}")
    debug_log(f"Metrics:\n{metrics_df}")
    knio.output_tables[2] = knio.Table.from_pandas(metrics_df)
else:
    debug_log("Output 3 (Metrics): Empty DataFrame")
    knio.output_tables[2] = knio.Table.from_pandas(pd.DataFrame())

debug_log("=" * 80)
debug_log("MODEL ANALYZER DEBUG VERSION COMPLETED SUCCESSFULLY")
debug_log(f"Completion time: {datetime.now().isoformat()}")
debug_log("=" * 80)

# =============================================================================
# Cleanup
# =============================================================================
sys.stdout.flush()

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

gc.collect()
debug_log("Cleanup completed")

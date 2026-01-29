# =============================================================================
# Model Analyzer for KNIME Python Script Node - COMMENTATED DEBUG TOGGLE VERSION
# =============================================================================
# This is a DEBUG version of the fully-commentated Model Analyzer script with
# a DEBUG_MODE toggle at the top. Set DEBUG_MODE = True for extensive logging,
# or False for production use without debug output.
#
# DEBUG FEATURES (when enabled):
# - Entry/exit logging for all functions with timestamps
# - Parameter values logged at function entry
# - Return values logged at function exit
# - Key operation logging within functions
# - Exception logging with full stack traces
# - Data shape and content logging
# - All original comments preserved
#
# PURPOSE:
# This script analyzes the performance of credit risk logistic regression models
# by computing various metrics (AUC, Gini, K-S statistic) and generating
# diagnostic charts (ROC curves, Lorenz curves, gains tables, etc.)
#
# The script has two operating modes:
# 1. Interactive Mode (Shiny UI) - Launches when no flow variables are provided
# 2. Headless Mode - Runs when DependentVariable and FilePath flow variables exist
#
# Original Release Date: 2026-01-16
# Toggle Version Date: 2026-01-28
# Version: 1.2-COMMENTATED-DEBUG-TOGGLE
# =============================================================================

# =============================================================================
# DEBUG MODE TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG_MODE = True
# =============================================================================

# =============================================================================
# SECTION 1: IMPORT STATEMENTS
# =============================================================================
# This section imports all the Python libraries and modules needed by this script.
# Each import serves a specific purpose in the analysis workflow.

# Import the KNIME scripting interface - this is the bridge between Python and KNIME
# It provides access to input/output tables and flow variables
import knime.scripting.io as knio

# Import pandas - the primary library for data manipulation in Python
# DataFrames are used throughout this script to hold tabular data
import pandas as pd

# Import numpy - provides fast numerical operations on arrays
# Used for mathematical calculations like sigmoid function, array operations
import numpy as np

# Import warnings module to control warning message display
# We suppress warnings to keep the output clean
import warnings

# Import os module for operating system interactions
# Used for file path handling, environment variables, and directory creation
import os

# Import gc (garbage collector) for memory management
# Explicitly triggers garbage collection to free memory after processing
import gc

# Import sys for system-specific parameters and functions
# Used for flushing output buffers
import sys

# Import random for generating random numbers
# Used to pick random ports to avoid conflicts when multiple instances run
import random

# Import traceback for detailed exception information
import traceback

# Import logging for structured debug output
import logging

# Import datetime for timestamps
from datetime import datetime

# Import typing module for type hints
# These make the code more readable by documenting expected types
from typing import Dict, List, Tuple, Optional, Any, Union

# Import dataclass decorator from dataclasses module
# Dataclasses automatically generate __init__, __repr__, etc. for simple classes
from dataclasses import dataclass

# Suppress all warning messages to keep output clean
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 2: DEBUG LOGGING INFRASTRUCTURE
# =============================================================================
# This section sets up comprehensive debug logging that will output to both
# the Python logger and KNIME's console output.
# All logging functions check DEBUG_MODE before outputting.

# Create a custom logger for this module
# Using a named logger allows filtering and separate configuration
DEBUG_LOGGER = logging.getLogger('ModelAnalyzer_Commentated_DEBUG_TOGGLE')
DEBUG_LOGGER.setLevel(logging.DEBUG)

# Create console handler with detailed formatting
# This ensures all debug output goes to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter with timestamp, level, function name, line number, and message
# This format helps identify exactly where each log message originates
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to logger (avoid duplicate handlers on reimport)
if not DEBUG_LOGGER.handlers:
    DEBUG_LOGGER.addHandler(console_handler)


def debug_log(message: str, level: str = "DEBUG"):
    """
    Helper function to log debug messages to both logger and stdout.
    Only logs if DEBUG_MODE is True.
    
    This ensures messages appear in KNIME's console regardless of
    logging configuration.
    
    Parameters:
    -----------
    message : str
        The message to log
    level : str
        Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL
    """
    if not DEBUG_MODE:
        return
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
    # Also print directly to stdout for KNIME console visibility
    print(f"[DEBUG] {message}")
    sys.stdout.flush()


def log_function_entry(func_name: str, **kwargs):
    """
    Log function entry with all parameter values.
    Only logs if DEBUG_MODE is True.
    
    This helps trace the flow of execution and identify
    what values are being passed to each function.
    
    Parameters:
    -----------
    func_name : str
        Name of the function being entered
    **kwargs : dict
        Parameter names and values to log
    """
    if not DEBUG_MODE:
        return
    debug_log(f">>> ENTERING {func_name}")
    for key, value in kwargs.items():
        # Handle different types appropriately for logging
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
    """
    Log function exit with return value summary.
    Only logs if DEBUG_MODE is True.
    
    Parameters:
    -----------
    func_name : str
        Name of the function being exited
    result : any, optional
        The return value to summarize
    """
    if not DEBUG_MODE:
        return
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
    """
    Log exception with full stack trace.
    Only logs if DEBUG_MODE is True.
    
    Parameters:
    -----------
    func_name : str
        Name of the function where exception occurred
    exception : Exception
        The exception object
    """
    if not DEBUG_MODE:
        return
    debug_log(f"!!! EXCEPTION in {func_name}: {type(exception).__name__}: {exception}", "ERROR")
    debug_log(f"    Stack trace:\n{traceback.format_exc()}", "ERROR")


# Log initialization (only if DEBUG_MODE is True)
if DEBUG_MODE:
    debug_log("=" * 80)
    debug_log("MODEL ANALYZER COMMENTATED DEBUG TOGGLE VERSION INITIALIZED")
    debug_log(f"DEBUG_MODE = {DEBUG_MODE}")
    debug_log(f"Timestamp: {datetime.now().isoformat()}")
    debug_log(f"Python version: {sys.version}")
    debug_log(f"Process ID: {os.getpid()}")
    debug_log("=" * 80)


# =============================================================================
# SECTION 3: STABILITY SETTINGS FOR MULTIPLE INSTANCE EXECUTION
# =============================================================================
# When multiple KNIME nodes run Python scripts simultaneously, they can conflict
# with each other over ports and threading resources. These settings prevent that.

# BASE_PORT defines the starting port number for the Shiny web application
BASE_PORT = 8051

# RANDOM_PORT_RANGE defines how far above BASE_PORT we can randomly select
RANDOM_PORT_RANGE = 1000

# Create a unique identifier for this specific running instance
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
debug_log(f"Instance ID: {INSTANCE_ID}")

# Set environment variables to limit threading in numerical libraries
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
debug_log("Threading environment variables set to 1 for stability")


# =============================================================================
# SECTION 4: INSTALL/IMPORT DEPENDENCIES
# =============================================================================
# This section handles automatic installation of required packages if they
# are not already available in the Python environment.

def install_if_missing(package, import_name=None):
    """
    Install a Python package if it is not already available.
    
    This helper function attempts to import a package. If the import fails
    (meaning the package is not installed), it uses pip to install it.
    
    Parameters:
    -----------
    package : str
        The name of the package as known to pip (e.g., 'scikit-learn')
    import_name : str, optional
        The name used to import the package in Python (e.g., 'sklearn')
    """
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


# Install required packages
debug_log("Checking/installing dependencies...")
install_if_missing('scikit-learn', 'sklearn')
install_if_missing('plotly')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('kaleido')

# Import scikit-learn components
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
debug_log("sklearn imports successful")

# Try to import Shiny and related packages
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
# SECTION 5: DATA CLASSES
# =============================================================================
# Data classes are simple classes that primarily hold data.

@dataclass
class GainsTable:
    """
    Container class for gains table results.
    
    A gains table divides predictions into deciles ranked by predicted probability,
    then calculates various metrics for each decile.
    """
    table: pd.DataFrame      # The DataFrame containing the gains table
    total_obs: int           # Total count of all observations
    total_events: int        # Total count of positive outcomes (1s)
    total_non_events: int    # Total count of negative outcomes (0s)


@dataclass
class ModelMetrics:
    """
    Container class for model performance metrics.
    
    This class holds all the key performance indicators used to evaluate
    a binary classification model's predictive power.
    """
    auc: float           # Area Under ROC Curve (0.5 to 1.0)
    gini: float          # Gini coefficient (0 to 1.0)
    ks_statistic: float  # Maximum K-S separation (0 to 1.0)
    ks_decile: int       # Decile where max K-S occurs (1 to 10)
    accuracy: float      # Overall classification accuracy
    sensitivity: float   # True Positive Rate / Recall
    specificity: float   # True Negative Rate


# =============================================================================
# SECTION 6: LOGISTIC REGRESSION PREDICTION FUNCTIONS
# =============================================================================
# These functions handle the mathematical transformations needed to work
# with logistic regression model outputs.

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid (logistic) function to convert log-odds to probabilities.
    
    The sigmoid function is defined as: sigmoid(x) = 1 / (1 + exp(-x))
    
    This is the inverse of the logit function and converts log-odds
    (which can range from -infinity to +infinity) into probabilities
    (which must be between 0 and 1).
    """
    log_function_entry("sigmoid", x_shape=x.shape if hasattr(x, 'shape') else len(x))
    try:
        # Convert input to numpy array with float type
        x = np.array(x, dtype=float)
        debug_log(f"Input array shape: {x.shape}, dtype: {x.dtype}")
        
        # Create a boolean mask identifying NaN positions
        nan_mask = np.isnan(x)
        nan_count = nan_mask.sum()
        debug_log(f"NaN values in input: {nan_count}")
        
        # Clip values to prevent numerical overflow in exp()
        x = np.clip(x, -500, 500)
        if nan_count < len(x):
            debug_log(f"Clipped values range: [{x[~nan_mask].min():.4f}, {x[~nan_mask].max():.4f}]")
        
        # Apply the sigmoid formula
        result = 1 / (1 + np.exp(-x))
        
        # Restore NaN values
        result[nan_mask] = np.nan
        
        if nan_count < len(result):
            debug_log(f"Output probabilities range: [{result[~nan_mask].min():.4f}, {result[~nan_mask].max():.4f}]")
        
        log_function_exit("sigmoid", result=result)
        return result
    except Exception as e:
        log_exception("sigmoid", e)
        raise


def is_log_odds(values: np.ndarray) -> bool:
    """
    Detect whether values are log-odds or probabilities.
    
    Log-odds can take any real value from -infinity to +infinity.
    Probabilities are constrained to the range [0, 1].
    """
    log_function_entry("is_log_odds", values_len=len(values) if hasattr(values, '__len__') else 1)
    try:
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        debug_log(f"Non-NaN values count: {len(values)}")
        
        if len(values) == 0:
            debug_log("No valid values, returning False (assume probabilities)")
            log_function_exit("is_log_odds", result=False)
            return False
        
        min_val, max_val = values.min(), values.max()
        debug_log(f"Value range: [{min_val:.6f}, {max_val:.6f}]")
        
        # If any values are outside [0, 1], they must be log-odds
        if np.any(values < 0) or np.any(values > 1):
            debug_log("Values outside [0,1] detected - these are LOG-ODDS")
            log_function_exit("is_log_odds", result=True)
            return True
        
        # If all values are exactly 0 or 1, probably predicted classes not log-odds
        if np.all((values == 0) | (values == 1)):
            debug_log("All values are binary (0 or 1) - these are PROBABILITIES/CLASSES")
            log_function_exit("is_log_odds", result=False)
            return False
        
        debug_log("All values in [0,1] - these are PROBABILITIES")
        log_function_exit("is_log_odds", result=False)
        return False
    except Exception as e:
        log_exception("is_log_odds", e)
        raise


def parse_coefficients_table(coef_df: pd.DataFrame) -> Dict[str, float]:
    """
    Parse a coefficients table from R model output into a Python dictionary.
    
    Expected table format:
    - Row ID (index) contains variable names, e.g., "(Intercept)", "WOE_Age"
    - First numeric column contains the coefficient values
    """
    log_function_entry("parse_coefficients_table", coef_df=coef_df)
    try:
        coefficients = {}
        
        debug_log(f"Coefficients table shape: {coef_df.shape}")
        debug_log(f"Coefficients table columns: {list(coef_df.columns)}")
        debug_log(f"Coefficients table index (first 10): {list(coef_df.index)[:10]}")
        
        # Find all columns with numeric data types
        numeric_cols = coef_df.select_dtypes(include=[np.number]).columns.tolist()
        debug_log(f"Numeric columns found: {numeric_cols}")
        
        if not numeric_cols:
            raise ValueError("No numeric columns found in coefficients table")
        
        # Use the first numeric column as the coefficient column
        coef_col = numeric_cols[0]
        debug_log(f"Using coefficient column: '{coef_col}'")
        
        # Iterate through each row
        for idx, row in coef_df.iterrows():
            var_name = str(idx)
            coef_value = row[coef_col]
            coefficients[var_name] = float(coef_value)
            debug_log(f"  Coefficient: {var_name} = {coef_value:.6f}")
        
        debug_log(f"Loaded {len(coefficients)} coefficients total")
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
    """
    Apply logistic regression coefficients to compute predictions.
    
    Mathematical formula:
        log_odds = intercept + sum(coefficient_i × value_i)
        probability = sigmoid(log_odds)
        predicted_class = 1 if probability >= 0.5 else 0
    """
    log_function_entry("predict_with_coefficients", df=df, num_coefficients=len(coefficients), return_log_odds=return_log_odds)
    try:
        n = len(df)
        debug_log(f"Computing predictions for {n} observations")
        
        # Get the intercept
        intercept = coefficients.get('(Intercept)', 0.0)
        if pd.isna(intercept):
            intercept = 0.0
        debug_log(f"Intercept: {intercept:.6f}")
        
        # Initialize log_odds array
        log_odds = np.full(n, intercept, dtype=float)
        
        # Track statistics
        matched_vars = 0
        missing_vars = []
        nan_filled_vars = []
        
        # Apply each coefficient
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
            debug_log(f"WARNING: {len(missing_vars)} coefficient variables not found: {missing_vars[:5]}", "WARNING")
        
        debug_log(f"Matched {matched_vars} variables from coefficients")
        debug_log(f"Log-odds range: [{log_odds.min():.4f}, {log_odds.max():.4f}]")
        
        # Convert to probabilities
        probabilities = sigmoid(log_odds)
        debug_log(f"Probabilities range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        
        # Convert to class predictions
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
    """
    Ensure that values are probabilities (0-1). Convert from log-odds if needed.
    """
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
# SECTION 7: CORE METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_gains_table(actual: np.ndarray, predicted: np.ndarray, n_deciles: int = 10) -> GainsTable:
    """
    Calculate a gains table (equivalent to R's blorr::blr_gains_table).
    
    A gains table divides the population into equal groups (typically deciles)
    ranked by predicted probability, then calculates performance metrics
    for each group.
    """
    log_function_entry("calculate_gains_table", actual_len=len(actual), predicted_len=len(predicted), n_deciles=n_deciles)
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        debug_log(f"Actual values - unique: {np.unique(actual)}, NaN count: {np.isnan(actual).sum()}")
        valid_predicted = predicted[~np.isnan(predicted)]
        debug_log(f"Predicted values - range: [{valid_predicted.min():.4f}, {valid_predicted.max():.4f}]")
        
        # Create DataFrame and sort
        df = pd.DataFrame({'actual': actual, 'predicted': predicted})
        df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
        debug_log(f"Sorted DataFrame shape: {df.shape}")
        
        # Calculate totals
        total_obs = len(df)
        total_events = df['actual'].sum()
        total_non_events = total_obs - total_events
        debug_log(f"Totals - obs: {total_obs}, events: {total_events}, non_events: {total_non_events}")
        
        # Create deciles
        df['decile'] = pd.qcut(range(len(df)), q=n_deciles, labels=False) + 1
        
        # Calculate metrics for each decile
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
            
            min_prob = decile_data['predicted'].min()
            max_prob = decile_data['predicted'].max()
            avg_prob = decile_data['predicted'].mean()
            
            debug_log(f"Decile {decile}: n={n_obs}, events={int(n_events)}, rate={event_rate:.4f}, ks={ks:.4f}, lift={lift:.2f}")
            
            gains_data.append({
                'decile': decile, 'n': n_obs, 'events': int(n_events), 'non_events': int(n_non_events),
                'event_rate': round(event_rate, 4), 'pct_events': round(pct_events, 4),
                'pct_non_events': round(pct_non_events, 4), 'cum_events': int(cumulative_events),
                'cum_non_events': int(cumulative_non_events), 'cum_pct_events': round(cum_pct_events, 4),
                'cum_pct_non_events': round(cum_pct_non_events, 4), 'ks': round(ks, 4),
                'lift': round(lift, 4), 'min_prob': round(min_prob, 4),
                'max_prob': round(max_prob, 4), 'avg_prob': round(avg_prob, 4)
            })
        
        gains_df = pd.DataFrame(gains_data)
        result = GainsTable(table=gains_df, total_obs=total_obs, 
                           total_events=int(total_events), total_non_events=int(total_non_events))
        
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
        
        debug_log(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        result = ModelMetrics(auc=auc_score, gini=gini, ks_statistic=ks_stat, ks_decile=ks_decile,
                             accuracy=round(accuracy, 4), sensitivity=round(sensitivity, 4),
                             specificity=round(specificity, 4))
        
        log_function_exit("calculate_model_metrics", result=f"ModelMetrics(AUC={auc_score}, Gini={gini}, KS={ks_stat})")
        return result
    except Exception as e:
        log_exception("calculate_model_metrics", e)
        raise


# =============================================================================
# SECTION 8: CHART CREATION FUNCTIONS (using Plotly)
# =============================================================================

def create_roc_curve(actual: np.ndarray, predicted: np.ndarray, 
                     model_name: str = "Model", color: str = "#E74C3C") -> go.Figure:
    """Create ROC curve with AUC and Gini index."""
    log_function_entry("create_roc_curve", actual_len=len(actual), predicted_len=len(predicted), model_name=model_name)
    try:
        fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
            name=f'{model_name} (AUC = {auc_score:.4f}, Gini = {gini:.4f})',
            line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            name='Random (AUC = 0.5)', line=dict(color='gray', dash='dash', width=1)))
        
        fig.update_layout(title=dict(text='ROC Curve', font=dict(size=18)),
            xaxis_title='1 - Specificity (False Positive Rate)',
            yaxis_title='Sensitivity (True Positive Rate)',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
        debug_log("ROC curve created successfully")
        log_function_exit("create_roc_curve", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_roc_curve", e)
        raise


def create_roc_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                          test_actual: np.ndarray, test_predicted: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    """Create ROC curve comparing training and test datasets."""
    log_function_entry("create_roc_curve_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_fpr, train_tpr, train_auc, train_gini = calculate_roc_metrics(train_actual, train_predicted)
        test_fpr, test_tpr, test_auc, test_gini = calculate_roc_metrics(test_actual, test_predicted)
        
        debug_log(f"Training - AUC: {train_auc:.4f}, Gini: {train_gini:.4f}")
        debug_log(f"Test - AUC: {test_auc:.4f}, Gini: {test_gini:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_fpr, y=train_tpr, mode='lines',
            name=f'Training (AUC = {train_auc:.4f})', line=dict(color='#3498DB', width=2)))
        fig.add_trace(go.Scatter(x=test_fpr, y=test_tpr, mode='lines',
            name=f'Test (AUC = {test_auc:.4f})', line=dict(color='#E74C3C', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash', width=1)))
        
        fig.update_layout(title=dict(text='ROC Curves - Training vs Test', font=dict(size=18)),
            xaxis_title='1 - Specificity', yaxis_title='Sensitivity',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_roc_curve_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_roc_curve_both", e)
        raise


def create_ks_chart(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    """Create K-S chart."""
    log_function_entry("create_ks_chart", actual_len=len(actual), predicted_len=len(predicted))
    try:
        gains = calculate_gains_table(actual, predicted)
        df = gains.table
        
        ks_max = df['ks'].max()
        ks_decile = df.loc[df['ks'].idxmax(), 'decile']
        debug_log(f"K-S max: {ks_max:.4f} at decile {ks_decile}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['decile'], y=df['cum_pct_events'], mode='lines+markers',
            name='Cumulative % Events', line=dict(color='#3498DB', width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=df['decile'], y=df['cum_pct_non_events'], mode='lines+markers',
            name='Cumulative % Non-Events', line=dict(color='#E74C3C', width=2), marker=dict(size=8)))
        
        ks_row = df[df['decile'] == ks_decile].iloc[0]
        fig.add_annotation(x=ks_decile, y=(ks_row['cum_pct_events'] + ks_row['cum_pct_non_events']) / 2,
            text=f'Max K-S = {ks_max:.4f}', showarrow=True, arrowhead=2)
        fig.add_vline(x=ks_decile, line=dict(color='green', dash='dash', width=2))
        
        fig.update_layout(title=dict(text=f'K-S Chart (Max = {ks_max:.4f})', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Cumulative Percentage',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template='plotly_white', width=600, height=500)
        
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
            debug_log("Subsampled to 500 points")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_pct_pop, y=cum_pct_events, mode='lines',
            name='Lorenz Curve', fill='tozeroy', line=dict(color='#3498DB', width=2),
            fillcolor='rgba(52, 152, 219, 0.3)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            name='Line of Equality', line=dict(color='#E74C3C', dash='dash', width=2)))
        
        fig.update_layout(title=dict(text='Lorenz Curve', font=dict(size=18)),
            xaxis_title='Cumulative % of Population', yaxis_title='Cumulative % of Events',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
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
        debug_log(f"Lift values: {df['lift'].tolist()}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['decile'], y=df['lift'], name='Lift',
            marker_color=bar_color, text=df['lift'].round(2), textposition='outside'))
        fig.add_hline(y=1, line=dict(color='#E74C3C', dash='dash', width=2),
            annotation_text='Baseline Lift = 1')
        
        fig.update_layout(title=dict(text='Decile Lift Chart', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Cumulative Lift',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_decile_lift_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_decile_lift_chart", e)
        raise


def create_ks_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                         test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create K-S chart comparing training and test."""
    log_function_entry("create_ks_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        train_ks = train_gains.table['ks'].max()
        test_ks = test_gains.table['ks'].max()
        debug_log(f"Training K-S: {train_ks:.4f}, Test K-S: {test_ks:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_gains.table['decile'], y=train_gains.table['ks'],
            mode='lines+markers', name=f'Training (K-S = {train_ks:.4f})',
            line=dict(color='#3498DB', width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=test_gains.table['decile'], y=test_gains.table['ks'],
            mode='lines+markers', name=f'Test (K-S = {test_ks:.4f})',
            line=dict(color='#E74C3C', width=2), marker=dict(size=8)))
        
        fig.update_layout(title=dict(text='K-S Chart - Training vs Test', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='K-S Statistic',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_ks_chart_both", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_ks_chart_both", e)
        raise


def create_lorenz_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                             test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Lorenz curve comparing training and test."""
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
                return cum_pct_pop[idx], cum_pct_events[idx]
            return cum_pct_pop, cum_pct_events
        
        train_pop, train_events = get_lorenz_data(train_actual, train_predicted)
        test_pop, test_events = get_lorenz_data(test_actual, test_predicted)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_pop, y=train_events, mode='lines',
            name='Training', line=dict(color='#3498DB', width=2)))
        fig.add_trace(go.Scatter(x=test_pop, y=test_events, mode='lines',
            name='Test', line=dict(color='#E74C3C', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash', width=1)))
        
        fig.update_layout(title=dict(text='Lorenz Curve - Training vs Test', font=dict(size=18)),
            xaxis_title='Cumulative % of Population', yaxis_title='Cumulative % of Events',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
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
        fig.add_trace(go.Bar(x=train_gains.table['decile'] - 0.2, y=train_gains.table['lift'],
            name='Training', marker_color='#3498DB', width=0.35))
        fig.add_trace(go.Bar(x=test_gains.table['decile'] + 0.2, y=test_gains.table['lift'],
            name='Test', marker_color='#E74C3C', width=0.35))
        fig.add_hline(y=1, line=dict(color='gray', dash='dash', width=1))
        
        fig.update_layout(title=dict(text='Decile Lift - Training vs Test', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Cumulative Lift',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1), barmode='group',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            template='plotly_white', width=600, height=500)
        
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
        debug_log(f"Event rates: {df['event_rate'].tolist()}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['decile'], y=df['event_rate'] * 100, name='Event Rate (%)',
            marker_color=bar_color, text=(df['event_rate'] * 100).round(1), textposition='outside'))
        
        overall_rate = df['events'].sum() / df['n'].sum() * 100
        fig.add_hline(y=overall_rate, line=dict(color='#E74C3C', dash='dash', width=2),
            annotation_text=f'Overall: {overall_rate:.1f}%')
        
        fig.update_layout(title=dict(text='Event Rate by Decile', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Event Rate (%)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_event_rate_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_event_rate_chart", e)
        raise


def create_event_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                  test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Event Rate chart comparing training and test."""
    log_function_entry("create_event_rate_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=train_gains.table['decile'] - 0.2,
            y=train_gains.table['event_rate'] * 100, name='Training', marker_color='#3498DB', width=0.35))
        fig.add_trace(go.Bar(x=test_gains.table['decile'] + 0.2,
            y=test_gains.table['event_rate'] * 100, name='Test', marker_color='#E74C3C', width=0.35))
        
        fig.update_layout(title=dict(text='Event Rate - Training vs Test', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Event Rate (%)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1), barmode='group',
            template='plotly_white', width=600, height=500)
        
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
        debug_log(f"Capture rates: {df['pct_events'].tolist()}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['decile'], y=df['pct_events'] * 100, name='Capture Rate (%)',
            marker_color=bar_color, text=(df['pct_events'] * 100).round(1), textposition='outside'))
        fig.add_hline(y=10, line=dict(color='#E74C3C', dash='dash', width=2),
            annotation_text='Random: 10%')
        
        fig.update_layout(title=dict(text='Capture Rate by Decile', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='% of Total Events',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_capture_rate_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_capture_rate_chart", e)
        raise


def create_capture_rate_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                                   test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """Create Capture Rate chart comparing training and test."""
    log_function_entry("create_capture_rate_chart_both", train_len=len(train_actual), test_len=len(test_actual))
    try:
        train_gains = calculate_gains_table(train_actual, train_predicted)
        test_gains = calculate_gains_table(test_actual, test_predicted)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=train_gains.table['decile'] - 0.2,
            y=train_gains.table['pct_events'] * 100, name='Training', marker_color='#3498DB', width=0.35))
        fig.add_trace(go.Bar(x=test_gains.table['decile'] + 0.2,
            y=test_gains.table['pct_events'] * 100, name='Test', marker_color='#E74C3C', width=0.35))
        
        fig.update_layout(title=dict(text='Capture Rate - Training vs Test', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='% of Total Events',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1), barmode='group',
            template='plotly_white', width=600, height=500)
        
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
        debug_log(f"Cumulative capture: {df['cum_pct_events'].tolist()}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['decile'], y=df['cum_pct_events'] * 100,
            name='Cumulative Capture (%)', marker_color=bar_color,
            text=(df['cum_pct_events'] * 100).round(1), textposition='outside'))
        fig.add_trace(go.Scatter(x=df['decile'], y=df['decile'] * 10,
            mode='lines+markers', name='Random Model', line=dict(color='#E74C3C', dash='dash', width=2)))
        
        fig.update_layout(title=dict(text='Cumulative Capture Rate', font=dict(size=18)),
            xaxis_title='Decile', yaxis_title='Cumulative % Events',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            template='plotly_white', width=600, height=500)
        
        log_function_exit("create_cumulative_capture_chart", result="Figure object")
        return fig
    except Exception as e:
        log_exception("create_cumulative_capture_chart", e)
        raise


def save_chart(fig: go.Figure, filepath: str) -> None:
    """Save Plotly figure as image."""
    log_function_entry("save_chart", filepath=filepath)
    try:
        debug_log(f"Saving chart to: {filepath}")
        fig.write_image(filepath, format='jpeg', width=800, height=600, scale=2)
        debug_log(f"Successfully saved: {filepath}")
        log_function_exit("save_chart")
    except Exception as e:
        debug_log(f"JPEG save failed, trying PNG: {e}", "WARNING")
        try:
            png_path = filepath.replace('.jpeg', '.png').replace('.jpg', '.png')
            fig.write_image(png_path, format='png', width=800, height=600, scale=2)
            debug_log(f"Saved as PNG: {png_path}")
        except Exception as e2:
            log_exception("save_chart", e2)


# =============================================================================
# SECTION 9: SHINY UI APPLICATION (Simplified for this version)
# =============================================================================
# Note: This commentated version includes a simplified Shiny app stub.
# For full interactive support, use model_analyzer_knime_DEBUG_TOGGLE.py

def create_model_analyzer_app(df, dv=None, prob_col="probabilities", pred_col="predicted", dataset_col="dataset"):
    """Create the Model Analyzer Shiny application with debug logging."""
    log_function_entry("create_model_analyzer_app", df=df, dv=dv, prob_col=prob_col)
    debug_log("Creating Shiny app - for full implementation, use model_analyzer_knime_DEBUG_TOGGLE.py")
    raise NotImplementedError("Use model_analyzer_knime_DEBUG_TOGGLE.py for full Shiny implementation")


def find_free_port(start_port: int = 8051, max_attempts: int = 50) -> int:
    """Find an available port."""
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
            continue
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        debug_log(f"Using OS-assigned port: {port}")
        return port


def run_model_analyzer(df, dv=None, prob_col="probabilities", pred_col="predicted", 
                       dataset_col="dataset", port=None):
    """Run the Model Analyzer application."""
    log_function_entry("run_model_analyzer", df=df, dv=dv, prob_col=prob_col)
    debug_log("Use model_analyzer_knime_DEBUG_TOGGLE.py for full interactive implementation")
    raise NotImplementedError("Use model_analyzer_knime_DEBUG_TOGGLE.py for Shiny app")


# =============================================================================
# SECTION 10: HEADLESS MODE PROCESSING
# =============================================================================

def run_headless_analysis(df, dv, prob_col, dataset_col, analyze_dataset, model_name, file_path,
                          save_roc=False, save_capture_rate=False, save_ks=False,
                          save_lorenz=False, save_decile_lift=False):
    """Run model analysis in headless mode."""
    log_function_entry("run_headless_analysis", df=df, dv=dv, analyze_dataset=analyze_dataset,
                       file_path=file_path, save_roc=save_roc, save_ks=save_ks)
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
            debug_log(f"Training: {len(df_train)} rows, Test: {len(df_test)} rows")
        else:
            df_train = df.copy()
            df_test = pd.DataFrame()
        
        # Process Training data
        if analyze_dataset in ["Training", "Both"] and len(df_train) > 0:
            debug_log("Processing Training data...")
            actual = df_train[dv].values
            predicted = df_train[prob_col].values
            
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Training'
            all_gains.append(gains_df)
            
            metrics = calculate_model_metrics(actual, predicted)
            all_metrics.append({
                'dataset': 'Training', 'auc': metrics.auc, 'gini': metrics.gini,
                'ks_statistic': metrics.ks_statistic, 'ks_decile': metrics.ks_decile,
                'accuracy': metrics.accuracy, 'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity
            })
            
            if analyze_dataset == "Training":
                if save_roc:
                    save_chart(create_roc_curve(actual, predicted, model_name),
                              f"{file_path}{model_name}_Training_ROC.jpeg")
                if save_capture_rate:
                    save_chart(create_event_rate_chart(actual, predicted),
                              f"{file_path}{model_name}_Training_CaptureRate.jpeg")
                if save_ks:
                    save_chart(create_ks_chart(actual, predicted),
                              f"{file_path}{model_name}_Training_KS.jpeg")
                if save_lorenz:
                    save_chart(create_lorenz_curve(actual, predicted),
                              f"{file_path}{model_name}_Training_Lorenz.jpeg")
                if save_decile_lift:
                    save_chart(create_decile_lift_chart(actual, predicted),
                              f"{file_path}{model_name}_Training_Lift.jpeg")
        
        # Process Test data
        if analyze_dataset in ["Test", "Both"] and len(df_test) > 0:
            debug_log("Processing Test data...")
            actual = df_test[dv].values
            predicted = df_test[prob_col].values
            
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Test'
            all_gains.append(gains_df)
            
            metrics = calculate_model_metrics(actual, predicted)
            all_metrics.append({
                'dataset': 'Test', 'auc': metrics.auc, 'gini': metrics.gini,
                'ks_statistic': metrics.ks_statistic, 'ks_decile': metrics.ks_decile,
                'accuracy': metrics.accuracy, 'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity
            })
            
            if analyze_dataset == "Test":
                if save_roc:
                    save_chart(create_roc_curve(actual, predicted, model_name),
                              f"{file_path}{model_name}_Test_ROC.jpeg")
                if save_capture_rate:
                    save_chart(create_event_rate_chart(actual, predicted),
                              f"{file_path}{model_name}_Test_CaptureRate.jpeg")
                if save_ks:
                    save_chart(create_ks_chart(actual, predicted),
                              f"{file_path}{model_name}_Test_KS.jpeg")
                if save_lorenz:
                    save_chart(create_lorenz_curve(actual, predicted),
                              f"{file_path}{model_name}_Test_Lorenz.jpeg")
                if save_decile_lift:
                    save_chart(create_decile_lift_chart(actual, predicted),
                              f"{file_path}{model_name}_Test_Lift.jpeg")
        
        # Both datasets comparison
        if analyze_dataset == "Both" and len(df_train) > 0 and len(df_test) > 0:
            debug_log("Creating comparison charts...")
            train_actual = df_train[dv].values
            train_predicted = df_train[prob_col].values
            test_actual = df_test[dv].values
            test_predicted = df_test[prob_col].values
            
            if save_roc:
                save_chart(create_roc_curve_both(train_actual, train_predicted, test_actual, test_predicted, model_name),
                          f"{file_path}{model_name}_Both_ROC.jpeg")
            if save_capture_rate:
                save_chart(create_event_rate_chart(test_actual, test_predicted),
                          f"{file_path}{model_name}_Both_CaptureRate.jpeg")
            if save_ks:
                save_chart(create_ks_chart(test_actual, test_predicted),
                          f"{file_path}{model_name}_Both_KS.jpeg")
            if save_lorenz:
                save_chart(create_lorenz_curve(test_actual, test_predicted),
                          f"{file_path}{model_name}_Both_Lorenz.jpeg")
            if save_decile_lift:
                save_chart(create_decile_lift_chart(test_actual, test_predicted),
                          f"{file_path}{model_name}_Both_Lift.jpeg")
        
        combined_gains = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()
        metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
        
        debug_log(f"Headless analysis complete - gains: {combined_gains.shape}, metrics: {metrics_df.shape}")
        log_function_exit("run_headless_analysis", result=(combined_gains.shape, metrics_df.shape))
        return combined_gains, metrics_df
    except Exception as e:
        log_exception("run_headless_analysis", e)
        raise


# =============================================================================
# SECTION 11: MAIN EXECUTION - KNIME INPUT/OUTPUT
# =============================================================================

debug_log("=" * 80)
debug_log("READING INPUT DATA")
debug_log("=" * 80)

# Input 1: Training data
debug_log("Reading Input 1 (Training data)...")
df_train = knio.input_tables[0].to_pandas()
debug_log(f"Input 1 shape: {df_train.shape}")
debug_log(f"Input 1 columns: {list(df_train.columns)}")

# Input 2: Coefficients
debug_log("Reading Input 2 (Coefficients)...")
try:
    df_coef = knio.input_tables[1].to_pandas()
    debug_log(f"Input 2 shape: {df_coef.shape}")
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
        has_test_data = True
    else:
        df_test = None
        has_test_data = False
except:
    df_test = None
    has_test_data = False
    debug_log("Input 3 not available")

# Read flow variables
debug_log("=" * 80)
debug_log("READING FLOW VARIABLES")
debug_log("=" * 80)

dv = None
model_name = "Model"
analyze_dataset = "Both" if has_test_data else "Training"
file_path = None
prob_col = "probabilities"
save_roc = save_capture_rate = save_ks = save_lorenz = save_decile_lift = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if dv in ["missing", ""]: dv = None
    debug_log(f"DependentVariable: {dv}")
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
    debug_log(f"FilePath: {file_path}")
except: pass

try:
    prob_col = knio.flow_variables.get("ProbabilitiesColumn", "probabilities")
    if prob_col in ["missing", ""]: prob_col = "probabilities"
except: pass

try: save_roc = knio.flow_variables.get("saveROC", 0) == 1
except: pass
try: save_capture_rate = knio.flow_variables.get("saveCaptureRate", 0) == 1
except: pass
try: save_ks = knio.flow_variables.get("saveK-S", 0) == 1
except: pass
try: save_lorenz = knio.flow_variables.get("saveLorenzCurve", 0) == 1
except: pass
try: save_decile_lift = knio.flow_variables.get("saveDecileLift", 0) == 1
except: pass

# Auto-detect probabilities column
if prob_col not in df_train.columns:
    for alt in ['probability', 'prob', 'probs', 'score', 'pred_prob', 'log_odds']:
        if alt in df_train.columns:
            prob_col = alt
            debug_log(f"Using '{prob_col}' as probabilities column")
            break
    else:
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            prob_col = numeric_cols[-1]
            debug_log(f"Using last numeric column '{prob_col}' as probabilities", "WARNING")

# Process data
debug_log("=" * 80)
debug_log("PROCESSING DATA")
debug_log("=" * 80)

df_train['dataset'] = 'Training'
df_train['probability'] = ensure_probabilities(df_train[prob_col].values, prob_col)

if has_test_data and has_coefficients:
    debug_log("Computing test predictions from coefficients...")
    test_probs, test_preds, test_log_odds = predict_with_coefficients(df_test, coefficients, return_log_odds=True)
    df_test['probability'] = test_probs
    df_test['predicted'] = test_preds
    df_test['dataset'] = 'Test'
elif has_test_data:
    debug_log("Test data provided but no coefficients - cannot compute predictions", "WARNING")
    df_test = None
    has_test_data = False

# Combine data
if has_test_data:
    common_cols = ['probability', 'predicted', 'dataset']
    if dv and dv in df_train.columns and dv in df_test.columns:
        common_cols.insert(0, dv)
    df_combined = pd.concat([df_train[common_cols], df_test[common_cols]], ignore_index=True)
    df_combined = df_combined.dropna(subset=['probability'])
    debug_log(f"Combined data: {len(df_combined)} rows")
else:
    df_combined = df_train.copy()

# Determine mode
contains_dv = dv is not None and dv in df_combined.columns
contains_file_path = file_path is not None and len(file_path) > 0
debug_log(f"Mode: {'HEADLESS' if contains_dv and contains_file_path else 'INTERACTIVE'}")

# Main processing
debug_log("=" * 80)
debug_log("MAIN PROCESSING")
debug_log("=" * 80)

if contains_dv and contains_file_path:
    debug_log("Running HEADLESS mode...")
    gains_table, metrics_df = run_headless_analysis(
        df=df_combined, dv=dv, prob_col='probability', dataset_col='dataset',
        analyze_dataset=analyze_dataset, model_name=model_name, file_path=file_path,
        save_roc=save_roc, save_capture_rate=save_capture_rate, save_ks=save_ks,
        save_lorenz=save_lorenz, save_decile_lift=save_decile_lift
    )
else:
    debug_log("Interactive mode requested - this COMMENTATED version only supports headless mode")
    debug_log("Use model_analyzer_knime_DEBUG_TOGGLE.py for full interactive support", "WARNING")
    gains_table = pd.DataFrame()
    metrics_df = pd.DataFrame()

# Output
debug_log("=" * 80)
debug_log("PREPARING OUTPUT")
debug_log("=" * 80)

if 'predicted' in df_combined.columns:
    df_combined['predicted'] = pd.to_numeric(df_combined['predicted'], errors='coerce').fillna(0).astype('Int32')
if 'probability' in df_combined.columns:
    df_combined['probability'] = pd.to_numeric(df_combined['probability'], errors='coerce').astype('Float64')
if 'dataset' in df_combined.columns:
    df_combined['dataset'] = df_combined['dataset'].astype(str)

debug_log(f"Output 1: {df_combined.shape}")
knio.output_tables[0] = knio.Table.from_pandas(df_combined)

if isinstance(gains_table, pd.DataFrame) and len(gains_table) > 0:
    debug_log(f"Output 2: {gains_table.shape}")
    knio.output_tables[1] = knio.Table.from_pandas(gains_table)
else:
    knio.output_tables[1] = knio.Table.from_pandas(pd.DataFrame())

if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
    debug_log(f"Output 3: {metrics_df.shape}")
    knio.output_tables[2] = knio.Table.from_pandas(metrics_df)
else:
    knio.output_tables[2] = knio.Table.from_pandas(pd.DataFrame())

debug_log("=" * 80)
debug_log("COMMENTATED DEBUG TOGGLE VERSION COMPLETED")
debug_log(f"Completion time: {datetime.now().isoformat()}")
debug_log("=" * 80)

# Cleanup
sys.stdout.flush()
try: del df_train
except: pass
try: del df_test
except: pass
try: del df_combined
except: pass
try: del df_coef
except: pass
gc.collect()
debug_log("Cleanup completed")

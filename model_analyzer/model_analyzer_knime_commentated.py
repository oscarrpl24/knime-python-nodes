# =============================================================================
# Model Analyzer for KNIME Python Script Node - FULLY COMMENTATED VERSION
# =============================================================================
# This is a comprehensive, line-by-line commented version of the Model Analyzer
# Python script designed to run inside KNIME 5.9 Python Script nodes.
#
# PURPOSE:
# This script analyzes the performance of credit risk logistic regression models
# by computing various metrics (AUC, Gini, K-S statistic) and generating
# diagnostic charts (ROC curves, Lorenz curves, gains tables, etc.)
#
# The script has two operating modes:
# 1. Interactive Mode (Shiny UI) - Launches when no flow variables are provided
#    - User can explore charts and metrics interactively in a web browser
# 2. Headless Mode - Runs when DependentVariable and FilePath flow variables exist
#    - Automatically generates charts and saves them to disk without user interaction
#
# INPUT PORTS (from KNIME):
# Port 1 (Required) - Training Data:
#    - Contains WOE (Weight of Evidence) transformed feature columns (prefixed WOE_*)
#    - Dependent variable column containing actual binary outcomes (0 = good, 1 = bad)
#    - "predicted" column (Int): The predicted class label (0 or 1)
#    - "probabilities" column (Float): The linear predictor / log-odds from regression
#      Note: If values are outside 0-1, they are converted using sigmoid function
#
# Port 2 (Required) - Coefficients Table:
#    - Row ID contains variable names including "(Intercept)"
#    - First numeric column contains the coefficient values from logistic regression
#
# Port 3 (Optional) - Test Data:
#    - Same structure as training data (WOE columns + dependent variable)
#    - Predictions are computed using the coefficients from Port 2
#
# OUTPUT PORTS (to KNIME):
# Port 1 - Combined data with predictions (training + test datasets merged)
# Port 2 - Gains table DataFrame (decile-based performance metrics)
# Port 3 - Model performance metrics DataFrame (AUC, Gini, K-S, accuracy, etc.)
#
# FLOW VARIABLES (for headless/automated mode):
# - DependentVariable (string): Name of the binary target variable column
# - ModelName (string, optional): Prefix for saved chart filenames
# - AnalyzeDataset (string): "Training", "Test", or "Both" - which subset to analyze
# - FilePath (string): Directory path where chart images will be saved
# - ProbabilitiesColumn (string, optional): Column name for log-odds (default: "probabilities")
# - saveROC (int): Set to 1 to save ROC curve image
# - saveCaptureRate (int): Set to 1 to save Capture Rate chart image
# - saveK-S (int): Set to 1 to save K-S chart image
# - saveLorenzCurve (int): Set to 1 to save Lorenz Curve image
# - saveDecileLift (int): Set to 1 to save Decile Lift chart image
#
# Release Date: 2026-01-16
# Version: 1.2
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

# Import typing module for type hints
# These make the code more readable by documenting expected types
# Dict = dictionary, List = list, Tuple = tuple of values
# Optional = can be None, Any = any type, Union = one of several types
from typing import Dict, List, Tuple, Optional, Any, Union

# Import dataclass decorator from dataclasses module
# Dataclasses automatically generate __init__, __repr__, etc. for simple classes
from dataclasses import dataclass

# Suppress all warning messages to keep output clean
# This prevents cluttering the KNIME console with non-critical warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: STABILITY SETTINGS FOR MULTIPLE INSTANCE EXECUTION
# =============================================================================
# When multiple KNIME nodes run Python scripts simultaneously, they can conflict
# with each other over ports and threading resources. These settings prevent that.

# BASE_PORT defines the starting port number for the Shiny web application
# Port 8051 is commonly used for Shiny/Dash applications
BASE_PORT = 8051

# RANDOM_PORT_RANGE defines how far above BASE_PORT we can randomly select
# This means ports 8051 through 9051 are potential candidates
# Using random ports prevents collisions when multiple analyzer nodes run at once
RANDOM_PORT_RANGE = 1000

# Create a unique identifier for this specific running instance
# Combines the process ID (unique per Python process) with a random number
# This is used to create unique temporary directories if needed
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Set environment variables to limit threading in numerical libraries
# When multiple instances run in KNIME, thread contention can cause hangs or crashes

# NUMEXPR_MAX_THREADS controls numexpr library threading
# numexpr is used by pandas for fast numerical expressions
os.environ['NUMEXPR_MAX_THREADS'] = '1'

# OMP_NUM_THREADS controls OpenMP threading (used by many scientific libraries)
# OpenMP is a parallel processing API used by numpy and others
os.environ['OMP_NUM_THREADS'] = '1'

# OPENBLAS_NUM_THREADS controls OpenBLAS threading
# OpenBLAS is a fast linear algebra library that numpy may use
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# MKL_NUM_THREADS controls Intel Math Kernel Library threading
# MKL is Intel's optimized math library that numpy may use on Intel CPUs
os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# SECTION 3: INSTALL/IMPORT DEPENDENCIES
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
        If not provided, assumes the import name matches the package name
    
    Example:
    --------
    install_if_missing('scikit-learn', 'sklearn')
    # This will try: import sklearn
    # If that fails, it runs: pip install scikit-learn
    """
    # If no import name provided, assume it matches the package name
    if import_name is None:
        import_name = package
    
    # Try to import the package
    try:
        # __import__ is the built-in function that implements the import statement
        # It returns the imported module (which we don't need here)
        __import__(import_name)
    except ImportError:
        # ImportError means the package is not installed
        # We need to install it using pip
        import subprocess
        # subprocess.check_call runs a command and waits for it to complete
        # If the command fails, it raises an exception
        subprocess.check_call(['pip', 'install', package])

# Install scikit-learn if not present - provides machine learning algorithms
# Note: package name is 'scikit-learn' but import name is 'sklearn'
install_if_missing('scikit-learn', 'sklearn')

# Install plotly if not present - provides interactive charting capabilities
install_if_missing('plotly')

# Install shiny if not present - provides the interactive web UI framework
install_if_missing('shiny')

# Install shinywidgets if not present - provides plotly integration with shiny
install_if_missing('shinywidgets')

# Install kaleido if not present - enables saving plotly figures as static images
# Kaleido is a headless browser for rendering plotly charts to image files
install_if_missing('kaleido')

# Now import the required functions from scikit-learn
# roc_curve: computes the ROC (Receiver Operating Characteristic) curve points
# auc: calculates the Area Under the Curve
# confusion_matrix: creates the 2x2 matrix of TP, TN, FP, FN
from sklearn.metrics import roc_curve, auc, confusion_matrix

# LogisticRegression is imported but not actually used in this script
# It's kept here for potential future use or compatibility
from sklearn.linear_model import LogisticRegression

# Try to import Shiny and related packages for the interactive UI
# These imports are wrapped in try/except because Shiny might not work
# in all environments (e.g., headless servers without display)
try:
    # Import Shiny components for building the web application
    # App: the main application class that ties UI and server together
    # Inputs: container for all user input values
    # Outputs: container for all rendered outputs
    # Session: represents a single user's session with the app
    # reactive: decorators for creating reactive computations
    # render: decorators for rendering outputs
    # ui: functions for building the user interface
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Import shinywidgets for rendering plotly charts in Shiny
    # render_plotly: decorator for rendering plotly figures
    # output_widget: UI placeholder for plotly chart output
    from shinywidgets import render_plotly, output_widget
    
    # Import plotly.graph_objects for creating detailed custom charts
    # go.Figure, go.Scatter, go.Bar, etc. are used for building charts
    import plotly.graph_objects as go
    
    # Import plotly.express for quick, high-level charting (not heavily used here)
    import plotly.express as px
    
    # If we got here, Shiny is available and working
    SHINY_AVAILABLE = True
    
except ImportError:
    # If import fails, Shiny is not available
    # The script will still work in headless mode but cannot launch interactive UI
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# SECTION 4: DATA CLASSES
# =============================================================================
# Data classes are simple classes that primarily hold data.
# The @dataclass decorator automatically generates __init__, __repr__, etc.

@dataclass
class GainsTable:
    """
    Container class for gains table results.
    
    A gains table (also called a "lift table" or "response table") divides
    predictions into deciles (10 equal groups) ranked by predicted probability,
    then calculates various metrics for each decile.
    
    Attributes:
    -----------
    table : pd.DataFrame
        The actual gains table with columns:
        - decile: The decile number (1-10, where 1 has highest predicted prob)
        - n: Number of observations in the decile
        - events: Number of actual events (1s) in the decile
        - non_events: Number of actual non-events (0s) in the decile
        - event_rate: Proportion of events in this decile (events/n)
        - pct_events: This decile's events as % of total events
        - pct_non_events: This decile's non-events as % of total non-events
        - cum_events: Cumulative count of events through this decile
        - cum_non_events: Cumulative count of non-events through this decile
        - cum_pct_events: Cumulative % of events through this decile
        - cum_pct_non_events: Cumulative % of non-events through this decile
        - ks: K-S statistic at this decile (|cum_pct_events - cum_pct_non_events|)
        - lift: Cumulative lift at this decile
        - min_prob, max_prob, avg_prob: Probability statistics for this decile
    
    total_obs : int
        Total number of observations across all deciles
    
    total_events : int
        Total number of events (1s) in the dataset
    
    total_non_events : int
        Total number of non-events (0s) in the dataset
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
    
    Attributes:
    -----------
    auc : float
        Area Under the ROC Curve. Ranges from 0 to 1.
        - 0.5 = random model (no discrimination)
        - 1.0 = perfect model
        - Values below 0.5 indicate inverted predictions
        Typical good credit models have AUC between 0.70 and 0.85.
    
    gini : float
        Gini coefficient, also called the Gini index or Accuracy Ratio.
        Calculated as: Gini = 2 * AUC - 1
        Ranges from -1 to 1, where:
        - 0 = random model
        - 1 = perfect model
        Gini is popular in credit risk because it's directly comparable
        across different base rates.
    
    ks_statistic : float
        Kolmogorov-Smirnov statistic.
        The maximum separation between the cumulative distribution of
        events and non-events when sorted by predicted probability.
        Ranges from 0 to 1. Higher is better.
        Typical values for credit models: 0.30 to 0.50.
    
    ks_decile : int
        The decile number (1-10) where the maximum K-S occurs.
        Usually occurs around decile 3-5 for well-calibrated models.
    
    accuracy : float
        Overall accuracy = (TP + TN) / (TP + TN + FP + FN)
        Proportion of correctly classified cases.
        Not ideal for imbalanced datasets (common in credit risk).
    
    sensitivity : float
        Also called Recall or True Positive Rate.
        Sensitivity = TP / (TP + FN)
        Proportion of actual positives correctly identified.
        "Of all the actual bads, how many did we catch?"
    
    specificity : float
        Also called True Negative Rate.
        Specificity = TN / (TN + FP)
        Proportion of actual negatives correctly identified.
        "Of all the actual goods, how many did we correctly approve?"
    """
    auc: float           # Area Under ROC Curve (0.5 to 1.0)
    gini: float          # Gini coefficient (0 to 1.0)
    ks_statistic: float  # Maximum K-S separation (0 to 1.0)
    ks_decile: int       # Decile where max K-S occurs (1 to 10)
    accuracy: float      # Overall classification accuracy
    sensitivity: float   # True Positive Rate / Recall
    specificity: float   # True Negative Rate


# =============================================================================
# SECTION 5: LOGISTIC REGRESSION PREDICTION FUNCTIONS
# =============================================================================
# These functions handle the mathematical transformations needed to work
# with logistic regression model outputs.

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid (logistic) function to convert log-odds to probabilities.
    
    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    
    This is the inverse of the logit function and converts log-odds
    (which can range from -infinity to +infinity) into probabilities
    (which must be between 0 and 1).
    
    Mathematical background:
    - In logistic regression, the model predicts log-odds: ln(p / (1-p))
    - The sigmoid function converts this back to probability p
    - When x = 0, sigmoid(0) = 0.5 (50% probability)
    - As x → +∞, sigmoid(x) → 1.0
    - As x → -∞, sigmoid(x) → 0.0
    
    Parameters:
    -----------
    x : array-like
        Log-odds (linear predictor) values from logistic regression.
        These are the raw outputs before conversion to probability.
    
    Returns:
    --------
    np.ndarray
        Probabilities between 0 and 1. NaN values in input remain NaN.
    
    Example:
    --------
    >>> sigmoid(np.array([0, 2, -2]))
    array([0.5, 0.88079708, 0.11920292])
    """
    # Convert input to numpy array with float type
    # This ensures we can perform mathematical operations
    x = np.array(x, dtype=float)
    
    # Create a boolean mask identifying which positions are NaN
    # We need to preserve NaN positions in the output
    nan_mask = np.isnan(x)
    
    # Clip values to prevent numerical overflow in exp()
    # exp(500) is already astronomically large (≈ 10^217)
    # exp(-500) is effectively 0
    # Without clipping, exp(710) would overflow to infinity
    x = np.clip(x, -500, 500)
    
    # Apply the sigmoid formula: 1 / (1 + e^(-x))
    # np.exp(-x) computes e raised to the power of -x for each element
    result = 1 / (1 + np.exp(-x))
    
    # Restore NaN values in positions that were originally NaN
    # This ensures we don't lose track of missing data
    result[nan_mask] = np.nan
    
    # Return the probability array
    return result


def is_log_odds(values: np.ndarray) -> bool:
    """
    Detect whether values are log-odds or probabilities.
    
    Log-odds can take any real value from -infinity to +infinity.
    Probabilities are constrained to the range [0, 1].
    
    This function examines the values to determine which type they are:
    - If ANY values are outside [0, 1], they must be log-odds
    - If all values are within [0, 1], they're assumed to be probabilities
    
    This auto-detection is important because different upstream nodes
    might output either format, and we need to handle both correctly.
    
    Parameters:
    -----------
    values : array-like
        Array of numeric values that are either probabilities or log-odds
    
    Returns:
    --------
    bool
        True if values appear to be log-odds (some outside 0-1 range)
        False if values appear to be probabilities (all within 0-1 range)
    
    Example:
    --------
    >>> is_log_odds(np.array([0.2, 0.5, 0.8]))
    False  # All values in [0, 1], so these are probabilities
    
    >>> is_log_odds(np.array([-1.5, 0.0, 2.3]))
    True   # Values outside [0, 1], so these are log-odds
    """
    # Convert to numpy array for consistent handling
    values = np.array(values)
    
    # Remove NaN values - we can't use them for detection
    # ~np.isnan creates a boolean mask where True = not NaN
    values = values[~np.isnan(values)]
    
    # If no valid values remain, assume not log-odds (default to probabilities)
    if len(values) == 0:
        return False
    
    # Check if any values are outside the [0, 1] range
    # If so, they cannot be probabilities and must be log-odds
    if np.any(values < 0) or np.any(values > 1):
        return True
    
    # If all values are exactly 0 or 1 (binary), probably predicted classes
    # not log-odds (which would typically have some spread)
    if np.all((values == 0) | (values == 1)):
        return False
    
    # If all values are within [0, 1] and not all binary,
    # assume they are already probabilities
    return False


def parse_coefficients_table(coef_df: pd.DataFrame) -> Dict[str, float]:
    """
    Parse a coefficients table from R model output into a Python dictionary.
    
    In KNIME workflows, logistic regression models output their coefficients
    in a table format. This function extracts those coefficients into a
    dictionary that can be used for making predictions.
    
    Expected table format:
    - Row ID (index) contains variable names, e.g., "(Intercept)", "WOE_Age"
    - First numeric column contains the coefficient values
    
    The function handles R-style output where:
    - Row IDs are used as variable names
    - "(Intercept)" is a special variable name for the constant term
    
    Parameters:
    -----------
    coef_df : pd.DataFrame
        DataFrame with Row ID as index containing variable names
        and at least one numeric column with coefficient values
    
    Returns:
    --------
    Dict[str, float]
        Dictionary mapping variable names to their coefficient values
        Example: {"(Intercept)": -1.234, "WOE_Age": 0.456, "WOE_Income": 0.789}
    
    Raises:
    -------
    ValueError
        If no numeric columns are found in the coefficients table
    """
    # Initialize empty dictionary to store coefficients
    coefficients = {}
    
    # Find all columns with numeric data types
    # select_dtypes filters columns by their dtype
    # np.number includes int, float, etc.
    numeric_cols = coef_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Raise error if no numeric columns found
    # We need at least one numeric column for coefficient values
    if not numeric_cols:
        raise ValueError("No numeric columns found in coefficients table")
    
    # Use the first numeric column as the coefficient column
    # This matches R's typical output format
    coef_col = numeric_cols[0]
    
    # Print which column we're using (helpful for debugging)
    print(f"Using coefficient column: '{coef_col}'")
    
    # Iterate through each row of the coefficients table
    # idx = row index (variable name), row = the row data
    for idx, row in coef_df.iterrows():
        # Convert index to string (variable name)
        var_name = str(idx)
        
        # Get the coefficient value from the identified column
        coef_value = row[coef_col]
        
        # Store in dictionary, converting to float
        coefficients[var_name] = float(coef_value)
    
    # Print summary of loaded coefficients
    print(f"Loaded {len(coefficients)} coefficients")
    
    # Print the intercept value if it exists (helpful for verification)
    if '(Intercept)' in coefficients:
        print(f"  Intercept: {coefficients['(Intercept)']:.6f}")
    
    # Return the dictionary of coefficients
    return coefficients


def predict_with_coefficients(
    df: pd.DataFrame,
    coefficients: Dict[str, float],
    return_log_odds: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply logistic regression coefficients to compute predictions.
    
    This function manually applies a logistic regression model by:
    1. Starting with the intercept term
    2. Adding each variable × coefficient contribution
    3. Converting the sum (log-odds) to probabilities via sigmoid
    4. Converting probabilities to class predictions using 0.5 threshold
    
    Mathematical formula:
        log_odds = intercept + sum(coefficient_i × value_i)
        probability = sigmoid(log_odds)
        predicted_class = 1 if probability >= 0.5 else 0
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with feature columns whose names match coefficient keys.
        Typically WOE-transformed columns like "WOE_Age", "WOE_Income", etc.
    
    coefficients : Dict[str, float]
        Dictionary mapping variable names to coefficients.
        Should include "(Intercept)" for the constant term.
    
    return_log_odds : bool, default=False
        If True, also return the raw log-odds values.
        Useful for debugging or for use in other calculations.
    
    Returns:
    --------
    If return_log_odds=False:
        Tuple of (probabilities, predicted_class) numpy arrays
    
    If return_log_odds=True:
        Tuple of (probabilities, predicted_class, log_odds) numpy arrays
    
    Note:
    -----
    Missing values (NaN) in feature columns are filled with 0.
    This is appropriate for WOE variables where 0 represents the
    population average risk (neutral contribution).
    """
    # Get the number of observations (rows) in the data
    n = len(df)
    
    # Get the intercept from coefficients, default to 0 if not present
    intercept = coefficients.get('(Intercept)', 0.0)
    
    # Handle case where intercept itself is NaN
    if pd.isna(intercept):
        intercept = 0.0
    
    # Initialize log_odds array with intercept value for all observations
    # np.full creates an array of size n, all filled with intercept value
    log_odds = np.full(n, intercept, dtype=float)
    
    # Track statistics about coefficient matching
    matched_vars = 0          # Count of coefficients found in data
    missing_vars = []          # List of coefficients not found in data
    nan_filled_vars = []       # Track which variables had NaN values filled
    
    # Iterate through each coefficient
    for var_name, coef in coefficients.items():
        # Skip the intercept - already handled above
        if var_name == '(Intercept)':
            continue
        
        # Skip coefficients that are NaN (invalid)
        if pd.isna(coef):
            continue
        
        # Check if this variable exists in the input data
        if var_name in df.columns:
            # Get the values for this variable as float array
            values = df[var_name].values.astype(float)
            
            # Count NaN values before filling
            nan_count = np.isnan(values).sum()
            if nan_count > 0:
                # Track this variable for reporting
                nan_filled_vars.append((var_name, nan_count))
            
            # Fill NaN with 0 - neutral value for WOE variables
            # WOE of 0 means average risk, so it doesn't push prediction either way
            values = np.nan_to_num(values, nan=0.0)
            
            # Add this variable's contribution to log-odds
            # contribution = coefficient × value for each observation
            log_odds += coef * values
            
            # Increment matched count
            matched_vars += 1
        else:
            # Variable in coefficients not found in data
            missing_vars.append(var_name)
    
    # Report on NaN filling if it occurred
    if nan_filled_vars:
        print(f"Note: Filled NaN with 0 in {len(nan_filled_vars)} variables:")
        # Show first 3 variables
        for var, count in nan_filled_vars[:3]:
            print(f"  - {var}: {count} NaN values")
        # If more than 3, show count of remaining
        if len(nan_filled_vars) > 3:
            print(f"  ... and {len(nan_filled_vars) - 3} more")
    
    # Report on missing variables if any
    if missing_vars:
        print(f"Warning: {len(missing_vars)} coefficient variables not found in data:")
        # Show first 5 missing variables
        for var in missing_vars[:5]:
            print(f"  - {var}")
        # If more than 5, show count of remaining
        if len(missing_vars) > 5:
            print(f"  ... and {len(missing_vars) - 5} more")
    
    # Print count of successfully matched variables
    print(f"Matched {matched_vars} variables from coefficients")
    
    # Convert log-odds to probabilities using sigmoid function
    probabilities = sigmoid(log_odds)
    
    # Convert probabilities to class predictions (0 or 1)
    # Using 0.5 as the default threshold
    # If probability >= 0.5, predict class 1 (bad/default)
    # If probability < 0.5, predict class 0 (good/non-default)
    predicted_class = (probabilities >= 0.5).astype(int)
    
    # Return appropriate outputs based on return_log_odds parameter
    if return_log_odds:
        return probabilities, predicted_class, log_odds
    
    return probabilities, predicted_class


def ensure_probabilities(values: np.ndarray, col_name: str = "values") -> np.ndarray:
    """
    Ensure that values are probabilities (0-1). Convert from log-odds if needed.
    
    This function is a safety wrapper that:
    1. Checks if values are already probabilities (0-1 range)
    2. If not, converts them from log-odds using sigmoid function
    
    This is important because different upstream nodes might output
    either format, and all downstream analysis expects probabilities.
    
    Parameters:
    -----------
    values : array-like
        Either probabilities (already in 0-1 range) or log-odds (any real number)
    
    col_name : str, default="values"
        Name of the column (for logging purposes only)
    
    Returns:
    --------
    np.ndarray
        Probabilities guaranteed to be between 0 and 1
    
    Example:
    --------
    >>> ensure_probabilities([0.2, 0.5, 0.8], "prob")
    # Returns same values (already probabilities)
    
    >>> ensure_probabilities([-1.0, 0.0, 1.0], "logit")
    # Returns sigmoid-transformed values: [0.269, 0.5, 0.731]
    """
    # Convert to numpy array with float type
    values = np.array(values, dtype=float)
    
    # Count and report NaN values
    nan_count = np.isnan(values).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in '{col_name}'")
    
    # Check if values are log-odds (outside 0-1 range)
    if is_log_odds(values):
        # Values are log-odds, need to convert to probabilities
        print(f"Converting '{col_name}' from log-odds to probabilities (values outside 0-1 detected)")
        return sigmoid(values)
    else:
        # Values are already probabilities
        print(f"'{col_name}' appears to already be probabilities (all values in 0-1 range)")
        return values


# =============================================================================
# SECTION 6: CORE METRIC CALCULATION FUNCTIONS
# =============================================================================
# These functions calculate the key performance metrics used to evaluate
# credit risk models.

def calculate_gains_table(actual: np.ndarray, predicted: np.ndarray, n_deciles: int = 10) -> GainsTable:
    """
    Calculate a gains table (equivalent to R's blorr::blr_gains_table).
    
    A gains table divides the population into equal groups (typically deciles)
    ranked by predicted probability, then calculates performance metrics
    for each group. This helps visualize how well the model separates
    high-risk from low-risk cases.
    
    How it works:
    1. Sort all observations by predicted probability (highest first)
    2. Divide into n equal groups (deciles)
    3. For each group, calculate various metrics
    
    The gains table is fundamental to credit risk model validation.
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1). In credit risk:
        - 1 typically represents "bad" (default/delinquency)
        - 0 typically represents "good" (no default)
    
    predicted : array-like
        Predicted probabilities (0 to 1). Higher values should
        correspond to higher probability of the event (actual = 1).
    
    n_deciles : int, default=10
        Number of groups to divide the data into.
        10 (deciles) is standard, but 20 (vigintiles) is also common.
    
    Returns:
    --------
    GainsTable
        Dataclass containing:
        - table: DataFrame with metrics for each decile
        - total_obs: Total number of observations
        - total_events: Total count of events (1s)
        - total_non_events: Total count of non-events (0s)
    
    Example gains table interpretation:
    - Decile 1 (highest predicted probability) should have highest event rate
    - Decile 10 (lowest predicted probability) should have lowest event rate
    - If this pattern doesn't hold, the model is not discriminating well
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Create DataFrame with actual and predicted values
    # This makes it easy to sort and group
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    })
    
    # Sort by predicted probability in DESCENDING order
    # Highest predicted probabilities come first (highest risk)
    # reset_index(drop=True) renumbers the index 0, 1, 2, ...
    df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
    
    # Calculate totals for later use
    total_obs = len(df)                    # Total number of observations
    total_events = df['actual'].sum()       # Total events (sum of 1s)
    total_non_events = total_obs - total_events  # Total non-events
    
    # Create decile labels (1, 2, 3, ..., 10)
    # pd.qcut divides data into quantiles
    # We're dividing row indices into n_deciles equal groups
    # labels=False gives us 0, 1, 2, ... so we add 1 to get 1, 2, 3, ...
    df['decile'] = pd.qcut(range(len(df)), q=n_deciles, labels=False) + 1
    
    # Initialize list to store metrics for each decile
    gains_data = []
    
    # Initialize cumulative counters
    cumulative_events = 0      # Running total of events
    cumulative_non_events = 0  # Running total of non-events
    cumulative_obs = 0         # Running total of observations
    
    # Calculate metrics for each decile
    for decile in range(1, n_deciles + 1):
        # Filter data to just this decile
        decile_data = df[df['decile'] == decile]
        
        # Count observations and events in this decile
        n_obs = len(decile_data)                    # Observations in this decile
        n_events = decile_data['actual'].sum()       # Events in this decile
        n_non_events = n_obs - n_events              # Non-events in this decile
        
        # Update cumulative totals
        cumulative_obs += n_obs
        cumulative_events += n_events
        cumulative_non_events += n_non_events
        
        # Calculate event rate within this decile
        # This is the probability of an event for cases in this decile
        event_rate = n_events / n_obs if n_obs > 0 else 0
        
        # Calculate this decile's share of total events and non-events
        pct_events = n_events / total_events if total_events > 0 else 0
        pct_non_events = n_non_events / total_non_events if total_non_events > 0 else 0
        
        # Calculate cumulative percentages
        # These are used for the K-S chart and Lorenz curve
        cum_pct_events = cumulative_events / total_events if total_events > 0 else 0
        cum_pct_non_events = cumulative_non_events / total_non_events if total_non_events > 0 else 0
        
        # Calculate K-S statistic for this decile
        # K-S = absolute difference between cumulative distributions
        ks = abs(cum_pct_events - cum_pct_non_events)
        
        # Calculate cumulative lift
        # Lift = how many times better than random
        # If decile 3 captures 45% of events, and random would capture 30%,
        # then lift = 45% / 30% = 1.5
        decile_pct = decile / n_deciles  # Expected % if random (10%, 20%, 30%, ...)
        lift = cum_pct_events / decile_pct if decile_pct > 0 else 0
        
        # Calculate probability statistics for this decile
        min_prob = decile_data['predicted'].min()   # Lowest probability in decile
        max_prob = decile_data['predicted'].max()   # Highest probability in decile
        avg_prob = decile_data['predicted'].mean()  # Average probability in decile
        
        # Append all metrics for this decile to our list
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
    
    # Convert list of dictionaries to DataFrame
    gains_df = pd.DataFrame(gains_data)
    
    # Return GainsTable dataclass with all results
    return GainsTable(
        table=gains_df,
        total_obs=total_obs,
        total_events=int(total_events),
        total_non_events=int(total_non_events)
    )


def calculate_roc_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate ROC (Receiver Operating Characteristic) curve metrics.
    
    The ROC curve plots True Positive Rate vs False Positive Rate at
    various classification thresholds. It's a standard tool for
    evaluating binary classification models.
    
    Key concepts:
    - FPR (False Positive Rate) = FP / (FP + TN) = 1 - Specificity
      "Of all actual negatives, what fraction did we incorrectly predict positive?"
    - TPR (True Positive Rate) = TP / (TP + FN) = Sensitivity = Recall
      "Of all actual positives, what fraction did we correctly predict?"
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    Returns:
    --------
    Tuple containing:
    - fpr : np.ndarray - False Positive Rates at various thresholds
    - tpr : np.ndarray - True Positive Rates at various thresholds
    - auc_score : float - Area Under the ROC Curve (rounded to 5 decimal places)
    - gini_index : float - Gini coefficient = 2 * AUC - 1 (rounded to 5 decimal places)
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Use scikit-learn's roc_curve function
    # This returns FPR, TPR arrays and the thresholds used
    # thresholds are the probability cutoffs used to generate each point
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    
    # Calculate Area Under the Curve using trapezoidal integration
    auc_score = auc(fpr, tpr)
    
    # Calculate Gini index (also called Gini coefficient or Accuracy Ratio)
    # Gini = 2 * AUC - 1
    # This scales AUC from [0.5, 1.0] to [0.0, 1.0]
    # A random model has AUC = 0.5 and Gini = 0
    # A perfect model has AUC = 1.0 and Gini = 1.0
    gini_index = 2 * auc_score - 1
    
    # Return all metrics, rounding floats to 5 decimal places
    return fpr, tpr, round(auc_score, 5), round(gini_index, 5)


def calculate_ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, int]:
    """
    Calculate the Kolmogorov-Smirnov (K-S) statistic.
    
    The K-S statistic measures the maximum separation between the
    cumulative distribution functions of events and non-events.
    It represents the point where the model best separates the two classes.
    
    In credit risk:
    - K-S of 0.40 means the model achieves 40% separation
    - Typical good models have K-S between 0.30 and 0.50
    - K-S > 0.50 is excellent
    - K-S < 0.20 indicates poor discrimination
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    Returns:
    --------
    Tuple containing:
    - ks_statistic : float - Maximum K-S value (rounded to 4 decimal places)
    - ks_decile : int - Decile number (1-10) where maximum K-S occurs
    """
    # Calculate gains table to get K-S at each decile
    gains = calculate_gains_table(actual, predicted)
    
    # Extract K-S values for all deciles
    ks_values = gains.table['ks'].values
    
    # Find the maximum K-S value
    ks_statistic = ks_values.max()
    
    # Find which decile has the maximum K-S
    # np.argmax returns the index (0-based), so add 1 to get decile number
    ks_decile = int(np.argmax(ks_values) + 1)
    
    # Return K-S statistic and the decile where it occurs
    return round(ks_statistic, 4), ks_decile


def calculate_model_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.5) -> ModelMetrics:
    """
    Calculate comprehensive model performance metrics.
    
    This function computes all the key metrics needed to evaluate
    a binary classification model's performance.
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    threshold : float, default=0.5
        Classification threshold for converting probabilities to classes.
        Cases with probability >= threshold are classified as 1.
    
    Returns:
    --------
    ModelMetrics
        Dataclass containing: auc, gini, ks_statistic, ks_decile,
        accuracy, sensitivity, specificity
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate ROC curve metrics (AUC and Gini)
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
    
    # Calculate K-S statistic and decile
    ks_stat, ks_decile = calculate_ks_statistic(actual, predicted)
    
    # Convert probabilities to predicted classes using threshold
    # 1 if probability >= threshold, else 0
    predicted_class = (predicted >= threshold).astype(int)
    
    # Calculate confusion matrix
    # Returns array: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(actual, predicted_class)
    
    # Extract individual counts from confusion matrix
    # .ravel() flattens the 2x2 matrix to 1D array: [TN, FP, FN, TP]
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate accuracy: proportion of correct predictions
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Calculate sensitivity (True Positive Rate, Recall)
    # Sensitivity = TP / (TP + FN)
    # "Of all actual positives, what fraction did we catch?"
    # Handle division by zero if no actual positives
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate specificity (True Negative Rate)
    # Specificity = TN / (TN + FP)
    # "Of all actual negatives, what fraction did we correctly identify?"
    # Handle division by zero if no actual negatives
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Return all metrics in a ModelMetrics dataclass
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
# SECTION 7: CHART CREATION FUNCTIONS (using Plotly)
# =============================================================================
# These functions create interactive charts using the Plotly library.
# Each function creates a specific type of model diagnostic chart.

def create_roc_curve(actual: np.ndarray, predicted: np.ndarray, 
                     model_name: str = "Model", color: str = "#E74C3C") -> go.Figure:
    """
    Create an ROC (Receiver Operating Characteristic) curve with AUC and Gini.
    
    The ROC curve shows the trade-off between sensitivity and specificity
    at various classification thresholds. The diagonal line represents
    a random classifier (AUC = 0.5).
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    model_name : str, default="Model"
        Name to display in the chart legend
    
    color : str, default="#E74C3C"
        Hex color code for the ROC curve line (default is red)
    
    Returns:
    --------
    go.Figure
        Plotly figure object containing the ROC curve
    """
    # Calculate ROC curve data points and metrics
    fpr, tpr, auc_score, gini = calculate_roc_metrics(actual, predicted)
    
    # Create a new Plotly figure
    fig = go.Figure()
    
    # Add the ROC curve as a line trace
    # x = False Positive Rate (1 - Specificity)
    # y = True Positive Rate (Sensitivity)
    fig.add_trace(go.Scatter(
        x=fpr,                    # FPR values on x-axis
        y=tpr,                    # TPR values on y-axis
        mode='lines',             # Draw as a continuous line
        name=f'{model_name} (AUC = {auc_score:.4f}, Gini = {gini:.4f})',  # Legend label
        line=dict(color=color, width=2)  # Line styling
    ))
    
    # Add diagonal reference line (represents random classifier)
    # A random model has AUC = 0.5 and follows the diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],                 # From (0,0) to (1,1)
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', dash='dash', width=1)  # Dashed gray line
    ))
    
    # Update the layout with titles, axis labels, and styling
    fig.update_layout(
        title=dict(text='ROC Curve', font=dict(size=18)),  # Chart title
        xaxis_title='1 - Specificity (False Positive Rate)',  # X-axis label
        yaxis_title='Sensitivity (True Positive Rate)',       # Y-axis label
        legend=dict(
            yanchor="bottom",     # Anchor legend at bottom
            y=0.01,               # Position near bottom of chart
            xanchor="right",      # Anchor legend at right
            x=0.99,               # Position near right of chart
            bgcolor="rgba(255,255,255,0.8)"  # Semi-transparent white background
        ),
        template='plotly_white',  # Clean white background theme
        width=600,                # Chart width in pixels
        height=500                # Chart height in pixels
    )
    
    # Return the completed figure
    return fig


def create_roc_curve_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                          test_actual: np.ndarray, test_predicted: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    """
    Create ROC curve comparing training and test datasets on same chart.
    
    This is useful for detecting overfitting:
    - If training AUC >> test AUC, model may be overfitting
    - If both AUCs are similar, model generalizes well
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    model_name : str
        Name to display in chart title
    
    Returns:
    --------
    go.Figure
        Plotly figure with both ROC curves overlaid
    """
    # Calculate metrics for training data
    train_fpr, train_tpr, train_auc, train_gini = calculate_roc_metrics(train_actual, train_predicted)
    
    # Calculate metrics for test data
    test_fpr, test_tpr, test_auc, test_gini = calculate_roc_metrics(test_actual, test_predicted)
    
    # Create new figure
    fig = go.Figure()
    
    # Add training ROC curve (blue)
    fig.add_trace(go.Scatter(
        x=train_fpr, y=train_tpr,
        mode='lines',
        name=f'Training (AUC = {train_auc:.4f}, Gini = {train_gini:.4f})',
        line=dict(color='#3498DB', width=2)  # Blue color for training
    ))
    
    # Add test ROC curve (red)
    fig.add_trace(go.Scatter(
        x=test_fpr, y=test_tpr,
        mode='lines',
        name=f'Test (AUC = {test_auc:.4f}, Gini = {test_gini:.4f})',
        line=dict(color='#E74C3C', width=2)  # Red color for test
    ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    # Update layout with labels and styling
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
    """
    Create K-S (Kolmogorov-Smirnov) chart (equivalent to blorr::blr_ks_chart).
    
    The K-S chart shows:
    - Cumulative % of events (sensitivity) by decile
    - Cumulative % of non-events (1 - specificity) by decile
    - The maximum separation between these two curves is the K-S statistic
    
    This chart helps visualize where the model provides the best separation
    between good and bad cases.
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    Returns:
    --------
    go.Figure
        Plotly figure showing K-S chart with annotated max K-S point
    """
    # Calculate gains table to get cumulative percentages by decile
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    # Find maximum K-S value and which decile it occurs at
    ks_max = df['ks'].max()
    ks_decile = df.loc[df['ks'].idxmax(), 'decile']
    
    # Create new figure
    fig = go.Figure()
    
    # Add cumulative events line (sensitivity)
    # This shows what % of all events we've captured by each decile
    fig.add_trace(go.Scatter(
        x=df['decile'],                    # X-axis: decile number
        y=df['cum_pct_events'],            # Y-axis: cumulative % of events
        mode='lines+markers',               # Line with point markers
        name='Cumulative % Events (Sensitivity)',
        line=dict(color='#3498DB', width=2),  # Blue line
        marker=dict(size=8)                   # Marker size
    ))
    
    # Add cumulative non-events line (1 - specificity)
    # This shows what % of non-events we've included by each decile
    fig.add_trace(go.Scatter(
        x=df['decile'],
        y=df['cum_pct_non_events'],
        mode='lines+markers',
        name='Cumulative % Non-Events (1 - Specificity)',
        line=dict(color='#E74C3C', width=2),  # Red line
        marker=dict(size=8)
    ))
    
    # Add annotation marking the maximum K-S point
    # Position it between the two curves at the max K-S decile
    ks_row = df[df['decile'] == ks_decile].iloc[0]
    fig.add_annotation(
        x=ks_decile,  # X position at the max K-S decile
        y=(ks_row['cum_pct_events'] + ks_row['cum_pct_non_events']) / 2,  # Y position between curves
        text=f'Max K-S = {ks_max:.4f}',  # Annotation text
        showarrow=True,                   # Show arrow pointing to position
        arrowhead=2,                      # Arrow style
        font=dict(size=12, color='#2C3E50')  # Text font
    )
    
    # Add vertical dashed line at max K-S decile
    fig.add_vline(
        x=ks_decile,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f'Decile {ks_decile}'
    )
    
    # Update layout with titles and styling
    fig.update_layout(
        title=dict(text=f'K-S Chart (Max K-S = {ks_max:.4f} at Decile {ks_decile})', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Percentage',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Show all decile numbers
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
    """
    Create Lorenz curve (equivalent to blorr::blr_lorenz_curve).
    
    The Lorenz curve shows cumulative % of events vs cumulative % of population
    when sorted by predicted probability. It visualizes model concentration power.
    
    Interpretation:
    - The diagonal line represents a random model (no discrimination)
    - The further the curve bows away from the diagonal, the better the model
    - The area between the curve and diagonal is related to the Gini coefficient
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    Returns:
    --------
    go.Figure
        Plotly figure showing Lorenz curve with shaded area
    """
    # Sort observations by predicted probability (highest first)
    # np.argsort with negative returns indices for descending sort
    sorted_idx = np.argsort(-predicted)
    actual_sorted = actual[sorted_idx]
    
    # Get total counts
    n = len(actual_sorted)
    total_events = actual_sorted.sum()
    
    # Calculate cumulative percentages
    # cum_pct_pop: cumulative % of population (1/n, 2/n, 3/n, ...)
    cum_pct_pop = np.arange(1, n + 1) / n
    # cum_pct_events: cumulative % of events captured
    cum_pct_events = np.cumsum(actual_sorted) / total_events
    
    # If there are too many points, subsample for performance
    # 1000+ points slows down the chart without improving visual quality
    if n > 1000:
        # Take 500 evenly spaced indices
        idx = np.linspace(0, n - 1, 500, dtype=int)
        cum_pct_pop = cum_pct_pop[idx]
        cum_pct_events = cum_pct_events[idx]
    
    # Create new figure
    fig = go.Figure()
    
    # Add Lorenz curve with shaded area below
    fig.add_trace(go.Scatter(
        x=cum_pct_pop,                # X: cumulative % of population
        y=cum_pct_events,             # Y: cumulative % of events
        mode='lines',
        name='Lorenz Curve',
        fill='tozeroy',               # Fill area from curve to y=0
        line=dict(color='#3498DB', width=2),
        fillcolor='rgba(52, 152, 219, 0.3)'  # Light blue fill
    ))
    
    # Add diagonal reference line (line of equality / random model)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Line of Equality',
        line=dict(color='#E74C3C', dash='dash', width=2)  # Red dashed line
    ))
    
    # Update layout
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
    """
    Create Decile Lift chart (equivalent to blorr::blr_decile_lift_chart).
    
    Lift measures how much better the model performs compared to random.
    - Lift of 2.0 means the model is 2x better than random at that point
    - Lift of 1.0 means the model is no better than random
    
    The chart shows cumulative lift at each decile, demonstrating how much
    value the model provides when targeting customers in order of predicted risk.
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    bar_color : str, default='#40E0D0'
        Hex color code for the bars (default is turquoise)
    
    Returns:
    --------
    go.Figure
        Plotly figure showing lift as bar chart
    """
    # Calculate gains table to get lift values
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    # Create new figure
    fig = go.Figure()
    
    # Add bar chart for lift values
    fig.add_trace(go.Bar(
        x=df['decile'],              # X: decile numbers
        y=df['lift'],                # Y: cumulative lift values
        name='Lift',
        marker_color=bar_color,      # Bar color
        text=df['lift'].round(2),    # Show lift value on each bar
        textposition='outside'        # Position text above bars
    ))
    
    # Add horizontal reference line at lift = 1 (random baseline)
    fig.add_hline(
        y=1,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text='Baseline Lift = 1'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text='Decile Lift Chart', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Lift',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Show all decile numbers
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_ks_chart_both(train_actual: np.ndarray, train_predicted: np.ndarray,
                         test_actual: np.ndarray, test_predicted: np.ndarray) -> go.Figure:
    """
    Create K-S chart comparing training and test datasets.
    
    Shows the K-S statistic (difference between cumulative distributions)
    for both datasets on the same chart.
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    Returns:
    --------
    go.Figure
        Plotly figure with both K-S curves
    """
    # Calculate gains tables for both datasets
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    train_df = train_gains.table
    test_df = test_gains.table
    
    # Get max K-S values for legend
    train_ks = train_df['ks'].max()
    test_ks = test_df['ks'].max()
    
    # Create figure
    fig = go.Figure()
    
    # Training K-S curve (blue)
    fig.add_trace(go.Scatter(
        x=train_df['decile'], y=train_df['ks'],
        mode='lines+markers',
        name=f'Training (K-S = {train_ks:.4f})',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=8)
    ))
    
    # Test K-S curve (red)
    fig.add_trace(go.Scatter(
        x=test_df['decile'], y=test_df['ks'],
        mode='lines+markers',
        name=f'Test (K-S = {test_ks:.4f})',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout
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
    """
    Create Lorenz curve comparing training and test datasets.
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    Returns:
    --------
    go.Figure
        Plotly figure with both Lorenz curves
    """
    
    def get_lorenz_data(actual, predicted):
        """Helper function to calculate Lorenz curve data for one dataset."""
        # Sort by predicted probability descending
        sorted_idx = np.argsort(-predicted)
        actual_sorted = actual[sorted_idx]
        n = len(actual_sorted)
        total_events = actual_sorted.sum()
        
        # Handle edge case of no events
        if total_events == 0:
            return np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        
        # Calculate cumulative percentages
        cum_pct_pop = np.arange(1, n + 1) / n
        cum_pct_events = np.cumsum(actual_sorted) / total_events
        
        # Subsample if too many points
        if n > 500:
            idx = np.linspace(0, n - 1, 500, dtype=int)
            cum_pct_pop = cum_pct_pop[idx]
            cum_pct_events = cum_pct_events[idx]
        
        return cum_pct_pop, cum_pct_events
    
    # Get Lorenz data for both datasets
    train_pop, train_events = get_lorenz_data(train_actual, train_predicted)
    test_pop, test_events = get_lorenz_data(test_actual, test_predicted)
    
    # Create figure
    fig = go.Figure()
    
    # Training Lorenz curve (blue)
    fig.add_trace(go.Scatter(
        x=train_pop, y=train_events,
        mode='lines',
        name='Training',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Test Lorenz curve (red)
    fig.add_trace(go.Scatter(
        x=test_pop, y=test_events,
        mode='lines',
        name='Test',
        line=dict(color='#E74C3C', width=2)
    ))
    
    # Diagonal reference line (random model)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    # Update layout
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
    """
    Create Decile Lift chart comparing training and test datasets.
    
    Shows side-by-side bars for each decile comparing lift between datasets.
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    Returns:
    --------
    go.Figure
        Plotly figure with grouped bar chart
    """
    # Calculate gains tables for both datasets
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    # Create figure
    fig = go.Figure()
    
    # Training bars (slightly left of center for each decile)
    fig.add_trace(go.Bar(
        x=train_gains.table['decile'] - 0.2,  # Offset left
        y=train_gains.table['lift'],
        name='Training',
        marker_color='#3498DB',               # Blue
        text=train_gains.table['lift'].round(2),
        textposition='outside',
        width=0.35                            # Bar width
    ))
    
    # Test bars (slightly right of center for each decile)
    fig.add_trace(go.Bar(
        x=test_gains.table['decile'] + 0.2,   # Offset right
        y=test_gains.table['lift'],
        name='Test',
        marker_color='#E74C3C',               # Red
        text=test_gains.table['lift'].round(2),
        textposition='outside',
        width=0.35
    ))
    
    # Reference line at lift = 1
    fig.add_hline(y=1, line=dict(color='gray', dash='dash', width=1))
    
    # Update layout
    fig.update_layout(
        title=dict(text='Decile Lift - Training vs Test', font=dict(size=18)),
        xaxis_title='Decile',
        yaxis_title='Cumulative Lift',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        barmode='group',                      # Group bars together
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_event_rate_chart(actual: np.ndarray, predicted: np.ndarray,
                            bar_color: str = '#00CED1') -> go.Figure:
    """
    Create Event Rate by Decile chart.
    
    Shows the event rate (proportion of events) within each decile.
    A good model should show decreasing event rate as decile increases
    (since decile 1 has highest predicted probability).
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    bar_color : str, default='#00CED1'
        Hex color code for bars (default is dark turquoise)
    
    Returns:
    --------
    go.Figure
        Plotly bar chart showing event rate by decile
    """
    # Calculate gains table
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    # Create figure
    fig = go.Figure()
    
    # Event rate as bar chart (convert to percentage for display)
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['event_rate'] * 100,            # Convert to percentage
        name='Event Rate (%)',
        marker_color=bar_color,
        text=(df['event_rate'] * 100).round(1),  # Show rate on bars
        textposition='outside'
    ))
    
    # Calculate and add overall event rate reference line
    overall_rate = df['events'].sum() / df['n'].sum() * 100
    fig.add_hline(
        y=overall_rate,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text=f'Overall Rate: {overall_rate:.1f}%'
    )
    
    # Update layout
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
    """
    Create Event Rate by Decile chart comparing training and test.
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    Returns:
    --------
    go.Figure
        Plotly grouped bar chart
    """
    # Calculate gains tables
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    # Create figure
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
    
    # Update layout
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
    """
    Create Capture Rate by Decile chart.
    
    Shows what percentage of total events are captured in each decile.
    A good model concentrates events in the top deciles (highest predicted probability).
    
    Example interpretation:
    - If decile 1 has 25% capture rate, the top 10% of predictions contains 25% of all events
    - A random model would have 10% capture rate in each decile
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    bar_color : str, default='#27AE60'
        Hex color code for bars (default is green)
    
    Returns:
    --------
    go.Figure
        Plotly bar chart showing capture rate by decile
    """
    # Calculate gains table
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    # Create figure
    fig = go.Figure()
    
    # pct_events = what % of all events are in this decile
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['pct_events'] * 100,            # Convert to percentage
        name='Capture Rate (%)',
        marker_color=bar_color,
        text=(df['pct_events'] * 100).round(1),
        textposition='outside'
    ))
    
    # Reference line at 10% (expected if random, since 10% of population)
    fig.add_hline(
        y=10,
        line=dict(color='#E74C3C', dash='dash', width=2),
        annotation_text='Random: 10%'
    )
    
    # Update layout
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
    """
    Create Capture Rate by Decile chart comparing training and test.
    
    Parameters:
    -----------
    train_actual, train_predicted : array-like
        Actual and predicted values for training set
    
    test_actual, test_predicted : array-like
        Actual and predicted values for test set
    
    Returns:
    --------
    go.Figure
        Plotly grouped bar chart
    """
    # Calculate gains tables
    train_gains = calculate_gains_table(train_actual, train_predicted)
    test_gains = calculate_gains_table(test_actual, test_predicted)
    
    # Create figure
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
    
    # Update layout
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
    """
    Create Cumulative Capture Rate chart.
    
    Shows the cumulative percentage of events captured by each decile.
    This is the same data as the Lorenz curve but displayed as a bar chart.
    
    Example interpretation:
    - If cumulative capture at decile 3 is 60%, the top 30% of predictions
      (deciles 1-3) contain 60% of all events
    
    Parameters:
    -----------
    actual : array-like
        Binary actual values (0 or 1)
    
    predicted : array-like
        Predicted probabilities (0 to 1)
    
    bar_color : str, default='#9B59B6'
        Hex color code for bars (default is purple)
    
    Returns:
    --------
    go.Figure
        Plotly bar chart with line overlay for random model
    """
    # Calculate gains table
    gains = calculate_gains_table(actual, predicted)
    df = gains.table
    
    # Create figure
    fig = go.Figure()
    
    # Cumulative capture rate bars
    fig.add_trace(go.Bar(
        x=df['decile'],
        y=df['cum_pct_events'] * 100,        # Convert to percentage
        name='Cumulative Capture Rate (%)',
        marker_color=bar_color,
        text=(df['cum_pct_events'] * 100).round(1),
        textposition='outside'
    ))
    
    # Reference line for random model (cumulative: 10%, 20%, 30%, ...)
    fig.add_trace(go.Scatter(
        x=df['decile'],
        y=df['decile'] * 10,                 # Random: decile × 10%
        mode='lines+markers',
        name='Random Model',
        line=dict(color='#E74C3C', dash='dash', width=2)
    ))
    
    # Update layout
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
    """
    Save a Plotly figure as a JPEG image file.
    
    Uses the kaleido library to render the figure to an image.
    Falls back to PNG format if JPEG fails.
    
    Parameters:
    -----------
    fig : go.Figure
        Plotly figure to save
    
    filepath : str
        Full path including filename where to save the image
    
    Note:
    -----
    Requires kaleido package to be installed for image export.
    """
    try:
        # Attempt to save as JPEG
        # width/height: image dimensions in pixels
        # scale: multiplier for resolution (2 = 2x resolution)
        fig.write_image(filepath, format='jpeg', width=800, height=600, scale=2)
        print(f"Saved chart to: {filepath}")
    except Exception as e:
        # JPEG export failed
        print(f"Error saving chart to {filepath}: {e}")
        
        # Try PNG as fallback (usually more compatible)
        try:
            png_path = filepath.replace('.jpeg', '.png').replace('.jpg', '.png')
            fig.write_image(png_path, format='png', width=800, height=600, scale=2)
            print(f"Saved chart as PNG to: {png_path}")
        except Exception as e2:
            # Both formats failed
            print(f"Could not save chart: {e2}")


# =============================================================================
# SECTION 8: SHINY UI APPLICATION
# =============================================================================
# This section contains the interactive Shiny web application that allows
# users to explore model performance metrics and charts interactively.

def create_model_analyzer_app(
    df: pd.DataFrame,
    dv: Optional[str] = None,
    prob_col: str = "probabilities",
    pred_col: str = "predicted",
    dataset_col: str = "dataset"
):
    """
    Create the Model Analyzer Shiny application.
    
    This function builds a complete Shiny web application for interactive
    model analysis. The app displays:
    - Configuration options (select DV, dataset, probability column)
    - Model performance metrics (AUC, Gini, K-S, accuracy, sensitivity, specificity)
    - ROC curve
    - K-S chart
    - Lorenz curve
    - Additional charts (event rate, capture rate, lift)
    - Gains table
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with predictions and probabilities columns
    
    dv : str, optional
        Dependent variable (actual values) column name.
        If None, user must select it in the UI.
    
    prob_col : str, default="probabilities"
        Column name for predicted probabilities
    
    pred_col : str, default="predicted"
        Column name for predicted class (0/1)
    
    dataset_col : str, default="dataset"
        Column name for dataset indicator ("Training" or "Test")
    
    Returns:
    --------
    App
        Shiny App object ready to be run
    """
    
    # Dictionary to store results when user closes the app
    # This allows us to return data after the UI session ends
    app_results = {
        'gains_table': None,    # Will hold the gains table DataFrame
        'metrics': None,        # Will hold the metrics dictionary
        'completed': False      # Flag indicating if user clicked close button
    }
    
    # Get list of column names for dropdowns
    columns = list(df.columns)
    
    # Get list of numeric columns only (for probability column dropdown)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if dataset column exists and determine what datasets are available
    has_dataset_col = dataset_col in df.columns
    
    if has_dataset_col:
        # Get unique values in the dataset column
        unique_datasets = df[dataset_col].dropna().unique().tolist()
        
        # Check if "training" and "test" are present (case-insensitive)
        has_training = any(d.lower() == 'training' for d in unique_datasets if isinstance(d, str))
        has_test = any(d.lower() == 'test' for d in unique_datasets if isinstance(d, str))
        
        # Build dropdown choices based on what's available
        if has_training and has_test:
            dataset_choices = ["Training", "Test", "Both"]
        elif has_training:
            dataset_choices = ["Training"]
        elif has_test:
            dataset_choices = ["Test"]
        else:
            dataset_choices = ["All Data"]
    else:
        # No dataset column - treat all data as one dataset
        dataset_choices = ["All Data"]
        has_test = False
    
    # Set default probability column (use specified or fall back to last numeric column)
    if prob_col not in df.columns:
        prob_col = numeric_cols[-1] if numeric_cols else columns[0]
    
    # ==========================================================================
    # Build the UI (User Interface)
    # ==========================================================================
    # The UI is built using Shiny's ui module with a fluid page layout
    
    app_ui = ui.page_fluid(
        # Add custom CSS styles in the page head
        ui.tags.head(
            ui.tags.style("""
                /* Import Google font for a professional look */
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                
                /* Body styling - sets the main page appearance */
                body { 
                    font-family: 'Source Sans Pro', sans-serif; 
                    background: #f5f7fa;        /* Light gray background */
                    min-height: 100vh;          /* Full viewport height */
                    color: #2c3e50;             /* Dark blue-gray text */
                }
                
                /* Card component styling - used for grouping content */
                .card { 
                    background: #ffffff;        /* White background */
                    border: 1px solid #e1e8ed;  /* Light gray border */
                    border-radius: 8px;         /* Rounded corners */
                    padding: 20px;              /* Internal spacing */
                    margin: 10px 0;             /* Vertical spacing between cards */
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);  /* Subtle shadow */
                }
                
                /* Card header styling */
                .card-header {
                    color: #2c3e50;
                    font-weight: 700;           /* Bold */
                    font-size: 1.1rem;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #3498db;  /* Blue underline */
                    padding-bottom: 8px;
                }
                
                /* Main heading styling */
                h3 { 
                    color: #2c3e50; 
                    text-align: center; 
                    font-weight: 700;
                    margin-bottom: 24px;
                }
                
                /* Primary button styling */
                .btn-primary { 
                    background: #3498db;        /* Blue */
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 10px 24px;
                    border-radius: 6px;
                }
                .btn-primary:hover {
                    background: #2980b9;        /* Darker blue on hover */
                }
                
                /* Success/Complete button styling */
                .btn-success { 
                    background: #27ae60;        /* Green */
                    border: none;
                    color: white;
                    font-weight: 700;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-success:hover {
                    background: #219a52;        /* Darker green on hover */
                }
                
                /* Form input styling */
                .form-control, .form-select {
                    background: #ffffff;
                    border: 1px solid #ced4da;
                    color: #2c3e50;
                    border-radius: 6px;
                }
                .form-control:focus, .form-select:focus {
                    background: #ffffff;
                    border-color: #3498db;      /* Blue border when focused */
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
                    color: #2c3e50;
                }
                
                /* Form label styling */
                .form-label {
                    color: #2c3e50;
                    font-weight: 600;
                }
                
                /* Metric display styling - for large numbers */
                .metric-value {
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #2c3e50;
                    text-align: center;
                }
                .metric-label {
                    color: #7f8c8d;             /* Gray for labels */
                    text-align: center;
                    font-size: 0.85rem;
                    margin-top: 4px;
                }
                
                /* Grid layout for metrics display */
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);  /* 4 equal columns */
                    gap: 12px;
                    margin-top: 12px;
                }
                
                /* Individual metric box styling */
                .metric-box {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 14px;
                    text-align: center;
                    border: 1px solid #e1e8ed;
                }
                
                /* Training metrics box - blue tint */
                .metric-box-train {
                    background: #ebf5fb;
                    border: 1px solid #3498db;
                }
                
                /* Test metrics box - red tint */
                .metric-box-test {
                    background: #fdedec;
                    border: 1px solid #e74c3c;
                }
                
                /* Section title styling */
                .section-title {
                    color: #2c3e50;
                    font-weight: 600;
                    margin-bottom: 8px;
                    padding-bottom: 4px;
                    border-bottom: 2px solid;
                }
                .section-title-train {
                    border-color: #3498db;      /* Blue for training */
                }
                .section-title-test {
                    border-color: #e74c3c;      /* Red for test */
                }
            """)
        ),
        
        # Page title
        ui.h3("Model Analyzer"),
        
        # Configuration Panel - allows user to select analysis options
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Configuration"),
            ui.row(
                # Column 1: Analyzer name input
                ui.column(3,
                    ui.input_text("analyzer_name", "Analyzer Name", value="Logistic Regression"),
                ),
                # Column 2: Dataset dropdown
                ui.column(3,
                    ui.input_select("dataset", "Dataset", choices=dataset_choices),
                ),
                # Column 3: Dependent variable dropdown
                ui.column(3,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=columns,
                                   selected=dv if dv and dv in columns else columns[0]),
                ),
                # Column 4: Probabilities column dropdown
                ui.column(3,
                    ui.input_select("prob_col", "Probabilities Column", 
                                   choices=numeric_cols,
                                   selected=prob_col if prob_col in numeric_cols else (numeric_cols[-1] if numeric_cols else None)),
                ),
            ),
        ),
        
        # Metrics Display Panel - shows calculated performance metrics
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Performance Metrics"),
            ui.output_ui("metrics_display"),  # Dynamic content rendered by server
        ),
        
        # Charts Row 1: ROC and K-S charts side by side
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 550px;"},
                    ui.div({"class": "card-header"}, "ROC Curve"),
                    output_widget("roc_chart", height="480px")  # Plotly widget output
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
        
        # Charts Row 2: Lorenz and selectable other charts
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
                    # Dropdown to select which additional chart to view
                    ui.input_select("other_chart", "Select Chart", 
                                   choices=["Event Rate by Decile", "Capture Rate by Decile",
                                           "Decile Lift Chart", "Cumulative Capture Rate"]),
                    output_widget("other_chart_display", height="420px")
                )
            ),
        ),
        
        # Gains Table Panel - shows the decile-based metrics table
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Gains Table"),
            ui.output_data_frame("gains_table"),  # DataFrame display
        ),
        
        # Close Button Panel - completes the analysis and closes the app
        ui.div(
            {"class": "card", "style": "text-align: center; margin-top: 20px;"},
            ui.input_action_button("close_btn", "✓ Complete Analysis", class_="btn btn-success btn-lg"),
        ),
    )
    
    # ==========================================================================
    # Define the Server Logic
    # ==========================================================================
    # The server function contains all the reactive logic that responds to
    # user inputs and generates outputs
    
    def server(input: Inputs, output: Outputs, session: Session):
        
        @reactive.Calc
        def get_data():
            """
            Reactive calculation that returns data based on selected dataset.
            
            This is recalculated whenever input.dv(), input.prob_col(), or
            input.dataset() changes. It filters the data appropriately.
            
            Returns:
            --------
            Tuple of arrays depending on dataset selection:
            - For single dataset: (actual, predicted, None, None)
            - For "Both": (train_actual, train_predicted, test_actual, test_predicted)
            """
            # Get current values from input controls
            dv_col = input.dv()                    # Selected dependent variable column
            selected_prob_col = input.prob_col()   # Selected probability column
            dataset_choice = input.dataset()        # Selected dataset option
            
            # Filter data based on dataset column if it exists
            if has_dataset_col and dataset_col in df.columns:
                if dataset_choice == "Training":
                    # Filter to training data only
                    subset = df[df[dataset_col].str.lower() == 'training']
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                    
                elif dataset_choice == "Test":
                    # Filter to test data only
                    subset = df[df[dataset_col].str.lower() == 'test']
                    actual = subset[dv_col].values
                    predicted = subset[selected_prob_col].values
                    return actual, predicted, None, None
                    
                elif dataset_choice == "Both":
                    # Get both training and test data separately
                    train_subset = df[df[dataset_col].str.lower() == 'training']
                    test_subset = df[df[dataset_col].str.lower() == 'test']
                    
                    train_actual = train_subset[dv_col].values
                    train_predicted = train_subset[selected_prob_col].values
                    test_actual = test_subset[dv_col].values
                    test_predicted = test_subset[selected_prob_col].values
                    
                    return train_actual, train_predicted, test_actual, test_predicted
                else:
                    # "All Data" - use everything
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
            """
            Reactive calculation that computes model metrics.
            
            Depends on get_data(), so it updates whenever the data changes.
            
            Returns:
            --------
            Tuple of (train_metrics, test_metrics) or (metrics, None)
            """
            data = get_data()
            if input.dataset() == "Both":
                # Calculate metrics for both datasets
                train_metrics = calculate_model_metrics(data[0], data[1])
                test_metrics = calculate_model_metrics(data[2], data[3])
                return train_metrics, test_metrics
            else:
                # Calculate metrics for single dataset
                metrics = calculate_model_metrics(data[0], data[1])
                return metrics, None
        
        @output
        @render.ui
        def metrics_display():
            """
            Render the metrics display as HTML.
            
            Creates a grid of metric boxes showing AUC, Gini, K-S, etc.
            Shows both training and test metrics if "Both" is selected.
            """
            metrics_data = get_metrics()
            
            if input.dataset() == "Both":
                # Display metrics for both training and test
                train_m, test_m = metrics_data
                return ui.div(
                    # Training section
                    ui.div({"class": "section-title section-title-train"}, "Training"),
                    ui.div(
                        {"class": "metrics-grid"},
                        # AUC metric box
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.auc:.4f}"),
                               ui.div({"class": "metric-label"}, "AUC")),
                        # Gini metric box
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.gini:.4f}"),
                               ui.div({"class": "metric-label"}, "Gini")),
                        # K-S metric box
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.ks_statistic:.4f}"),
                               ui.div({"class": "metric-label"}, f"K-S (Decile {train_m.ks_decile})")),
                        # Accuracy metric box
                        ui.div({"class": "metric-box metric-box-train"},
                               ui.div({"class": "metric-value"}, f"{train_m.accuracy:.1%}"),
                               ui.div({"class": "metric-label"}, "Accuracy")),
                    ),
                    # Test section
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
                # Single dataset display - show all 6 metrics
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
            """Render the ROC curve chart."""
            data = get_data()
            if input.dataset() == "Both":
                # Create comparison chart with both datasets
                return create_roc_curve_both(data[0], data[1], data[2], data[3], input.analyzer_name())
            else:
                # Create single dataset chart
                return create_roc_curve(data[0], data[1], input.analyzer_name())
        
        @output
        @render_plotly
        def ks_chart():
            """Render the K-S chart."""
            data = get_data()
            if input.dataset() == "Both":
                return create_ks_chart_both(data[0], data[1], data[2], data[3])
            else:
                return create_ks_chart(data[0], data[1])
        
        @output
        @render_plotly
        def lorenz_chart():
            """Render the Lorenz curve chart."""
            data = get_data()
            if input.dataset() == "Both":
                return create_lorenz_curve_both(data[0], data[1], data[2], data[3])
            else:
                return create_lorenz_curve(data[0], data[1])
        
        @output
        @render_plotly
        def other_chart_display():
            """Render the selected additional chart based on dropdown choice."""
            data = get_data()
            chart_type = input.other_chart()  # Get selected chart type
            
            if input.dataset() == "Both":
                # Use comparison versions of charts
                if chart_type == "Event Rate by Decile":
                    return create_event_rate_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Capture Rate by Decile":
                    return create_capture_rate_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Decile Lift Chart":
                    return create_decile_lift_chart_both(data[0], data[1], data[2], data[3])
                elif chart_type == "Cumulative Capture Rate":
                    # Use test data for cumulative capture
                    return create_cumulative_capture_chart(data[2], data[3])
            else:
                # Single dataset charts
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
            """Render the gains table as an interactive DataFrame."""
            data = get_data()
            if input.dataset() == "Both":
                # Combine gains tables from both datasets
                train_gains = calculate_gains_table(data[0], data[1])
                test_gains = calculate_gains_table(data[2], data[3])
                
                # Add dataset labels
                train_df = train_gains.table.copy()
                test_df = test_gains.table.copy()
                train_df['dataset'] = 'Training'
                test_df['dataset'] = 'Test'
                
                # Concatenate into single table
                combined = pd.concat([train_df, test_df], ignore_index=True)
                
                # Reorder columns to put dataset first
                cols = ['dataset'] + [c for c in combined.columns if c != 'dataset']
                return combined[cols]
            else:
                # Single dataset gains table
                gains = calculate_gains_table(data[0], data[1])
                return gains.table
        
        @reactive.Effect
        @reactive.event(input.close_btn)
        async def handle_close():
            """
            Handle the close button click.
            
            This saves the results and closes the Shiny session.
            The results are stored in app_results which can be accessed
            after the app closes.
            """
            # Get the current data
            data = get_data()
            
            # Calculate final results based on selected dataset
            if input.dataset() == "Both":
                # Use test data for final results
                gains = calculate_gains_table(data[2], data[3])
                metrics = calculate_model_metrics(data[2], data[3])
            else:
                gains = calculate_gains_table(data[0], data[1])
                metrics = calculate_model_metrics(data[0], data[1])
            
            # Store results in the app_results dictionary
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
            
            # Close the session (this stops the app)
            await session.close()
    
    # Create the Shiny App object by combining UI and server
    app = App(app_ui, server)
    
    # Attach the results dictionary to the app so it can be accessed externally
    app.results = app_results
    
    return app


def find_free_port(start_port: int = 8051, max_attempts: int = 50) -> int:
    """
    Find an available network port to run the Shiny server on.
    
    Tries random ports in a range to avoid conflicts when multiple
    instances of the Model Analyzer are running simultaneously.
    
    Parameters:
    -----------
    start_port : int, default=8051
        Base port number to start searching from
    
    max_attempts : int, default=50
        Maximum number of ports to try before falling back to OS-assigned port
    
    Returns:
    --------
    int
        An available port number
    """
    import socket
    
    # Try random ports in the range
    for offset in range(max_attempts):
        # Pick a random port in the allowed range
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        
        try:
            # Try to bind to the port
            # If successful, the port is free
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))  # Bind to localhost only
                return port
        except OSError:
            # Port is in use, try another
            continue
    
    # If all attempts failed, let the OS assign a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))  # 0 tells OS to assign any available port
        return s.getsockname()[1]  # Get the assigned port number


def run_model_analyzer(
    df: pd.DataFrame,
    dv: Optional[str] = None,
    prob_col: str = "probabilities",
    pred_col: str = "predicted",
    dataset_col: str = "dataset",
    port: int = None
):
    """
    Run the Model Analyzer Shiny application and wait for user to complete.
    
    This function:
    1. Creates the Shiny app
    2. Runs it on an available port
    3. Opens a browser window
    4. Waits for the user to complete analysis (click close button)
    5. Returns the results
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with predictions and probabilities
    
    dv : str, optional
        Dependent variable column name
    
    prob_col : str, default="probabilities"
        Probabilities column name
    
    pred_col : str, default="predicted"
        Predicted class column name
    
    dataset_col : str, default="dataset"
        Dataset indicator column name
    
    port : int, optional
        Specific port to use. If None, finds a free port automatically.
    
    Returns:
    --------
    dict
        Results dictionary containing:
        - 'gains_table': DataFrame or None
        - 'metrics': dict or None
        - 'completed': bool
    """
    import threading
    import time
    
    # Find a free port if none specified
    if port is None:
        port = find_free_port(BASE_PORT)
    
    # Log the port being used
    print(f"Starting Shiny app on port {port}")
    sys.stdout.flush()  # Ensure message is printed immediately
    
    # Create the Shiny application
    app = create_model_analyzer_app(df, dv, prob_col, pred_col, dataset_col)
    
    # Variable to capture any server exceptions
    server_exception = [None]  # Use list to allow modification in nested function
    
    def run_server():
        """Thread target function to run the Shiny server."""
        try:
            # Run the app - this blocks until the app closes
            app.run(port=port, launch_browser=True)
        except Exception as e:
            # Capture any exception for reporting
            server_exception[0] = e
            print(f"Server stopped: {e}")
            sys.stdout.flush()
    
    # Start the server in a separate thread so we can monitor completion
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for the app to complete (user clicks close button)
    timeout_counter = 0
    max_timeout = 7200  # 2 hours maximum wait time
    
    while not app.results.get('completed', False):
        # Sleep briefly to avoid busy-waiting
        time.sleep(0.5)
        timeout_counter += 0.5
        
        # Check if the server crashed
        if server_exception[0] is not None:
            print(f"Server encountered error: {server_exception[0]}")
            break
        
        # Check for timeout
        if timeout_counter >= max_timeout:
            print("Session timed out after 2 hours")
            break
    
    # Brief pause for cleanup
    time.sleep(0.5)
    
    print("Analysis complete - returning results")
    sys.stdout.flush()
    
    # Force garbage collection to free memory
    gc.collect()
    
    return app.results


# =============================================================================
# SECTION 9: HEADLESS MODE PROCESSING
# =============================================================================
# Functions for running the analysis without user interaction,
# automatically saving charts to files.

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
    
    This function performs all the same analysis as the interactive mode
    but without any UI. It's used when flow variables are provided to
    specify the analysis parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
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
        Name prefix for saved chart files
    
    file_path : str
        Directory path to save charts
    
    save_roc : bool
        If True, save ROC curve image
    
    save_capture_rate : bool
        If True, save Capture Rate chart image
    
    save_ks : bool
        If True, save K-S chart image
    
    save_lorenz : bool
        If True, save Lorenz Curve image
    
    save_decile_lift : bool
        If True, save Decile Lift chart image
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (gains_table, metrics_df)
    """
    # Ensure file path ends with directory separator
    if not file_path.endswith(os.sep):
        file_path += os.sep
    
    # Create the output directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    
    # Initialize lists to collect results
    all_gains = []    # Will hold gains table DataFrames
    all_metrics = []  # Will hold metrics dictionaries
    
    # Check if dataset column exists in the data
    has_dataset_col = dataset_col in df.columns
    
    # Split data by dataset if column exists
    if has_dataset_col:
        # Filter to training data (case-insensitive matching)
        df_train = df[df[dataset_col].str.lower() == 'training'].copy()
        # Filter to test data
        df_test = df[df[dataset_col].str.lower() == 'test'].copy()
    else:
        # No dataset column - treat all data as training
        df_train = df.copy()
        df_test = pd.DataFrame()  # Empty DataFrame
    
    # Process Training data if requested
    if analyze_dataset == "Training" or analyze_dataset == "Both":
        if len(df_train) > 0:
            # Extract arrays for analysis
            actual = df_train[dv].values
            predicted = df_train[prob_col].values
        
            # Calculate gains table
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Training'  # Label the source
            all_gains.append(gains_df)
            
            # Calculate metrics
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
            
            # Save charts for Training only mode
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
    
    # Process Test data if requested
    if analyze_dataset == "Test" or analyze_dataset == "Both":
        if len(df_test) == 0:
            # Warning: requested test analysis but no test data found
            print("Warning: No 'Test' rows found in dataset column but 'Test' or 'Both' was selected")
        else:
            # Extract arrays for analysis
            actual = df_test[dv].values
            predicted = df_test[prob_col].values
            
            # Calculate gains table
            gains = calculate_gains_table(actual, predicted)
            gains_df = gains.table.copy()
            gains_df['dataset'] = 'Test'
            all_gains.append(gains_df)
            
            # Calculate metrics
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
            
            # Save charts for Test only mode
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
    
    # Handle "Both" mode - create comparison charts
    if analyze_dataset == "Both" and len(df_test) > 0 and len(df_train) > 0:
        # Get arrays for both datasets
        train_actual = df_train[dv].values
        train_predicted = df_train[prob_col].values
        test_actual = df_test[dv].values
        test_predicted = df_test[prob_col].values
        
        # Save comparison charts
        if save_roc:
            fig = create_roc_curve_both(train_actual, train_predicted, 
                                        test_actual, test_predicted, model_name)
            save_chart(fig, f"{file_path}{model_name}_Both_ROC.jpeg")
        
        if save_capture_rate:
            # Use test data for the capture rate chart
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
    
    # Combine all gains tables into one DataFrame
    combined_gains = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()
    
    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    
    return combined_gains, metrics_df


# =============================================================================
# SECTION 10: READ INPUT DATA FROM KNIME
# =============================================================================
# This section reads the input tables provided by KNIME.
# The knio.input_tables list contains all input port data.

# -----------------------------------------------------------------------------
# Input Port 1: Training Data (Required)
# -----------------------------------------------------------------------------
# This is the main dataset containing:
# - All WOE (Weight of Evidence) feature columns (usually prefixed with WOE_)
# - The dependent variable column (binary 0/1 actual outcomes)
# - The "predicted" column containing predicted class labels (0 or 1)
# - The "probabilities" column containing linear predictor / log-odds from logistic regression

# Read the first input table and convert to pandas DataFrame
# knio.input_tables[0] is the first input port (0-indexed)
# .to_pandas() converts from KNIME's internal format to pandas DataFrame
df_train = knio.input_tables[0].to_pandas()

# Print information about the loaded data for debugging
print(f"Input 1 (Training data): {len(df_train)} rows, {len(df_train.columns)} columns")

# -----------------------------------------------------------------------------
# Input Port 2: Coefficients Table (Required)
# -----------------------------------------------------------------------------
# This table contains the logistic regression model coefficients.
# Expected format:
# - Row ID contains variable names (e.g., "(Intercept)", "WOE_Age", "WOE_Income")
# - First numeric column contains the coefficient values

# Try to read coefficients table - use try/except in case it's not provided
try:
    # Read the second input table
    df_coef = knio.input_tables[1].to_pandas()
    print(f"Input 2 (Coefficients): {len(df_coef)} rows")
    
    # Parse the coefficients table into a dictionary
    coefficients = parse_coefficients_table(df_coef)
    has_coefficients = True
    
except Exception as e:
    # Coefficients table not available or couldn't be parsed
    print(f"Warning: Could not read coefficients table: {e}")
    coefficients = {}
    has_coefficients = False

# -----------------------------------------------------------------------------
# Input Port 3: Test Data (Optional)
# -----------------------------------------------------------------------------
# This is an optional dataset for model validation.
# If provided, predictions will be computed using the coefficients from Input 2.
# Should have the same WOE feature columns as training data.

try:
    # Attempt to read the third input table
    df_test = knio.input_tables[2].to_pandas()
    
    # Check if the table is not empty
    if len(df_test) > 0:
        print(f"Input 3 (Test data): {len(df_test)} rows, {len(df_test.columns)} columns")
        has_test_data = True
    else:
        # Empty table provided
        df_test = None
        has_test_data = False
        
except:
    # Third input port not connected or error reading
    df_test = None
    has_test_data = False
    print("No test data provided (Input 3)")

# -----------------------------------------------------------------------------
# Check for Expected Columns in Training Data
# -----------------------------------------------------------------------------
# Verify that the expected columns are present

if 'probabilities' in df_train.columns:
    print("Found 'probabilities' column in training data")
elif 'probability' in df_train.columns:
    # Alternative column name
    print("Found 'probability' column (will use as probabilities)")
else:
    print("Warning: No 'probabilities' column found in training data")

if 'predicted' in df_train.columns:
    print("Found 'predicted' column in training data")


# =============================================================================
# SECTION 11: CHECK FOR FLOW VARIABLES (HEADLESS MODE DETECTION)
# =============================================================================
# Flow variables control the node's behavior. If certain flow variables
# are provided, the node runs in headless mode without UI.

# Initialize variables with default values
dv = None                                              # Dependent variable column name
model_name = "Model"                                   # Model name for chart file prefixes
analyze_dataset = "Both" if has_test_data else "Training"  # Which dataset to analyze

file_path = None                                       # Directory path for saving charts

# Column name defaults
prob_col = "probabilities"                             # Column with predicted probabilities
pred_col = "predicted"                                 # Column with predicted classes

# Chart saving flags (all default to False)
save_roc = False           # Save ROC curve
save_capture_rate = False  # Save Capture Rate chart
save_ks = False            # Save K-S chart
save_lorenz = False        # Save Lorenz curve
save_decile_lift = False   # Save Decile Lift chart

# -----------------------------------------------------------------------------
# Read Flow Variables with Error Handling
# -----------------------------------------------------------------------------
# Each flow variable is read with try/except to handle missing variables gracefully

# Read DependentVariable flow variable
try:
    # Get the flow variable value with None as default
    dv = knio.flow_variables.get("DependentVariable", None)
    
    # Check for "missing" placeholder or empty string
    if dv == "missing" or dv == "":
        dv = None
except:
    # Flow variable doesn't exist or can't be read
    pass

# Read ModelName flow variable
try:
    model_name = knio.flow_variables.get("ModelName", "Model")
    if model_name == "missing" or model_name == "":
        model_name = "Model"
except:
    pass

# Read AnalyzeDataset flow variable (which dataset to analyze)
try:
    analyze_dataset = knio.flow_variables.get("AnalyzeDataset", analyze_dataset)
except:
    pass

# Also check for alternative flow variable name "Dataset" for compatibility
try:
    analyze_dataset = knio.flow_variables.get("Dataset", analyze_dataset)
except:
    pass

# Read FilePath flow variable (directory for saving charts)
try:
    file_path = knio.flow_variables.get("FilePath", None)
    if file_path == "missing" or file_path == "":
        file_path = None
except:
    pass

# Read ProbabilitiesColumn flow variable
try:
    prob_col = knio.flow_variables.get("ProbabilitiesColumn", "probabilities")
    if prob_col == "missing" or prob_col == "":
        prob_col = "probabilities"
except:
    pass

# Read PredictedColumn flow variable
try:
    pred_col = knio.flow_variables.get("PredictedColumn", "predicted")
    if pred_col == "missing" or pred_col == "":
        pred_col = "predicted"
except:
    pass

# Read chart saving flags (integer flow variables where 1 = True)
try:
    # saveROC: 1 to save, 0 or missing to skip
    save_roc = knio.flow_variables.get("saveROC", 0) == 1
except:
    pass

try:
    save_capture_rate = knio.flow_variables.get("saveCaptureRate", 0) == 1
except:
    pass

try:
    # Note: "saveK-S" contains a hyphen in the name
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

# -----------------------------------------------------------------------------
# Auto-detect Probabilities Column if Not Found
# -----------------------------------------------------------------------------
# If the specified probability column doesn't exist, try common alternatives

if prob_col not in df_train.columns:
    # List of alternative column names to try
    for alt_name in ['probability', 'prob', 'probs', 'score', 'pred_prob', 'log_odds']:
        if alt_name in df_train.columns:
            prob_col = alt_name
            print(f"Using '{prob_col}' as probabilities column")
            break
    else:
        # No known alternatives found - use last numeric column as fallback
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            prob_col = numeric_cols[-1]
            print(f"Warning: Using '{prob_col}' as probabilities column (last numeric column)")


# =============================================================================
# SECTION 12: PROCESS TRAINING DATA
# =============================================================================
# Prepare the training data for analysis by converting log-odds to probabilities

print("\n--- Processing Training Data ---")

# Add dataset column to identify this as training data
df_train['dataset'] = 'Training'

# Convert log-odds to probabilities if needed
# The input might contain log-odds (linear predictor from R) which need
# to be converted to probabilities using the sigmoid function
if prob_col in df_train.columns:
    # Get raw values from the probability column
    raw_values = df_train[prob_col].values
    
    # Ensure values are probabilities (converts from log-odds if necessary)
    # The result is stored in a new 'probability' column
    df_train['probability'] = ensure_probabilities(raw_values, prob_col)
else:
    # Required column not found - raise an error
    raise ValueError(f"Probabilities column '{prob_col}' not found in training data")


# =============================================================================
# SECTION 13: PROCESS TEST DATA
# =============================================================================
# If test data is provided and we have coefficients, compute predictions for test data

if has_test_data and has_coefficients:
    print("\n--- Processing Test Data (computing predictions from coefficients) ---")
    
    # Compute predictions for test data using the coefficients table
    # This applies the logistic regression formula manually:
    # log_odds = intercept + sum(coefficient_i * value_i)
    # probability = sigmoid(log_odds)
    test_probs, test_preds, test_log_odds = predict_with_coefficients(
        df_test, coefficients, return_log_odds=True
    )
    
    # Check for NaN values in predictions
    nan_probs = np.isnan(test_probs).sum()
    if nan_probs > 0:
        # Fill NaN with 0.5 (neutral probability)
        print(f"Warning: {nan_probs} NaN values in predicted probabilities - filling with 0.5")
        test_probs = np.nan_to_num(test_probs, nan=0.5)
        test_preds = (test_probs >= 0.5).astype(int)
        test_log_odds = np.nan_to_num(test_log_odds, nan=0.0)
    
    # Add computed columns to test DataFrame
    df_test['probability'] = test_probs    # Predicted probabilities (0 to 1)
    df_test['predicted'] = test_preds       # Predicted class (0 or 1)
    df_test['log_odds'] = test_log_odds     # Raw log-odds for reference
    df_test['dataset'] = 'Test'             # Dataset identifier
    
    # Print summary statistics
    print(f"Test predictions computed: {len(df_test)} rows")
    print(f"  Probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    print(f"  Predicted class distribution: 0={np.sum(test_preds==0)}, 1={np.sum(test_preds==1)}")
    
    # Check for NaN in the dependent variable
    if dv and dv in df_test.columns:
        test_dv_nan = df_test[dv].isna().sum()
        if test_dv_nan > 0:
            print(f"Warning: {test_dv_nan} NaN values in test DV column '{dv}'")
            
elif has_test_data and not has_coefficients:
    # Test data provided but no coefficients to compute predictions
    print("\nWarning: Test data provided but no coefficients table - cannot compute predictions")
    df_test = None
    has_test_data = False


# =============================================================================
# SECTION 14: COMBINE DATA FOR ANALYSIS
# =============================================================================
# Create a unified DataFrame with both training and test data if available

if has_test_data:
    # Define columns to include in the combined DataFrame
    common_cols = ['probability', 'predicted', 'dataset']
    
    # Include DV column if it exists in both datasets
    if dv and dv in df_train.columns and dv in df_test.columns:
        common_cols.insert(0, dv)
    
    # Concatenate training and test data
    df_combined = pd.concat([
        df_train[common_cols], 
        df_test[common_cols]
    ], ignore_index=True)
    
    # Report any NaN values in the combined data
    for col in common_cols:
        nan_count = df_combined[col].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in combined '{col}' column")
    
    # Drop rows with NaN in critical columns
    # These are essential for the analysis to work correctly
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
    # No test data - use training data only
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


# =============================================================================
# SECTION 15: DETERMINE OPERATING MODE
# =============================================================================
# Decide whether to run in interactive (Shiny) or headless mode

# Check if dependent variable is specified and exists in the data
contains_dv = dv is not None and dv in df_combined.columns

# Check if file path is specified (needed for saving charts in headless mode)
contains_file_path = file_path is not None and len(file_path) > 0

# Print the configuration
print(f"\nDV: {dv}, Analyze Dataset: {analyze_dataset}")
print(f"Mode: {'Headless' if contains_dv and contains_file_path else 'Interactive'}")


# =============================================================================
# SECTION 16: MAIN PROCESSING LOGIC
# =============================================================================
# Execute the analysis in either headless or interactive mode

# Use the 'probability' column which has been converted from log-odds if needed
analysis_prob_col = 'probability'

if contains_dv and contains_file_path:
    # =========================================================================
    # HEADLESS MODE - Run without UI, save charts to files
    # =========================================================================
    print(f"\nRunning in headless mode...")
    print(f"Saving charts to: {file_path}")
    
    # Call the headless analysis function
    # This computes all metrics and saves charts as specified by flow variables
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
    # INTERACTIVE MODE - Launch Shiny UI for user interaction
    # =========================================================================
    
    # Check if Shiny is available
    if not SHINY_AVAILABLE:
        raise RuntimeError("Shiny is not available. Please install shiny and shinywidgets packages.")
    
    print("\nRunning in interactive mode - launching Shiny UI...")
    
    # Run the Shiny application and wait for user to complete
    results = run_model_analyzer(
        df=df_combined,
        dv=dv,
        prob_col=analysis_prob_col,
        pred_col='predicted',
        dataset_col='dataset'
    )
    
    # Check if the user completed the analysis or cancelled
    if results['completed']:
        # User clicked the complete button - extract results
        gains_table = results['gains_table']
        metrics_dict = results['metrics']
        
        # Convert metrics dictionary to DataFrame
        metrics_df = pd.DataFrame([metrics_dict]) if metrics_dict else pd.DataFrame()
        
        print("Interactive session completed successfully")
    else:
        # User cancelled or session timed out
        print("Interactive session cancelled - returning empty results")
        gains_table = pd.DataFrame()
        metrics_df = pd.DataFrame()


# =============================================================================
# SECTION 17: OUTPUT TABLES TO KNIME
# =============================================================================
# Write the results to KNIME output ports.
# There are 3 output ports:
# 1. Combined data with predictions
# 2. Gains table
# 3. Model performance metrics

print("\nPreparing output tables...")

# -----------------------------------------------------------------------------
# Fix Column Types for Arrow Conversion
# -----------------------------------------------------------------------------
# KNIME uses Apache Arrow for data transfer, which requires specific data types.
# We need to ensure columns have KNIME-compatible types.

# Fix 'predicted' column - use nullable Int32
if 'predicted' in df_combined.columns:
    # pd.to_numeric handles mixed types; 'coerce' converts invalid values to NaN
    # fillna(0) replaces NaN with 0
    # astype('Int32') converts to nullable integer (capital I is important!)
    df_combined['predicted'] = pd.to_numeric(df_combined['predicted'], errors='coerce').fillna(0).astype('Int32')

# Fix 'probability' column - use nullable Float64
if 'probability' in df_combined.columns:
    df_combined['probability'] = pd.to_numeric(df_combined['probability'], errors='coerce').astype('Float64')

# Fix 'dataset' column - ensure it's string type
if 'dataset' in df_combined.columns:
    df_combined['dataset'] = df_combined['dataset'].astype(str)

# -----------------------------------------------------------------------------
# Output Port 1: Combined Data with Predictions
# -----------------------------------------------------------------------------
# This includes the original data plus computed columns:
# - probability: predicted probabilities (0 to 1)
# - predicted: predicted class (0 or 1)
# - dataset: identifier ("Training" or "Test")

knio.output_tables[0] = knio.Table.from_pandas(df_combined)

# -----------------------------------------------------------------------------
# Output Port 2: Gains Table
# -----------------------------------------------------------------------------
# The gains table contains decile-based metrics for evaluating model performance

if isinstance(gains_table, pd.DataFrame) and len(gains_table) > 0:
    # Ensure numeric columns are properly typed
    for col in gains_table.columns:
        if col not in ['dataset']:  # Skip string columns
            gains_table[col] = pd.to_numeric(gains_table[col], errors='coerce')
    
    knio.output_tables[1] = knio.Table.from_pandas(gains_table)
else:
    # No gains table - output empty DataFrame
    knio.output_tables[1] = knio.Table.from_pandas(pd.DataFrame())

# -----------------------------------------------------------------------------
# Output Port 3: Model Performance Metrics
# -----------------------------------------------------------------------------
# Contains AUC, Gini, K-S statistic, accuracy, sensitivity, specificity

if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
    knio.output_tables[2] = knio.Table.from_pandas(metrics_df)
else:
    # No metrics - output empty DataFrame
    knio.output_tables[2] = knio.Table.from_pandas(pd.DataFrame())

# Print completion message
print("\n" + "="*60)
print("Model Analyzer completed successfully")
print("="*60)


# =============================================================================
# SECTION 18: CLEANUP FOR STABILITY
# =============================================================================
# Clean up resources to prevent memory issues when running multiple nodes

# Flush output buffer to ensure all messages are printed
sys.stdout.flush()

# Delete large objects to free memory
# Using try/except because variables might not exist in some code paths
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

# Force garbage collection to immediately free the deleted objects' memory
# This is especially important when running multiple nodes sequentially
gc.collect()

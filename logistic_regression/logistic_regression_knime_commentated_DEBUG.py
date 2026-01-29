# =============================================================================
# Logistic Regression for KNIME Python Script Node - DEBUG VERSION
# =============================================================================
# This is a comprehensive Python implementation that mirrors the functionality
# of R's Logistic Regression for credit risk modeling workflows.
# It includes a Shiny user interface for interactive variable selection.
# The script is designed to work within KNIME 5.9 using Python 3.9.
#
# DEBUG VERSION: Includes extensive logging on every function for debugging
#
# This script operates in two distinct modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided, a graphical
#    interface launches allowing users to select variables and configure the model
# 2. Headless - When DependentVariable and VarSelectionMethod are provided via
#    flow variables, the script runs automatically without user interaction
#
# Inputs:
# 1. A DataFrame containing variables (typically WOE-transformed data from the WOE Editor node)
#    This data should contain the target variable and predictor variables
#
# Outputs:
# 1. Input DataFrame with prediction columns added:
#    - 'probabilities': predicted probability of the positive class (rounded to 6 decimals)
#    - 'predicted': binary classification ("1" or "0") based on 0.5 threshold
# 2. Model coefficients table with variable names as row indices and coefficient values
#
# Flow Variables (for headless mode - when you want automated execution):
# - DependentVariable (string): The name of the binary target variable column
# - TargetCategory (optional): Which value in the target represents the "bad" outcome
# - VarSelectionMethod (string): Variable selection approach - one of:
#     "All" - use all provided variables without selection
#     "Stepwise" - bidirectional stepwise selection
#     "Forward" - start with no variables, add one at a time
#     "Backward" - start with all variables, remove one at a time
# - Cutoff (float, default 2): AIC penalty multiplier (k in stepAIC formula)
#     k=2 gives standard AIC (Akaike Information Criterion)
#     k=log(n) gives BIC (Bayesian Information Criterion) for stricter selection
#
# Release Date: 2026-01-17
# Version: 1.0-DEBUG
# =============================================================================

# Import the KNIME scripting interface - this module provides access to input/output
# tables and flow variables within the KNIME environment
import knime.scripting.io as knio

# Import pandas for data manipulation - the primary library for handling tabular data
# in Python, providing DataFrame and Series data structures
import pandas as pd

# Import numpy for numerical operations - provides efficient array operations,
# mathematical functions, and is the foundation for many scientific computing libraries
import numpy as np

# Import warnings module to control warning messages - we'll use this to suppress
# certain warnings during model fitting that are expected but not informative
import warnings

# Import gc (garbage collector) for memory management - allows explicit triggering
# of garbage collection to free up memory after large operations
import gc

# Import sys for system-specific parameters and functions - used here for
# stdout flushing and other system-level operations
import sys

# Import random for generating random numbers - used for selecting random ports
# to avoid conflicts when multiple instances run simultaneously
import random

# Import os for operating system interface - provides access to environment
# variables and file system operations
import os

# Import logging for comprehensive debug logging
import logging

# Import time for timing measurements
import time

# Import functools for function decorators
import functools

# Import traceback for detailed error information
import traceback as tb

# Import typing module components for type hints - these make the code more
# readable and help IDEs provide better autocompletion:
# - Dict: dictionary type hint (key-value mapping)
# - List: list type hint (ordered collection)
# - Tuple: tuple type hint (immutable ordered collection)
# - Optional: indicates a value can be None
# - Any: any type is acceptable
# - Union: value can be one of several types
from typing import Dict, List, Tuple, Optional, Any, Union

# Import dataclass decorator from dataclasses module - provides a clean way to
# create classes that are primarily used to store data with automatic __init__,
# __repr__, and other methods generated
from dataclasses import dataclass

# Note: Warning suppression during stepwise iterations is handled within the
# fit_logit_model function to avoid cluttering the console during optimization

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================
# This section sets up a comprehensive logging system for debugging purposes.
# All function calls, parameter values, return values, and timing information
# are logged to help trace execution flow and identify issues.

# Create a logger instance for this module
logger = logging.getLogger('LogisticRegression_Commentated_DEBUG')
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to prevent duplicate log messages
# This is important when the script might be reloaded
logger.handlers = []

# Create a console handler that outputs to stdout
# This allows logs to appear in KNIME's console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a detailed formatter that includes:
# - Timestamp with milliseconds for precise timing
# - Log level (DEBUG, INFO, WARNING, ERROR)
# - Function name and line number for easy source location
# - The actual log message
formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Counter for tracking function calls - helps understand execution order
_call_counter = {'count': 0}


def debug_log_function(func):
    """
    Decorator that adds comprehensive debug logging to any function.
    
    This decorator wraps a function to automatically log:
    1. Function entry with all parameters
    2. Function exit with return value
    3. Execution time
    4. Any exceptions that occur
    
    Usage:
        @debug_log_function
        def my_function(arg1, arg2):
            return result
    
    The decorator uses functools.wraps to preserve the original function's
    metadata (name, docstring, etc.) for introspection and debugging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Increment and capture call counter for this invocation
        _call_counter['count'] += 1
        call_id = _call_counter['count']
        func_name = func.__name__
        
        # Log function entry with a visual separator
        logger.debug(f"{'='*60}")
        logger.debug(f"[CALL #{call_id}] ENTERING: {func_name}")
        
        # Log all positional arguments with safe representation
        # Large objects like DataFrames are summarized to avoid log bloat
        for i, arg in enumerate(args):
            arg_repr = _safe_repr(arg)
            logger.debug(f"  arg[{i}]: {arg_repr}")
        
        # Log all keyword arguments
        for key, value in kwargs.items():
            value_repr = _safe_repr(value)
            logger.debug(f"  kwarg[{key}]: {value_repr}")
        
        # Record start time for duration calculation
        start_time = time.time()
        
        try:
            # Execute the actual function
            result = func(*args, **kwargs)
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            # Log successful completion with return value
            result_repr = _safe_repr(result)
            logger.debug(f"[CALL #{call_id}] EXITING: {func_name}")
            logger.debug(f"  elapsed_time: {elapsed:.4f}s")
            logger.debug(f"  return_value: {result_repr}")
            logger.debug(f"{'='*60}")
            
            return result
            
        except Exception as e:
            # Log exception details for debugging
            elapsed = time.time() - start_time
            logger.error(f"[CALL #{call_id}] EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
            logger.error(f"  elapsed_time: {elapsed:.4f}s")
            logger.error(f"  traceback: {tb.format_exc()}")
            logger.debug(f"{'='*60}")
            # Re-raise the exception to preserve normal error handling
            raise
    
    return wrapper


def _safe_repr(obj, max_len: int = 200) -> str:
    """
    Create a safe string representation of an object, truncating if too long.
    
    This function handles various object types specially to provide informative
    yet concise representations suitable for logging:
    - DataFrames: Shows shape and column names
    - Series: Shows length and dtype
    - ndarrays: Shows shape and dtype
    - Lists/Dicts: Shows length and sample of contents
    - Other objects: Uses repr with length limit
    
    Parameters:
        obj: Any Python object to represent
        max_len: Maximum string length before truncation
        
    Returns:
        A string representation of the object
    """
    try:
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, show shape and first few column names
            cols = list(obj.columns)[:5]
            suffix = '...' if len(obj.columns) > 5 else ''
            return f"DataFrame(shape={obj.shape}, columns={cols}{suffix})"
        elif isinstance(obj, pd.Series):
            # For Series, show length and data type
            return f"Series(len={len(obj)}, dtype={obj.dtype})"
        elif isinstance(obj, np.ndarray):
            # For numpy arrays, show shape and data type
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        elif isinstance(obj, list):
            # For lists, show length and first few elements
            if len(obj) > 10:
                return f"list(len={len(obj)}, first_5={obj[:5]}...)"
            return f"list({obj})"
        elif isinstance(obj, dict):
            # For dicts, show length and first few keys
            if len(obj) > 5:
                keys = list(obj.keys())[:5]
                return f"dict(len={len(obj)}, keys={keys}...)"
            return f"dict({obj})"
        else:
            # For other objects, use repr with truncation
            repr_str = repr(obj)
            if len(repr_str) > max_len:
                return repr_str[:max_len] + "..."
            return repr_str
    except Exception:
        # Fallback if repr fails
        return f"<{type(obj).__name__}>"


def log_variable(name: str, value: Any, context: str = ""):
    """
    Log a variable's value with optional context information.
    
    This helper function provides a consistent format for logging variable
    values throughout the script. It's useful for tracking state changes
    and debugging data flow.
    
    Parameters:
        name: The variable name for the log message
        value: The value to log (will be safely represented)
        context: Optional additional context (e.g., "flow_variable", "computed")
    """
    value_repr = _safe_repr(value)
    if context:
        logger.debug(f"[VAR] {context} | {name} = {value_repr}")
    else:
        logger.debug(f"[VAR] {name} = {value_repr}")


def log_checkpoint(message: str):
    """
    Log a checkpoint message marking significant points in execution.
    
    Checkpoints help identify where in the execution flow the script is,
    making it easier to trace issues and understand the processing sequence.
    
    Parameters:
        message: A descriptive message for the checkpoint
    """
    logger.info(f"[CHECKPOINT] {message}")


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log detailed information about a DataFrame.
    
    This function logs comprehensive DataFrame information including:
    - Shape (rows x columns)
    - Column names
    - Data types
    - Memory usage
    - Null value counts
    - First row sample
    
    Parameters:
        df: The DataFrame to describe
        name: A label for the DataFrame in the log
    """
    logger.debug(f"[DATAFRAME INFO] {name}:")
    logger.debug(f"  shape: {df.shape}")
    logger.debug(f"  columns: {list(df.columns)}")
    logger.debug(f"  dtypes: {dict(df.dtypes)}")
    logger.debug(f"  memory_usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    logger.debug(f"  null_counts: {dict(df.isnull().sum())}")
    if len(df) > 0:
        logger.debug(f"  first_row: {dict(df.iloc[0])}")


# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# When running multiple KNIME workflows simultaneously, each containing this
# logistic regression node, we need to ensure they don't conflict with each other.
# These settings help maintain isolation between instances.

log_checkpoint("Initializing stability settings")

# Base port number for the Shiny web application - set to 8053 which is different
# from other nodes in the credit risk toolkit (WOE Editor uses 8050, etc.)
# This minimizes the chance of port conflicts when running multiple nodes
BASE_PORT = 8053  # Different from other scripts to avoid conflicts

# Range of random port offsets - when the base port is busy, we'll try random
# ports within this range (8053 to 9053)
RANDOM_PORT_RANGE = 1000

# Create a unique identifier for this specific process instance by combining
# the process ID (unique per running Python process) with a random number
# This ensures each instance can be identified and tracked separately
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

log_variable("BASE_PORT", BASE_PORT)
log_variable("RANDOM_PORT_RANGE", RANDOM_PORT_RANGE)
log_variable("INSTANCE_ID", INSTANCE_ID)

# Set environment variable to limit numexpr to single thread - numexpr is used
# by pandas for fast numerical expression evaluation. Multiple threads can
# cause conflicts when running parallel instances
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts
logger.debug("Set NUMEXPR_MAX_THREADS=1")

# Set environment variable to limit OpenMP to single thread - OpenMP is a
# parallel programming interface used by many numerical libraries. Limiting
# it prevents resource contention between instances
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts
logger.debug("Set OMP_NUM_THREADS=1")

# =============================================================================
# Install/Import Dependencies
# =============================================================================
# This section handles the installation of required packages if they're not
# already installed, and then imports them. This ensures the script can run
# even in a fresh Python environment.

log_checkpoint("Installing/importing dependencies")


@debug_log_function
def install_if_missing(package, import_name=None):
    """
    Install a Python package if it's not already available in the environment.
    
    This function attempts to import a package, and if the import fails (meaning
    the package isn't installed), it uses pip to install it. This is useful for
    ensuring all dependencies are available before the main script runs.
    
    Parameters:
        package: The name of the package as it appears on PyPI (for pip install)
        import_name: The name used to import the package in Python code
                    If None, assumes the import name matches the package name
                    
    Example:
        install_if_missing('scikit-learn', 'sklearn')
        # pip install uses 'scikit-learn', but import uses 'sklearn'
    """
    # If no import name provided, assume it matches the package name
    # This is the common case for most packages
    if import_name is None:
        import_name = package
    
    logger.debug(f"Checking for package: {package} (import as: {import_name})")
    
    # Attempt to import the package using Python's __import__ function
    # This is equivalent to an 'import package' statement but allows
    # the package name to be a variable
    try:
        __import__(import_name)
        logger.debug(f"Package {package} already installed")
    except ImportError:
        # If import fails, the package isn't installed
        # Import subprocess to run external commands (pip)
        logger.info(f"Installing missing package: {package}")
        import subprocess
        # Run pip install command and wait for it to complete
        # check_call raises an exception if the command fails
        subprocess.check_call(['pip', 'install', package])
        logger.info(f"Successfully installed: {package}")


# Install statsmodels if not present - this is the core statistical modeling
# library that provides the Logit class for logistic regression
install_if_missing('statsmodels')

# Install scikit-learn if not present - provides metrics like ROC-AUC score
# Note: the PyPI package name is 'scikit-learn' but the import name is 'sklearn'
install_if_missing('scikit-learn', 'sklearn')

# Install Shiny for Python if not present - provides the interactive web UI
# This is different from R's Shiny but has similar functionality
install_if_missing('shiny')

# Install shinywidgets for interactive Plotly charts in Shiny
install_if_missing('shinywidgets')

# Install Plotly for creating interactive charts (coefficient plots, ROC curves)
install_if_missing('plotly')

# Now import the statsmodels API - 'sm' is the conventional alias
# statsmodels provides statistical models including logistic regression
import statsmodels.api as sm
logger.debug("Imported statsmodels")

# Import the Logit class specifically from statsmodels
# This class implements logistic regression for binary outcomes
from statsmodels.discrete.discrete_model import Logit
logger.debug("Imported Logit class")

# Import roc_auc_score from sklearn.metrics - this function calculates the
# Area Under the ROC Curve, a key metric for classification model performance
from sklearn.metrics import roc_auc_score
logger.debug("Imported roc_auc_score")

# Attempt to import Shiny and related packages for interactive UI
# These imports are wrapped in try/except because the UI is optional -
# the script can still run in headless mode without Shiny
try:
    # Import core Shiny components:
    # - App: the main application class that combines UI and server logic
    # - Inputs: type hint for input bindings (user interactions)
    # - Outputs: type hint for output bindings (rendered content)
    # - Session: represents a user's session in the app
    # - reactive: decorators for reactive programming patterns
    # - render: decorators for rendering outputs
    # - ui: functions for building the user interface
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Import render_plotly and output_widget from shinywidgets
    # These enable embedding interactive Plotly charts in Shiny apps
    from shinywidgets import render_plotly, output_widget
    
    # Import Plotly's graph_objects module for creating figures
    # This provides lower-level control over chart creation
    import plotly.graph_objects as go
    
    # Import Plotly express for quick chart creation
    # (though we primarily use graph_objects in this script)
    import plotly.express as px
    
    # Set flag indicating Shiny is available for interactive mode
    SHINY_AVAILABLE = True
    logger.info("Shiny components imported successfully")
except ImportError as e:
    # If any Shiny-related import fails, print a warning and disable interactive mode
    # The script will still work in headless mode with flow variables
    logger.warning(f"Shiny not available: {e}")
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False

log_variable("SHINY_AVAILABLE", SHINY_AVAILABLE)


# =============================================================================
# Diagnostic Functions
# =============================================================================
# These functions help identify potential issues with the data before fitting
# the logistic regression model, particularly multicollinearity which can
# cause numerical instability and unreliable coefficient estimates.

@debug_log_function
def check_multicollinearity(df: pd.DataFrame, x_vars: List[str], threshold: float = 0.85, 
                            vif_threshold: float = 10.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Check for multicollinearity and other data issues before model fitting.
    
    Multicollinearity occurs when predictor variables are highly correlated with
    each other. This can cause several problems in logistic regression:
    1. Unstable coefficient estimates (small data changes cause big swings)
    2. Inflated standard errors making significance tests unreliable
    3. Numerical issues during model fitting (Hessian matrix inversion warnings)
    
    This function checks for three types of issues:
    1. Low variance variables - nearly constant, provide little information
    2. High correlation pairs - variables that move together
    3. High VIF (Variance Inflation Factor) - multicollinearity indicator
    
    Parameters:
        df: DataFrame containing the data to check
        x_vars: List of predictor variable names to analyze
        threshold: Correlation coefficient threshold for flagging pairs (default 0.85)
                  Correlations above this are considered problematically high
        vif_threshold: VIF threshold for flagging variables (default 10.0)
                      VIF > 10 is commonly used as a multicollinearity indicator
        verbose: If True, print detailed diagnostic output to console
        
    Returns:
        Dictionary containing diagnostic results with keys:
        - 'high_correlations': list of (var1, var2, correlation) tuples
        - 'high_vif_vars': list of (variable, vif_value) tuples
        - 'low_variance_vars': list of variable names with near-zero variance
        - 'issues_found': boolean indicating if any issues were detected
    """
    log_checkpoint("Starting multicollinearity check")
    log_variable("threshold", threshold)
    log_variable("vif_threshold", vif_threshold)
    log_variable("num_x_vars", len(x_vars))
    
    # Initialize the results dictionary with empty lists and no issues flag
    # This structure will be populated as we run each diagnostic check
    results = {
        'high_correlations': [],     # Will store tuples of correlated variable pairs
        'high_vif_vars': [],         # Will store tuples of variables with high VIF
        'low_variance_vars': [],     # Will store names of low-variance variables
        'issues_found': False        # Flag that's set True if any issue is detected
    }
    
    # If we have fewer than 2 predictor variables, correlation/VIF checks
    # don't make sense (can't have multicollinearity with 1 variable)
    if len(x_vars) < 2:
        logger.debug("Less than 2 variables - skipping multicollinearity check")
        return results
    
    # Extract only the predictor columns and convert all to float type
    # This ensures consistent numeric type for correlation calculations
    X = df[x_vars].astype(float)
    logger.debug(f"Created X matrix with shape: {X.shape}")
    
    # Check for low variance variables - these are variables that are nearly
    # constant across all observations. They provide little predictive value
    # and can cause numerical issues in matrix operations
    logger.debug("Checking for low variance variables")
    variances = X.var()  # Calculate variance for each column
    
    # Find variables with variance less than 1e-10 (essentially zero)
    # These variables have almost no variation in their values
    low_var = variances[variances < 1e-10].index.tolist()
    
    # If any low-variance variables found, record them and set the flag
    if low_var:
        logger.warning(f"Found {len(low_var)} low variance variables: {low_var[:5]}")
        results['low_variance_vars'] = low_var
        results['issues_found'] = True
    
    # Check correlation matrix for highly correlated pairs
    # High correlation between predictors indicates redundant information
    logger.debug("Computing correlation matrix")
    try:
        # Calculate the correlation matrix and take absolute values
        # We care about the strength of relationship, not direction
        corr_matrix = X.corr().abs()
        logger.debug(f"Correlation matrix computed, shape: {corr_matrix.shape}")
        
        # Initialize list to store pairs with high correlation
        high_corr_pairs = []
        
        # Loop through all unique pairs of variables (upper triangle only)
        # We use indices to ensure we only check each pair once
        for i, var1 in enumerate(x_vars):
            for j, var2 in enumerate(x_vars):
                if i < j:  # Only upper triangle - avoids duplicates and self-correlation
                    # Get the correlation value for this pair
                    corr = corr_matrix.loc[var1, var2]
                    
                    # If correlation exceeds threshold, record this pair
                    if corr > threshold:
                        high_corr_pairs.append((var1, var2, corr))
        
        # If any highly correlated pairs found, sort by correlation (highest first)
        # and record them in results
        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} highly correlated pairs")
            results['high_correlations'] = sorted(high_corr_pairs, key=lambda x: -x[2])
            results['issues_found'] = True
    except Exception as e:
        # If correlation calculation fails for any reason, log and continue
        # This is a diagnostic tool, so failure shouldn't stop the main process
        logger.error(f"Error computing correlation matrix: {e}")
    
    # Calculate Variance Inflation Factor (VIF) for each variable
    # VIF measures how much the variance of a regression coefficient is inflated
    # due to multicollinearity. VIF = 1 means no correlation with other predictors.
    # VIF > 10 is commonly used as a threshold indicating problematic multicollinearity.
    logger.debug("Calculating VIF values")
    try:
        # Import the VIF calculation function from statsmodels
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Only calculate VIF if we have appropriate data dimensions:
        # - More than 1 variable (need at least 2 for multicollinearity)
        # - Fewer than 50 variables (VIF calculation can be slow with many variables)
        # - Enough observations (need more rows than variables + some buffer)
        if len(x_vars) > 1 and len(x_vars) < 50 and len(df) > len(x_vars) + 10:
            # Add a constant column (intercept) to the design matrix
            # VIF calculation requires this to match regression setup
            # dropna() removes rows with missing values which would cause issues
            X_with_const = sm.add_constant(X.dropna())
            logger.debug(f"X_with_const shape after dropna: {X_with_const.shape}")
            
            # Verify we have enough complete cases after dropping NA
            if len(X_with_const) > len(x_vars) + 1:
                # Initialize list to store variables with high VIF
                vif_data = []
                
                # Calculate VIF for each predictor variable
                for i, var in enumerate(x_vars):
                    try:
                        # variance_inflation_factor requires the data matrix and
                        # the index of the column to calculate VIF for
                        # We use i+1 because index 0 is the constant we added
                        vif = variance_inflation_factor(X_with_const.values, i + 1)
                        logger.debug(f"VIF for {var}: {vif}")
                        
                        # If VIF exceeds threshold and isn't infinite, record it
                        # Infinite VIF indicates perfect multicollinearity
                        if vif > vif_threshold and not np.isinf(vif):
                            vif_data.append((var, vif))
                    except Exception as e:
                        # If VIF calculation fails for one variable, continue to others
                        logger.warning(f"VIF calculation failed for {var}: {e}")
                
                # If any high-VIF variables found, sort by VIF (highest first)
                if vif_data:
                    logger.warning(f"Found {len(vif_data)} high VIF variables")
                    results['high_vif_vars'] = sorted(vif_data, key=lambda x: -x[1])
                    results['issues_found'] = True
    except ImportError:
        # If variance_inflation_factor isn't available, skip VIF calculation
        logger.debug("variance_inflation_factor not available")
    except Exception as e:
        # For any other error, log and continue
        logger.error(f"VIF calculation error: {e}")
    
    # Print detailed diagnostic output if verbose mode is enabled and issues were found
    if verbose and results['issues_found']:
        # Print header with separator lines and emoji for visibility
        print("\n" + "=" * 70)
        print("⚠️  MULTICOLLINEARITY DIAGNOSTICS")
        print("=" * 70)
        
        # Report low variance variables - most severe issue
        if results['low_variance_vars']:
            print(f"\n🔴 LOW VARIANCE VARIABLES ({len(results['low_variance_vars'])}):")
            print("   These variables have near-zero variance and may cause fitting issues:")
            # Print first 10 variables (could be many)
            for var in results['low_variance_vars'][:10]:
                print(f"     - {var}")
            # If more than 10, indicate how many more
            if len(results['low_variance_vars']) > 10:
                print(f"     ... and {len(results['low_variance_vars']) - 10} more")
        
        # Report highly correlated pairs - common issue with WOE variables
        if results['high_correlations']:
            print(f"\n🟠 HIGHLY CORRELATED PAIRS (r > {threshold}):")
            print("   Consider removing one variable from each pair:")
            # Print first 10 pairs with their correlation values
            for var1, var2, corr in results['high_correlations'][:10]:
                print(f"     - {var1} ↔ {var2}: r = {corr:.3f}")
            if len(results['high_correlations']) > 10:
                print(f"     ... and {len(results['high_correlations']) - 10} more pairs")
        
        # Report high VIF variables - indicates multicollinearity
        if results['high_vif_vars']:
            print(f"\n🟡 HIGH VIF VARIABLES (VIF > {vif_threshold}):")
            print("   These variables have high multicollinearity with other predictors:")
            # Print first 10 variables with their VIF values
            for var, vif in results['high_vif_vars'][:10]:
                print(f"     - {var}: VIF = {vif:.1f}")
            if len(results['high_vif_vars']) > 10:
                print(f"     ... and {len(results['high_vif_vars']) - 10} more")
        
        # Print recommendation footer
        print("\n" + "-" * 70)
        print("💡 RECOMMENDATION: Address these issues in earlier pipeline steps")
        print("   (e.g., remove correlated variables in Variable Selection node)")
        print("=" * 70 + "\n")
    
    log_variable("issues_found", results['issues_found'])
    # Return the results dictionary with all diagnostic findings
    return results


# =============================================================================
# Data Classes
# =============================================================================
# Data classes provide a clean, readable way to define classes that are primarily
# used to store data. The @dataclass decorator automatically generates __init__,
# __repr__, __eq__ and other methods based on the class attributes.

@dataclass
class StepwiseResult:
    """
    Container for stepwise selection results.
    
    This class holds the output of any stepwise variable selection process
    (forward, backward, or bidirectional). It stores which variables were
    selected, the AIC history showing how the model improved, and descriptions
    of each step taken during the selection process.
    
    Attributes:
        selected_vars: List of variable names that were selected for the final model
        aic_history: List of AIC values at each step of the selection process,
                    showing how the model's fit evolved
        steps: List of human-readable descriptions of what happened at each step
              (e.g., "+ WOE_Age: AIC=1234.5" for adding a variable)
    """
    selected_vars: List[str]  # Selected variable names
    aic_history: List[float]  # AIC at each step
    steps: List[str]  # Description of each step


@dataclass
class ModelResult:
    """
    Container for logistic regression model results.
    
    This class packages all outputs from fitting a logistic regression model
    into a single object. This makes it easy to pass results between functions
    and access different components of the model output.
    
    Attributes:
        model: The fitted statsmodels model object, which contains the full
              statistical output including coefficients, standard errors,
              p-values, summary tables, and prediction methods
        coefficients: DataFrame with variable names as index and coefficient
                     values as the single column. The intercept is labeled
                     '(Intercept)' to match R's convention
        predictions: DataFrame containing the original data plus added columns
                    'probabilities' (predicted probability) and 'predicted'
                    (binary "1" or "0" based on 0.5 threshold)
        selected_vars: List of variable names included in the final model
                      (may differ from input if stepwise selection was used)
    """
    model: Any  # Fitted statsmodels model
    coefficients: pd.DataFrame  # Coefficient table (variable name as index, coefficient value)
    predictions: pd.DataFrame  # DataFrame with probabilities and predicted columns added
    selected_vars: List[str]  # Variables in final model


# =============================================================================
# Stepwise Selection Functions (equivalent to R's MASS::stepAIC)
# =============================================================================
# These functions implement stepwise variable selection algorithms that are
# commonly used in statistical modeling to find a parsimonious model.
# The goal is to find a subset of variables that balances model fit (likelihood)
# with model complexity (number of parameters), using AIC as the criterion.

# Global variable to track which variables cause numerical issues during stepwise
# selection. This helps with debugging and understanding why certain variables
# might have been excluded or caused fitting problems.
_stepwise_numerical_issues = set()


@debug_log_function
def fit_logit_model(df: pd.DataFrame, y_var: str, x_vars: List[str], track_issues: bool = True) -> Tuple[Any, float]:
    """
    Fit a logistic regression model and return the model and AIC.
    
    This is a utility function used by the stepwise selection algorithms.
    It fits a single logistic regression model with the specified variables
    and returns both the fitted model and its AIC value. The function handles
    various edge cases and numerical issues that can occur during fitting.
    
    Parameters:
        df: DataFrame containing all the data (both predictors and target)
        y_var: Name of the dependent (target) variable column
        x_vars: List of independent (predictor) variable names to include
               Can be empty for null model (intercept only)
        track_issues: If True, record variables that cause numerical warnings
                     in the global _stepwise_numerical_issues set
        
    Returns:
        Tuple containing:
        - The fitted model object (or None if fitting failed)
        - The AIC value (or infinity if fitting failed)
        
    The function tries multiple optimization methods with fallbacks:
    1. BFGS (quasi-Newton method) - fast and usually works
    2. L1 regularized fitting - if BFGS fails
    3. Newton-Raphson - last resort fallback
    """
    # Access the global variable for tracking problematic variables
    global _stepwise_numerical_issues
    
    log_variable("y_var", y_var)
    log_variable("num_x_vars", len(x_vars))
    log_variable("x_vars_sample", x_vars[:5] if x_vars else [])
    log_variable("df_shape", df.shape)
    
    # Handle the case of no predictor variables (null/intercept-only model)
    if not x_vars:
        # For null model, we need to create a design matrix with just a constant
        logger.debug("Fitting null model (intercept only)")
        X = sm.add_constant(pd.DataFrame(index=df.index, columns=[]))
        X = np.ones((len(df), 1))  # Override with explicit ones column
    else:
        # For models with predictors, add a constant column (intercept) and
        # convert all predictor columns to float type for consistent computation
        X = sm.add_constant(df[x_vars].astype(float))
    
    # Extract the target variable and convert to float for modeling
    y = df[y_var].astype(float)
    
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.debug(f"y value counts: {dict(y.value_counts())}")
    
    # Flag to track if we encountered numerical issues during fitting
    had_numerical_issues = False
    
    try:
        # Use context manager to capture warnings during model fitting
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            
            # Try BFGS optimization first - it's generally fast and reliable
            logger.debug("Attempting BFGS optimization")
            try:
                model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
                logger.debug("BFGS optimization succeeded")
            except Exception as e:
                logger.warning(f"BFGS failed: {e}")
                had_numerical_issues = True
                
                # Try L1 regularized fitting
                logger.debug("Attempting L1 regularized fitting")
                try:
                    model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
                    logger.debug("L1 regularized fitting succeeded")
                except Exception as e2:
                    logger.warning(f"L1 regularized failed: {e2}")
                    # Last resort: try basic Newton-Raphson method
                    logger.debug("Attempting Newton-Raphson method")
                    model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
                    logger.debug("Newton-Raphson succeeded")
            
            # Check if any captured warnings indicate Hessian issues
            for w in caught_warnings:
                if 'Hessian' in str(w.message) or 'cov_params' in str(w.message):
                    logger.warning(f"Numerical warning detected: {w.message}")
                    had_numerical_issues = True
                    break
        
        # Track which variables caused numerical issues for reporting
        if track_issues and had_numerical_issues and x_vars:
            logger.debug(f"Tracking numerical issue for variable: {x_vars[-1]}")
            _stepwise_numerical_issues.add(x_vars[-1])
        
        # Get the AIC from the fitted model
        aic = model.aic
        logger.debug(f"Model AIC: {aic}")
        return model, aic
        
    except Exception as e:
        # If all fitting attempts fail, track the issue and return failure values
        logger.error(f"Model fitting failed: {e}")
        if track_issues and x_vars:
            _stepwise_numerical_issues.add(x_vars[-1])
        return None, float('inf')


@debug_log_function
def stepwise_forward(
    df: pd.DataFrame,
    y_var: str,
    candidate_vars: List[str],
    k: float = 2.0,
    verbose: bool = True
) -> StepwiseResult:
    """
    Forward stepwise selection using AIC.
    
    Forward selection starts with an empty model (no predictors) and iteratively
    adds the variable that most improves the model's AIC at each step. The process
    continues until no variable addition improves the AIC.
    
    This is analogous to R's stepAIC with direction="forward".
    
    Parameters:
        df: DataFrame with all the data (predictors and target)
        y_var: Dependent variable name
        candidate_vars: List of candidate predictor variables to consider
        k: Penalty multiplier for AIC calculation
           - k=2 gives standard AIC
           - k=log(n) gives BIC (more conservative selection)
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
    """
    log_checkpoint("Starting forward stepwise selection")
    log_variable("y_var", y_var)
    log_variable("num_candidate_vars", len(candidate_vars))
    log_variable("k", k)
    
    # Initialize lists for tracking the selection process
    selected = []
    remaining = list(candidate_vars)
    aic_history = []
    steps = []
    
    # Fit the null model to get starting AIC
    logger.debug("Fitting null model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Forward Selection: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    iteration = 0
    improved = True
    while improved and remaining:
        iteration += 1
        logger.debug(f"Forward iteration {iteration}: {len(selected)} selected, {len(remaining)} remaining")
        improved = False
        best_var = None
        best_aic = current_aic
        
        # Try adding each remaining variable to the model
        for var in remaining:
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                improved = True
        
        # If we found an improvement, add the best variable
        if improved and best_var:
            logger.info(f"Forward: adding {best_var} (AIC: {current_aic:.4f} -> {best_aic:.4f})")
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"+ {best_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  + {best_var}: AIC = {current_aic:.4f}")
    
    log_checkpoint(f"Forward selection complete: {len(selected)} variables selected")
    if verbose:
        print(f"Forward Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


@debug_log_function
def stepwise_backward(
    df: pd.DataFrame,
    y_var: str,
    current_vars: List[str],
    k: float = 2.0,
    verbose: bool = True
) -> StepwiseResult:
    """
    Backward stepwise elimination using AIC.
    
    Backward selection starts with a full model (all predictors) and iteratively
    removes the variable whose removal most improves (or least worsens) the model's
    AIC. The process continues until no removal improves the AIC.
    
    This is analogous to R's stepAIC with direction="backward".
    
    Parameters:
        df: DataFrame with all the data (predictors and target)
        y_var: Name of the dependent variable column
        current_vars: List of all predictor variables to start with
        k: Penalty multiplier for AIC calculation
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
    """
    log_checkpoint("Starting backward stepwise elimination")
    log_variable("y_var", y_var)
    log_variable("num_current_vars", len(current_vars))
    log_variable("k", k)
    
    # Start with all variables and create a copy to modify
    selected = list(current_vars)
    aic_history = []
    steps = []
    
    # Fit the full model to get starting AIC
    logger.debug("Fitting full model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Backward Elimination: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    iteration = 0
    improved = True
    while improved and len(selected) > 0:
        iteration += 1
        logger.debug(f"Backward iteration {iteration}: {len(selected)} variables remaining")
        improved = False
        worst_var = None
        best_aic = current_aic
        
        # Try removing each variable from the model
        for var in selected:
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                worst_var = var
                improved = True
        
        # If we found an improvement, remove the worst variable
        if improved and worst_var:
            logger.info(f"Backward: removing {worst_var} (AIC: {current_aic:.4f} -> {best_aic:.4f})")
            selected.remove(worst_var)
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"- {worst_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  - {worst_var}: AIC = {current_aic:.4f}")
    
    log_checkpoint(f"Backward elimination complete: {len(selected)} variables remaining")
    if verbose:
        print(f"Backward Elimination: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


@debug_log_function
def stepwise_both(
    df: pd.DataFrame,
    y_var: str,
    current_vars: List[str],
    k: float = 2.0,
    verbose: bool = True
) -> StepwiseResult:
    """
    Stepwise selection (both directions) using AIC.
    Like R's stepAIC with direction="both".
    
    Bidirectional stepwise selection combines forward and backward steps.
    At each iteration, it considers both adding a removed variable and removing
    a current variable, choosing whichever action most improves the AIC.
    
    Parameters:
        df: DataFrame with all the data (predictors and target)
        y_var: Name of the dependent variable column
        current_vars: List of predictor variables to start with (typically all)
        k: Penalty multiplier for AIC calculation
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
    """
    log_checkpoint("Starting bidirectional stepwise selection")
    log_variable("y_var", y_var)
    log_variable("num_current_vars", len(current_vars))
    log_variable("k", k)
    
    # Initialize with current variables
    selected = list(current_vars)
    remaining = []
    aic_history = []
    steps = []
    all_vars = list(current_vars)
    
    # Fit current model to get starting AIC
    logger.debug("Fitting initial model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Stepwise Selection: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    improved = True
    iteration = 0
    max_iterations = len(all_vars) * 2
    
    while improved and iteration < max_iterations:
        iteration += 1
        logger.debug(f"Stepwise iteration {iteration}: {len(selected)} selected, {len(remaining)} remaining")
        improved = False
        best_action = None
        best_var = None
        best_aic = current_aic
        
        # Try removing each variable (backward step)
        for var in selected:
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                best_action = 'remove'
                improved = True
        
        # Try adding each removed variable (forward step)
        for var in remaining:
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                best_action = 'add'
                improved = True
        
        # If we found an improvement, apply the best action
        if improved and best_var:
            if best_action == 'remove':
                logger.info(f"Stepwise: removing {best_var} (AIC: {current_aic:.4f} -> {best_aic:.4f})")
                selected.remove(best_var)
                remaining.append(best_var)
                steps.append(f"- {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  - {best_var}: AIC = {best_aic:.4f}")
            else:
                logger.info(f"Stepwise: adding {best_var} (AIC: {current_aic:.4f} -> {best_aic:.4f})")
                selected.append(best_var)
                remaining.remove(best_var)
                steps.append(f"+ {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  + {best_var}: AIC = {best_aic:.4f}")
            
            current_aic = best_aic
            aic_history.append(current_aic)
    
    log_checkpoint(f"Stepwise selection complete: {len(selected)} variables selected after {iteration} iterations")
    if verbose:
        print(f"Stepwise Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


# =============================================================================
# Model Fitting and Evaluation
# =============================================================================

@debug_log_function
def fit_logistic_regression(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    method: str = "All",
    k: float = 2.0,
    verbose: bool = True
) -> ModelResult:
    """
    Fit logistic regression with optional variable selection.
    
    This is the main entry point for fitting a logistic regression model.
    It handles data preparation, optional stepwise variable selection,
    model fitting, and prediction generation.
    
    Parameters:
        df: DataFrame containing both predictor variables and the target variable
        y_var: Name of the dependent (target) variable column
        x_vars: List of predictor variable names to consider
        method: Variable selection method - one of:
               - "All": Use all provided variables (no selection)
               - "Stepwise": Bidirectional stepwise selection
               - "Forward": Forward stepwise selection
               - "Backward": Backward elimination
        k: AIC penalty multiplier
        verbose: If True, print progress and diagnostics
        
    Returns:
        ModelResult containing the fitted model, coefficients, predictions,
        and list of variables in the final model
    """
    log_checkpoint("Starting logistic regression fitting")
    log_variable("y_var", y_var)
    log_variable("num_x_vars", len(x_vars))
    log_variable("method", method)
    log_variable("k", k)
    log_dataframe_info(df, "Input DataFrame")
    
    # Create list of all columns needed
    cols_to_use = [y_var] + x_vars
    
    # Remove rows with missing values
    df_clean = df[cols_to_use].dropna()
    logger.debug(f"After dropna: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")
    
    if len(df_clean) == 0:
        logger.error("No complete cases after removing missing values")
        raise ValueError("No complete cases after removing missing values")
    
    if verbose:
        print(f"Fitting logistic regression: {len(df_clean)} observations, {len(x_vars)} variables")
        print(f"Method: {method}, k = {k}")
    
    # Run multicollinearity diagnostics
    if verbose:
        logger.debug("Running multicollinearity diagnostics")
        diagnostics = check_multicollinearity(df_clean, x_vars, threshold=0.85, vif_threshold=10.0, verbose=True)
    
    # Reset numerical issues tracker
    global _stepwise_numerical_issues
    _stepwise_numerical_issues = set()
    
    # Apply the selected variable selection method
    logger.debug(f"Applying variable selection method: {method}")
    if method == "All":
        selected_vars = x_vars
        stepwise_result = None
    elif method == "Forward":
        stepwise_result = stepwise_forward(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    elif method == "Backward":
        stepwise_result = stepwise_backward(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    elif method == "Stepwise":
        stepwise_result = stepwise_both(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    else:
        logger.warning(f"Unknown method '{method}', using all variables")
        selected_vars = x_vars
        stepwise_result = None
    
    log_variable("num_selected_vars", len(selected_vars))
    log_variable("selected_vars", selected_vars)
    
    if not selected_vars:
        logger.error("No variables selected - model cannot be fit")
        raise ValueError("No variables selected - model cannot be fit")
    
    # Report numerical issues
    if verbose and _stepwise_numerical_issues and method != "All":
        print("\n" + "-" * 70)
        print("⚠️  NUMERICAL ISSUES DURING STEPWISE SELECTION")
        print("-" * 70)
        print("The following variables caused Hessian inversion warnings or fit failures:")
        for var in sorted(_stepwise_numerical_issues):
            status = "✓ selected" if var in selected_vars else "✗ not selected"
            print(f"  - {var} ({status})")
        print("\nThis typically indicates multicollinearity or separation issues.")
        print("Consider reviewing correlated variables in earlier pipeline steps.")
        print("-" * 70 + "\n")
    
    # Fit final model
    logger.debug("Fitting final model with selected variables")
    X = sm.add_constant(df_clean[selected_vars].astype(float))
    y = df_clean[y_var].astype(float)
    
    logger.debug(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            logger.debug("Attempting final model fit with BFGS")
            model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
            logger.debug("Final model BFGS succeeded")
        except Exception as e:
            logger.warning(f"Final model BFGS failed: {e}")
            if verbose:
                print("  Note: BFGS optimization had issues, trying Newton method...")
            try:
                logger.debug("Attempting Newton method")
                model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
                logger.debug("Final model Newton succeeded")
            except Exception as e2:
                logger.warning(f"Final model Newton failed: {e2}")
                if verbose:
                    print("  Note: Newton method had issues, using L1 regularization...")
                logger.debug("Attempting L1 regularization")
                model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
                logger.debug("Final model L1 regularized succeeded")
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL SUMMARY")
        print("="*70)
        print(model.summary())
    
    # Create coefficient table
    logger.debug("Creating coefficient table")
    coef_df = model.params.to_frame(name='coefficients')
    coef_df.index.name = None
    
    if 'const' in coef_df.index:
        coef_df = coef_df.rename(index={'const': '(Intercept)'})
    
    log_variable("coefficients", dict(coef_df['coefficients']))
    
    # Calculate predictions
    logger.debug("Calculating predictions for original dataframe")
    predictions = df.copy()
    X_full = sm.add_constant(df[selected_vars].astype(float), has_constant='add')
    complete_mask = X_full.notna().all(axis=1)
    
    logger.debug(f"Complete cases for prediction: {complete_mask.sum()} of {len(df)}")
    
    predictions['probabilities'] = np.nan
    predictions['predicted'] = None
    
    if complete_mask.any():
        X_complete = X_full[complete_mask]
        proba = model.predict(X_complete)
        predictions.loc[complete_mask, 'probabilities'] = np.round(proba.values, 6)
        predictions.loc[complete_mask, 'predicted'] = np.where(proba.values > 0.5, "1", "0")
        
        logger.debug(f"Predictions made: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    
    # Calculate AUC
    if verbose:
        try:
            y_actual = df_clean[y_var].astype(float)
            y_proba = model.predict(X)
            auc = roc_auc_score(y_actual, y_proba)
            logger.info(f"Model AUC: {auc:.4f}, Gini: {2*auc - 1:.4f}")
            print(f"\nModel AUC: {auc:.4f}")
            print(f"Gini: {2*auc - 1:.4f}")
        except Exception as e:
            logger.error(f"AUC calculation failed: {e}")
            print(f"Could not calculate AUC: {e}")
    
    log_checkpoint("Logistic regression fitting complete")
    
    return ModelResult(
        model=model,
        coefficients=coef_df,
        predictions=predictions,
        selected_vars=selected_vars
    )


# =============================================================================
# Shiny UI Application
# =============================================================================
# Note: The Shiny UI code is included here but with minimal comments in the
# debug version since the main focus is on the core modeling logic.
# See the non-debug version for detailed UI comments.

@debug_log_function
def create_logistic_regression_app(df: pd.DataFrame):
    """Create the Logistic Regression Shiny application."""
    log_checkpoint("Creating Shiny application")
    log_dataframe_info(df, "App input DataFrame")
    
    app_results = {
        'coefficients': None,
        'predictions': None,
        'selected_vars': None,
        'dv': None,
        'completed': False
    }
    
    # UI definition (abbreviated for debug version - see non-debug for full comments)
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&display=swap');
                body { font-family: 'Raleway', sans-serif; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh; color: #e8e8e8; }
                .card { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; 
                    padding: 24px; margin: 12px 0; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
                .card-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px 12px 0 0; margin: -24px -24px 20px -24px;
                    padding: 16px 24px; color: white; font-weight: 600; }
                h4 { font-weight: 700; text-align: center; margin: 20px 0; color: #fff;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
                h5 { color: #a8dadc; font-weight: 600; margin-bottom: 16px; }
                .btn { font-weight: 600; border-radius: 50px; padding: 10px 24px;
                    text-transform: uppercase; letter-spacing: 1px; font-size: 13px;
                    transition: all 0.3s ease; border: none; }
                .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
                .btn-primary { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
                .btn-success { background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); }
                .btn-info { background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%); }
                .form-control, .form-select { background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2); color: #fff; border-radius: 10px; }
                .form-control:focus, .form-select:focus { background: rgba(255, 255, 255, 0.15);
                    border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3); color: #fff; }
                .form-select option { background: #1a1a2e; color: #fff; }
                .form-label { color: #a8dadc; font-weight: 500; }
                .form-check-input:checked { background-color: #667eea; border-color: #667eea; }
                table { color: #e8e8e8 !important; }
                .var-checkbox-container { max-height: 400px; overflow-y: auto;
                    background: rgba(0, 0, 0, 0.2); border-radius: 10px; padding: 16px; }
            """)
        ),
        ui.h4("🔬 Logistic Regression [DEBUG MODE]"),
        ui.div({"class": "card"}, ui.div({"class": "card-header"}, "Model Configuration"),
            ui.row(
                ui.column(4, ui.input_select("dv", "Dependent Variable", 
                    choices=list(df.columns), selected=df.columns[-1] if len(df.columns) > 0 else None)),
                ui.column(4, ui.input_select("tc", "Target Category", choices=[])),
                ui.column(4, ui.input_select("method", "Variable Selection Method",
                    choices=["Must include all", "Stepwise Selection", "Forward Selection", "Backward Selection"],
                    selected="Must include all"))
            ),
            ui.row(
                ui.column(4, ui.input_numeric("cutoff", "AIC Penalty (k)", value=2, min=0, step=0.5),
                    ui.tags.small("k=2 for AIC, k=log(n) for BIC", style="color: #888;")),
                ui.column(4, ui.input_action_button("select_woe", "Select WOE Variables", class_="btn btn-info")),
                ui.column(4, ui.input_action_button("select_all", "Select All Variables", class_="btn btn-primary"))
            )
        ),
        ui.div({"class": "card"}, ui.div({"class": "card-header"}, "Variable Selection"),
            ui.row(ui.column(12, ui.div({"class": "var-checkbox-container"}, ui.output_ui("var_checkboxes"))))
        ),
        ui.div({"class": "card"}, ui.div({"class": "card-header"}, "Model Results"),
            ui.row(ui.column(12, ui.h5("Coefficients"), ui.output_data_frame("coef_table")))
        ),
        ui.row(
            ui.column(6, ui.div({"class": "card", "style": "height: 400px;"}, ui.h5("Coefficient Plot"), output_widget("coef_plot"))),
            ui.column(6, ui.div({"class": "card", "style": "height: 400px;"}, ui.h5("ROC Curve"), output_widget("roc_plot")))
        ),
        ui.div({"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "🚀 Run Model & Close", class_="btn btn-success btn-lg"))
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        logger.debug("Shiny server function initialized")
        model_result_rv = reactive.Value(None)
        
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            logger.debug(f"[Shiny] DV changed to: {dv}")
            if dv and dv in df.columns:
                unique_vals = [str(v) for v in sorted(df[dv].dropna().unique().tolist())]
                logger.debug(f"[Shiny] Target categories: {unique_vals}")
                ui.update_select("tc", choices=unique_vals, selected=unique_vals[-1] if unique_vals else None)
        
        @output
        @render.ui
        def var_checkboxes():
            dv = input.dv()
            logger.debug(f"[Shiny] Rendering variable checkboxes, DV={dv}")
            if not dv:
                return ui.p("Select a dependent variable first")
            available_vars = [col for col in df.columns if col != dv]
            logger.debug(f"[Shiny] Available vars: {len(available_vars)}")
            return ui.div(*[ui.input_checkbox(f"var_{var}", var, value=var.startswith('WOE_')) for var in available_vars])
        
        @reactive.Effect
        @reactive.event(input.select_woe)
        def select_woe_vars():
            logger.debug("[Shiny] Select WOE variables button clicked")
            dv = input.dv()
            if not dv:
                return
            for var in [col for col in df.columns if col != dv]:
                try: ui.update_checkbox(f"var_{var}", value=var.startswith('WOE_'))
                except: pass
        
        @reactive.Effect
        @reactive.event(input.select_all)
        def select_all_vars():
            logger.debug("[Shiny] Select all variables button clicked")
            dv = input.dv()
            if not dv:
                return
            for var in [col for col in df.columns if col != dv]:
                try: ui.update_checkbox(f"var_{var}", value=not var.startswith('b_'))
                except: pass
        
        @reactive.Calc
        def get_selected_vars():
            dv = input.dv()
            if not dv:
                return []
            selected = []
            for var in [col for col in df.columns if col != dv]:
                try:
                    if input[f"var_{var}"]():
                        selected.append(var)
                except: pass
            return selected
        
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_model():
            logger.info("[Shiny] Run Model button clicked")
            dv = input.dv()
            selected = get_selected_vars()
            method_raw = input.method()
            cutoff = input.cutoff()
            
            logger.debug(f"[Shiny] DV: {dv}, Method: {method_raw}, Cutoff: {cutoff}, Selected: {len(selected)}")
            
            if not dv or not selected:
                logger.warning("[Shiny] No DV or no variables selected")
                return
            
            method = {"Must include all": "All", "Stepwise Selection": "Stepwise",
                     "Forward Selection": "Forward", "Backward Selection": "Backward"}.get(method_raw, "All")
            
            try:
                logger.info(f"[Shiny] Fitting model with method: {method}")
                result = fit_logistic_regression(df=df, y_var=dv, x_vars=selected,
                    method=method, k=cutoff if cutoff else 2.0, verbose=True)
                model_result_rv.set(result)
                app_results['coefficients'] = result.coefficients
                app_results['predictions'] = result.predictions
                app_results['selected_vars'] = result.selected_vars
                app_results['dv'] = dv
                app_results['completed'] = True
                logger.info("[Shiny] Model fitting complete, closing session")
                await session.close()
            except Exception as e:
                logger.error(f"[Shiny] Error fitting model: {e}")
                logger.error(f"[Shiny] Traceback: {tb.format_exc()}")
        
        @output
        @render.data_frame
        def coef_table():
            result = model_result_rv.get()
            if result is None:
                return render.DataGrid(pd.DataFrame())
            display_df = result.coefficients.reset_index()
            display_df.columns = ['Variable', 'Coefficient']
            display_df['Coefficient'] = display_df['Coefficient'].round(6)
            return render.DataGrid(display_df, height="300px")
        
        @output
        @render_plotly
        def coef_plot():
            result = model_result_rv.get()
            if result is None:
                return go.Figure()
            coef_df = result.coefficients.reset_index()
            coef_df.columns = ['Variable', 'Coefficient']
            coef_df = coef_df[coef_df['Variable'] != '(Intercept)'].copy()
            if coef_df.empty:
                return go.Figure()
            coef_df['abs_coef'] = abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('abs_coef', ascending=True)
            colors = ['#38ef7d' if c > 0 else '#ff6b6b' for c in coef_df['Coefficient']]
            fig = go.Figure(data=[go.Bar(y=coef_df['Variable'], x=coef_df['Coefficient'],
                orientation='h', marker_color=colors, text=[f"{c:.3f}" for c in coef_df['Coefficient']], textposition='outside')])
            fig.update_layout(title='Coefficients', height=350, margin=dict(l=150, r=50, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e8e8e8'))
            return fig
        
        @output
        @render_plotly
        def roc_plot():
            result = model_result_rv.get()
            if result is None:
                return go.Figure()
            dv = app_results.get('dv')
            if dv is None:
                return go.Figure()
            pred_df = result.predictions.dropna(subset=['probabilities'])
            if pred_df.empty:
                return go.Figure()
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(pred_df[dv].values, pred_df['probabilities'].values)
            auc = roc_auc_score(pred_df[dv].values, pred_df['probabilities'].values)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc:.3f})', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
            fig.update_layout(title=f'ROC Curve (AUC = {auc:.3f})', height=350, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e8e8e8'), legend=dict(x=0.6, y=0.1))
            return fig
    
    app = App(app_ui, server)
    app.results = app_results
    logger.debug("Shiny app created successfully")
    return app


@debug_log_function
def find_free_port(start_port: int = 8053, max_attempts: int = 50) -> int:
    """Find an available port starting from start_port."""
    import socket
    logger.debug(f"Finding free port starting from {start_port}")
    for offset in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                logger.debug(f"Found free port: {port}")
                return port
        except OSError:
            logger.debug(f"Port {port} in use")
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        logger.debug(f"Using OS-assigned port: {port}")
        return port


@debug_log_function
def run_logistic_regression_ui(df: pd.DataFrame, port: int = None):
    """Run the Logistic Regression application and return results."""
    log_checkpoint("Starting Shiny UI application")
    if port is None:
        port = find_free_port(BASE_PORT)
    logger.info(f"Starting Shiny app on port {port} (Instance: {INSTANCE_ID})")
    print(f"Starting Shiny app on port {port} (Instance: {INSTANCE_ID})")
    sys.stdout.flush()
    app = create_logistic_regression_app(df)
    try:
        logger.debug(f"Running app on port {port}")
        app.run(port=port, launch_browser=True)
    except Exception as e:
        logger.error(f"Error running Shiny app on port {port}: {e}")
        print(f"Error running Shiny app on port {port}: {e}")
        try:
            fallback_port = find_free_port(port + 100)
            logger.info(f"Retrying on fallback port {fallback_port}")
            app.run(port=fallback_port, launch_browser=True)
        except Exception as e2:
            logger.error(f"Failed on fallback port: {e2}")
            app.results['completed'] = False
    gc.collect()
    log_checkpoint("Shiny UI application finished")
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
log_checkpoint("=" * 70)
log_checkpoint("LOGISTIC REGRESSION NODE - COMMENTATED DEBUG VERSION - STARTING")
log_checkpoint("=" * 70)

print("Logistic Regression Node - COMMENTATED DEBUG VERSION - Starting...")
print("=" * 70)

logger.info("Reading input data from KNIME")
df = knio.input_tables[0].to_pandas()
log_dataframe_info(df, "Input data")
print(f"Input data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
log_checkpoint("Checking flow variables")

contains_dv = False
contains_method = False
dv = None
target = None
sel_method = None
k = 2.0

try:
    dv = knio.flow_variables.get("DependentVariable", None)
    log_variable("DependentVariable", dv, "flow_variable")
except Exception as e:
    logger.debug(f"DependentVariable flow variable not found: {e}")

try:
    target = knio.flow_variables.get("TargetCategory", None)
    log_variable("TargetCategory", target, "flow_variable")
except Exception as e:
    logger.debug(f"TargetCategory flow variable not found: {e}")

try:
    sel_method = knio.flow_variables.get("VarSelectionMethod", None)
    log_variable("VarSelectionMethod", sel_method, "flow_variable")
except Exception as e:
    logger.debug(f"VarSelectionMethod flow variable not found: {e}")

try:
    k = knio.flow_variables.get("Cutoff", 2.0)
    if k is None:
        k = 2.0
    log_variable("Cutoff", k, "flow_variable")
except Exception as e:
    logger.debug(f"Cutoff flow variable not found: {e}")
    k = 2.0

# Validate DependentVariable
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        logger.info(f"[OK] DependentVariable: {dv}")
        print(f"[OK] DependentVariable: {dv}")
    else:
        logger.warning(f"DependentVariable '{dv}' not found in columns")

# Validate VarSelectionMethod
selection_methods = ["All", "Stepwise", "Forward", "Backward"]
if sel_method is not None and sel_method in selection_methods:
    contains_method = True
    logger.info(f"[OK] VarSelectionMethod: {sel_method}")
    print(f"[OK] VarSelectionMethod: {sel_method}")
else:
    logger.debug(f"VarSelectionMethod '{sel_method}' not valid")

log_variable("contains_dv", contains_dv)
log_variable("contains_method", contains_method)
print(f"Cutoff (k): {k}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================
log_checkpoint("Starting main processing logic")

coefficients = pd.DataFrame()
predictions = df.copy()

if contains_dv and contains_method:
    # HEADLESS MODE
    log_checkpoint("Running in HEADLESS mode")
    print("Running in HEADLESS mode")
    
    x_vars = [col for col in df.columns if col != dv]
    log_variable("initial_x_vars_count", len(x_vars))
    
    woe_vars = [col for col in x_vars if col.startswith('WOE_')]
    interaction_vars = [col for col in woe_vars if '_x_' in col]
    single_woe_vars = [col for col in woe_vars if '_x_' not in col]
    
    log_variable("woe_vars_count", len(woe_vars))
    log_variable("interaction_vars_count", len(interaction_vars))
    
    if woe_vars:
        print(f"Found {len(woe_vars)} WOE variables total:")
        print(f"  - {len(single_woe_vars)} single WOE variables")
        print(f"  - {len(interaction_vars)} interaction variables")
        if interaction_vars:
            print(f"  Interactions: {interaction_vars[:5]}{'...' if len(interaction_vars) > 5 else ''}")
        x_vars = woe_vars
    else:
        x_vars = [col for col in x_vars if not col.startswith('b_')]
        print(f"Using {len(x_vars)} predictor variables")
    
    try:
        logger.info("Fitting logistic regression model")
        result = fit_logistic_regression(df=df, y_var=dv, x_vars=x_vars, method=sel_method, k=k, verbose=True)
        coefficients = result.coefficients
        predictions = result.predictions
        
        logger.info(f"Model fitted with {len(result.selected_vars)} variables")
        print(f"\nFinal model uses {len(result.selected_vars)} variables:")
        
        selected_single = [v for v in result.selected_vars if '_x_' not in v]
        selected_interactions = [v for v in result.selected_vars if '_x_' in v]
        
        print(f"  Single WOE variables ({len(selected_single)}):")
        for var in selected_single:
            print(f"    - {var}")
        
        if selected_interactions:
            print(f"  Interaction variables ({len(selected_interactions)}):")
            for var in selected_interactions:
                print(f"    - {var}")
        else:
            print(f"  Interaction variables: None selected")
        
        if interaction_vars:
            dropped = [v for v in interaction_vars if v not in result.selected_vars]
            if dropped:
                print(f"\n  Dropped interactions ({len(dropped)}):")
                for var in dropped[:10]:
                    print(f"    - {var}")
                if len(dropped) > 10:
                    print(f"    ... and {len(dropped) - 10} more")
        
    except Exception as e:
        logger.error(f"ERROR fitting model: {e}")
        logger.error(f"Traceback: {tb.format_exc()}")
        print(f"ERROR fitting model: {e}")
        import traceback
        traceback.print_exc()
        
else:
    # INTERACTIVE MODE
    log_checkpoint("Running in INTERACTIVE mode")
    
    if SHINY_AVAILABLE:
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        logger.info("Launching Shiny UI")
        
        results = run_logistic_regression_ui(df)
        
        if results['completed']:
            coefficients = results['coefficients']
            predictions = results['predictions']
            dv = results['dv']
            logger.info("Interactive session completed successfully")
            print("Interactive session completed successfully")
        else:
            logger.warning("Interactive session cancelled")
            print("Interactive session cancelled - returning empty results")
    else:
        logger.error("Shiny not available for interactive mode")
        print("=" * 70)
        print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
        print("Please provide flow variables for headless mode:")
        print("  - DependentVariable (string): e.g., 'IsFPD'")
        print("  - VarSelectionMethod (string): 'All', 'Stepwise', 'Forward', or 'Backward'")
        print("  - Cutoff (float, optional): AIC penalty, default 2")
        print("=" * 70)

# =============================================================================
# Output Tables
# =============================================================================
log_checkpoint("Preparing output tables")

if coefficients is None:
    coefficients = pd.DataFrame()
if predictions is None:
    predictions = df.copy()

log_variable("coefficients_shape", coefficients.shape if hasattr(coefficients, 'shape') else 'N/A')
log_variable("predictions_shape", predictions.shape)

logger.debug("Writing output table 0 (predictions)")
knio.output_tables[0] = knio.Table.from_pandas(predictions)

coef_output = coefficients.copy()
if len(coef_output.columns) > 0:
    coef_output.columns = ['model$coefficients']

logger.debug("Writing output table 1 (coefficients)")
knio.output_tables[1] = knio.Table.from_pandas(coef_output)

print("=" * 70)
print("Logistic Regression completed successfully")
print("=" * 70)
print(f"Output 1 (Data with Predictions): {len(predictions)} rows")
print(f"  - Added columns: 'probabilities' (rounded to 6 decimals), 'predicted' ('1' or '0')")
print(f"Output 2 (Coefficients): {len(coefficients)} terms")
print("=" * 70)

log_checkpoint("Logistic Regression node completed")

# =============================================================================
# Cleanup for Stability
# =============================================================================
logger.debug("Starting cleanup")
sys.stdout.flush()

try:
    del df
    logger.debug("Deleted df")
except: pass

try:
    del predictions
    logger.debug("Deleted predictions")
except: pass

try:
    del coefficients
    logger.debug("Deleted coefficients")
except: pass

gc.collect()
logger.debug("Garbage collection complete")

log_checkpoint("=" * 70)
log_checkpoint("LOGISTIC REGRESSION NODE - COMMENTATED DEBUG VERSION - FINISHED")
log_checkpoint("=" * 70)

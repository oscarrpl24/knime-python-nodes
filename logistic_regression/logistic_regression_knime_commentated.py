# =============================================================================
# Logistic Regression for KNIME Python Script Node
# =============================================================================
# This is a comprehensive Python implementation that mirrors the functionality
# of R's Logistic Regression for credit risk modeling workflows.
# It includes a Shiny user interface for interactive variable selection.
# The script is designed to work within KNIME 5.9 using Python 3.9.
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
# Version: 1.0
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
# Stability Settings for Multiple Instance Execution
# =============================================================================
# When running multiple KNIME workflows simultaneously, each containing this
# logistic regression node, we need to ensure they don't conflict with each other.
# These settings help maintain isolation between instances.

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

# Set environment variable to limit numexpr to single thread - numexpr is used
# by pandas for fast numerical expression evaluation. Multiple threads can
# cause conflicts when running parallel instances
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts

# Set environment variable to limit OpenMP to single thread - OpenMP is a
# parallel programming interface used by many numerical libraries. Limiting
# it prevents resource contention between instances
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts

# =============================================================================
# Install/Import Dependencies
# =============================================================================
# This section handles the installation of required packages if they're not
# already installed, and then imports them. This ensures the script can run
# even in a fresh Python environment.

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
    
    # Attempt to import the package using Python's __import__ function
    # This is equivalent to an 'import package' statement but allows
    # the package name to be a variable
    try:
        __import__(import_name)
    except ImportError:
        # If import fails, the package isn't installed
        # Import subprocess to run external commands (pip)
        import subprocess
        # Run pip install command and wait for it to complete
        # check_call raises an exception if the command fails
        subprocess.check_call(['pip', 'install', package])

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

# Import the Logit class specifically from statsmodels
# This class implements logistic regression for binary outcomes
from statsmodels.discrete.discrete_model import Logit

# Import roc_auc_score from sklearn.metrics - this function calculates the
# Area Under the ROC Curve, a key metric for classification model performance
from sklearn.metrics import roc_auc_score

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
except ImportError:
    # If any Shiny-related import fails, print a warning and disable interactive mode
    # The script will still work in headless mode with flow variables
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Diagnostic Functions
# =============================================================================
# These functions help identify potential issues with the data before fitting
# the logistic regression model, particularly multicollinearity which can
# cause numerical instability and unreliable coefficient estimates.

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
        return results
    
    # Extract only the predictor columns and convert all to float type
    # This ensures consistent numeric type for correlation calculations
    X = df[x_vars].astype(float)
    
    # Check for low variance variables - these are variables that are nearly
    # constant across all observations. They provide little predictive value
    # and can cause numerical issues in matrix operations
    variances = X.var()  # Calculate variance for each column
    
    # Find variables with variance less than 1e-10 (essentially zero)
    # These variables have almost no variation in their values
    low_var = variances[variances < 1e-10].index.tolist()
    
    # If any low-variance variables found, record them and set the flag
    if low_var:
        results['low_variance_vars'] = low_var
        results['issues_found'] = True
    
    # Check correlation matrix for highly correlated pairs
    # High correlation between predictors indicates redundant information
    try:
        # Calculate the correlation matrix and take absolute values
        # We care about the strength of relationship, not direction
        corr_matrix = X.corr().abs()
        
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
            results['high_correlations'] = sorted(high_corr_pairs, key=lambda x: -x[2])
            results['issues_found'] = True
    except Exception:
        # If correlation calculation fails for any reason, silently continue
        # This is a diagnostic tool, so failure shouldn't stop the main process
        pass
    
    # Calculate Variance Inflation Factor (VIF) for each variable
    # VIF measures how much the variance of a regression coefficient is inflated
    # due to multicollinearity. VIF = 1 means no correlation with other predictors.
    # VIF > 10 is commonly used as a threshold indicating problematic multicollinearity.
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
                        
                        # If VIF exceeds threshold and isn't infinite, record it
                        # Infinite VIF indicates perfect multicollinearity
                        if vif > vif_threshold and not np.isinf(vif):
                            vif_data.append((var, vif))
                    except Exception:
                        # If VIF calculation fails for one variable, continue to others
                        pass
                
                # If any high-VIF variables found, sort by VIF (highest first)
                if vif_data:
                    results['high_vif_vars'] = sorted(vif_data, key=lambda x: -x[1])
                    results['issues_found'] = True
    except ImportError:
        # If variance_inflation_factor isn't available, skip VIF calculation
        pass
    except Exception:
        # For any other error, silently continue
        pass
    
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
    
    # Handle the case of no predictor variables (null/intercept-only model)
    if not x_vars:
        # For null model, we need to create a design matrix with just a constant
        # add_constant on empty DataFrame doesn't work as expected, so we create
        # a matrix of ones with the same number of rows as the data
        X = sm.add_constant(pd.DataFrame(index=df.index, columns=[]))
        X = np.ones((len(df), 1))  # Override with explicit ones column
    else:
        # For models with predictors, add a constant column (intercept) and
        # convert all predictor columns to float type for consistent computation
        X = sm.add_constant(df[x_vars].astype(float))
    
    # Extract the target variable and convert to float for modeling
    y = df[y_var].astype(float)
    
    # Flag to track if we encountered numerical issues during fitting
    had_numerical_issues = False
    
    try:
        # Use context manager to capture warnings during model fitting
        # This allows us to detect Hessian inversion warnings without
        # printing them to the console
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Set filter to capture all warnings (not just those matching a filter)
            warnings.simplefilter("always")
            
            # Try BFGS optimization first - it's generally fast and reliable
            # BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method
            # that approximates the Hessian matrix, making it efficient
            try:
                # disp=0 suppresses the fit summary printout
                # maxiter=1000 allows enough iterations for convergence
                model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
            except Exception:
                # If BFGS fails, it often indicates numerical issues
                had_numerical_issues = True
                
                # Try L1 regularized fitting (Lasso-type regularization)
                # Regularization can help with multicollinearity by shrinking coefficients
                try:
                    # alpha=0.01 is a small regularization penalty
                    model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
                except Exception:
                    # Last resort: try basic Newton-Raphson method
                    # This is slower and can have convergence issues, but may work
                    # when other methods fail
                    model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
            
            # Check if any captured warnings indicate Hessian issues
            # Hessian inversion problems suggest the model matrix is near-singular
            for w in caught_warnings:
                if 'Hessian' in str(w.message) or 'cov_params' in str(w.message):
                    had_numerical_issues = True
                    break
        
        # Track which variables caused numerical issues for reporting
        # The assumption is that the last variable added is often the culprit
        if track_issues and had_numerical_issues and x_vars:
            # Add the last variable to the set of problematic variables
            _stepwise_numerical_issues.add(x_vars[-1])
        
        # Get the AIC (Akaike Information Criterion) from the fitted model
        # Lower AIC indicates better model (balancing fit and complexity)
        aic = model.aic
        return model, aic
    except Exception as e:
        # If all fitting attempts fail, track the issue and return failure values
        if track_issues and x_vars:
            _stepwise_numerical_issues.add(x_vars[-1])
        # Return None for model and infinity for AIC (worst possible value)
        return None, float('inf')


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
        y_var: Name of the dependent variable column
        candidate_vars: List of candidate predictor variables to consider
        k: Penalty multiplier for AIC calculation
           - k=2 gives standard AIC
           - k=log(n) gives BIC (more conservative selection)
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
        
    Algorithm:
    1. Start with null model (intercept only)
    2. For each remaining candidate variable:
       a. Try adding it to the current model
       b. Calculate the new AIC
    3. If any addition improves AIC, add the best variable
    4. Repeat until no improvement is possible
    """
    # Initialize lists for tracking the selection process
    selected = []  # Variables currently in the model
    remaining = list(candidate_vars)  # Variables not yet added
    aic_history = []  # AIC values at each step
    steps = []  # Human-readable step descriptions
    
    # Fit the null model (no predictors, just intercept) to get starting AIC
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    # Print initial state if verbose
    if verbose:
        print(f"Forward Selection: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    improved = True
    while improved and remaining:
        improved = False  # Reset flag - will be set True if we find an improvement
        best_var = None   # Track the best variable to add
        best_aic = current_aic  # Track the best AIC found
        
        # Try adding each remaining variable to the model
        for var in remaining:
            # Create a test model with current variables plus this candidate
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            
            # Adjust AIC with the penalty multiplier
            # Standard AIC uses k=2, so (k-2) is 0 for standard AIC
            # For BIC or other penalties, this adjusts the selection criterion
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            # Check if this is the best improvement so far
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                improved = True
        
        # If we found an improvement, add the best variable
        if improved and best_var:
            selected.append(best_var)  # Add to selected set
            remaining.remove(best_var)  # Remove from candidates
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"+ {best_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  + {best_var}: AIC = {current_aic:.4f}")
    
    # Print final summary if verbose
    if verbose:
        print(f"Forward Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    # Return the results packaged in a StepwiseResult object
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


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
           - k=2 gives standard AIC
           - k=log(n) gives BIC (more aggressive elimination)
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
        
    Algorithm:
    1. Start with full model (all candidate predictors)
    2. For each variable in the current model:
       a. Try removing it from the model
       b. Calculate the new AIC
    3. If any removal improves AIC, remove the worst variable
    4. Repeat until no improvement is possible or no variables remain
    """
    # Start with all variables and create a copy to modify
    selected = list(current_vars)
    aic_history = []  # AIC values at each step
    steps = []  # Human-readable step descriptions
    
    # Fit the full model to get starting AIC
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    # Print initial state if verbose
    if verbose:
        print(f"Backward Elimination: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    improved = True
    while improved and len(selected) > 0:
        improved = False  # Reset flag
        worst_var = None  # Track the variable to remove
        best_aic = current_aic  # Track the best AIC after removal
        
        # Try removing each variable from the model
        for var in selected:
            # Create a test model without this variable
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            
            # Adjust AIC with the penalty multiplier
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            # Check if removing this variable improves AIC
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                worst_var = var
                improved = True
        
        # If we found an improvement, remove the worst variable
        if improved and worst_var:
            selected.remove(worst_var)  # Remove from selected set
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"- {worst_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  - {worst_var}: AIC = {current_aic:.4f}")
    
    # Print final summary if verbose
    if verbose:
        print(f"Backward Elimination: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    # Return the results packaged in a StepwiseResult object
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


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
    
    This approach is more flexible than pure forward or backward selection
    and can escape local optima by reconsidering previously made decisions.
    
    Parameters:
        df: DataFrame with all the data (predictors and target)
        y_var: Name of the dependent variable column
        current_vars: List of predictor variables to start with (typically all)
        k: Penalty multiplier for AIC calculation
           - k=2 gives standard AIC
           - k=log(n) gives BIC
        verbose: If True, print progress to console
        
    Returns:
        StepwiseResult containing selected variables, AIC history, and step descriptions
        
    Algorithm:
    1. Start with the specified set of variables
    2. At each step, consider all possible single-variable changes:
       a. Remove any variable currently in the model
       b. Add any variable previously removed
    3. Make the change that most improves AIC (if any)
    4. Repeat until no improvement is possible or max iterations reached
    """
    # Initialize with current variables
    selected = list(current_vars)
    remaining = []  # Variables removed from model (empty when starting with all)
    aic_history = []  # AIC values at each step
    steps = []  # Human-readable step descriptions
    
    # Track all variables ever considered (for potential re-addition)
    all_vars = list(current_vars)
    
    # Fit current model to get starting AIC
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    # Print initial state if verbose
    if verbose:
        print(f"Stepwise Selection: Start AIC = {current_aic:.4f}")
    
    # Continue until no improvement is found
    improved = True
    iteration = 0
    # Set a maximum number of iterations to prevent infinite loops
    # In theory, stepwise should converge, but this is a safety measure
    max_iterations = len(all_vars) * 2
    
    while improved and iteration < max_iterations:
        iteration += 1
        improved = False  # Reset flag
        best_action = None  # Will be 'add' or 'remove'
        best_var = None  # Variable to add or remove
        best_aic = current_aic  # Best AIC found
        
        # Try removing each variable currently in the model (backward step)
        for var in selected:
            # Create test model without this variable
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            # Check if this is the best change so far
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                best_action = 'remove'
                improved = True
        
        # Try adding each removed variable back (forward step)
        for var in remaining:
            # Create test model with this variable added back
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            # Check if this is better than current best (including removals)
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                best_action = 'add'
                improved = True
        
        # If we found an improvement, apply the best action
        if improved and best_var:
            if best_action == 'remove':
                # Remove the variable from selected, add to remaining
                selected.remove(best_var)
                remaining.append(best_var)
                steps.append(f"- {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  - {best_var}: AIC = {best_aic:.4f}")
            else:
                # Add the variable back from remaining to selected
                selected.append(best_var)
                remaining.remove(best_var)
                steps.append(f"+ {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  + {best_var}: AIC = {best_aic:.4f}")
            
            # Update current AIC and record in history
            current_aic = best_aic
            aic_history.append(current_aic)
    
    # Print final summary if verbose
    if verbose:
        print(f"Stepwise Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    # Return the results packaged in a StepwiseResult object
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


# =============================================================================
# Model Fitting and Evaluation
# =============================================================================
# This section contains the main function for fitting logistic regression models
# with optional variable selection. It brings together the stepwise functions
# and model diagnostics into a single high-level interface.

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
               This should be binary (0/1) for logistic regression
        x_vars: List of predictor variable names to consider
               These are the candidate variables for the model
        method: Variable selection method - one of:
               - "All": Use all provided variables (no selection)
               - "Stepwise": Bidirectional stepwise selection
               - "Forward": Forward stepwise selection (start empty)
               - "Backward": Backward elimination (start with all)
        k: AIC penalty multiplier
           - k=2 gives standard AIC (default)
           - k=log(n) gives BIC for more parsimonious models
        verbose: If True, print progress and diagnostics to console
        
    Returns:
        ModelResult containing the fitted model, coefficients, predictions,
        and list of variables in the final model
        
    Raises:
        ValueError: If no complete cases remain after removing missing values
        ValueError: If no variables are selected (model cannot be fit)
    """
    # Create a list of all columns we'll use (target + predictors)
    # This is used for subsetting and missing value handling
    cols_to_use = [y_var] + x_vars
    
    # Remove rows with any missing values in the columns we're using
    # Logistic regression cannot handle missing values directly
    df_clean = df[cols_to_use].dropna()
    
    # Check that we have data remaining after removing missing values
    if len(df_clean) == 0:
        raise ValueError("No complete cases after removing missing values")
    
    # Print initial information about the model fitting
    if verbose:
        print(f"Fitting logistic regression: {len(df_clean)} observations, {len(x_vars)} variables")
        print(f"Method: {method}, k = {k}")
    
    # Run multicollinearity diagnostics before fitting
    # This warns users about potential issues that might cause model problems
    if verbose:
        diagnostics = check_multicollinearity(df_clean, x_vars, threshold=0.85, vif_threshold=10.0, verbose=True)
    
    # Reset the global tracker for numerical issues before running stepwise
    # This ensures we only track issues from this model fitting session
    global _stepwise_numerical_issues
    _stepwise_numerical_issues = set()
    
    # Apply the selected variable selection method
    if method == "All":
        # No selection - use all provided variables
        selected_vars = x_vars
        stepwise_result = None
    elif method == "Forward":
        # Forward selection starting from empty model
        stepwise_result = stepwise_forward(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    elif method == "Backward":
        # Backward elimination starting from full model
        stepwise_result = stepwise_backward(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    elif method == "Stepwise":
        # Bidirectional stepwise selection
        stepwise_result = stepwise_both(df_clean, y_var, x_vars, k=k, verbose=verbose)
        selected_vars = stepwise_result.selected_vars
    else:
        # Unknown method - default to using all variables
        selected_vars = x_vars
        stepwise_result = None
    
    # Verify that we have at least one variable selected
    if not selected_vars:
        raise ValueError("No variables selected - model cannot be fit")
    
    # Report any variables that caused numerical issues during stepwise selection
    # This helps users understand why certain variables might be problematic
    if verbose and _stepwise_numerical_issues and method != "All":
        print("\n" + "-" * 70)
        print("⚠️  NUMERICAL ISSUES DURING STEPWISE SELECTION")
        print("-" * 70)
        print("The following variables caused Hessian inversion warnings or fit failures:")
        # Show each problematic variable with whether it ended up in the model
        for var in sorted(_stepwise_numerical_issues):
            status = "✓ selected" if var in selected_vars else "✗ not selected"
            print(f"  - {var} ({status})")
        print("\nThis typically indicates multicollinearity or separation issues.")
        print("Consider reviewing correlated variables in earlier pipeline steps.")
        print("-" * 70 + "\n")
    
    # Fit the final model with selected variables
    # Add constant column for the intercept term
    X = sm.add_constant(df_clean[selected_vars].astype(float))
    # Convert target to float for modeling
    y = df_clean[y_var].astype(float)
    
    # Suppress warnings during final model fitting since we've already
    # run diagnostics and user is aware of potential issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Try BFGS optimization first - generally fastest and most reliable
        try:
            model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
        except Exception:
            # If BFGS fails, inform user and try Newton method
            if verbose:
                print("  Note: BFGS optimization had issues, trying Newton method...")
            try:
                model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
            except Exception:
                # If Newton also fails, use L1 regularization as last resort
                if verbose:
                    print("  Note: Newton method had issues, using L1 regularization...")
                model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
    
    # Print the full model summary if verbose
    if verbose:
        print("\n" + "="*70)
        print("MODEL SUMMARY")
        print("="*70)
        # statsmodels summary() provides coefficients, std errors, p-values, etc.
        print(model.summary())
    
    # Create coefficient table matching R format: model$coefficients
    # Row names are variable names, single column with coefficient values
    # model.params is a pandas Series with variable names as index
    coef_df = model.params.to_frame(name='coefficients')
    coef_df.index.name = None  # Remove index name to match R's format
    
    # Rename 'const' to '(Intercept)' for compatibility with R conventions
    # R uses '(Intercept)' as the name for the intercept term
    if 'const' in coef_df.index:
        coef_df = coef_df.rename(index={'const': '(Intercept)'})
    
    # Calculate predictions for the entire original dataframe
    # This matches R's: predict(model, df, type = "response")
    predictions = df.copy()  # Start with a copy of the original data
    
    # For prediction, we need to add a constant column for the intercept
    # has_constant='add' handles cases where constant might already exist
    X_full = sm.add_constant(df[selected_vars].astype(float), has_constant='add')
    
    # Identify rows with complete data (no missing values)
    # We can only make predictions for rows with all required values
    complete_mask = X_full.notna().all(axis=1)
    
    # Initialize prediction columns with missing values
    # These will be filled in for rows with complete data
    predictions['probabilities'] = np.nan
    predictions['predicted'] = None
    
    # Make predictions for rows with complete data
    if complete_mask.any():
        # Extract complete cases
        X_complete = X_full[complete_mask]
        
        # Get predicted probabilities from the model
        # predict() returns P(Y=1|X) for logistic regression
        proba = model.predict(X_complete)
        
        # Store probabilities rounded to 6 decimal places (matches R behavior)
        predictions.loc[complete_mask, 'probabilities'] = np.round(proba.values, 6)
        
        # Create binary predictions using 0.5 as threshold
        # Values stored as strings "1" or "0" to match R behavior
        predictions.loc[complete_mask, 'predicted'] = np.where(proba.values > 0.5, "1", "0")
    
    # Calculate and print model performance metrics
    if verbose:
        try:
            # Get actual target values and predicted probabilities for complete cases
            y_actual = df_clean[y_var].astype(float)
            y_proba = model.predict(X)
            
            # Calculate AUC (Area Under the ROC Curve)
            # AUC of 0.5 = random guessing, 1.0 = perfect prediction
            auc = roc_auc_score(y_actual, y_proba)
            print(f"\nModel AUC: {auc:.4f}")
            
            # Calculate Gini coefficient (another common credit risk metric)
            # Gini = 2 * AUC - 1, ranges from 0 (random) to 1 (perfect)
            print(f"Gini: {2*auc - 1:.4f}")
        except Exception as e:
            # If AUC calculation fails (e.g., all predictions same class), report error
            print(f"Could not calculate AUC: {e}")
    
    # Return all results packaged in a ModelResult object
    return ModelResult(
        model=model,
        coefficients=coef_df,
        predictions=predictions,
        selected_vars=selected_vars
    )


# =============================================================================
# Shiny UI Application
# =============================================================================
# This section defines the interactive user interface for the logistic regression
# node. When run without flow variables, this UI allows users to:
# - Select the dependent variable
# - Choose which predictor variables to include
# - Select a variable selection method
# - View model results (coefficients, ROC curve)
# - Run the model and export results

def create_logistic_regression_app(df: pd.DataFrame):
    """
    Create the Logistic Regression Shiny application.
    
    This function builds the complete Shiny application with its user interface
    and server logic. The app provides an interactive way to configure and run
    logistic regression models.
    
    Parameters:
        df: The input DataFrame containing all variables to choose from
        
    Returns:
        A Shiny App object that can be run with app.run()
        The app has an additional 'results' attribute containing the final
        model outputs after the user clicks "Run Model & Close"
    """
    
    # Dictionary to store results that will be returned after the app closes
    # This allows communication between the Shiny session and the main script
    app_results = {
        'coefficients': None,    # Will hold the coefficient DataFrame
        'predictions': None,     # Will hold predictions DataFrame
        'selected_vars': None,   # Will hold list of selected variables
        'dv': None,              # Will hold the dependent variable name
        'completed': False       # Flag indicating if user completed the workflow
    }
    
    # Define the user interface layout using Shiny's page_fluid (responsive layout)
    app_ui = ui.page_fluid(
        # Add custom CSS styles in the page header
        ui.tags.head(
            ui.tags.style("""
                /* Import Google Fonts - Raleway for a modern, clean look */
                @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&display=swap');
                
                /* Body styling - dark gradient background for modern appearance */
                body { 
                    font-family: 'Raleway', sans-serif; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    color: #e8e8e8;
                }
                
                /* Card styling - glassmorphism effect with blur and transparency */
                .card { 
                    background: rgba(255, 255, 255, 0.08);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px; 
                    padding: 24px; 
                    margin: 12px 0; 
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }
                
                /* Card header with gradient background */
                .card-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px 12px 0 0;
                    margin: -24px -24px 20px -24px;
                    padding: 16px 24px;
                    color: white;
                    font-weight: 600;
                }
                
                /* Main title styling */
                h4 { 
                    font-weight: 700; 
                    text-align: center; 
                    margin: 20px 0; 
                    color: #fff;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }
                
                /* Section header styling */
                h5 {
                    color: #a8dadc;
                    font-weight: 600;
                    margin-bottom: 16px;
                }
                
                /* Button styling with rounded corners and gradient */
                .btn { 
                    font-weight: 600; 
                    border-radius: 50px; 
                    padding: 10px 24px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    font-size: 13px;
                    transition: all 0.3s ease;
                    border: none;
                }
                
                /* Button hover effect - slight lift and shadow */
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                }
                
                /* Primary button gradient (purple) */
                .btn-primary { 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                }
                
                /* Success button gradient (green) */
                .btn-success { 
                    background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                }
                
                /* Info button gradient (blue) */
                .btn-info { 
                    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%); 
                }
                
                /* Spacer class for inline spacing */
                .divider { width: 12px; display: inline-block; }
                
                /* Form input styling - dark theme with transparency */
                .form-control, .form-select {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    color: #fff;
                    border-radius: 10px;
                }
                
                /* Form input focus state */
                .form-control:focus, .form-select:focus {
                    background: rgba(255, 255, 255, 0.15);
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
                    color: #fff;
                }
                
                /* Dropdown option styling */
                .form-select option {
                    background: #1a1a2e;
                    color: #fff;
                }
                
                /* Form label styling */
                .form-label {
                    color: #a8dadc;
                    font-weight: 500;
                }
                
                /* Checkbox styling when checked */
                .form-check-input:checked {
                    background-color: #667eea;
                    border-color: #667eea;
                }
                
                /* Table text color override */
                table {
                    color: #e8e8e8 !important;
                }
                
                /* DataFrame container styling */
                .dataframe {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                }
                
                /* Scrollable container for variable checkboxes */
                .var-checkbox-container {
                    max-height: 400px;
                    overflow-y: auto;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 10px;
                    padding: 16px;
                }
                
                /* Custom scrollbar for checkbox container - track */
                .var-checkbox-container::-webkit-scrollbar {
                    width: 8px;
                }
                .var-checkbox-container::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                }
                
                /* Custom scrollbar - thumb */
                .var-checkbox-container::-webkit-scrollbar-thumb {
                    background: #667eea;
                    border-radius: 4px;
                }
            """)
        ),
        
        # Page title with microscope emoji
        ui.h4("🔬 Logistic Regression"),
        
        # Configuration Card - contains model setup options
        ui.div(
            {"class": "card"},  # Apply card styling
            ui.div({"class": "card-header"}, "Model Configuration"),  # Card title
            ui.row(
                # Dependent variable dropdown - user selects target variable
                ui.column(4,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=list(df.columns),  # All columns as options
                                   selected=df.columns[-1] if len(df.columns) > 0 else None)
                ),
                # Target category dropdown - which value is "bad" outcome
                ui.column(4,
                    ui.input_select("tc", "Target Category", choices=[])  # Populated dynamically
                ),
                # Variable selection method dropdown
                ui.column(4,
                    ui.input_select("method", "Variable Selection Method",
                                   choices=["Must include all", "Stepwise Selection", 
                                           "Forward Selection", "Backward Selection"],
                                   selected="Must include all")
                )
            ),
            ui.row(
                # AIC penalty input - controls stepwise selection stringency
                ui.column(4,
                    ui.input_numeric("cutoff", "AIC Penalty (k)", value=2, min=0, step=0.5),
                    ui.tags.small("k=2 for AIC, k=log(n) for BIC", style="color: #888;")
                ),
                # Button to auto-select WOE variables
                ui.column(4,
                    ui.input_action_button("select_woe", "Select WOE Variables", class_="btn btn-info"),
                    ui.tags.small("Auto-select WOE_ prefixed vars", style="color: #888;")
                ),
                # Button to select all variables
                ui.column(4,
                    ui.input_action_button("select_all", "Select All Variables", class_="btn btn-primary"),
                )
            )
        ),
        
        # Variable Selection Card - contains checkboxes for each variable
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Variable Selection"),
            ui.row(
                ui.column(12,
                    # Scrollable container for variable checkboxes
                    ui.div(
                        {"class": "var-checkbox-container"},
                        ui.output_ui("var_checkboxes")  # Dynamic output - generated by server
                    )
                )
            )
        ),
        
        # Results Card - displays coefficient table
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Results"),
            ui.row(
                ui.column(12,
                    ui.h5("Coefficients"),
                    ui.output_data_frame("coef_table")  # Dynamic DataGrid output
                )
            )
        ),
        
        # Charts Card - side-by-side coefficient and ROC plots
        ui.row(
            # Coefficient bar chart
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    ui.h5("Coefficient Plot"),
                    output_widget("coef_plot")  # Plotly widget output
                )
            ),
            # ROC curve chart
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    ui.h5("ROC Curve"),
                    output_widget("roc_plot")  # Plotly widget output
                )
            )
        ),
        
        # Action Button - runs model and closes the UI
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "🚀 Run Model & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    # Define the server function - contains all reactive logic
    def server(input: Inputs, output: Outputs, session: Session):
        # Reactive value to store model results - updates UI when changed
        model_result_rv = reactive.Value(None)
        # Reactive value for selected variables
        selected_vars_rv = reactive.Value([])
        
        # Effect that runs when dependent variable changes
        # Updates the target category dropdown with unique values
        @reactive.Effect
        @reactive.event(input.dv)  # Trigger when dv input changes
        def update_tc():
            # Get the selected dependent variable
            dv = input.dv()
            if dv and dv in df.columns:
                # Get unique values from the DV column
                unique_vals = df[dv].dropna().unique().tolist()
                # Sort and convert to strings for display
                unique_vals = [str(v) for v in sorted(unique_vals)]
                # Update the target category dropdown
                ui.update_select("tc", choices=unique_vals, 
                               selected=unique_vals[-1] if unique_vals else None)
        
        # Output renderer for variable checkboxes
        # Creates a checkbox for each column (except DV)
        @output
        @render.ui
        def var_checkboxes():
            dv = input.dv()  # Get current DV selection
            if not dv:
                return ui.p("Select a dependent variable first")
            
            # Get all columns except the dependent variable
            available_vars = [col for col in df.columns if col != dv]
            
            # Create checkbox inputs for each variable
            checkboxes = []
            for var in available_vars:
                # Pre-select variables that start with 'WOE_' (Weight of Evidence)
                is_woe = var.startswith('WOE_')
                # Create checkbox with unique ID based on variable name
                checkbox = ui.input_checkbox(f"var_{var}", var, value=is_woe)
                checkboxes.append(checkbox)
            
            # Return all checkboxes wrapped in a div
            return ui.div(*checkboxes)
        
        # Effect that runs when "Select WOE Variables" button is clicked
        @reactive.Effect
        @reactive.event(input.select_woe)
        def select_woe_vars():
            dv = input.dv()
            if not dv:
                return
            
            # Get all columns except DV
            available_vars = [col for col in df.columns if col != dv]
            # Set each checkbox based on whether variable starts with WOE_
            for var in available_vars:
                is_woe = var.startswith('WOE_')
                try:
                    ui.update_checkbox(f"var_{var}", value=is_woe)
                except:
                    pass  # Ignore errors if checkbox doesn't exist
        
        # Effect that runs when "Select All Variables" button is clicked
        @reactive.Effect
        @reactive.event(input.select_all)
        def select_all_vars():
            dv = input.dv()
            if not dv:
                return
            
            # Get all columns except DV
            available_vars = [col for col in df.columns if col != dv]
            for var in available_vars:
                # Select everything except b_ prefixed variables (binned)
                # b_ variables are binned versions, not for regression
                should_select = not var.startswith('b_')
                try:
                    ui.update_checkbox(f"var_{var}", value=should_select)
                except:
                    pass
        
        # Reactive calculation that returns list of currently selected variables
        @reactive.Calc
        def get_selected_vars():
            dv = input.dv()
            if not dv:
                return []
            
            available_vars = [col for col in df.columns if col != dv]
            selected = []
            # Check each variable's checkbox state
            for var in available_vars:
                try:
                    # Access the checkbox input dynamically by name
                    if input[f"var_{var}"]():
                        selected.append(var)
                except:
                    pass
            return selected
        
        # Effect that runs when "Run Model & Close" button is clicked
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_model():
            # Get all user inputs
            dv = input.dv()
            selected = get_selected_vars()
            method_raw = input.method()
            cutoff = input.cutoff()
            
            # Validate inputs
            if not dv or not selected:
                return
            
            # Map UI method names to internal method names
            method_map = {
                "Must include all": "All",
                "Stepwise Selection": "Stepwise",
                "Forward Selection": "Forward",
                "Backward Selection": "Backward"
            }
            method = method_map.get(method_raw, "All")
            
            try:
                # Fit the logistic regression model
                result = fit_logistic_regression(
                    df=df,
                    y_var=dv,
                    x_vars=selected,
                    method=method,
                    k=cutoff if cutoff else 2.0,
                    verbose=True
                )
                
                # Store result in reactive value (updates UI)
                model_result_rv.set(result)
                
                # Store results for output to main script
                app_results['coefficients'] = result.coefficients
                app_results['predictions'] = result.predictions
                app_results['selected_vars'] = result.selected_vars
                app_results['dv'] = dv
                app_results['completed'] = True
                
                # Close the Shiny session, ending the app
                await session.close()
                
            except Exception as e:
                # Print error for debugging
                print(f"Error fitting model: {e}")
                import traceback
                traceback.print_exc()
        
        # Output renderer for coefficient table
        @output
        @render.data_frame
        def coef_table():
            result = model_result_rv.get()  # Get current model result
            if result is None:
                # Return empty DataGrid if no model yet
                return render.DataGrid(pd.DataFrame())
            
            # Format coefficients for display
            display_df = result.coefficients.reset_index()
            display_df.columns = ['Variable', 'Coefficient']
            display_df['Coefficient'] = display_df['Coefficient'].round(6)
            
            # Return as DataGrid with fixed height
            return render.DataGrid(display_df, height="300px")
        
        # Output renderer for coefficient bar chart
        @output
        @render_plotly
        def coef_plot():
            result = model_result_rv.get()
            if result is None:
                return go.Figure()  # Return empty figure
            
            # Prepare coefficient data
            coef_df = result.coefficients.reset_index()
            coef_df.columns = ['Variable', 'Coefficient']
            
            # Exclude intercept from visualization
            coef_df = coef_df[coef_df['Variable'] != '(Intercept)'].copy()
            if coef_df.empty:
                return go.Figure()
            
            # Sort by absolute coefficient value for better visualization
            coef_df['abs_coef'] = abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('abs_coef', ascending=True)
            
            # Color bars based on sign (green=positive, red=negative)
            colors = ['#38ef7d' if c > 0 else '#ff6b6b' for c in coef_df['Coefficient']]
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=coef_df['Variable'],  # Variables on y-axis
                    x=coef_df['Coefficient'],  # Values on x-axis
                    orientation='h',  # Horizontal orientation
                    marker_color=colors,
                    text=[f"{c:.3f}" for c in coef_df['Coefficient']],  # Value labels
                    textposition='outside'
                )
            ])
            
            # Update layout for dark theme
            fig.update_layout(
                title='Coefficients (excluding intercept)',
                xaxis_title='Coefficient',
                yaxis_title='Variable',
                height=350,
                margin=dict(l=150, r=50, t=50, b=50),  # Left margin for variable names
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8')  # Light text
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            return fig
        
        # Output renderer for ROC curve
        @output
        @render_plotly
        def roc_plot():
            result = model_result_rv.get()
            if result is None:
                return go.Figure()
            
            dv = app_results.get('dv')
            if dv is None:
                return go.Figure()
            
            # Get predictions with complete data
            pred_df = result.predictions.dropna(subset=['probabilities'])
            if pred_df.empty:
                return go.Figure()
            
            # Extract actual and predicted values
            y_true = pred_df[dv].values
            y_score = pred_df['probabilities'].values
            
            # Calculate ROC curve points
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            
            # Calculate AUC for legend
            auc = roc_auc_score(y_true, y_score)
            
            # Create figure
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc:.3f})',
                line=dict(color='#667eea', width=3)
            ))
            
            # Add diagonal reference line (random classifier baseline)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash')
            ))
            
            # Update layout for dark theme
            fig.update_layout(
                title=f'ROC Curve (AUC = {auc:.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8'),
                legend=dict(x=0.6, y=0.1)  # Position legend in lower right
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
            
            return fig
    
    # Create the Shiny App by combining UI and server
    app = App(app_ui, server)
    # Attach results dictionary to app for access after it closes
    app.results = app_results
    return app


def find_free_port(start_port: int = 8053, max_attempts: int = 50) -> int:
    """
    Find an available port starting from start_port.
    
    This function attempts to bind to ports starting from the given start_port.
    If a port is in use, it tries random ports within the RANDOM_PORT_RANGE.
    This is necessary when running multiple instances of the node simultaneously.
    
    Parameters:
        start_port: The first port to try (default 8053)
        max_attempts: Maximum number of ports to try before falling back
        
    Returns:
        An available port number that can be used for the Shiny app
    """
    # Import socket for network operations
    import socket
    
    # Try random ports within the defined range
    for offset in range(max_attempts):
        # Pick a random port offset from start_port
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            # Try to bind to the port - if successful, it's available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port  # Port is available, return it
        except OSError:
            continue  # Port in use, try another
    
    # Fallback: let the OS assign an ephemeral port
    # This should always work but gives us less control
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))  # 0 means let OS choose
        return s.getsockname()[1]  # Get the assigned port


def run_logistic_regression_ui(df: pd.DataFrame, port: int = None):
    """
    Run the Logistic Regression application and return results.
    
    This function launches the Shiny web application for interactive model
    configuration. The app runs until the user clicks "Run Model & Close",
    at which point the results are returned.
    
    Parameters:
        df: The input DataFrame to use for model fitting
        port: Optional specific port to use (if None, finds a free port)
        
    Returns:
        Dictionary containing model results:
        - 'coefficients': coefficient DataFrame
        - 'predictions': predictions DataFrame
        - 'selected_vars': list of selected variables
        - 'dv': dependent variable name
        - 'completed': True if user completed, False if cancelled/error
    """
    # Find an available port if none specified
    if port is None:
        port = find_free_port(BASE_PORT)
    
    # Print startup info to console
    print(f"Starting Shiny app on port {port} (Instance: {INSTANCE_ID})")
    sys.stdout.flush()  # Ensure message is displayed immediately
    
    # Create the Shiny application
    app = create_logistic_regression_app(df)
    
    try:
        # Run the app - this blocks until app closes
        # launch_browser=True opens the user's default browser to the app
        app.run(port=port, launch_browser=True)
    except Exception as e:
        # If running fails (e.g., port became unavailable), try fallback
        print(f"Error running Shiny app on port {port}: {e}")
        sys.stdout.flush()
        try:
            # Try a port 100 higher
            fallback_port = find_free_port(port + 100)
            print(f"Retrying on port {fallback_port}")
            sys.stdout.flush()
            app.run(port=fallback_port, launch_browser=True)
        except Exception as e2:
            # If fallback also fails, mark as not completed
            print(f"Failed on fallback port: {e2}")
            app.results['completed'] = False
    
    # Clean up resources
    gc.collect()  # Force garbage collection
    sys.stdout.flush()  # Flush output buffer
    
    # Return the results from the app
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
# This section is the main entry point when the script runs in KNIME.
# It reads the input data table and prepares for processing.

# Print startup banner
print("Logistic Regression Node - Starting...")
print("=" * 70)

# Read the input data from KNIME's first input port (index 0)
# knio.input_tables[0] returns a KNIME Table object
# .to_pandas() converts it to a pandas DataFrame for processing
df = knio.input_tables[0].to_pandas()

# Print data dimensions for logging
print(f"Input data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
# Flow variables allow automated/batch processing without the UI.
# If required flow variables are provided, the script runs in headless mode.
# Otherwise, it launches the interactive Shiny UI.

# Initialize flags and values for flow variable checking
contains_dv = False      # Flag: is DependentVariable valid?
contains_method = False  # Flag: is VarSelectionMethod valid?
dv = None               # Dependent variable name
target = None           # Target category value
sel_method = None       # Selection method string
k = 2.0                 # AIC penalty (default to standard AIC)

# Try to read DependentVariable flow variable
# Using try/except because flow variable might not exist
try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass  # Variable doesn't exist, dv remains None

# Try to read TargetCategory flow variable
try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

# Try to read VarSelectionMethod flow variable
try:
    sel_method = knio.flow_variables.get("VarSelectionMethod", None)
except:
    pass

# Try to read Cutoff flow variable with default value
try:
    k = knio.flow_variables.get("Cutoff", 2.0)
    # Handle case where Cutoff is None
    if k is None:
        k = 2.0
except:
    k = 2.0  # Use default if reading fails

# Validate DependentVariable - must be a non-empty string that exists in data
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True  # Valid DV found
        print(f"[OK] DependentVariable: {dv}")

# Validate VarSelectionMethod - must be one of the allowed values
selection_methods = ["All", "Stepwise", "Forward", "Backward"]
if sel_method is not None and sel_method in selection_methods:
    contains_method = True  # Valid method found
    print(f"[OK] VarSelectionMethod: {sel_method}")

# Print the cutoff value being used
print(f"Cutoff (k): {k}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================
# Based on whether flow variables are provided, either run in headless mode
# (automated) or interactive mode (Shiny UI).

# Initialize output variables with defaults
coefficients = pd.DataFrame()  # Empty DataFrame for coefficients
predictions = df.copy()        # Copy of input for predictions

# Check if we have all required flow variables for headless mode
if contains_dv and contains_method:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    # Flow variables provided - run automatically without UI
    print("Running in HEADLESS mode")
    
    # Get predictor variables (all columns except the dependent variable)
    x_vars = [col for col in df.columns if col != dv]
    
    # Check if WOE-transformed variables are present (preferred for credit risk)
    # WOE variables start with 'WOE_' prefix
    woe_vars = [col for col in x_vars if col.startswith('WOE_')]
    
    # Identify interaction columns (contain '_x_' pattern)
    # These are products of two WOE variables for capturing interactions
    interaction_vars = [col for col in woe_vars if '_x_' in col]
    single_woe_vars = [col for col in woe_vars if '_x_' not in col]
    
    # If WOE variables found, use those; otherwise use all non-binned variables
    if woe_vars:
        print(f"Found {len(woe_vars)} WOE variables total:")
        print(f"  - {len(single_woe_vars)} single WOE variables")
        print(f"  - {len(interaction_vars)} interaction variables")
        # Show sample of interaction variables if present
        if interaction_vars:
            print(f"  Interactions: {interaction_vars[:5]}{'...' if len(interaction_vars) > 5 else ''}")
        x_vars = woe_vars
    else:
        # No WOE variables - exclude b_ prefixed (binned) variables
        # Binned variables are for display, not regression
        x_vars = [col for col in x_vars if not col.startswith('b_')]
        print(f"Using {len(x_vars)} predictor variables")
    
    # Attempt to fit the model
    try:
        # Call the main fitting function with flow variable parameters
        result = fit_logistic_regression(
            df=df,
            y_var=dv,
            x_vars=x_vars,
            method=sel_method,
            k=k,
            verbose=True
        )
        
        # Extract outputs from result
        coefficients = result.coefficients
        predictions = result.predictions
        
        # Print summary of final model
        print(f"\nFinal model uses {len(result.selected_vars)} variables:")
        
        # Separate selected variables into single and interaction for clarity
        selected_single = [v for v in result.selected_vars if '_x_' not in v]
        selected_interactions = [v for v in result.selected_vars if '_x_' in v]
        
        # Print single WOE variables
        print(f"  Single WOE variables ({len(selected_single)}):")
        for var in selected_single:
            print(f"    - {var}")
        
        # Print interaction variables
        if selected_interactions:
            print(f"  Interaction variables ({len(selected_interactions)}):")
            for var in selected_interactions:
                print(f"    - {var}")
        else:
            print(f"  Interaction variables: None selected (may have been dropped by stepwise)")
        
        # Show what interaction terms were dropped during selection
        if interaction_vars:
            dropped_interactions = [v for v in interaction_vars if v not in result.selected_vars]
            if dropped_interactions:
                print(f"\n  Dropped interactions ({len(dropped_interactions)}):")
                for var in dropped_interactions[:10]:
                    print(f"    - {var}")
                if len(dropped_interactions) > 10:
                    print(f"    ... and {len(dropped_interactions) - 10} more")
        
    except Exception as e:
        # Print error and traceback for debugging
        print(f"ERROR fitting model: {e}")
        import traceback
        traceback.print_exc()
        
else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    # Flow variables not provided - launch Shiny UI for user configuration
    if SHINY_AVAILABLE:
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        
        # Run the Shiny app and get results when it closes
        results = run_logistic_regression_ui(df)
        
        # Check if user completed the workflow
        if results['completed']:
            # Extract results from app
            coefficients = results['coefficients']
            predictions = results['predictions']
            dv = results['dv']
            print("Interactive session completed successfully")
        else:
            # User cancelled or error occurred
            print("Interactive session cancelled - returning empty results")
    else:
        # Shiny not available - print instructions for headless mode
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
# Write the results to KNIME output ports

# Ensure outputs are valid DataFrames (not None)
if coefficients is None:
    coefficients = pd.DataFrame()
if predictions is None:
    predictions = df.copy()

# Output 1: DataFrame with predictions added
# This matches R behavior:
#   df$probabilities <- predict(model, df, type = "response") %>% round(6)
#   df$predicted <- ifelse(df$probabilities > 0.5, "1", "0")
knio.output_tables[0] = knio.Table.from_pandas(predictions)

# Output 2: Model coefficients table
# This matches R format: as.data.frame(model$coefficients)
# Row ID = variable name (like R's row names), column = "model$coefficients"
coef_output = coefficients.copy()
if len(coef_output.columns) > 0:
    # Rename column to match R's naming convention
    coef_output.columns = ['model$coefficients']
knio.output_tables[1] = knio.Table.from_pandas(coef_output)

# Print completion summary
print("=" * 70)
print("Logistic Regression completed successfully")
print("=" * 70)
print(f"Output 1 (Data with Predictions): {len(predictions)} rows")
print(f"  - Added columns: 'probabilities' (rounded to 6 decimals), 'predicted' ('1' or '0')")
print(f"Output 2 (Coefficients): {len(coefficients)} terms")
print("=" * 70)

# =============================================================================
# Cleanup for Stability
# =============================================================================
# Clean up resources to prevent memory issues when running multiple nodes

# Flush stdout to ensure all messages are displayed
sys.stdout.flush()

# Delete large objects to free memory immediately
# Using try/except because variables might not exist in error cases
try:
    del df  # Delete input DataFrame
except:
    pass

try:
    del predictions  # Delete predictions DataFrame
except:
    pass

try:
    del coefficients  # Delete coefficients DataFrame
except:
    pass

# Force garbage collection to reclaim memory
# This is especially important when running multiple nodes in KNIME
gc.collect()

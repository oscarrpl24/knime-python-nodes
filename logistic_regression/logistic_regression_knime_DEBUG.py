# =============================================================================
# Logistic Regression for KNIME Python Script Node - DEBUG VERSION
# =============================================================================
# Python implementation matching R's Logistic Regression functionality
# with Shiny UI for variable selection
# Compatible with KNIME 5.9, Python 3.9
#
# DEBUG VERSION: Includes extensive logging on every function
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When DependentVariable and VarSelectionMethod are provided
#
# Inputs:
# 1. DataFrame with variables (typically WOE-transformed from WOE Editor)
#
# Outputs:
# 1. Input DataFrame with predictions added (probabilities, predicted columns)
# 2. Model coefficients table (variable name as row, coefficient value)
#
# Flow Variables (for headless mode):
# - DependentVariable (string): Binary target variable name
# - TargetCategory (optional): Which value represents the "bad" outcome
# - VarSelectionMethod (string): "All", "Stepwise", "Forward", "Backward"
# - Cutoff (float, default 2): AIC penalty multiplier (k in stepAIC)
#     k=2 for AIC (default), k=log(n) for BIC
#
# Release Date: 2026-01-17
# Version: 1.0-DEBUG
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import gc
import sys
import random
import os
import logging
import time
import functools
import traceback as tb
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================
# Configure comprehensive debug logging

# Create logger
logger = logging.getLogger('LogisticRegression_DEBUG')
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicates
logger.handlers = []

# Create console handler with detailed formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter with timestamp, function name, and line number
formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Function call counter for tracking execution flow
_call_counter = {'count': 0}

def debug_log_function(func):
    """
    Decorator to add comprehensive debug logging to functions.
    Logs entry, exit, parameters, return values, and timing.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _call_counter['count'] += 1
        call_id = _call_counter['count']
        func_name = func.__name__
        
        # Log function entry with parameters
        logger.debug(f"{'='*60}")
        logger.debug(f"[CALL #{call_id}] ENTERING: {func_name}")
        
        # Log positional arguments (truncate large objects)
        for i, arg in enumerate(args):
            arg_repr = _safe_repr(arg)
            logger.debug(f"  arg[{i}]: {arg_repr}")
        
        # Log keyword arguments
        for key, value in kwargs.items():
            value_repr = _safe_repr(value)
            logger.debug(f"  kwarg[{key}]: {value_repr}")
        
        # Execute function with timing
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log successful return
            result_repr = _safe_repr(result)
            logger.debug(f"[CALL #{call_id}] EXITING: {func_name}")
            logger.debug(f"  elapsed_time: {elapsed:.4f}s")
            logger.debug(f"  return_value: {result_repr}")
            logger.debug(f"{'='*60}")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[CALL #{call_id}] EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
            logger.error(f"  elapsed_time: {elapsed:.4f}s")
            logger.error(f"  traceback: {tb.format_exc()}")
            logger.debug(f"{'='*60}")
            raise
    
    return wrapper

def _safe_repr(obj, max_len: int = 200) -> str:
    """Create a safe string representation of an object, truncating if too long."""
    try:
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame(shape={obj.shape}, columns={list(obj.columns)[:5]}{'...' if len(obj.columns) > 5 else ''})"
        elif isinstance(obj, pd.Series):
            return f"Series(len={len(obj)}, dtype={obj.dtype})"
        elif isinstance(obj, np.ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        elif isinstance(obj, list):
            if len(obj) > 10:
                return f"list(len={len(obj)}, first_5={obj[:5]}...)"
            return f"list({obj})"
        elif isinstance(obj, dict):
            if len(obj) > 5:
                keys = list(obj.keys())[:5]
                return f"dict(len={len(obj)}, keys={keys}...)"
            return f"dict({obj})"
        else:
            repr_str = repr(obj)
            if len(repr_str) > max_len:
                return repr_str[:max_len] + "..."
            return repr_str
    except Exception:
        return f"<{type(obj).__name__}>"

def log_variable(name: str, value: Any, context: str = ""):
    """Log a variable's value with optional context."""
    value_repr = _safe_repr(value)
    if context:
        logger.debug(f"[VAR] {context} | {name} = {value_repr}")
    else:
        logger.debug(f"[VAR] {name} = {value_repr}")

def log_checkpoint(message: str):
    """Log a checkpoint in the execution flow."""
    logger.info(f"[CHECKPOINT] {message}")

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Log detailed information about a DataFrame."""
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
log_checkpoint("Initializing stability settings")

BASE_PORT = 8053
RANDOM_PORT_RANGE = 1000
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

log_variable("BASE_PORT", BASE_PORT)
log_variable("RANDOM_PORT_RANGE", RANDOM_PORT_RANGE)
log_variable("INSTANCE_ID", INSTANCE_ID)

os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

logger.debug("Set NUMEXPR_MAX_THREADS=1")
logger.debug("Set OMP_NUM_THREADS=1")

# =============================================================================
# Install/Import Dependencies
# =============================================================================
log_checkpoint("Installing/importing dependencies")

@debug_log_function
def install_if_missing(package, import_name=None):
    """Install package if not available."""
    if import_name is None:
        import_name = package
    logger.debug(f"Checking for package: {package} (import as: {import_name})")
    try:
        __import__(import_name)
        logger.debug(f"Package {package} already installed")
    except ImportError:
        logger.info(f"Installing missing package: {package}")
        import subprocess
        subprocess.check_call(['pip', 'install', package])
        logger.info(f"Successfully installed: {package}")

install_if_missing('statsmodels')
install_if_missing('scikit-learn', 'sklearn')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('plotly')

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score

logger.debug("Imported statsmodels and sklearn")

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    import plotly.express as px
    SHINY_AVAILABLE = True
    logger.info("Shiny components imported successfully")
except ImportError as e:
    logger.warning(f"Shiny not available: {e}")
    SHINY_AVAILABLE = False

log_variable("SHINY_AVAILABLE", SHINY_AVAILABLE)

# =============================================================================
# Diagnostic Functions
# =============================================================================

@debug_log_function
def check_multicollinearity(df: pd.DataFrame, x_vars: List[str], threshold: float = 0.85, 
                            vif_threshold: float = 10.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Check for multicollinearity and other data issues before model fitting.
    """
    log_checkpoint("Starting multicollinearity check")
    log_variable("threshold", threshold)
    log_variable("vif_threshold", vif_threshold)
    log_variable("num_x_vars", len(x_vars))
    
    results = {
        'high_correlations': [],
        'high_vif_vars': [],
        'low_variance_vars': [],
        'issues_found': False
    }
    
    if len(x_vars) < 2:
        logger.debug("Less than 2 variables - skipping multicollinearity check")
        return results
    
    X = df[x_vars].astype(float)
    logger.debug(f"Created X matrix with shape: {X.shape}")
    
    # Check for low variance variables
    logger.debug("Checking for low variance variables")
    variances = X.var()
    low_var = variances[variances < 1e-10].index.tolist()
    if low_var:
        logger.warning(f"Found {len(low_var)} low variance variables: {low_var[:5]}")
        results['low_variance_vars'] = low_var
        results['issues_found'] = True
    
    # Check correlation matrix
    logger.debug("Computing correlation matrix")
    try:
        corr_matrix = X.corr().abs()
        logger.debug(f"Correlation matrix computed, shape: {corr_matrix.shape}")
        
        high_corr_pairs = []
        for i, var1 in enumerate(x_vars):
            for j, var2 in enumerate(x_vars):
                if i < j:
                    corr = corr_matrix.loc[var1, var2]
                    if corr > threshold:
                        high_corr_pairs.append((var1, var2, corr))
        
        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} highly correlated pairs")
            results['high_correlations'] = sorted(high_corr_pairs, key=lambda x: -x[2])
            results['issues_found'] = True
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
    
    # Calculate VIF
    logger.debug("Calculating VIF values")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        if len(x_vars) > 1 and len(x_vars) < 50 and len(df) > len(x_vars) + 10:
            X_with_const = sm.add_constant(X.dropna())
            logger.debug(f"X_with_const shape after dropna: {X_with_const.shape}")
            
            if len(X_with_const) > len(x_vars) + 1:
                vif_data = []
                for i, var in enumerate(x_vars):
                    try:
                        vif = variance_inflation_factor(X_with_const.values, i + 1)
                        logger.debug(f"VIF for {var}: {vif}")
                        if vif > vif_threshold and not np.isinf(vif):
                            vif_data.append((var, vif))
                    except Exception as e:
                        logger.warning(f"VIF calculation failed for {var}: {e}")
                
                if vif_data:
                    logger.warning(f"Found {len(vif_data)} high VIF variables")
                    results['high_vif_vars'] = sorted(vif_data, key=lambda x: -x[1])
                    results['issues_found'] = True
    except ImportError:
        logger.debug("variance_inflation_factor not available")
    except Exception as e:
        logger.error(f"VIF calculation error: {e}")
    
    # Print diagnostics
    if verbose and results['issues_found']:
        print("\n" + "=" * 70)
        print("⚠️  MULTICOLLINEARITY DIAGNOSTICS")
        print("=" * 70)
        
        if results['low_variance_vars']:
            print(f"\n🔴 LOW VARIANCE VARIABLES ({len(results['low_variance_vars'])}):")
            print("   These variables have near-zero variance and may cause fitting issues:")
            for var in results['low_variance_vars'][:10]:
                print(f"     - {var}")
            if len(results['low_variance_vars']) > 10:
                print(f"     ... and {len(results['low_variance_vars']) - 10} more")
        
        if results['high_correlations']:
            print(f"\n🟠 HIGHLY CORRELATED PAIRS (r > {threshold}):")
            print("   Consider removing one variable from each pair:")
            for var1, var2, corr in results['high_correlations'][:10]:
                print(f"     - {var1} ↔ {var2}: r = {corr:.3f}")
            if len(results['high_correlations']) > 10:
                print(f"     ... and {len(results['high_correlations']) - 10} more pairs")
        
        if results['high_vif_vars']:
            print(f"\n🟡 HIGH VIF VARIABLES (VIF > {vif_threshold}):")
            print("   These variables have high multicollinearity with other predictors:")
            for var, vif in results['high_vif_vars'][:10]:
                print(f"     - {var}: VIF = {vif:.1f}")
            if len(results['high_vif_vars']) > 10:
                print(f"     ... and {len(results['high_vif_vars']) - 10} more")
        
        print("\n" + "-" * 70)
        print("💡 RECOMMENDATION: Address these issues in earlier pipeline steps")
        print("   (e.g., remove correlated variables in Variable Selection node)")
        print("=" * 70 + "\n")
    
    log_variable("issues_found", results['issues_found'])
    return results


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepwiseResult:
    """Container for stepwise selection results"""
    selected_vars: List[str]
    aic_history: List[float]
    steps: List[str]


@dataclass
class ModelResult:
    """Container for logistic regression model results"""
    model: Any
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    selected_vars: List[str]


# =============================================================================
# Stepwise Selection Functions
# =============================================================================

_stepwise_numerical_issues = set()


@debug_log_function
def fit_logit_model(df: pd.DataFrame, y_var: str, x_vars: List[str], track_issues: bool = True) -> Tuple[Any, float]:
    """
    Fit a logistic regression model and return the model and AIC.
    """
    global _stepwise_numerical_issues
    
    log_variable("y_var", y_var)
    log_variable("num_x_vars", len(x_vars))
    log_variable("x_vars_sample", x_vars[:5] if x_vars else [])
    log_variable("df_shape", df.shape)
    
    if not x_vars:
        logger.debug("Fitting null model (intercept only)")
        X = sm.add_constant(pd.DataFrame(index=df.index, columns=[]))
        X = np.ones((len(df), 1))
    else:
        X = sm.add_constant(df[x_vars].astype(float))
    
    y = df[y_var].astype(float)
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.debug(f"y value counts: {dict(y.value_counts())}")
    
    had_numerical_issues = False
    
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            
            # Try BFGS first
            logger.debug("Attempting BFGS optimization")
            try:
                model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
                logger.debug("BFGS optimization succeeded")
            except Exception as e:
                logger.warning(f"BFGS failed: {e}")
                had_numerical_issues = True
                
                # Fallback to regularized
                logger.debug("Attempting L1 regularized fitting")
                try:
                    model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
                    logger.debug("L1 regularized fitting succeeded")
                except Exception as e2:
                    logger.warning(f"L1 regularized failed: {e2}")
                    logger.debug("Attempting Newton-Raphson method")
                    model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
                    logger.debug("Newton-Raphson succeeded")
            
            # Check for warnings
            for w in caught_warnings:
                if 'Hessian' in str(w.message) or 'cov_params' in str(w.message):
                    logger.warning(f"Numerical warning detected: {w.message}")
                    had_numerical_issues = True
                    break
        
        if track_issues and had_numerical_issues and x_vars:
            logger.debug(f"Tracking numerical issue for variable: {x_vars[-1]}")
            _stepwise_numerical_issues.add(x_vars[-1])
        
        aic = model.aic
        logger.debug(f"Model AIC: {aic}")
        return model, aic
        
    except Exception as e:
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
    """
    log_checkpoint("Starting forward stepwise selection")
    log_variable("y_var", y_var)
    log_variable("num_candidate_vars", len(candidate_vars))
    log_variable("k", k)
    
    selected = []
    remaining = list(candidate_vars)
    aic_history = []
    steps = []
    
    # Fit null model
    logger.debug("Fitting null model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Forward Selection: Start AIC = {current_aic:.4f}")
    
    iteration = 0
    improved = True
    while improved and remaining:
        iteration += 1
        logger.debug(f"Forward iteration {iteration}: {len(selected)} selected, {len(remaining)} remaining")
        improved = False
        best_var = None
        best_aic = current_aic
        
        for var in remaining:
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                improved = True
        
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
    """
    log_checkpoint("Starting backward stepwise elimination")
    log_variable("y_var", y_var)
    log_variable("num_current_vars", len(current_vars))
    log_variable("k", k)
    
    selected = list(current_vars)
    aic_history = []
    steps = []
    
    # Fit full model
    logger.debug("Fitting full model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Backward Elimination: Start AIC = {current_aic:.4f}")
    
    iteration = 0
    improved = True
    while improved and len(selected) > 0:
        iteration += 1
        logger.debug(f"Backward iteration {iteration}: {len(selected)} variables remaining")
        improved = False
        worst_var = None
        best_aic = current_aic
        
        for var in selected:
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                worst_var = var
                improved = True
        
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
    """
    log_checkpoint("Starting bidirectional stepwise selection")
    log_variable("y_var", y_var)
    log_variable("num_current_vars", len(current_vars))
    log_variable("k", k)
    
    selected = list(current_vars)
    remaining = []
    aic_history = []
    steps = []
    all_vars = list(current_vars)
    
    # Fit current model
    logger.debug("Fitting initial model")
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Stepwise Selection: Start AIC = {current_aic:.4f}")
    
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
    """
    log_checkpoint("Starting logistic regression fitting")
    log_variable("y_var", y_var)
    log_variable("num_x_vars", len(x_vars))
    log_variable("method", method)
    log_variable("k", k)
    log_dataframe_info(df, "Input DataFrame")
    
    cols_to_use = [y_var] + x_vars
    df_clean = df[cols_to_use].dropna()
    
    logger.debug(f"After dropna: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")
    
    if len(df_clean) == 0:
        logger.error("No complete cases after removing missing values")
        raise ValueError("No complete cases after removing missing values")
    
    if verbose:
        print(f"Fitting logistic regression: {len(df_clean)} observations, {len(x_vars)} variables")
        print(f"Method: {method}, k = {k}")
    
    # Run diagnostics
    if verbose:
        logger.debug("Running multicollinearity diagnostics")
        diagnostics = check_multicollinearity(df_clean, x_vars, threshold=0.85, vif_threshold=10.0, verbose=True)
    
    # Reset numerical issues tracker
    global _stepwise_numerical_issues
    _stepwise_numerical_issues = set()
    
    # Variable selection
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
    
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&display=swap');
                body { 
                    font-family: 'Raleway', sans-serif; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    color: #e8e8e8;
                }
                .card { 
                    background: rgba(255, 255, 255, 0.08);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px; 
                    padding: 24px; 
                    margin: 12px 0; 
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }
                .card-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px 12px 0 0;
                    margin: -24px -24px 20px -24px;
                    padding: 16px 24px;
                    color: white;
                    font-weight: 600;
                }
                h4 { 
                    font-weight: 700; 
                    text-align: center; 
                    margin: 20px 0; 
                    color: #fff;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }
                h5 {
                    color: #a8dadc;
                    font-weight: 600;
                    margin-bottom: 16px;
                }
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
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                }
                .btn-primary { 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                }
                .btn-success { 
                    background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                }
                .btn-info { 
                    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%); 
                }
                .divider { width: 12px; display: inline-block; }
                .form-control, .form-select {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    color: #fff;
                    border-radius: 10px;
                }
                .form-control:focus, .form-select:focus {
                    background: rgba(255, 255, 255, 0.15);
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
                    color: #fff;
                }
                .form-select option {
                    background: #1a1a2e;
                    color: #fff;
                }
                .form-label {
                    color: #a8dadc;
                    font-weight: 500;
                }
                .form-check-input:checked {
                    background-color: #667eea;
                    border-color: #667eea;
                }
                table {
                    color: #e8e8e8 !important;
                }
                .dataframe {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                }
                .var-checkbox-container {
                    max-height: 400px;
                    overflow-y: auto;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 10px;
                    padding: 16px;
                }
                .var-checkbox-container::-webkit-scrollbar {
                    width: 8px;
                }
                .var-checkbox-container::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                }
                .var-checkbox-container::-webkit-scrollbar-thumb {
                    background: #667eea;
                    border-radius: 4px;
                }
            """)
        ),
        
        ui.h4("🔬 Logistic Regression [DEBUG MODE]"),
        
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Configuration"),
            ui.row(
                ui.column(4,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=list(df.columns),
                                   selected=df.columns[-1] if len(df.columns) > 0 else None)
                ),
                ui.column(4,
                    ui.input_select("tc", "Target Category", choices=[])
                ),
                ui.column(4,
                    ui.input_select("method", "Variable Selection Method",
                                   choices=["Must include all", "Stepwise Selection", 
                                           "Forward Selection", "Backward Selection"],
                                   selected="Must include all")
                )
            ),
            ui.row(
                ui.column(4,
                    ui.input_numeric("cutoff", "AIC Penalty (k)", value=2, min=0, step=0.5),
                    ui.tags.small("k=2 for AIC, k=log(n) for BIC", style="color: #888;")
                ),
                ui.column(4,
                    ui.input_action_button("select_woe", "Select WOE Variables", class_="btn btn-info"),
                    ui.tags.small("Auto-select WOE_ prefixed vars", style="color: #888;")
                ),
                ui.column(4,
                    ui.input_action_button("select_all", "Select All Variables", class_="btn btn-primary"),
                )
            )
        ),
        
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Variable Selection"),
            ui.row(
                ui.column(12,
                    ui.div(
                        {"class": "var-checkbox-container"},
                        ui.output_ui("var_checkboxes")
                    )
                )
            )
        ),
        
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Results"),
            ui.row(
                ui.column(12,
                    ui.h5("Coefficients"),
                    ui.output_data_frame("coef_table")
                )
            )
        ),
        
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    ui.h5("Coefficient Plot"),
                    output_widget("coef_plot")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    ui.h5("ROC Curve"),
                    output_widget("roc_plot")
                )
            )
        ),
        
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "🚀 Run Model & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        logger.debug("Shiny server function initialized")
        model_result_rv = reactive.Value(None)
        selected_vars_rv = reactive.Value([])
        
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            logger.debug(f"[Shiny] DV changed to: {dv}")
            if dv and dv in df.columns:
                unique_vals = df[dv].dropna().unique().tolist()
                unique_vals = [str(v) for v in sorted(unique_vals)]
                logger.debug(f"[Shiny] Target categories: {unique_vals}")
                ui.update_select("tc", choices=unique_vals, 
                               selected=unique_vals[-1] if unique_vals else None)
        
        @output
        @render.ui
        def var_checkboxes():
            dv = input.dv()
            logger.debug(f"[Shiny] Rendering variable checkboxes, DV={dv}")
            if not dv:
                return ui.p("Select a dependent variable first")
            
            available_vars = [col for col in df.columns if col != dv]
            logger.debug(f"[Shiny] Available vars: {len(available_vars)}")
            
            checkboxes = []
            for var in available_vars:
                is_woe = var.startswith('WOE_')
                checkbox = ui.input_checkbox(f"var_{var}", var, value=is_woe)
                checkboxes.append(checkbox)
            
            return ui.div(*checkboxes)
        
        @reactive.Effect
        @reactive.event(input.select_woe)
        def select_woe_vars():
            logger.debug("[Shiny] Select WOE variables button clicked")
            dv = input.dv()
            if not dv:
                return
            
            available_vars = [col for col in df.columns if col != dv]
            woe_count = 0
            for var in available_vars:
                is_woe = var.startswith('WOE_')
                if is_woe:
                    woe_count += 1
                try:
                    ui.update_checkbox(f"var_{var}", value=is_woe)
                except:
                    pass
            logger.debug(f"[Shiny] Selected {woe_count} WOE variables")
        
        @reactive.Effect
        @reactive.event(input.select_all)
        def select_all_vars():
            logger.debug("[Shiny] Select all variables button clicked")
            dv = input.dv()
            if not dv:
                return
            
            available_vars = [col for col in df.columns if col != dv]
            for var in available_vars:
                should_select = not var.startswith('b_')
                try:
                    ui.update_checkbox(f"var_{var}", value=should_select)
                except:
                    pass
        
        @reactive.Calc
        def get_selected_vars():
            dv = input.dv()
            if not dv:
                return []
            
            available_vars = [col for col in df.columns if col != dv]
            selected = []
            for var in available_vars:
                try:
                    if input[f"var_{var}"]():
                        selected.append(var)
                except:
                    pass
            return selected
        
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_model():
            logger.info("[Shiny] Run Model button clicked")
            dv = input.dv()
            selected = get_selected_vars()
            method_raw = input.method()
            cutoff = input.cutoff()
            
            logger.debug(f"[Shiny] DV: {dv}, Method: {method_raw}, Cutoff: {cutoff}")
            logger.debug(f"[Shiny] Selected variables: {len(selected)}")
            
            if not dv or not selected:
                logger.warning("[Shiny] No DV or no variables selected")
                return
            
            method_map = {
                "Must include all": "All",
                "Stepwise Selection": "Stepwise",
                "Forward Selection": "Forward",
                "Backward Selection": "Backward"
            }
            method = method_map.get(method_raw, "All")
            
            try:
                logger.info(f"[Shiny] Fitting model with method: {method}")
                result = fit_logistic_regression(
                    df=df,
                    y_var=dv,
                    x_vars=selected,
                    method=method,
                    k=cutoff if cutoff else 2.0,
                    verbose=True
                )
                
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
            
            fig = go.Figure(data=[
                go.Bar(
                    y=coef_df['Variable'],
                    x=coef_df['Coefficient'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{c:.3f}" for c in coef_df['Coefficient']],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Coefficients (excluding intercept)',
                xaxis_title='Coefficient',
                yaxis_title='Variable',
                height=350,
                margin=dict(l=150, r=50, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8')
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
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
            
            y_true = pred_df[dv].values
            y_score = pred_df['probabilities'].values
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            
            auc = roc_auc_score(y_true, y_score)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc:.3f})',
                line=dict(color='#667eea', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve (AUC = {auc:.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=350,
                margin=dict(l=50, r=50, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8'),
                legend=dict(x=0.6, y=0.1)
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
            
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
            logger.debug(f"Port {port} in use, trying next")
            continue
    
    # Fallback
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
        sys.stdout.flush()
        try:
            fallback_port = find_free_port(port + 100)
            logger.info(f"Retrying on fallback port {fallback_port}")
            print(f"Retrying on port {fallback_port}")
            sys.stdout.flush()
            app.run(port=fallback_port, launch_browser=True)
        except Exception as e2:
            logger.error(f"Failed on fallback port: {e2}")
            print(f"Failed on fallback port: {e2}")
            app.results['completed'] = False
    
    gc.collect()
    sys.stdout.flush()
    
    log_checkpoint("Shiny UI application finished")
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
log_checkpoint("=" * 70)
log_checkpoint("LOGISTIC REGRESSION NODE - DEBUG VERSION - STARTING")
log_checkpoint("=" * 70)

print("Logistic Regression Node - DEBUG VERSION - Starting...")
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
    logger.debug(f"VarSelectionMethod '{sel_method}' not valid, must be one of: {selection_methods}")

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
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    log_checkpoint("Running in HEADLESS mode")
    print("Running in HEADLESS mode")
    
    x_vars = [col for col in df.columns if col != dv]
    log_variable("initial_x_vars_count", len(x_vars))
    
    woe_vars = [col for col in x_vars if col.startswith('WOE_')]
    interaction_vars = [col for col in woe_vars if '_x_' in col]
    single_woe_vars = [col for col in woe_vars if '_x_' not in col]
    
    log_variable("woe_vars_count", len(woe_vars))
    log_variable("interaction_vars_count", len(interaction_vars))
    log_variable("single_woe_vars_count", len(single_woe_vars))
    
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
    
    log_variable("final_x_vars_count", len(x_vars))
    
    try:
        logger.info("Fitting logistic regression model")
        result = fit_logistic_regression(
            df=df,
            y_var=dv,
            x_vars=x_vars,
            method=sel_method,
            k=k,
            verbose=True
        )
        
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
            print(f"  Interaction variables: None selected (may have been dropped by stepwise)")
        
        if interaction_vars:
            dropped_interactions = [v for v in interaction_vars if v not in result.selected_vars]
            if dropped_interactions:
                print(f"\n  Dropped interactions ({len(dropped_interactions)}):")
                for var in dropped_interactions[:10]:
                    print(f"    - {var}")
                if len(dropped_interactions) > 10:
                    print(f"    ... and {len(dropped_interactions) - 10} more")
        
    except Exception as e:
        logger.error(f"ERROR fitting model: {e}")
        logger.error(f"Traceback: {tb.format_exc()}")
        print(f"ERROR fitting model: {e}")
        import traceback
        traceback.print_exc()
        
else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
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
except:
    pass

try:
    del predictions
    logger.debug("Deleted predictions")
except:
    pass

try:
    del coefficients
    logger.debug("Deleted coefficients")
except:
    pass

gc.collect()
logger.debug("Garbage collection complete")

log_checkpoint("=" * 70)
log_checkpoint("LOGISTIC REGRESSION NODE - DEBUG VERSION - FINISHED")
log_checkpoint("=" * 70)

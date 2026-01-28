# =============================================================================
# Logistic Regression for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's Logistic Regression functionality
# with Shiny UI for variable selection
# Compatible with KNIME 5.9, Python 3.9
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
# Version: 1.0
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import gc
import sys
import random
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Only suppress during stepwise iterations (handled in fit_logit_model)

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# Use random port to avoid conflicts when running multiple instances
BASE_PORT = 8053  # Different from other scripts to avoid conflicts
RANDOM_PORT_RANGE = 1000

# Process isolation: Set unique temp directories per instance
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts

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

install_if_missing('statsmodels')
install_if_missing('scikit-learn', 'sklearn')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('plotly')

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score

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
# Diagnostic Functions
# =============================================================================

def check_multicollinearity(df: pd.DataFrame, x_vars: List[str], threshold: float = 0.85, 
                            vif_threshold: float = 10.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Check for multicollinearity and other data issues before model fitting.
    
    Parameters:
        df: DataFrame with data
        x_vars: List of predictor variable names
        threshold: Correlation threshold for flagging (default 0.85)
        vif_threshold: VIF threshold for flagging (default 10.0)
        verbose: Print diagnostics
        
    Returns:
        Dictionary with diagnostic results
    """
    results = {
        'high_correlations': [],
        'high_vif_vars': [],
        'low_variance_vars': [],
        'issues_found': False
    }
    
    if len(x_vars) < 2:
        return results
    
    # Get numeric data for selected variables
    X = df[x_vars].astype(float)
    
    # Check for low variance variables
    variances = X.var()
    low_var = variances[variances < 1e-10].index.tolist()
    if low_var:
        results['low_variance_vars'] = low_var
        results['issues_found'] = True
    
    # Check correlation matrix
    try:
        corr_matrix = X.corr().abs()
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for i, var1 in enumerate(x_vars):
            for j, var2 in enumerate(x_vars):
                if i < j:  # Only upper triangle
                    corr = corr_matrix.loc[var1, var2]
                    if corr > threshold:
                        high_corr_pairs.append((var1, var2, corr))
        
        if high_corr_pairs:
            results['high_correlations'] = sorted(high_corr_pairs, key=lambda x: -x[2])
            results['issues_found'] = True
    except Exception:
        pass
    
    # Calculate VIF for each variable (if statsmodels variance_inflation_factor is available)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Only calculate VIF if we have enough data and not too many variables
        if len(x_vars) > 1 and len(x_vars) < 50 and len(df) > len(x_vars) + 10:
            X_with_const = sm.add_constant(X.dropna())
            if len(X_with_const) > len(x_vars) + 1:
                vif_data = []
                for i, var in enumerate(x_vars):
                    try:
                        vif = variance_inflation_factor(X_with_const.values, i + 1)  # +1 to skip constant
                        if vif > vif_threshold and not np.isinf(vif):
                            vif_data.append((var, vif))
                    except Exception:
                        pass
                
                if vif_data:
                    results['high_vif_vars'] = sorted(vif_data, key=lambda x: -x[1])
                    results['issues_found'] = True
    except ImportError:
        pass
    except Exception:
        pass
    
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
    
    return results


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepwiseResult:
    """Container for stepwise selection results"""
    selected_vars: List[str]  # Selected variable names
    aic_history: List[float]  # AIC at each step
    steps: List[str]  # Description of each step


@dataclass
class ModelResult:
    """Container for logistic regression model results"""
    model: Any  # Fitted statsmodels model
    coefficients: pd.DataFrame  # Coefficient table (variable name as index, coefficient value)
    predictions: pd.DataFrame  # DataFrame with probabilities and predicted columns added
    selected_vars: List[str]  # Variables in final model


# =============================================================================
# Stepwise Selection Functions (equivalent to R's MASS::stepAIC)
# =============================================================================

# Track variables that cause numerical issues during stepwise
_stepwise_numerical_issues = set()


def fit_logit_model(df: pd.DataFrame, y_var: str, x_vars: List[str], track_issues: bool = True) -> Tuple[Any, float]:
    """
    Fit a logistic regression model and return the model and AIC.
    
    Parameters:
        df: DataFrame with data
        y_var: Name of the dependent variable
        x_vars: List of independent variable names
        track_issues: Whether to track variables that cause numerical issues
        
    Returns:
        Tuple of (fitted model, AIC value)
    """
    global _stepwise_numerical_issues
    
    if not x_vars:
        # Null model (intercept only)
        X = sm.add_constant(pd.DataFrame(index=df.index, columns=[]))
        X = np.ones((len(df), 1))
    else:
        X = sm.add_constant(df[x_vars].astype(float))
    
    y = df[y_var].astype(float)
    
    had_numerical_issues = False
    
    try:
        # Use context manager to suppress all warnings during fit
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            
            # Try BFGS first (faster, usually works)
            try:
                model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
            except Exception:
                had_numerical_issues = True
                # Fallback to Newton-Raphson with regularization if BFGS fails
                try:
                    model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
                except Exception:
                    # Last resort: simple Newton method
                    model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
            
            # Check if any Hessian warnings were raised
            for w in caught_warnings:
                if 'Hessian' in str(w.message) or 'cov_params' in str(w.message):
                    had_numerical_issues = True
                    break
        
        # Track which variables caused issues
        if track_issues and had_numerical_issues and x_vars:
            # The last variable added is likely the problematic one
            _stepwise_numerical_issues.add(x_vars[-1])
        
        aic = model.aic
        return model, aic
    except Exception as e:
        # Track failure - likely the last variable caused it
        if track_issues and x_vars:
            _stepwise_numerical_issues.add(x_vars[-1])
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
    
    Parameters:
        df: DataFrame with data
        y_var: Dependent variable name
        candidate_vars: List of candidate predictor variables
        k: Penalty multiplier (2 for AIC, log(n) for BIC)
        verbose: Print progress
        
    Returns:
        StepwiseResult with selected variables
    """
    selected = []
    remaining = list(candidate_vars)
    aic_history = []
    steps = []
    
    # Fit null model
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Forward Selection: Start AIC = {current_aic:.4f}")
    
    improved = True
    while improved and remaining:
        improved = False
        best_var = None
        best_aic = current_aic
        
        for var in remaining:
            test_vars = selected + [var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            
            # Adjust AIC with penalty
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                best_var = var
                improved = True
        
        if improved and best_var:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"+ {best_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  + {best_var}: AIC = {current_aic:.4f}")
    
    if verbose:
        print(f"Forward Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
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
    
    Parameters:
        df: DataFrame with data
        y_var: Dependent variable name
        current_vars: List of current predictor variables (starting set)
        k: Penalty multiplier (2 for AIC, log(n) for BIC)
        verbose: Print progress
        
    Returns:
        StepwiseResult with selected variables
    """
    selected = list(current_vars)
    aic_history = []
    steps = []
    
    # Fit full model
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Backward Elimination: Start AIC = {current_aic:.4f}")
    
    improved = True
    while improved and len(selected) > 0:
        improved = False
        worst_var = None
        best_aic = current_aic
        
        for var in selected:
            test_vars = [v for v in selected if v != var]
            _, test_aic = fit_logit_model(df, y_var, test_vars)
            
            # Adjust AIC with penalty
            adjusted_aic = test_aic + (k - 2) * len(test_vars)
            
            if adjusted_aic < best_aic:
                best_aic = adjusted_aic
                worst_var = var
                improved = True
        
        if improved and worst_var:
            selected.remove(worst_var)
            current_aic = best_aic
            aic_history.append(current_aic)
            steps.append(f"- {worst_var}: AIC={current_aic:.4f}")
            
            if verbose:
                print(f"  - {worst_var}: AIC = {current_aic:.4f}")
    
    if verbose:
        print(f"Backward Elimination: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
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
    
    Parameters:
        df: DataFrame with data
        y_var: Dependent variable name
        current_vars: List of current predictor variables (starting set)
        k: Penalty multiplier (2 for AIC, log(n) for BIC)
        verbose: Print progress
        
    Returns:
        StepwiseResult with selected variables
    """
    selected = list(current_vars)
    remaining = []  # Variables not in model (initially empty for "both" from full model)
    aic_history = []
    steps = []
    
    # Track all variables ever considered
    all_vars = list(current_vars)
    
    # Fit current model
    _, current_aic = fit_logit_model(df, y_var, selected)
    aic_history.append(current_aic)
    steps.append(f"Start: AIC={current_aic:.4f}")
    
    if verbose:
        print(f"Stepwise Selection: Start AIC = {current_aic:.4f}")
    
    improved = True
    iteration = 0
    max_iterations = len(all_vars) * 2  # Prevent infinite loops
    
    while improved and iteration < max_iterations:
        iteration += 1
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
                selected.remove(best_var)
                remaining.append(best_var)
                steps.append(f"- {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  - {best_var}: AIC = {best_aic:.4f}")
            else:
                selected.append(best_var)
                remaining.remove(best_var)
                steps.append(f"+ {best_var}: AIC={best_aic:.4f}")
                if verbose:
                    print(f"  + {best_var}: AIC = {best_aic:.4f}")
            
            current_aic = best_aic
            aic_history.append(current_aic)
    
    if verbose:
        print(f"Stepwise Selection: Final AIC = {current_aic:.4f}, {len(selected)} variables")
    
    return StepwiseResult(selected_vars=selected, aic_history=aic_history, steps=steps)


# =============================================================================
# Model Fitting and Evaluation
# =============================================================================

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
    
    Parameters:
        df: DataFrame with data
        y_var: Dependent variable name
        x_vars: List of predictor variable names
        method: Variable selection method ("All", "Stepwise", "Forward", "Backward")
        k: AIC penalty multiplier
        verbose: Print progress
        
    Returns:
        ModelResult with fitted model and statistics
    """
    # Clean data - remove rows with missing values in selected columns
    cols_to_use = [y_var] + x_vars
    df_clean = df[cols_to_use].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No complete cases after removing missing values")
    
    if verbose:
        print(f"Fitting logistic regression: {len(df_clean)} observations, {len(x_vars)} variables")
        print(f"Method: {method}, k = {k}")
    
    # Run multicollinearity diagnostics before fitting
    if verbose:
        diagnostics = check_multicollinearity(df_clean, x_vars, threshold=0.85, vif_threshold=10.0, verbose=True)
    
    # Reset numerical issues tracker before stepwise
    global _stepwise_numerical_issues
    _stepwise_numerical_issues = set()
    
    # Variable selection
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
        selected_vars = x_vars
        stepwise_result = None
    
    if not selected_vars:
        raise ValueError("No variables selected - model cannot be fit")
    
    # Report any variables that caused numerical issues during stepwise
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
    X = sm.add_constant(df_clean[selected_vars].astype(float))
    y = df_clean[y_var].astype(float)
    
    # Suppress warnings during final model fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Try BFGS first, with fallbacks for numerical issues
        try:
            model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=1000)
        except Exception:
            if verbose:
                print("  Note: BFGS optimization had issues, trying Newton method...")
            try:
                model = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=500)
            except Exception:
                if verbose:
                    print("  Note: Newton method had issues, using L1 regularization...")
                model = sm.Logit(y, X).fit_regularized(disp=0, method='l1', alpha=0.01)
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL SUMMARY")
        print("="*70)
        print(model.summary())
    
    # Create coefficient table matching R format: model$coefficients
    # Row names are variable names, single column with coefficient values
    # Use actual param names from model to handle cases where constant wasn't added
    coef_df = model.params.to_frame(name='coefficients')
    coef_df.index.name = None  # Match R's row names without explicit index name
    
    # Rename 'const' to '(Intercept)' for R compatibility
    if 'const' in coef_df.index:
        coef_df = coef_df.rename(index={'const': '(Intercept)'})
    
    # Calculate predictions for the entire original dataframe
    # This matches R's: predict(model, df, type = "response")
    predictions = df.copy()
    
    # For rows with complete data, calculate predictions
    X_full = sm.add_constant(df[selected_vars].astype(float), has_constant='add')
    
    # Handle missing values - predict only where we have complete cases
    complete_mask = X_full.notna().all(axis=1)
    
    # Initialize columns
    predictions['probabilities'] = np.nan
    predictions['predicted'] = None
    
    # Predict probabilities for complete cases
    if complete_mask.any():
        X_complete = X_full[complete_mask]
        proba = model.predict(X_complete)
        predictions.loc[complete_mask, 'probabilities'] = np.round(proba.values, 6)
        predictions.loc[complete_mask, 'predicted'] = np.where(proba.values > 0.5, "1", "0")
    
    # Print summary statistics
    if verbose:
        try:
            y_actual = df_clean[y_var].astype(float)
            y_proba = model.predict(X)
            auc = roc_auc_score(y_actual, y_proba)
            print(f"\nModel AUC: {auc:.4f}")
            print(f"Gini: {2*auc - 1:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
    
    return ModelResult(
        model=model,
        coefficients=coef_df,
        predictions=predictions,
        selected_vars=selected_vars
    )


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_logistic_regression_app(df: pd.DataFrame):
    """Create the Logistic Regression Shiny application."""
    
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
        
        ui.h4("🔬 Logistic Regression"),
        
        # Configuration Card
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
        
        # Variable Selection Card
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
        
        # Results Card
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
        
        # Charts Card
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
        
        # Action Button
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "🚀 Run Model & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        model_result_rv = reactive.Value(None)
        selected_vars_rv = reactive.Value([])
        
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            if dv and dv in df.columns:
                unique_vals = df[dv].dropna().unique().tolist()
                # Sort and convert to strings for display
                unique_vals = [str(v) for v in sorted(unique_vals)]
                ui.update_select("tc", choices=unique_vals, 
                               selected=unique_vals[-1] if unique_vals else None)
        
        @output
        @render.ui
        def var_checkboxes():
            dv = input.dv()
            if not dv:
                return ui.p("Select a dependent variable first")
            
            available_vars = [col for col in df.columns if col != dv]
            
            # Create checkbox inputs for each variable
            checkboxes = []
            for var in available_vars:
                # Pre-select WOE variables
                is_woe = var.startswith('WOE_')
                checkbox = ui.input_checkbox(f"var_{var}", var, value=is_woe)
                checkboxes.append(checkbox)
            
            return ui.div(*checkboxes)
        
        @reactive.Effect
        @reactive.event(input.select_woe)
        def select_woe_vars():
            dv = input.dv()
            if not dv:
                return
            
            available_vars = [col for col in df.columns if col != dv]
            for var in available_vars:
                is_woe = var.startswith('WOE_')
                # Update checkbox
                try:
                    ui.update_checkbox(f"var_{var}", value=is_woe)
                except:
                    pass
        
        @reactive.Effect
        @reactive.event(input.select_all)
        def select_all_vars():
            dv = input.dv()
            if not dv:
                return
            
            available_vars = [col for col in df.columns if col != dv]
            for var in available_vars:
                # Exclude b_ prefixed variables (binned, not for regression)
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
            dv = input.dv()
            selected = get_selected_vars()
            method_raw = input.method()
            cutoff = input.cutoff()
            
            if not dv or not selected:
                return
            
            # Map method name
            method_map = {
                "Must include all": "All",
                "Stepwise Selection": "Stepwise",
                "Forward Selection": "Forward",
                "Backward Selection": "Backward"
            }
            method = method_map.get(method_raw, "All")
            
            try:
                result = fit_logistic_regression(
                    df=df,
                    y_var=dv,
                    x_vars=selected,
                    method=method,
                    k=cutoff if cutoff else 2.0,
                    verbose=True
                )
                
                model_result_rv.set(result)
                
                # Store results for output
                app_results['coefficients'] = result.coefficients
                app_results['predictions'] = result.predictions
                app_results['selected_vars'] = result.selected_vars
                app_results['dv'] = dv
                app_results['completed'] = True
                
                await session.close()
                
            except Exception as e:
                print(f"Error fitting model: {e}")
                import traceback
                traceback.print_exc()
        
        @output
        @render.data_frame
        def coef_table():
            result = model_result_rv.get()
            if result is None:
                return render.DataGrid(pd.DataFrame())
            
            # Display coefficients with variable name
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
            
            # Reset index to get variable names as column
            coef_df = result.coefficients.reset_index()
            coef_df.columns = ['Variable', 'Coefficient']
            
            # Exclude intercept
            coef_df = coef_df[coef_df['Variable'] != '(Intercept)'].copy()
            if coef_df.empty:
                return go.Figure()
            
            # Sort by absolute coefficient value
            coef_df['abs_coef'] = abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('abs_coef', ascending=True)
            
            # Color based on sign
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
            
            # Get actual and predicted values (using new column names)
            pred_df = result.predictions.dropna(subset=['probabilities'])
            if pred_df.empty:
                return go.Figure()
            
            y_true = pred_df[dv].values
            y_score = pred_df['probabilities'].values
            
            # Calculate ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            
            auc = roc_auc_score(y_true, y_score)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc:.3f})',
                line=dict(color='#667eea', width=3)
            ))
            
            # Diagonal reference line
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
    return app


def find_free_port(start_port: int = 8053, max_attempts: int = 50) -> int:
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


def run_logistic_regression_ui(df: pd.DataFrame, port: int = None):
    """Run the Logistic Regression application and return results."""
    # Find a free port to avoid conflicts with multiple instances
    if port is None:
        port = find_free_port(BASE_PORT)
    
    print(f"Starting Shiny app on port {port} (Instance: {INSTANCE_ID})")
    sys.stdout.flush()
    
    app = create_logistic_regression_app(df)
    
    try:
        app.run(port=port, launch_browser=True)
    except Exception as e:
        print(f"Error running Shiny app on port {port}: {e}")
        sys.stdout.flush()
        # Try with a different port
        try:
            fallback_port = find_free_port(port + 100)
            print(f"Retrying on port {fallback_port}")
            sys.stdout.flush()
            app.run(port=fallback_port, launch_browser=True)
        except Exception as e2:
            print(f"Failed on fallback port: {e2}")
            app.results['completed'] = False
    
    # Cleanup
    gc.collect()
    sys.stdout.flush()
    
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
print("Logistic Regression Node - Starting...")
print("=" * 70)

df = knio.input_tables[0].to_pandas()
print(f"Input data: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
contains_dv = False
contains_method = False
dv = None
target = None
sel_method = None
k = 2.0

try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

try:
    sel_method = knio.flow_variables.get("VarSelectionMethod", None)
except:
    pass

try:
    k = knio.flow_variables.get("Cutoff", 2.0)
    if k is None:
        k = 2.0
except:
    k = 2.0

# Validate DependentVariable
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        print(f"[OK] DependentVariable: {dv}")

# Validate VarSelectionMethod
selection_methods = ["All", "Stepwise", "Forward", "Backward"]
if sel_method is not None and sel_method in selection_methods:
    contains_method = True
    print(f"[OK] VarSelectionMethod: {sel_method}")

print(f"Cutoff (k): {k}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================

# Initialize outputs
coefficients = pd.DataFrame()
predictions = df.copy()

if contains_dv and contains_method:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    print("Running in HEADLESS mode")
    
    # Get predictor variables (all columns except DV)
    x_vars = [col for col in df.columns if col != dv]
    
    # Optionally filter to WOE variables only if available
    woe_vars = [col for col in x_vars if col.startswith('WOE_')]
    
    # Identify interaction columns (contain '_x_')
    interaction_vars = [col for col in woe_vars if '_x_' in col]
    single_woe_vars = [col for col in woe_vars if '_x_' not in col]
    
    if woe_vars:
        print(f"Found {len(woe_vars)} WOE variables total:")
        print(f"  - {len(single_woe_vars)} single WOE variables")
        print(f"  - {len(interaction_vars)} interaction variables")
        if interaction_vars:
            print(f"  Interactions: {interaction_vars[:5]}{'...' if len(interaction_vars) > 5 else ''}")
        x_vars = woe_vars
    else:
        # Exclude b_ prefixed variables (binned, not for regression)
        x_vars = [col for col in x_vars if not col.startswith('b_')]
        print(f"Using {len(x_vars)} predictor variables")
    
    try:
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
        
        print(f"\nFinal model uses {len(result.selected_vars)} variables:")
        
        # Separate selected vars into single and interactions for clarity
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
        
        # Show what was dropped
        if interaction_vars:
            dropped_interactions = [v for v in interaction_vars if v not in result.selected_vars]
            if dropped_interactions:
                print(f"\n  Dropped interactions ({len(dropped_interactions)}):")
                for var in dropped_interactions[:10]:
                    print(f"    - {var}")
                if len(dropped_interactions) > 10:
                    print(f"    ... and {len(dropped_interactions) - 10} more")
        
    except Exception as e:
        print(f"ERROR fitting model: {e}")
        import traceback
        traceback.print_exc()
        
else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    if SHINY_AVAILABLE:
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        
        results = run_logistic_regression_ui(df)
        
        if results['completed']:
            coefficients = results['coefficients']
            predictions = results['predictions']
            dv = results['dv']
            print("Interactive session completed successfully")
        else:
            print("Interactive session cancelled - returning empty results")
    else:
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

# Ensure outputs are valid DataFrames
if coefficients is None:
    coefficients = pd.DataFrame()
if predictions is None:
    predictions = df.copy()

# Output 1: DataFrame with predictions (probabilities, predicted columns)
# Matches R: df$probabilities <- predict(model, df, type = "response") %>% round(6)
#            df$predicted <- ifelse(df$probabilities > 0.5, "1", "0")
knio.output_tables[0] = knio.Table.from_pandas(predictions)

# Output 2: Model coefficients table
# Matches R: as.data.frame(model$coefficients)
# Row ID = variable name (like R's row names), column = "model$coefficients"
coef_output = coefficients.copy()
if len(coef_output.columns) > 0:
    coef_output.columns = ['model$coefficients']
knio.output_tables[1] = knio.Table.from_pandas(coef_output)

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
sys.stdout.flush()

# Delete large objects to free memory
try:
    del df
except:
    pass

try:
    del predictions
except:
    pass

try:
    del coefficients
except:
    pass

# Force garbage collection
gc.collect()

# =============================================================================
# Scorecard Generator for KNIME Python Script Node - DEBUG TOGGLE VERSION
# =============================================================================
# This version has TOGGLEABLE debug logging. Set DEBUG_ENABLED to True/False
# to enable/disable all debug output without removing any code.
# 
# Python implementation matching R's scorecard creation functionality
# with Shiny UI for parameter configuration
# Compatible with KNIME 5.9, Python 3.9
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When Points, Odds, and PDO are provided via flow variables
#
# Inputs:
# 1. Coefficients table from Logistic Regression node (Output 2)
#    - Row ID = variable name (e.g., "(Intercept)", "WOE_Age")
#    - Column "model$coefficients" = coefficient value
# 2. Bins table from WOE Editor node (Output 4)
#    - var, bin, binValue, woe columns required
#
# Outputs:
# 1. Scorecard table - all bins with points (var, bin, woe, points columns)
#
# Flow Variables (for headless mode):
# - Points (int, default 600): Base score at target odds
# - Odds (int, default 20): Target odds ratio (1:Odds, e.g., 20 means 1:19)
# - PDO (int, default 50): Points to Double the Odds
#
# Scorecard Formula:
#   b = PDO / log(2)
#   a = Points + b * log(1/(Odds-1))
#   basepoints = a - b * intercept_coefficient
#   bin_points = round(-b * coefficient * woe, digits)
#
# Release Date: 2026-01-28
# Version: 1.0-DEBUG-TOGGLE
# =============================================================================

# =============================================================================
# DEBUG TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG_ENABLED = True  # <-- Change this to False to disable all debug logging
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from functools import wraps

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING CONFIGURATION
# =============================================================================

class DebugLogger:
    """Custom debug logger with indentation for nested function calls.
    Respects the DEBUG_ENABLED flag - all logging is skipped when disabled."""
    
    def __init__(self, name: str = "ScorecardDebug"):
        self.indent_level = 0
        self.indent_str = "  "
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S.%f')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_prefix(self) -> str:
        return self.indent_str * self.indent_level
    
    def enter_function(self, func_name: str, **kwargs):
        """Log function entry with parameters."""
        if not DEBUG_ENABLED:
            return
        self.logger.debug(f"{self._get_prefix()}>>> ENTER: {func_name}")
        self.indent_level += 1
        if kwargs:
            for key, value in kwargs.items():
                value_str = self._format_value(value)
                self.logger.debug(f"{self._get_prefix()}PARAM {key}: {value_str}")
    
    def exit_function(self, func_name: str, result=None):
        """Log function exit with return value."""
        if not DEBUG_ENABLED:
            return
        if result is not None:
            result_str = self._format_value(result)
            self.logger.debug(f"{self._get_prefix()}RETURN: {result_str}")
        self.indent_level = max(0, self.indent_level - 1)
        self.logger.debug(f"{self._get_prefix()}<<< EXIT: {func_name}")
    
    def log(self, message: str):
        """Log a general debug message."""
        if not DEBUG_ENABLED:
            return
        self.logger.debug(f"{self._get_prefix()}{message}")
    
    def log_var(self, name: str, value: Any):
        """Log a variable value."""
        if not DEBUG_ENABLED:
            return
        value_str = self._format_value(value)
        self.logger.debug(f"{self._get_prefix()}VAR {name} = {value_str}")
    
    def log_dataframe(self, name: str, df: pd.DataFrame, max_rows: int = 5):
        """Log DataFrame info and sample rows."""
        if not DEBUG_ENABLED:
            return
        if df is None:
            self.logger.debug(f"{self._get_prefix()}DF {name}: None")
            return
        self.logger.debug(f"{self._get_prefix()}DF {name}: shape={df.shape}, columns={list(df.columns)}")
        if len(df) > 0:
            sample = df.head(max_rows).to_string().replace('\n', f'\n{self._get_prefix()}   ')
            self.logger.debug(f"{self._get_prefix()}   {sample}")
    
    def log_error(self, message: str, exc: Exception = None):
        """Log an error message with optional exception."""
        if not DEBUG_ENABLED:
            return
        self.logger.error(f"{self._get_prefix()}ERROR: {message}")
        if exc:
            tb = traceback.format_exc()
            for line in tb.split('\n'):
                self.logger.error(f"{self._get_prefix()}  {line}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        if not DEBUG_ENABLED:
            return
        self.logger.warning(f"{self._get_prefix()}WARNING: {message}")
    
    def _format_value(self, value: Any, max_len: int = 200) -> str:
        """Format a value for logging, truncating if necessary."""
        if isinstance(value, pd.DataFrame):
            return f"DataFrame(shape={value.shape}, columns={list(value.columns)})"
        elif isinstance(value, pd.Series):
            return f"Series(len={len(value)}, dtype={value.dtype})"
        elif isinstance(value, (list, tuple)):
            if len(value) > 10:
                return f"{type(value).__name__}(len={len(value)}, first_5={value[:5]}...)"
            return str(value)
        elif isinstance(value, dict):
            if len(value) > 5:
                sample = dict(list(value.items())[:5])
                return f"dict(len={len(value)}, sample={sample}...)"
            return str(value)
        else:
            s = str(value)
            if len(s) > max_len:
                return s[:max_len] + "..."
            return s


# Create global debug logger instance
debug = DebugLogger()


def debug_trace(func):
    """Decorator to automatically trace function entry/exit.
    When DEBUG_ENABLED is False, this becomes a passthrough decorator."""
    if not DEBUG_ENABLED:
        # Return the function unchanged when debugging is disabled
        return func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Build parameter dict for logging
        param_dict = {}
        if args:
            param_dict['args'] = args
        if kwargs:
            param_dict.update(kwargs)
        
        debug.enter_function(func.__name__, **param_dict)
        try:
            result = func(*args, **kwargs)
            debug.exit_function(func.__name__, result)
            return result
        except Exception as e:
            debug.log_error(f"Exception in {func.__name__}: {e}", e)
            debug.exit_function(func.__name__)
            raise
    return wrapper


# =============================================================================
# Install/Import Dependencies
# =============================================================================

@debug_trace
def install_if_missing(package, import_name=None):
    """Install package if not available."""
    if import_name is None:
        import_name = package
    debug.log(f"Checking if package '{package}' (import as '{import_name}') is available")
    try:
        __import__(import_name)
        debug.log(f"Package '{package}' is already installed")
    except ImportError:
        debug.log(f"Package '{package}' not found, attempting to install...")
        import subprocess
        subprocess.check_call(['pip', 'install', package])
        debug.log(f"Package '{package}' installed successfully")

debug.log("=" * 70)
debug.log("SCORECARD GENERATOR DEBUG TOGGLE VERSION - Starting initialization")
debug.log(f"DEBUG_ENABLED = {DEBUG_ENABLED}")
debug.log("=" * 70)

install_if_missing('shiny')
install_if_missing('shinywidgets')

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    SHINY_AVAILABLE = True
    debug.log("Shiny imported successfully - interactive mode available")
except ImportError:
    debug.log_warning("Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Scorecard Creation Functions
# =============================================================================

@debug_trace
def calculate_ab(points0: float = 600, odds0: float = 1/19, pdo: float = 50) -> Tuple[float, float]:
    """
    Calculate scaling parameters a and b for scorecard.
    
    Parameters:
        points0: Base score at target odds
        odds0: Target odds ratio (as decimal, e.g., 1/19 for 1:19)
        pdo: Points to Double the Odds
        
    Returns:
        Tuple of (a, b) scaling parameters
    """
    debug.log(f"Calculating scaling parameters: points0={points0}, odds0={odds0:.6f}, pdo={pdo}")
    
    b = pdo / np.log(2)
    debug.log_var("b (pdo/log(2))", round(b, 6))
    
    log_odds0 = np.log(odds0)
    debug.log_var("log(odds0)", round(log_odds0, 6))
    
    a = points0 + b * log_odds0
    debug.log_var("a (points0 + b*log(odds0))", round(a, 6))
    
    return a, b


@debug_trace
def is_interaction_term(var_name: str) -> bool:
    """Check if a variable name represents an interaction term."""
    result = '_x_WOE_' in var_name or '_x_' in var_name
    debug.log(f"Checking if '{var_name}' is interaction term: {result}")
    return result


@debug_trace
def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term into its two component variable names.
    
    Input formats:
        - "WOE_var1_x_WOE_var2" -> ("var1", "var2")
        - "var1_x_WOE_var2" -> ("var1", "var2")
        - "var1_x_var2" -> ("var1", "var2")
    
    Returns:
        Tuple of (var1_name, var2_name)
    """
    debug.log(f"Parsing interaction term: '{var_name}'")
    
    # Remove leading WOE_ if present
    clean_name = var_name
    if clean_name.startswith('WOE_'):
        clean_name = clean_name[4:]  # Remove 'WOE_'
        debug.log(f"Removed WOE_ prefix, clean_name='{clean_name}'")
    
    # Split on '_x_WOE_' first (most specific pattern)
    if '_x_WOE_' in clean_name:
        parts = clean_name.split('_x_WOE_', 1)
        debug.log(f"Split on '_x_WOE_': var1='{parts[0]}', var2='{parts[1]}'")
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_'
    if '_x_' in clean_name:
        parts = clean_name.split('_x_', 1)
        var2 = parts[1]
        # Remove WOE_ prefix from var2 if present
        if var2.startswith('WOE_'):
            var2 = var2[4:]
        debug.log(f"Split on '_x_': var1='{parts[0]}', var2='{var2}'")
        return parts[0], var2
    
    debug.log_error(f"Cannot parse interaction term: {var_name}")
    raise ValueError(f"Cannot parse interaction term: {var_name}")


@debug_trace
def create_interaction_bins(
    bins: pd.DataFrame,
    var1: str,
    var2: str,
    interaction_name: str,
    coef: float,
    b: float,
    digits: int = 0
) -> List[Dict]:
    """
    Create scorecard entries for an interaction term.
    
    For interaction WOE_var1_x_WOE_var2:
    - Creates all combinations of var1 bins × var2 bins
    - Interaction WOE = woe1 × woe2
    - Points = round(-b × coefficient × woe1 × woe2)
    """
    debug.log(f"Creating interaction bins for: {interaction_name}")
    debug.log_var("var1", var1)
    debug.log_var("var2", var2)
    debug.log_var("coef", coef)
    debug.log_var("b", b)
    debug.log_var("digits", digits)
    
    rows = []
    
    # Get bins for each component variable
    var1_bins = bins[bins['var'] == var1].copy()
    var2_bins = bins[bins['var'] == var2].copy()
    
    debug.log(f"var1 '{var1}' bins count: {len(var1_bins)}")
    debug.log(f"var2 '{var2}' bins count: {len(var2_bins)}")
    
    if var1_bins.empty or var2_bins.empty:
        debug.log_warning(f"Cannot create interaction bins for {interaction_name}")
        debug.log_warning(f"  var1 '{var1}' has {len(var1_bins)} bins")
        debug.log_warning(f"  var2 '{var2}' has {len(var2_bins)} bins")
        return rows
    
    # Log combination count
    total_combinations = len(var1_bins) * len(var2_bins)
    debug.log(f"Total combinations to create: {total_combinations}")
    
    if total_combinations > 1000:
        debug.log_warning(f"Large interaction: {total_combinations:,} combinations may take time")
    
    # Create all combinations
    combination_count = 0
    for idx1, row1 in var1_bins.iterrows():
        woe1 = row1.get('woe', 0)
        if pd.isna(woe1):
            woe1 = 0
        bin1 = row1.get('binValue', row1.get('bin', ''))
        
        for idx2, row2 in var2_bins.iterrows():
            woe2 = row2.get('woe', 0)
            if pd.isna(woe2):
                woe2 = 0
            bin2 = row2.get('binValue', row2.get('bin', ''))
            
            # Interaction WOE is the product
            interaction_woe = woe1 * woe2
            
            # Calculate points
            points = round(-b * coef * interaction_woe, digits)
            
            # Create combined bin label
            combined_bin = f"{var1}:{bin1} × {var2}:{bin2}"
            
            rows.append({
                'var': interaction_name,
                'bin': combined_bin,
                'binValue': combined_bin,
                'woe': round(interaction_woe, 6),
                'points': points
            })
            
            combination_count += 1
            
            # Log every 100th combination for large interactions
            if total_combinations > 100 and combination_count % 100 == 0:
                debug.log(f"  Processed {combination_count}/{total_combinations} combinations...")
    
    debug.log(f"Created {len(rows)} interaction bin rows")
    return rows


@debug_trace
def create_scorecard(
    bins: pd.DataFrame,
    coefficients: pd.DataFrame,
    points0: float = 600,
    odds0: float = 1/19,
    pdo: float = 50,
    basepoints_eq0: bool = False,
    digits: int = 0
) -> pd.DataFrame:
    """
    Create a scorecard from binning rules and logistic regression coefficients.
    
    Handles both regular variables and interaction terms.
    """
    debug.log("=" * 60)
    debug.log("CREATE SCORECARD - Starting main scorecard creation")
    debug.log("=" * 60)
    
    debug.log_dataframe("bins", bins)
    debug.log_dataframe("coefficients", coefficients)
    debug.log_var("points0", points0)
    debug.log_var("odds0", odds0)
    debug.log_var("pdo", pdo)
    debug.log_var("basepoints_eq0", basepoints_eq0)
    debug.log_var("digits", digits)
    
    # Calculate scaling parameters
    debug.log("Step 1: Calculate scaling parameters a and b")
    a, b = calculate_ab(points0, odds0, pdo)
    debug.log_var("a", round(a, 6))
    debug.log_var("b", round(b, 6))
    
    # Prepare coefficients
    debug.log("Step 2: Process coefficients")
    coef_col = coefficients.columns[0] if len(coefficients.columns) > 0 else 'coefficients'
    debug.log_var("coef_col", coef_col)
    
    # Create coefficient lookup
    coef_dict = {}
    coef_dict_clean = {}
    intercept = 0.0
    
    debug.log("Processing coefficient rows:")
    for var_name, row in coefficients.iterrows():
        coef_value = row.iloc[0] if len(row) > 0 else row[coef_col]
        
        if var_name == '(Intercept)' or var_name.lower() == 'intercept':
            intercept = coef_value
            debug.log(f"  INTERCEPT: {coef_value}")
        else:
            coef_dict[var_name] = coef_value
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            coef_dict_clean[clean_var] = coef_value
            debug.log(f"  {var_name} -> {clean_var}: coef={coef_value}")
    
    # Calculate base points
    debug.log("Step 3: Calculate basepoints")
    if basepoints_eq0:
        basepoints = 0
        debug.log("basepoints_eq0=True, setting basepoints=0")
    else:
        basepoints = round(a - b * intercept, digits)
        debug.log(f"basepoints = round(a - b * intercept, {digits})")
        debug.log(f"           = round({a:.6f} - {b:.6f} * {intercept}, {digits})")
        debug.log(f"           = {basepoints}")
    
    # Create scorecard entries
    debug.log("Step 4: Create scorecard rows")
    scorecard_rows = []
    
    # Add basepoints row
    scorecard_rows.append({
        'var': 'basepoints',
        'bin': None,
        'binValue': None,
        'woe': None,
        'points': basepoints
    })
    debug.log(f"Added basepoints row: points={basepoints}")
    
    # Process each variable in bins
    bins_copy = bins.copy()
    
    # Ensure we have the required columns
    if 'var' not in bins_copy.columns:
        debug.log_error("Bins table must have 'var' column")
        raise ValueError("Bins table must have 'var' column")
    if 'woe' not in bins_copy.columns:
        debug.log_error("Bins table must have 'woe' column")
        raise ValueError("Bins table must have 'woe' column")
    
    # Separate regular variables from interaction terms
    debug.log("Step 5: Separate regular vars from interaction terms")
    regular_vars = []
    interaction_vars = []
    
    for var_name in coef_dict.keys():
        if is_interaction_term(var_name):
            interaction_vars.append(var_name)
            debug.log(f"  Interaction term: {var_name}")
        else:
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            regular_vars.append((var_name, clean_var))
            debug.log(f"  Regular var: {var_name} -> {clean_var}")
    
    debug.log(f"Regular variables: {len(regular_vars)}")
    debug.log(f"Interaction terms: {len(interaction_vars)}")
    
    # Process regular variables
    debug.log("Step 6: Process regular variables")
    for full_var, clean_var in regular_vars:
        debug.log(f"Processing variable: {clean_var}")
        
        if clean_var not in bins_copy['var'].unique():
            debug.log_warning(f"Variable '{clean_var}' not found in bins table")
            continue
            
        var_bins = bins_copy[bins_copy['var'] == clean_var].copy()
        coef = coef_dict[full_var]
        debug.log(f"  Found {len(var_bins)} bins, coef={coef}")
        
        bin_count = 0
        for _, row in var_bins.iterrows():
            woe = row.get('woe', 0)
            if pd.isna(woe):
                woe = 0
            
            # Calculate points: -b * coefficient * woe
            points = round(-b * coef * woe, digits)
            
            scorecard_rows.append({
                'var': clean_var,
                'bin': row.get('bin', None),
                'binValue': row.get('binValue', None),
                'woe': woe,
                'points': points
            })
            bin_count += 1
        
        debug.log(f"  Added {bin_count} bin rows for {clean_var}")
    
    # Process interaction terms
    debug.log("Step 7: Process interaction terms")
    for interaction_name in interaction_vars:
        debug.log(f"Processing interaction: {interaction_name}")
        try:
            var1, var2 = parse_interaction_term(interaction_name)
            coef = coef_dict[interaction_name]
            debug.log(f"  Components: var1={var1}, var2={var2}, coef={coef}")
            
            interaction_rows = create_interaction_bins(
                bins=bins_copy,
                var1=var1,
                var2=var2,
                interaction_name=f"{var1}_x_{var2}",
                coef=coef,
                b=b,
                digits=digits
            )
            
            scorecard_rows.extend(interaction_rows)
            debug.log(f"  Created {len(interaction_rows)} bins for interaction: {var1} × {var2}")
            
        except Exception as e:
            debug.log_error(f"Error processing interaction '{interaction_name}': {e}")
    
    debug.log("Step 8: Create final DataFrame")
    scorecard_df = pd.DataFrame(scorecard_rows)
    debug.log(f"Initial scorecard shape: {scorecard_df.shape}")
    
    # Reorder columns
    col_order = ['var', 'bin', 'binValue', 'woe', 'points']
    col_order = [c for c in col_order if c in scorecard_df.columns]
    scorecard_df = scorecard_df[col_order]
    
    debug.log(f"Final scorecard shape: {scorecard_df.shape}")
    debug.log_dataframe("scorecard_df (first 10 rows)", scorecard_df.head(10))
    
    debug.log("CREATE SCORECARD - Complete")
    return scorecard_df


@debug_trace
def create_scorecard_list(scorecard_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert scorecard DataFrame to list format (dictionary of DataFrames per variable).
    """
    debug.log(f"Converting scorecard to list format")
    debug.log(f"Input shape: {scorecard_df.shape}")
    
    card_list = {}
    
    for var in scorecard_df['var'].unique():
        var_df = scorecard_df[scorecard_df['var'] == var].copy()
        card_list[var] = var_df.reset_index(drop=True)
        debug.log(f"  {var}: {len(var_df)} rows")
    
    debug.log(f"Created list with {len(card_list)} variables")
    return card_list


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_scorecard_app(coefficients: pd.DataFrame, bins: pd.DataFrame):
    """Create the Scorecard Generator Shiny application."""
    debug.log("Creating Shiny application")
    debug.log_dataframe("coefficients", coefficients)
    debug.log_dataframe("bins", bins)
    
    app_results = {
        'scorecard': None,
        'completed': False
    }
    
    # Determine title based on debug mode
    title_text = "Scorecard Generator (DEBUG MODE)" if DEBUG_ENABLED else "Scorecard Generator"
    
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
                .btn-secondary { 
                    background: #95a5a6;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-secondary:hover {
                    background: #7f8c8d;
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
                .param-box {
                    background: #f8f9fa;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                }
                .metric-box {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 14px;
                    text-align: center;
                    border: 1px solid #e1e8ed;
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
                .scorecard-table-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: auto;
                    width: 100%;
                }
                .scorecard-table-container > div {
                    width: 100% !important;
                    min-width: 100% !important;
                }
                .scorecard-table-container table {
                    width: 100% !important;
                    min-width: 600px;
                    table-layout: fixed !important;
                }
                .scorecard-table-container th,
                .scorecard-table-container td {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    padding: 8px 12px;
                }
                .scorecard-table-container th:nth-child(1),
                .scorecard-table-container td:nth-child(1) {
                    width: 150px !important;
                    min-width: 150px !important;
                    max-width: 150px !important;
                }
                .scorecard-table-container th:nth-child(2),
                .scorecard-table-container td:nth-child(2) {
                    width: 250px !important;
                    min-width: 250px !important;
                    max-width: 250px !important;
                }
                .scorecard-table-container th:nth-child(3),
                .scorecard-table-container td:nth-child(3) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
                .scorecard-table-container th:nth-child(4),
                .scorecard-table-container td:nth-child(4) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
            """)
        ),
        
        ui.h3(title_text),
        
        # Configuration Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard Parameters"),
            ui.row(
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("points", "Base Points", value=600, min=0, step=50),
                        ui.tags.small("Score at target odds", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("odds", "Odds Ratio (1:X)", value=20, min=2, step=1),
                        ui.tags.small("Target odds (e.g., 20 = 1:19)", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("pdo", "Points to Double Odds", value=50, min=10, step=10),
                        ui.tags.small("PDO scaling factor", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_select("output_format", "Output Format", 
                                       choices=["Table", "List"],
                                       selected="Table"),
                        ui.tags.small("Scorecard output style", style="color: #7f8c8d;")
                    )
                )
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        {"style": "text-align: center; margin-top: 15px;"},
                        ui.input_action_button("analyze", "Generate Scorecard", class_="btn btn-primary btn-lg")
                    )
                )
            )
        ),
        
        # Summary Stats
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Summary"),
            ui.output_ui("summary_stats")
        ),
        
        # Scorecard Table Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard"),
            ui.div(
                {"class": "scorecard-table-container"},
                ui.output_data_frame("scorecard_table")
            )
        ),
        
        # Action Buttons
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run", "Run & Close", class_="btn btn-success btn-lg"),
            ui.input_action_button("close", "Close", class_="btn btn-secondary btn-lg")
        )
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        scorecard_rv = reactive.Value(None)
        
        @reactive.Effect
        @reactive.event(input.analyze)
        def generate_scorecard():
            debug.log("UI: Generate Scorecard button clicked")
            points = input.points() or 600
            odds = input.odds() or 20
            pdo = input.pdo() or 50
            
            debug.log(f"UI Parameters: points={points}, odds={odds}, pdo={pdo}")
            
            # Convert odds input (1:X format) to decimal
            odds_decimal = 1 / (odds - 1)
            debug.log(f"Calculated odds_decimal: {odds_decimal}")
            
            try:
                card = create_scorecard(
                    bins=bins,
                    coefficients=coefficients,
                    points0=points,
                    odds0=odds_decimal,
                    pdo=pdo,
                    basepoints_eq0=False,
                    digits=0
                )
                
                # Use binValue for display instead of bin
                if 'binValue' in card.columns:
                    card['bin'] = card['binValue']
                    card = card.drop(columns=['binValue'])
                
                scorecard_rv.set(card)
                debug.log(f"UI: Scorecard generated with {len(card)} rows")
                
            except Exception as e:
                debug.log_error(f"UI: Error generating scorecard: {e}", e)
        
        @output
        @render.ui
        def summary_stats():
            card = scorecard_rv.get()
            if card is None:
                return ui.p("Click 'Generate Scorecard' to view summary", style="text-align: center; color: #7f8c8d;")
            
            debug.log("UI: Rendering summary stats")
            
            # Calculate summary statistics
            num_vars = len([v for v in card['var'].unique() if v != 'basepoints'])
            total_bins = len(card) - 1  # Exclude basepoints row
            
            basepoints_row = card[card['var'] == 'basepoints']
            basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
            
            # Calculate min and max possible scores
            min_score = basepoints
            max_score = basepoints
            
            for var in card['var'].unique():
                if var == 'basepoints':
                    continue
                var_points = card[card['var'] == var]['points']
                if not var_points.empty:
                    min_score += var_points.min()
                    max_score += var_points.max()
            
            debug.log(f"UI Stats: vars={num_vars}, bins={total_bins}, range=[{min_score}, {max_score}]")
            
            return ui.div(
                {"class": "metrics-grid"},
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{num_vars}"),
                    ui.div({"class": "metric-label"}, "Variables")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{total_bins}"),
                    ui.div({"class": "metric-label"}, "Total Bins")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(min_score)}"),
                    ui.div({"class": "metric-label"}, "Min Score")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(max_score)}"),
                    ui.div({"class": "metric-label"}, "Max Score")
                )
            )
        
        @output
        @render.data_frame
        def scorecard_table():
            card = scorecard_rv.get()
            if card is None:
                return render.DataGrid(pd.DataFrame())
            
            debug.log(f"UI: Rendering scorecard table with {len(card)} rows")
            
            # Format for display
            display_df = card.copy()
            
            # Round woe for display
            if 'woe' in display_df.columns:
                display_df['woe'] = display_df['woe'].round(4)
            
            return render.DataGrid(display_df, height="450px", width="100%")
        
        @reactive.Effect
        @reactive.event(input.run)
        async def run_and_close():
            debug.log("UI: Run & Close button clicked")
            card = scorecard_rv.get()
            if card is not None:
                app_results['scorecard'] = card
                app_results['completed'] = True
                debug.log(f"UI: Results saved, scorecard has {len(card)} rows")
            await session.close()
        
        @reactive.Effect
        @reactive.event(input.close)
        async def close_app():
            debug.log("UI: Close button clicked (without saving)")
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    debug.log("Shiny app created successfully")
    return app


@debug_trace
def run_scorecard_ui(coefficients: pd.DataFrame, bins: pd.DataFrame, port: int = 8052):
    """Run the Scorecard Generator application and return results."""
    import threading
    import time
    import socket
    
    debug.log(f"Starting Shiny UI on port {port}")
    
    # Check if port is available
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except socket.error:
                return False
    
    if not is_port_available(port):
        debug.log_warning(f"Port {port} is already in use!")
        debug.log(f"Trying to use port {port+1} instead...")
        port = port + 1
        if not is_port_available(port):
            debug.log_error(f"Port {port} is also in use.")
    
    app = create_scorecard_app(coefficients, bins)
    
    # Run app in a separate thread
    def run_server():
        try:
            debug.log(f"Server thread starting on port {port}")
            print("=" * 70)
            print(f"Starting Shiny UI on http://127.0.0.1:{port}")
            print("=" * 70)
            print("IMPORTANT: A browser window should open automatically.")
            print("If it doesn't, manually open: http://127.0.0.1:{port}")
            print("")
            print("STEPS TO COMPLETE:")
            print("  1. Configure parameters in the browser UI")
            print("  2. Click 'Generate Scorecard' button")
            print("  3. Review the scorecard table")
            print("  4. Click 'Run & Close' button (green button at bottom)")
            print("")
            print("Waiting for you to complete the UI workflow...")
            print("=" * 70)
            app.run(port=port, launch_browser=True)
        except Exception as e:
            debug.log_error(f"Server stopped: {e}", e)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give the server time to start
    time.sleep(2)
    
    # Wait for the app to complete
    wait_count = 0
    while not app.results.get('completed', False):
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            debug.log(f"Still waiting for UI completion... ({wait_count} seconds elapsed)")
            print(f"Still waiting... ({wait_count} seconds elapsed)")
    
    time.sleep(0.5)
    debug.log("UI workflow completed, returning results")
    print("=" * 70)
    print("Scorecard generation complete - returning results")
    print("=" * 70)
    
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
debug.log("=" * 70)
debug.log("MAIN EXECUTION - Reading input data")
debug.log("=" * 70)

print("Scorecard Generator Node (DEBUG TOGGLE) - Starting...")
if DEBUG_ENABLED:
    print("DEBUG MODE: ENABLED - Verbose logging active")
else:
    print("DEBUG MODE: DISABLED - Minimal logging")
print("=" * 70)

# Input 1: Coefficients from Logistic Regression
debug.log("Reading Input 1: Coefficients table")
coefficients = knio.input_tables[0].to_pandas()
debug.log_dataframe("coefficients", coefficients)
print(f"Input 1 (Coefficients): {len(coefficients)} terms")

# Input 2: Bins from WOE Editor
debug.log("Reading Input 2: Bins table")
bins = knio.input_tables[1].to_pandas()
debug.log_dataframe("bins", bins)
print(f"Input 2 (Bins): {len(bins)} rows")

# Bins table summary
if 'var' in bins.columns:
    bins_per_var = bins.groupby('var').size()
    max_bins = bins_per_var.max()
    avg_bins = bins_per_var.mean()
    
    debug.log(f"Bins per variable stats: min={bins_per_var.min()}, avg={avg_bins:.1f}, max={max_bins}")
    print(f"\nBins per variable: min={bins_per_var.min()}, avg={avg_bins:.1f}, max={max_bins}")
    
    if max_bins > 20:
        debug.log("Variables with most bins:")
        print(f"\nVariables with most bins:")
        for var, count in bins_per_var.nlargest(5).items():
            debug.log(f"  - {var}: {count} bins")
            print(f"  - {var}: {count} bins")

# Debug: Show coefficient variable names
debug.log("Coefficient variable names:")
print("\nCoefficients:")
for var_name in coefficients.index:
    debug.log(f"  - {var_name}")
    print(f"  - {var_name}")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
debug.log("=" * 70)
debug.log("Checking for flow variables")
debug.log("=" * 70)

has_flow_vars = False
points = 600
odds = 20
pdo = 50
output_format = "Table"

try:
    points_fv = knio.flow_variables.get("Points", None)
    debug.log(f"Flow variable 'Points': {points_fv}")
    if points_fv is not None:
        points = int(points_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'Points' flow variable: {e}")

try:
    odds_fv = knio.flow_variables.get("Odds", None)
    debug.log(f"Flow variable 'Odds': {odds_fv}")
    if odds_fv is not None:
        odds = int(odds_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'Odds' flow variable: {e}")

try:
    pdo_fv = knio.flow_variables.get("PDO", None)
    debug.log(f"Flow variable 'PDO': {pdo_fv}")
    if pdo_fv is not None:
        pdo = int(pdo_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'PDO' flow variable: {e}")

try:
    output_format = knio.flow_variables.get("OutputFormat", "Table")
    debug.log(f"Flow variable 'OutputFormat': {output_format}")
except Exception as e:
    debug.log(f"Error reading 'OutputFormat' flow variable: {e}")

debug.log(f"has_flow_vars = {has_flow_vars}")
debug.log(f"Final parameters: Points={points}, Odds={odds}, PDO={pdo}")
print(f"\nParameters: Points={points}, Odds={odds}, PDO={pdo}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================
debug.log("=" * 70)
debug.log("MAIN PROCESSING")
debug.log("=" * 70)

scorecard = pd.DataFrame()

if has_flow_vars:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    debug.log("Running in HEADLESS mode")
    print("Running in HEADLESS mode")
    
    # Convert odds input (1:X format) to decimal
    odds_decimal = 1 / (odds - 1)
    debug.log(f"Calculated odds_decimal: {odds_decimal}")
    
    try:
        scorecard = create_scorecard(
            bins=bins,
            coefficients=coefficients,
            points0=points,
            odds0=odds_decimal,
            pdo=pdo,
            basepoints_eq0=False,
            digits=0
        )
        
        # Use binValue for display instead of bin
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])
        
        debug.log(f"Scorecard created successfully with {len(scorecard)} rows")
        print(f"\nScorecard created with {len(scorecard)} rows")
        
    except Exception as e:
        debug.log_error(f"Error creating scorecard: {e}", e)
        print(f"ERROR creating scorecard: {e}")
        import traceback
        traceback.print_exc()

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    if SHINY_AVAILABLE:
        debug.log("Running in INTERACTIVE mode - launching Shiny UI")
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        
        results = run_scorecard_ui(coefficients, bins)
        
        if results['completed']:
            scorecard = results['scorecard']
            debug.log("Interactive session completed successfully")
            print("Interactive session completed successfully")
        else:
            debug.log("Interactive session cancelled - returning empty results")
            print("Interactive session cancelled - returning empty results")
    else:
        debug.log_warning("Shiny not available, falling back to defaults")
        print("=" * 70)
        print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
        print("Please provide flow variables for headless mode:")
        print("  - Points (int): Base score at target odds, default 600")
        print("  - Odds (int): Odds ratio (1:X), default 20")
        print("  - PDO (int): Points to Double the Odds, default 50")
        print("=" * 70)
        
        # Run with defaults
        odds_decimal = 1 / (odds - 1)
        scorecard = create_scorecard(
            bins=bins,
            coefficients=coefficients,
            points0=points,
            odds0=odds_decimal,
            pdo=pdo,
            basepoints_eq0=False,
            digits=0
        )
        
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])

# =============================================================================
# Output Table
# =============================================================================
debug.log("=" * 70)
debug.log("WRITING OUTPUT")
debug.log("=" * 70)

# Ensure scorecard is a valid DataFrame
if scorecard is None or scorecard.empty:
    debug.log_warning("Scorecard is empty, creating empty DataFrame with columns")
    scorecard = pd.DataFrame(columns=['var', 'bin', 'woe', 'points'])

debug.log_dataframe("Final scorecard", scorecard.head(20))

# Output 1: Scorecard table
knio.output_tables[0] = knio.Table.from_pandas(scorecard)
debug.log(f"Output written to port 0: {len(scorecard)} rows")

# =============================================================================
# Print Summary
# =============================================================================
debug.log("=" * 70)
debug.log("SUMMARY")
debug.log("=" * 70)

print("=" * 70)
print("Scorecard Generator completed successfully")
print("=" * 70)

if not scorecard.empty:
    # Calculate score range
    basepoints_row = scorecard[scorecard['var'] == 'basepoints']
    basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
    
    min_score = basepoints
    max_score = basepoints
    
    for var in scorecard['var'].unique():
        if var == 'basepoints':
            continue
        var_points = scorecard[scorecard['var'] == var]['points']
        if not var_points.empty:
            min_score += var_points.min()
            max_score += var_points.max()
    
    num_vars = len([v for v in scorecard['var'].unique() if v != 'basepoints'])
    
    debug.log(f"Variables in scorecard: {num_vars}")
    debug.log(f"Base points: {int(basepoints)}")
    debug.log(f"Score range: {int(min_score)} to {int(max_score)}")
    
    print(f"Variables in scorecard: {num_vars}")
    print(f"Base points: {int(basepoints)}")
    print(f"Score range: {int(min_score)} to {int(max_score)}")
    
debug.log(f"Output (Scorecard): {len(scorecard)} rows")
print(f"\nOutput (Scorecard): {len(scorecard)} rows")
print("=" * 70)

debug.log("=" * 70)
debug.log("SCORECARD GENERATOR DEBUG TOGGLE VERSION - Complete")
debug.log("=" * 70)

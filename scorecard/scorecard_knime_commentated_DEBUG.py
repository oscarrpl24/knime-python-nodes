# =============================================================================
# Scorecard Generator for KNIME Python Script Node - COMMENTATED DEBUG VERSION
# =============================================================================
# This is the COMMENTATED DEBUG VERSION of scorecard_knime.py
# Every line of code is explained in plain English for educational purposes,
# AND extensive debug logging is enabled to trace execution flow.
# =============================================================================
#
# WHAT IS A SCORECARD?
# --------------------
# A scorecard is a points-based system used in credit risk modeling to convert
# a logistic regression model's predictions into an easy-to-understand score.
# Instead of dealing with probabilities and log-odds, users see simple integer
# points that sum up to a final credit score (like FICO scores).
#
# SCORECARD FORMULA EXPLAINED:
# ----------------------------
# The core formula converts logistic regression coefficients to points:
#
#   b = PDO / log(2)
#       - PDO = "Points to Double the Odds" (e.g., 50 points)
#       - This determines how many points it takes to double the odds of default
#       - log(2) ≈ 0.693, so b ≈ 72.13 when PDO=50
#
#   a = Points + b * log(1/(Odds-1))
#       - Points = base score at target odds (e.g., 600)
#       - Odds = target odds ratio (e.g., 1:19 means 1 bad for every 19 goods)
#       - This anchors the score scale
#
#   basepoints = a - b * intercept
#       - Distributes the intercept term into a fixed base score
#
#   bin_points = round(-b * coefficient * WOE)
#       - Converts each bin's WOE value into points using its coefficient
#       - Negative sign: higher WOE (riskier) = lower points
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided, opens a
#    browser-based interface where users can configure parameters
# 2. Headless - When Points, Odds, and PDO are provided via KNIME flow
#    variables, runs automatically without user interaction
#
# INPUTS (from KNIME workflow):
# 1. Coefficients table from Logistic Regression node (Output 2)
#    - Row ID = variable name (e.g., "(Intercept)", "WOE_Age")
#    - Column "model$coefficients" = coefficient value
# 2. Bins table from WOE Editor node (Output 4)
#    - var, bin, binValue, woe columns required
#
# OUTPUTS:
# 1. Scorecard table - all bins with points (var, bin, woe, points columns)
#
# Flow Variables (for headless mode):
# - Points (int, default 600): Base score at target odds
# - Odds (int, default 20): Target odds ratio (1:Odds, e.g., 20 means 1:19)
# - PDO (int, default 50): Points to Double the Odds
#
# Release Date: 2026-01-19
# Version: 1.0-COMMENTATED-DEBUG
# =============================================================================

# =============================================================================
# IMPORT SECTION
# =============================================================================
# This section imports all the libraries needed for the script to work.

# Import the KNIME scripting interface - this is how we read data from KNIME
# and write results back. 'knio' is the standard abbreviation used in KNIME
# Python nodes. Without this, the script cannot interact with the KNIME workflow.
import knime.scripting.io as knio

# Import pandas - the main library for working with tabular data in Python.
# 'pd' is the conventional abbreviation. We use it to:
# - Read and manipulate DataFrames (tables)
# - Filter rows, select columns, group data
# - Handle missing values (NA/NaN)
import pandas as pd

# Import numpy - the fundamental library for numerical computing in Python.
# 'np' is the conventional abbreviation. We use it for:
# - Mathematical functions like log() and round()
# - Array operations
# - Handling NaN values
import numpy as np

# Import warnings module - allows us to control warning messages.
# Some operations generate warnings that aren't actually problems,
# so we can suppress them to keep the output clean.
import warnings

# Import logging and related modules for extensive debug tracing
import logging
import sys
import traceback
from datetime import datetime

# Import type hints from the typing module - these don't affect runtime
# but help developers understand what types of data functions expect/return.
# Dict = dictionary (key-value pairs), List = list of items
# Tuple = fixed-length ordered collection, Optional = can be None
# Any = any type of data
from typing import Dict, List, Tuple, Optional, Any

# Import functools for the wraps decorator used in debug_trace
from functools import wraps

# Suppress all warning messages. This keeps the KNIME console clean by hiding
# non-critical warnings from pandas, numpy, and other libraries.
# Note: In production, you might want to be more selective about which
# warnings to ignore, but for this script it's safe to ignore all.
warnings.filterwarnings('ignore')


# =============================================================================
# DEBUG LOGGING CONFIGURATION
# =============================================================================
# This section sets up a custom debug logger that provides detailed tracing
# of function execution. The logger supports:
# - Indentation for nested function calls (visual hierarchy)
# - Function entry/exit tracking with parameters and return values
# - Variable value logging
# - DataFrame summary logging
# - Error and warning logging with stack traces

class DebugLogger:
    """
    Custom debug logger with indentation for nested function calls.
    
    This logger provides structured output that makes it easy to follow
    the execution flow through the codebase. Each nested function call
    increases the indentation level, creating a visual hierarchy.
    
    Example output:
        >>> ENTER: create_scorecard
          PARAM bins: DataFrame(shape=(50, 5), columns=['var', 'bin', ...])
          >>> ENTER: calculate_ab
            PARAM points0: 600
            VAR b = 72.134752
            RETURN: (387.6, 72.13)
          <<< EXIT: calculate_ab
        <<< EXIT: create_scorecard
    """
    
    def __init__(self, name: str = "ScorecardDebug"):
        """
        Initialize the debug logger.
        
        Parameters:
            name: Logger name (appears in log output for filtering)
        """
        # Track the current indentation level (increases with nested calls)
        self.indent_level = 0
        # String used for each level of indentation (2 spaces)
        self.indent_str = "  "
        # Create a Python logger instance with the given name
        self.logger = logging.getLogger(name)
        # Set to DEBUG level to capture all messages
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if not already present
        # This ensures we only add one handler even if this class is instantiated multiple times
        if not self.logger.handlers:
            # Create a handler that outputs to stdout (KNIME console)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            # Custom format: timestamp | message
            formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S.%f')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_prefix(self) -> str:
        """
        Get the current indentation prefix string.
        
        Returns:
            String of spaces based on current indent level
        """
        return self.indent_str * self.indent_level
    
    def enter_function(self, func_name: str, **kwargs):
        """
        Log function entry with parameters.
        
        This should be called at the start of a function to log
        that we're entering it and what parameters were passed.
        
        Parameters:
            func_name: Name of the function being entered
            **kwargs: Named parameters to log
        """
        # Log the function entry with >>> arrow marker
        self.logger.debug(f"{self._get_prefix()}>>> ENTER: {func_name}")
        # Increase indent for the function body
        self.indent_level += 1
        # Log each parameter if any were provided
        if kwargs:
            for key, value in kwargs.items():
                # Format the value (truncate if too long)
                value_str = self._format_value(value)
                self.logger.debug(f"{self._get_prefix()}PARAM {key}: {value_str}")
    
    def exit_function(self, func_name: str, result=None):
        """
        Log function exit with return value.
        
        This should be called when exiting a function to log
        the return value and mark the end of the function.
        
        Parameters:
            func_name: Name of the function being exited
            result: The return value (optional)
        """
        # Log the return value if provided
        if result is not None:
            result_str = self._format_value(result)
            self.logger.debug(f"{self._get_prefix()}RETURN: {result_str}")
        # Decrease indent (min 0 to prevent negative indent)
        self.indent_level = max(0, self.indent_level - 1)
        # Log the function exit with <<< arrow marker
        self.logger.debug(f"{self._get_prefix()}<<< EXIT: {func_name}")
    
    def log(self, message: str):
        """
        Log a general debug message.
        
        Parameters:
            message: The message to log
        """
        self.logger.debug(f"{self._get_prefix()}{message}")
    
    def log_var(self, name: str, value: Any):
        """
        Log a variable value.
        
        This is useful for tracking intermediate values during computation.
        
        Parameters:
            name: Variable name
            value: Variable value
        """
        value_str = self._format_value(value)
        self.logger.debug(f"{self._get_prefix()}VAR {name} = {value_str}")
    
    def log_dataframe(self, name: str, df: pd.DataFrame, max_rows: int = 5):
        """
        Log DataFrame info and sample rows.
        
        This provides a summary of a DataFrame without dumping all data.
        Shows shape, columns, and first few rows.
        
        Parameters:
            name: Name to identify this DataFrame
            df: The DataFrame to log
            max_rows: Maximum number of sample rows to show
        """
        if df is None:
            self.logger.debug(f"{self._get_prefix()}DF {name}: None")
            return
        # Log summary: shape and column names
        self.logger.debug(f"{self._get_prefix()}DF {name}: shape={df.shape}, columns={list(df.columns)}")
        # If DataFrame has data, show first few rows
        if len(df) > 0:
            # Convert to string and add indentation to each line
            sample = df.head(max_rows).to_string().replace('\n', f'\n{self._get_prefix()}   ')
            self.logger.debug(f"{self._get_prefix()}   {sample}")
    
    def log_error(self, message: str, exc: Exception = None):
        """
        Log an error message with optional exception traceback.
        
        Parameters:
            message: Error description
            exc: Optional exception object
        """
        self.logger.error(f"{self._get_prefix()}ERROR: {message}")
        if exc:
            # Get the full traceback and log each line
            tb = traceback.format_exc()
            for line in tb.split('\n'):
                self.logger.error(f"{self._get_prefix()}  {line}")
    
    def log_warning(self, message: str):
        """
        Log a warning message.
        
        Parameters:
            message: Warning description
        """
        self.logger.warning(f"{self._get_prefix()}WARNING: {message}")
    
    def _format_value(self, value: Any, max_len: int = 200) -> str:
        """
        Format a value for logging, truncating if necessary.
        
        Different types get different formatting:
        - DataFrames: show shape and columns
        - Series: show length and dtype
        - Lists/Tuples: show length and first few items if large
        - Dicts: show length and sample if large
        - Others: convert to string, truncate if too long
        
        Parameters:
            value: The value to format
            max_len: Maximum string length before truncation
            
        Returns:
            Formatted string representation
        """
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
# This is used throughout the script for consistent logging
debug = DebugLogger()


def debug_trace(func):
    """
    Decorator to automatically trace function entry/exit.
    
    Apply this decorator to any function to automatically log:
    - Function entry with all parameters
    - Function exit with return value
    - Any exceptions that occur
    
    Usage:
        @debug_trace
        def my_function(param1, param2):
            return result
    
    This is a Python decorator - it wraps the original function with
    additional logging behavior without modifying the function itself.
    """
    @wraps(func)  # Preserves the original function's name and docstring
    def wrapper(*args, **kwargs):
        # Build parameter dict for logging
        param_dict = {}
        if args:
            param_dict['args'] = args
        if kwargs:
            param_dict.update(kwargs)
        
        # Log function entry
        debug.enter_function(func.__name__, **param_dict)
        try:
            # Call the original function
            result = func(*args, **kwargs)
            # Log successful exit with return value
            debug.exit_function(func.__name__, result)
            return result
        except Exception as e:
            # Log error and re-raise the exception
            debug.log_error(f"Exception in {func.__name__}: {e}", e)
            debug.exit_function(func.__name__)
            raise
    return wrapper


# =============================================================================
# Install/Import Dependencies Section
# =============================================================================
# This section handles optional dependencies that might not be installed.
# The Shiny library is used for the interactive web UI, but it's not always
# available or needed (headless mode doesn't require it).

@debug_trace
def install_if_missing(package, import_name=None):
    """
    Attempt to install a Python package if it's not already installed.
    
    This function tries to import a package, and if that fails (ImportError),
    it uses pip to install the package automatically. This is useful for
    KNIME environments where packages might not be pre-installed.
    
    Parameters:
        package: The pip package name to install (e.g., 'shiny')
        import_name: The Python import name if different from package name
                     (e.g., package='PIL' but import_name='pillow')
                     If None, uses the package name for both.
    
    WARNING: Auto-installing packages can be a security risk in some
    environments. In production, packages should be pre-installed.
    """
    # If no separate import name is provided, use the package name for both
    # pip installation and Python import (they're usually the same)
    if import_name is None:
        import_name = package
    
    debug.log(f"Checking if package '{package}' (import as '{import_name}') is available")
    
    # Try to import the package to check if it's already installed
    try:
        # __import__ is Python's built-in function to import by string name
        # This is equivalent to "import shiny" but allows dynamic names
        __import__(import_name)
        # If we get here, the import succeeded - package is installed
        debug.log(f"Package '{package}' is already installed")
    except ImportError:
        # ImportError means the package is not installed
        debug.log(f"Package '{package}' not found, attempting to install...")
        # Use subprocess to run pip install command
        import subprocess
        # subprocess.check_call runs a command and raises error if it fails
        # ['pip', 'install', package] is equivalent to: pip install package
        subprocess.check_call(['pip', 'install', package])
        debug.log(f"Package '{package}' installed successfully")


# Print startup banner
debug.log("=" * 70)
debug.log("SCORECARD GENERATOR COMMENTATED DEBUG VERSION - Starting initialization")
debug.log("=" * 70)

# Try to install the 'shiny' package if not present
# Shiny is a web application framework that creates interactive UIs
# Originally from R, now available for Python too
install_if_missing('shiny')

# Try to install 'shinywidgets' if not present
# shinywidgets provides additional UI components for Shiny apps
install_if_missing('shinywidgets')

# Now try to actually import the Shiny components we need
# This is wrapped in try/except because even after installation,
# imports can sometimes fail due to environment issues
try:
    # Import specific components from the shiny package:
    # - App: The main application class that combines UI and server logic
    # - Inputs: Gives access to user input values from the UI
    # - Outputs: Allows setting output values to display in the UI
    # - Session: Represents a single user's browser session
    # - reactive: Decorators for reactive programming (automatic updates)
    # - render: Decorators for rendering outputs (tables, text, etc.)
    # - ui: Functions to build the user interface (buttons, inputs, etc.)
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Set flag to True indicating Shiny is available for use
    SHINY_AVAILABLE = True
    debug.log("Shiny imported successfully - interactive mode available")
    
except ImportError:
    # If import fails, print a warning and set flag to False
    # The script will still work but only in headless mode
    debug.log_warning("Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Scorecard Creation Functions
# =============================================================================
# These are the core mathematical functions that convert logistic regression
# coefficients and WOE bins into scorecard points.

@debug_trace
def calculate_ab(points0: float = 600, odds0: float = 1/19, pdo: float = 50) -> Tuple[float, float]:
    """
    Calculate the 'a' and 'b' scaling parameters for the scorecard formula.
    
    These parameters translate the logistic regression's log-odds scale
    into a user-friendly point scale. The formulas are standard in credit
    scoring and come from solving two equations:
    
    1. At odds = odds0, score should equal points0
    2. When odds double, score should change by PDO points
    
    The mathematical derivation:
    - Score = a + b * log(odds)  [linear in log-odds]
    - At odds0: points0 = a + b * log(odds0)  ... equation 1
    - At 2*odds0: points0 + PDO = a + b * log(2*odds0)  ... equation 2
    
    Subtracting equation 1 from 2:
    - PDO = b * log(2*odds0) - b * log(odds0)
    - PDO = b * [log(2) + log(odds0) - log(odds0)]
    - PDO = b * log(2)
    - Therefore: b = PDO / log(2)
    
    Substituting back:
    - a = points0 - b * log(odds0)
    - But we use: a = points0 + b * log(odds0) because of sign convention
    
    Parameters:
        points0: The base score when odds equal the target odds (default 600)
                 This is the "anchor point" of the scorecard scale.
                 Common values: 600, 660, 700
        odds0: The target odds ratio as a decimal (default 1/19 ≈ 0.0526)
               1/19 means "1 bad outcome for every 19 good outcomes"
               Also expressed as "1:19 odds" or "20:1 goods to bads"
        pdo: Points to Double the Odds (default 50)
             How many points it takes to double the odds of being "good"
             Higher PDO = more spread in scores, finer granularity
             Common values: 20, 50, 100
        
    Returns:
        Tuple containing:
        - a (float): The intercept scaling parameter
        - b (float): The slope scaling parameter (coefficient multiplier)
        
    Example:
        With default values:
        - b = 50 / log(2) ≈ 72.13
        - a = 600 + 72.13 * log(1/19) ≈ 600 + 72.13 * (-2.944) ≈ 387.6
    """
    debug.log(f"Calculating scaling parameters: points0={points0}, odds0={odds0:.6f}, pdo={pdo}")
    
    # Calculate b: the scaling factor for the slope
    # np.log() is the natural logarithm (base e)
    # np.log(2) ≈ 0.693, so if PDO=50, then b ≈ 72.13
    # This means each unit of log-odds converts to about 72 points
    b = pdo / np.log(2)
    debug.log_var("b (pdo/log(2))", round(b, 6))
    
    # Calculate log of odds for the next step
    log_odds0 = np.log(odds0)
    debug.log_var("log(odds0)", round(log_odds0, 6))
    
    # Calculate a: the intercept that anchors the scale
    # This ensures that at odds = odds0, the score equals points0
    # Note: We add (not subtract) because odds0 < 1 gives negative log
    a = points0 + b * log_odds0
    debug.log_var("a (points0 + b*log(odds0))", round(a, 6))
    
    # Return both values as a tuple for the caller to use
    return a, b


@debug_trace
def is_interaction_term(var_name: str) -> bool:
    """
    Determine if a variable name represents an interaction term.
    
    In logistic regression, interaction terms capture how two variables
    jointly affect the outcome beyond their individual effects.
    For example, if Age and Income have an interaction, the effect of
    Age on default risk depends on the Income level (and vice versa).
    
    In WOE-based scorecards, interaction terms are created by multiplying
    the WOE values of two variables: WOE_interaction = WOE_var1 × WOE_var2
    
    Interaction term naming conventions:
    - "WOE_Age_x_WOE_Income" - both variables have WOE_ prefix
    - "Age_x_WOE_Income" - first variable might not have prefix
    - "Age_x_Income" - neither has prefix (less common)
    
    Parameters:
        var_name: The variable name to check (string)
        
    Returns:
        True if the name contains interaction pattern, False otherwise
        
    Examples:
        is_interaction_term("WOE_Age") -> False (regular variable)
        is_interaction_term("WOE_Age_x_WOE_Income") -> True (interaction)
        is_interaction_term("Age_x_Income") -> True (interaction)
    """
    # Check for the most specific pattern first: '_x_WOE_'
    # This appears in names like "WOE_Age_x_WOE_Income"
    # OR check for the more general pattern: '_x_'
    # This appears in names like "Age_x_Income"
    # The '_x_' separator is a common convention for denoting interactions
    result = '_x_WOE_' in var_name or '_x_' in var_name
    debug.log(f"Checking if '{var_name}' is interaction term: {result}")
    return result


@debug_trace
def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term name into its two component variable names.
    
    This function takes a combined interaction name and extracts the
    individual variable names that make up the interaction. It handles
    various naming formats that might come from the logistic regression.
    
    The function must handle these input formats:
    - "WOE_var1_x_WOE_var2" -> returns ("var1", "var2")
    - "var1_x_WOE_var2" -> returns ("var1", "var2")
    - "var1_x_var2" -> returns ("var1", "var2")
    
    Note: Variable names themselves might contain underscores, but we
    rely on the '_x_' or '_x_WOE_' patterns to identify the split point.
    
    Parameters:
        var_name: The full interaction term name (string)
        
    Returns:
        Tuple of (var1_name, var2_name) as strings
        
    Raises:
        ValueError: If the name doesn't match any known interaction pattern
        
    Example:
        parse_interaction_term("WOE_Age_x_WOE_Income")
        -> Returns ("Age", "Income")
    """
    debug.log(f"Parsing interaction term: '{var_name}'")
    
    # Start with the full name - we'll clean it up step by step
    clean_name = var_name
    
    # First, remove the leading 'WOE_' prefix if present
    # Many interaction terms start with "WOE_" from the logistic regression
    # but we need the base variable name to look up in the bins table
    if clean_name.startswith('WOE_'):
        # Slice the string starting at position 4 to remove 'WOE_'
        # 'WOE_Age' becomes 'Age'
        clean_name = clean_name[4:]
        debug.log(f"Removed WOE_ prefix, clean_name='{clean_name}'")
    
    # Try splitting on '_x_WOE_' first - this is the most specific pattern
    # This handles cases like "Age_x_WOE_Income" after WOE_ removal
    # (original: "WOE_Age_x_WOE_Income")
    if '_x_WOE_' in clean_name:
        # split('_x_WOE_', 1) splits only on the first occurrence
        # and returns at most 2 parts
        # "Age_x_WOE_Income" -> ["Age", "Income"]
        parts = clean_name.split('_x_WOE_', 1)
        debug.log(f"Split on '_x_WOE_': var1='{parts[0]}', var2='{parts[1]}'")
        # Return first part and second part as the two variable names
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_' - more general pattern
    # This handles cases like "Age_x_Income"
    if '_x_' in clean_name:
        # Split on '_x_' to get the two parts
        parts = clean_name.split('_x_', 1)
        var2 = parts[1]
        
        # The second variable might still have 'WOE_' prefix
        # (e.g., if original was "Age_x_WOE_Income" but we didn't match above)
        if var2.startswith('WOE_'):
            # Remove the WOE_ prefix from var2 as well
            var2 = var2[4:]
        
        debug.log(f"Split on '_x_': var1='{parts[0]}', var2='{var2}'")
        # Return both cleaned variable names
        return parts[0], var2
    
    # If we get here, the name doesn't match any known pattern
    # This is an error case - raise an exception with helpful message
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
    
    For an interaction between two variables, we need to create scorecard
    rows for every combination of their bins. If var1 has 5 bins and var2
    has 4 bins, we create 5 × 4 = 20 interaction bin combinations.
    
    The interaction WOE is calculated as: woe_interaction = woe1 × woe2
    This captures how the combined bin effects multiply together.
    
    The points formula is the same as regular variables:
    points = round(-b × coefficient × WOE)
    But WOE is now the product of two WOE values.
    
    Example:
        If Age bin "[25,35)" has WOE = 0.3
        And Income bin "[50k,75k)" has WOE = 0.2
        Then interaction WOE = 0.3 × 0.2 = 0.06
        If coefficient = 0.5 and b = 72.13
        Then points = round(-72.13 × 0.5 × 0.06) = round(-2.16) = -2
    
    Parameters:
        bins: DataFrame containing the binning rules for all variables
              Must have 'var', 'bin'/'binValue', and 'woe' columns
        var1: Name of the first component variable (without WOE_ prefix)
        var2: Name of the second component variable (without WOE_ prefix)
        interaction_name: Display name for the interaction (e.g., "Age_x_Income")
        coef: The logistic regression coefficient for this interaction term
        b: The scaling parameter from calculate_ab()
        digits: Number of decimal places for rounding points (default 0 = integers)
        
    Returns:
        List of dictionaries, each representing one row in the scorecard.
        Each dict has keys: 'var', 'bin', 'binValue', 'woe', 'points'
        
    Note:
        If either component variable is not found in the bins table,
        returns an empty list and prints a warning.
    """
    debug.log(f"Creating interaction bins for: {interaction_name}")
    debug.log_var("var1", var1)
    debug.log_var("var2", var2)
    debug.log_var("coef", coef)
    debug.log_var("b", b)
    debug.log_var("digits", digits)
    
    # Initialize empty list to collect scorecard rows
    rows = []
    
    # Get the bins for the first variable from the bins DataFrame
    # Filter to only rows where 'var' column equals var1
    # .copy() creates a new DataFrame to avoid modifying the original
    var1_bins = bins[bins['var'] == var1].copy()
    
    # Same for the second variable
    var2_bins = bins[bins['var'] == var2].copy()
    
    debug.log(f"var1 '{var1}' bins count: {len(var1_bins)}")
    debug.log(f"var2 '{var2}' bins count: {len(var2_bins)}")
    
    # Check if we found bins for both variables
    # .empty is True if the DataFrame has no rows
    if var1_bins.empty or var2_bins.empty:
        # Print warning with details about what's missing
        debug.log_warning(f"Cannot create interaction bins for {interaction_name}")
        debug.log_warning(f"  var1 '{var1}' has {len(var1_bins)} bins")
        debug.log_warning(f"  var2 '{var2}' has {len(var2_bins)} bins")
        # Return empty list - no bins to create
        return rows
    
    # Calculate total number of combinations (for informational logging)
    # This is the Cartesian product: n1 × n2
    total_combinations = len(var1_bins) * len(var2_bins)
    debug.log(f"Total combinations to create: {total_combinations}")
    
    # If there are many combinations, print an info message
    # 1000 is arbitrary threshold - just to warn about potentially slow processing
    if total_combinations > 1000:
        debug.log_warning(f"Large interaction: {total_combinations:,} combinations may take time")
    
    # Create all combinations using nested loops
    # Outer loop: iterate over each bin of variable 1
    combination_count = 0
    for _, row1 in var1_bins.iterrows():
        # iterrows() returns index and row as a tuple
        # We use _ for the index because we don't need it
        
        # Get the WOE value for this bin
        # .get() with default 0 handles the case where 'woe' column might not exist
        woe1 = row1.get('woe', 0)
        
        # Handle missing WOE values (NaN/NA)
        # pd.isna() returns True if the value is NaN, None, or NA
        if pd.isna(woe1):
            woe1 = 0
        
        # Get the bin label for display
        # Try 'binValue' first (preferred display name), fall back to 'bin'
        bin1 = row1.get('binValue', row1.get('bin', ''))
        
        # Inner loop: iterate over each bin of variable 2
        for _, row2 in var2_bins.iterrows():
            # Get WOE for var2's bin
            woe2 = row2.get('woe', 0)
            if pd.isna(woe2):
                woe2 = 0
            
            # Get bin label for var2
            bin2 = row2.get('binValue', row2.get('bin', ''))
            
            # Calculate the interaction WOE as the product of individual WOEs
            # This is the standard approach for WOE interactions
            # Mathematically: log-odds contribution = coef × woe1 × woe2
            interaction_woe = woe1 * woe2
            
            # Calculate points using the standard formula
            # Negative sign: higher WOE (riskier) = fewer points
            # b: scaling factor from calculate_ab()
            # coef: logistic regression coefficient
            # round(..., digits): round to specified decimal places
            points = round(-b * coef * interaction_woe, digits)
            
            # Create a combined bin label that shows both components
            # Format: "Age:[25,35) × Income:[50k,75k)"
            # This makes it clear which combination of bins this row represents
            combined_bin = f"{var1}:{bin1} × {var2}:{bin2}"
            
            # Append a dictionary with all the scorecard columns
            # This will become one row in the final scorecard DataFrame
            rows.append({
                'var': interaction_name,      # The interaction name (e.g., "Age_x_Income")
                'bin': combined_bin,          # Combined bin label
                'binValue': combined_bin,     # Same as bin (for consistency)
                'woe': round(interaction_woe, 6),  # WOE rounded to 6 decimal places
                'points': points              # The calculated points (integer if digits=0)
            })
            
            combination_count += 1
            
            # Log every 100th combination for large interactions
            if total_combinations > 100 and combination_count % 100 == 0:
                debug.log(f"  Processed {combination_count}/{total_combinations} combinations...")
    
    debug.log(f"Created {len(rows)} interaction bin rows")
    # Return the list of all combination rows
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
    Create a complete scorecard from binning rules and logistic regression coefficients.
    
    This is the main function that orchestrates the entire scorecard creation process.
    It handles both regular variables and interaction terms.
    
    The process:
    1. Calculate scaling parameters (a, b) from the scorecard settings
    2. Parse coefficients and identify the intercept
    3. Calculate base points from the intercept
    4. For each regular variable: calculate points for each bin
    5. For each interaction term: create all bin combinations with points
    6. Combine everything into a single scorecard DataFrame
    
    Parameters:
        bins: DataFrame with binning rules from WOE Editor
              Required columns: 'var', 'woe'
              Optional columns: 'bin', 'binValue'
        coefficients: DataFrame with model coefficients from Logistic Regression
                      Index = variable name (e.g., "(Intercept)", "WOE_Age")
                      First column = coefficient value
        points0: Base score at target odds (default 600)
        odds0: Target odds ratio as decimal (default 1/19)
        pdo: Points to Double the Odds (default 50)
        basepoints_eq0: If True, force basepoints to 0 (default False)
                        Used when you want to distribute all points to variables
        digits: Number of decimal places for rounding points (default 0)
        
    Returns:
        DataFrame with columns: ['var', 'bin', 'binValue', 'woe', 'points']
        First row is always 'basepoints' with the base score
        Remaining rows are the bins for each variable with their points
        
    Example output:
        var         bin           woe     points
        basepoints  None          None    450
        Age         [18,25)       -0.5    25
        Age         [25,35)       0.0     0
        Age         [35,50)       0.3     -15
        Income      [0,30k)       -0.8    40
        ...
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
    
    # Step 1: Calculate the scaling parameters a and b
    # These convert log-odds to points using the scorecard formula
    debug.log("Step 1: Calculate scaling parameters a and b")
    a, b = calculate_ab(points0, odds0, pdo)
    debug.log_var("a", round(a, 6))
    debug.log_var("b", round(b, 6))
    
    # Step 2: Prepare coefficients
    # The coefficients DataFrame has variable names as index and values in columns
    # Get the name of the first column (usually "model$coefficients" or similar)
    debug.log("Step 2: Process coefficients")
    coef_col = coefficients.columns[0] if len(coefficients.columns) > 0 else 'coefficients'
    debug.log_var("coef_col", coef_col)
    
    # Create dictionaries to look up coefficients by variable name
    # coef_dict: keeps original names (with WOE_ prefix) for matching
    # coef_dict_clean: strips WOE_ prefix for matching with bins table
    coef_dict = {}         # Full variable names -> coefficient values
    coef_dict_clean = {}   # Cleaned variable names -> coefficient values
    intercept = 0.0        # Will store the intercept coefficient
    
    # Iterate through each row of the coefficients DataFrame
    # .iterrows() yields (index, row) pairs
    # The index is the variable name (e.g., "(Intercept)", "WOE_Age")
    debug.log("Processing coefficient rows:")
    for var_name, row in coefficients.iterrows():
        # Get the coefficient value from the first column
        # row.iloc[0] gets the first value in the row (by position)
        # Fall back to row[coef_col] if iloc fails
        coef_value = row.iloc[0] if len(row) > 0 else row[coef_col]
        
        # Check if this is the intercept term
        # R uses "(Intercept)" with parentheses, so we check for both formats
        if var_name == '(Intercept)' or var_name.lower() == 'intercept':
            # Store the intercept separately - it gets special treatment
            intercept = coef_value
            debug.log(f"  INTERCEPT: {coef_value}")
        else:
            # Regular variable - store in both dictionaries
            coef_dict[var_name] = coef_value
            
            # Create a "clean" version with WOE_ prefix removed
            # This helps match with the bins table which uses base names
            # "WOE_Age" -> "Age"
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            coef_dict_clean[clean_var] = coef_value
            debug.log(f"  {var_name} -> {clean_var}: coef={coef_value}")
    
    # Step 3: Calculate base points
    # The intercept from logistic regression gets converted to base points
    debug.log("Step 3: Calculate basepoints")
    if basepoints_eq0:
        # If user wants basepoints = 0, set it directly
        # (All points will be distributed to the variable bins)
        basepoints = 0
        debug.log("basepoints_eq0=True, setting basepoints=0")
    else:
        # Standard formula: basepoints = a - b × intercept
        # This anchors the score scale based on the model's intercept
        basepoints = round(a - b * intercept, digits)
        debug.log(f"basepoints = round(a - b * intercept, {digits})")
        debug.log(f"           = round({a:.6f} - {b:.6f} * {intercept}, {digits})")
        debug.log(f"           = {basepoints}")
    
    # Step 4: Initialize the list that will hold all scorecard rows
    debug.log("Step 4: Create scorecard rows")
    scorecard_rows = []
    
    # Add the basepoints row first
    # This is a special row that represents the starting score before
    # any variable contributions are added
    scorecard_rows.append({
        'var': 'basepoints',  # Special variable name
        'bin': None,          # No bin for basepoints
        'binValue': None,     # No bin value for basepoints
        'woe': None,          # No WOE for basepoints
        'points': basepoints  # The base score value
    })
    debug.log(f"Added basepoints row: points={basepoints}")
    
    # Step 5: Process each variable in the bins table
    # Make a copy to avoid modifying the original DataFrame
    bins_copy = bins.copy()
    
    # Validate that required columns exist in the bins table
    # These columns are essential for scorecard creation
    if 'var' not in bins_copy.columns:
        debug.log_error("Bins table must have 'var' column")
        raise ValueError("Bins table must have 'var' column")
    if 'woe' not in bins_copy.columns:
        debug.log_error("Bins table must have 'woe' column")
        raise ValueError("Bins table must have 'woe' column")
    
    # Step 6: Separate regular variables from interaction terms
    # They need different processing logic
    debug.log("Step 5: Separate regular vars from interaction terms")
    regular_vars = []      # Will hold tuples of (full_name, clean_name)
    interaction_vars = []  # Will hold interaction term names
    
    # Iterate through all variables that have coefficients
    for var_name in coef_dict.keys():
        # Check if this is an interaction term
        if is_interaction_term(var_name):
            # Add to interaction list for later processing
            interaction_vars.append(var_name)
            debug.log(f"  Interaction term: {var_name}")
        else:
            # Regular variable - strip WOE_ prefix for matching with bins
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            # Store both names as a tuple: (original_for_coefficient, cleaned_for_bins)
            regular_vars.append((var_name, clean_var))
            debug.log(f"  Regular var: {var_name} -> {clean_var}")
    
    debug.log(f"Regular variables: {len(regular_vars)}")
    debug.log(f"Interaction terms: {len(interaction_vars)}")
    
    # Step 7: Process regular (non-interaction) variables
    debug.log("Step 6: Process regular variables")
    for full_var, clean_var in regular_vars:
        debug.log(f"Processing variable: {clean_var}")
        
        # Check if this variable exists in the bins table
        # .unique() gets distinct values in the 'var' column
        if clean_var not in bins_copy['var'].unique():
            # Variable not found - print warning and skip
            debug.log_warning(f"Variable '{clean_var}' not found in bins table")
            continue
        
        # Get all bins for this variable from the bins table
        var_bins = bins_copy[bins_copy['var'] == clean_var].copy()
        
        # Get the coefficient for this variable
        coef = coef_dict[full_var]
        debug.log(f"  Found {len(var_bins)} bins, coef={coef}")
        
        # Iterate through each bin of this variable
        bin_count = 0
        for _, row in var_bins.iterrows():
            # Get the WOE value for this bin
            woe = row.get('woe', 0)
            
            # Handle missing WOE values
            if pd.isna(woe):
                woe = 0
            
            # Calculate points for this bin
            # Formula: points = round(-b × coefficient × WOE)
            # Negative sign: higher WOE (riskier) = fewer points
            points = round(-b * coef * woe, digits)
            
            # Add this bin as a row in the scorecard
            scorecard_rows.append({
                'var': clean_var,                      # Variable name (cleaned)
                'bin': row.get('bin', None),           # Bin label (e.g., "[25,35)")
                'binValue': row.get('binValue', None), # Bin display value
                'woe': woe,                            # WOE value for this bin
                'points': points                       # Calculated points
            })
            bin_count += 1
        
        debug.log(f"  Added {bin_count} bin rows for {clean_var}")
    
    # Step 8: Process interaction terms
    debug.log("Step 7: Process interaction terms")
    for interaction_name in interaction_vars:
        debug.log(f"Processing interaction: {interaction_name}")
        try:
            # Parse the interaction name to get component variable names
            # e.g., "WOE_Age_x_WOE_Income" -> ("Age", "Income")
            var1, var2 = parse_interaction_term(interaction_name)
            
            # Get the coefficient for this interaction term
            coef = coef_dict[interaction_name]
            debug.log(f"  Components: var1={var1}, var2={var2}, coef={coef}")
            
            # Create all combination bins for this interaction
            # This function handles the nested loop logic
            interaction_rows = create_interaction_bins(
                bins=bins_copy,
                var1=var1,
                var2=var2,
                interaction_name=f"{var1}_x_{var2}",  # Clean display name
                coef=coef,
                b=b,
                digits=digits
            )
            
            # Add all interaction rows to the main list
            # extend() adds each item from the list individually
            scorecard_rows.extend(interaction_rows)
            
            # Print info about how many bins were created
            debug.log(f"  Created {len(interaction_rows)} bins for interaction: {var1} × {var2}")
            
        except Exception as e:
            # If anything goes wrong, print error and continue with other variables
            debug.log_error(f"Error processing interaction '{interaction_name}': {e}")
    
    # Step 9: Convert the list of dictionaries to a DataFrame
    debug.log("Step 8: Create final DataFrame")
    scorecard_df = pd.DataFrame(scorecard_rows)
    debug.log(f"Initial scorecard shape: {scorecard_df.shape}")
    
    # Step 10: Reorder columns to a standard order
    # This ensures consistent output regardless of how rows were added
    col_order = ['var', 'bin', 'binValue', 'woe', 'points']
    
    # Only include columns that actually exist in the DataFrame
    # (in case some optional columns weren't created)
    col_order = [c for c in col_order if c in scorecard_df.columns]
    
    # Reorder the DataFrame columns
    scorecard_df = scorecard_df[col_order]
    
    debug.log(f"Final scorecard shape: {scorecard_df.shape}")
    debug.log_dataframe("scorecard_df (first 10 rows)", scorecard_df.head(10))
    
    debug.log("CREATE SCORECARD - Complete")
    # Return the complete scorecard DataFrame
    return scorecard_df


@debug_trace
def create_scorecard_list(scorecard_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert a scorecard DataFrame to a list format (dictionary of DataFrames).
    
    This is an alternative output format where instead of one big table,
    you get a separate DataFrame for each variable. This can be useful for:
    - Displaying scorecards in a more compact format
    - Processing variables independently
    - Creating per-variable reports
    
    Parameters:
        scorecard_df: The scorecard DataFrame from create_scorecard()
        
    Returns:
        Dictionary where:
        - Keys are variable names (strings)
        - Values are DataFrames containing just that variable's bins
        
    Example:
        Input scorecard_df:
            var         bin           woe    points
            basepoints  None          None   450
            Age         [18,25)       -0.5   25
            Age         [25,35)       0.0    0
            Income      [0,30k)       -0.8   40
            
        Output:
            {
                'basepoints': DataFrame with 1 row (basepoints),
                'Age': DataFrame with 2 rows (Age bins),
                'Income': DataFrame with 1 row (Income bins)
            }
    """
    debug.log(f"Converting scorecard to list format")
    debug.log(f"Input shape: {scorecard_df.shape}")
    
    # Initialize empty dictionary to hold the results
    card_list = {}
    
    # Iterate through each unique variable name in the scorecard
    # .unique() returns array of distinct values
    for var in scorecard_df['var'].unique():
        # Filter to get only rows for this variable
        var_df = scorecard_df[scorecard_df['var'] == var].copy()
        
        # Reset the index so each mini-DataFrame has clean 0-based indices
        # drop=True means don't keep the old index as a column
        card_list[var] = var_df.reset_index(drop=True)
        debug.log(f"  {var}: {len(var_df)} rows")
    
    debug.log(f"Created list with {len(card_list)} variables")
    # Return the dictionary of DataFrames
    return card_list


# =============================================================================
# Shiny UI Application
# =============================================================================
# This section defines the interactive web-based user interface using Shiny.
# Shiny is a reactive web framework that automatically updates outputs when
# inputs change. Originally from R, now available for Python.
# 
# The UI and server code are extensively logged to trace user interactions.

def create_scorecard_app(coefficients: pd.DataFrame, bins: pd.DataFrame):
    """
    Create the Scorecard Generator Shiny application.
    
    This function defines both the UI layout and the server logic for an
    interactive scorecard generation tool. Users can:
    1. Adjust scorecard parameters (Points, Odds, PDO)
    2. Generate a preview of the scorecard
    3. View summary statistics
    4. Save results and close the app
    
    The app uses reactive programming: when users change inputs, the
    dependent outputs automatically update.
    
    Parameters:
        coefficients: DataFrame with model coefficients (passed to create_scorecard)
        bins: DataFrame with binning rules (passed to create_scorecard)
        
    Returns:
        A Shiny App object that can be run with app.run()
        The app has a 'results' attribute to retrieve the final scorecard
    """
    debug.log("Creating Shiny application")
    debug.log_dataframe("coefficients", coefficients)
    debug.log_dataframe("bins", bins)
    
    # Dictionary to store results from the app
    # This is how we get data back after the user closes the app
    app_results = {
        'scorecard': None,    # Will hold the final scorecard DataFrame
        'completed': False    # Flag to indicate if user clicked "Run & Close"
    }
    
    # ==========================================================================
    # Define the User Interface (UI) Layout
    # ==========================================================================
    # ui.page_fluid() creates a responsive page that adjusts to screen width
    app_ui = ui.page_fluid(
        # Add custom CSS styles in the page header
        ui.tags.head(
            ui.tags.style("""
                /* Import Google Font for a clean, professional look */
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                
                /* Base body styles - light gray background, dark text */
                body { 
                    font-family: 'Source Sans Pro', sans-serif; 
                    background: #f5f7fa;
                    min-height: 100vh;
                    color: #2c3e50;
                }
                
                /* Card component styles - white background with shadow */
                .card { 
                    background: #ffffff;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                /* Card header with blue underline */
                .card-header {
                    color: #2c3e50;
                    font-weight: 700;
                    font-size: 1.1rem;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }
                
                /* Main heading styles */
                h3 { 
                    color: #2c3e50; 
                    text-align: center; 
                    font-weight: 700;
                    margin-bottom: 24px;
                }
                
                /* Primary button (blue) */
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
                
                /* Success button (green) - for "Run & Close" */
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
                
                /* Secondary button (gray) - for "Close" */
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
                
                /* Form input styling */
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
                
                /* Form labels */
                .form-label {
                    color: #2c3e50;
                    font-weight: 600;
                }
                
                /* Parameter input boxes */
                .param-box {
                    background: #f8f9fa;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                }
                
                /* Metric display boxes */
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
                
                /* Grid layout for metrics */
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                    margin-top: 12px;
                }
                
                /* Scorecard table container with scroll */
                .scorecard-table-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: auto;
                    width: 100%;
                }
                
                /* Table width fixes */
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
                
                /* Fixed column widths for scorecard table */
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
        
        # Page title - centered heading (indicates DEBUG MODE)
        ui.h3("Scorecard Generator (COMMENTATED DEBUG MODE)"),
        
        # ==========================================================================
        # Configuration Card - Parameter Inputs
        # ==========================================================================
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard Parameters"),
            
            # Row with 4 columns for input parameters
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
            
            # Row for the Generate button
            ui.row(
                ui.column(12,
                    ui.div(
                        {"style": "text-align: center; margin-top: 15px;"},
                        ui.input_action_button("analyze", "Generate Scorecard", class_="btn btn-primary btn-lg")
                    )
                )
            )
        ),
        
        # ==========================================================================
        # Summary Stats Card
        # ==========================================================================
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Summary"),
            ui.output_ui("summary_stats")
        ),
        
        # ==========================================================================
        # Scorecard Table Card
        # ==========================================================================
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard"),
            ui.div(
                {"class": "scorecard-table-container"},
                ui.output_data_frame("scorecard_table")
            )
        ),
        
        # ==========================================================================
        # Action Buttons Card
        # ==========================================================================
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run", "Run & Close", class_="btn btn-success btn-lg"),
            ui.input_action_button("close", "Close", class_="btn btn-secondary btn-lg")
        )
    )
    
    # ==========================================================================
    # Define the Server Logic
    # ==========================================================================
    def server(input: Inputs, output: Outputs, session: Session):
        # Create a reactive value to hold the current scorecard
        scorecard_rv = reactive.Value(None)
        
        # ==========================
        # Generate Scorecard Handler
        # ==========================
        @reactive.Effect
        @reactive.event(input.analyze)
        def generate_scorecard():
            """Generate the scorecard when button is clicked."""
            debug.log("UI: Generate Scorecard button clicked")
            
            # Get current values from inputs, with defaults if empty
            points = input.points() or 600
            odds = input.odds() or 20
            pdo = input.pdo() or 50
            
            debug.log(f"UI Parameters: points={points}, odds={odds}, pdo={pdo}")
            
            # Convert odds input (1:X format) to decimal
            odds_decimal = 1 / (odds - 1)
            debug.log(f"Calculated odds_decimal: {odds_decimal}")
            
            try:
                # Call the scorecard creation function
                card = create_scorecard(
                    bins=bins,
                    coefficients=coefficients,
                    points0=points,
                    odds0=odds_decimal,
                    pdo=pdo,
                    basepoints_eq0=False,
                    digits=0
                )
                
                # Use binValue for display instead of bin column
                if 'binValue' in card.columns:
                    card['bin'] = card['binValue']
                    card = card.drop(columns=['binValue'])
                
                # Update the reactive value
                scorecard_rv.set(card)
                debug.log(f"UI: Scorecard generated with {len(card)} rows")
                
            except Exception as e:
                debug.log_error(f"UI: Error generating scorecard: {e}", e)
        
        # ==========================
        # Summary Statistics Renderer
        # ==========================
        @output
        @render.ui
        def summary_stats():
            """Render the summary statistics panel."""
            card = scorecard_rv.get()
            
            if card is None:
                return ui.p("Click 'Generate Scorecard' to view summary", 
                           style="text-align: center; color: #7f8c8d;")
            
            debug.log("UI: Rendering summary stats")
            
            # Calculate summary statistics
            num_vars = len([v for v in card['var'].unique() if v != 'basepoints'])
            total_bins = len(card) - 1
            
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
            
            # Return the UI with metrics grid
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
        
        # ==========================
        # Scorecard Table Renderer
        # ==========================
        @output
        @render.data_frame
        def scorecard_table():
            """Render the scorecard table."""
            card = scorecard_rv.get()
            
            if card is None:
                return render.DataGrid(pd.DataFrame())
            
            debug.log(f"UI: Rendering scorecard table with {len(card)} rows")
            
            # Create a copy for display formatting
            display_df = card.copy()
            
            # Round WOE values for cleaner display
            if 'woe' in display_df.columns:
                display_df['woe'] = display_df['woe'].round(4)
            
            return render.DataGrid(display_df, height="450px", width="100%")
        
        # ==========================
        # Run & Close Button Handler
        # ==========================
        @reactive.Effect
        @reactive.event(input.run)
        async def run_and_close():
            """Save results and close the application."""
            debug.log("UI: Run & Close button clicked")
            card = scorecard_rv.get()
            
            if card is not None:
                app_results['scorecard'] = card
                app_results['completed'] = True
                debug.log(f"UI: Results saved, scorecard has {len(card)} rows")
            
            await session.close()
        
        # ==========================
        # Close Button Handler
        # ==========================
        @reactive.Effect
        @reactive.event(input.close)
        async def close_app():
            """Close the application without saving."""
            debug.log("UI: Close button clicked (without saving)")
            await session.close()
    
    # Create the Shiny App object
    app = App(app_ui, server)
    
    # Attach the results dictionary to the app object
    app.results = app_results
    
    debug.log("Shiny app created successfully")
    return app


@debug_trace
def run_scorecard_ui(coefficients: pd.DataFrame, bins: pd.DataFrame, port: int = 8052):
    """
    Run the Scorecard Generator application and wait for user to complete.
    
    This function:
    1. Creates the Shiny app
    2. Starts it in a background thread
    3. Opens the user's browser
    4. Waits for the user to click "Run & Close"
    5. Returns the results
    
    Parameters:
        coefficients: DataFrame with model coefficients
        bins: DataFrame with binning rules
        port: HTTP port to run the app on (default 8052)
        
    Returns:
        Dictionary with:
        - 'scorecard': The generated scorecard DataFrame (or None)
        - 'completed': Boolean indicating if user completed the workflow
    """
    import threading
    import time
    import socket
    
    debug.log(f"Starting Shiny UI on port {port}")
    
    def is_port_available(port):
        """Check if a TCP port is available for use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except socket.error:
                return False
    
    # Check if our desired port is available
    if not is_port_available(port):
        debug.log_warning(f"Port {port} is already in use!")
        debug.log(f"Trying to use port {port+1} instead...")
        port = port + 1
        
        if not is_port_available(port):
            debug.log_error(f"Port {port} is also in use.")
    
    # Create the Shiny app
    app = create_scorecard_app(coefficients, bins)
    
    def run_server():
        """Function to run the Shiny server in a separate thread."""
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
    
    # Create and start server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give the server time to start up
    time.sleep(2)
    
    # Wait loop - check if user has completed the workflow
    wait_count = 0
    
    while not app.results.get('completed', False):
        time.sleep(1)
        wait_count += 1
        
        if wait_count % 10 == 0:
            debug.log(f"Still waiting for UI completion... ({wait_count} seconds elapsed)")
            print(f"Still waiting... ({wait_count} seconds elapsed)")
    
    # Give a moment for cleanup
    time.sleep(0.5)
    
    debug.log("UI workflow completed, returning results")
    print("=" * 70)
    print("Scorecard generation complete - returning results")
    print("=" * 70)
    
    return app.results


# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================
# This section runs when the script is executed by KNIME.
# It reads input data, determines the mode, and generates the scorecard.
# Extensive debug logging traces every step of the process.

debug.log("=" * 70)
debug.log("MAIN EXECUTION - Reading input data")
debug.log("=" * 70)

# Print startup message to KNIME console
print("Scorecard Generator Node (COMMENTATED DEBUG) - Starting...")
print("=" * 70)

# =============================================================================
# Read Input Data from KNIME
# =============================================================================

# Read Input 1: Coefficients table
debug.log("Reading Input 1: Coefficients table")
coefficients = knio.input_tables[0].to_pandas()
debug.log_dataframe("coefficients", coefficients)
print(f"Input 1 (Coefficients): {len(coefficients)} terms")

# Read Input 2: Bins table
debug.log("Reading Input 2: Bins table")
bins = knio.input_tables[1].to_pandas()
debug.log_dataframe("bins", bins)
print(f"Input 2 (Bins): {len(bins)} rows")

# Calculate and display bins per variable statistics
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

# Print the coefficient variable names for debugging
debug.log("Coefficient variable names:")
print("\nCoefficients:")
for var_name in coefficients.index:
    debug.log(f"  - {var_name}")
    print(f"  - {var_name}")

# =============================================================================
# Check for Flow Variables (Headless Mode Detection)
# =============================================================================
debug.log("=" * 70)
debug.log("Checking for flow variables")
debug.log("=" * 70)

# Initialize flag and default values
has_flow_vars = False
points = 600
odds = 20
pdo = 50
output_format = "Table"

# Try to read "Points" flow variable
try:
    points_fv = knio.flow_variables.get("Points", None)
    debug.log(f"Flow variable 'Points': {points_fv}")
    if points_fv is not None:
        points = int(points_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'Points' flow variable: {e}")

# Try to read "Odds" flow variable
try:
    odds_fv = knio.flow_variables.get("Odds", None)
    debug.log(f"Flow variable 'Odds': {odds_fv}")
    if odds_fv is not None:
        odds = int(odds_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'Odds' flow variable: {e}")

# Try to read "PDO" flow variable
try:
    pdo_fv = knio.flow_variables.get("PDO", None)
    debug.log(f"Flow variable 'PDO': {pdo_fv}")
    if pdo_fv is not None:
        pdo = int(pdo_fv)
        has_flow_vars = True
except Exception as e:
    debug.log(f"Error reading 'PDO' flow variable: {e}")

# Try to read "OutputFormat" flow variable
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

# Initialize empty DataFrame to hold the scorecard result
scorecard = pd.DataFrame()

# Check if we should run in headless mode
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
        # Shiny not available
        debug.log_warning("Shiny not available, falling back to defaults")
        print("=" * 70)
        print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
        print("Please provide flow variables for headless mode:")
        print("  - Points (int): Base score at target odds, default 600")
        print("  - Odds (int): Odds ratio (1:X), default 20")
        print("  - PDO (int): Points to Double the Odds, default 50")
        print("=" * 70)
        
        # Fall back to running with default parameters
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
        
        # Clean up binValue column
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])

# =============================================================================
# Output Table to KNIME
# =============================================================================
debug.log("=" * 70)
debug.log("WRITING OUTPUT")
debug.log("=" * 70)

# Ensure scorecard is a valid DataFrame with expected columns
if scorecard is None or scorecard.empty:
    debug.log_warning("Scorecard is empty, creating empty DataFrame with columns")
    scorecard = pd.DataFrame(columns=['var', 'bin', 'woe', 'points'])

debug.log_dataframe("Final scorecard", scorecard.head(20))

# Write the scorecard to KNIME's first output port
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

# Only print statistics if we have a non-empty scorecard
if not scorecard.empty:
    # Get the basepoints value
    basepoints_row = scorecard[scorecard['var'] == 'basepoints']
    basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
    
    # Calculate score range
    min_score = basepoints
    max_score = basepoints
    
    for var in scorecard['var'].unique():
        if var == 'basepoints':
            continue
        var_points = scorecard[scorecard['var'] == var]['points']
        if not var_points.empty:
            min_score += var_points.min()
            max_score += var_points.max()
    
    # Count variables
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
debug.log("SCORECARD GENERATOR COMMENTATED DEBUG VERSION - Complete")
debug.log("=" * 70)

# =============================================================================
# END OF SCRIPT
# =============================================================================
# The scorecard is now available in KNIME for further processing or export.
# Connect this node's output to:
# - A CSV Writer to save the scorecard
# - The Scorecard Apply node to score new data
# - A Table View to inspect the results
# =============================================================================

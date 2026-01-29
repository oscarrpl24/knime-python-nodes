# =============================================================================
# WOE Editor for KNIME Python Script Node - FULLY COMMENTATED VERSION
# =============================================================================
# 
# PURPOSE:
# This script implements a Weight of Evidence (WOE) Editor for credit risk modeling.
# WOE is a technique used to transform categorical and continuous variables into
# a format suitable for logistic regression in credit scoring models.
#
# WOE measures the "strength" of a grouping technique to separate good and bad accounts.
# Positive WOE means the bin has more "bads" (defaults) than average.
# Negative WOE means the bin has more "goods" (non-defaults) than average.
#
# This is a Python implementation that matches the functionality of R's WOE Editor.
# It is designed to run inside KNIME 5.9 using the Python Script node.
#
# The script supports two operational modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided, launches a web UI
# 2. Headless - When DependentVariable flow variable is set, runs automatically
#
# OUTPUT PORTS (5 total):
# 1. Original input DataFrame (unchanged) - Pass-through of input data
# 2. df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# 3. df_only_woe - Only WOE columns + dependent variable (for logistic regression)
# 4. df_only_bins - ONLY binned columns (b_*) for scorecard scoring (LEAN!)
# 5. bins - Binning rules with WOE values (metadata for scorecard creation)
#
# Release Date: 2026-01-15
# Version: 1.0
# =============================================================================

# =============================================================================
# IMPORT SECTION - Loading Required Libraries
# =============================================================================

# Import KNIME's Python scripting interface
# This module provides access to input/output tables and flow variables
import knime.scripting.io as knio

# Import pandas for DataFrame manipulation
# pandas is the primary data structure library for tabular data in Python
import pandas as pd

# Import numpy for numerical operations
# numpy provides efficient array operations and mathematical functions
import numpy as np

# Import re for regular expression pattern matching
# Used to parse bin rules and extract numeric values from strings
import re

# Import warnings module to control warning messages
# We'll suppress warnings to keep the output clean
import warnings

# Import time module for timing operations and progress tracking
# Used to measure how long each step takes
import time

# Import sys for system-level operations
# Used to flush stdout for real-time progress updates
import sys

# Import os for operating system interactions
# Used to set environment variables for thread control
import os

# Import random for generating random numbers
# Used to randomize port numbers to avoid conflicts
import random

# Import typing module for type hints
# These make the code more readable and help with IDE autocomplete
from typing import Dict, List, Tuple, Optional, Any, Union

# Import dataclass decorator for creating simple data container classes
# dataclass automatically generates __init__, __repr__, etc.
from dataclasses import dataclass

# Suppress all warnings to keep output clean
# This prevents numpy/pandas deprecation warnings from cluttering the console
warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# These settings prevent conflicts when multiple KNIME nodes run simultaneously

# BASE_PORT: Starting port number for the Shiny web UI
# Shiny needs a TCP port to serve the web interface
BASE_PORT = 8050

# RANDOM_PORT_RANGE: Range of random port offsets to add to BASE_PORT
# If port 8050 is busy, we'll try a random port between 8050 and 9050
RANDOM_PORT_RANGE = 1000

# INSTANCE_ID: Unique identifier for this script instance
# Combines process ID (unique per running process) with random number
# Used to create unique temporary directories if needed
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Set environment variables to limit threading in numerical libraries
# This prevents threading conflicts when multiple KNIME nodes run in parallel
# Each library has its own environment variable:

# NUMEXPR_MAX_THREADS: Limits threads for numexpr (used by pandas internally)
os.environ['NUMEXPR_MAX_THREADS'] = '1'

# OMP_NUM_THREADS: Limits OpenMP threads (used by many scientific libraries)
os.environ['OMP_NUM_THREADS'] = '1'

# OPENBLAS_NUM_THREADS: Limits OpenBLAS threads (linear algebra library)
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# MKL_NUM_THREADS: Limits Intel MKL threads (Intel's math library)
os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# Progress Logging Utilities
# =============================================================================
# These functions help track progress during long-running operations


def log_progress(message: str, flush: bool = True):
    """
    Print a progress message with a timestamp prefix.
    
    This function prints messages in a consistent format with the current time,
    making it easy to track when each step occurred and how long operations take.
    
    Parameters:
        message (str): The message to print to the console
        flush (bool): If True, immediately flush the output buffer
                      This ensures the message appears immediately, not buffered
    
    Example output: "[14:32:05] Processing variable Age..."
    """
    # Get current time formatted as HH:MM:SS
    timestamp = time.strftime("%H:%M:%S")
    
    # Print the timestamp in brackets followed by the message
    print(f"[{timestamp}] {message}")
    
    # If flush is True, force Python to write the output immediately
    # Without this, output might be buffered and not appear until later
    if flush:
        sys.stdout.flush()


def format_time(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string.
    
    This function takes a number of seconds and returns a formatted string
    that's easier to read, using appropriate units (seconds, minutes, hours).
    
    Parameters:
        seconds (float): Duration in seconds (can be fractional)
    
    Returns:
        str: Formatted duration like "45.2s", "3m 25s", or "2h 15m"
    
    Examples:
        format_time(45.2) -> "45.2s"
        format_time(185) -> "3m 5s"
        format_time(7500) -> "2h 5m"
    """
    # If less than 60 seconds, show seconds with one decimal place
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    # If less than 3600 seconds (1 hour), show minutes and seconds
    elif seconds < 3600:
        # Integer division to get whole minutes
        mins = int(seconds // 60)
        # Modulo to get remaining seconds
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    
    # If 1 hour or more, show hours and minutes
    else:
        # Integer division by 3600 to get hours
        hours = int(seconds // 3600)
        # Modulo 3600 gives remaining seconds, then divide by 60 for minutes
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# =============================================================================
# Install/Import Dependencies
# =============================================================================
# This section ensures required libraries are installed before importing them
# Using try/except pattern: try to import, if that fails, install then import

# Try to import scikit-learn's DecisionTreeClassifier
# This is used to find optimal split points for numeric variables
try:
    # Attempt to import DecisionTreeClassifier from sklearn
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    # If import fails (library not installed), install it using pip
    import subprocess
    # subprocess.check_call runs a command and raises exception if it fails
    # We're running: pip install scikit-learn
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    # Now that it's installed, import it again
    from sklearn.tree import DecisionTreeClassifier

# Try to import Shiny and related libraries for the interactive UI
try:
    # Import core Shiny components for building the web application
    # App: Main application class
    # Inputs: Container for all input values from UI widgets
    # Outputs: Container for all output renderers
    # Session: Represents a user's browser session
    # reactive: For creating reactive values and effects
    # render: For rendering outputs (tables, text, etc.)
    # ui: For building the user interface layout
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Import shinywidgets for Plotly integration in Shiny
    # render_plotly: Decorator to render Plotly figures
    # output_widget: UI placeholder for widget outputs
    from shinywidgets import render_plotly, output_widget
    
    # Import Plotly's graph_objects for creating interactive charts
    import plotly.graph_objects as go
    
except ImportError:
    # If any of the above imports fail, install all three packages
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    
    # Now import them again after installation
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Data Classes
# =============================================================================
# Data classes are simple classes that mainly hold data
# The @dataclass decorator automatically generates __init__ and other methods


@dataclass
class BinResult:
    """
    Container class for binning results.
    
    This class holds the output of the binning process, which consists of:
    1. A summary of each variable (IV, entropy, trend, etc.)
    2. Detailed bin information (bin rules, counts, WOE values, etc.)
    
    Using a dataclass makes it easy to pass around both pieces of data together
    and provides a clean interface for accessing them.
    
    Attributes:
        var_summary (pd.DataFrame): Summary statistics for each variable
            Columns: var, varType, iv, ent, trend, monTrend, flipRatio, numBins, purNode
        bin (pd.DataFrame): Detailed bin information for all variables
            Columns: var, bin, count, bads, goods, propn, bad_rate, iv, ent, etc.
    """
    # DataFrame containing summary statistics for each variable
    # One row per variable with aggregated metrics
    var_summary: pd.DataFrame
    
    # DataFrame containing detailed bin information
    # Multiple rows per variable (one per bin plus a Total row)
    bin: pd.DataFrame


# =============================================================================
# Core Binning Functions
# =============================================================================
# These functions implement the core WOE binning logic


def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray) -> np.ndarray:
    """
    Calculate Weight of Evidence (WOE) for each bin.
    
    WOE Formula: WOE = ln((% of Bads in bin) / (% of Goods in bin))
    
    WOE measures how much more likely a bin is to contain "bads" vs "goods".
    - Positive WOE: More bads than expected (higher risk bin)
    - Negative WOE: More goods than expected (lower risk bin)
    - Zero WOE: Proportion matches overall (neutral risk)
    
    Parameters:
        freq_good (np.ndarray): Array of good counts for each bin
        freq_bad (np.ndarray): Array of bad counts for each bin
    
    Returns:
        np.ndarray: Array of WOE values for each bin, rounded to 5 decimal places
    
    Example:
        If a bin has 10% of all goods but 20% of all bads:
        WOE = ln(0.20 / 0.10) = ln(2) ≈ 0.693 (high risk bin)
    """
    # Convert inputs to float arrays to ensure proper division
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate total goods and bads across all bins
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # Handle edge case: if no goods or no bads at all, WOE is undefined
    # Return zeros (neutral) in this case
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    # Calculate distribution of goods across bins (what % of goods are in each bin)
    dist_good = freq_good / total_good
    
    # Calculate distribution of bads across bins (what % of bads are in each bin)
    dist_bad = freq_bad / total_bad
    
    # Replace zeros with small value (0.0001) to avoid division by zero in log
    # This is a standard technique in WOE calculation
    dist_good = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    # Calculate WOE using the formula: ln(dist_bad / dist_good)
    # Round to 5 decimal places for cleaner output
    woe = np.round(np.log(dist_bad / dist_good), 5)
    
    return woe


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """
    Calculate Information Value (IV) for a variable.
    
    IV Formula: IV = Σ (dist_good - dist_bad) * WOE
    
    IV measures the overall predictive power of a variable:
    - IV < 0.02: Not useful for prediction
    - 0.02 ≤ IV < 0.1: Weak predictor
    - 0.1 ≤ IV < 0.3: Medium predictor
    - 0.3 ≤ IV < 0.5: Strong predictor
    - IV ≥ 0.5: Suspicious (possible overfitting or data issue)
    
    Parameters:
        freq_good (np.ndarray): Array of good counts for each bin
        freq_bad (np.ndarray): Array of bad counts for each bin
    
    Returns:
        float: The Information Value, rounded to 4 decimal places
    """
    # Convert inputs to float arrays
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # If no goods or bads, IV is 0 (no predictive power)
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    # Calculate distributions
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Create "safe" versions with small values instead of zeros
    # This prevents log(0) which would be -infinity
    dist_good_safe = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    # Calculate WOE for each bin
    woe = np.log(dist_bad_safe / dist_good_safe)
    
    # Calculate IV as sum of (dist_bad - dist_good) * WOE
    # This weights each bin by how different its distribution is from overall
    iv = np.sum((dist_bad - dist_good) * woe)
    
    # Handle any infinite or NaN values by replacing with 0
    if not np.isfinite(iv):
        iv = 0.0
    
    # Round to 4 decimal places and return
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """
    Calculate entropy for a bin.
    
    Entropy measures the "purity" or "disorder" of a bin:
    - Entropy = 0: Pure bin (all goods OR all bads)
    - Entropy = 1: Maximum impurity (50% goods, 50% bads)
    
    Formula: Entropy = -Σ p_i * log2(p_i)
    
    Lower entropy is better for prediction because it means the bin
    is more "pure" and can better separate goods from bads.
    
    Parameters:
        goods (int): Number of good outcomes in the bin
        bads (int): Number of bad outcomes in the bin
    
    Returns:
        float: Entropy value between 0 and 1, rounded to 4 decimal places
    """
    # Total count in this bin
    total = goods + bads
    
    # Edge cases: empty bin or pure bin (no goods or no bads)
    # In these cases, entropy is 0 (perfectly pure or empty)
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    # Calculate probability of good and bad outcomes
    p_good = goods / total
    p_bad = bads / total
    
    # Calculate entropy using the binary entropy formula
    # H = -(p_bad * log2(p_bad) + p_good * log2(p_good))
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    
    # Round to 4 decimal places
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """
    Determine if a variable is numeric or factor (categorical).
    
    This function examines a pandas Series and decides whether to treat it
    as a numeric variable (continuous, requires binning) or a factor
    (categorical, each unique value becomes its own bin).
    
    Decision logic:
    1. If the dtype is numeric (int, float) AND has more than 10 unique values,
       treat as numeric (will be binned using decision tree)
    2. If numeric with 10 or fewer unique values, treat as factor
       (likely ordinal categorical like age groups 1-5)
    3. If non-numeric (string, object), always treat as factor
    
    Parameters:
        series (pd.Series): The column to analyze
    
    Returns:
        str: Either 'numeric' or 'factor'
    """
    # Check if the series has a numeric data type (int, float, etc.)
    if pd.api.types.is_numeric_dtype(series):
        # If numeric but only 10 or fewer unique values,
        # treat as categorical (factor)
        # This handles cases like encoded categories (1, 2, 3)
        if series.nunique() <= 10:
            return 'factor'
        # Otherwise, treat as true numeric (continuous)
        return 'numeric'
    
    # Non-numeric types (string, object, etc.) are always factors
    return 'factor'


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.05,
    max_bins: int = 10
) -> List[float]:
    """
    Use a decision tree to find optimal split points for numeric variables.
    
    This function fits a DecisionTreeClassifier to find the thresholds that
    best separate goods from bads. Decision trees naturally find splits that
    maximize information gain, which correlates with IV.
    
    The tree is constrained by:
    - max_leaf_nodes: Maximum number of bins (splits + 1)
    - min_samples_leaf: Minimum proportion of data in each bin
    
    Parameters:
        x (pd.Series): The numeric variable to bin
        y (pd.Series): The binary target variable (0/1)
        min_prop (float): Minimum proportion of data that must be in each bin
                          Default 0.05 = 5% minimum per bin
        max_bins (int): Maximum number of bins to create
                        Default 10 bins maximum
    
    Returns:
        List[float]: Sorted list of split points (thresholds)
                     Empty list if no valid splits found
    
    Example:
        If splits = [25, 50, 75] for Age variable:
        - Bin 1: Age ≤ 25
        - Bin 2: 25 < Age ≤ 50
        - Bin 3: 50 < Age ≤ 75
        - Bin 4: Age > 75
    """
    # Create a mask for non-null values in both x and y
    # We can only use complete cases for tree fitting
    mask = x.notna() & y.notna()
    
    # Extract clean data as numpy arrays
    # reshape(-1, 1) converts 1D array to 2D column vector (required by sklearn)
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    # If no valid data points, return empty list (no splits)
    if len(x_clean) == 0:
        return []
    
    # Calculate minimum samples per leaf based on proportion
    # Ensure at least 1 sample per leaf
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    
    # Create the decision tree classifier
    # max_leaf_nodes limits the number of final bins
    # min_samples_leaf ensures each bin has enough data
    # random_state=42 makes results reproducible
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Try to fit the tree; return empty list if fitting fails
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        return []
    
    # Extract thresholds from the fitted tree
    # tree.tree_.threshold contains the split values at each node
    # -2 is a placeholder for leaf nodes (no split), so we filter those out
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    
    # Sort and deduplicate the thresholds
    thresholds = sorted(set(thresholds))
    
    return thresholds


def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """
    Create a bin DataFrame for a numeric variable based on split points.
    
    This function takes a list of split points and creates bins for a numeric
    variable. Each bin is defined by a range (lower, upper] and counts the
    goods and bads that fall within that range.
    
    Bins are created as follows:
    - First bin: x ≤ first_split
    - Middle bins: prev_split < x ≤ current_split
    - Last bin: x > last_split
    - NA bin: All null/missing values
    
    Parameters:
        df (pd.DataFrame): The input data
        var (str): Name of the variable to bin
        y_var (str): Name of the target variable (0/1)
        splits (List[float]): Sorted list of split points
    
    Returns:
        pd.DataFrame: DataFrame with columns [var, bin, count, bads, goods]
                      One row per bin (including NA bin if there are nulls)
    """
    # Extract the variable and target columns
    x = df[var]
    y = df[y_var]
    
    # List to collect bin data
    bins_data = []
    
    # Ensure splits are sorted
    splits = sorted(splits)
    
    # Create edges: negative infinity, all splits, positive infinity
    # This ensures we capture all values including extremes
    edges = [-np.inf] + splits + [np.inf]
    
    # Iterate through each pair of edges to create bins
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        
        # Create the bin rule string and mask based on edge values
        if lower == -np.inf:
            # First bin: no lower bound, just x ≤ upper
            mask = (x <= upper) & x.notna()
            # R-style bin rule format
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            # Last bin: x > lower, no upper bound
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            # Middle bins: lower < x ≤ upper
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        # Count observations in this bin
        count = mask.sum()
        
        # Only create bin if there are observations
        if count > 0:
            # Count bads (sum of 1s in target) and goods (count - bads)
            bads = y[mask].sum()
            goods = count - bads
            
            # Add bin data to list
            bins_data.append({
                'var': var,
                'bin': bin_rule,
                'count': count,
                'bads': int(bads),
                'goods': int(goods)
            })
    
    # Handle NA (null/missing) values as a separate bin
    na_mask = x.isna()
    if na_mask.sum() > 0:
        na_count = na_mask.sum()
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        
        # R-style NA indicator
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
    
    # Convert list of dicts to DataFrame and return
    return pd.DataFrame(bins_data)


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """
    Create a bin DataFrame for a factor/categorical variable.
    
    For categorical variables, each unique value becomes its own bin.
    No optimization is done here - every distinct value is a separate bin.
    
    Parameters:
        df (pd.DataFrame): The input data
        var (str): Name of the categorical variable
        y_var (str): Name of the target variable (0/1)
    
    Returns:
        pd.DataFrame: DataFrame with columns [var, bin, count, bads, goods]
                      One row per unique category (plus NA if present)
    """
    # Extract the variable and target columns
    x = df[var]
    y = df[y_var]
    
    # List to collect bin data
    bins_data = []
    
    # Get all unique non-null values
    unique_vals = x.dropna().unique()
    
    # Create a bin for each unique value
    for val in unique_vals:
        # Mask for rows where x equals this value
        mask = x == val
        count = mask.sum()
        
        if count > 0:
            bads = y[mask].sum()
            goods = count - bads
            
            # R-style %in% c() format for factor bins
            bins_data.append({
                'var': var,
                'bin': f'{var} %in% c("{val}")',
                'count': int(count),
                'bads': int(bads),
                'goods': int(goods)
            })
    
    # Handle NA values as a separate bin
    na_mask = x.isna()
    if na_mask.sum() > 0:
        na_count = na_mask.sum()
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
    
    return pd.DataFrame(bins_data)


def update_bin_stats(bin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update bin statistics including proportion, bad rate, IV, entropy, and trend.
    
    This function takes a bin DataFrame with basic counts and adds calculated
    statistics that are useful for analysis and display.
    
    Added columns:
    - propn: Percentage of total observations in this bin
    - bad_rate: Percentage of observations in this bin that are "bad"
    - goodCap: Proportion of all goods that are in this bin (for IV calc)
    - badCap: Proportion of all bads that are in this bin (for IV calc)
    - iv: Information Value contribution from this bin
    - ent: Entropy of this bin (0=pure, 1=mixed)
    - purNode: 'Y' if bin is pure (all goods or all bads), 'N' otherwise
    - trend: 'I' for increasing bad rate, 'D' for decreasing from previous bin
    
    Parameters:
        bin_df (pd.DataFrame): DataFrame with var, bin, count, bads, goods columns
    
    Returns:
        pd.DataFrame: Same DataFrame with additional calculated columns
    """
    # Return empty DataFrame if input is empty
    if bin_df.empty:
        return bin_df
    
    # Create a copy to avoid modifying the original
    df = bin_df.copy()
    
    # Calculate totals for proportion calculations
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # Calculate proportion: what percentage of all data is in this bin
    # Rounded to 2 decimal places for display
    df['propn'] = round(df['count'] / total_count * 100, 2)
    
    # Calculate bad rate: what percentage of this bin is "bad"
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    
    # Calculate good capture rate: what fraction of all goods are in this bin
    # Used for IV calculation
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    
    # Calculate bad capture rate: what fraction of all bads are in this bin
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    # Calculate IV contribution for each bin
    # IV = (goodCap - badCap) * ln(goodCap / badCap)
    # We handle zeros by replacing with 0.0001 before taking log
    df['iv'] = round((df['goodCap'] - df['badCap']) * np.log(
        np.where(df['goodCap'] == 0, 0.0001, df['goodCap']) / 
        np.where(df['badCap'] == 0, 0.0001, df['badCap'])
    ), 4)
    
    # Replace any infinite values with 0 (can happen with extreme distributions)
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    # Calculate entropy for each bin using our entropy function
    df['ent'] = df.apply(
        lambda row: calculate_entropy(row['goods'], row['bads']), 
        axis=1
    )
    
    # Mark pure nodes: bins where either goods=0 or bads=0
    # Pure nodes cause issues with WOE (infinite values)
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    # Calculate trend (I=increasing, D=decreasing) for bad rate
    # This is None for the first bin and compares to previous bin otherwise
    df['trend'] = None
    bad_rates = df['bad_rate'].values
    
    for i in range(1, len(bad_rates)):
        # Skip NA bins when calculating trend
        if 'is.na' not in str(df.iloc[i]['bin']):
            # Compare this bin's bad rate to previous bin's bad rate
            if bad_rates[i] >= bad_rates[i-1]:
                # Bad rate increased or stayed same
                df.iloc[i, df.columns.get_loc('trend')] = 'I'
            else:
                # Bad rate decreased
                df.iloc[i, df.columns.get_loc('trend')] = 'D'
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Add a summary "Total" row to the bin DataFrame.
    
    The Total row aggregates statistics across all bins and includes
    additional metrics that summarize the variable's overall characteristics:
    - Total count, goods, bads across all bins
    - Overall IV (sum of individual bin IVs)
    - Weighted average entropy
    - Whether monotonicity is maintained (monTrend)
    - Flip ratio (how often the trend changes direction)
    - Number of bins
    
    Parameters:
        bin_df (pd.DataFrame): DataFrame with individual bin rows
        var (str): Variable name (for the var column in total row)
    
    Returns:
        pd.DataFrame: Original DataFrame with Total row appended
    """
    # Create a copy to avoid modifying original
    df = bin_df.copy()
    
    # Calculate aggregate statistics
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # Sum of IV contributions (replace infinities with 0 first)
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()
    
    # Weighted average entropy (weighted by count in each bin)
    if total_count > 0:
        total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4)
    else:
        total_ent = 0
    
    # Check if monotonic trend is maintained
    # A variable is monotonic if all non-NA bins trend in same direction
    trends = df[df['trend'].notna()]['trend'].unique()
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    # Count increasing and decreasing trends
    incr_count = len(df[df['trend'] == 'I'])
    decr_count = len(df[df['trend'] == 'D'])
    total_trend_count = incr_count + decr_count
    
    # Calculate flip ratio: how often does the trend change?
    # Lower is better (0 = perfectly monotonic)
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    # Determine overall trend (whichever direction is more common)
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    
    # Check if any bins are pure (100% goods or 100% bads)
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    
    # Count number of bins (excluding any existing Total row)
    num_bins = len(df)
    
    # Create the Total row as a single-row DataFrame
    total_row = pd.DataFrame([{
        'var': var,
        'bin': 'Total',
        'count': total_count,
        'bads': total_bads,
        'goods': total_goods,
        'propn': 100.0,  # Total is always 100%
        'bad_rate': round(total_bads / total_count * 100, 2) if total_count > 0 else 0,
        'goodCap': 1.0,  # Total captures all goods
        'badCap': 1.0,   # Total captures all bads
        'iv': round(total_iv, 4),
        'ent': total_ent,
        'purNode': has_pure_node,
        'trend': overall_trend,
        'monTrend': mon_trend,
        'flipRatio': round(flip_ratio, 4),
        'numBins': num_bins
    }])
    
    # Concatenate original bins with Total row and return
    return pd.concat([df, total_row], ignore_index=True)


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    min_prop: float = 0.05,
    max_bins: int = 10
) -> BinResult:
    """
    Get optimal bins for multiple variables - main entry point for binning.
    
    This is the primary function for WOE binning, equivalent to logiBin::getBins in R.
    It processes multiple variables, determines their types, and creates optimal bins
    using decision trees for numeric variables or unique values for categorical.
    
    The function also provides progress logging for long-running operations.
    
    Parameters:
        df (pd.DataFrame): The input data with all variables
        y_var (str): Name of the target/dependent variable (binary 0/1)
        x_vars (List[str]): List of independent variable names to bin
        min_prop (float): Minimum proportion of data per bin (default 5%)
        max_bins (int): Maximum number of bins per variable (default 10)
    
    Returns:
        BinResult: Contains var_summary (one row per variable) and 
                   bin (detailed bin info for all variables)
    """
    # Lists to accumulate results
    all_bins = []      # Will hold bin DataFrames for all variables
    var_summaries = [] # Will hold summary stats for all variables
    
    # Tracking variables for progress logging
    total_vars = len(x_vars)
    start_time = time.time()
    last_log_time = start_time
    processed_count = 0
    times_per_var = []  # Track processing time per variable for ETA
    
    # Log start of processing
    log_progress(f"Starting binning for {total_vars} variables (Algorithm: DecisionTree)")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    
    # Process each variable
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        # Skip if variable not in DataFrame
        if var not in df.columns:
            continue
        
        # Determine if numeric or factor (categorical)
        var_type = get_var_type(df[var])
        
        # Create bins based on variable type
        if var_type == 'numeric':
            # For numeric: use decision tree to find optimal splits
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            # For factor/categorical: each unique value is a bin
            bin_df = _create_factor_bins(df, var, y_var)
        
        # Skip if no bins were created (e.g., all nulls)
        if bin_df.empty:
            continue
        
        # Update bin statistics (propn, bad_rate, IV, entropy, etc.)
        bin_df = update_bin_stats(bin_df)
        
        # Add the Total row with aggregate statistics
        bin_df = add_total_row(bin_df, var)
        
        # Extract summary info from the Total row
        total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
        var_summaries.append({
            'var': var,
            'varType': var_type,
            'iv': total_row['iv'],
            'ent': total_row['ent'],
            'trend': total_row['trend'],
            'monTrend': total_row.get('monTrend', 'N'),
            'flipRatio': total_row.get('flipRatio', 0),
            'numBins': total_row.get('numBins', len(bin_df) - 1),
            'purNode': total_row['purNode']
        })
        
        # Add this variable's bins to the master list
        all_bins.append(bin_df)
        
        # Progress logging logic
        var_time = time.time() - var_start
        times_per_var.append(var_time)
        processed_count += 1
        
        # Determine if we should log progress
        # Log every 10 variables, every 5 seconds, first variable, or last variable
        current_time = time.time()
        should_log = (
            processed_count % 10 == 0 or 
            processed_count == 1 or
            current_time - last_log_time >= 5.0 or
            processed_count == total_vars
        )
        
        if should_log:
            # Calculate progress metrics
            pct = (processed_count / total_vars) * 100
            elapsed = current_time - start_time
            avg_time = sum(times_per_var) / len(times_per_var)
            remaining = (total_vars - processed_count) * avg_time
            
            # Log progress with variable name, IV, elapsed time, and ETA
            log_progress(
                f"[{processed_count}/{total_vars}] {pct:.1f}% | "
                f"Variable: {var[:30]:30} | "
                f"IV: {total_row['iv']:.4f} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(remaining)}"
            )
            last_log_time = current_time
    
    # Log completion
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {processed_count} variables in {format_time(total_time)}")
    
    # Combine all bin DataFrames into one
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    # Create variable summary DataFrame
    var_summary_df = pd.DataFrame(var_summaries)
    
    # Return as BinResult container
    return BinResult(var_summary=var_summary_df, bin=combined_bins)


def manual_split(
    bin_result: BinResult,
    var: str,
    y_var: str,
    splits: List[float],
    df: pd.DataFrame
) -> BinResult:
    """
    Manually split a numeric variable at specified points.
    
    This function allows a user to override the automatic binning for a variable
    by specifying exact split points. Useful when domain knowledge suggests
    specific thresholds (e.g., age 18, 65 for legal/retirement ages).
    
    Parameters:
        bin_result (BinResult): Existing binning results to update
        var (str): Name of the variable to re-bin
        y_var (str): Name of the target variable
        splits (List[float]): User-specified split points
        df (pd.DataFrame): The original data
    
    Returns:
        BinResult: Updated binning results with new bins for specified variable
    """
    # Create new bins based on manual splits
    bin_df = _create_numeric_bins(df, var, y_var, splits)
    
    # If binning failed, return original unchanged
    if bin_df.empty:
        return bin_result
    
    # Update statistics and add total row
    bin_df = update_bin_stats(bin_df)
    bin_df = add_total_row(bin_df, var)
    
    # Get bins for other variables (not the one we're updating)
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
    
    # Combine other bins with new bins for this variable
    new_bins = pd.concat([other_bins, bin_df], ignore_index=True)
    
    # Update the variable summary
    total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
    var_summary = bin_result.var_summary.copy()
    
    # Find and update the row for this variable
    mask = var_summary['var'] == var
    if mask.any():
        var_summary.loc[mask, 'iv'] = total_row['iv']
        var_summary.loc[mask, 'ent'] = total_row['ent']
        var_summary.loc[mask, 'trend'] = total_row['trend']
        var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'N')
        var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
        var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(bin_df) - 1)
        var_summary.loc[mask, 'purNode'] = total_row['purNode']
    
    return BinResult(var_summary=var_summary, bin=new_bins)


# =============================================================================
# Bin Operations Functions
# =============================================================================
# These functions modify bins after initial creation (merging, splitting, etc.)


def _parse_numeric_from_rule(rule: str) -> List[float]:
    """
    Extract numeric values from a bin rule string.
    
    Bin rules are formatted like "Age > '25' & Age <= '50'" and this function
    extracts the numeric values (25 and 50 in this case).
    
    Parameters:
        rule (str): The bin rule string in R-style format
    
    Returns:
        List[float]: List of numeric values found in the rule
    
    Example:
        _parse_numeric_from_rule("Age > '25' & Age <= '50'") 
        -> [25.0, 50.0]
    """
    # Pattern matches numbers inside single quotes: '123' or '-45.67'
    pattern = r"'(-?\d+\.?\d*)'"
    
    # Find all matches and convert to floats
    matches = re.findall(pattern, rule)
    return [float(m) for m in matches]


def _parse_factor_values_from_rule(rule: str) -> List[str]:
    """
    Extract factor/categorical values from a bin rule string.
    
    Bin rules for factors are formatted like 'var %in% c("A", "B", "C")'
    and this function extracts the category values.
    
    Parameters:
        rule (str): The bin rule string in R-style format
    
    Returns:
        List[str]: List of category values found in the rule
    
    Example:
        _parse_factor_values_from_rule('Status %in% c("Active", "Pending")')
        -> ['Active', 'Pending']
    """
    # Pattern matches strings inside double quotes: "value"
    pattern = r'"([^"]*)"'
    
    # Find all matches
    matches = re.findall(pattern, rule)
    return matches


def na_combine(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """
    Combine NA bin with the adjacent bin that has the closest bad rate.
    
    NA (missing) values often need to be combined with a non-missing bin
    for WOE calculation and model building. This function merges the NA bin
    with whichever non-NA bin has the most similar bad rate.
    
    This is a common practice because:
    1. It reduces the number of bins
    2. It groups missing values with similar-risk observations
    3. It ensures the model can handle missing values
    
    Parameters:
        bin_result (BinResult): Current binning results
        vars_to_process (Union[str, List[str]]): Variable(s) to process
    
    Returns:
        BinResult: Updated binning results with NA bins merged
    """
    # Convert single variable to list for uniform processing
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    # Create copies to avoid modifying originals
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process each variable
    for var in vars_to_process:
        # Get bins for this variable
        var_bins = new_bins[new_bins['var'] == var].copy()
        
        # Skip if no bins exist
        if var_bins.empty:
            continue
        
        # Find the NA bin (contains 'is.na' in the rule)
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        
        # Skip if no NA bin exists
        if not na_mask.any():
            continue
        
        # Get the NA bin row
        na_bin = var_bins[na_mask].iloc[0]
        
        # Get non-NA bins (excluding Total row)
        non_na_bins = var_bins[~na_mask & (var_bins['bin'] != 'Total')]
        
        # Skip if no non-NA bins to merge with
        if non_na_bins.empty:
            continue
        
        # Calculate bad rate for NA bin
        na_bad_rate = na_bin['bads'] / na_bin['count'] if na_bin['count'] > 0 else 0
        
        # Find the non-NA bin with closest bad rate
        non_na_bins = non_na_bins.copy()
        non_na_bins['bad_rate_calc'] = non_na_bins['bads'] / non_na_bins['count']
        non_na_bins['rate_diff'] = abs(non_na_bins['bad_rate_calc'] - na_bad_rate)
        
        # Get index of closest bin
        closest_idx = non_na_bins['rate_diff'].idxmin()
        closest_bin = non_na_bins.loc[closest_idx]
        
        # Create combined bin rule (original rule | is.na())
        combined_rule = f"{closest_bin['bin']} | is.na({var})"
        
        # Combine counts
        combined_count = closest_bin['count'] + na_bin['count']
        combined_goods = closest_bin['goods'] + na_bin['goods']
        combined_bads = closest_bin['bads'] + na_bin['bads']
        
        # Update the closest bin with combined values
        new_bins.loc[closest_idx, 'bin'] = combined_rule
        new_bins.loc[closest_idx, 'count'] = combined_count
        new_bins.loc[closest_idx, 'goods'] = combined_goods
        new_bins.loc[closest_idx, 'bads'] = combined_bads
        
        # Remove the NA bin row
        na_idx = var_bins[na_mask].index[0]
        new_bins = new_bins.drop(na_idx)
        
        # Recalculate statistics for this variable
        var_new_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        var_new_bins = update_bin_stats(var_new_bins)
        var_new_bins = add_total_row(var_new_bins, var)
        
        # Replace all bins for this variable with updated bins
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, var_new_bins], ignore_index=True)
        
        # Update variable summary
        total_row = var_new_bins[var_new_bins['bin'] == 'Total'].iloc[0]
        mask = var_summary['var'] == var
        if mask.any():
            var_summary.loc[mask, 'iv'] = total_row['iv']
            var_summary.loc[mask, 'ent'] = total_row['ent']
            var_summary.loc[mask, 'trend'] = total_row['trend']
            var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'N')
            var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
            var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(var_new_bins) - 1)
            var_summary.loc[mask, 'purNode'] = total_row['purNode']
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def merge_pure_bins(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None
) -> BinResult:
    """
    Merge pure bins (100% goods or 100% bads) with the closest non-pure bin.
    
    Pure bins are problematic because they cause infinite WOE values:
    - 0 bads: WOE = ln(0/something) = -infinity
    - 0 goods: WOE = ln(something/0) = +infinity
    
    This function iteratively merges pure bins with their closest neighbor
    (by bad rate) until no pure bins remain.
    
    Parameters:
        bin_result (BinResult): Current binning results
        vars_to_process (Union[str, List[str]]): Variables to process (default: all)
    
    Returns:
        BinResult: Updated binning results with pure bins merged
    """
    # Default to all variables if none specified
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    # Create copies
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process each variable
    for var in vars_to_process:
        max_iterations = 100  # Safety limit to prevent infinite loops
        iteration = 0
        
        # Keep merging until no pure bins remain
        while iteration < max_iterations:
            iteration += 1
            
            # Get current bins for this variable (excluding Total)
            var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
            
            # If only one bin left, can't merge further
            if len(var_bins) <= 1:
                break
            
            # Find pure bins (goods=0 OR bads=0)
            pure_mask = (var_bins['goods'] == 0) | (var_bins['bads'] == 0)
            
            # If no pure bins, we're done with this variable
            if not pure_mask.any():
                break
            
            # Get the first pure bin to merge
            pure_bin = var_bins[pure_mask].iloc[0]
            pure_idx = var_bins[pure_mask].index[0]
            
            # Find non-pure bins to merge with
            non_pure_bins = var_bins[~pure_mask]
            
            if non_pure_bins.empty:
                # Edge case: all bins are pure
                # Merge with the bin that has closest count
                other_bins = var_bins[var_bins.index != pure_idx]
                if other_bins.empty:
                    break
                other_bins = other_bins.copy()
                other_bins['count_diff'] = abs(other_bins['count'] - pure_bin['count'])
                closest_idx = other_bins['count_diff'].idxmin()
                closest_bin = other_bins.loc[closest_idx]
            else:
                # Normal case: merge with closest non-pure bin by bad rate
                pure_bad_rate = pure_bin['bads'] / pure_bin['count'] if pure_bin['count'] > 0 else 0.5
                
                non_pure_bins = non_pure_bins.copy()
                non_pure_bins['bad_rate_calc'] = non_pure_bins['bads'] / non_pure_bins['count']
                non_pure_bins['rate_diff'] = abs(non_pure_bins['bad_rate_calc'] - pure_bad_rate)
                
                closest_idx = non_pure_bins['rate_diff'].idxmin()
                closest_bin = non_pure_bins.loc[closest_idx]
            
            # Create combined bin rule
            combined_rule = f"({closest_bin['bin']}) | ({pure_bin['bin']})"
            combined_count = closest_bin['count'] + pure_bin['count']
            combined_goods = closest_bin['goods'] + pure_bin['goods']
            combined_bads = closest_bin['bads'] + pure_bin['bads']
            
            # Update the closest bin
            new_bins.loc[closest_idx, 'bin'] = combined_rule
            new_bins.loc[closest_idx, 'count'] = combined_count
            new_bins.loc[closest_idx, 'goods'] = combined_goods
            new_bins.loc[closest_idx, 'bads'] = combined_bads
            
            # Remove the pure bin
            new_bins = new_bins.drop(pure_idx)
        
        # Recalculate statistics for this variable
        var_new_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        if not var_new_bins.empty:
            var_new_bins = update_bin_stats(var_new_bins)
            var_new_bins = add_total_row(var_new_bins, var)
            
            # Replace variable bins
            new_bins = new_bins[new_bins['var'] != var]
            new_bins = pd.concat([new_bins, var_new_bins], ignore_index=True)
            
            # Update var_summary
            total_row = var_new_bins[var_new_bins['bin'] == 'Total'].iloc[0]
            mask = var_summary['var'] == var
            if mask.any():
                var_summary.loc[mask, 'iv'] = total_row['iv']
                var_summary.loc[mask, 'ent'] = total_row['ent']
                var_summary.loc[mask, 'trend'] = total_row['trend']
                var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'N')
                var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
                var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(var_new_bins) - 1)
                var_summary.loc[mask, 'purNode'] = 'N'  # No more pure nodes
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def break_bin(
    bin_result: BinResult,
    var: str,
    y_var: str,
    df: pd.DataFrame
) -> BinResult:
    """
    Break all bins for a factor variable - each unique value becomes its own bin.
    
    This is useful when bins have been previously merged but you want to
    start fresh with one bin per category.
    
    Parameters:
        bin_result (BinResult): Current binning results
        var (str): Variable to break into individual bins
        y_var (str): Target variable name
        df (pd.DataFrame): Original data
    
    Returns:
        BinResult: Updated binning results with broken bins
    """
    # Create new bins from scratch (one per unique value)
    new_var_bins = _create_factor_bins(df, var, y_var)
    new_var_bins = update_bin_stats(new_var_bins)
    new_var_bins = add_total_row(new_var_bins, var)
    
    # Keep other variables' bins unchanged
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
    new_bins = pd.concat([other_bins, new_var_bins], ignore_index=True)
    
    # Update variable summary
    total_row = new_var_bins[new_var_bins['bin'] == 'Total'].iloc[0]
    var_summary = bin_result.var_summary.copy()
    mask = var_summary['var'] == var
    if mask.any():
        var_summary.loc[mask, 'iv'] = total_row['iv']
        var_summary.loc[mask, 'ent'] = total_row['ent']
        var_summary.loc[mask, 'trend'] = total_row['trend']
        var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'N')
        var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
        var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(new_var_bins) - 1)
        var_summary.loc[mask, 'purNode'] = total_row['purNode']
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def force_incr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """
    Force an increasing monotonic trend in bad rates by combining adjacent bins.
    
    Monotonic WOE (always increasing or always decreasing) is often preferred
    in credit scoring because:
    1. It's easier to interpret (higher value = higher/lower risk)
    2. It's more stable and less prone to overfitting
    3. It makes business sense for many variables (e.g., income)
    
    This function merges adjacent bins where bad rate decreases to force
    a strictly increasing pattern.
    
    Parameters:
        bin_result (BinResult): Current binning results
        vars_to_process (Union[str, List[str]]): Variables to process
    
    Returns:
        BinResult: Updated binning results with increasing bad rate trend
    """
    # Convert single variable to list
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    # Create copies
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process each variable
    for var in vars_to_process:
        # Get bins for this variable (excluding Total)
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        
        # Skip if insufficient bins
        if var_bins.empty or len(var_bins) < 2:
            continue
        
        # Separate NA bins (don't include in trend optimization)
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
        working_bins = var_bins[~na_mask].copy()
        
        if working_bins.empty:
            continue
        
        # Reset index for easier iteration
        working_bins = working_bins.reset_index(drop=True)
        
        # Iteratively merge bins where bad rate decreases
        changed = True
        while changed and len(working_bins) > 1:
            changed = False
            
            # Calculate current bad rates
            working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
            
            # Find first pair where bad rate decreases
            for i in range(1, len(working_bins)):
                if working_bins.iloc[i]['bad_rate_calc'] < working_bins.iloc[i-1]['bad_rate_calc']:
                    # Merge bin i into bin i-1
                    working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                    
                    # Update bin rule to reflect merged range
                    old_rule = working_bins.iloc[i-1]['bin']
                    new_rule = working_bins.iloc[i]['bin']
                    
                    # Try to create a cleaner combined rule for numeric bins
                    if '<=' in new_rule:
                        new_upper = _parse_numeric_from_rule(new_rule)
                        if new_upper:
                            max_upper = max(new_upper)
                            if '<=' in old_rule and '>' in old_rule:
                                lower_vals = _parse_numeric_from_rule(old_rule.split('&')[0]) if '&' in old_rule else []
                                if lower_vals:
                                    working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} > '{min(lower_vals)}' & {var} <= '{max_upper}'"
                                else:
                                    working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} <= '{max_upper}'"
                            elif '<=' in old_rule:
                                working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} <= '{max_upper}'"
                    elif '>' in new_rule and '<=' not in new_rule:
                        if '>' in old_rule:
                            old_lower = _parse_numeric_from_rule(old_rule.split('&')[0]) if '&' in old_rule else _parse_numeric_from_rule(old_rule)
                            if old_lower:
                                working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} > '{min(old_lower)}'"
                    
                    # Remove the merged bin
                    working_bins = working_bins.drop(working_bins.index[i]).reset_index(drop=True)
                    changed = True
                    break
        
        # Add back NA bin if it existed
        if not na_bin.empty:
            working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
        
        # Clean up temporary column
        if 'bad_rate_calc' in working_bins.columns:
            working_bins = working_bins.drop('bad_rate_calc', axis=1)
        
        # Recalculate statistics
        working_bins = update_bin_stats(working_bins)
        working_bins = add_total_row(working_bins, var)
        
        # Replace bins for this variable
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
        
        # Update variable summary
        total_row = working_bins[working_bins['bin'] == 'Total'].iloc[0]
        mask = var_summary['var'] == var
        if mask.any():
            var_summary.loc[mask, 'iv'] = total_row['iv']
            var_summary.loc[mask, 'ent'] = total_row['ent']
            var_summary.loc[mask, 'trend'] = total_row['trend']
            var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'Y')
            var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
            var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(working_bins) - 1)
            var_summary.loc[mask, 'purNode'] = total_row['purNode']
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def force_decr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """
    Force a decreasing monotonic trend in bad rates by combining adjacent bins.
    
    Similar to force_incr_trend but enforces a decreasing pattern instead.
    This is useful for variables where higher values indicate lower risk
    (e.g., income, credit score).
    
    Parameters:
        bin_result (BinResult): Current binning results
        vars_to_process (Union[str, List[str]]): Variables to process
    
    Returns:
        BinResult: Updated binning results with decreasing bad rate trend
    """
    # Convert single variable to list
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    # Create copies
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process each variable
    for var in vars_to_process:
        # Get bins for this variable (excluding Total)
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        
        # Skip if insufficient bins
        if var_bins.empty or len(var_bins) < 2:
            continue
        
        # Separate NA bins
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
        working_bins = var_bins[~na_mask].copy()
        
        if working_bins.empty:
            continue
        
        working_bins = working_bins.reset_index(drop=True)
        
        # Iteratively merge bins where bad rate INCREASES (to force decrease)
        changed = True
        while changed and len(working_bins) > 1:
            changed = False
            working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
            
            for i in range(1, len(working_bins)):
                # Note: > instead of < for decreasing trend
                if working_bins.iloc[i]['bad_rate_calc'] > working_bins.iloc[i-1]['bad_rate_calc']:
                    # Merge bin i into bin i-1
                    working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                    
                    # Update bin rule (same logic as force_incr_trend)
                    old_rule = working_bins.iloc[i-1]['bin']
                    new_rule = working_bins.iloc[i]['bin']
                    
                    if '<=' in new_rule:
                        new_upper = _parse_numeric_from_rule(new_rule)
                        if new_upper:
                            max_upper = max(new_upper)
                            if '<=' in old_rule and '>' in old_rule:
                                lower_vals = _parse_numeric_from_rule(old_rule.split('&')[0]) if '&' in old_rule else []
                                if lower_vals:
                                    working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} > '{min(lower_vals)}' & {var} <= '{max_upper}'"
                                else:
                                    working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} <= '{max_upper}'"
                            elif '<=' in old_rule:
                                working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} <= '{max_upper}'"
                    elif '>' in new_rule and '<=' not in new_rule:
                        if '>' in old_rule:
                            old_lower = _parse_numeric_from_rule(old_rule.split('&')[0]) if '&' in old_rule else _parse_numeric_from_rule(old_rule)
                            if old_lower:
                                working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} > '{min(old_lower)}'"
                    
                    working_bins = working_bins.drop(working_bins.index[i]).reset_index(drop=True)
                    changed = True
                    break
        
        # Add back NA bin
        if not na_bin.empty:
            working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
        
        if 'bad_rate_calc' in working_bins.columns:
            working_bins = working_bins.drop('bad_rate_calc', axis=1)
        
        # Recalculate statistics
        working_bins = update_bin_stats(working_bins)
        working_bins = add_total_row(working_bins, var)
        
        # Replace bins
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
        
        # Update summary
        total_row = working_bins[working_bins['bin'] == 'Total'].iloc[0]
        mask = var_summary['var'] == var
        if mask.any():
            var_summary.loc[mask, 'iv'] = total_row['iv']
            var_summary.loc[mask, 'ent'] = total_row['ent']
            var_summary.loc[mask, 'trend'] = total_row['trend']
            var_summary.loc[mask, 'monTrend'] = total_row.get('monTrend', 'Y')
            var_summary.loc[mask, 'flipRatio'] = total_row.get('flipRatio', 0)
            var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(working_bins) - 1)
            var_summary.loc[mask, 'purNode'] = total_row['purNode']
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def create_binned_columns(
    bin_result: BinResult,
    df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_"
) -> pd.DataFrame:
    """
    Create binned columns in the DataFrame based on binning rules.
    
    This function applies the bin rules to the original data to create
    new columns that contain the bin label for each row. These binned
    columns are used for:
    1. Looking up WOE values
    2. Applying scorecards
    3. Visual analysis
    
    Parameters:
        bin_result (BinResult): Binning rules to apply
        df (pd.DataFrame): Original data to transform
        x_vars (List[str]): Variables to create bins for
        prefix (str): Prefix for new column names (default "b_")
    
    Returns:
        pd.DataFrame: Original data with new binned columns (b_varname)
    
    Example:
        If Age has bins [<25, 25-50, >50], a new column b_Age is created
        with values like "<= '25'", "> '25' & <= '50'", "> '50'"
    """
    # Create copy to avoid modifying original
    result_df = df.copy()
    
    # Process each variable
    for var in x_vars:
        # Get bins for this variable (excluding Total row)
        var_bins = bin_result.bin[(bin_result.bin['var'] == var) & 
                                   (bin_result.bin['bin'] != 'Total')]
        
        # Skip if no bins exist
        if var_bins.empty:
            continue
        
        # Create new column name with prefix
        new_col = prefix + var
        
        # Initialize column with None
        result_df[new_col] = None
        
        # Track NA rule for later application
        na_rule = None
        
        # Apply each bin rule
        for _, row in var_bins.iterrows():
            rule = row['bin']
            
            # Extract the bin value (remove variable name and formatting)
            bin_value = rule.replace(var, '').replace(' %in% c', '').strip()
            
            # Check if this rule includes NA handling
            if '| is.na' in rule:
                na_rule = bin_value
                main_rule = rule.split('|')[0].strip()
            else:
                main_rule = rule
            
            # Try to apply the rule and assign bin values
            try:
                is_na_bin = False
                
                # Handle different rule formats
                if 'is.na' in main_rule and '|' not in main_rule:
                    # Standalone NA bin
                    mask = result_df[var].isna()
                    is_na_bin = True
                    
                elif '%in%' in main_rule:
                    # Factor/categorical bin: var %in% c("A", "B")
                    values = _parse_factor_values_from_rule(main_rule)
                    mask = result_df[var].isin(values)
                    
                elif '<=' in main_rule and '>' in main_rule:
                    # Numeric range bin: var > 'lower' & var <= 'upper'
                    nums = _parse_numeric_from_rule(main_rule)
                    if len(nums) >= 2:
                        lower, upper = min(nums), max(nums)
                        mask = (result_df[var] > lower) & (result_df[var] <= upper)
                    else:
                        continue
                        
                elif '<=' in main_rule:
                    # Upper-bounded numeric bin: var <= 'upper'
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        upper = max(nums)
                        mask = result_df[var] <= upper
                    else:
                        continue
                        
                elif '>' in main_rule:
                    # Lower-bounded numeric bin: var > 'lower'
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        lower = min(nums)
                        mask = result_df[var] > lower
                    else:
                        continue
                        
                elif '==' in main_rule:
                    # Exact match: var == 'value'
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        result_df.loc[result_df[var] == nums[0], new_col] = bin_value
                    continue
                else:
                    # Unknown format, skip
                    continue
                
                # Apply the mask to assign bin values
                if is_na_bin:
                    # For NA bins, apply directly
                    result_df.loc[mask, new_col] = bin_value
                else:
                    # For non-NA bins, exclude NA values
                    result_df.loc[mask & result_df[var].notna(), new_col] = bin_value
                
            except Exception:
                # Skip rules that can't be parsed
                continue
        
        # Handle NA values explicitly if there's an NA rule
        if na_rule is not None:
            result_df.loc[result_df[var].isna(), new_col] = na_rule
        elif result_df[var].isna().any():
            # Try to find and apply standalone NA bin
            na_bins = var_bins[var_bins['bin'].str.match(r'^is\.na\(', na=False)]
            if not na_bins.empty:
                bin_value = na_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
                result_df.loc[result_df[var].isna(), new_col] = bin_value
        
        # Handle any remaining unassigned rows
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
            # Assign to NA bin or first bin as fallback
            if na_rule is not None:
                fallback_bin = na_rule
            elif not var_bins.empty:
                fallback_bin = var_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
            else:
                fallback_bin = "Unmatched"
            result_df.loc[unassigned_mask, new_col] = fallback_bin
    
    return result_df


def add_woe_columns(
    df: pd.DataFrame,
    bins_df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    woe_prefix: str = "WOE_"
) -> pd.DataFrame:
    """
    Add WOE columns to the DataFrame by joining with binning rules.
    
    This function looks up the WOE value for each row based on its bin
    assignment and creates new WOE columns. These WOE columns are the
    actual inputs to logistic regression.
    
    Missing/unmatched bin values are assigned WOE=0 (neutral - no information).
    
    Parameters:
        df (pd.DataFrame): Data with binned columns (b_*)
        bins_df (pd.DataFrame): Binning rules with WOE values
        x_vars (List[str]): Variables to add WOE for
        prefix (str): Prefix of binned columns (default "b_")
        woe_prefix (str): Prefix for new WOE columns (default "WOE_")
    
    Returns:
        pd.DataFrame: Original data with new WOE columns (WOE_varname)
    """
    # Create copy
    result_df = df.copy()
    
    # Process each variable
    for var in x_vars:
        # Get bins for this variable (excluding Total)
        var_bins = bins_df[(bins_df['var'] == var) & (bins_df['bin'] != 'Total')].copy()
        
        if var_bins.empty:
            continue
        
        # Calculate WOE if not already present
        if 'woe' not in var_bins.columns:
            var_bins['woe'] = calculate_woe(var_bins['goods'].values, var_bins['bads'].values)
        
        # Create bin value column (for matching with binned data)
        var_bins['binValue'] = var_bins['bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
        
        # Column names
        bin_col = prefix + var
        woe_col = woe_prefix + var
        
        # Map bin values to WOE values
        if bin_col in result_df.columns:
            woe_map = dict(zip(var_bins['binValue'], var_bins['woe']))
            result_df[woe_col] = result_df[bin_col].map(woe_map)
            
            # Check for unmapped values (indicates a bug)
            missing_woe_count = result_df[woe_col].isna().sum()
            if missing_woe_count > 0:
                unmapped_bins = result_df.loc[result_df[woe_col].isna(), bin_col].unique()
                print(f"[ERROR] {var}: {missing_woe_count} rows have unmapped bin values!")
                print(f"        Unmapped bins: {list(unmapped_bins)}")
                print(f"        Available bins in woe_map: {list(woe_map.keys())}")
                
                # Try to recover by finding matching WOE values
                for unmapped_bin in unmapped_bins:
                    if unmapped_bin is None or pd.isna(unmapped_bin):
                        # Find NA bin WOE
                        na_woe_bins = var_bins[var_bins['bin'].str.contains('is.na', na=False)]
                        if not na_woe_bins.empty:
                            na_woe = na_woe_bins.iloc[0]['woe']
                            result_df.loc[result_df[bin_col].isna(), woe_col] = na_woe
                            print(f"        -> Assigned NA bin WOE: {na_woe}")
                    else:
                        # Try exact match in original bin rules
                        for _, bin_row in var_bins.iterrows():
                            if unmapped_bin in bin_row['bin'] or bin_row['binValue'] == unmapped_bin:
                                result_df.loc[result_df[bin_col] == unmapped_bin, woe_col] = bin_row['woe']
                                print(f"        -> Matched '{unmapped_bin}' to WOE: {bin_row['woe']}")
                                break
    
    return result_df


# =============================================================================
# Shiny UI Application
# =============================================================================
# This section contains the interactive web interface for manual WOE editing


def create_woe_editor_app(df: pd.DataFrame, min_prop: float = 0.05):
    """
    Create the WOE Editor Shiny application.
    
    This function builds a complete interactive web application for:
    1. Selecting dependent and independent variables
    2. Viewing and editing bin configurations
    3. Optimizing monotonicity
    4. Visualizing WOE and bad rates
    5. Exporting final results
    
    Parameters:
        df (pd.DataFrame): Input data to bin
        min_prop (float): Minimum proportion per bin (default 5%)
    
    Returns:
        App: Shiny application object that can be run
    """
    
    # Dictionary to store results when user clicks "Run & Close"
    # This is how we pass data back from the UI to the main script
    app_results = {
        'df_with_woe': None,    # Full data with WOE columns
        'df_only_woe': None,    # Just WOE columns + target
        'bins': None,           # Binning rules
        'dv': None,             # Selected dependent variable
        'completed': False      # Whether user completed workflow
    }
    
    # ==========================================================================
    # Define the User Interface Layout
    # ==========================================================================
    
    app_ui = ui.page_fluid(
        # Custom CSS styling for the application
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css?family=Raleway');
                body { font-family: 'Raleway', sans-serif; background-color: #f5f5f5; }
                .card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .btn-primary { background-color: #75AFD7; border-color: #75AFD7; }
                .btn-success { background-color: #9ECC53; border-color: #9ECC53; }
                .btn-danger { background-color: #B5202E; border-color: #B5202E; }
                .btn-secondary { background-color: #8A9399; border-color: #8A9399; }
                .btn-dark { background-color: #525E66; border-color: #525E66; }
                h4 { font-weight: bold; text-align: center; margin: 20px 0; }
                .divider { width: 10px; display: inline-block; }
            """)
        ),
        
        # Main title
        ui.h4("WOE Editor"),
        
        # Card 1: Variable Selection
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    # Dropdown to select the target/dependent variable
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=list(df.columns),
                                   selected=df.columns[0] if len(df.columns) > 0 else None)
                ),
                ui.column(6,
                    # Dropdown for target category (which value = "bad")
                    ui.input_select("tc", "Target Category", choices=[])
                )
            )
        ),
        
        # Card 2: Independent Variable Selection and Actions
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    # Dropdown to select variable to view/edit
                    ui.input_select("iv", "Independent Variable", choices=[]),
                    # Navigation buttons
                    ui.div(
                        ui.input_action_button("prev_btn", "← Previous", class_="btn btn-secondary"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("next_btn", "Next →", class_="btn btn-success"),
                    )
                ),
                ui.column(6,
                    # Bin operation buttons
                    ui.div(
                        ui.input_action_button("group_na_btn", "Group NA", class_="btn btn-primary"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("break_btn", "Break Bin", class_="btn btn-danger"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("reset_btn", "Reset", class_="btn btn-danger"),
                    ),
                    ui.br(),
                    # Optimization buttons
                    ui.div(
                        ui.input_action_button("optimize_btn", "Optimize", class_="btn btn-dark"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("optimize_all_btn", "Optimize All", class_="btn btn-dark"),
                    ),
                )
            )
        ),
        
        # Row with bin table and WOE graph
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px; overflow-y: auto;"},
                    ui.h5("Bin Details"),
                    ui.output_data_frame("woe_table")  # DataGrid showing bin stats
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px;"},
                    ui.h5("WOE & Bad Rate"),
                    output_widget("woe_graph")  # Plotly line chart
                )
            )
        ),
        
        # Row with count and proportion bar charts
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 350px;"},
                    output_widget("count_bar")  # Count distribution
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 350px;"},
                    output_widget("prop_bar")  # Good/Bad proportion
                )
            )
        ),
        
        # Measurements summary table
        ui.div(
            {"class": "card"},
            ui.h5("Measurements"),
            ui.output_data_frame("measurements_table")
        ),
        
        # Run button to complete and close
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "Run & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    # ==========================================================================
    # Define the Server Logic
    # ==========================================================================
    
    def server(input: Inputs, output: Outputs, session: Session):
        """
        Server function containing all reactive logic.
        
        This function handles:
        - Updating UI elements based on selections
        - Button click actions
        - Rendering tables and charts
        """
        
        # Reactive values to store state
        bins_rv = reactive.Value(None)           # Current variable's bins
        all_bins_rv = reactive.Value(None)       # All variables' bins
        all_bins_mod_rv = reactive.Value(None)   # Modified bins after optimize all
        modified_action_rv = reactive.Value(False)  # Flag if optimize_all was used
        initial_bins_rv = reactive.Value(None)   # Initial bins for comparison
        
        # ----------------------------------------------------------------------
        # Reactive effect: Update target category when DV changes
        # ----------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            if dv and dv in df.columns:
                # Get unique values for target category dropdown
                unique_vals = df[dv].dropna().unique().tolist()
                ui.update_select("tc", choices=unique_vals, 
                               selected=max(unique_vals) if unique_vals else None)
                
                # Get list of independent variables (all except DV)
                iv_list = [col for col in df.columns if col != dv]
                
                # Calculate bins for all variables if DV has no nulls
                if df[dv].isna().sum() <= 0:
                    try:
                        all_bins = get_bins(df, dv, iv_list, min_prop=min_prop)
                        all_bins_rv.set(all_bins)
                        
                        # Update IV dropdown with binnable variables
                        bin_vars = all_bins.var_summary['var'].tolist()
                        ui.update_select("iv", choices=bin_vars, 
                                       selected=bin_vars[0] if bin_vars else None)
                    except Exception as e:
                        print(f"Error calculating bins: {e}")
        
        # ----------------------------------------------------------------------
        # Reactive effect: Update bins when IV selection changes
        # ----------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.iv)
        def update_iv_bins():
            iv = input.iv()
            dv = input.dv()
            # Only update if not in "optimize all" mode
            if iv and dv and not modified_action_rv.get():
                try:
                    bins = get_bins(df, dv, [iv], min_prop=min_prop)
                    bins_rv.set(bins)
                    initial_bins_rv.set(bins)
                except Exception as e:
                    print(f"Error getting bins for {iv}: {e}")
        
        # ----------------------------------------------------------------------
        # Button handlers: Previous/Next variable navigation
        # ----------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.prev_btn)
        def prev_var():
            current = input.iv()
            all_bins = all_bins_rv.get()
            if all_bins is not None and current:
                vars_list = all_bins.var_summary['var'].tolist()
                if current in vars_list:
                    idx = vars_list.index(current)
                    if idx > 0:
                        ui.update_select("iv", selected=vars_list[idx - 1])
        
        @reactive.Effect
        @reactive.event(input.next_btn)
        def next_var():
            current = input.iv()
            all_bins = all_bins_rv.get()
            if all_bins is not None and current:
                vars_list = all_bins.var_summary['var'].tolist()
                if current in vars_list:
                    idx = vars_list.index(current)
                    if idx < len(vars_list) - 1:
                        ui.update_select("iv", selected=vars_list[idx + 1])
        
        # ----------------------------------------------------------------------
        # Button handlers: Bin operations
        # ----------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.group_na_btn)
        def group_na():
            """Combine NA bin with closest bin by bad rate."""
            bins = bins_rv.get()
            iv = input.iv()
            if bins is not None and iv:
                new_bins = na_combine(bins, iv)
                bins_rv.set(new_bins)
                modified_action_rv.set(False)
        
        @reactive.Effect
        @reactive.event(input.break_btn)
        def break_bins():
            """Break all bins into individual values."""
            bins = bins_rv.get()
            iv = input.iv()
            dv = input.dv()
            if bins is not None and iv and dv:
                new_bins = break_bin(bins, iv, dv, df)
                bins_rv.set(new_bins)
                modified_action_rv.set(False)
        
        @reactive.Effect
        @reactive.event(input.reset_btn)
        def reset_bins():
            """Reset bins to original automatic binning."""
            iv = input.iv()
            dv = input.dv()
            if iv and dv:
                modified_action_rv.set(False)
                bins = get_bins(df, dv, [iv], min_prop=min_prop)
                bins_rv.set(bins)
        
        @reactive.Effect
        @reactive.event(input.optimize_btn)
        def optimize_var():
            """Optimize monotonicity for current variable only."""
            bins = bins_rv.get()
            iv = input.iv()
            if bins is not None and iv:
                var_info = bins.var_summary[bins.var_summary['var'] == iv]
                if not var_info.empty:
                    trend = var_info.iloc[0]['trend']
                    if trend == 'I':
                        new_bins = force_incr_trend(bins, iv)
                    elif trend == 'D':
                        new_bins = force_decr_trend(bins, iv)
                    else:
                        new_bins = bins
                    bins_rv.set(new_bins)
                    modified_action_rv.set(False)
        
        @reactive.Effect
        @reactive.event(input.optimize_all_btn)
        def optimize_all():
            """Optimize monotonicity for all variables."""
            all_bins = all_bins_rv.get()
            if all_bins is not None:
                modified_action_rv.set(True)
                
                # First, combine NA bins for all variables
                bins_mod = na_combine(all_bins, all_bins.var_summary['var'].tolist())
                
                # Force decreasing trend on variables with D trend
                decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
                if decr_vars:
                    bins_mod = force_decr_trend(bins_mod, decr_vars)
                
                # Force increasing trend on variables with I trend
                incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
                if incr_vars:
                    bins_mod = force_incr_trend(bins_mod, incr_vars)
                
                all_bins_mod_rv.set(bins_mod)
        
        # ----------------------------------------------------------------------
        # Reactive calculation: Get bins to display
        # ----------------------------------------------------------------------
        @reactive.Calc
        def get_display_bins():
            """Get the appropriate bins DataFrame for display."""
            if modified_action_rv.get():
                # If optimize_all was used, show from modified bins
                all_mod = all_bins_mod_rv.get()
                iv = input.iv()
                if all_mod is not None and iv:
                    var_bins = all_mod.bin[all_mod.bin['var'] == iv].copy()
                    return var_bins
            else:
                # Otherwise show from individual variable bins
                bins = bins_rv.get()
                if bins is not None:
                    return bins.bin.copy()
            return pd.DataFrame()
        
        # ----------------------------------------------------------------------
        # Output renderers: Tables
        # ----------------------------------------------------------------------
        @output
        @render.data_frame
        def woe_table():
            """Render the bin details table."""
            display_bins = get_display_bins()
            if display_bins.empty:
                return render.DataGrid(pd.DataFrame())
            
            # Calculate WOE for non-Total rows
            non_total = display_bins[display_bins['bin'] != 'Total'].copy()
            if not non_total.empty:
                non_total['woe'] = calculate_woe(non_total['goods'].values, non_total['bads'].values)
            
            # Total row gets NaN for WOE
            total_row = display_bins[display_bins['bin'] == 'Total'].copy()
            if not total_row.empty:
                total_row['woe'] = np.nan
            
            # Combine and select display columns
            result = pd.concat([non_total, total_row], ignore_index=True)
            display_cols = ['bin', 'count', 'goods', 'bads', 'propn', 'bad_rate', 'woe', 'iv']
            display_cols = [c for c in display_cols if c in result.columns]
            
            return render.DataGrid(result[display_cols], selection_mode="rows", height="350px")
        
        # ----------------------------------------------------------------------
        # Output renderers: Charts
        # ----------------------------------------------------------------------
        @output
        @render_plotly
        def woe_graph():
            """Render the WOE and rates line chart."""
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            # Get non-Total bins for plotting
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            # Calculate WOE
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
            # Create figure with multiple traces
            fig = go.Figure()
            
            # Bad rate line (left axis)
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=plot_data['bad_rate'] / 100,
                name='Bad Rate', mode='lines+markers', line=dict(color='#3498db')
            ))
            
            # Good rate line (left axis)
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=(100 - plot_data['bad_rate']) / 100,
                name='Good Rate', mode='lines+markers', line=dict(color='#2ecc71')
            ))
            
            # WOE line (right axis)
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=plot_data['woe'],
                name='WOE', mode='lines+markers', line=dict(color='#e74c3c'), yaxis='y2'
            ))
            
            # Layout with dual y-axes
            fig.update_layout(
                title='WOE & Rates by Bin', xaxis_title='Bin', yaxis_title='Rate',
                yaxis2=dict(title='WOE', overlaying='y', side='right', showgrid=False),
                height=380, margin=dict(l=50, r=50, t=50, b=100),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            return fig
        
        @output
        @render_plotly
        def count_bar():
            """Render the count distribution bar chart."""
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
            # Bar chart with count, showing WOE in hover text
            fig = go.Figure(data=[
                go.Bar(
                    x=plot_data['bin'], y=plot_data['count'],
                    text=[f"Count: {c}<br>Propn: {p}%<br>WOE: {w:.4f}" 
                          for c, p, w in zip(plot_data['count'], plot_data['propn'], plot_data['woe'])],
                    textposition='outside', marker_color='#1F77B4'
                )
            ])
            
            fig.update_layout(
                title='Count Distribution', xaxis_title='Bin', yaxis_title='Count',
                height=300, margin=dict(l=50, r=50, t=50, b=100)
            )
            
            return fig
        
        @output
        @render_plotly
        def prop_bar():
            """Render the good/bad proportion stacked bar chart."""
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            
            # Good proportion (green)
            fig.add_trace(go.Bar(
                y=plot_data['bin'], x=100 - plot_data['bad_rate'],
                name='Good', orientation='h', marker_color='#9ECC53',
                text=100 - plot_data['bad_rate'], textposition='inside'
            ))
            
            # Bad proportion (red)
            fig.add_trace(go.Bar(
                y=plot_data['bin'], x=plot_data['bad_rate'],
                name='Bad', orientation='h', marker_color='#F25563',
                text=plot_data['bad_rate'], textposition='inside'
            ))
            
            fig.update_layout(
                title='Good/Bad Proportion', barmode='stack', height=300,
                margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            return fig
        
        @output
        @render.data_frame
        def measurements_table():
            """Render the measurements summary table."""
            display_bins = get_display_bins()
            initial = initial_bins_rv.get()
            
            if display_bins.empty:
                return render.DataGrid(pd.DataFrame({
                    'Initial IV': [0], 'Final IV': [0],
                    'Initial Entropy': [0], 'Final Entropy': [0]
                }))
            
            # Get final values from current bins
            total_row = display_bins[display_bins['bin'] == 'Total']
            final_iv = total_row['iv'].iloc[0] if not total_row.empty else 0
            final_ent = total_row['ent'].iloc[0] if not total_row.empty else 0
            
            # Get initial values for comparison
            initial_iv = 0
            initial_ent = 0
            if initial is not None:
                init_total = initial.bin[initial.bin['bin'] == 'Total']
                if not init_total.empty:
                    initial_iv = init_total['iv'].iloc[0]
                    initial_ent = init_total['ent'].iloc[0]
            
            # Create summary DataFrame
            measurements = pd.DataFrame({
                'Initial IV': [round(initial_iv, 4)],
                'Final IV': [round(final_iv, 4)],
                'Initial Entropy': [round(initial_ent, 4)],
                'Final Entropy': [round(final_ent, 4)]
            })
            
            return render.DataGrid(measurements)
        
        # ----------------------------------------------------------------------
        # Run & Close button handler
        # ----------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_and_close():
            """Finalize and close the application."""
            dv = input.dv()
            
            # Get final bins (from optimize_all or individual edits)
            if modified_action_rv.get():
                final_bins = all_bins_mod_rv.get()
            else:
                final_bins = all_bins_rv.get()
            
            if final_bins is None or dv is None:
                return
            
            # Get all variables that were binned
            all_vars = final_bins.var_summary['var'].tolist()
            
            # Prepare binning rules with WOE values
            rules = final_bins.bin[final_bins.bin['bin'] != 'Total'].copy()
            rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
            
            # Add binValue column for mapping
            for var in all_vars:
                var_mask = rules['var'] == var
                rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
                    lambda x: x.replace(var, '').replace(' %in% c', '').strip()
                )
            
            # Create output DataFrames
            df_with_bins = create_binned_columns(final_bins, df, all_vars)
            df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
            
            # Create WOE-only DataFrame for logistic regression
            woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
            df_only_woe = df_with_woe[woe_cols + [dv]].copy()
            
            # Store results
            app_results['df_with_woe'] = df_with_woe
            app_results['df_only_woe'] = df_only_woe
            app_results['bins'] = rules
            app_results['dv'] = dv
            app_results['completed'] = True
            
            # Close the session (closes browser window)
            await session.close()
    
    # Create and return the Shiny app
    app = App(app_ui, server)
    app.results = app_results
    return app


def run_woe_editor(df: pd.DataFrame, min_prop: float = 0.05) -> Dict[str, Any]:
    """
    Run the WOE Editor application and return results.
    
    This function handles:
    1. Finding an available port
    2. Creating the Shiny app
    3. Opening a browser window
    4. Running the app until user closes it
    5. Returning the results
    
    Parameters:
        df (pd.DataFrame): Input data
        min_prop (float): Minimum proportion per bin
    
    Returns:
        Dict: Results dictionary with df_with_woe, df_only_woe, bins, dv, completed
    """
    import socket
    import webbrowser
    
    # Find an available port by trying random ports
    port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
    
    for attempt in range(10):
        try:
            # Try to bind to the port to check if it's available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            break
        except OSError:
            # Port in use, try another random port
            port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
    
    log_progress(f"Starting Shiny UI on port {port} (Instance: {INSTANCE_ID})")
    
    # Create the app and open browser
    app = create_woe_editor_app(df, min_prop)
    webbrowser.open(f'http://127.0.0.1:{port}')
    
    # Run the app (blocks until user closes)
    app.run(host='127.0.0.1', port=port)
    
    # Return results after app closes
    return app.results


# =============================================================================
# Configuration
# =============================================================================

# Minimum proportion of data that must be in each bin
# 0.05 = 5%, meaning each bin must contain at least 5% of the data
# This prevents bins with too few observations (unreliable statistics)
min_prop = 0.05

# =============================================================================
# Read Input Data
# =============================================================================

# Read the first input table from KNIME into a pandas DataFrame
# knio.input_tables[0] is the first input port of the Python Script node
# .to_pandas() converts the KNIME table to a pandas DataFrame
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
# Flow variables determine whether to run in headless or interactive mode

# Initialize control variables
contains_dv = False   # Whether a valid dependent variable was specified
dv = None            # Dependent variable name
target = None        # Target category (which value = "bad")
optimize_all = False # Whether to auto-optimize all variables
group_na = False     # Whether to auto-group NA bins

# Try to read DependentVariable flow variable
# This is the main switch between headless and interactive mode
try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

# Try to read TargetCategory flow variable
# Specifies which value of the DV represents "bad" outcome
try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

# Try to read OptimizeAll flow variable
# If True, automatically apply monotonicity optimization to all variables
try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
except:
    pass

# Try to read GroupNA flow variable
# If True, automatically combine NA bins with closest non-NA bin
try:
    group_na = knio.flow_variables.get("GroupNA", False)
except:
    pass

# Check if we have a valid DependentVariable
# Must be: non-null, a string, non-empty, not "missing", and exist in data
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    # Run automatically without user interaction when flow variables are set
    
    # Log header
    log_progress("=" * 60)
    log_progress("WOE EDITOR - HEADLESS MODE (Original/DecisionTree)")
    log_progress("=" * 60)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"OptimizeAll: {optimize_all}, GroupNA: {group_na}")
    
    # Get list of independent variables (all columns except DV)
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter out constant variables (only 1 unique value)
    # These provide no predictive power and can cause issues
    constant_vars = []
    valid_vars = []
    for col in iv_list:
        # Count unique non-null values
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_vars.append(col)
        else:
            valid_vars.append(col)
    
    # Log removed constant variables
    if constant_vars:
        log_progress(f"Removed {len(constant_vars)} constant variables (only 1 unique value)")
        if len(constant_vars) <= 10:
            log_progress(f"  Constant vars: {constant_vars}")
        else:
            log_progress(f"  First 10: {constant_vars[:10]}...")
    
    # Update list to only valid variables
    iv_list = valid_vars
    log_progress(f"Variables to process: {len(iv_list)}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Compute initial bins using decision tree algorithm
    # -------------------------------------------------------------------------
    step_start = time.time()
    log_progress("STEP 1/5: Computing initial bins...")
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    log_progress(f"STEP 1/5 complete in {format_time(time.time() - step_start)}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Merge pure bins (prevents infinite WOE)
    # -------------------------------------------------------------------------
    step_start = time.time()
    
    # Check how many variables have pure bins
    if 'purNode' in bins_result.var_summary.columns:
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    
    if pure_count > 0:
        log_progress(f"STEP 2/5: Merging {int(pure_count)} pure bins (prevents infinite WOE)...")
        bins_result = merge_pure_bins(bins_result)
        log_progress(f"STEP 2/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 2/5: Skipped (no pure bins found)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Group NA bins (optional, based on flow variable)
    # -------------------------------------------------------------------------
    if group_na:
        step_start = time.time()
        log_progress("STEP 3/5: Grouping NA values...")
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        log_progress(f"STEP 3/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 3/5: Skipped (GroupNA=False)")
    
    # -------------------------------------------------------------------------
    # STEP 4: Optimize All (optional, based on flow variable)
    # -------------------------------------------------------------------------
    if optimize_all:
        step_start = time.time()
        log_progress("STEP 4/5: Optimizing monotonicity for all variables...")
        
        # First, ensure NA bins are grouped
        bins_mod = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        
        # Force decreasing trend on variables trending down
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            log_progress(f"  - Forcing decreasing trend on {len(decr_vars)} variables...")
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        # Force increasing trend on variables trending up
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            log_progress(f"  - Forcing increasing trend on {len(incr_vars)} variables...")
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
        log_progress(f"STEP 4/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 4/5: Skipped (OptimizeAll=False)")
    
    # -------------------------------------------------------------------------
    # STEP 5: Apply WOE transformation to data
    # -------------------------------------------------------------------------
    step_start = time.time()
    log_progress("STEP 5/5: Applying WOE transformation to data...")
    
    # Prepare binning rules with WOE values
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
    
    # Add binValue column for each variable
    for var in bins_result.var_summary['var'].tolist():
        var_mask = rules['var'] == var
        rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
    
    # Get list of all binned variables
    all_vars = bins_result.var_summary['var'].tolist()
    
    # Create binned columns (b_*) for each variable
    log_progress(f"  - Creating binned columns for {len(all_vars)} variables...")
    df_with_bins = create_binned_columns(bins_result, df, all_vars)
    
    # Add WOE columns (WOE_*) for each variable
    log_progress(f"  - Adding WOE columns...")
    df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
    
    # Create WOE-only DataFrame for logistic regression
    woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
    df_only_woe = df_with_woe[woe_cols + [dv]].copy()
    
    # Store bins for output
    bins = rules
    
    # Log completion
    log_progress(f"STEP 4/4 complete in {format_time(time.time() - step_start)}")
    log_progress("=" * 60)
    log_progress(f"COMPLETE: Processed {len(all_vars)} variables")
    log_progress("=" * 60)

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    # Launch the Shiny web UI for manual WOE editing
    
    print("Running in interactive mode - launching Shiny UI...")
    
    # Run the WOE editor app (blocks until user closes)
    results = run_woe_editor(df, min_prop=min_prop)
    
    # Extract results from the app
    if results['completed']:
        df_with_woe = results['df_with_woe']
        df_only_woe = results['df_only_woe']
        bins = results['bins']
        dv = results['dv']
        print("Interactive session completed successfully")
    else:
        # User cancelled or closed without completing
        print("Interactive session cancelled - returning empty results")
        df_with_woe = df.copy()
        df_only_woe = pd.DataFrame()
        bins = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================
# Write results to KNIME output ports

# Output Port 1: Original input DataFrame (unchanged)
# Useful for comparison or downstream nodes that need raw data
knio.output_tables[0] = knio.Table.from_pandas(df)

# Output Port 2: df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# Contains everything: original columns, bin assignments, and WOE values
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)

# Output Port 3: df_only_woe - Only WOE columns + dependent variable
# This is the direct input for logistic regression
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)

# Output Port 4: df_only_bins - ONLY binned columns (b_*) for scorecard scoring
# This is used when applying a scorecard to new data
# Extract only columns starting with "b_"
b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
df_only_bins = df_with_woe[b_columns].copy()
knio.output_tables[3] = knio.Table.from_pandas(df_only_bins)

# Output Port 5: bins - Binning rules with WOE values (metadata)
# Contains the binning logic needed to recreate the transformation
knio.output_tables[4] = knio.Table.from_pandas(bins)

# Print summary of outputs for user
print("=" * 70)
print("OUTPUT SUMMARY:")
print(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
print(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
print(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
print(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols) ** USE FOR SCORECARD **")
print(f"  Port 5: Bin Rules ({len(bins)} rows - metadata)")
print("=" * 70)
print("WOE Editor completed successfully")

# =============================================================================
# WOE Editor for KNIME Python Script Node - PARALLEL VERSION
# =============================================================================
# This script implements Weight of Evidence (WOE) binning for credit risk modeling.
# WOE is a technique used to transform categorical and continuous variables into
# numeric values that represent the predictive power of each bin/category.
#
# Python implementation matching R's WOE Editor functionality
# Compatible with KNIME 5.9, Python 3.9
#
# This version uses parallel processing to utilize multiple CPU cores
# for faster processing of large datasets with many variables.
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When DependentVariable flow variable is provided
#
# Outputs:
# 1. Original input DataFrame (unchanged)
# 2. df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# 3. df_only_woe - Only WOE columns + dependent variable
# 4. bins - Binning rules with WOE values
#
# Release Date: 2026-01-15
# Version: 2.0 (Parallel)
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

# Import the KNIME scripting interface - this is required to read input tables
# and write output tables when running inside a KNIME Python Script node
import knime.scripting.io as knio

# pandas is the primary data manipulation library - used for DataFrame operations
# DataFrames are 2D tabular data structures similar to Excel spreadsheets
import pandas as pd

# numpy provides numerical computing capabilities - used for array operations,
# mathematical functions like logarithms, and handling special values like infinity
import numpy as np

# re (regular expressions) is used to parse bin rule strings to extract
# numeric thresholds and categorical values from the bin definitions
import re

# warnings module allows us to suppress warning messages that might clutter output
# We suppress them because some numpy/pandas operations generate non-critical warnings
import warnings

# os module provides operating system interface - used for environment variables,
# process IDs, and CPU count detection
import os

# gc (garbage collector) allows manual memory management - we use it to free
# memory after processing large datasets to prevent memory leaks
import gc

# sys provides access to Python interpreter - used here for stdout.flush()
# to ensure print statements are immediately displayed in KNIME console
import sys

# random generates pseudo-random numbers - used to create unique port numbers
# for the Shiny web server to avoid conflicts when multiple instances run
import random

# multiprocessing provides parallel processing capabilities - used to detect
# the number of CPU cores available on the system
import multiprocessing

# typing provides type hints for function signatures - these don't affect
# runtime behavior but improve code readability and IDE support
# Dict = dictionary, List = list, Tuple = tuple, Optional = can be None,
# Any = any type, Union = one of several types
from typing import Dict, List, Tuple, Optional, Any, Union

# dataclasses provide a decorator to automatically generate __init__, __repr__,
# and other methods for classes that primarily store data
from dataclasses import dataclass

# Suppress all warning messages to keep the console output clean
# This prevents numpy and pandas from showing deprecation warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: STABILITY SETTINGS FOR MULTIPLE INSTANCE EXECUTION
# =============================================================================
# When running multiple KNIME nodes simultaneously, each needs its own port
# and thread settings to avoid conflicts

# Base port number for the Shiny web server UI
# Different from other scripts (8051, 8052, etc.) to avoid port conflicts
# Port 8054 was chosen as a unique identifier for this parallel version
BASE_PORT = 8054

# Range of random port offsets - the actual port will be BASE_PORT + random(0, 1000)
# This helps when the base port is already in use
RANDOM_PORT_RANGE = 1000

# Create a unique instance identifier by combining the process ID with a random number
# This is used for logging and debugging when multiple instances run simultaneously
# os.getpid() returns the current process ID (unique per running Python process)
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Set environment variables to limit threading in numerical libraries
# This prevents conflicts when KNIME runs multiple Python nodes in parallel
# Each library has its own environment variable for thread control

# NUMEXPR_MAX_THREADS controls numexpr library (used internally by pandas)
# Setting to '1' forces single-threaded operation
os.environ['NUMEXPR_MAX_THREADS'] = '1'

# OMP_NUM_THREADS controls OpenMP (parallel processing library used by numpy)
# Setting to '1' prevents internal parallelization that conflicts with our joblib parallelization
os.environ['OMP_NUM_THREADS'] = '1'

# OPENBLAS_NUM_THREADS controls OpenBLAS (linear algebra library)
# Single-thread prevents threading conflicts in matrix operations
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# MKL_NUM_THREADS controls Intel Math Kernel Library (alternative to OpenBLAS)
# Single-thread for stability
os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# SECTION 3: INSTALL/IMPORT DEPENDENCIES
# =============================================================================
# These try/except blocks attempt to import required libraries, and if they're
# not installed, they install them automatically using pip

# Try to import DecisionTreeClassifier from scikit-learn
# This is used to find optimal split points for numeric variables
try:
    # sklearn.tree contains decision tree algorithms
    # DecisionTreeClassifier is used because it naturally finds splits that
    # maximize information gain, which correlates with WOE predictive power
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    # If import fails, the library isn't installed
    # subprocess.check_call runs pip install and waits for completion
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    # After installation, import again
    from sklearn.tree import DecisionTreeClassifier

# Try to import joblib for parallel processing
try:
    # Parallel is the main class for running functions in parallel
    # delayed is a decorator that marks a function for parallel execution
    from joblib import Parallel, delayed
except ImportError:
    # Install joblib if not present
    import subprocess
    subprocess.check_call(['pip', 'install', 'joblib'])
    from joblib import Parallel, delayed

# Try to import Shiny for the interactive web UI
try:
    # App is the main Shiny application class
    # Inputs, Outputs, Session handle user interactions
    # reactive enables reactive programming (auto-updating when values change)
    # render and ui provide UI components and rendering functions
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    # shinywidgets provides Plotly integration for interactive charts
    from shinywidgets import render_plotly, output_widget
    # plotly.graph_objects provides the charting library for interactive visualizations
    import plotly.graph_objects as go
except ImportError:
    # Install all Shiny-related packages
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# SECTION 4: RESOURCE DETECTION
# =============================================================================
# This section determines how many CPU cores to use for parallel processing

def get_optimal_n_jobs(reserve_cores: int = 1, max_usage_percent: float = 0.75) -> int:
    """
    Determine optimal number of parallel jobs based on available system resources.
    
    This function analyzes the CPU configuration and returns a safe number of
    parallel workers that won't overload the system.
    
    Parameters:
    -----------
    reserve_cores : int
        Number of cores to reserve for system/other processes (default: 1)
        Leaving at least one core free prevents the system from becoming unresponsive
    max_usage_percent : float
        Maximum percentage of cores to use (default: 0.75 = 75%)
        Using less than 100% prevents thermal throttling and leaves headroom
    
    Returns:
    --------
    int : Optimal number of parallel jobs (workers)
    """
    try:
        # Get the count of logical processors (includes hyperthreading)
        # Logical cores = physical cores * 2 on hyperthreaded CPUs
        # os.cpu_count() can return None on some systems, so we default to 1
        logical_cores = os.cpu_count() or 1
        
        # Try to get the physical core count (more accurate for CPU-bound tasks)
        # Hyperthreading provides ~30% boost, not 100%, so physical cores matter more
        try:
            # On Windows, this typically gives the same as logical cores
            # On Linux/Mac, there are better ways to get physical cores
            physical_cores = multiprocessing.cpu_count()
        except:
            # If detection fails, assume physical = logical
            physical_cores = logical_cores
        
        # For CPU-bound tasks (like decision tree fitting), physical cores matter most
        # but we can benefit from some hyperthreading, hence the 1.5 multiplier
        # This balances CPU utilization without excessive context switching
        effective_cores = min(logical_cores, int(physical_cores * 1.5))
        
        # Apply maximum usage percentage to leave headroom
        # max(1, ...) ensures we always have at least 1 core
        max_cores = max(1, int(effective_cores * max_usage_percent))
        
        # Subtract reserved cores for system stability
        # Ensures background processes and the OS can still function smoothly
        available_cores = max(1, max_cores - reserve_cores)
        
        # Print configuration for debugging and user information
        print(f"[Parallel Config] Detected: {logical_cores} logical processors, "
              f"{physical_cores} physical cores")
        print(f"[Parallel Config] Using: {available_cores} parallel workers "
              f"(reserved {reserve_cores} for system)")
        
        # Return the calculated number of workers
        return available_cores
        
    except Exception as e:
        # If anything goes wrong during detection, use a safe default of 4
        # 4 workers work well on most modern systems (4-16 cores typical)
        print(f"[Parallel Config] Error detecting cores: {e}, defaulting to 4")
        return 4


# Call the function once at module load time to determine worker count
# This value is used throughout the script for all parallel operations
# Making it global avoids repeated detection calls
N_JOBS = get_optimal_n_jobs(reserve_cores=1, max_usage_percent=0.75)


# =============================================================================
# SECTION 5: DATA CLASSES
# =============================================================================
# Data classes are simple classes that primarily hold data
# The @dataclass decorator auto-generates __init__, __repr__, etc.

@dataclass
class BinResult:
    """
    Container for binning results returned by the get_bins function.
    
    This class bundles together two related DataFrames that describe
    the binning output for one or more variables.
    
    Attributes:
    -----------
    var_summary : pd.DataFrame
        Summary statistics for each variable including:
        - var: variable name
        - varType: 'numeric' or 'factor'
        - iv: Information Value (predictive power measure)
        - ent: Entropy (measure of uncertainty/purity)
        - trend: 'I' (increasing) or 'D' (decreasing) bad rate trend
        - monTrend: 'Y' if trend is monotonic, 'N' otherwise
        - flipRatio: ratio of trend reversals
        - numBins: number of bins created
        - purNode: 'Y' if any bin has 100% good or 100% bad
    
    bin : pd.DataFrame
        Detailed bin information including:
        - var: variable name
        - bin: the binning rule (e.g., "Age > '25' & Age <= '35'")
        - count: number of observations in bin
        - goods: count of good outcomes (target = 0)
        - bads: count of bad outcomes (target = 1)
        - propn: percentage of total observations
        - bad_rate: percentage of bads in bin
        - iv: contribution to total IV
        - woe: Weight of Evidence value
        - Plus a "Total" row for each variable with aggregates
    """
    var_summary: pd.DataFrame  # Summary stats for each variable
    bin: pd.DataFrame  # Detailed bin information


# =============================================================================
# SECTION 6: CORE BINNING FUNCTIONS
# =============================================================================
# These functions implement the mathematical calculations for WOE and IV

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray) -> np.ndarray:
    """
    Calculate Weight of Evidence (WOE) for each bin.
    
    WOE measures the predictive power of a bin by comparing the distribution
    of goods vs bads. The formula is:
    
    WOE = ln((% of Bads in bin) / (% of Goods in bin))
    
    Interpretation:
    - WOE > 0: More bads than expected (higher risk)
    - WOE < 0: Fewer bads than expected (lower risk)
    - WOE = 0: Neutral (average risk)
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good counts (target=0) for each bin
    freq_bad : np.ndarray
        Array of bad counts (target=1) for each bin
    
    Returns:
    --------
    np.ndarray : WOE values for each bin, rounded to 5 decimal places
    """
    # Convert inputs to float arrays to enable division
    # This handles cases where inputs might be integers or lists
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals across all bins
    # These are used to compute distribution percentages
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # Edge case: if there are no goods or no bads at all, WOE is undefined
    # Return zeros to avoid division by zero errors
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    # Calculate the distribution (percentage) of goods and bads in each bin
    # dist_good[i] = what percentage of all goods are in bin i
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Replace zeros with small value (0.0001) to avoid log(0) = -infinity
    # This is a standard epsilon replacement technique
    # np.where returns the second argument where condition is True, else third argument
    dist_good = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    # Calculate WOE using the natural logarithm
    # WOE = ln(dist_bad / dist_good)
    # Round to 5 decimal places for cleaner output
    woe = np.round(np.log(dist_bad / dist_good), 5)
    
    return woe


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """
    Calculate Information Value (IV) for a variable.
    
    IV measures the overall predictive power of a variable. It's calculated
    by summing the IV contribution of each bin:
    
    IV = Σ (dist_bad - dist_good) * WOE
    
    IV Interpretation (industry standard thresholds):
    - IV < 0.02: Not useful for prediction
    - 0.02 ≤ IV < 0.1: Weak predictor
    - 0.1 ≤ IV < 0.3: Medium predictor
    - 0.3 ≤ IV < 0.5: Strong predictor
    - IV ≥ 0.5: Suspiciously good (possible data leakage or overfit)
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good counts for each bin
    freq_bad : np.ndarray
        Array of bad counts for each bin
    
    Returns:
    --------
    float : Total IV value for the variable, rounded to 4 decimal places
    """
    # Convert to float arrays for calculation
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # If no goods or bads exist, IV is zero (no predictive power)
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    # Calculate distributions (percentages)
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Create "safe" versions with epsilon replacement to avoid log(0)
    # We keep original distributions for the difference calculation
    dist_good_safe = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    # Calculate WOE for each bin
    woe = np.log(dist_bad_safe / dist_good_safe)
    
    # IV formula: sum of (dist_bad - dist_good) * WOE for each bin
    # This captures both the magnitude and direction of prediction
    iv = np.sum((dist_bad - dist_good) * woe)
    
    # Handle any numerical issues (infinity or NaN)
    # np.isfinite returns False for inf, -inf, and nan
    if not np.isfinite(iv):
        iv = 0.0
    
    # Round to 4 decimal places for cleaner output
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """
    Calculate entropy for a bin.
    
    Entropy measures the impurity or uncertainty in a bin.
    Lower entropy means the bin is more "pure" (mostly goods or mostly bads).
    
    Formula: H = -Σ(p * log2(p))
    
    Where p is the proportion of each class (good/bad).
    
    Entropy ranges:
    - 0: Perfect purity (all goods or all bads)
    - 1: Maximum impurity (50% goods, 50% bads)
    
    Parameters:
    -----------
    goods : int
        Count of good outcomes in the bin
    bads : int
        Count of bad outcomes in the bin
    
    Returns:
    --------
    float : Entropy value between 0 and 1, rounded to 4 decimal places
    """
    # Calculate total count in this bin
    total = goods + bads
    
    # Edge cases: if bin is empty or pure (one class only), entropy is 0
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    # Calculate proportions (probabilities) of each class
    p_good = goods / total
    p_bad = bads / total
    
    # Shannon entropy formula with base-2 logarithm
    # The negative sign makes the result positive (since log of fraction is negative)
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """
    Determine if variable is numeric or factor (categorical).
    
    This function decides how to bin a variable:
    - Numeric variables use decision tree splits (thresholds)
    - Factor variables use each unique value as a bin
    
    The distinction matters because:
    - Numeric: ordered, can use < > comparisons, bins are ranges
    - Factor: unordered categories, each value is its own bin
    
    Parameters:
    -----------
    series : pd.Series
        A pandas Series containing the variable's values
    
    Returns:
    --------
    str : Either 'numeric' or 'factor'
    """
    # Check if the pandas dtype is numeric (int, float, etc.)
    # pd.api.types.is_numeric_dtype handles all numeric types
    if pd.api.types.is_numeric_dtype(series):
        # Even if numeric, treat as factor if very few unique values
        # Variables with ≤10 unique values are likely coded categories
        # Examples: days of week (1-7), months (1-12), rating (1-10)
        if series.nunique() <= 10:
            return 'factor'
        # True numeric variable with many values
        return 'numeric'
    
    # Non-numeric types (string, object, etc.) are always factors
    return 'factor'


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.01,
    max_bins: int = 10
) -> List[float]:
    """
    Use decision tree to find optimal split points for numeric variables.
    
    Decision trees naturally find splits that maximize information gain,
    which correlates with WOE predictive power. This is more sophisticated
    than simple equal-width or equal-frequency binning.
    
    The tree finds splits that best separate goods from bads.
    
    Parameters:
    -----------
    x : pd.Series
        The numeric variable to find splits for
    y : pd.Series
        The binary target variable (0/1)
    min_prop : float
        Minimum proportion of samples in each bin (default: 0.01 = 1%)
        Prevents creating tiny bins with unreliable statistics
    max_bins : int
        Maximum number of bins to create (default: 10)
        Controls complexity vs granularity tradeoff
    
    Returns:
    --------
    List[float] : Sorted list of split thresholds
                  Empty list if no valid splits found
    """
    # Create a mask for rows where both x and y have valid (non-null) values
    # We can only use complete cases for tree fitting
    mask = x.notna() & y.notna()
    
    # Extract clean data as numpy arrays
    # reshape(-1, 1) converts 1D array to 2D column (required by sklearn)
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    # If no valid data remains, return empty list
    if len(x_clean) == 0:
        return []
    
    # Calculate minimum samples per leaf node
    # This ensures each bin has enough observations for reliable statistics
    # max(, 1) ensures at least 1 sample per leaf
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    
    # Create the decision tree classifier
    # max_leaf_nodes=max_bins limits the number of bins created
    # min_samples_leaf enforces minimum bin size
    # random_state=42 ensures reproducible results
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Fit the tree to find optimal splits
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        # If fitting fails (e.g., all same values), return empty list
        return []
    
    # Extract thresholds from the fitted tree
    # tree.tree_.threshold contains the split values at each node
    # -2 indicates a leaf node (no split), so we filter those out
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    
    # Sort and deduplicate the thresholds
    # set() removes duplicates, sorted() orders them
    thresholds = sorted(set(thresholds))
    
    return thresholds


def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """
    Create bin DataFrame for numeric variable based on splits.
    
    This function takes a list of split points and creates bin definitions
    with counts of goods and bads in each bin.
    
    Bins are created as ranges:
    - First bin: x <= first_split
    - Middle bins: prev_split < x <= current_split
    - Last bin: x > last_split
    - NA bin: for missing values (handled separately)
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset
    var : str
        Name of the numeric variable to bin
    y_var : str
        Name of the target variable (0/1)
    splits : List[float]
        List of split thresholds from decision tree
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns [var, bin, count, bads, goods]
    """
    # Get the variable and target as Series
    x = df[var]
    y = df[y_var]
    
    # List to accumulate bin data
    bins_data = []
    
    # Sort splits and add -inf and +inf as boundaries
    # This creates the complete set of bin edges
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    # Create a bin for each pair of adjacent edges
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        
        # Create the appropriate mask and rule string based on position
        if lower == -np.inf:
            # First bin: x <= upper (no lower bound)
            mask = (x <= upper) & x.notna()
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            # Last bin: x > lower (no upper bound)
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            # Middle bin: lower < x <= upper
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        # Count observations in this bin
        count = mask.sum()
        
        # Only add bin if it has observations
        if count > 0:
            # Count bads (target = 1) and goods (target = 0)
            bads = y[mask].sum()
            goods = count - bads
            
            # Append bin data as dictionary
            bins_data.append({
                'var': var,
                'bin': bin_rule,
                'count': count,
                'bads': int(bads),
                'goods': int(goods)
            })
    
    # Handle missing values (NA) separately
    # Create a mask for NA values
    na_mask = x.isna()
    
    # If there are any NA values, create an NA bin
    if na_mask.sum() > 0:
        na_count = na_mask.sum()
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        
        # Use R-style is.na() notation for compatibility
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
    
    # Convert list of dictionaries to DataFrame
    return pd.DataFrame(bins_data)


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """
    Create bin DataFrame for factor/categorical variable.
    
    Each unique value becomes its own bin. This is appropriate for
    categorical variables where ordering doesn't make sense.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset
    var : str
        Name of the categorical variable to bin
    y_var : str
        Name of the target variable (0/1)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns [var, bin, count, bads, goods]
    """
    # Get the variable and target as Series
    x = df[var]
    y = df[y_var]
    
    # List to accumulate bin data
    bins_data = []
    
    # Get all unique non-null values
    unique_vals = x.dropna().unique()
    
    # Create a bin for each unique value
    for val in unique_vals:
        # Create mask for this value (exact equality)
        mask = x == val
        count = mask.sum()
        
        if count > 0:
            bads = y[mask].sum()
            goods = count - bads
            
            # Use R-style %in% c() notation for compatibility
            bins_data.append({
                'var': var,
                'bin': f'{var} %in% c("{val}")',
                'count': int(count),
                'bads': int(bads),
                'goods': int(goods)
            })
    
    # Handle missing values (NA) separately
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
    Update bin statistics (propn, bad_rate, iv, ent, trend, etc.)
    
    This function calculates derived statistics for each bin after
    the basic counts (goods, bads) have been determined.
    
    Parameters:
    -----------
    bin_df : pd.DataFrame
        DataFrame with basic bin info (var, bin, count, goods, bads)
    
    Returns:
    --------
    pd.DataFrame : Same DataFrame with additional calculated columns:
        - propn: percentage of total observations in this bin
        - bad_rate: percentage of observations in bin that are bad
        - goodCap: percentage of all goods that fall in this bin
        - badCap: percentage of all bads that fall in this bin
        - iv: Information Value contribution of this bin
        - ent: Entropy of this bin
        - purNode: 'Y' if bin is pure (100% good or 100% bad)
        - trend: 'I' (increasing) or 'D' (decreasing) vs previous bin
    """
    # Return empty DataFrame as-is
    if bin_df.empty:
        return bin_df
    
    # Make a copy to avoid modifying the original
    df = bin_df.copy()
    
    # Calculate totals for percentage calculations
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # propn: what percentage of all observations are in each bin
    # Multiply by 100 for percentage, round to 2 decimal places
    df['propn'] = round(df['count'] / total_count * 100, 2)
    
    # bad_rate: what percentage of observations in this bin are bad
    # This is the key metric for understanding risk in each bin
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    
    # goodCap: what percentage of ALL goods are in this bin
    # Also called "distribution of goods"
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    
    # badCap: what percentage of ALL bads are in this bin
    # Also called "distribution of bads"
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    # iv: Information Value contribution of this bin
    # IV = (goodCap - badCap) * ln(goodCap / badCap)
    # Replace zeros with small epsilon to avoid log(0)
    df['iv'] = round((df['goodCap'] - df['badCap']) * np.log(
        np.where(df['goodCap'] == 0, 0.0001, df['goodCap']) / 
        np.where(df['badCap'] == 0, 0.0001, df['badCap'])
    ), 4)
    
    # Replace any inf/-inf values with 0 (occurs when epsilon is used)
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    # ent: Entropy of each bin (measure of purity)
    # Apply the calculate_entropy function to each row
    df['ent'] = df.apply(
        lambda row: calculate_entropy(row['goods'], row['bads']), 
        axis=1  # axis=1 means apply across columns for each row
    )
    
    # purNode: 'Y' if this is a "pure" bin (all goods or all bads)
    # Pure bins cause problems because WOE becomes infinite
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    # trend: Is bad_rate Increasing or Decreasing vs previous bin
    # This helps identify if the binning shows a monotonic relationship
    df['trend'] = None  # Initialize column with None
    bad_rates = df['bad_rate'].values  # Get as numpy array for faster access
    
    # Compare each bin's bad_rate to the previous bin's
    for i in range(1, len(bad_rates)):
        # Skip NA bins (they're conceptually separate from the ordering)
        if 'is.na' not in str(df.iloc[i]['bin']):
            if bad_rates[i] >= bad_rates[i-1]:
                # Bad rate increased or stayed same
                df.iloc[i, df.columns.get_loc('trend')] = 'I'
            else:
                # Bad rate decreased
                df.iloc[i, df.columns.get_loc('trend')] = 'D'
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Add a total/summary row to the bin DataFrame.
    
    The total row aggregates statistics across all bins and provides
    overall metrics for the variable like total IV and trend consistency.
    
    Parameters:
    -----------
    bin_df : pd.DataFrame
        DataFrame with bin statistics
    var : str
        Variable name (for labeling the total row)
    
    Returns:
    --------
    pd.DataFrame : Original DataFrame with total row appended
    """
    df = bin_df.copy()
    
    # Aggregate basic counts
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # Sum IV across all bins (replace inf/-inf with 0 first)
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()
    
    # Weighted average entropy (weighted by count in each bin)
    if total_count > 0:
        total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4)
    else:
        total_ent = 0
    
    # Check if trends are monotonic (all same direction)
    trends = df[df['trend'].notna()]['trend'].unique()
    # monTrend = 'Y' if all bins have same trend direction
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    # Calculate flip ratio (measure of trend consistency)
    # Lower is better (means more consistent trend)
    incr_count = len(df[df['trend'] == 'I'])  # Count increasing trends
    decr_count = len(df[df['trend'] == 'D'])  # Count decreasing trends
    total_trend_count = incr_count + decr_count
    
    # Flip ratio is the proportion of the minority trend direction
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    # Overall trend is whichever direction is more common
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    
    # Check if any bin has pure nodes
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    
    # Count number of bins
    num_bins = len(df)
    
    # Create the total row as a single-row DataFrame
    total_row = pd.DataFrame([{
        'var': var,
        'bin': 'Total',  # Special label to identify total row
        'count': total_count,
        'bads': total_bads,
        'goods': total_goods,
        'propn': 100.0,  # Total is always 100%
        'bad_rate': round(total_bads / total_count * 100, 2) if total_count > 0 else 0,
        'goodCap': 1.0,  # Total is always 100%
        'badCap': 1.0,
        'iv': round(total_iv, 4),  # Sum of all bin IVs
        'ent': total_ent,  # Weighted average entropy
        'purNode': has_pure_node,  # Any pure nodes present?
        'trend': overall_trend,  # Dominant trend direction
        'monTrend': mon_trend,  # Is trend monotonic?
        'flipRatio': round(flip_ratio, 4),  # Trend consistency measure
        'numBins': num_bins  # Number of bins
    }])
    
    # Concatenate original bins with total row
    return pd.concat([df, total_row], ignore_index=True)


# =============================================================================
# SECTION 7: PARALLEL PROCESSING FUNCTIONS
# =============================================================================
# These functions enable processing multiple variables simultaneously

def _process_single_var(
    df: pd.DataFrame, 
    var: str, 
    y_var: str, 
    min_prop: float, 
    max_bins: int
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """
    Process a single variable for binning - designed for parallel execution.
    
    This function is called by joblib's Parallel to process each variable
    independently. It handles all error cases and returns None for
    variables that can't be binned.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset
    var : str
        Name of the variable to bin
    y_var : str
        Name of the target variable
    min_prop : float
        Minimum proportion per bin
    max_bins : int
        Maximum number of bins
    
    Returns:
    --------
    Tuple of (var_summary dict, bin_df DataFrame) or (None, None) if failed
    """
    try:
        # Check if the variable exists in the DataFrame
        if var not in df.columns:
            print(f"  [SKIP] {var}: Column not found in DataFrame")
            return None, None
        
        # Check for all-NaN columns (no usable data)
        non_na_count = df[var].notna().sum()
        if non_na_count == 0:
            print(f"  [SKIP] {var}: All values are NaN/missing")
            return None, None
        
        # Check for constant columns (no variance = no predictive power)
        if df[var].nunique() <= 1:
            print(f"  [SKIP] {var}: Constant column (only 1 unique value)")
            return None, None
        
        # Determine if variable is numeric or categorical
        var_type = get_var_type(df[var])
        
        # Create bins based on variable type
        if var_type == 'numeric':
            # Use decision tree to find optimal splits
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            # Create bins from the splits
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            # Each unique value becomes its own bin
            bin_df = _create_factor_bins(df, var, y_var)
        
        # Check if any bins were created
        if bin_df.empty:
            print(f"  [SKIP] {var}: No valid bins could be created (empty bin result)")
            return None, None
        
        # Calculate all the derived statistics
        bin_df = update_bin_stats(bin_df)
        
        # Add the total/summary row
        bin_df = add_total_row(bin_df, var)
        
        # Extract summary statistics from the total row
        total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
        
        # Create dictionary with variable-level summary
        var_summary = {
            'var': var,
            'varType': var_type,
            'iv': total_row['iv'],
            'ent': total_row['ent'],
            'trend': total_row['trend'],
            'monTrend': total_row.get('monTrend', 'N'),
            'flipRatio': total_row.get('flipRatio', 0),
            'numBins': total_row.get('numBins', len(bin_df) - 1),
            'purNode': total_row['purNode']
        }
        
        return var_summary, bin_df
        
    except Exception as e:
        # Catch any unexpected errors and skip the variable
        # Truncate error message to 80 chars for cleaner output
        print(f"  [SKIP] {var}: Error during processing - {str(e)[:80]}")
        return None, None


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    min_prop: float = 0.01,
    max_bins: int = 10,
    n_jobs: Optional[int] = None
) -> BinResult:
    """
    Get optimal bins for multiple variables using parallel processing.
    
    This is the main entry point for binning. It processes all specified
    variables and returns a BinResult containing summary and detail info.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data containing all variables
    y_var : str
        Name of the dependent variable (binary target 0/1)
    x_vars : List[str]
        List of independent variable names to bin
    min_prop : float
        Minimum proportion of samples in each bin (default: 0.01 = 1%)
    max_bins : int
        Maximum number of bins per variable (default: 10)
    n_jobs : int, optional
        Number of parallel jobs. If None, uses global N_JOBS setting.
    
    Returns:
    --------
    BinResult : Container with var_summary and bin DataFrames
    """
    # Use global N_JOBS if not specified
    if n_jobs is None:
        n_jobs = N_JOBS
    
    # Use parallel processing if we have multiple variables and cores
    if len(x_vars) > 1 and n_jobs > 1:
        print(f"[Parallel] Processing {len(x_vars)} variables using {n_jobs} workers...")
        
        try:
            # Parallel() creates a parallel executor
            # delayed() wraps the function for lazy evaluation
            # prefer="processes" uses separate processes (better for CPU-bound tasks)
            # verbose=0 suppresses joblib's progress output
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_single_var)(df, var, y_var, min_prop, max_bins)
                for var in x_vars
            )
        except Exception as e:
            # Fall back to sequential processing if parallel fails
            print(f"[Parallel] Parallel processing failed: {e}, falling back to sequential")
            results = [_process_single_var(df, var, y_var, min_prop, max_bins) for var in x_vars]
    else:
        # Sequential processing for single variable or single core
        results = [_process_single_var(df, var, y_var, min_prop, max_bins) for var in x_vars]
    
    # Collect results from all workers
    var_summaries = []  # List of summary dictionaries
    all_bins = []  # List of bin DataFrames
    skipped_count = 0  # Count of variables that couldn't be processed
    
    # Unpack results and handle None values (failed variables)
    for var_summary, bin_df in results:
        if var_summary is not None:
            var_summaries.append(var_summary)
            all_bins.append(bin_df)
        else:
            skipped_count += 1
    
    # Combine all bin DataFrames into one
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    # Convert summary list to DataFrame
    var_summary_df = pd.DataFrame(var_summaries)
    
    # Print summary of processing results
    print(f"[Parallel] Completed binning for {len(var_summaries)} of {len(x_vars)} variables")
    if skipped_count > 0:
        print(f"[Parallel] WARNING: {skipped_count} variables were skipped (see [SKIP] messages above)")
    
    # Return packaged results
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
    
    This allows users to override the automatic decision tree splits
    with their own domain-knowledge-based cutpoints.
    
    Parameters:
    -----------
    bin_result : BinResult
        Existing binning results to update
    var : str
        Variable to re-bin
    y_var : str
        Target variable name
    splits : List[float]
        Custom split points
    df : pd.DataFrame
        The data
    
    Returns:
    --------
    BinResult : Updated binning results with new splits for the variable
    """
    # Create new bins using the specified splits
    bin_df = _create_numeric_bins(df, var, y_var, splits)
    
    # If no bins created, return unchanged
    if bin_df.empty:
        return bin_result
    
    # Calculate statistics and add total row
    bin_df = update_bin_stats(bin_df)
    bin_df = add_total_row(bin_df, var)
    
    # Replace old bins for this variable with new bins
    # Keep bins for all other variables unchanged
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
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
# SECTION 8: BIN OPERATIONS FUNCTIONS
# =============================================================================
# These functions modify existing bins (combine, split, force trends, etc.)

def _parse_numeric_from_rule(rule: str) -> List[float]:
    """
    Extract numeric values from a bin rule string.
    
    Bin rules like "Age > '25' & Age <= '35'" contain numeric thresholds
    enclosed in single quotes. This function extracts them.
    
    Parameters:
    -----------
    rule : str
        A bin rule string
    
    Returns:
    --------
    List[float] : List of numeric values found in the rule
    """
    # Regular expression pattern to match numbers in single quotes
    # r"'(-?\d+\.?\d*)'" matches: 'optional-minus digit(s) optional-decimal digit(s)'
    pattern = r"'(-?\d+\.?\d*)'"
    
    # Find all matches
    matches = re.findall(pattern, rule)
    
    # Convert strings to floats
    return [float(m) for m in matches]


def _parse_factor_values_from_rule(rule: str) -> List[str]:
    """
    Extract factor values from a bin rule string.
    
    Bin rules like 'Status %in% c("Active", "Pending")' contain categorical
    values enclosed in double quotes. This function extracts them.
    
    Parameters:
    -----------
    rule : str
        A bin rule string
    
    Returns:
    --------
    List[str] : List of categorical values found in the rule
    """
    # Regular expression pattern to match strings in double quotes
    # r'"([^"]*)"' matches: anything between double quotes
    pattern = r'"([^"]*)"'
    
    # Find all matches
    matches = re.findall(pattern, rule)
    
    return matches


def _process_na_combine_single_var(
    new_bins: pd.DataFrame,
    var: str
) -> Tuple[str, pd.DataFrame, Optional[Dict]]:
    """
    Process NA combine for a single variable - for parallel execution.
    
    This function merges the NA bin with the bin that has the most similar
    bad rate. This is useful because:
    1. Reduces number of bins
    2. NA values are often similar to some observed pattern
    3. Makes the binning more robust
    
    Parameters:
    -----------
    new_bins : pd.DataFrame
        All bins (may contain multiple variables)
    var : str
        Variable to process
    
    Returns:
    --------
    Tuple of (var name, processed bins DataFrame, var_summary update dict or None)
    """
    # Get bins for this variable only
    var_bins = new_bins[new_bins['var'] == var].copy()
    
    # Return empty if no bins
    if var_bins.empty:
        return var, pd.DataFrame(), None
    
    # Find the NA bin using string matching
    na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
    
    # If no NA bin exists, return unchanged
    if not na_mask.any():
        return var, var_bins, None
    
    # Get the NA bin and non-NA bins (excluding Total row)
    na_bin = var_bins[na_mask].iloc[0]
    non_na_bins = var_bins[~na_mask & (var_bins['bin'] != 'Total')]
    
    # If no non-NA bins to merge with, return unchanged
    if non_na_bins.empty:
        return var, var_bins, None
    
    # Calculate bad rate for NA bin
    na_bad_rate = na_bin['bads'] / na_bin['count'] if na_bin['count'] > 0 else 0
    
    # Find the non-NA bin with the closest bad rate
    non_na_bins = non_na_bins.copy()
    non_na_bins['bad_rate_calc'] = non_na_bins['bads'] / non_na_bins['count']
    non_na_bins['rate_diff'] = abs(non_na_bins['bad_rate_calc'] - na_bad_rate)
    
    # Get the index of the closest bin
    closest_idx = non_na_bins['rate_diff'].idxmin()
    closest_bin = non_na_bins.loc[closest_idx]
    
    # Create a combined bin rule (original rule | NA condition)
    combined_rule = f"{closest_bin['bin']} | is.na({var})"
    
    # Combine counts
    combined_count = closest_bin['count'] + na_bin['count']
    combined_goods = closest_bin['goods'] + na_bin['goods']
    combined_bads = closest_bin['bads'] + na_bin['bads']
    
    # Update the closest bin with combined values
    modified_bins = var_bins.copy()
    modified_bins.loc[closest_idx, 'bin'] = combined_rule
    modified_bins.loc[closest_idx, 'count'] = combined_count
    modified_bins.loc[closest_idx, 'goods'] = combined_goods
    modified_bins.loc[closest_idx, 'bads'] = combined_bads
    
    # Remove the original NA bin (now merged)
    na_idx = var_bins[na_mask].index[0]
    modified_bins = modified_bins.drop(na_idx)
    
    # Recalculate statistics with the merged bins
    var_new_bins = modified_bins[modified_bins['bin'] != 'Total'].copy()
    var_new_bins = update_bin_stats(var_new_bins)
    var_new_bins = add_total_row(var_new_bins, var)
    
    # Prepare summary update
    total_row = var_new_bins[var_new_bins['bin'] == 'Total'].iloc[0]
    var_summary_update = {
        'var': var,
        'iv': total_row['iv'],
        'ent': total_row['ent'],
        'trend': total_row['trend'],
        'monTrend': total_row.get('monTrend', 'N'),
        'flipRatio': total_row.get('flipRatio', 0),
        'numBins': total_row.get('numBins', len(var_new_bins) - 1),
        'purNode': total_row['purNode']
    }
    
    return var, var_new_bins, var_summary_update


def na_combine(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    n_jobs: Optional[int] = None
) -> BinResult:
    """
    Combine NA bin with the adjacent bin that has the closest bad rate.
    Uses parallel processing for multiple variables.
    
    This operation is commonly used in WOE binning to handle missing values
    by merging them with a behaviorally similar group.
    
    Parameters:
    -----------
    bin_result : BinResult
        Current binning results
    vars_to_process : str or List[str]
        Variable(s) to process
    n_jobs : int, optional
        Number of parallel workers
    
    Returns:
    --------
    BinResult : Updated binning results with NA bins merged
    """
    # Handle single variable as list
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    # Use global N_JOBS if not specified
    if n_jobs is None:
        n_jobs = N_JOBS
    
    # Make copies to avoid modifying originals
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process variables in parallel if multiple
    if len(vars_to_process) > 1 and n_jobs > 1:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_na_combine_single_var)(new_bins, var)
                for var in vars_to_process
            )
        except:
            # Fall back to sequential if parallel fails
            results = [_process_na_combine_single_var(new_bins, var) for var in vars_to_process]
    else:
        results = [_process_na_combine_single_var(new_bins, var) for var in vars_to_process]
    
    # Combine results from all workers
    processed_vars = set()
    all_processed_bins = []
    
    for var, var_bins, var_summary_update in results:
        if not var_bins.empty:
            processed_vars.add(var)
            all_processed_bins.append(var_bins)
            
            # Update var_summary with new values
            if var_summary_update is not None:
                mask = var_summary['var'] == var
                if mask.any():
                    for key, value in var_summary_update.items():
                        if key != 'var' and key in var_summary.columns:
                            var_summary.loc[mask, key] = value
    
    # Combine unprocessed vars with processed vars
    unprocessed_bins = new_bins[~new_bins['var'].isin(processed_vars)]
    if all_processed_bins:
        final_bins = pd.concat([unprocessed_bins] + all_processed_bins, ignore_index=True)
    else:
        final_bins = unprocessed_bins
    
    return BinResult(var_summary=var_summary, bin=final_bins)


def merge_pure_bins(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None
) -> BinResult:
    """
    Merge pure bins (100% goods or 100% bads) with the closest non-pure bin.
    
    Pure bins cause infinite WOE values (division by zero) which break
    logistic regression. This function iteratively merges pure bins
    until no pure bins remain.
    
    Parameters:
    -----------
    bin_result : BinResult
        Current binning results
    vars_to_process : str or List[str], optional
        Variable(s) to process. If None, process all variables.
    
    Returns:
    --------
    BinResult : Updated results with pure bins merged
    """
    # Default to all variables if not specified
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Process each variable
    for var in vars_to_process:
        max_iterations = 100  # Safety limit to prevent infinite loops
        iteration = 0
        
        # Keep merging until no pure bins remain or max iterations reached
        while iteration < max_iterations:
            iteration += 1
            
            # Get non-Total bins for this variable
            var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
            
            # If only one bin left, can't merge further
            if len(var_bins) <= 1:
                break
            
            # Find pure bins (bins where goods=0 OR bads=0)
            pure_mask = (var_bins['goods'] == 0) | (var_bins['bads'] == 0)
            
            # If no pure bins, we're done with this variable
            if not pure_mask.any():
                break
            
            # Get the first pure bin to process
            pure_bin = var_bins[pure_mask].iloc[0]
            pure_idx = var_bins[pure_mask].index[0]
            
            # Find non-pure bins to potentially merge with
            non_pure_bins = var_bins[~pure_mask]
            
            if non_pure_bins.empty:
                # All bins are pure - merge with closest by count
                other_bins = var_bins[var_bins.index != pure_idx]
                if other_bins.empty:
                    break
                other_bins = other_bins.copy()
                other_bins['count_diff'] = abs(other_bins['count'] - pure_bin['count'])
                closest_idx = other_bins['count_diff'].idxmin()
                closest_bin = other_bins.loc[closest_idx]
            else:
                # Find non-pure bin with closest bad rate
                pure_bad_rate = pure_bin['bads'] / pure_bin['count'] if pure_bin['count'] > 0 else 0.5
                
                non_pure_bins = non_pure_bins.copy()
                non_pure_bins['bad_rate_calc'] = non_pure_bins['bads'] / non_pure_bins['count']
                non_pure_bins['rate_diff'] = abs(non_pure_bins['bad_rate_calc'] - pure_bad_rate)
                
                closest_idx = non_pure_bins['rate_diff'].idxmin()
                closest_bin = non_pure_bins.loc[closest_idx]
            
            # Merge the pure bin into the closest bin
            # Create combined rule
            combined_rule = f"({closest_bin['bin']}) | ({pure_bin['bin']})"
            
            # Sum the counts
            combined_count = closest_bin['count'] + pure_bin['count']
            combined_goods = closest_bin['goods'] + pure_bin['goods']
            combined_bads = closest_bin['bads'] + pure_bin['bads']
            
            # Update the closest bin with combined values
            new_bins.loc[closest_idx, 'bin'] = combined_rule
            new_bins.loc[closest_idx, 'count'] = combined_count
            new_bins.loc[closest_idx, 'goods'] = combined_goods
            new_bins.loc[closest_idx, 'bads'] = combined_bads
            
            # Remove the pure bin (now merged)
            new_bins = new_bins.drop(pure_idx)
        
        # Recalculate statistics after all merging
        var_new_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        if not var_new_bins.empty:
            var_new_bins = update_bin_stats(var_new_bins)
            var_new_bins = add_total_row(var_new_bins, var)
            
            # Replace variable bins in the main DataFrame
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
                var_summary.loc[mask, 'purNode'] = 'N'  # No more pure nodes after merging
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def break_bin(
    bin_result: BinResult,
    var: str,
    y_var: str,
    df: pd.DataFrame
) -> BinResult:
    """
    Break all bins for a factor variable - each unique value becomes its own bin.
    
    This is useful when you want to reset a variable's binning to the most
    granular level (one bin per unique value).
    
    Parameters:
    -----------
    bin_result : BinResult
        Current binning results
    var : str
        Variable to break
    y_var : str
        Target variable name
    df : pd.DataFrame
        The data
    
    Returns:
    --------
    BinResult : Updated results with variable broken into individual value bins
    """
    # Create new bins with each value as its own bin
    new_var_bins = _create_factor_bins(df, var, y_var)
    
    # Calculate statistics
    new_var_bins = update_bin_stats(new_var_bins)
    new_var_bins = add_total_row(new_var_bins, var)
    
    # Replace old bins for this variable
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
    new_bins = pd.concat([other_bins, new_var_bins], ignore_index=True)
    
    # Update var_summary
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


def _process_force_trend_single_var(
    var_bins: pd.DataFrame,
    var: str,
    increasing: bool
) -> Tuple[str, pd.DataFrame, Optional[Dict]]:
    """
    Process force trend for a single variable - for parallel execution.
    
    This function merges adjacent bins until the bad rate trend is monotonic
    (either always increasing or always decreasing).
    
    Monotonic trends are preferred in credit scoring because:
    1. They're easier to interpret (higher value = higher/lower risk)
    2. They prevent non-intuitive score behavior
    3. They're more stable over time
    
    Parameters:
    -----------
    var_bins : pd.DataFrame
        Bins for this variable (excluding Total row)
    var : str
        Variable name
    increasing : bool
        If True, force increasing trend. If False, force decreasing.
    
    Returns:
    --------
    Tuple of (var name, processed bins DataFrame, var_summary update dict or None)
    """
    # Return unchanged if empty or only one bin
    if var_bins.empty or len(var_bins) < 2:
        return var, var_bins, None
    
    # Separate NA bins from regular bins (NA bins don't participate in trend)
    na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
    na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
    working_bins = var_bins[~na_mask].copy()
    
    if working_bins.empty:
        return var, var_bins, None
    
    # Reset index for easier manipulation
    working_bins = working_bins.reset_index(drop=True)
    
    # Iteratively merge bins until trend is monotonic
    changed = True
    while changed and len(working_bins) > 1:
        changed = False
        
        # Recalculate bad rates
        working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
        
        # Check each adjacent pair of bins
        for i in range(1, len(working_bins)):
            # Determine if this pair violates the desired trend
            should_merge = (
                # For increasing: current should be >= previous
                (increasing and working_bins.iloc[i]['bad_rate_calc'] < working_bins.iloc[i-1]['bad_rate_calc']) or
                # For decreasing: current should be <= previous
                (not increasing and working_bins.iloc[i]['bad_rate_calc'] > working_bins.iloc[i-1]['bad_rate_calc'])
            )
            
            if should_merge:
                # Merge bin i into bin i-1
                # Add counts to previous bin
                working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                
                # Update the bin rule to reflect the merge
                old_rule = working_bins.iloc[i-1]['bin']
                new_rule = working_bins.iloc[i]['bin']
                
                # Try to create a clean combined rule for numeric variables
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
                break  # Restart the loop after merge
    
    # Add NA bin back if it existed
    if not na_bin.empty:
        working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
    
    # Clean up temporary column
    if 'bad_rate_calc' in working_bins.columns:
        working_bins = working_bins.drop('bad_rate_calc', axis=1)
    
    # Recalculate statistics
    working_bins = update_bin_stats(working_bins)
    working_bins = add_total_row(working_bins, var)
    
    # Prepare summary update
    total_row = working_bins[working_bins['bin'] == 'Total'].iloc[0]
    var_summary_update = {
        'var': var,
        'iv': total_row['iv'],
        'ent': total_row['ent'],
        'trend': total_row['trend'],
        'monTrend': total_row.get('monTrend', 'Y'),  # Should be Y after forcing
        'flipRatio': total_row.get('flipRatio', 0),
        'numBins': total_row.get('numBins', len(working_bins) - 1),
        'purNode': total_row['purNode']
    }
    
    return var, working_bins, var_summary_update


def force_incr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    n_jobs: Optional[int] = None
) -> BinResult:
    """
    Force an increasing monotonic trend in bad rates by combining adjacent bins.
    Uses parallel processing for multiple variables.
    
    After this operation, bad_rate[i] >= bad_rate[i-1] for all bins.
    
    Parameters:
    -----------
    bin_result : BinResult
        Current binning results
    vars_to_process : str or List[str]
        Variable(s) to process
    n_jobs : int, optional
        Number of parallel workers
    
    Returns:
    --------
    BinResult : Updated results with increasing trend enforced
    """
    # Handle single variable as list
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    if n_jobs is None:
        n_jobs = N_JOBS
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Prepare data for each variable (bins, name, direction flag)
    var_data = []
    for var in vars_to_process:
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        var_data.append((var_bins, var, True))  # True = increasing
    
    # Process in parallel
    if len(vars_to_process) > 1 and n_jobs > 1:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_force_trend_single_var)(vb, v, inc)
                for vb, v, inc in var_data
            )
        except:
            results = [_process_force_trend_single_var(vb, v, inc) for vb, v, inc in var_data]
    else:
        results = [_process_force_trend_single_var(vb, v, inc) for vb, v, inc in var_data]
    
    # Combine results
    for var, working_bins, var_summary_update in results:
        if not working_bins.empty:
            # Replace bins for this variable
            new_bins = new_bins[new_bins['var'] != var]
            new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
            
            # Update summary
            if var_summary_update is not None:
                mask = var_summary['var'] == var
                if mask.any():
                    for key, value in var_summary_update.items():
                        if key != 'var' and key in var_summary.columns:
                            var_summary.loc[mask, key] = value
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def force_decr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    n_jobs: Optional[int] = None
) -> BinResult:
    """
    Force a decreasing monotonic trend in bad rates by combining adjacent bins.
    Uses parallel processing for multiple variables.
    
    After this operation, bad_rate[i] <= bad_rate[i-1] for all bins.
    
    Parameters:
    -----------
    bin_result : BinResult
        Current binning results
    vars_to_process : str or List[str]
        Variable(s) to process
    n_jobs : int, optional
        Number of parallel workers
    
    Returns:
    --------
    BinResult : Updated results with decreasing trend enforced
    """
    # Handle single variable as list
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    if n_jobs is None:
        n_jobs = N_JOBS
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    # Prepare data for each variable
    var_data = []
    for var in vars_to_process:
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        var_data.append((var_bins, var, False))  # False = decreasing
    
    # Process in parallel
    if len(vars_to_process) > 1 and n_jobs > 1:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_force_trend_single_var)(vb, v, inc)
                for vb, v, inc in var_data
            )
        except:
            results = [_process_force_trend_single_var(vb, v, inc) for vb, v, inc in var_data]
    else:
        results = [_process_force_trend_single_var(vb, v, inc) for vb, v, inc in var_data]
    
    # Combine results
    for var, working_bins, var_summary_update in results:
        if not working_bins.empty:
            new_bins = new_bins[new_bins['var'] != var]
            new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
            
            if var_summary_update is not None:
                mask = var_summary['var'] == var
                if mask.any():
                    for key, value in var_summary_update.items():
                        if key != 'var' and key in var_summary.columns:
                            var_summary.loc[mask, key] = value
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def _process_binned_column_single_var(
    df: pd.DataFrame,
    var: str,
    var_bins: pd.DataFrame,
    prefix: str
) -> Tuple[str, pd.Series]:
    """
    Process binned column creation for a single variable - for parallel execution.
    
    This function assigns each observation to a bin based on the bin rules.
    The result is a new column where each row contains the bin label.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data
    var : str
        Variable name
    var_bins : pd.DataFrame
        Bin definitions for this variable
    prefix : str
        Prefix for the new column name (e.g., 'b_')
    
    Returns:
    --------
    Tuple of (new column name, Series with bin assignments)
    """
    # Create column name (e.g., "b_Age" for variable "Age")
    new_col_name = prefix + var
    
    # Initialize new column with None values
    new_col = pd.Series(index=df.index, dtype=object)
    new_col[:] = None
    
    # Track if we found an NA rule that's combined with another bin
    na_rule = None
    
    # Process each bin rule
    for _, row in var_bins.iterrows():
        rule = row['bin']
        
        # Extract the bin value (for display in the column)
        # Remove the variable name and %in% c parts
        bin_value = rule.replace(var, '').replace(' %in% c', '').strip()
        
        # Check if this rule includes NA handling
        if '| is.na' in rule:
            na_rule = bin_value  # Save for later NA handling
            main_rule = rule.split('|')[0].strip()  # Use the non-NA part of the rule
        else:
            main_rule = rule
        
        try:
            is_na_bin = False
            
            # Determine the type of rule and create appropriate mask
            if 'is.na' in main_rule and '|' not in main_rule:
                # Standalone NA bin
                mask = df[var].isna()
                is_na_bin = True
            elif '%in%' in main_rule:
                # Factor variable: check if value is in the list
                values = _parse_factor_values_from_rule(main_rule)
                mask = df[var].isin(values)
            elif '<=' in main_rule and '>' in main_rule:
                # Numeric range: lower < x <= upper
                nums = _parse_numeric_from_rule(main_rule)
                if len(nums) >= 2:
                    lower, upper = min(nums), max(nums)
                    mask = (df[var] > lower) & (df[var] <= upper)
                else:
                    continue
            elif '<=' in main_rule:
                # Upper bound only: x <= upper
                nums = _parse_numeric_from_rule(main_rule)
                if nums:
                    upper = max(nums)
                    mask = df[var] <= upper
                else:
                    continue
            elif '>' in main_rule:
                # Lower bound only: x > lower
                nums = _parse_numeric_from_rule(main_rule)
                if nums:
                    lower = min(nums)
                    mask = df[var] > lower
                else:
                    continue
            elif '==' in main_rule:
                # Exact equality
                nums = _parse_numeric_from_rule(main_rule)
                if nums:
                    new_col.loc[df[var] == nums[0]] = bin_value
                continue
            else:
                continue
            
            # Apply the mask
            if is_na_bin:
                # For NA bins, apply mask directly
                new_col.loc[mask] = bin_value
            else:
                # For other bins, exclude NA values
                new_col.loc[mask & df[var].notna()] = bin_value
            
        except Exception:
            continue
    
    # Handle NA values
    if na_rule is not None:
        # If we found a combined NA rule, assign NA rows to it
        new_col.loc[df[var].isna()] = na_rule
    elif df[var].isna().any():
        # If there's a standalone NA bin, find and use it
        na_bins = var_bins[var_bins['bin'].str.match(r'^is\.na\(', na=False)]
        if not na_bins.empty:
            bin_value = na_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
            new_col.loc[df[var].isna()] = bin_value
    
    # Handle any remaining unassigned rows (edge cases)
    unassigned_mask = new_col.isna() | (new_col == None)
    if unassigned_mask.any():
        # Assign to first available bin as fallback
        if na_rule is not None:
            fallback_bin = na_rule
        elif not var_bins.empty:
            fallback_bin = var_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
        else:
            fallback_bin = "Unmatched"
        new_col.loc[unassigned_mask] = fallback_bin
    
    return new_col_name, new_col


def create_binned_columns(
    bin_result: BinResult,
    df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Create binned columns in the DataFrame based on binning rules.
    Uses parallel processing for multiple variables.
    
    This function adds new columns (e.g., b_Age, b_Income) that contain
    the bin label for each observation.
    
    Parameters:
    -----------
    bin_result : BinResult
        Binning results containing the bin rules
    df : pd.DataFrame
        The data
    x_vars : List[str]
        Variables to create binned columns for
    prefix : str
        Prefix for new column names (default: "b_")
    n_jobs : int, optional
        Number of parallel workers
    
    Returns:
    --------
    pd.DataFrame : Original data with binned columns added
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    # Start with a copy of the original data
    result_df = df.copy()
    
    # Prepare data for parallel processing
    var_data = []
    for var in x_vars:
        # Get bins for this variable (excluding Total row)
        var_bins = bin_result.bin[(bin_result.bin['var'] == var) & 
                                   (bin_result.bin['bin'] != 'Total')]
        if not var_bins.empty:
            var_data.append((df, var, var_bins, prefix))
    
    if not var_data:
        return result_df
    
    # Process in parallel
    if len(var_data) > 1 and n_jobs > 1:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_binned_column_single_var)(d, v, vb, p)
                for d, v, vb, p in var_data
            )
        except:
            results = [_process_binned_column_single_var(d, v, vb, p) for d, v, vb, p in var_data]
    else:
        results = [_process_binned_column_single_var(d, v, vb, p) for d, v, vb, p in var_data]
    
    # Add all new columns to result DataFrame
    for col_name, col_series in results:
        result_df[col_name] = col_series
    
    return result_df


def _process_woe_column_single_var(
    df: pd.DataFrame,
    var: str,
    var_bins: pd.DataFrame,
    prefix: str,
    woe_prefix: str
) -> Tuple[str, pd.Series]:
    """
    Process WOE column creation for a single variable - for parallel execution.
    
    This function maps each observation's bin to its WOE value.
    Missing/unmatched bin values are assigned WOE=0 (neutral - no information).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with binned columns already added
    var : str
        Variable name
    var_bins : pd.DataFrame
        Bin definitions with WOE values
    prefix : str
        Prefix for binned column (e.g., 'b_')
    woe_prefix : str
        Prefix for WOE column (e.g., 'WOE_')
    
    Returns:
    --------
    Tuple of (WOE column name, Series with WOE values)
    """
    # Calculate WOE if not already present
    if 'woe' not in var_bins.columns:
        var_bins = var_bins.copy()
        var_bins['woe'] = calculate_woe(var_bins['goods'].values, var_bins['bads'].values)
    
    # Create bin value mapping (extract the display value from the rule)
    var_bins['binValue'] = var_bins['bin'].apply(
        lambda x: x.replace(var, '').replace(' %in% c', '').strip()
    )
    
    # Column names
    bin_col = prefix + var  # e.g., 'b_Age'
    woe_col = woe_prefix + var  # e.g., 'WOE_Age'
    
    # Map bin values to WOE values
    if bin_col in df.columns:
        # Create mapping dictionary: bin_value -> WOE
        woe_map = dict(zip(var_bins['binValue'], var_bins['woe']))
        
        # Apply mapping
        woe_series = df[bin_col].map(woe_map)
        
        # Check for unmapped bins (indicates a bug)
        missing_woe_count = woe_series.isna().sum()
        if missing_woe_count > 0:
            unmapped_bins = df.loc[woe_series.isna(), bin_col].unique()
            print(f"[ERROR] {var}: {missing_woe_count} rows have unmapped bin values!")
            print(f"        Unmapped bins: {list(unmapped_bins)}")
            print(f"        Available bins in woe_map: {list(woe_map.keys())}")
            
            # Try to fix unmapped bins
            for unmapped_bin in unmapped_bins:
                if unmapped_bin is None or pd.isna(unmapped_bin):
                    # Find NA bin WOE
                    na_woe_bins = var_bins[var_bins['bin'].str.contains('is.na', na=False)]
                    if not na_woe_bins.empty:
                        na_woe = na_woe_bins.iloc[0]['woe']
                        woe_series.loc[df[bin_col].isna()] = na_woe
                        print(f"        -> Assigned NA bin WOE: {na_woe}")
                else:
                    # Try exact match in original bin rules
                    for _, bin_row in var_bins.iterrows():
                        if unmapped_bin in bin_row['bin'] or bin_row['binValue'] == unmapped_bin:
                            woe_series.loc[df[bin_col] == unmapped_bin] = bin_row['woe']
                            print(f"        -> Matched '{unmapped_bin}' to WOE: {bin_row['woe']}")
                            break
        
        return woe_col, woe_series
    
    # Return empty series if bin column doesn't exist
    return woe_col, pd.Series(index=df.index, dtype=float)


def add_woe_columns(
    df: pd.DataFrame,
    bins_df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    woe_prefix: str = "WOE_",
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Add WOE columns to the DataFrame by joining with binning rules.
    Uses parallel processing for multiple variables.
    
    This function adds WOE columns (e.g., WOE_Age, WOE_Income) that contain
    the Weight of Evidence value for each observation's bin.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with binned columns already added
    bins_df : pd.DataFrame
        Binning rules with WOE values
    x_vars : List[str]
        Variables to create WOE columns for
    prefix : str
        Prefix for binned columns (default: "b_")
    woe_prefix : str
        Prefix for WOE columns (default: "WOE_")
    n_jobs : int, optional
        Number of parallel workers
    
    Returns:
    --------
    pd.DataFrame : Data with WOE columns added
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    result_df = df.copy()
    
    # Prepare data for parallel processing
    var_data = []
    for var in x_vars:
        var_bins = bins_df[(bins_df['var'] == var) & (bins_df['bin'] != 'Total')].copy()
        if not var_bins.empty:
            var_data.append((df, var, var_bins, prefix, woe_prefix))
    
    if not var_data:
        return result_df
    
    # Process in parallel
    if len(var_data) > 1 and n_jobs > 1:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_woe_column_single_var)(d, v, vb, p, wp)
                for d, v, vb, p, wp in var_data
            )
        except:
            results = [_process_woe_column_single_var(d, v, vb, p, wp) for d, v, vb, p, wp in var_data]
    else:
        results = [_process_woe_column_single_var(d, v, vb, p, wp) for d, v, vb, p, wp in var_data]
    
    # Add all WOE columns to result DataFrame
    for col_name, col_series in results:
        if not col_series.empty:
            result_df[col_name] = col_series
    
    return result_df


# =============================================================================
# SECTION 9: SHINY UI APPLICATION
# =============================================================================
# This section defines the interactive web UI for manual WOE editing

def create_woe_editor_app(df: pd.DataFrame, min_prop: float = 0.01):
    """
    Create the WOE Editor Shiny application.
    
    This function builds a complete web application that allows users to:
    - Select dependent and independent variables
    - View bin statistics and charts
    - Manually adjust binning (group NA, break bins, reset)
    - Force monotonic trends
    - Export final WOE values
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    min_prop : float
        Minimum proportion for binning
    
    Returns:
    --------
    App : Shiny application object
    """
    
    # Dictionary to store results when user clicks "Run & Close"
    # This is how results are passed back to the main script
    app_results = {
        'df_with_woe': None,
        'df_only_woe': None,
        'bins': None,
        'dv': None,
        'completed': False
    }
    
    # Define the UI layout using Shiny's fluid page layout
    app_ui = ui.page_fluid(
        # Inject custom CSS styles into the page header
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
        ui.h4("WOE Editor (Parallel Processing Enabled)"),
        
        # Card 1: Variable selection dropdowns
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    # Dropdown for dependent variable (target)
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=list(df.columns),
                                   selected=df.columns[0] if len(df.columns) > 0 else None)
                ),
                ui.column(6,
                    # Dropdown for target category (which value is "bad")
                    ui.input_select("tc", "Target Category", choices=[])
                )
            )
        ),
        
        # Card 2: Independent variable selection and action buttons
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    # Dropdown for independent variable
                    ui.input_select("iv", "Independent Variable", choices=[]),
                    # Navigation buttons to move through variables
                    ui.div(
                        ui.input_action_button("prev_btn", "← Previous", class_="btn btn-secondary"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("next_btn", "Next →", class_="btn btn-success"),
                    )
                ),
                ui.column(6,
                    # Bin manipulation buttons
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
        
        # Row: WOE table and chart side by side
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px; overflow-y: auto;"},
                    ui.h5("Bin Details"),
                    # Data grid showing bin statistics
                    ui.output_data_frame("woe_table")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px;"},
                    ui.h5("WOE & Bad Rate"),
                    # Plotly chart showing WOE and rates
                    output_widget("woe_graph")
                )
            )
        ),
        
        # Row: Count and proportion bar charts
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 350px;"},
                    output_widget("count_bar")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 350px;"},
                    output_widget("prop_bar")
                )
            )
        ),
        
        # Measurements table (IV and Entropy before/after)
        ui.div(
            {"class": "card"},
            ui.h5("Measurements"),
            ui.output_data_frame("measurements_table")
        ),
        
        # Run button to finalize and close
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "Run & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    # Server function: contains all the reactive logic
    def server(input: Inputs, output: Outputs, session: Session):
        # Reactive values to store state between user interactions
        bins_rv = reactive.Value(None)  # Current variable's bins
        all_bins_rv = reactive.Value(None)  # All variables' bins
        all_bins_mod_rv = reactive.Value(None)  # Modified bins (after Optimize All)
        modified_action_rv = reactive.Value(False)  # Flag if Optimize All was used
        initial_bins_rv = reactive.Value(None)  # Original bins for comparison
        
        # Effect: Update target category dropdown when DV changes
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            if dv and dv in df.columns:
                # Get unique values of the dependent variable
                unique_vals = df[dv].dropna().unique().tolist()
                # Update target category dropdown
                ui.update_select("tc", choices=unique_vals, 
                               selected=max(unique_vals) if unique_vals else None)
                
                # Create list of independent variables (all except DV)
                iv_list = [col for col in df.columns if col != dv]
                
                # Calculate bins for all IVs if DV has no missing values
                if df[dv].isna().sum() <= 0:
                    try:
                        all_bins = get_bins(df, dv, iv_list, min_prop=min_prop)
                        all_bins_rv.set(all_bins)
                        
                        # Update IV dropdown with binned variables
                        bin_vars = all_bins.var_summary['var'].tolist()
                        ui.update_select("iv", choices=bin_vars, 
                                       selected=bin_vars[0] if bin_vars else None)
                    except Exception as e:
                        print(f"Error calculating bins: {e}")
        
        # Effect: Update bins when IV selection changes
        @reactive.Effect
        @reactive.event(input.iv)
        def update_iv_bins():
            iv = input.iv()
            dv = input.dv()
            if iv and dv and not modified_action_rv.get():
                try:
                    bins = get_bins(df, dv, [iv], min_prop=min_prop)
                    bins_rv.set(bins)
                    initial_bins_rv.set(bins)  # Store for comparison
                except Exception as e:
                    print(f"Error getting bins for {iv}: {e}")
        
        # Effect: Previous button - navigate to previous variable
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
        
        # Effect: Next button - navigate to next variable
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
        
        # Effect: Group NA button - merge NA bin with closest bin
        @reactive.Effect
        @reactive.event(input.group_na_btn)
        def group_na():
            bins = bins_rv.get()
            iv = input.iv()
            if bins is not None and iv:
                new_bins = na_combine(bins, iv)
                bins_rv.set(new_bins)
                modified_action_rv.set(False)
        
        # Effect: Break Bin button - split variable into individual value bins
        @reactive.Effect
        @reactive.event(input.break_btn)
        def break_bins():
            bins = bins_rv.get()
            iv = input.iv()
            dv = input.dv()
            if bins is not None and iv and dv:
                new_bins = break_bin(bins, iv, dv, df)
                bins_rv.set(new_bins)
                modified_action_rv.set(False)
        
        # Effect: Reset button - return to original binning
        @reactive.Effect
        @reactive.event(input.reset_btn)
        def reset_bins():
            iv = input.iv()
            dv = input.dv()
            if iv and dv:
                modified_action_rv.set(False)
                bins = get_bins(df, dv, [iv], min_prop=min_prop)
                bins_rv.set(bins)
        
        # Effect: Optimize button - force monotonic trend for current variable
        @reactive.Effect
        @reactive.event(input.optimize_btn)
        def optimize_var():
            bins = bins_rv.get()
            iv = input.iv()
            if bins is not None and iv:
                var_info = bins.var_summary[bins.var_summary['var'] == iv]
                if not var_info.empty:
                    trend = var_info.iloc[0]['trend']
                    # Force trend based on current dominant direction
                    if trend == 'I':
                        new_bins = force_incr_trend(bins, iv)
                    elif trend == 'D':
                        new_bins = force_decr_trend(bins, iv)
                    else:
                        new_bins = bins
                    bins_rv.set(new_bins)
                    modified_action_rv.set(False)
        
        # Effect: Optimize All button - optimize all variables at once
        @reactive.Effect
        @reactive.event(input.optimize_all_btn)
        def optimize_all():
            all_bins = all_bins_rv.get()
            if all_bins is not None:
                modified_action_rv.set(True)
                
                # First, combine NA bins for all variables
                bins_mod = na_combine(all_bins, all_bins.var_summary['var'].tolist())
                
                # Force decreasing trend for variables with D trend
                decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
                if decr_vars:
                    bins_mod = force_decr_trend(bins_mod, decr_vars)
                
                # Force increasing trend for variables with I trend
                incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
                if incr_vars:
                    bins_mod = force_incr_trend(bins_mod, incr_vars)
                
                all_bins_mod_rv.set(bins_mod)
        
        # Reactive calculation: Get bins to display (considers Optimize All state)
        @reactive.Calc
        def get_display_bins():
            if modified_action_rv.get():
                # Use optimized bins if Optimize All was clicked
                all_mod = all_bins_mod_rv.get()
                iv = input.iv()
                if all_mod is not None and iv:
                    var_bins = all_mod.bin[all_mod.bin['var'] == iv].copy()
                    return var_bins
            else:
                # Use current variable's bins
                bins = bins_rv.get()
                if bins is not None:
                    return bins.bin.copy()
            return pd.DataFrame()
        
        # Output: WOE table showing bin details
        @output
        @render.data_frame
        def woe_table():
            display_bins = get_display_bins()
            if display_bins.empty:
                return render.DataGrid(pd.DataFrame())
            
            # Calculate WOE for non-total rows
            non_total = display_bins[display_bins['bin'] != 'Total'].copy()
            if not non_total.empty:
                non_total['woe'] = calculate_woe(non_total['goods'].values, non_total['bads'].values)
            
            # Total row doesn't have WOE
            total_row = display_bins[display_bins['bin'] == 'Total'].copy()
            if not total_row.empty:
                total_row['woe'] = np.nan
            
            # Combine and select display columns
            result = pd.concat([non_total, total_row], ignore_index=True)
            
            display_cols = ['bin', 'count', 'goods', 'bads', 'propn', 'bad_rate', 'woe', 'iv']
            display_cols = [c for c in display_cols if c in result.columns]
            
            return render.DataGrid(result[display_cols], selection_mode="rows", height="350px")
        
        # Output: WOE and rate chart
        @output
        @render_plotly
        def woe_graph():
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            # Exclude total row from chart
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            # Calculate WOE for plotting
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
            # Create figure with multiple traces
            fig = go.Figure()
            
            # Bad rate line
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=plot_data['bad_rate'] / 100,
                name='Bad Rate', mode='lines+markers', line=dict(color='#3498db')
            ))
            
            # Good rate line
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=(100 - plot_data['bad_rate']) / 100,
                name='Good Rate', mode='lines+markers', line=dict(color='#2ecc71')
            ))
            
            # WOE line on secondary y-axis
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
        
        # Output: Count distribution bar chart
        @output
        @render_plotly
        def count_bar():
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
            # Bar chart with hover info
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
        
        # Output: Good/Bad proportion stacked bar chart
        @output
        @render_plotly
        def prop_bar():
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            
            # Good proportion (horizontal stacked bar)
            fig.add_trace(go.Bar(
                y=plot_data['bin'], x=100 - plot_data['bad_rate'],
                name='Good', orientation='h', marker_color='#9ECC53',
                text=100 - plot_data['bad_rate'], textposition='inside'
            ))
            
            # Bad proportion
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
        
        # Output: Measurements comparison table
        @output
        @render.data_frame
        def measurements_table():
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
            
            # Get initial values from stored initial bins
            initial_iv = 0
            initial_ent = 0
            if initial is not None:
                init_total = initial.bin[initial.bin['bin'] == 'Total']
                if not init_total.empty:
                    initial_iv = init_total['iv'].iloc[0]
                    initial_ent = init_total['ent'].iloc[0]
            
            # Create comparison table
            measurements = pd.DataFrame({
                'Initial IV': [round(initial_iv, 4)],
                'Final IV': [round(final_iv, 4)],
                'Initial Entropy': [round(initial_ent, 4)],
                'Final Entropy': [round(final_ent, 4)]
            })
            
            return render.DataGrid(measurements)
        
        # Effect: Run & Close button - finalize and close the app
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_and_close():
            dv = input.dv()
            
            # Use optimized bins if available, else use original
            if modified_action_rv.get():
                final_bins = all_bins_mod_rv.get()
            else:
                final_bins = all_bins_rv.get()
            
            if final_bins is None or dv is None:
                return
            
            # Get list of all variables that were binned
            all_vars = final_bins.var_summary['var'].tolist()
            
            # Prepare rules with WOE values
            rules = final_bins.bin[final_bins.bin['bin'] != 'Total'].copy()
            rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
            
            # Add bin value column for each variable
            for var in all_vars:
                var_mask = rules['var'] == var
                rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
                    lambda x: x.replace(var, '').replace(' %in% c', '').strip()
                )
            
            # Create output DataFrames
            df_with_bins = create_binned_columns(final_bins, df, all_vars)
            df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
            
            # Extract only WOE columns + DV for the WOE-only output
            woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
            df_only_woe = df_with_woe[woe_cols + [dv]].copy()
            
            # Store results for return to main script
            app_results['df_with_woe'] = df_with_woe
            app_results['df_only_woe'] = df_only_woe
            app_results['bins'] = rules
            app_results['dv'] = dv
            app_results['completed'] = True
            
            # Close the Shiny session
            await session.close()
    
    # Create and return the Shiny app
    app = App(app_ui, server)
    app.results = app_results  # Attach results dict to app for retrieval
    return app


def find_free_port(start_port: int = 8054, max_attempts: int = 50) -> int:
    """
    Find an available port starting from start_port.
    
    This prevents "Address already in use" errors when running multiple
    instances of the WOE Editor.
    
    Parameters:
    -----------
    start_port : int
        Port number to start searching from
    max_attempts : int
        Maximum number of ports to try
    
    Returns:
    --------
    int : An available port number
    """
    import socket
    
    # Try random ports in the configured range
    for offset in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port  # Port is available
        except OSError:
            continue  # Port in use, try another
    
    # Fallback: let the OS assign a port (bind to port 0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_woe_editor(df: pd.DataFrame, min_prop: float = 0.01, port: int = None):
    """
    Run the WOE Editor application and return results.
    
    This function starts the Shiny server, opens a browser, waits for
    the user to finish editing, and returns the results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    min_prop : float
        Minimum proportion for binning
    port : int, optional
        Specific port to use. If None, finds a free port.
    
    Returns:
    --------
    dict : Results dictionary with df_with_woe, df_only_woe, bins, dv, completed
    """
    # Find a free port to avoid conflicts
    if port is None:
        port = find_free_port(BASE_PORT)
    
    print(f"Starting Shiny app on port {port} (Instance: {INSTANCE_ID})")
    sys.stdout.flush()  # Ensure message is displayed immediately
    
    # Create the app
    app = create_woe_editor_app(df, min_prop)
    
    try:
        # Run the app (blocks until user closes it)
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
    
    # Cleanup memory
    gc.collect()
    sys.stdout.flush()
    
    return app.results


# =============================================================================
# SECTION 10: CONFIGURATION
# =============================================================================
# Global configuration settings

# Minimum proportion of samples required in each bin
# 0.01 = 1% means each bin must contain at least 1% of observations
# This prevents creating bins with too few samples for reliable statistics
min_prop = 0.01

# =============================================================================
# SECTION 11: READ INPUT DATA FROM KNIME
# =============================================================================
# This is where the script interfaces with KNIME

# Read the first input table from KNIME and convert to pandas DataFrame
# knio.input_tables[0] is the first input port of the Python Script node
# .to_pandas() converts KNIME's table format to a pandas DataFrame
df = knio.input_tables[0].to_pandas()

# =============================================================================
# SECTION 12: CHECK FOR FLOW VARIABLES (HEADLESS MODE)
# =============================================================================
# Flow variables allow KNIME to pass parameters to the Python script
# If DependentVariable is provided, run in headless (automated) mode
# Otherwise, launch the interactive Shiny UI

# Initialize flags
contains_dv = False  # Whether a valid dependent variable was provided
dv = None  # Dependent variable name
target = None  # Target category (which value is "bad")
optimize_all = False  # Whether to auto-optimize all variables
group_na = False  # Whether to auto-group NA values

# Try to read the DependentVariable flow variable
try:
    # get() returns the value if it exists, or the default (None) if not
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass  # Ignore if flow variables aren't accessible

# Try to read the TargetCategory flow variable
try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

# Try to read the OptimizeAll flow variable
try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
except:
    pass

# Try to read the GroupNA flow variable
try:
    group_na = knio.flow_variables.get("GroupNA", False)
except:
    pass

# Validate the dependent variable
# It must be a non-empty string and exist as a column in the data
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True  # Valid DV found - use headless mode

# =============================================================================
# SECTION 13: MAIN PROCESSING LOGIC
# =============================================================================
# Execute either headless or interactive mode based on flow variables

if contains_dv:
    # =========================================================================
    # HEADLESS MODE (automated processing with parallel processing)
    # =========================================================================
    # This mode runs without user interaction when DV is provided
    
    print(f"Running in headless mode with DV: {dv}")
    print(f"[Parallel] Parallel processing enabled with {N_JOBS} workers")
    
    # Create list of independent variables (all columns except DV)
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter out constant/zero-variance variables (only 1 unique value)
    # These can't be binned and would cause errors
    constant_vars = []
    valid_vars = []
    for col in iv_list:
        # Count unique non-null values
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_vars.append(col)
        else:
            valid_vars.append(col)
    
    # Log removed variables
    if constant_vars:
        print(f"[Parallel] Removed {len(constant_vars)} constant variables (only 1 unique value)")
        if len(constant_vars) <= 10:
            print(f"  Constant vars: {constant_vars}")
        else:
            print(f"  First 10: {constant_vars[:10]}...")
    
    # Update IV list to only include valid variables
    iv_list = valid_vars
    print(f"[Parallel] Variables to process: {len(iv_list)}")
    
    # Calculate initial bins using parallel processing
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    
    # Merge pure bins (always do this - prevents infinite WOE values)
    # Pure bins have 100% goods or 100% bads, causing log(0) errors
    if 'purNode' in bins_result.var_summary.columns:
        # Count variables with pure bins (purNode = 'Y')
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    
    if pure_count > 0:
        print(f"[Parallel] Merging {int(pure_count)} pure bins (prevents infinite WOE)...")
        bins_result = merge_pure_bins(bins_result)
    
    # Optionally group NA values with closest bin
    if group_na:
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist())
    
    # Optionally optimize all variables (force monotonic trends)
    if optimize_all:
        # First group NAs
        bins_mod = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        
        # Force decreasing trend for D-trend variables
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        # Force increasing trend for I-trend variables
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
    
    # Prepare final rules with WOE values
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
    
    # Add bin value column for joining
    for var in bins_result.var_summary['var'].tolist():
        var_mask = rules['var'] == var
        rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
    
    # Create output DataFrames
    all_vars = bins_result.var_summary['var'].tolist()
    
    # Add binned columns (b_*) to data
    df_with_bins = create_binned_columns(bins_result, df, all_vars)
    
    # Add WOE columns (WOE_*) to data
    df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
    
    # Create WOE-only DataFrame (just WOE columns + DV)
    woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
    df_only_woe = df_with_woe[woe_cols + [dv]].copy()
    
    # Store bins for output
    bins = rules
    
    print(f"Processed {len(all_vars)} variables")

else:
    # =========================================================================
    # INTERACTIVE MODE (Shiny UI)
    # =========================================================================
    # This mode launches when no DV is provided via flow variables
    
    print("Running in interactive mode - launching Shiny UI...")
    print(f"[Parallel] Parallel processing enabled with {N_JOBS} workers")
    
    # Run the Shiny app and wait for user to finish
    results = run_woe_editor(df, min_prop=min_prop)
    
    # Check if user completed the session (vs. just closing the window)
    if results['completed']:
        df_with_woe = results['df_with_woe']
        df_only_woe = results['df_only_woe']
        bins = results['bins']
        dv = results['dv']
        print("Interactive session completed successfully")
    else:
        # User cancelled - return empty results
        print("Interactive session cancelled - returning empty results")
        df_with_woe = df.copy()
        df_only_woe = pd.DataFrame()
        bins = pd.DataFrame()

# =============================================================================
# SECTION 14: OUTPUT TABLES TO KNIME
# =============================================================================
# Write results to KNIME output ports

# Output 1: Original input DataFrame (unchanged)
# This allows downstream nodes to access the original data
knio.output_tables[0] = knio.Table.from_pandas(df)

# Output 2: df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# Contains all original columns plus the binning and WOE transformations
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)

# Output 3: df_only_woe - Only WOE columns + dependent variable
# This is the typical input for logistic regression in credit scoring
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)

# Output 4: bins - Binning rules with WOE values
# Contains the mapping from bin rules to WOE values for scorecard creation
knio.output_tables[3] = knio.Table.from_pandas(bins)

print("WOE Editor (Parallel) completed successfully")

# =============================================================================
# SECTION 15: CLEANUP FOR STABILITY
# =============================================================================
# Free memory and ensure clean exit

# Flush any buffered output to ensure all messages are displayed
sys.stdout.flush()

# Delete large objects to free memory
# This is especially important in KNIME where nodes may run repeatedly
try:
    del df
except:
    pass

try:
    del df_with_woe
except:
    pass

try:
    del df_only_woe
except:
    pass

try:
    del bins
except:
    pass

# Force garbage collection to immediately reclaim memory
# This prevents memory accumulation across multiple node executions
gc.collect()

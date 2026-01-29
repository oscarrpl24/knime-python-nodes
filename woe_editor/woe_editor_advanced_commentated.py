# =============================================================================
# WOE Editor for KNIME Python Script Node - ADVANCED BINNING VERSION
# =============================================================================
# This is a comprehensive Python script that performs Weight of Evidence (WOE)
# binning for credit risk modeling. WOE is a technique that transforms categorical
# and continuous variables into values that represent their predictive power
# relative to a binary target variable (e.g., default/no default).
#
# The script uses state-of-the-art optimal binning algorithms based on academic
# research and industry best practices for credit scoring and fraud detection.
#
# Compatible with KNIME 5.9, Python 3.9
#
# ALGORITHM DIFFERENCES FROM ORIGINAL:
# - Original: Uses DecisionTree (CART) for initial bin splits
# - Advanced: Uses ChiMerge + Monotonic Optimization + IV Maximization
#
# DEFAULT BEHAVIOR (R-compatible):
# - Uses same Decision Tree (CART) algorithm as R's logiBin::getBins
# - Same minProp=0.01 (1%) default as R
# - Output matches R WOE Editor exactly
#
# OPTIONAL ENHANCEMENTS:
# Enhancements can be enabled ALL AT ONCE (UseEnhancements=True) or INDIVIDUALLY:
# 1. Adaptive min_prop for sparse data (AdaptiveMinProp=True)
# 2. Chi-square validation to merge similar bins (ChiSquareValidation=True)
# 3. Minimum event count per bin (MinEventCount=True)
# 4. Automatic retry with relaxed constraints (AutoRetry=True)
# 5. Single-bin protection in GroupNA (SingleBinProtection=True, default ON)
#
# ADDITIONAL OPTIONS:
# - ChiMerge algorithm (set Algorithm="ChiMerge")
# - Shrinkage estimators for WOE (set UseShrinkage=True)
# - Diagnostic logging for problematic variables
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When DependentVariable flow variable is provided
#
# FLOW VARIABLES:
#
# Basic Settings:
# - DependentVariable (string, required for headless): Binary target variable name
# - TargetCategory (string, optional): Which value represents "bad" outcome
# - OptimizeAll (boolean, default False): Force monotonic trends on all vars
# - GroupNA (boolean, default False): Combine NA bins with closest bin
#
# Algorithm Settings:
# - Algorithm (string, default "DecisionTree"): "DecisionTree" or "ChiMerge"
# - MinBinPct (float, default 0.01): Min percentage per bin (0.01=1%, 0.05=5%)
# - MinBinCount (int, default 20): Min absolute count per bin (ChiMerge only)
# - UseShrinkage (boolean, default False): Apply shrinkage to WOE (for rare events)
#
# Enhancement Master Switch:
# - UseEnhancements (boolean, default False): Enable ALL enhancements at once
#
# Individual Enhancement Flags (override master switch when set):
# - AdaptiveMinProp (boolean, default False): Relax min_prop for sparse data
# - MinEventCount (boolean, default False): Ensure minimum events per bin
# - AutoRetry (boolean, default False): Retry with relaxed constraints if no splits
# - ChiSquareValidation (boolean, default False): Merge statistically similar bins
# - SingleBinProtection (boolean, default True): Prevent na_combine from creating WOE=0
#
# ALGORITHM OPTIONS:
#
# "DecisionTree" (DEFAULT - R-compatible):
#   - Uses CART decision tree, same as R's logiBin::getBins
#   - Produces identical output to R WOE Editor
#   - Recommended for consistency with existing R workflows
#
# "ChiMerge":
#   - Uses chi-square based bin merging
#   - More statistically rigorous but different from R
#   - May produce fewer bins for sparse data
#
# "IVOptimal":
#   - Directly maximizes Information Value (IV)
#   - Does NOT enforce monotonicity (allows "sweet spots")
#   - Dynamic starting granularity based on variable characteristics
#   - Merges pure bins to closest adjacent by WOE
#   - Best for fraud detection where non-monotonic patterns exist
#
# TUNING FOR FRAUD vs CREDIT SCORING:
# 
# FRAUD MODELS (low event rates 1-5%):
#   - MinBinPct = 0.01 (1%)
#   - MinBinCount = 20
#   - UseShrinkage = True (optional)
#
# CREDIT SCORING (higher event rates 10-20%):
#   - MinBinPct = 0.05 (5% - industry standard)
#   - MinBinCount = 50
#   - UseShrinkage = False
#
# Outputs:
# 1. Original input DataFrame (unchanged)
# 2. df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# 3. df_only_woe - Only WOE columns + dependent variable (for logistic regression)
# 4. df_only_bins - ONLY binned columns (b_*) for scorecard scoring (LEAN!)
# 5. bins - Binning rules with WOE values (metadata)
#
# Release Date: 2026-01-26
# Version: 1.4 (IVOptimal Algorithm + Individual Enhancement Flags)
# 
# Version History:
#   v1.2 - R-Compatible Algorithm + Fraud Support + Diagnostics
#   v1.3 - Individual enhancement flags (AdaptiveMinProp, MinEventCount, etc.)
#   v1.4 - IVOptimal algorithm (IV-maximizing, non-monotonic patterns)
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORT SECTION: Load all required Python libraries and modules
# -----------------------------------------------------------------------------

# Import the KNIME scripting interface - this is how we read inputs and write outputs
# in KNIME Python Script nodes. The 'knio' module provides access to input_tables,
# output_tables, and flow_variables.
import knime.scripting.io as knio

# Import pandas - the primary library for data manipulation in Python.
# pandas provides DataFrame objects which are similar to Excel spreadsheets or SQL tables.
# We use 'pd' as a shorthand alias by convention.
import pandas as pd

# Import numpy - the fundamental library for numerical computing in Python.
# numpy provides efficient array operations and mathematical functions.
# We use 'np' as a shorthand alias by convention.
import numpy as np

# Import the 're' module for regular expression operations.
# Regular expressions allow us to search for and manipulate text patterns,
# which we use extensively for parsing bin rule strings.
import re

# Import the warnings module to control Python warning messages.
# We'll suppress warnings to keep the output clean during processing.
import warnings

# Import the time module for measuring execution time.
# We use this to track how long each step takes and estimate remaining time.
import time

# Import the sys module for system-specific parameters and functions.
# We use sys.stdout.flush() to ensure progress messages are displayed immediately.
import sys

# Import the os module for operating system interactions.
# We use this to set environment variables and get process information.
import os

# Import the random module for generating random numbers.
# We use this to generate random port numbers for the Shiny UI to avoid conflicts.
import random

# Import type hints from the typing module for better code documentation.
# Dict: dictionary type, List: list type, Tuple: tuple type, Optional: can be None,
# Any: any type, Union: one of multiple types.
from typing import Dict, List, Tuple, Optional, Any, Union

# Import the dataclass decorator for creating simple data container classes.
# Dataclasses automatically generate __init__, __repr__, and other methods.
from dataclasses import dataclass

# Import the Enum class for creating enumeration types.
# Enumerations provide a way to define symbolic names for constant values.
from enum import Enum

# Suppress all warning messages to keep the console output clean.
# This prevents numpy, pandas, and other libraries from cluttering the output
# with deprecation warnings or other non-critical messages.
warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# When running multiple KNIME nodes simultaneously, we need to ensure each
# instance operates independently without conflicts.

# BASE_PORT: The starting port number for the Shiny web UI.
# We use port 8055 (different from the original to avoid conflicts).
# The actual port used will be BASE_PORT + a random offset.
BASE_PORT = 8055  # Different from original to avoid conflicts

# RANDOM_PORT_RANGE: The range of random values to add to BASE_PORT.
# This means we'll use ports between 8055 and 9055 (8055 + 1000).
RANDOM_PORT_RANGE = 1000

# INSTANCE_ID: A unique identifier for this running instance of the script.
# Combines the process ID (unique per running Python process) with a random number.
# os.getpid() returns the current process ID from the operating system.
# random.randint(10000, 99999) generates a random 5-digit number.
# This helps distinguish between multiple instances running simultaneously.
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Set environment variables to prevent multi-threading conflicts.
# When multiple KNIME nodes run Python scripts simultaneously, threading libraries
# can conflict with each other. Setting these to '1' forces single-threaded execution.

# NUMEXPR_MAX_THREADS: Controls numexpr library threading (used by pandas internally)
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts

# OMP_NUM_THREADS: Controls OpenMP threading (used by many scientific libraries)
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts

# OPENBLAS_NUM_THREADS: Controls OpenBLAS threading (linear algebra library)
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading conflicts

# MKL_NUM_THREADS: Controls Intel MKL threading (another linear algebra library)
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading conflicts

# =============================================================================
# Progress Logging Utilities
# =============================================================================
# These functions help us provide feedback to users about the script's progress,
# which is especially important for long-running operations on large datasets.

def log_progress(message: str, flush: bool = True):
    """
    Print a progress message with a timestamp prefix.
    
    This function adds a timestamp to messages so users can track when each
    step occurred and how long operations are taking.
    
    Parameters:
        message (str): The message to display to the user.
        flush (bool): If True, immediately write the message to the console.
                     Default is True because we want immediate feedback.
    
    Example output: "[14:32:15] Processing variable: Age"
    """
    # time.strftime formats the current time as hours:minutes:seconds
    # %H = 24-hour format hour, %M = minutes, %S = seconds
    timestamp = time.strftime("%H:%M:%S")
    
    # Print the message with the timestamp in square brackets
    print(f"[{timestamp}] {message}")
    
    # If flush is True, force Python to write the output immediately
    # Without flushing, Python may buffer output and delay displaying it
    if flush:
        sys.stdout.flush()


def format_time(seconds: float) -> str:
    """
    Convert a number of seconds into a human-readable time string.
    
    This function intelligently formats time based on magnitude:
    - Less than 60 seconds: shows seconds with one decimal (e.g., "45.3s")
    - Less than 3600 seconds: shows minutes and seconds (e.g., "5m 30s")  
    - 3600+ seconds: shows hours and minutes (e.g., "2h 15m")
    
    Parameters:
        seconds (float): The time duration in seconds to format.
    
    Returns:
        str: A human-readable string representation of the time.
    
    Examples:
        format_time(45.3) -> "45.3s"
        format_time(150) -> "2m 30s"
        format_time(7500) -> "2h 5m"
    """
    # If less than 60 seconds, show with one decimal place
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    # If less than 3600 seconds (1 hour), show minutes and seconds
    elif seconds < 3600:
        # Integer division (//) gives whole minutes
        mins = int(seconds // 60)
        # Modulo (%) gives remaining seconds after extracting minutes
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    
    # For 1 hour or more, show hours and minutes
    else:
        # Integer division by 3600 gives whole hours
        hours = int(seconds // 3600)
        # Modulo 3600 gives remaining seconds, then divide by 60 for minutes
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# =============================================================================
# Install/Import Dependencies
# =============================================================================
# This section attempts to import required libraries, and if they're not
# installed, it automatically installs them using pip. This ensures the
# script works even in environments where dependencies aren't pre-installed.

# Try to import scipy.stats for statistical functions (chi-square tests, etc.)
try:
    # scipy.stats contains statistical distributions, tests, and functions
    # We use it for chi-square calculations in the ChiMerge algorithm
    from scipy import stats
except ImportError:
    # If scipy is not installed, install it using pip
    # subprocess.check_call runs a command and waits for it to complete
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    # After installation, import the module
    from scipy import stats

# Try to import DecisionTreeClassifier from scikit-learn
try:
    # DecisionTreeClassifier is used for the R-compatible binning algorithm
    # It finds optimal split points that maximize information gain
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    # If scikit-learn is not installed, install it
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier

# Try to import Shiny web framework components for the interactive UI
try:
    # Shiny is a web application framework that lets us create interactive UIs
    # App: main application class, Inputs/Outputs/Session: UI state management
    # reactive/render/ui: reactive programming components for dynamic updates
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # shinywidgets provides integration between Shiny and Plotly for charts
    from shinywidgets import render_plotly, output_widget
    
    # plotly.graph_objects provides low-level plotly chart building
    import plotly.graph_objects as go
except ImportError:
    # If any Shiny components are missing, install the full stack
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Configuration
# =============================================================================
# This class holds all the configurable parameters for the binning algorithms.
# Using a class allows us to easily modify settings via flow variables.

class BinningConfig:
    """
    Configuration class for advanced binning algorithms.
    
    This class stores all tunable parameters as class-level attributes.
    Using class attributes (not instance attributes) means all functions
    can access the same configuration without passing it around.
    
    ALGORITHM OPTIONS:
        - "DecisionTree" (default): Same as R logiBin::getBins - CART-based splitting
        - "ChiMerge": Chi-square based bin merging (more conservative)
    
    For R-COMPATIBLE OUTPUT (matches logiBin::getBins):
        - Algorithm = "DecisionTree"
        - MIN_BIN_PCT = 0.01 (1%) - matches R's minProp default
    
    For FRAUD MODELS (low event rates 1-5%):
        - MIN_BIN_PCT = 0.01 to 0.02 (1-2%)
        - MIN_BIN_COUNT = 20-30
        - USE_SHRINKAGE = True
    
    For CREDIT SCORING (higher event rates 10-20%):
        - MIN_BIN_PCT = 0.05 (5%) - industry standard
        - MIN_BIN_COUNT = 50
    
    INDIVIDUAL ENHANCEMENT FLAGS:
        These can be enabled independently or all at once via USE_ENHANCEMENTS=True
        - ADAPTIVE_MIN_PROP: Relaxes min_prop for sparse data (< 500 samples)
        - MIN_EVENT_COUNT: Ensures minimum number of events per bin
        - AUTO_RETRY: Retries with relaxed constraints if binning fails
        - CHI_SQUARE_VALIDATION: Merges statistically similar bins
        - SINGLE_BIN_PROTECTION: Prevents na_combine from creating single-bin variables
    """
    
    # ALGORITHM: Which binning algorithm to use.
    # "DecisionTree" matches R's logiBin package for consistency.
    # "ChiMerge" is more statistically rigorous but produces different results.
    ALGORITHM = "DecisionTree"  # "DecisionTree" (R-compatible) or "ChiMerge"
    
    # MIN_BIN_PCT: Minimum percentage of observations required in each bin.
    # 0.01 = 1%, meaning each bin must have at least 1% of the total data.
    # This matches R's minProp default for compatibility.
    MIN_BIN_PCT = 0.01  # Minimum 1% of observations per bin (matches R's minProp)
    
    # MIN_BIN_COUNT: Minimum absolute number of observations per bin.
    # Even if 1% would be less than 20, bins must have at least 20 observations.
    # This prevents bins with too few samples for stable WOE estimates.
    MIN_BIN_COUNT = 20  # Minimum absolute count per bin
    
    # MAX_BINS: Maximum number of bins allowed per variable.
    # More bins = more granular but more complex; fewer bins = simpler model.
    # 10 is a good balance between capturing patterns and avoiding overfitting.
    MAX_BINS = 10  # Maximum number of bins
    
    # MIN_BINS: Minimum number of bins required per variable.
    # Must have at least 2 bins to have any discriminatory power (WOE variation).
    MIN_BINS = 2  # Minimum number of bins
    
    # MAX_CATEGORIES: Maximum unique values for categorical variables.
    # Variables with more unique values than this will be skipped.
    # High-cardinality categoricals (like ZIP codes) need special handling.
    MAX_CATEGORIES = 50  # Maximum unique categories for categorical variables
    
    # CHI_MERGE_THRESHOLD: P-value threshold for the ChiMerge algorithm.
    # Adjacent bins are merged if their chi-square p-value exceeds this.
    # Lower values = more aggressive merging, fewer bins.
    CHI_MERGE_THRESHOLD = 0.05  # Chi-square p-value threshold
    
    # MIN_IV_GAIN: Minimum improvement in Information Value to continue splitting.
    # If a split doesn't improve IV by at least this amount, stop splitting.
    MIN_IV_GAIN = 0.005  # Minimum IV improvement to continue splitting
    
    # USE_SHRINKAGE: Whether to apply Bayesian shrinkage to WOE values.
    # Shrinkage helps stabilize WOE estimates when bin counts are low (fraud models).
    USE_SHRINKAGE = False  # Apply shrinkage to WOE values (optional)
    
    # SHRINKAGE_STRENGTH: How much shrinkage to apply (if USE_SHRINKAGE is True).
    # Higher values = more shrinkage toward the overall mean.
    SHRINKAGE_STRENGTH = 0.1  # Shrinkage regularization strength
    
    # USE_ENHANCEMENTS: Master switch to enable all advanced enhancements.
    # When True, turns on all individual enhancement flags below.
    USE_ENHANCEMENTS = False  # Master switch: enables ALL enhancements when True
    
    # --- Individual enhancement flags (can be set independently) ---
    
    # ADAPTIVE_MIN_PROP: Automatically relax the minimum bin proportion for sparse data.
    # When a variable has very few non-null values (<500), standard constraints may
    # result in no bins being created. This enhancement relaxes constraints adaptively.
    ADAPTIVE_MIN_PROP = False  # Relaxes min_prop for sparse data (< 500 samples)
    
    # MIN_EVENT_COUNT: Ensure each bin has enough "bad" events for stable WOE.
    # For fraud models with very low event rates, bins might have 0 or 1 events,
    # leading to unstable WOE values. This ensures a minimum event count.
    MIN_EVENT_COUNT = False  # Ensures at least N events per potential bin
    
    # AUTO_RETRY: Automatically retry binning with relaxed constraints if initial fails.
    # If the first attempt produces no splits (single bin), try again with looser settings.
    AUTO_RETRY = False  # Retry with relaxed constraints if no splits found
    
    # CHI_SQUARE_VALIDATION: Post-binning validation using chi-square tests.
    # Merges adjacent bins that are not statistically significantly different,
    # preventing over-binning where splits don't add real predictive value.
    CHI_SQUARE_VALIDATION = False  # Merge statistically similar bins post-binning
    
    # SINGLE_BIN_PROTECTION: Prevent na_combine from creating single-bin variables.
    # When grouping NA values, if merging would leave only one bin (WOE=0 everywhere),
    # skip the merge for that variable. Default is ON because single-bin vars are useless.
    SINGLE_BIN_PROTECTION = True  # Prevent na_combine from creating single-bin vars


# =============================================================================
# Data Classes
# =============================================================================
# Data classes provide a convenient way to create classes that primarily hold data.
# They automatically generate __init__, __repr__, __eq__, and other methods.

@dataclass
class BinResult:
    """
    Container class for binning results.
    
    This class holds the two main outputs of the binning process:
    1. A summary table with one row per variable (IV, entropy, trend, etc.)
    2. A detailed bin table with one row per bin per variable
    
    Using a dataclass makes it easy to pass these two related DataFrames together
    and access them with clear, readable attribute names.
    
    Attributes:
        var_summary (pd.DataFrame): Summary statistics for each variable.
            Columns: var, varType, iv, ent, trend, monTrend, flipRatio, numBins, purNode
        bin (pd.DataFrame): Detailed bin information.
            Columns: var, bin, count, bads, goods, propn, bad_rate, goodCap, badCap,
                    iv, ent, purNode, trend, woe (optional), binValue (optional)
    """
    # var_summary: DataFrame with one row per variable, containing summary statistics
    var_summary: pd.DataFrame  # Summary stats for each variable
    
    # bin: DataFrame with one row per bin per variable, containing detailed bin info
    bin: pd.DataFrame  # Detailed bin information


# =============================================================================
# Core WOE/IV Calculation Functions
# =============================================================================
# These functions implement the fundamental calculations for Weight of Evidence
# and Information Value - the core concepts in WOE binning.

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray, 
                  use_shrinkage: bool = False, shrinkage_strength: float = 0.1) -> np.ndarray:
    """
    Calculate Weight of Evidence (WOE) for each bin.
    
    WOE measures the relative risk of an outcome in a bin compared to the population.
    Formula: WOE = ln((% of Bads in bin) / (% of Goods in bin))
    
    - Positive WOE: More bads than expected (higher risk bin)
    - Negative WOE: Fewer bads than expected (lower risk bin)
    - Zero WOE: Same proportion as population (neutral)
    
    With optional shrinkage for rare events (fraud detection).
    Shrinkage pulls extreme WOE values toward zero, stabilizing estimates when
    sample sizes are small (empirical Bayes approach).
    
    Parameters:
        freq_good (np.ndarray): Array of good (non-event) counts for each bin.
        freq_bad (np.ndarray): Array of bad (event) counts for each bin.
        use_shrinkage (bool): If True, apply Bayesian shrinkage to stabilize estimates.
        shrinkage_strength (float): How much shrinkage to apply (0=none, higher=more).
    
    Returns:
        np.ndarray: Array of WOE values for each bin, rounded to 5 decimal places.
    
    Example:
        If bin has 10% of all bads but only 5% of all goods:
        WOE = ln(0.10 / 0.05) = ln(2) = 0.693 (higher risk bin)
    """
    # Convert inputs to float numpy arrays to ensure consistent math operations
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals across all bins
    total_good = freq_good.sum()  # Total number of goods across all bins
    total_bad = freq_bad.sum()    # Total number of bads across all bins
    
    # Handle edge case: if no goods or no bads at all, WOE is undefined
    # Return zeros for all bins in this case
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    # Calculate distribution (proportion) of goods and bads in each bin
    # dist_good[i] = what fraction of all goods are in bin i
    dist_good = freq_good / total_good
    # dist_bad[i] = what fraction of all bads are in bin i
    dist_bad = freq_bad / total_bad
    
    # Apply Laplace smoothing to prevent division by zero and log(0)
    # When a bin has 0 goods or 0 bads, we'd get infinity or negative infinity
    # epsilon is a small value to add stability without changing results much
    epsilon = 0.0001  # Small constant for smoothing
    
    # Replace zero distributions with epsilon to avoid log(0) or division by 0
    dist_good = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad = np.where(dist_bad == 0, epsilon, dist_bad)
    
    # Calculate WOE using the standard formula
    # WOE = natural log of (bad distribution / good distribution)
    woe = np.log(dist_bad / dist_good)
    
    # Apply shrinkage (empirical Bayes) for rare events if requested
    # Shrinkage pulls extreme WOE values toward zero based on sample size
    # Larger samples get less shrinkage (more trust in the data)
    if use_shrinkage and shrinkage_strength > 0:
        # Total observations in each bin
        n_obs = freq_good + freq_bad
        # Total observations across all bins
        total_obs = n_obs.sum()
        
        # Calculate shrinkage weights based on sample size
        # Formula: weight = n / (n + k * N / B)
        # where n=bin count, k=shrinkage strength, N=total count, B=number of bins
        # Larger samples get weight closer to 1 (less shrinkage)
        weights = n_obs / (n_obs + shrinkage_strength * total_obs / len(n_obs))
        
        # Apply shrinkage by multiplying WOE by weight
        # Small samples: weight near 0, WOE pulled toward 0
        # Large samples: weight near 1, WOE unchanged
        woe = woe * weights
    
    # Round to 5 decimal places for clean output
    return np.round(woe, 5)


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """
    Calculate Information Value (IV) for a variable.
    
    IV measures the overall predictive power of a variable. It sums the contribution
    from each bin, where each bin's contribution depends on:
    1. How different its WOE is from zero
    2. How much of the population it contains
    
    Formula: IV = Σ (dist_bad - dist_good) * WOE
    
    IV Interpretation Guidelines:
    - < 0.02: Not useful for prediction
    - 0.02 - 0.1: Weak predictive power
    - 0.1 - 0.3: Medium predictive power
    - 0.3 - 0.5: Strong predictive power
    - > 0.5: Suspicious (too good, possible data leakage)
    
    Parameters:
        freq_good (np.ndarray): Array of good (non-event) counts for each bin.
        freq_bad (np.ndarray): Array of bad (event) counts for each bin.
    
    Returns:
        float: The Information Value, rounded to 4 decimal places.
    """
    # Convert to float arrays for consistent calculations
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # If no goods or no bads, IV is undefined - return 0
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    # Calculate distributions
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Apply epsilon smoothing for the log calculation (same as in calculate_woe)
    epsilon = 0.0001
    dist_good_safe = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, epsilon, dist_bad)
    
    # Calculate WOE for each bin
    woe = np.log(dist_bad_safe / dist_good_safe)
    
    # Calculate IV as sum of (dist_bad - dist_good) * WOE across all bins
    # Note: we use original distributions (not smoothed) for the difference
    iv = np.sum((dist_bad - dist_good) * woe)
    
    # Handle any numerical issues (infinity or NaN)
    if not np.isfinite(iv):
        iv = 0.0
    
    # Round to 4 decimal places
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """
    Calculate entropy for a bin.
    
    Entropy measures the "purity" or "disorder" of a bin.
    - Entropy = 0: Pure bin (all goods OR all bads)
    - Entropy = 1: Maximum impurity (50% goods, 50% bads)
    
    Formula: Entropy = -Σ p(i) * log2(p(i))
    For binary case: -[p_bad * log2(p_bad) + p_good * log2(p_good)]
    
    Lower entropy is generally better (more predictive bins).
    
    Parameters:
        goods (int): Number of good outcomes in the bin.
        bads (int): Number of bad outcomes in the bin.
    
    Returns:
        float: Entropy value between 0 and 1, rounded to 4 decimal places.
    """
    # Total observations in the bin
    total = goods + bads
    
    # Handle edge cases: empty bin or pure bin (all one class)
    # In these cases, entropy is 0 (perfectly "pure")
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    # Calculate proportions (probabilities)
    p_good = goods / total  # Probability of good outcome
    p_bad = bads / total    # Probability of bad outcome
    
    # Calculate entropy using the binary entropy formula
    # We use log base 2 so entropy is between 0 and 1
    # Negative sign because log of probabilities (<1) is negative
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """
    Determine if a variable is numeric or factor (categorical).
    
    This function classifies variables into two types:
    1. 'numeric': Continuous or discrete numeric values (treated with splits)
    2. 'factor': Categorical values (treated with grouping)
    
    Special case: Numeric variables with ≤10 unique values are treated as factors.
    This is because variables like "NumberOfDependents" (0, 1, 2, 3...) are better
    handled as discrete categories than as continuous variables.
    
    Parameters:
        series (pd.Series): The pandas Series (column) to classify.
    
    Returns:
        str: Either 'numeric' or 'factor'.
    """
    # Check if the pandas dtype is numeric (int, float, etc.)
    if pd.api.types.is_numeric_dtype(series):
        # Even if numeric, treat as factor if there are few unique values
        # This handles coded variables like 0/1/2/3 for education level
        if series.nunique() <= 10:
            return 'factor'
        return 'numeric'
    
    # Non-numeric types (strings, objects, categories) are factors
    return 'factor'


# =============================================================================
# ChiMerge Algorithm - Core of Advanced Binning
# =============================================================================
# ChiMerge is a bottom-up binning algorithm that starts with many small bins
# and progressively merges adjacent bins that are statistically similar.
# It uses chi-square tests to determine when bins are different enough to keep separate.

def _chi_square_statistic(bin1_good: int, bin1_bad: int, 
                          bin2_good: int, bin2_bad: int) -> float:
    """
    Calculate chi-square statistic for two adjacent bins.
    
    The chi-square test measures whether the good/bad proportions in two bins
    are statistically different from each other. A high chi-square value means
    the bins are significantly different and should be kept separate.
    
    Uses Yates continuity correction for 2x2 contingency tables, which improves
    accuracy when cell counts are small.
    
    Parameters:
        bin1_good (int): Number of goods in first bin.
        bin1_bad (int): Number of bads in first bin.
        bin2_good (int): Number of goods in second bin.
        bin2_bad (int): Number of bads in second bin.
    
    Returns:
        float: Chi-square statistic. Higher values = more different bins.
               Returns infinity if calculation is not possible.
    """
    # Create a 2x2 contingency table (observed frequencies)
    # Rows: bin1, bin2
    # Columns: goods, bads
    observed = np.array([[bin1_good, bin1_bad], [bin2_good, bin2_bad]])
    
    # If the table is empty, return infinity (can't merge nothing)
    if observed.sum() == 0:
        return np.inf
    
    # Calculate marginal totals (row sums and column sums)
    row_totals = observed.sum(axis=1)  # [bin1_total, bin2_total]
    col_totals = observed.sum(axis=0)  # [total_goods, total_bads]
    total = observed.sum()              # Grand total
    
    # Check for zero marginals (can't compute expected frequencies)
    if total == 0 or any(row_totals == 0) or any(col_totals == 0):
        return np.inf
    
    # Calculate expected frequencies under the null hypothesis
    # Expected = (row_total * col_total) / grand_total
    # np.outer computes the outer product of row and column totals
    expected = np.outer(row_totals, col_totals) / total
    
    # Calculate chi-square with Yates continuity correction
    # Yates correction subtracts 0.5 from each |observed - expected|
    # This helps when expected counts are small
    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i, j] > 0:
                # Yates correction: subtract 0.5 from the absolute difference
                diff = abs(observed[i, j] - expected[i, j]) - 0.5
                # Don't let the corrected difference go negative
                diff = max(diff, 0)
                # Add to chi-square: (observed - expected)^2 / expected
                chi2 += (diff ** 2) / expected[i, j]
    
    return chi2


def _chimerge_get_splits(
    x: pd.Series,
    y: pd.Series,
    min_bin_pct: float = 0.05,
    min_bin_count: int = 50,
    max_bins: int = 10,
    min_bins: int = 2,
    chi_threshold: float = 0.05
) -> List[float]:
    """
    ChiMerge algorithm: Find optimal split points for numeric variables.
    
    Algorithm Overview:
    1. Start with fine-grained bins (many small bins based on percentiles)
    2. Calculate chi-square statistic for each pair of adjacent bins
    3. Merge the pair with the smallest chi-square (most similar)
    4. Repeat until stopping criteria are met
    
    This is more statistically rigorous than decision tree-based splitting
    because it explicitly tests whether adjacent bins are different.
    
    Parameters:
        x (pd.Series): The feature values to bin.
        y (pd.Series): The binary target variable (0/1).
        min_bin_pct (float): Minimum percentage of data per bin (default 5%).
        min_bin_count (int): Minimum absolute count per bin (default 50).
        max_bins (int): Maximum number of bins to create (default 10).
        min_bins (int): Minimum number of bins to create (default 2).
        chi_threshold (float): P-value threshold for stopping (default 0.05).
    
    Returns:
        List[float]: Split points (bin boundaries) for the variable.
    """
    # Remove rows where either x or y is missing
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    # If no valid data, return no splits
    if len(x_clean) == 0:
        return []
    
    # Calculate minimum count required per bin
    # Use the larger of: percentage-based or absolute minimum
    total_count = len(x_clean)
    min_count_required = max(int(total_count * min_bin_pct), min_bin_count)
    
    # Start with fine-grained bins based on percentiles
    # Use up to 100 initial bins, or number of unique values if less
    n_initial_bins = min(100, len(x_clean.unique()))
    
    # Create initial bins using quantile-based binning
    try:
        # pd.qcut creates bins with approximately equal frequencies
        # duplicates='drop' handles cases where many values are the same
        initial_bins = pd.qcut(x_clean, q=n_initial_bins, duplicates='drop')
    except ValueError:
        # If qcut fails, fall back to equal-width binning
        try:
            initial_bins = pd.cut(x_clean, bins=min(20, len(x_clean.unique())), duplicates='drop')
        except:
            # If both methods fail, return no splits
            return []
    
    # Extract bin edges from the categorical result
    if hasattr(initial_bins, 'categories') and len(initial_bins.categories) > 0:
        # Get unique edges: left boundary of first bin + right boundaries of all bins
        edges = sorted(set(
            [initial_bins.categories[0].left] + 
            [cat.right for cat in initial_bins.categories]
        ))
    else:
        return []
    
    # Helper function to build statistics for each bin
    def build_bin_stats(edges):
        """Create a list of dictionaries with stats for each bin."""
        bins_stats = []
        for i in range(len(edges) - 1):
            # Define the mask for this bin
            # First bin includes left boundary (>=), others are half-open (>)
            if i == 0:
                bin_mask = (x_clean >= edges[i]) & (x_clean <= edges[i + 1])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            # Calculate bin statistics
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())  # Sum of 1s = number of bads
            goods = int(count - bads)             # Remainder = goods
            
            bins_stats.append({
                'left': edges[i],     # Left boundary
                'right': edges[i + 1], # Right boundary
                'goods': goods,        # Good count
                'bads': bads,          # Bad count
                'count': count         # Total count
            })
        return bins_stats
    
    # Build initial bin statistics
    bins_stats = build_bin_stats(edges)
    
    # Remove empty bins (can occur due to duplicates or edge cases)
    bins_stats = [b for b in bins_stats if b['count'] > 0]
    
    # If we already have min_bins or fewer, just extract splits and return
    if len(bins_stats) <= min_bins:
        return [b['right'] for b in bins_stats[:-1]]
    
    # Calculate chi-square critical value for the given threshold
    # Using 1 degree of freedom for 2x2 contingency table
    chi2_threshold = stats.chi2.ppf(1 - chi_threshold, df=1)
    
    # Main merging loop: iteratively merge bins with smallest chi-square
    while len(bins_stats) > min_bins:
        # Find the pair of adjacent bins with minimum chi-square
        min_chi2 = np.inf
        merge_idx = -1
        
        for i in range(len(bins_stats) - 1):
            # Calculate chi-square between bin i and bin i+1
            chi2 = _chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            # Track the minimum
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        # Stopping rule: if minimum chi-square exceeds threshold AND
        # we have an acceptable number of bins, stop merging
        if min_chi2 > chi2_threshold and len(bins_stats) <= max_bins:
            break
        
        # Merge the pair with smallest chi-square
        if merge_idx >= 0:
            # Combine statistics of bins at merge_idx and merge_idx+1
            bins_stats[merge_idx] = {
                'left': bins_stats[merge_idx]['left'],        # Keep left boundary
                'right': bins_stats[merge_idx + 1]['right'],  # Take right boundary
                'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
                'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
                'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
            }
            # Remove the second bin (it's now merged into the first)
            bins_stats.pop(merge_idx + 1)
        else:
            break
    
    # Enforce minimum bin size by merging small bins
    changed = True
    while changed and len(bins_stats) > min_bins:
        changed = False
        for i in range(len(bins_stats)):
            if bins_stats[i]['count'] < min_count_required:
                # This bin is too small - merge with an adjacent bin
                if i == 0 and len(bins_stats) > 1:
                    # First bin: merge with second bin
                    bins_stats[0] = {
                        'left': bins_stats[0]['left'],
                        'right': bins_stats[1]['right'],
                        'goods': bins_stats[0]['goods'] + bins_stats[1]['goods'],
                        'bads': bins_stats[0]['bads'] + bins_stats[1]['bads'],
                        'count': bins_stats[0]['count'] + bins_stats[1]['count']
                    }
                    bins_stats.pop(1)
                    changed = True
                    break
                elif i > 0:
                    # Other bins: merge with previous bin
                    bins_stats[i - 1] = {
                        'left': bins_stats[i - 1]['left'],
                        'right': bins_stats[i]['right'],
                        'goods': bins_stats[i - 1]['goods'] + bins_stats[i]['goods'],
                        'bads': bins_stats[i - 1]['bads'] + bins_stats[i]['bads'],
                        'count': bins_stats[i - 1]['count'] + bins_stats[i]['count']
                    }
                    bins_stats.pop(i)
                    changed = True
                    break
    
    # If still too many bins, continue merging using chi-square
    while len(bins_stats) > max_bins:
        min_chi2 = np.inf
        merge_idx = 0
        
        # Find pair with smallest chi-square
        for i in range(len(bins_stats) - 1):
            chi2 = _chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        # Merge
        bins_stats[merge_idx] = {
            'left': bins_stats[merge_idx]['left'],
            'right': bins_stats[merge_idx + 1]['right'],
            'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
            'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
            'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
        }
        bins_stats.pop(merge_idx + 1)
    
    # Extract split points (right boundaries of all bins except the last)
    splits = [b['right'] for b in bins_stats[:-1]]
    return splits


def _enforce_monotonicity(
    x: pd.Series,
    y: pd.Series,
    splits: List[float],
    direction: str = 'auto'
) -> List[float]:
    """
    Enforce monotonic WOE by merging bins that violate monotonicity.
    
    Monotonic WOE is often desired in credit scoring because:
    1. It makes business sense (e.g., higher income should always be better)
    2. It prevents overfitting to noise
    3. It creates more interpretable scorecards
    
    This function merges bins until WOE is monotonically increasing or decreasing.
    
    Parameters:
        x (pd.Series): The feature values.
        y (pd.Series): The binary target variable.
        splits (List[float]): Current split points.
        direction (str): 'auto' (detect from correlation), 'increasing', or 'decreasing'.
    
    Returns:
        List[float]: Modified split points that result in monotonic WOE.
    """
    # If no splits, nothing to enforce
    if len(splits) == 0:
        return splits
    
    # Remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Detect optimal direction if auto
    if direction == 'auto':
        # Use correlation to determine natural direction
        corr = x_clean.corr(y_clean)
        # If correlation is too weak or undefined, don't enforce monotonicity
        if pd.isna(corr) or abs(corr) < 0.01:
            return splits  # Not enough correlation to enforce monotonicity
        # Positive correlation: higher x = more bads = increasing WOE
        # Negative correlation: higher x = fewer bads = decreasing WOE
        direction = 'increasing' if corr > 0 else 'decreasing'
    
    def get_bin_woes(current_splits):
        """Calculate WOE for each bin given current splits."""
        # Create bin edges including -inf and +inf
        edges = [-np.inf] + list(current_splits) + [np.inf]
        bin_data = []
        
        # Calculate stats for each bin
        for i in range(len(edges) - 1):
            # Define bin boundaries
            if i == 0:
                bin_mask = (x_clean <= edges[i + 1])
            elif i == len(edges) - 2:
                bin_mask = (x_clean > edges[i])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())
            goods = int(count - bads)
            bin_data.append({'goods': goods, 'bads': bads, 'count': count, 'edge': edges[i + 1]})
        
        # Calculate WOE for all bins
        goods_arr = np.array([b['goods'] for b in bin_data])
        bads_arr = np.array([b['bads'] for b in bin_data])
        woes = calculate_woe(goods_arr, bads_arr)
        
        return list(woes), bin_data
    
    # Iteratively merge bins that violate monotonicity
    current_splits = list(splits)
    max_iterations = 50  # Safety limit to prevent infinite loops
    
    for _ in range(max_iterations):
        # Can't enforce monotonicity with no splits
        if len(current_splits) == 0:
            break
        
        # Get current WOE values
        woes, bin_data = get_bin_woes(current_splits)
        
        # Check for monotonicity violations
        violating_idx = -1
        for i in range(1, len(woes)):
            # Check if WOE violates expected direction
            if direction == 'increasing' and woes[i] < woes[i - 1]:
                violating_idx = i
                break
            elif direction == 'decreasing' and woes[i] > woes[i - 1]:
                violating_idx = i
                break
        
        # If no violations found, we're done
        if violating_idx == -1:
            break
        
        # Remove the split between violating bins (merge them)
        if violating_idx > 0 and violating_idx <= len(current_splits):
            current_splits.pop(violating_idx - 1)
        elif len(current_splits) > 0:
            current_splits.pop(0)
        else:
            break
    
    return current_splits


# =============================================================================
# Decision Tree Algorithm (R-compatible - matches logiBin::getBins)
# =============================================================================
# This algorithm uses scikit-learn's DecisionTreeClassifier to find optimal
# split points. It produces results that match R's logiBin package.

def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_events: int = 5,
    use_enhancements: bool = False,
    adaptive_min_prop: bool = None,
    min_event_count: bool = None,
    auto_retry: bool = None
) -> List[float]:
    """
    Use decision tree (CART) to find optimal split points for numeric variables.
    
    This matches R's logiBin::getBins algorithm, which uses CART decision trees
    to find splits that maximize information gain (minimize entropy).
    
    Individual enhancements can be enabled/disabled:
    1. Adaptive min_prop (adaptive_min_prop): Relaxes min_prop for sparse data
    2. Minimum event count (min_event_count): Prevents unstable bins in low-event data
    3. Automatic retry (auto_retry): Retries with relaxed constraints if no splits found
    
    Parameters:
        x (pd.Series): Feature values to bin.
        y (pd.Series): Binary target values (0/1).
        min_prop (float): Minimum proportion of samples per bin (R default is 0.01 = 1%).
        max_bins (int): Maximum number of bins (leaf nodes).
        min_events (int): Minimum number of events (bads) required per potential bin.
        use_enhancements (bool): Master switch - if True, enables all enhancements.
        adaptive_min_prop (bool): Enable adaptive min_prop for sparse data (overrides master).
        min_event_count (bool): Enable minimum event count per bin (overrides master).
        auto_retry (bool): Enable auto-retry with relaxed constraints (overrides master).
    
    Returns:
        List[float]: Split thresholds found by the decision tree.
    """
    # Resolve individual flags: use explicit value if provided, else fall back to master switch
    # This allows both the master switch and individual overrides
    use_adaptive = adaptive_min_prop if adaptive_min_prop is not None else use_enhancements
    use_min_events = min_event_count if min_event_count is not None else use_enhancements
    use_auto_retry = auto_retry if auto_retry is not None else use_enhancements
    
    # Remove rows with missing values in either x or y
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)  # Reshape for sklearn (needs 2D array)
    y_clean = y[mask].values
    
    # If no valid data, return no splits
    if len(x_clean) == 0:
        return []
    
    # Calculate sample size and event statistics
    n_samples = len(x_clean)
    n_events = int(y_clean.sum())  # Number of "bads" (1s)
    event_rate = n_events / n_samples if n_samples > 0 else 0
    
    # Default: Use min_prop directly (R-compatible)
    effective_min_prop = min_prop
    
    # ENHANCEMENT 1: Adaptive min_prop for sparse data
    # Relaxes constraints when there's limited data to work with
    if use_adaptive:
        # For very sparse data (few non-null values), relax the constraint
        if n_samples < 500:
            # Reduce min_prop by half, but not below 0.5%
            effective_min_prop = max(min_prop / 2, 0.005)
        
        # For very low event rates (fraud), ensure enough events per bin
        if event_rate < 0.05 and n_events > 0:
            # Need at least min_events per bin
            max_possible_bins = max(n_events // min_events, 2)
            # Calculate what min_samples would give us that many bins
            min_samples_for_events = n_samples / max_possible_bins
            adaptive_prop = min_samples_for_events / n_samples
            # Use the larger of current or adaptive value
            effective_min_prop = max(effective_min_prop, adaptive_prop * 0.8)
    
    # Convert proportion to absolute count for sklearn
    min_samples_leaf = max(int(n_samples * effective_min_prop), 1)
    
    # ENHANCEMENT 2: Ensure minimum event count per leaf
    # This prevents bins with 0 or 1 events, which have unstable WOE
    if use_min_events and n_events > 0 and min_events > 0:
        # Each leaf should have at least min_events bads on average
        # Calculate how many samples we need to expect min_events bads
        min_samples_for_min_events = int(min_events / max(event_rate, 0.001))
        min_samples_leaf = max(min_samples_leaf, min_samples_for_min_events)
    
    # Don't let min_samples_leaf exceed half the data (need at least 2 leaves)
    min_samples_leaf = min(min_samples_leaf, n_samples // 2)
    min_samples_leaf = max(min_samples_leaf, 1)  # At least 1
    
    # Create and fit the decision tree
    # max_leaf_nodes controls the number of bins (leaves)
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,      # Maximum number of bins
        min_samples_leaf=min_samples_leaf,  # Minimum samples per bin
        random_state=42                # Fixed seed for reproducibility
    )
    
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        # If fitting fails, return no splits
        return []
    
    # Extract thresholds from the tree structure
    # tree_.threshold contains the split values; -2 indicates a leaf node
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]  # Remove leaf indicators
    thresholds = sorted(set(thresholds))       # Remove duplicates and sort
    
    # ENHANCEMENT 3: Retry with relaxed constraints if no splits found
    # If the first attempt failed to create splits, try again with looser settings
    if use_auto_retry and len(thresholds) == 0 and min_samples_leaf > 10:
        # Try again with smaller min_samples_leaf (half, minimum 1)
        tree_retry = DecisionTreeClassifier(
            max_leaf_nodes=max_bins,
            min_samples_leaf=max(min_samples_leaf // 2, 1),
            random_state=42
        )
        try:
            tree_retry.fit(x_clean, y_clean)
            thresholds = tree_retry.tree_.threshold
            thresholds = thresholds[thresholds != -2]
            thresholds = sorted(set(thresholds))
        except Exception:
            pass  # Keep empty thresholds if retry also fails
    
    return thresholds


# =============================================================================
# IV-Optimal Algorithm - Maximizes Information Value
# =============================================================================
# This algorithm directly optimizes for Information Value, the measure of
# predictive power. Unlike other methods, it doesn't enforce monotonicity.

def _calculate_bin_iv(goods: int, bads: int, total_goods: int, total_bads: int) -> float:
    """
    Calculate Information Value (IV) contribution for a single bin.
    
    Each bin contributes to the total IV based on:
    1. Its share of goods and bads
    2. How different those shares are from each other
    
    Formula: IV_bin = (dist_bad - dist_good) * WOE
    
    Parameters:
        goods (int): Number of goods in this bin.
        bads (int): Number of bads in this bin.
        total_goods (int): Total goods across all bins.
        total_bads (int): Total bads across all bins.
    
    Returns:
        float: This bin's contribution to total IV.
    """
    # Handle edge cases where totals are zero
    if total_goods == 0 or total_bads == 0:
        return 0.0
    
    # Calculate this bin's share of goods and bads
    dist_good = goods / total_goods if total_goods > 0 else 0
    dist_bad = bads / total_bads if total_bads > 0 else 0
    
    # Handle edge cases to avoid log(0)
    if dist_good == 0 or dist_bad == 0:
        return 0.0
    
    # Calculate WOE and IV contribution
    woe = np.log(dist_bad / dist_good)
    iv = (dist_bad - dist_good) * woe
    return iv


def _calculate_total_iv(bins_list: List[dict], total_goods: int, total_bads: int) -> float:
    """
    Calculate total Information Value for a binning configuration.
    
    Total IV is the sum of IV contributions from all bins.
    
    Parameters:
        bins_list (List[dict]): List of bin dictionaries with 'goods' and 'bads' keys.
        total_goods (int): Total goods across all bins.
        total_bads (int): Total bads across all bins.
    
    Returns:
        float: Total IV for this binning configuration.
    """
    total_iv = 0.0
    for bin_info in bins_list:
        total_iv += _calculate_bin_iv(
            bin_info['goods'], bin_info['bads'], 
            total_goods, total_bads
        )
    return total_iv


def _get_bin_woe(goods: int, bads: int, total_goods: int, total_bads: int) -> float:
    """
    Calculate WOE for a single bin.
    
    Parameters:
        goods (int): Number of goods in this bin.
        bads (int): Number of bads in this bin.
        total_goods (int): Total goods across all bins.
        total_bads (int): Total bads across all bins.
    
    Returns:
        float: WOE value for this bin.
    """
    # Handle edge cases
    if total_goods == 0 or total_bads == 0:
        return 0.0
    
    dist_good = goods / total_goods if total_goods > 0 else 0
    dist_bad = bads / total_bads if total_bads > 0 else 0
    
    # Apply epsilon smoothing to avoid log(0) or division by zero
    if dist_good == 0:
        dist_good = 0.0001  # Small epsilon
    if dist_bad == 0:
        dist_bad = 0.0001
    
    return np.log(dist_bad / dist_good)


def _is_pure_bin(goods: int, bads: int) -> bool:
    """
    Check if a bin is pure (100% goods or 100% bads).
    
    Pure bins are problematic because:
    - They have infinite or undefined WOE
    - They indicate possible data issues or overfitting
    
    Parameters:
        goods (int): Number of goods in the bin.
        bads (int): Number of bads in the bin.
    
    Returns:
        bool: True if the bin is pure (all one class), False otherwise.
    """
    # Pure if either count is zero
    return goods == 0 or bads == 0


def _iv_optimal_get_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_bin_count: int = 20,
    min_iv_loss: float = 0.001
) -> List[float]:
    """
    IV-optimal binning algorithm that maximizes Information Value.
    
    Algorithm Overview:
    1. Start with dynamic granularity based on variable characteristics
    2. Merge pure bins first (to closest adjacent bin by WOE)
    3. Iteratively merge adjacent bins with smallest IV loss
    4. Stop at max_bins or when IV loss exceeds threshold
    
    Unlike DecisionTree and ChiMerge, this algorithm does NOT enforce
    monotonicity, making it suitable for fraud detection where non-monotonic
    patterns (e.g., "sweet spots") may exist.
    
    Parameters:
        x (pd.Series): Feature values to bin.
        y (pd.Series): Binary target values.
        min_prop (float): Minimum proportion of samples per bin.
        max_bins (int): Maximum number of bins (stopping rule).
        min_bin_count (int): Minimum count per bin.
        min_iv_loss (float): Stop merging when IV loss exceeds this threshold.
    
    Returns:
        List[float]: Split thresholds for the variable.
    """
    # Clean data - remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    # Return empty if no valid data
    if len(x_clean) == 0:
        return []
    
    # Calculate basic statistics
    n_samples = len(x_clean)
    n_unique = len(np.unique(x_clean))
    total_goods = int((y_clean == 0).sum())  # Count of 0s
    total_bads = int((y_clean == 1).sum())   # Count of 1s
    
    # If all same class, can't compute IV
    if total_goods == 0 or total_bads == 0:
        return []
    
    # ==========================================================================
    # Step 1: Dynamic starting granularity
    # ==========================================================================
    # Choose initial number of bins based on data characteristics
    
    if n_unique <= 20:
        # Low cardinality: use unique values as initial bins
        initial_splits = sorted(np.unique(x_clean))[:-1]  # All unique values except last
    else:
        # High cardinality: use quantile-based initial bins
        # Number of initial bins scales with data size and uniqueness
        n_initial = min(
            max(20, n_unique // 5),  # At least 20, scale with uniqueness
            min(100, n_samples // 50),  # Cap at 100 or based on sample size
            n_unique - 1  # Can't have more bins than unique values
        )
        
        try:
            # Use quantiles for initial splits
            quantiles = np.linspace(0, 100, n_initial + 1)[1:-1]
            initial_splits = list(np.percentile(x_clean, quantiles))
            initial_splits = sorted(set(initial_splits))  # Remove duplicates
        except Exception:
            # Fallback to unique values
            initial_splits = sorted(np.unique(x_clean))[:-1]
    
    # If no splits possible, return empty
    if len(initial_splits) == 0:
        return []
    
    # ==========================================================================
    # Step 2: Create initial bin structure
    # ==========================================================================
    
    def create_bins_from_splits(splits: List[float]) -> List[dict]:
        """Create bin info list from splits."""
        bins_list = []
        edges = [-np.inf] + sorted(splits) + [np.inf]
        
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            
            # Create mask for this bin
            if lower == -np.inf:
                bin_mask = x_clean <= upper
            elif upper == np.inf:
                bin_mask = x_clean > lower
            else:
                bin_mask = (x_clean > lower) & (x_clean <= upper)
            
            count = int(bin_mask.sum())
            if count > 0:
                bads = int(y_clean[bin_mask].sum())
                goods = count - bads
                bins_list.append({
                    'lower': lower,
                    'upper': upper,
                    'count': count,
                    'goods': goods,
                    'bads': bads
                })
        
        return bins_list
    
    # Create initial bins
    current_splits = list(initial_splits)
    bins_list = create_bins_from_splits(current_splits)
    
    # If only 1 or fewer bins, return empty
    if len(bins_list) <= 1:
        return []
    
    # ==========================================================================
    # Step 3: Merge pure bins first (to closest by WOE)
    # ==========================================================================
    
    def merge_pure_bins_by_woe(bins_list: List[dict], splits: List[float]) -> Tuple[List[dict], List[float]]:
        """Merge pure bins to adjacent bin with closest WOE."""
        if len(bins_list) <= 2:
            return bins_list, splits
        
        # Calculate WOE for each bin
        woes = []
        for b in bins_list:
            woe = _get_bin_woe(b['goods'], b['bads'], total_goods, total_bads)
            woes.append(woe)
        
        # Iteratively merge pure bins
        merged = True
        while merged and len(bins_list) > 2:
            merged = False
            for i, b in enumerate(bins_list):
                if _is_pure_bin(b['goods'], b['bads']):
                    # Find adjacent bin with closest WOE
                    current_woe = woes[i]
                    
                    # Get adjacent WOEs
                    left_woe = woes[i - 1] if i > 0 else None
                    right_woe = woes[i + 1] if i < len(bins_list) - 1 else None
                    
                    # Determine which to merge with
                    if left_woe is None:
                        merge_with = i + 1
                    elif right_woe is None:
                        merge_with = i - 1
                    else:
                        # Merge with closer WOE
                        if abs(left_woe - current_woe) <= abs(right_woe - current_woe):
                            merge_with = i - 1
                        else:
                            merge_with = i + 1
                    
                    # Perform merge
                    merge_idx = min(i, merge_with)
                    other_idx = max(i, merge_with)
                    
                    merged_bin = {
                        'lower': bins_list[merge_idx]['lower'],
                        'upper': bins_list[other_idx]['upper'],
                        'count': bins_list[merge_idx]['count'] + bins_list[other_idx]['count'],
                        'goods': bins_list[merge_idx]['goods'] + bins_list[other_idx]['goods'],
                        'bads': bins_list[merge_idx]['bads'] + bins_list[other_idx]['bads']
                    }
                    
                    # Update bins list
                    bins_list = bins_list[:merge_idx] + [merged_bin] + bins_list[other_idx + 1:]
                    
                    # Update splits
                    if merge_idx < len(splits):
                        splits = splits[:merge_idx] + splits[merge_idx + 1:]
                    
                    # Recalculate WOEs
                    woes = [_get_bin_woe(b['goods'], b['bads'], total_goods, total_bads) for b in bins_list]
                    merged = True
                    break
        
        return bins_list, splits
    
    # Merge pure bins
    bins_list, current_splits = merge_pure_bins_by_woe(bins_list, current_splits)
    
    # ==========================================================================
    # Step 4: Iteratively merge bins with smallest IV loss until max_bins
    # ==========================================================================
    
    current_iv = _calculate_total_iv(bins_list, total_goods, total_bads)
    
    while len(bins_list) > max_bins:
        if len(bins_list) <= 2:
            break
        
        # Find pair of adjacent bins whose merge loses least IV
        min_iv_loss_found = float('inf')
        best_merge_idx = 0
        
        for i in range(len(bins_list) - 1):
            # Calculate IV loss if we merge bins i and i+1
            merged_goods = bins_list[i]['goods'] + bins_list[i + 1]['goods']
            merged_bads = bins_list[i]['bads'] + bins_list[i + 1]['bads']
            
            # IV of current two bins
            iv_before = (
                _calculate_bin_iv(bins_list[i]['goods'], bins_list[i]['bads'], total_goods, total_bads) +
                _calculate_bin_iv(bins_list[i + 1]['goods'], bins_list[i + 1]['bads'], total_goods, total_bads)
            )
            
            # IV of merged bin
            iv_after = _calculate_bin_iv(merged_goods, merged_bads, total_goods, total_bads)
            
            iv_loss = iv_before - iv_after
            
            if iv_loss < min_iv_loss_found:
                min_iv_loss_found = iv_loss
                best_merge_idx = i
        
        # Check stopping rule: if IV loss is too high, stop
        if min_iv_loss_found > min_iv_loss and len(bins_list) <= max_bins * 2:
            break
        
        # Perform the merge
        i = best_merge_idx
        merged_bin = {
            'lower': bins_list[i]['lower'],
            'upper': bins_list[i + 1]['upper'],
            'count': bins_list[i]['count'] + bins_list[i + 1]['count'],
            'goods': bins_list[i]['goods'] + bins_list[i + 1]['goods'],
            'bads': bins_list[i]['bads'] + bins_list[i + 1]['bads']
        }
        
        bins_list = bins_list[:i] + [merged_bin] + bins_list[i + 2:]
        
        # Update splits
        if i < len(current_splits):
            current_splits = current_splits[:i] + current_splits[i + 1:]
    
    # ==========================================================================
    # Step 5: Ensure minimum bin size constraints
    # ==========================================================================
    
    min_count = max(int(n_samples * min_prop), min_bin_count)
    
    # Merge small bins
    merged = True
    while merged and len(bins_list) > 2:
        merged = False
        for i, b in enumerate(bins_list):
            if b['count'] < min_count:
                # Merge with smaller adjacent bin
                if i == 0:
                    merge_with = 1
                elif i == len(bins_list) - 1:
                    merge_with = i - 1
                else:
                    # Merge with smaller neighbor
                    if bins_list[i - 1]['count'] <= bins_list[i + 1]['count']:
                        merge_with = i - 1
                    else:
                        merge_with = i + 1
                
                merge_idx = min(i, merge_with)
                other_idx = max(i, merge_with)
                
                merged_bin = {
                    'lower': bins_list[merge_idx]['lower'],
                    'upper': bins_list[other_idx]['upper'],
                    'count': bins_list[merge_idx]['count'] + bins_list[other_idx]['count'],
                    'goods': bins_list[merge_idx]['goods'] + bins_list[other_idx]['goods'],
                    'bads': bins_list[merge_idx]['bads'] + bins_list[other_idx]['bads']
                }
                
                bins_list = bins_list[:merge_idx] + [merged_bin] + bins_list[other_idx + 1:]
                
                if merge_idx < len(current_splits):
                    current_splits = current_splits[:merge_idx] + current_splits[merge_idx + 1:]
                
                merged = True
                break
    
    # Extract final splits from bins
    final_splits = []
    for b in bins_list[:-1]:  # All bins except last
        if b['upper'] != np.inf:
            final_splits.append(b['upper'])
    
    return sorted(final_splits)


# =============================================================================
# Main Binning Functions
# =============================================================================
# These functions orchestrate the binning process and create the output tables.

def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """
    Create bin DataFrame for a numeric variable based on split points.
    
    This function takes a list of split points and creates a DataFrame with
    bin rules, counts, and good/bad breakdowns for each bin.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        var (str): Name of the numeric variable to bin.
        y_var (str): Name of the binary target variable.
        splits (List[float]): Split points defining bin boundaries.
    
    Returns:
        pd.DataFrame: DataFrame with columns [var, bin, count, bads, goods]
    """
    # Get the variable and target as Series
    x = df[var]
    y = df[y_var]
    
    # Initialize list to hold bin data
    bins_data = []
    
    # Sort splits and add -inf and +inf as boundaries
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    # Create a bin for each pair of adjacent edges
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        
        # Create mask and rule string based on bin position
        if lower == -np.inf:
            # First bin: x <= upper
            mask = (x <= upper) & x.notna()
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            # Last bin: x > lower
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            # Middle bin: lower < x <= upper
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        # Calculate statistics for this bin
        count = mask.sum()
        if count > 0:
            bads = y[mask].sum()    # Sum of 1s = bad count
            goods = count - bads     # Remainder = good count
            bins_data.append({
                'var': var,          # Variable name
                'bin': bin_rule,     # Rule string
                'count': int(count), # Total observations
                'bads': int(bads),   # Bad (event) count
                'goods': int(goods)  # Good (non-event) count
            })
    
    # Handle missing values separately - they get their own bin
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


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    max_categories: int = 50,
    max_bins: int = None
) -> pd.DataFrame:
    """
    Create bin DataFrame for a factor/categorical variable.
    
    Strategy:
    - Low-cardinality (≤ max_bins): Each category becomes its own bin
    - Medium-cardinality (> max_bins, ≤ max_categories): Groups categories by WOE similarity
      * Unlike numeric bins, categorical bins don't need adjacency
      * Any categories with similar WOE (risk profile) can be grouped together
      * Example: Group high-risk states together, low-risk states together
    - High-cardinality (> max_categories): Skip variable with warning
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        var (str): Variable name.
        y_var (str): Target variable name.
        max_categories (int): Maximum categories before skipping (default 50).
        max_bins (int): Maximum bins to create (uses MAX_BINS from config if None).
        
    Returns:
        pd.DataFrame: DataFrame with bin statistics.
    """
    # Get variable and target
    x = df[var]
    y = df[y_var]
    
    # Initialize list for bin data
    bins_data = []
    
    # Get unique non-null values
    unique_vals = x.dropna().unique()
    n_unique = len(unique_vals)
    
    # Use config value if max_bins not specified
    if max_bins is None:
        max_bins = BinningConfig.MAX_BINS
    
    # Strategy 1: Low cardinality - one bin per category
    if n_unique <= max_bins:
        for val in unique_vals:
            mask = x == val
            count = mask.sum()
            if count > 0:
                bads = y[mask].sum()
                goods = count - bads
                # R-style rule: var %in% c("value")
                bins_data.append({
                    'var': var,
                    'bin': f'{var} %in% c("{val}")',
                    'count': int(count),
                    'bads': int(bads),
                    'goods': int(goods)
                })
    
    # Strategy 2: Medium cardinality - group by WOE similarity
    elif n_unique <= max_categories:
        print(f"  INFO: Variable '{var}' has {n_unique} categories - grouping into {max_bins} bins by WOE similarity")
        
        # Calculate WOE for each category
        cat_stats = []
        total_goods = (y == 0).sum()
        total_bads = (y == 1).sum()
        
        # Check for degenerate case
        if total_goods == 0 or total_bads == 0:
            print(f"  WARNING: Cannot calculate WOE for '{var}' - no variation in target")
            return pd.DataFrame()
        
        # Calculate WOE for each unique value
        for val in unique_vals:
            mask = x == val
            count = mask.sum()
            if count > 0:
                bads = y[mask].sum()
                goods = count - bads
                
                # Calculate WOE for this category
                dist_goods = (goods / total_goods) if total_goods > 0 else 0.0001
                dist_bads = (bads / total_bads) if total_bads > 0 else 0.0001
                
                # Apply smoothing to avoid inf/-inf
                dist_goods = max(dist_goods, 0.0001)
                dist_bads = max(dist_bads, 0.0001)
                
                woe = np.log(dist_bads / dist_goods)
                
                cat_stats.append({
                    'value': val,
                    'count': count,
                    'bads': int(bads),
                    'goods': int(goods),
                    'woe': woe
                })
        
        if not cat_stats:
            return pd.DataFrame()
        
        # Create DataFrame from category stats
        cat_df = pd.DataFrame(cat_stats)
        
        # Group categories by WOE similarity (not by order!)
        n_bins = min(max_bins, len(cat_df))
        
        try:
            # Create bins based on WOE quantiles
            cat_df['bin_group'] = pd.qcut(cat_df['woe'], q=n_bins, labels=False, duplicates='drop')
        except (ValueError, IndexError):
            # If qcut fails, use simple cut
            cat_df['bin_group'] = pd.cut(cat_df['woe'], bins=n_bins, labels=False, duplicates='drop')
        
        # Handle any NaN bin_groups
        if cat_df['bin_group'].isna().any():
            median_group = cat_df['bin_group'].median()
            cat_df['bin_group'] = cat_df['bin_group'].fillna(median_group)
            cat_df['bin_group'] = cat_df['bin_group'].fillna(0)
        
        # Aggregate by bin group
        for bin_idx in sorted(cat_df['bin_group'].unique()):
            bin_cats = cat_df[cat_df['bin_group'] == bin_idx]
            values_str = '", "'.join([str(v) for v in bin_cats['value'].tolist()])
            
            total_count = bin_cats['count'].sum()
            total_bads = bin_cats['bads'].sum()
            total_goods = bin_cats['goods'].sum()
            
            avg_woe = bin_cats['woe'].mean()
            
            bins_data.append({
                'var': var,
                'bin': f'{var} %in% c("{values_str}")',
                'count': int(total_count),
                'bads': int(total_bads),
                'goods': int(total_goods)
            })
            
            # Log the grouping for transparency
            if len(bin_cats) > 1:
                print(f"    Grouped {len(bin_cats)} categories (avg WOE: {avg_woe:.3f})")
    
    # Strategy 3: Very high cardinality - skip with warning
    else:
        print(f"  WARNING: Variable '{var}' has {n_unique} unique categories!")
        print(f"    Exceeds max_categories={max_categories}. Skipping variable.")
        print(f"    Recommendation: Recode variable before WOE binning or increase MaxCategories")
        return pd.DataFrame()
    
    # Handle missing values separately
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
    Update bin statistics (proportion, bad_rate, iv, entropy, trend, etc.)
    
    This function takes a basic bin DataFrame and adds calculated statistics:
    - propn: Percentage of total observations in this bin
    - bad_rate: Percentage of bads in this bin
    - goodCap: Capture rate of goods (% of all goods in this bin)
    - badCap: Capture rate of bads (% of all bads in this bin)
    - iv: Information Value contribution from this bin
    - ent: Entropy of this bin
    - purNode: 'Y' if bin is pure (all goods or all bads), 'N' otherwise
    - trend: 'I' (increasing) or 'D' (decreasing) compared to previous bin
    
    Parameters:
        bin_df (pd.DataFrame): DataFrame with basic bin info (var, bin, count, bads, goods)
    
    Returns:
        pd.DataFrame: DataFrame with additional calculated columns.
    """
    # Return empty if input is empty
    if bin_df.empty:
        return bin_df
    
    # Make a copy to avoid modifying original
    df = bin_df.copy()
    
    # Calculate totals
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # Calculate proportion (percentage of total in each bin)
    df['propn'] = round(df['count'] / total_count * 100, 2)
    
    # Calculate bad rate (percentage of bads within each bin)
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    
    # Calculate capture rates (what % of all goods/bads are in this bin)
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    # Calculate IV contribution for each bin
    # IV = (goodCap - badCap) * ln(goodCap / badCap)
    df['iv'] = round((df['goodCap'] - df['badCap']) * np.log(
        np.where(df['goodCap'] == 0, 0.0001, df['goodCap']) / 
        np.where(df['badCap'] == 0, 0.0001, df['badCap'])
    ), 4)
    
    # Replace any infinity values with 0
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    # Calculate entropy for each bin
    df['ent'] = df.apply(
        lambda row: calculate_entropy(row['goods'], row['bads']), 
        axis=1
    )
    
    # Mark pure nodes (bins with 0 goods or 0 bads)
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    # Calculate trend (increasing or decreasing bad rate)
    df['trend'] = None
    bad_rates = df['bad_rate'].values
    for i in range(1, len(bad_rates)):
        # Skip NA bins for trend calculation
        if 'is.na' not in str(df.iloc[i]['bin']):
            if bad_rates[i] >= bad_rates[i-1]:
                df.iloc[i, df.columns.get_loc('trend')] = 'I'  # Increasing
            else:
                df.iloc[i, df.columns.get_loc('trend')] = 'D'  # Decreasing
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Add a total (summary) row to the bin DataFrame.
    
    The total row contains aggregate statistics for the entire variable:
    - Total count, goods, bads
    - Total IV
    - Weighted average entropy
    - Overall trend direction
    - Whether monotonicity is achieved
    - Flip ratio (how many trend changes)
    - Number of bins
    
    Parameters:
        bin_df (pd.DataFrame): DataFrame with bin statistics.
        var (str): Variable name (for the 'var' column in total row).
    
    Returns:
        pd.DataFrame: Original DataFrame with total row appended.
    """
    df = bin_df.copy()
    
    # Calculate totals
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()  # Sum of IV contributions
    
    # Calculate weighted average entropy
    if total_count > 0:
        total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4)
    else:
        total_ent = 0
    
    # Check if trend is monotonic (all same direction)
    trends = df[df['trend'].notna()]['trend'].unique()
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    # Calculate flip ratio (measure of non-monotonicity)
    incr_count = len(df[df['trend'] == 'I'])
    decr_count = len(df[df['trend'] == 'D'])
    total_trend_count = incr_count + decr_count
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    # Determine overall trend direction (majority)
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    
    # Check if any bin is pure
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    
    # Count number of bins
    num_bins = len(df)
    
    # Create total row as DataFrame
    total_row = pd.DataFrame([{
        'var': var,
        'bin': 'Total',
        'count': total_count,
        'bads': total_bads,
        'goods': total_goods,
        'propn': 100.0,
        'bad_rate': round(total_bads / total_count * 100, 2) if total_count > 0 else 0,
        'goodCap': 1.0,
        'badCap': 1.0,
        'iv': round(total_iv, 4),
        'ent': total_ent,
        'purNode': has_pure_node,
        'trend': overall_trend,
        'monTrend': mon_trend,
        'flipRatio': round(flip_ratio, 4),
        'numBins': num_bins
    }])
    
    # Concatenate and return
    return pd.concat([df, total_row], ignore_index=True)


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    min_prop: float = 0.01,  # Default to 1% (matches R's logiBin minProp)
    max_bins: int = 10,
    enforce_monotonic: bool = True,
    algorithm: str = None,  # "DecisionTree" or "ChiMerge"
    use_enhancements: bool = None,  # Master switch for all enhancements
    adaptive_min_prop: bool = None,  # Individual: adaptive min_prop for sparse data
    min_event_count: bool = None,  # Individual: minimum event count per bin
    auto_retry: bool = None  # Individual: auto-retry with relaxed constraints
) -> BinResult:
    """
    Get optimal bins for multiple variables.
    
    This is the main entry point for binning, equivalent to logiBin::getBins in R.
    It processes multiple variables and returns comprehensive binning results.
    
    Algorithm Options:
    - "DecisionTree" (default): Uses CART decision tree - matches R's logiBin::getBins
    - "ChiMerge": Uses chi-square based bin merging (more conservative)
    - "IVOptimal": Maximizes Information Value (non-monotonic allowed)
    
    Parameters:
        df (pd.DataFrame): DataFrame with data.
        y_var (str): Name of binary dependent variable (0/1).
        x_vars (List[str]): List of independent variable names to bin.
        min_prop (float): Minimum proportion of samples per bin (default 0.01 = 1%).
        max_bins (int): Maximum number of bins per variable.
        enforce_monotonic (bool): Whether to enforce monotonic WOE trends.
        algorithm (str): Binning algorithm to use.
        use_enhancements (bool): Master switch - enables all enhancements when True.
        adaptive_min_prop (bool): Individual flag for adaptive min_prop.
        min_event_count (bool): Individual flag for minimum event count.
        auto_retry (bool): Individual flag for auto-retry.
    
    Returns:
        BinResult: Named tuple with var_summary and bin DataFrames.
    """
    # Initialize containers for results
    all_bins = []        # Will hold bin DataFrames for each variable
    var_summaries = []   # Will hold summary info for each variable
    
    # Use config values if not specified
    if algorithm is None:
        algorithm = BinningConfig.ALGORITHM
    if use_enhancements is None:
        use_enhancements = BinningConfig.USE_ENHANCEMENTS
    
    # Resolve individual flags: use config if not explicitly passed
    if adaptive_min_prop is None:
        adaptive_min_prop = BinningConfig.ADAPTIVE_MIN_PROP
    if min_event_count is None:
        min_event_count = BinningConfig.MIN_EVENT_COUNT
    if auto_retry is None:
        auto_retry = BinningConfig.AUTO_RETRY
    
    # Use the min_prop as passed (don't override)
    min_bin_pct = min_prop
    
    # Progress tracking
    total_vars = len(x_vars)
    start_time = time.time()
    last_log_time = start_time
    processed_count = 0
    times_per_var = []
    
    # Determine display string for algorithm mode
    any_enhancement = use_enhancements or adaptive_min_prop or min_event_count or auto_retry
    if algorithm == "DecisionTree":
        algo_display = "DecisionTree (R-compatible)" if not any_enhancement else "DecisionTree (Enhanced)"
    elif algorithm == "IVOptimal":
        algo_display = "IVOptimal (Maximize IV)"
    else:
        algo_display = "ChiMerge"
    
    # Log start
    log_progress(f"Starting binning for {total_vars} variables (Algorithm: {algo_display})")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    log_progress(f"Settings: min_bin_pct={min_bin_pct:.1%}, max_bins={max_bins}, monotonic={enforce_monotonic}")
    
    if any_enhancement:
        enhancements_list = []
        if use_enhancements:
            enhancements_list.append("ALL")
        else:
            if adaptive_min_prop:
                enhancements_list.append("AdaptiveMinProp")
            if min_event_count:
                enhancements_list.append("MinEventCount")
            if auto_retry:
                enhancements_list.append("AutoRetry")
        log_progress(f"Enhancements: {', '.join(enhancements_list)}")
    
    # Process each variable
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        # Skip if variable not in DataFrame
        if var not in df.columns:
            continue
        
        # Determine variable type (numeric or factor)
        var_type = get_var_type(df[var])
        
        # Process based on variable type
        if var_type == 'numeric':
            # Use appropriate algorithm for numeric variables
            if algorithm == "ChiMerge":
                # ChiMerge algorithm
                splits = _chimerge_get_splits(
                    df[var], 
                    df[y_var], 
                    min_bin_pct=min_bin_pct,
                    min_bin_count=BinningConfig.MIN_BIN_COUNT,
                    max_bins=max_bins,
                    min_bins=BinningConfig.MIN_BINS,
                    chi_threshold=BinningConfig.CHI_MERGE_THRESHOLD
                )
            elif algorithm == "IVOptimal":
                # IV-optimal algorithm
                splits = _iv_optimal_get_splits(
                    df[var],
                    df[y_var],
                    min_prop=min_bin_pct,
                    max_bins=max_bins,
                    min_bin_count=BinningConfig.MIN_BIN_COUNT,
                    min_iv_loss=BinningConfig.MIN_IV_GAIN
                )
            else:
                # Decision Tree (R-compatible)
                splits = _get_decision_tree_splits(
                    df[var], 
                    df[y_var], 
                    min_prop=min_bin_pct,
                    max_bins=max_bins,
                    use_enhancements=use_enhancements,
                    adaptive_min_prop=adaptive_min_prop,
                    min_event_count=min_event_count,
                    auto_retry=auto_retry
                )
            
            # Enforce monotonicity if requested
            if enforce_monotonic and len(splits) > 0:
                splits = _enforce_monotonicity(df[var], df[y_var], splits, direction='auto')
            
            # Create bin DataFrame
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            # Factor variables - use category-based binning
            bin_df = _create_factor_bins(df, var, y_var, 
                                         max_categories=BinningConfig.MAX_CATEGORIES,
                                         max_bins=max_bins)
        
        # Skip if no bins created
        if bin_df.empty:
            continue
        
        # Add calculated statistics
        bin_df = update_bin_stats(bin_df)
        bin_df = add_total_row(bin_df, var)
        
        # Extract summary info from total row
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
        
        # Add to results
        all_bins.append(bin_df)
        
        # Progress logging
        var_time = time.time() - var_start
        times_per_var.append(var_time)
        processed_count += 1
        
        # Log every 10 variables, every 5 seconds, or at milestones
        current_time = time.time()
        should_log = (
            processed_count % 10 == 0 or 
            processed_count == 1 or
            current_time - last_log_time >= 5.0 or
            processed_count == total_vars
        )
        
        if should_log:
            pct = (processed_count / total_vars) * 100
            elapsed = current_time - start_time
            avg_time = sum(times_per_var) / len(times_per_var)
            remaining = (total_vars - processed_count) * avg_time
            mono_status = "Y" if total_row.get('monTrend', 'N') == 'Y' else "N"
            
            log_progress(
                f"[{processed_count}/{total_vars}] {pct:.1f}% | "
                f"Variable: {var[:25]:25} | "
                f"IV: {total_row['iv']:.4f} | "
                f"Mono: {mono_status} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(remaining)}"
            )
            last_log_time = current_time
    
    # Final logging
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {processed_count} variables in {format_time(total_time)}")
    log_progress(f"Average time per variable: {format_time(total_time/max(processed_count,1))}")
    
    # Combine all bin DataFrames
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    # Create summary DataFrame
    var_summary_df = pd.DataFrame(var_summaries)
    
    return BinResult(var_summary=var_summary_df, bin=combined_bins)

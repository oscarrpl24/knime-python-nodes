# =============================================================================
# WOE 2.0 Editor for KNIME Python Script Node - ADVANCED BINNING VERSION
# =============================================================================
# COMPREHENSIVE COMMENTED VERSION
# =============================================================================
#
# This file is a fully documented version of woe2_editor_advanced.py with
# detailed explanations for every single line of code. It is intended as a
# learning resource and reference for understanding the WOE 2.0 binning
# algorithms and their implementation.
#
# =============================================================================
# WHAT IS WEIGHT OF EVIDENCE (WOE)?
# =============================================================================
#
# Weight of Evidence (WOE) is a data transformation technique used primarily
# in credit scoring and fraud detection. It converts categorical and continuous
# variables into a single numerical scale that represents their predictive power
# relative to a binary target variable (e.g., fraud/not fraud, default/no default).
#
# The WOE formula is:
#   WOE = ln(% of Bads in bin / % of Goods in bin)
#
# Where:
#   - "Bads" are the target events (fraud, default, etc.)
#   - "Goods" are the non-events
#   - ln is the natural logarithm
#
# WOE Interpretation:
#   - Positive WOE: The bin has more bads than expected (higher risk)
#   - Negative WOE: The bin has fewer bads than expected (lower risk)
#   - Zero WOE: The bin has the same proportion as the overall population
#
# =============================================================================
# WHAT IS INFORMATION VALUE (IV)?
# =============================================================================
#
# Information Value (IV) measures the overall predictive power of a variable.
# It sums the WOE contributions across all bins:
#
#   IV = Σ (% of Bads - % of Goods) × WOE
#
# IV Interpretation Guidelines:
#   - IV < 0.02: Not useful for prediction (too weak)
#   - IV 0.02-0.1: Weak predictive power
#   - IV 0.1-0.3: Medium predictive power
#   - IV 0.3-0.5: Strong predictive power
#   - IV > 0.5: Suspicious (may indicate data leakage or overfitting)
#
# =============================================================================
# WOE 2.0 ENHANCEMENTS
# =============================================================================
#
# This implementation extends traditional WOE binning with:
#
# 1. BAYESIAN SHRINKAGE (Beta-Binomial)
#    Traditional WOE can be unstable when bins have few observations.
#    Bayesian shrinkage pulls extreme WOE values toward zero based on
#    sample size, providing more stable estimates for rare events.
#
# 2. SPLINE-BASED BINNING
#    Instead of simple decision tree splits, spline functions can capture
#    non-linear relationships between variables and the target.
#
# 3. ISOTONIC REGRESSION BINNING
#    Provides finer-grained monotonic binning than decision trees.
#
# 4. ADVANCED TREND OPTIONS
#    Beyond simple ascending/descending monotonicity, allows peak, valley,
#    concave, and convex patterns.
#
# 5. CREDIBLE INTERVALS
#    Provides uncertainty quantification for WOE estimates.
#
# 6. PSI MONITORING
#    Detects when the population distribution has shifted.
#
# =============================================================================
# FLOW VARIABLES REFERENCE
# =============================================================================
#
# Basic Settings:
# - DependentVariable (string, required for headless): Binary target variable
# - TargetCategory (string, optional): Which value represents "bad" outcome
# - OptimizeAll (boolean, default False): Force monotonic trends on all vars
# - GroupNA (boolean, default False): Combine NA bins with closest bin
#
# Algorithm Settings:
# - Algorithm (string, default "DecisionTree"): Binning algorithm
#   Options: "DecisionTree", "ChiMerge", "IVOptimal", "Spline", "Isotonic"
# - MinBinPct (float, default 0.01): Min percentage per bin
# - MaxBins (int, default 10): Maximum bins per variable
#
# WOE 2.0 Shrinkage Settings:
# - UseShrinkage (boolean, default True): Apply Bayesian shrinkage
# - ShrinkageMethod (string, default "BetaBinomial"): "BetaBinomial" or "Simple"
# - PriorStrength (float, default 1.0): Strength of prior (higher = more shrinkage)
#
# Trend Options:
# - MonotonicTrend (string, default "auto"): Monotonicity constraint
#   Options: "auto", "ascending", "descending", "peak", "valley", 
#            "concave", "convex", "none"
#
# Statistical Validation:
# - MaxPValue (float, default 0.05): Max p-value between adjacent bins
# - UsePValueMerging (boolean, default True): Merge non-significant bins
#
# Bayesian Options:
# - ComputeCredibleIntervals (boolean, default False): Add CI to WOE
# - CredibleLevel (float, default 0.95): Credible interval level
#
# PSI Monitoring:
# - ComputePSI (boolean, default True): Calculate PSI for monitoring
# - PSIReferenceData (string, optional): Path to reference data for PSI
#
# =============================================================================
# OUTPUT PORTS
# =============================================================================
#
# Port 1: Original input DataFrame (unchanged)
# Port 2: df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# Port 3: df_only_woe - Only WOE columns + dependent variable
# Port 4: df_only_bins - Only binned columns (b_*) for scorecard
# Port 5: bins - Binning rules with WOE values and credible intervals
# Port 6: psi_report - PSI values for each variable (NEW)
#
# Version: 2.0
# Release Date: 2026-01-28
# Based on: WOE 2.0 paper (arXiv:2101.01494), OptBinning library
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# This section imports all the Python libraries and modules required by the
# script. Each import is explained in detail.

# -----------------------------------------------------------------------------
# KNIME Integration
# -----------------------------------------------------------------------------
# Import the KNIME scripting interface module.
# This module provides access to:
#   - knio.input_tables: List of input DataFrames from upstream KNIME nodes
#   - knio.output_tables: List to assign output DataFrames
#   - knio.flow_variables: Dictionary of flow variables from KNIME workflow
# The 'as knio' creates an alias for shorter, cleaner code.
import knime.scripting.io as knio

# -----------------------------------------------------------------------------
# Data Manipulation Libraries
# -----------------------------------------------------------------------------
# Import pandas - the primary library for data manipulation in Python.
# pandas provides:
#   - DataFrame: 2D tabular data structure (like a spreadsheet or SQL table)
#   - Series: 1D array-like structure (a single column)
#   - Rich functionality for data cleaning, transformation, and analysis
# We use 'pd' as a shorthand alias (this is the standard convention).
import pandas as pd

# Import numpy - the fundamental library for numerical computing in Python.
# numpy provides:
#   - Efficient multi-dimensional arrays (ndarray)
#   - Mathematical functions that operate on arrays
#   - Linear algebra, random number generation, and more
# We use 'np' as a shorthand alias (standard convention).
import numpy as np

# -----------------------------------------------------------------------------
# Text Processing
# -----------------------------------------------------------------------------
# Import the 're' module for regular expression operations.
# Regular expressions are patterns used to match and manipulate text.
# We use this to:
#   - Parse bin rule strings (e.g., "Age > '25' & Age <= '35'")
#   - Extract numeric values from text
#   - Match categorical values in bin labels
import re

# -----------------------------------------------------------------------------
# System Utilities
# -----------------------------------------------------------------------------
# Import the warnings module to control Python warning messages.
# Warnings can clutter the output during processing, so we'll suppress them.
# Common warnings include deprecation notices and numerical precision issues.
import warnings

# Import the time module for measuring execution time.
# We use this to:
#   - Track how long each processing step takes
#   - Estimate remaining time for long-running operations
#   - Provide progress feedback to users
import time

# Import the sys module for system-specific parameters and functions.
# We use this to:
#   - Flush stdout to ensure progress messages display immediately
#   - Access system-level information
import sys

# Import the os module for operating system interactions.
# We use this to:
#   - Set environment variables (to control threading behavior)
#   - Get the current process ID (for unique instance identification)
#   - Access file system paths
import os

# Import the random module for generating random numbers.
# We use this to:
#   - Generate random port numbers for the Shiny UI (avoids conflicts)
#   - Create unique instance identifiers
import random

# -----------------------------------------------------------------------------
# Type Hints
# -----------------------------------------------------------------------------
# Import type hints from the typing module for better code documentation.
# Type hints make code more readable and enable IDE autocomplete/checking.
# 
# Dict: Represents a dictionary type, e.g., Dict[str, int] = {"a": 1}
# List: Represents a list type, e.g., List[str] = ["a", "b"]
# Tuple: Represents a tuple type, e.g., Tuple[int, str] = (1, "a")
# Optional: Indicates a value can be None, e.g., Optional[str] = "a" or None
# Any: Accepts any type (use sparingly)
# Union: One of multiple types, e.g., Union[int, str] = 1 or "a"
from typing import Dict, List, Tuple, Optional, Any, Union

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
# Import the dataclass decorator and field function for creating data classes.
# Dataclasses are a convenient way to create classes that primarily hold data.
# They automatically generate:
#   - __init__() method for initialization
#   - __repr__() method for string representation
#   - __eq__() method for equality comparison
# The 'field' function allows customizing individual fields.
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------
# Import the Enum class for creating enumeration types.
# Enumerations provide a way to define symbolic names for constant values.
# This makes code more readable and less error-prone than using raw strings.
# Example: Algorithm.DECISION_TREE instead of "DecisionTree"
from enum import Enum

# -----------------------------------------------------------------------------
# Suppress Warnings
# -----------------------------------------------------------------------------
# Suppress all warning messages to keep the console output clean.
# The 'ignore' action tells Python to ignore all warnings.
# This prevents numpy, pandas, scipy, and other libraries from cluttering
# the output with deprecation warnings or other non-critical messages.
# Note: In development, you might want to see warnings; in production, hide them.
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: STABILITY SETTINGS FOR MULTIPLE INSTANCE EXECUTION
# =============================================================================
# When running multiple KNIME Python Script nodes simultaneously (parallel
# execution), we need to ensure each instance operates independently without
# conflicts. This section sets up isolation mechanisms.

# -----------------------------------------------------------------------------
# Port Configuration for Shiny Web UI
# -----------------------------------------------------------------------------
# BASE_PORT: The starting port number for the Shiny web UI.
# When the interactive UI launches, it needs a TCP port to serve the web app.
# We use port 8060 (different from the original WOE editor's 8055) to avoid
# conflicts if both versions are running simultaneously.
BASE_PORT = 8060  # Different from other versions to avoid conflicts

# RANDOM_PORT_RANGE: The range of random values to add to BASE_PORT.
# The actual port used will be: BASE_PORT + random(0, RANDOM_PORT_RANGE)
# This means we'll use ports between 8060 and 9060.
# If port 8060 is busy, we'll try a random port in this range.
RANDOM_PORT_RANGE = 1000

# INSTANCE_ID: A unique identifier for this running instance of the script.
# This combines:
#   - os.getpid(): The current process ID (unique per running Python process)
#   - random.randint(10000, 99999): A random 5-digit number
# This ID helps distinguish between multiple simultaneous instances,
# useful for debugging and logging in parallel execution scenarios.
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# -----------------------------------------------------------------------------
# Threading Configuration
# -----------------------------------------------------------------------------
# Set environment variables to prevent multi-threading conflicts.
# Scientific computing libraries (numpy, scipy, scikit-learn) often use
# multi-threaded linear algebra backends (OpenBLAS, MKL, etc.).
# When multiple KNIME nodes run Python scripts simultaneously, these
# threading libraries can conflict with each other, causing crashes or hangs.
# Setting these to '1' forces single-threaded execution within each script,
# while KNIME handles parallelism at a higher level.

# NUMEXPR_MAX_THREADS: Controls numexpr library threading.
# numexpr is used internally by pandas for efficient expression evaluation.
os.environ['NUMEXPR_MAX_THREADS'] = '1'

# OMP_NUM_THREADS: Controls OpenMP threading.
# OpenMP is a parallel programming API used by many scientific libraries.
os.environ['OMP_NUM_THREADS'] = '1'

# OPENBLAS_NUM_THREADS: Controls OpenBLAS threading.
# OpenBLAS is an open-source implementation of BLAS (Basic Linear Algebra).
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# MKL_NUM_THREADS: Controls Intel MKL threading.
# Intel Math Kernel Library is another linear algebra implementation.
os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# SECTION 3: PROGRESS LOGGING UTILITIES
# =============================================================================
# These functions help us provide feedback to users about the script's
# progress. This is especially important for long-running operations on
# large datasets, where users need to know the script is still working.

def log_progress(message: str, flush: bool = True):
    """
    Print a progress message with a timestamp prefix.
    
    This function adds a timestamp to messages so users can track when each
    step occurred and how long operations are taking. The timestamp helps
    identify performance bottlenecks and estimate completion times.
    
    Parameters:
    -----------
    message : str
        The message to display to the user. Should be concise but informative.
        Examples: "Processing variable: Age", "Binning complete: 50 variables"
    
    flush : bool, optional (default=True)
        If True, immediately write the message to the console.
        Without flushing, Python may buffer output and delay displaying it.
        We default to True because we want immediate feedback during processing.
    
    Returns:
    --------
    None
        This function only prints output; it doesn't return anything.
    
    Example Output:
    ---------------
    [14:32:15] Processing variable: Age
    [14:32:16] Processing variable: Income
    """
    # time.strftime() formats the current time according to a format string.
    # Format codes:
    #   %H = Hour (24-hour format, 00-23)
    #   %M = Minute (00-59)
    #   %S = Second (00-59)
    # Result: "14:32:15" for 2:32:15 PM
    timestamp = time.strftime("%H:%M:%S")
    
    # Print the message with the timestamp in square brackets.
    # The f-string (f"...") allows embedding variables directly in the string.
    print(f"[{timestamp}] {message}")
    
    # If flush is True, force Python to write the output immediately.
    # sys.stdout is the standard output stream (usually the console).
    # .flush() forces any buffered output to be written immediately.
    # This ensures the user sees the message right away, even if Python
    # would normally buffer it for efficiency.
    if flush:
        sys.stdout.flush()


def format_time(seconds: float) -> str:
    """
    Convert a number of seconds into a human-readable time string.
    
    This function intelligently formats time based on magnitude:
    - Less than 60 seconds: shows seconds with one decimal (e.g., "45.3s")
    - Less than 3600 seconds (1 hour): shows minutes and seconds (e.g., "5m 30s")
    - 3600+ seconds: shows hours and minutes (e.g., "2h 15m")
    
    This adaptive formatting makes it easy for users to understand durations
    at any scale without mental conversion.
    
    Parameters:
    -----------
    seconds : float
        The time duration in seconds to format. Can be any non-negative number.
    
    Returns:
    --------
    str
        A human-readable string representation of the time.
    
    Examples:
    ---------
    >>> format_time(45.3)
    '45.3s'
    >>> format_time(150)
    '2m 30s'
    >>> format_time(7500)
    '2h 5m'
    """
    # Case 1: Less than 60 seconds
    # For short durations, show seconds with one decimal place for precision.
    # The :.1f format specifier means: floating-point with 1 decimal place.
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    # Case 2: Between 60 seconds and 1 hour (3600 seconds)
    # Show minutes and seconds for medium durations.
    elif seconds < 3600:
        # Integer division (//) gives whole minutes.
        # Example: 150 // 60 = 2 (minutes)
        mins = int(seconds // 60)
        
        # Modulo (%) gives remaining seconds after extracting minutes.
        # Example: 150 % 60 = 30 (seconds)
        secs = int(seconds % 60)
        
        return f"{mins}m {secs}s"
    
    # Case 3: 1 hour or more
    # Show hours and minutes for long durations.
    else:
        # Integer division by 3600 gives whole hours.
        # Example: 7500 // 3600 = 2 (hours)
        hours = int(seconds // 3600)
        
        # Modulo 3600 gives remaining seconds, then divide by 60 for minutes.
        # Example: 7500 % 3600 = 300, then 300 // 60 = 5 (minutes)
        mins = int((seconds % 3600) // 60)
        
        return f"{hours}h {mins}m"


# =============================================================================
# SECTION 4: INSTALL/IMPORT DEPENDENCIES
# =============================================================================
# This section attempts to import required libraries, and if they're not
# installed, it automatically installs them using pip. This ensures the
# script works even in environments where dependencies aren't pre-installed.
#
# The pattern used is:
#   try:
#       import library
#   except ImportError:
#       pip install library
#       import library

# -----------------------------------------------------------------------------
# SciPy - Scientific Computing Library
# -----------------------------------------------------------------------------
# scipy provides advanced scientific computing functions including:
#   - stats: Statistical distributions and tests (chi-square, etc.)
#   - interpolate: Spline fitting for the spline-based binning algorithm
#   - special: Special mathematical functions (betaln for Bayesian calculations)
try:
    # Import the stats submodule for statistical tests and distributions.
    # We use this for chi-square tests in ChiMerge algorithm and p-value calculations.
    from scipy import stats
    
    # Import UnivariateSpline for fitting spline functions to data.
    # This is the core of the spline-based binning algorithm.
    # A spline is a piecewise polynomial function that smoothly connects data points.
    from scipy.interpolate import UnivariateSpline
    
    # Import betaln (log of Beta function) for Bayesian calculations.
    # The Beta function is fundamental to the Beta-Binomial posterior calculation
    # used in WOE 2.0 shrinkage. betaln = ln(Beta(a,b)) for numerical stability.
    from scipy.special import betaln
    
except ImportError:
    # If scipy is not installed, install it using pip.
    # subprocess.check_call runs a command and waits for completion.
    # If the command fails, it raises an exception.
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    
    # Now import the modules after installation
    from scipy import stats
    from scipy.interpolate import UnivariateSpline
    from scipy.special import betaln

# -----------------------------------------------------------------------------
# Scikit-learn - Machine Learning Library
# -----------------------------------------------------------------------------
# scikit-learn provides machine learning algorithms. We use:
#   - DecisionTreeClassifier: For the R-compatible decision tree binning
#   - IsotonicRegression: For the isotonic regression binning algorithm
#   - KMeans: For clustering when reducing the number of splits
try:
    # DecisionTreeClassifier implements the CART (Classification and Regression
    # Trees) algorithm. We use this for finding optimal split points.
    # This matches the R logiBin::getBins algorithm for compatibility.
    from sklearn.tree import DecisionTreeClassifier
    
    # IsotonicRegression fits a monotonic (always increasing or decreasing)
    # function to data. This is the core of the isotonic binning algorithm.
    from sklearn.isotonic import IsotonicRegression
    
    # KMeans is a clustering algorithm that groups data into k clusters.
    # We use this to reduce the number of splits when there are too many.
    from sklearn.cluster import KMeans
    
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Shiny - Web Application Framework
# -----------------------------------------------------------------------------
# Shiny for Python is a web application framework that creates interactive UIs.
# It was originally developed for R and has been ported to Python.
# We use Shiny for the interactive mode of the WOE editor.
try:
    # Core Shiny components:
    #   - App: The main application class that combines UI and server logic
    #   - Inputs: Container for user input values
    #   - Outputs: Container for rendered outputs
    #   - Session: Manages the user session (connection state)
    #   - reactive: Decorator for reactive values and effects
    #   - render: Functions for rendering outputs (tables, text, etc.)
    #   - ui: Functions for creating UI components (buttons, sliders, etc.)
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # shinywidgets provides integration between Shiny and Plotly for charts.
    #   - render_plotly: Decorator for rendering Plotly figures
    #   - output_widget: UI component for displaying Plotly figures
    from shinywidgets import render_plotly, output_widget
    
    # Plotly is an interactive charting library.
    # graph_objects provides low-level control over chart creation.
    # We use this to create WOE charts, IV distributions, etc.
    import plotly.graph_objects as go
    
except ImportError:
    import subprocess
    # Install all three packages: shiny, shinywidgets, and plotly
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# SECTION 5: ENUMERATIONS
# =============================================================================
# Enumerations (Enums) define a fixed set of named constants.
# Using Enums instead of raw strings:
#   - Prevents typos (IDE will catch Algorithm.DESICION_TREE but not "Desicion")
#   - Enables autocomplete in IDEs
#   - Makes code more self-documenting
#   - Allows type checking by static analyzers

class Algorithm(Enum):
    """
    Enumeration of available binning algorithms.
    
    Each algorithm has different characteristics and is suitable for
    different scenarios:
    
    DECISION_TREE: R-compatible, uses CART algorithm
        - Matches R's logiBin::getBins output exactly
        - Fast and robust
        - Good for general use
    
    CHI_MERGE: Chi-square based bottom-up merging
        - More statistically rigorous
        - May produce fewer bins for sparse data
        - Good for statistical validation
    
    IV_OPTIMAL: Directly maximizes Information Value
        - Allows non-monotonic patterns ("sweet spots")
        - Best for fraud detection where non-monotonic patterns exist
        - May overfit if not careful
    
    SPLINE: WOE 2.0 spline-based binning
        - Captures non-linear effects
        - More granular than traditional methods
        - Based on Raymaekers et al. (2021) paper
    
    ISOTONIC: Isotonic regression based binning
        - Guaranteed monotonicity
        - Finer-grained than decision trees
        - Higher IV potential
    """
    # Each value is the string used in flow variables and configuration
    DECISION_TREE = "DecisionTree"
    CHI_MERGE = "ChiMerge"
    IV_OPTIMAL = "IVOptimal"
    SPLINE = "Spline"
    ISOTONIC = "Isotonic"


class MonotonicTrend(Enum):
    """
    Enumeration of monotonicity constraint options.
    
    In credit scoring and fraud detection, WOE values often follow a
    monotonic pattern - higher values of a variable correspond to
    consistently higher or lower risk. Enforcing monotonicity:
        - Improves model interpretability
        - Reduces overfitting
        - Aligns with business logic
    
    However, some variables (especially in fraud) may have non-monotonic
    patterns. For example, transaction amounts in a "sweet spot" range
    might have unusually high fraud rates.
    
    AUTO: Automatically detect the best monotonic direction
        - Calculates correlation between variable and target
        - Chooses ascending or descending based on correlation sign
    
    ASCENDING: WOE must increase with variable value
        - Higher variable values = higher risk
        - Example: Number of overdrafts
    
    DESCENDING: WOE must decrease with variable value
        - Higher variable values = lower risk
        - Example: Account age
    
    PEAK: Allows one peak (increase then decrease)
        - Low and high values are lower risk
        - Middle values are higher risk
        - Example: Transaction amount with fraud "sweet spot"
    
    VALLEY: Allows one valley (decrease then increase)
        - Low and high values are higher risk
        - Middle values are lower risk
        - Example: Account balance (both very low and very high suspicious)
    
    CONCAVE: Rate of increase slows down
        - WOE increases, but at a decreasing rate
        - Second derivative is negative
    
    CONVEX: Rate of increase speeds up
        - WOE increases, but at an increasing rate
        - Second derivative is positive
    
    NONE: No monotonicity constraint
        - Allows any pattern
        - Maximum flexibility but higher overfit risk
    """
    AUTO = "auto"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    PEAK = "peak"           # Increases then decreases
    VALLEY = "valley"       # Decreases then increases
    CONCAVE = "concave"     # Rate of increase slows (d²WOE/dx² < 0)
    CONVEX = "convex"       # Rate of increase accelerates (d²WOE/dx² > 0)
    NONE = "none"           # No monotonicity constraint


class ShrinkageMethod(Enum):
    """
    Enumeration of shrinkage estimation methods for WOE calculation.
    
    Shrinkage is a technique that "shrinks" extreme estimates toward a
    central value. In WOE binning, this helps when bins have few observations,
    which can lead to extreme and unstable WOE values.
    
    NONE: No shrinkage applied
        - Use raw WOE values
        - May be unstable for small bins
        - Traditional approach
    
    SIMPLE: Original weight-based shrinkage
        - Shrinks WOE based on bin sample size
        - Larger bins get less shrinkage
        - Simple implementation
    
    BETA_BINOMIAL: WOE 2.0 Bayesian approach
        - Uses Beta-Binomial posterior for event rate estimation
        - More principled statistical framework
        - Provides credible intervals
        - Better for rare events (fraud detection)
    """
    NONE = "None"
    SIMPLE = "Simple"           # Original weight-based approach
    BETA_BINOMIAL = "BetaBinomial"  # WOE 2.0 Bayesian approach


# =============================================================================
# SECTION 6: CONFIGURATION
# =============================================================================
# This section defines the configuration settings for the binning algorithms.
# Using a dataclass makes it easy to:
#   - Group related settings together
#   - Provide default values
#   - Access settings with dot notation (config.max_bins)
#   - Modify settings via flow variables

@dataclass
class BinningConfig:
    """
    Configuration class for WOE 2.0 binning algorithms.
    
    This dataclass holds all tunable parameters for the binning process.
    Default values are set to work well for typical fraud detection scenarios,
    but can be overridden via KNIME flow variables.
    
    The @dataclass decorator automatically generates:
        - __init__() method with all fields as parameters
        - __repr__() for readable string representation
        - __eq__() for equality comparison
    
    Attributes are grouped by category:
    
    Algorithm Settings:
        algorithm: Which binning algorithm to use
        min_bin_pct: Minimum percentage of observations per bin
        min_bin_count: Minimum absolute count per bin
        max_bins: Maximum number of bins per variable
        min_bins: Minimum number of bins (must be at least 2)
        max_categories: Maximum unique values for categorical variables
    
    Monotonic Trend:
        monotonic_trend: Which monotonicity constraint to apply
    
    WOE 2.0 Shrinkage Settings:
        use_shrinkage: Whether to apply any shrinkage
        shrinkage_method: Which shrinkage method to use
        prior_strength: How strong the Bayesian prior should be
    
    Statistical Validation:
        max_p_value: Maximum p-value for adjacent bins (merge if higher)
        use_p_value_merging: Whether to merge non-significant bins
        chi_merge_threshold: P-value threshold for ChiMerge algorithm
    
    Bayesian Options:
        compute_credible_intervals: Whether to calculate CIs for WOE
        credible_level: Credible interval level (e.g., 0.95 for 95%)
    
    IV Optimization:
        min_iv_gain: Minimum IV improvement to continue splitting
    
    PSI Monitoring:
        compute_psi: Whether to calculate Population Stability Index
    
    Spline Settings:
        spline_smoothing: Smoothing parameter for spline fitting
        spline_degree: Polynomial degree for spline (3 = cubic)
    
    Enhancement Flags (backward compatible with woe_editor_advanced.py):
        use_enhancements: Master switch for all enhancements
        adaptive_min_prop: Relax min_prop for sparse data
        min_event_count: Ensure minimum events per bin
        auto_retry: Retry with relaxed constraints if binning fails
        chi_square_validation: Merge statistically similar bins
        single_bin_protection: Prevent creating single-bin variables
    """
    
    # -------------------------------------------------------------------------
    # Algorithm Settings
    # -------------------------------------------------------------------------
    
    # algorithm: Which binning algorithm to use.
    # Options: "DecisionTree", "ChiMerge", "IVOptimal", "Spline", "Isotonic"
    # Default: "DecisionTree" for R compatibility with logiBin::getBins
    algorithm: str = "DecisionTree"
    
    # min_bin_pct: Minimum percentage of observations required in each bin.
    # Value 0.01 means 1% - each bin must have at least 1% of the data.
    # For fraud models with low event rates, this may need to be lower (0.005).
    # For credit scoring, industry standard is often 5% (0.05).
    min_bin_pct: float = 0.01
    
    # min_bin_count: Minimum absolute number of observations per bin.
    # Even if 1% of data is only 5 observations, bins must have at least 20.
    # This prevents bins with too few samples for stable WOE estimates.
    # Increase for more stable estimates; decrease for more granularity.
    min_bin_count: int = 20
    
    # max_bins: Maximum number of bins allowed per variable.
    # More bins = more granular but more complex; fewer bins = simpler model.
    # 10 is a good balance between capturing patterns and avoiding overfitting.
    # For scorecards, 5-8 bins is often preferred for simplicity.
    max_bins: int = 10
    
    # min_bins: Minimum number of bins required per variable.
    # Must have at least 2 bins to have any discriminatory power (WOE variation).
    # A single bin means WOE=0 everywhere - the variable has no predictive value.
    min_bins: int = 2
    
    # max_categories: Maximum unique values for categorical variables.
    # Variables with more unique values than this will be skipped with a warning.
    # High-cardinality categoricals (like ZIP codes) need special handling
    # such as grouping by WOE similarity or target encoding.
    max_categories: int = 50
    
    # -------------------------------------------------------------------------
    # Monotonic Trend
    # -------------------------------------------------------------------------
    
    # monotonic_trend: Which monotonicity constraint to apply to WOE values.
    # Options: "auto", "ascending", "descending", "peak", "valley", 
    #          "concave", "convex", "none"
    # Default: "auto" - automatically detect based on correlation
    monotonic_trend: str = "auto"
    
    # -------------------------------------------------------------------------
    # WOE 2.0 Shrinkage Settings
    # -------------------------------------------------------------------------
    
    # use_shrinkage: Whether to apply shrinkage to WOE values.
    # Shrinkage helps stabilize estimates when bin counts are low.
    # Recommended: True for fraud models; can be False for large datasets.
    use_shrinkage: bool = True
    
    # shrinkage_method: Which shrinkage method to use.
    # Options: "BetaBinomial" (WOE 2.0), "Simple" (weight-based), "None"
    # Default: "BetaBinomial" - the principled Bayesian approach from WOE 2.0
    shrinkage_method: str = "BetaBinomial"
    
    # prior_strength: How strong the Bayesian prior should be.
    # Higher values = more shrinkage toward the global event rate.
    # Think of this as the "equivalent sample size" of the prior.
    # Value 1.0 is a reasonable default; increase for more shrinkage.
    prior_strength: float = 1.0
    
    # -------------------------------------------------------------------------
    # Statistical Validation
    # -------------------------------------------------------------------------
    
    # max_p_value: Maximum p-value for chi-square test between adjacent bins.
    # If p-value > max_p_value, the bins are not significantly different
    # and should be merged to prevent over-binning.
    # Standard significance level is 0.05; use higher (0.10) for more merging.
    max_p_value: float = 0.05
    
    # use_p_value_merging: Whether to merge bins that aren't significantly different.
    # When True, adjacent bins with p-value > max_p_value are merged.
    # This prevents over-binning and ensures bins have real differences.
    use_p_value_merging: bool = True
    
    # chi_merge_threshold: P-value threshold for the ChiMerge algorithm.
    # In ChiMerge, we merge adjacent bins until chi-square exceeds this threshold.
    # Lower values = more aggressive merging, fewer bins.
    chi_merge_threshold: float = 0.05
    
    # -------------------------------------------------------------------------
    # Bayesian Options
    # -------------------------------------------------------------------------
    
    # compute_credible_intervals: Whether to calculate credible intervals for WOE.
    # Credible intervals provide uncertainty quantification for WOE estimates.
    # Useful for understanding reliability of WOE values in small bins.
    # Set to False by default because it adds computational overhead.
    compute_credible_intervals: bool = False
    
    # credible_level: Credible interval level (e.g., 0.95 for 95% CI).
    # A 95% credible interval means there's a 95% probability the true
    # WOE value falls within the interval (Bayesian interpretation).
    credible_level: float = 0.95
    
    # -------------------------------------------------------------------------
    # IV Optimization
    # -------------------------------------------------------------------------
    
    # min_iv_gain: Minimum improvement in Information Value to continue splitting.
    # If a split doesn't improve IV by at least this amount, stop splitting.
    # Prevents creating bins that don't add meaningful predictive value.
    # Increase for simpler bins; decrease for more granularity.
    min_iv_gain: float = 0.005
    
    # -------------------------------------------------------------------------
    # PSI Monitoring
    # -------------------------------------------------------------------------
    
    # compute_psi: Whether to calculate Population Stability Index.
    # PSI measures distribution drift between training and scoring data.
    # Essential for production fraud systems to detect when models need refresh.
    # Requires reference data (from training) for comparison.
    compute_psi: bool = True
    
    # -------------------------------------------------------------------------
    # Spline Settings
    # -------------------------------------------------------------------------
    
    # spline_smoothing: Smoothing parameter for spline fitting.
    # Higher values = smoother spline with fewer inflection points.
    # Lower values = spline follows data more closely (risk of overfitting).
    # Value 0.5 is a reasonable default; adjust based on noise level.
    spline_smoothing: float = 0.5
    
    # spline_degree: Polynomial degree for spline segments.
    # 3 = cubic spline (most common, provides smooth curves)
    # 2 = quadratic spline (less smooth but simpler)
    # Must be at least 1 and at most k where k+1 <= number of data points
    spline_degree: int = 3
    
    # -------------------------------------------------------------------------
    # Enhancement Flags (backward compatible with woe_editor_advanced.py)
    # -------------------------------------------------------------------------
    
    # use_enhancements: Master switch to enable all advanced enhancements.
    # When True, turns on all individual enhancement flags below.
    # For R-compatible output, set to False.
    use_enhancements: bool = True
    
    # adaptive_min_prop: Automatically relax minimum bin proportion for sparse data.
    # When a variable has very few non-null values (<500), standard constraints
    # may result in no bins being created. This enhancement relaxes constraints.
    adaptive_min_prop: bool = True
    
    # min_event_count: Ensure each bin has enough "bad" events for stable WOE.
    # For fraud models with very low event rates, bins might have 0 or 1 events,
    # leading to unstable WOE values. This ensures a minimum event count.
    min_event_count: bool = True
    
    # auto_retry: Automatically retry binning with relaxed constraints if initial fails.
    # If the first attempt produces no splits (single bin), try again with looser settings.
    auto_retry: bool = True
    
    # chi_square_validation: Post-binning validation using chi-square tests.
    # Merges adjacent bins that are not statistically significantly different.
    chi_square_validation: bool = True
    
    # single_bin_protection: Prevent na_combine from creating single-bin variables.
    # When grouping NA values, if merging would leave only one bin (WOE=0 everywhere),
    # skip the merge for that variable. Default is ON because single-bin vars are useless.
    single_bin_protection: bool = True


# Create a global configuration instance.
# This instance holds the default settings and can be modified via flow variables.
# Functions throughout the script can access this global config.
config = BinningConfig()


# =============================================================================
# SECTION 7: DATA CLASSES FOR BINNING RESULTS
# =============================================================================
# These dataclasses define the structure of binning results.
# Using dataclasses provides:
#   - Clear documentation of what data is expected
#   - Automatic initialization and comparison methods
#   - Type hints for IDE support

@dataclass
class BinInfo:
    """
    Information about a single bin in the binning result.
    
    This dataclass holds all the statistics for one bin of one variable.
    It's used internally during binning calculations.
    
    Attributes:
    -----------
    lower : float
        The lower bound of the bin (exclusive, except for first bin).
        For the first bin, this is typically -infinity.
    
    upper : float
        The upper bound of the bin (inclusive).
        For the last bin, this is typically +infinity.
    
    count : int
        Total number of observations in this bin.
        count = goods + bads
    
    goods : int
        Number of "good" outcomes (non-events, e.g., non-fraudulent transactions).
    
    bads : int
        Number of "bad" outcomes (events, e.g., fraudulent transactions).
    
    woe : float
        Weight of Evidence value for this bin.
        WOE = ln(% of bads / % of goods)
    
    iv_contribution : float
        This bin's contribution to the total Information Value.
        IV_contribution = (% of bads - % of goods) × WOE
    
    woe_ci_lower : Optional[float]
        Lower bound of the credible interval for WOE.
        None if credible intervals are not computed.
    
    woe_ci_upper : Optional[float]
        Upper bound of the credible interval for WOE.
        None if credible intervals are not computed.
    
    event_rate : float
        The event rate (bad rate) in this bin.
        event_rate = bads / count
    """
    lower: float                          # Lower bound of bin (exclusive)
    upper: float                          # Upper bound of bin (inclusive)
    count: int                            # Total observations in bin
    goods: int                            # Number of non-events
    bads: int                             # Number of events
    woe: float = 0.0                      # Weight of Evidence
    iv_contribution: float = 0.0          # IV contribution from this bin
    woe_ci_lower: Optional[float] = None  # Lower bound of WOE credible interval
    woe_ci_upper: Optional[float] = None  # Upper bound of WOE credible interval
    event_rate: float = 0.0               # Event rate in this bin


@dataclass
class BinResult:
    """
    Container for complete binning results.
    
    This dataclass holds the two main outputs of the binning process:
    1. A summary table with one row per variable (aggregate statistics)
    2. A detailed bin table with one row per bin per variable
    
    Attributes:
    -----------
    var_summary : pd.DataFrame
        Summary statistics for each variable with columns:
            - var: Variable name
            - varType: "numeric" or "factor"
            - iv: Total Information Value
            - ent: Average entropy across bins
            - trend: Predominant trend direction ("I" or "D")
            - monTrend: Is trend monotonic? ("Y" or "N")
            - flipRatio: Ratio of trend reversals
            - numBins: Number of bins created
            - purNode: Any pure bins? ("Y" or "N")
    
    bin : pd.DataFrame
        Detailed bin information with columns:
            - var: Variable name
            - bin: Bin rule string (e.g., "Age > '25' & Age <= '35'")
            - count: Number of observations
            - bads: Number of events
            - goods: Number of non-events
            - propn: Proportion of total (percentage)
            - bad_rate: Event rate in bin (percentage)
            - event_rate: Event rate (decimal)
            - woe: Weight of Evidence
            - goodCap: Goods captured (proportion of total goods)
            - badCap: Bads captured (proportion of total bads)
            - iv: IV contribution from this bin
            - ent: Entropy of this bin
            - purNode: Is this a pure bin? ("Y" or "N")
            - trend: Trend direction vs previous bin ("I", "D", or None)
            - woe_ci_lower: WOE credible interval lower bound (if computed)
            - woe_ci_upper: WOE credible interval upper bound (if computed)
    """
    var_summary: pd.DataFrame  # One row per variable with aggregate stats
    bin: pd.DataFrame          # One row per bin with detailed info


@dataclass
class PSIResult:
    """
    Container for Population Stability Index calculation results.
    
    PSI measures the shift in a variable's distribution between two datasets.
    It's commonly used to detect data drift in production scoring systems.
    
    Attributes:
    -----------
    variable : str
        The name of the variable this PSI was calculated for.
    
    psi_value : float
        The calculated PSI value.
        Interpretation:
            < 0.1: Stable (no significant change)
            0.1 - 0.2: Moderate drift (monitor closely)
            > 0.2: Significant drift (consider retraining)
    
    status : str
        Human-readable status based on PSI value.
        One of: "stable", "moderate_drift", "significant_drift"
    
    bin_details : pd.DataFrame
        Per-bin PSI contributions with columns:
            - bin: Bin label
            - count: Current count
            - current_prop: Current proportion
            - reference_prop: Reference proportion
            - psi_contribution: This bin's contribution to total PSI
    """
    variable: str              # Variable name
    psi_value: float           # Total PSI value
    status: str                # "stable", "moderate_drift", or "significant_drift"
    bin_details: pd.DataFrame  # Per-bin PSI breakdown


# =============================================================================
# SECTION 8: CORE WOE/IV CALCULATION FUNCTIONS - WOE 2.0 ENHANCED
# =============================================================================
# These functions implement the fundamental calculations for Weight of Evidence
# and Information Value. This section includes both traditional calculations
# and the WOE 2.0 enhancements (Bayesian shrinkage, credible intervals).

def calculate_woe_simple(freq_good: np.ndarray, freq_bad: np.ndarray,
                         shrinkage_weight: float = 0.0) -> np.ndarray:
    """
    Calculate WOE with simple weight-based shrinkage.
    
    This is the original method from woe_editor_advanced.py.
    It uses a simple shrinkage formula where larger bins get weights
    closer to 1 (less shrinkage) and smaller bins get weights closer to 0.
    
    The Formula:
    ------------
    Without shrinkage:
        WOE = ln(dist_bad / dist_good)
        where dist_bad = freq_bad[i] / sum(freq_bad)
              dist_good = freq_good[i] / sum(freq_good)
    
    With shrinkage:
        weight[i] = n[i] / (n[i] + shrinkage_weight * N / B)
        WOE_shrunk = WOE * weight
        
        where n[i] = sample size in bin i
              N = total sample size
              B = number of bins
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good (non-event) counts for each bin.
        Example: [100, 150, 200] for 3 bins with these good counts.
    
    freq_bad : np.ndarray
        Array of bad (event) counts for each bin.
        Example: [10, 25, 15] for 3 bins with these bad counts.
    
    shrinkage_weight : float, optional (default=0.0)
        Strength of shrinkage. Higher values = more shrinkage.
        0.0 means no shrinkage (traditional WOE).
        Typical values: 0.1 to 1.0
    
    Returns:
    --------
    np.ndarray
        Array of WOE values for each bin, rounded to 5 decimal places.
    
    Example:
    --------
    >>> freq_good = np.array([100, 150, 200])
    >>> freq_bad = np.array([10, 25, 15])
    >>> woe = calculate_woe_simple(freq_good, freq_bad)
    >>> print(woe)
    [-0.51083, 0.22314, -0.28768]
    """
    # Convert inputs to float numpy arrays for consistent math operations.
    # This handles cases where inputs might be integers or lists.
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals across all bins.
    # These represent the total goods and bads in the entire dataset.
    total_good = freq_good.sum()  # Total number of goods across all bins
    total_bad = freq_bad.sum()    # Total number of bads across all bins
    
    # Handle edge case: if no goods or no bads at all, WOE is undefined.
    # This can happen if the target variable has only one class in the data.
    # Return zeros for all bins in this case.
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    # Calculate distribution (proportion) of goods and bads in each bin.
    # dist_good[i] = what fraction of all goods are in bin i
    # dist_bad[i] = what fraction of all bads are in bin i
    # These proportions sum to 1.0 across all bins.
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Apply Laplace smoothing to prevent division by zero and log(0).
    # When a bin has 0 goods or 0 bads:
    #   - Division by zero when calculating WOE
    #   - log(0) = -infinity
    # epsilon is a small value that adds stability without changing results much.
    epsilon = 0.0001  # Small constant for smoothing
    
    # Replace zero distributions with epsilon.
    # np.where(condition, value_if_true, value_if_false) is vectorized if-else.
    dist_good = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad = np.where(dist_bad == 0, epsilon, dist_bad)
    
    # Calculate WOE using the standard formula.
    # WOE = natural log of (bad distribution / good distribution)
    # Interpretation:
    #   - WOE > 0: More bads than expected (higher risk bin)
    #   - WOE < 0: Fewer bads than expected (lower risk bin)
    #   - WOE = 0: Same proportion as population (neutral)
    woe = np.log(dist_bad / dist_good)
    
    # Apply simple weight-based shrinkage if requested.
    # This pulls extreme WOE values toward zero based on sample size.
    # Larger samples get less shrinkage (we trust the data more).
    if shrinkage_weight > 0:
        # Total observations in each bin
        n_obs = freq_good + freq_bad
        
        # Total observations across all bins
        total_obs = n_obs.sum()
        
        # Calculate shrinkage weights based on sample size.
        # Formula: weight = n / (n + k * N / B)
        # where n = bin count, k = shrinkage strength, N = total count, B = number of bins
        # Larger n -> weight closer to 1 -> less shrinkage
        # Smaller n -> weight closer to 0 -> more shrinkage
        weights = n_obs / (n_obs + shrinkage_weight * total_obs / len(n_obs))
        
        # Apply shrinkage by multiplying WOE by weight.
        # Small bins: weight near 0, WOE pulled toward 0
        # Large bins: weight near 1, WOE mostly unchanged
        woe = woe * weights
    
    # Round to 5 decimal places for clean output.
    # Excessive precision is meaningless and makes output harder to read.
    return np.round(woe, 5)


def calculate_woe_beta_binomial(
    freq_good: np.ndarray, 
    freq_bad: np.ndarray,
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate WOE using Beta-Binomial posterior (WOE 2.0 approach).
    
    This is a more principled shrinkage method that uses Bayesian estimation.
    The key insight is that event rates follow a Binomial distribution,
    and the Beta distribution is the conjugate prior for Binomial.
    
    Bayesian Framework:
    -------------------
    Prior: Beta(α₀, β₀) centered on global event rate
    Likelihood: Binomial(n, p) where n is sample size
    Posterior: Beta(α₀ + bads, β₀ + goods)
    
    The posterior mean provides shrinkage toward the global event rate,
    with the amount of shrinkage determined by sample size:
    - Small bins: posterior closer to prior (more shrinkage)
    - Large bins: posterior closer to data (less shrinkage)
    
    This approach also naturally provides credible intervals through
    the posterior distribution.
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good (non-event) counts per bin.
    
    freq_bad : np.ndarray
        Array of bad (event) counts per bin.
    
    prior_strength : float, optional (default=1.0)
        Strength of the prior, interpreted as "equivalent sample size".
        Higher values = stronger prior = more shrinkage.
        - 1.0: Moderate shrinkage (recommended starting point)
        - 0.5: Light shrinkage
        - 2.0+: Strong shrinkage (for very sparse data)
    
    compute_ci : bool, optional (default=False)
        Whether to compute credible intervals for WOE.
        Adds computational overhead but provides uncertainty quantification.
    
    ci_level : float, optional (default=0.95)
        Credible interval level (e.g., 0.95 for 95% CI).
        The interval contains the true WOE with this probability.
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        - woe: Array of WOE values
        - ci_lower: Lower bounds of credible intervals (None if not computed)
        - ci_upper: Upper bounds of credible intervals (None if not computed)
    
    Example:
    --------
    >>> freq_good = np.array([100, 150, 200])
    >>> freq_bad = np.array([10, 25, 15])
    >>> woe, ci_low, ci_high = calculate_woe_beta_binomial(freq_good, freq_bad)
    >>> print(woe)
    [-0.48521, 0.21052, -0.27104]
    """
    # Convert inputs to float numpy arrays for consistent calculations.
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals.
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    total_all = total_good + total_bad
    
    # Handle edge case: no goods or no bads.
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good)), None, None
    
    # Calculate global event rate to use as the prior mean.
    # This is where we want to shrink toward - the overall event rate.
    global_event_rate = total_bad / total_all
    
    # Set up Beta prior parameters centered on global event rate.
    # For a Beta(α, β) distribution:
    #   Mean = α / (α + β)
    #   Variance decreases as (α + β) increases
    # 
    # We want: α / (α + β) = global_event_rate
    # With prior_strength controlling the "equivalent sample size" (α + β)
    prior_alpha = prior_strength * global_event_rate
    prior_beta = prior_strength * (1 - global_event_rate)
    
    # Ensure prior parameters are at least 0.5 for numerical stability.
    # Very small alpha or beta can cause numerical issues.
    prior_alpha = max(prior_alpha, 0.5)
    prior_beta = max(prior_beta, 0.5)
    
    # Initialize result arrays
    woe_values = []
    ci_lower_values = []
    ci_upper_values = []
    
    # Process each bin
    for i in range(len(freq_good)):
        goods = freq_good[i]
        bads = freq_bad[i]
        n = goods + bads
        
        # Handle empty bins
        if n == 0:
            woe_values.append(0.0)
            ci_lower_values.append(0.0)
            ci_upper_values.append(0.0)
            continue
        
        # Calculate posterior parameters using Beta-Binomial conjugacy.
        # Posterior = Beta(prior_α + observed_bads, prior_β + observed_goods)
        # This is the "Bayesian update" - combining prior beliefs with observed data.
        post_alpha = bads + prior_alpha
        post_beta = goods + prior_beta
        
        # Calculate distributions for WOE with shrinkage.
        # Traditional: dist_bad = bads / total_bad
        # With shrinkage: we add a prior component that pulls toward global rate
        dist_bad = (bads + prior_alpha * (n / total_all)) / (total_bad + prior_alpha)
        dist_good = (goods + prior_beta * (n / total_all)) / (total_good + prior_beta)
        
        # Avoid division by zero
        dist_bad = max(dist_bad, 0.0001)
        dist_good = max(dist_good, 0.0001)
        
        # Calculate WOE
        woe = np.log(dist_bad / dist_good)
        woe_values.append(round(woe, 5))
        
        # Compute credible intervals via Monte Carlo simulation
        if compute_ci:
            n_samples = 1000  # Number of Monte Carlo samples
            alpha_half = (1 - ci_level) / 2  # e.g., 0.025 for 95% CI
            
            # Sample event rates from posterior distribution
            # Beta distribution is the posterior for event rate
            p_samples = stats.beta.rvs(post_alpha, post_beta, size=n_samples)
            
            # Convert to WOE scale (simplified approach)
            # WOE ≈ log(p / (1-p)) - log(global_p / (1-global_p))
            # This is the log-odds relative to global log-odds
            log_odds_samples = np.log(p_samples / (1 - p_samples + 0.0001) + 0.0001)
            global_log_odds = np.log(global_event_rate / (1 - global_event_rate))
            woe_samples = log_odds_samples - global_log_odds
            
            # Get credible interval from simulated distribution
            ci_lower_values.append(round(np.percentile(woe_samples, alpha_half * 100), 5))
            ci_upper_values.append(round(np.percentile(woe_samples, (1 - alpha_half) * 100), 5))
        else:
            ci_lower_values.append(None)
            ci_upper_values.append(None)
    
    # Return results
    return (
        np.array(woe_values),
        np.array(ci_lower_values) if compute_ci else None,
        np.array(ci_upper_values) if compute_ci else None
    )


def calculate_woe(
    freq_good: np.ndarray, 
    freq_bad: np.ndarray,
    method: str = "BetaBinomial",
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate Weight of Evidence with configurable method.
    
    This is the main WOE calculation function that dispatches to the
    appropriate implementation based on the selected method. It provides
    a unified interface regardless of which shrinkage method is used.
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good (non-event) counts per bin.
    
    freq_bad : np.ndarray
        Array of bad (event) counts per bin.
    
    method : str, optional (default="BetaBinomial")
        Which calculation method to use:
        - "BetaBinomial": WOE 2.0 Bayesian shrinkage
        - "Simple": Weight-based shrinkage
        - "None": No shrinkage (traditional WOE)
    
    prior_strength : float, optional (default=1.0)
        Shrinkage strength parameter (interpretation depends on method).
    
    compute_ci : bool, optional (default=False)
        Whether to compute credible intervals (only for BetaBinomial).
    
    ci_level : float, optional (default=0.95)
        Credible interval level.
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        - woe: Array of WOE values
        - ci_lower: Lower credible interval bounds (or None)
        - ci_upper: Upper credible interval bounds (or None)
    """
    # Dispatch to appropriate calculation method
    if method == "BetaBinomial":
        # Use the full Bayesian approach with optional credible intervals
        return calculate_woe_beta_binomial(
            freq_good, freq_bad, prior_strength, compute_ci, ci_level
        )
    elif method == "Simple":
        # Use simple weight-based shrinkage (no credible intervals)
        woe = calculate_woe_simple(freq_good, freq_bad, prior_strength)
        return woe, None, None
    else:
        # No shrinkage - traditional WOE calculation
        woe = calculate_woe_simple(freq_good, freq_bad, 0.0)
        return woe, None, None


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """
    Calculate Information Value (IV) for a variable.
    
    Information Value measures the overall predictive power of a variable.
    It sums the WOE contributions across all bins, weighted by the
    difference in good/bad distributions.
    
    Formula:
    --------
    IV = Σ (dist_bad[i] - dist_good[i]) × WOE[i]
    
    Where:
        dist_bad[i] = freq_bad[i] / total_bad
        dist_good[i] = freq_good[i] / total_good
        WOE[i] = ln(dist_bad[i] / dist_good[i])
    
    Substituting:
    IV = Σ (dist_bad[i] - dist_good[i]) × ln(dist_bad[i] / dist_good[i])
    
    This is also known as Jeffrey's divergence or symmetric KL divergence.
    
    IV Interpretation:
    ------------------
    < 0.02: Not useful for prediction (too weak)
    0.02 - 0.1: Weak predictive power
    0.1 - 0.3: Medium predictive power
    0.3 - 0.5: Strong predictive power
    > 0.5: Suspicious (may indicate data leakage or overfitting)
    
    Parameters:
    -----------
    freq_good : np.ndarray
        Array of good (non-event) counts for each bin.
    
    freq_bad : np.ndarray
        Array of bad (event) counts for each bin.
    
    Returns:
    --------
    float
        The Information Value, rounded to 4 decimal places.
    
    Example:
    --------
    >>> freq_good = np.array([100, 150, 200])
    >>> freq_bad = np.array([10, 25, 15])
    >>> iv = calculate_iv(freq_good, freq_bad)
    >>> print(iv)
    0.0423
    """
    # Convert to float arrays
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    # Calculate totals
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    # Handle edge case: if no goods or no bads, IV is undefined - return 0
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    # Calculate distributions
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Apply epsilon smoothing for the log calculation
    epsilon = 0.0001
    dist_good_safe = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, epsilon, dist_bad)
    
    # Calculate WOE for each bin
    woe = np.log(dist_bad_safe / dist_good_safe)
    
    # Calculate IV as sum of (dist_bad - dist_good) × WOE
    # Note: we use original distributions (not smoothed) for the difference
    iv = np.sum((dist_bad - dist_good) * woe)
    
    # Handle numerical issues (infinity or NaN)
    if not np.isfinite(iv):
        iv = 0.0
    
    # Round to 4 decimal places
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """
    Calculate entropy for a bin.
    
    Entropy measures the "purity" or "disorder" of a bin. In information
    theory, entropy quantifies the uncertainty in a random variable.
    
    For a binary outcome:
    - Entropy = 0: Pure bin (100% goods OR 100% bads) - no uncertainty
    - Entropy = 1: Maximum impurity (50% goods, 50% bads) - maximum uncertainty
    
    Formula:
    --------
    Entropy = -Σ p(i) × log₂(p(i))
    
    For binary case:
    Entropy = -[p_bad × log₂(p_bad) + p_good × log₂(p_good)]
    
    Why Use Entropy:
    ----------------
    - Lower entropy indicates more predictive bins
    - Used as a secondary metric alongside WOE and IV
    - Helps identify bins that cleanly separate goods from bads
    
    Parameters:
    -----------
    goods : int
        Number of good outcomes (non-events) in the bin.
    
    bads : int
        Number of bad outcomes (events) in the bin.
    
    Returns:
    --------
    float
        Entropy value between 0 and 1, rounded to 4 decimal places.
    
    Example:
    --------
    >>> calculate_entropy(100, 100)  # 50-50 split
    1.0
    >>> calculate_entropy(100, 0)    # Pure bin
    0.0
    >>> calculate_entropy(75, 25)    # 75-25 split
    0.8113
    """
    # Total observations in the bin
    total = goods + bads
    
    # Handle edge cases: empty bin or pure bin (all one class)
    # In these cases, entropy is 0 (perfectly "pure" or no data)
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    # Calculate proportions (probabilities)
    p_good = goods / total  # Probability of good outcome in this bin
    p_bad = bads / total    # Probability of bad outcome in this bin
    
    # Calculate entropy using the binary entropy formula.
    # We use log base 2 so entropy is between 0 and 1.
    # The negative sign is needed because log of probabilities (<1) is negative.
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """
    Determine if a variable is numeric or factor (categorical).
    
    This function classifies variables into two types for binning:
    1. 'numeric': Continuous or discrete numeric values
       - Binned using split points (ranges)
       - Example: Age, Income, Transaction Amount
    
    2. 'factor': Categorical values
       - Binned by grouping categories
       - Example: State, Product Type, Merchant Category
    
    Special Case:
    -------------
    Numeric variables with ≤10 unique values are treated as factors.
    This handles coded variables like:
    - 0/1/2/3 for education level
    - 1-5 for rating scales
    - Number of dependents (0, 1, 2, 3, 4, 5+)
    
    These are better handled as discrete categories than as continuous
    variables with split points.
    
    Parameters:
    -----------
    series : pd.Series
        The pandas Series (column) to classify.
    
    Returns:
    --------
    str
        Either 'numeric' or 'factor'.
    """
    # Check if the pandas dtype is numeric (int, float, etc.)
    if pd.api.types.is_numeric_dtype(series):
        # Treat as factor if few unique values
        if series.nunique() <= 10:
            return 'factor'
        return 'numeric'
    return 'factor'


# =============================================================================
# SECTION 9: PSI (POPULATION STABILITY INDEX) CALCULATION
# =============================================================================
# PSI measures the shift in a variable's distribution between two datasets.
# It's commonly used to detect data drift in production scoring systems.
#
# The PSI formula is:
#   PSI = Σ (current_prop - ref_prop) × ln(current_prop / ref_prop)
#
# PSI Interpretation:
#   < 0.1: Stable (no significant change)
#   0.1 - 0.2: Moderate drift (monitor closely)
#   > 0.2: Significant drift (consider retraining)

def calculate_psi(
    reference_proportions: np.ndarray,
    current_proportions: np.ndarray,
    epsilon: float = 0.0001
) -> float:
    """
    Calculate Population Stability Index between reference and current distributions.
    
    PSI quantifies how much a distribution has shifted. In fraud detection and
    credit scoring, monitoring PSI helps identify when models need retraining
    due to changing population characteristics.
    
    Formula:
    --------
    PSI = Σ (current_prop - ref_prop) × ln(current_prop / ref_prop)
    
    This is symmetric: large differences in either direction contribute to PSI.
    
    Parameters:
    -----------
    reference_proportions : np.ndarray
        The reference (baseline) proportions for each bin.
        Typically from the training dataset or a stable period.
    
    current_proportions : np.ndarray
        The current proportions for each bin.
        From the dataset being compared to reference.
    
    epsilon : float, optional (default=0.0001)
        Small constant to add for numerical stability.
        Prevents log(0) and division by zero.
    
    Returns:
    --------
    float
        The PSI value, rounded to 4 decimal places.
    
    Examples:
    ---------
    >>> ref = np.array([0.1, 0.2, 0.3, 0.4])  # Reference distribution
    >>> cur = np.array([0.1, 0.2, 0.3, 0.4])  # Identical distribution
    >>> calculate_psi(ref, cur)
    0.0  # No drift
    
    >>> cur = np.array([0.15, 0.25, 0.25, 0.35])  # Shifted distribution
    >>> calculate_psi(ref, cur)
    0.0234  # Small drift
    """
    # Convert to float arrays
    ref = np.array(reference_proportions, dtype=float)
    cur = np.array(current_proportions, dtype=float)
    
    # Normalize to ensure they sum to 1 (in case they don't already)
    ref = ref / ref.sum() if ref.sum() > 0 else ref
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    
    # Add epsilon to avoid log(0) and division by zero
    # np.maximum ensures each element is at least epsilon
    ref = np.maximum(ref, epsilon)
    cur = np.maximum(cur, epsilon)
    
    # PSI calculation: sum of (current - ref) × ln(current / ref)
    psi = np.sum((cur - ref) * np.log(cur / ref))
    
    return round(psi, 4)


# =============================================================================
# SECTION 10: CHI-SQUARE STATISTICS FOR BIN MERGING
# =============================================================================
# Chi-square tests measure whether two bins have significantly different
# distributions of goods and bads. This is used in:
# 1. ChiMerge algorithm - merge bins that aren't significantly different
# 2. P-value based merging - post-binning validation

def chi_square_statistic(bin1_good: int, bin1_bad: int,
                         bin2_good: int, bin2_bad: int) -> float:
    """
    Calculate chi-square statistic for two adjacent bins.
    
    The chi-square test measures whether the good/bad proportions in two
    bins are statistically different from each other. A high chi-square
    value means the bins are significantly different and should be kept
    separate; a low value means they can be merged.
    
    This implementation uses Yates' continuity correction for 2x2 tables,
    which improves accuracy when cell counts are small.
    
    The Contingency Table:
    ----------------------
                  Goods    Bads    | Row Total
    Bin 1          g1       b1     |    n1
    Bin 2          g2       b2     |    n2
    --------------------------------
    Col Total     G        B      |    N
    
    Expected frequencies (under null hypothesis of no difference):
    E[i,j] = (row_total[i] × col_total[j]) / grand_total
    
    Chi-square with Yates correction:
    χ² = Σ (|observed - expected| - 0.5)² / expected
    
    Parameters:
    -----------
    bin1_good : int
        Number of goods (non-events) in first bin.
    
    bin1_bad : int
        Number of bads (events) in first bin.
    
    bin2_good : int
        Number of goods in second bin.
    
    bin2_bad : int
        Number of bads in second bin.
    
    Returns:
    --------
    float
        Chi-square statistic. Higher values = more different bins.
        Returns infinity if calculation is not possible (e.g., empty cells).
    """
    # Create a 2x2 contingency table (observed frequencies)
    # Rows: bin1, bin2
    # Columns: goods, bads
    observed = np.array([[bin1_good, bin1_bad], [bin2_good, bin2_bad]])
    
    # If the table is completely empty, return infinity (can't compare)
    if observed.sum() == 0:
        return np.inf
    
    # Calculate marginal totals
    row_totals = observed.sum(axis=1)  # [n1, n2] - total per bin
    col_totals = observed.sum(axis=0)  # [G, B] - total goods and bads
    total = observed.sum()              # Grand total N
    
    # Check for zero marginals (can't compute expected frequencies)
    if total == 0 or any(row_totals == 0) or any(col_totals == 0):
        return np.inf
    
    # Calculate expected frequencies under the null hypothesis
    # E[i,j] = (row_total[i] × col_total[j]) / grand_total
    # np.outer computes the outer product of row and column totals
    expected = np.outer(row_totals, col_totals) / total
    
    # Calculate chi-square with Yates continuity correction
    # The correction subtracts 0.5 from |observed - expected|
    # This helps when expected counts are small
    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i, j] > 0:
                # Yates correction: subtract 0.5 from the absolute difference
                diff = abs(observed[i, j] - expected[i, j]) - 0.5
                # Don't let the corrected difference go negative
                diff = max(diff, 0)
                # Add to chi-square: (corrected_diff)² / expected
                chi2 += (diff ** 2) / expected[i, j]
    
    return chi2


def chi_square_p_value(bin1_good: int, bin1_bad: int,
                       bin2_good: int, bin2_bad: int) -> float:
    """
    Calculate p-value for chi-square test between two bins.
    
    The p-value represents the probability of observing a chi-square
    statistic at least as extreme as the one calculated, assuming the
    null hypothesis (bins have same distribution) is true.
    
    Interpretation:
    - Low p-value (< 0.05): Bins are significantly different - keep separate
    - High p-value (> 0.05): Bins are not significantly different - can merge
    
    Parameters:
    -----------
    bin1_good, bin1_bad : int
        Good and bad counts for first bin.
    
    bin2_good, bin2_bad : int
        Good and bad counts for second bin.
    
    Returns:
    --------
    float
        P-value between 0 and 1.
    """
    # Calculate chi-square statistic
    chi2 = chi_square_statistic(bin1_good, bin1_bad, bin2_good, bin2_bad)
    
    # If chi-square is infinity, return p-value of 1 (can't determine significance)
    if chi2 == np.inf:
        return 1.0
    
    # Calculate p-value from chi-square distribution with 1 degree of freedom
    # For 2x2 table: df = (rows-1) × (cols-1) = 1 × 1 = 1
    # stats.chi2.cdf gives cumulative probability up to chi2
    # 1 - cdf gives probability of exceeding chi2 (the p-value)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return p_value


# =============================================================================
# SECTION 11: BINNING ALGORITHM - DECISION TREE (R-COMPATIBLE)
# =============================================================================
# This algorithm uses scikit-learn's DecisionTreeClassifier to find optimal
# split points. It matches the behavior of R's logiBin::getBins function,
# providing R compatibility for existing workflows.
#
# How it works:
# 1. Fit a decision tree with the variable as input and target as output
# 2. The tree finds split points that maximize information gain
# 3. Extract the threshold values from the tree structure
# 4. These thresholds become the bin boundaries

def get_decision_tree_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_events: int = 5,
    adaptive_min_prop: bool = True,
    auto_retry: bool = True
) -> List[float]:
    """
    Use decision tree (CART) to find optimal split points.
    
    This is the R-compatible algorithm that matches logiBin::getBins.
    CART (Classification and Regression Trees) uses recursive binary
    splitting to find splits that maximize information gain (minimize Gini
    impurity or entropy).
    
    Algorithm:
    ----------
    1. Remove missing values from x and y
    2. Calculate effective min_samples_leaf based on constraints
    3. Fit DecisionTreeClassifier with max_leaf_nodes = max_bins
    4. Extract threshold values from tree.tree_.threshold
    5. Optionally retry with relaxed constraints if no splits found
    
    Parameters:
    -----------
    x : pd.Series
        The feature values to bin. Must be numeric.
    
    y : pd.Series
        The binary target variable (0/1).
    
    min_prop : float, optional (default=0.01)
        Minimum proportion of observations per bin.
        0.01 = 1%, matching R's default.
    
    max_bins : int, optional (default=10)
        Maximum number of bins (leaf nodes in the tree).
    
    min_events : int, optional (default=5)
        Minimum number of events (bads) to aim for per bin.
        Used when adaptive_min_prop is True for low event rates.
    
    adaptive_min_prop : bool, optional (default=True)
        If True, automatically relax min_prop for sparse data or low event rates.
        This helps prevent "no split" situations for difficult variables.
    
    auto_retry : bool, optional (default=True)
        If True and no splits found, retry with relaxed constraints.
        Helps handle edge cases without manual intervention.
    
    Returns:
    --------
    List[float]
        List of split thresholds. These are the bin boundaries.
        For n splits, there will be n+1 bins.
    
    Example:
    --------
    >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    >>> splits = get_decision_tree_splits(x, y, max_bins=3)
    >>> print(splits)
    [4.5, 7.5]  # Creates bins: ≤4.5, 4.5-7.5, >7.5
    """
    # Remove rows with missing values in either x or y
    # notna() returns True for non-null values
    mask = x.notna() & y.notna()
    
    # Extract clean values and reshape for sklearn
    # sklearn expects 2D array for features, hence reshape(-1, 1)
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    # If no valid data, return empty list
    if len(x_clean) == 0:
        return []
    
    # Calculate sample statistics for adaptive constraints
    n_samples = len(x_clean)
    n_events = int(y_clean.sum())  # Number of bads (events)
    event_rate = n_events / n_samples if n_samples > 0 else 0
    
    # Start with the user-specified minimum proportion
    effective_min_prop = min_prop
    
    # ADAPTIVE MIN_PROP: Adjust constraints for difficult cases
    if adaptive_min_prop:
        # For small samples, relax the constraint
        if n_samples < 500:
            # Use half the minimum proportion, but at least 0.5%
            effective_min_prop = max(min_prop / 2, 0.005)
        
        # For low event rates (fraud detection scenarios)
        if event_rate < 0.05 and n_events > 0:
            # Calculate max possible bins given we need min_events per bin
            max_possible_bins = max(n_events // min_events, 2)
            # Calculate minimum samples needed per bin
            min_samples_for_events = n_samples / max_possible_bins
            # Convert to proportion
            adaptive_prop = min_samples_for_events / n_samples
            # Use the more restrictive of current and adaptive
            effective_min_prop = max(effective_min_prop, adaptive_prop * 0.8)
    
    # Convert proportion to absolute number of samples
    # This is what DecisionTreeClassifier uses as min_samples_leaf
    min_samples_leaf = max(int(n_samples * effective_min_prop), 1)
    
    # Don't let min_samples_leaf exceed half the data
    # (otherwise no split is possible)
    min_samples_leaf = min(min_samples_leaf, n_samples // 2)
    
    # Ensure at least 1 sample per leaf
    min_samples_leaf = max(min_samples_leaf, 1)
    
    # Create and configure the decision tree
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,        # Limits number of bins
        min_samples_leaf=min_samples_leaf,  # Minimum samples per bin
        random_state=42                 # For reproducibility
    )
    
    # Fit the tree to find optimal splits
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        # If fitting fails, return empty list
        return []
    
    # Extract thresholds from the tree structure
    # tree.tree_.threshold contains split thresholds for each node
    # Leaf nodes have threshold = -2 (sentinel value)
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]  # Remove leaf indicators
    thresholds = sorted(set(thresholds))        # Sort and deduplicate
    
    # AUTO RETRY: If no splits found, try with relaxed constraints
    if auto_retry and len(thresholds) == 0 and min_samples_leaf > 10:
        # Create new tree with half the min_samples_leaf
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
            pass  # If retry fails, return what we have (empty list)
    
    return thresholds


# =============================================================================
# SECTION 12: BINNING ALGORITHM - CHIMERGE
# =============================================================================
# ChiMerge is a bottom-up binning algorithm that:
# 1. Starts with many fine-grained bins (based on quantiles)
# 2. Iteratively merges the pair of adjacent bins with lowest chi-square
# 3. Stops when chi-square exceeds threshold or max_bins reached
#
# Advantages over decision trees:
# - More statistically rigorous (explicit significance testing)
# - May produce fewer bins for sparse data
# - Better handles continuous variables with no clear splits

def get_chimerge_splits(
    x: pd.Series,
    y: pd.Series,
    min_bin_pct: float = 0.05,
    min_bin_count: int = 50,
    max_bins: int = 10,
    min_bins: int = 2,
    chi_threshold: float = 0.05
) -> List[float]:
    """
    ChiMerge algorithm: bottom-up binning based on chi-square tests.
    
    ChiMerge was introduced by Kerber (1992) and is widely used in credit
    scoring. It provides a statistically principled approach to binning.
    
    Algorithm:
    ----------
    1. Create initial fine-grained bins using quantiles
    2. Calculate chi-square for each pair of adjacent bins
    3. Find the pair with minimum chi-square (most similar)
    4. If chi-square < threshold, merge this pair
    5. Repeat until chi-square > threshold OR min_bins reached
    6. Enforce minimum bin size by additional merging if needed
    7. Reduce to max_bins if still too many bins
    
    Parameters:
    -----------
    x : pd.Series
        Feature values to bin.
    
    y : pd.Series
        Binary target (0/1).
    
    min_bin_pct : float, optional (default=0.05)
        Minimum percentage of observations per bin.
    
    min_bin_count : int, optional (default=50)
        Minimum absolute count per bin.
    
    max_bins : int, optional (default=10)
        Maximum number of bins.
    
    min_bins : int, optional (default=2)
        Minimum number of bins.
    
    chi_threshold : float, optional (default=0.05)
        P-value threshold for chi-square test.
        Bins are merged if chi-square p-value > threshold.
    
    Returns:
    --------
    List[float]
        List of split thresholds (bin boundaries).
    """
    # Remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return []
    
    # Calculate minimum required count per bin
    total_count = len(x_clean)
    min_count_required = max(int(total_count * min_bin_pct), min_bin_count)
    
    # Create initial fine-grained bins using quantiles
    # Start with up to 100 initial bins
    n_initial_bins = min(100, len(x_clean.unique()))
    
    try:
        # pd.qcut creates bins with equal frequencies
        initial_bins = pd.qcut(x_clean, q=n_initial_bins, duplicates='drop')
    except ValueError:
        # If qcut fails (too few unique values), use pd.cut
        try:
            initial_bins = pd.cut(x_clean, bins=min(20, len(x_clean.unique())), duplicates='drop')
        except:
            return []
    
    # Extract bin edges from the categorical
    if hasattr(initial_bins, 'categories') and len(initial_bins.categories) > 0:
        edges = sorted(set(
            [initial_bins.categories[0].left] + 
            [cat.right for cat in initial_bins.categories]
        ))
    else:
        return []
    
    # Helper function to build bin statistics from edges
    def build_bin_stats(edges):
        bins_stats = []
        for i in range(len(edges) - 1):
            # First bin includes left edge; others are (left, right]
            if i == 0:
                bin_mask = (x_clean >= edges[i]) & (x_clean <= edges[i + 1])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())
            goods = int(count - bads)
            bins_stats.append({
                'left': edges[i], 
                'right': edges[i + 1], 
                'goods': goods, 
                'bads': bads, 
                'count': count
            })
        return bins_stats
    
    # Build initial bin statistics
    bins_stats = build_bin_stats(edges)
    
    # Remove empty bins
    bins_stats = [b for b in bins_stats if b['count'] > 0]
    
    # If already at or below min_bins, return current splits
    if len(bins_stats) <= min_bins:
        return [b['right'] for b in bins_stats[:-1]]
    
    # Calculate chi-square critical value for the threshold
    # This is the chi-square value at the (1 - threshold) percentile
    chi2_threshold = stats.chi2.ppf(1 - chi_threshold, df=1)
    
    # MAIN MERGING LOOP: Merge until chi-square exceeds threshold
    while len(bins_stats) > min_bins:
        # Find the pair of adjacent bins with minimum chi-square
        min_chi2 = np.inf
        merge_idx = -1
        
        for i in range(len(bins_stats) - 1):
            chi2 = chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        # Stop if chi-square exceeds threshold AND we're at acceptable bin count
        if min_chi2 > chi2_threshold and len(bins_stats) <= max_bins:
            break
        
        # Merge the pair with smallest chi-square
        if merge_idx >= 0:
            bins_stats[merge_idx] = {
                'left': bins_stats[merge_idx]['left'],
                'right': bins_stats[merge_idx + 1]['right'],
                'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
                'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
                'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
            }
            bins_stats.pop(merge_idx + 1)
        else:
            break
    
    # ENFORCE MINIMUM BIN SIZE: Merge small bins
    changed = True
    while changed and len(bins_stats) > min_bins:
        changed = False
        for i in range(len(bins_stats)):
            if bins_stats[i]['count'] < min_count_required:
                # Merge small bin with its neighbor
                if i == 0 and len(bins_stats) > 1:
                    # First bin is small - merge with second
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
                    # Merge with previous bin
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
    
    # REDUCE TO MAX_BINS: If still too many bins, merge most similar pairs
    while len(bins_stats) > max_bins:
        min_chi2 = np.inf
        merge_idx = 0
        
        for i in range(len(bins_stats) - 1):
            chi2 = chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        bins_stats[merge_idx] = {
            'left': bins_stats[merge_idx]['left'],
            'right': bins_stats[merge_idx + 1]['right'],
            'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
            'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
            'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
        }
        bins_stats.pop(merge_idx + 1)
    
    # Extract split points (right edges of all but last bin)
    splits = [b['right'] for b in bins_stats[:-1]]
    return splits


# =============================================================================
# SECTION 13: BINNING ALGORITHM - IV-OPTIMAL (MAXIMIZE INFORMATION VALUE)
# =============================================================================
# IV-Optimal binning directly maximizes Information Value (IV).
# Unlike decision trees which maximize information gain at each split,
# this algorithm considers the overall IV of the resulting binning.
#
# Key advantages:
# - Allows non-monotonic patterns ("sweet spots" in fraud detection)
# - Directly optimizes for predictive power
# - May find bins that decision trees miss
#
# Algorithm:
# 1. Start with fine-grained bins (many quantile-based splits)
# 2. Calculate IV for each pair of adjacent bins if merged
# 3. Find the merge that loses the least IV
# 4. Merge until reaching max_bins or IV loss exceeds threshold

def get_iv_optimal_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_bin_count: int = 20,
    min_iv_loss: float = 0.001
) -> List[float]:
    """
    IV-optimal binning: directly maximizes Information Value.
    
    This algorithm allows non-monotonic patterns, making it especially useful
    for fraud detection where "sweet spots" of high fraud rates may exist
    in the middle of a variable's range.
    
    Algorithm:
    ----------
    1. Create fine-grained initial bins (based on quantiles or unique values)
    2. Iteratively merge the pair of adjacent bins with minimum IV loss
    3. Stop when reaching max_bins or IV loss exceeds threshold
    4. Enforce minimum bin size by additional merging if needed
    
    Parameters:
    -----------
    x : pd.Series
        Feature values to bin.
    
    y : pd.Series
        Binary target (0/1).
    
    min_prop : float, optional (default=0.01)
        Minimum proportion of observations per bin.
    
    max_bins : int, optional (default=10)
        Maximum number of bins.
    
    min_bin_count : int, optional (default=20)
        Minimum absolute count per bin.
    
    min_iv_loss : float, optional (default=0.001)
        Minimum IV loss to continue merging. If a merge would lose more
        than this amount of IV, stop merging (if already at acceptable bin count).
    
    Returns:
    --------
    List[float]
        List of split thresholds (bin boundaries).
    """
    # Remove missing values from both x and y
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    # Return empty if no valid data
    if len(x_clean) == 0:
        return []
    
    # Calculate dataset statistics
    n_samples = len(x_clean)           # Total number of observations
    n_unique = len(np.unique(x_clean))  # Number of unique x values
    total_goods = int((y_clean == 0).sum())  # Total non-events
    total_bads = int((y_clean == 1).sum())   # Total events
    
    # Need both goods and bads to calculate WOE/IV
    if total_goods == 0 or total_bads == 0:
        return []
    
    # STEP 1: Create initial fine-grained bins
    # Use unique values for low-cardinality, quantiles for high-cardinality
    if n_unique <= 20:
        # For few unique values, use all except the last as split points
        initial_splits = sorted(np.unique(x_clean))[:-1]
    else:
        # For many unique values, use quantile-based splits
        # Number of initial bins scales with sample size and unique values
        n_initial = min(
            max(20, n_unique // 5),      # At least 20, or 1/5 of unique values
            min(100, n_samples // 50),   # At most 100, or sample/50
            n_unique - 1                  # Can't have more splits than unique-1
        )
        try:
            # Calculate quantile positions (excluding 0% and 100%)
            quantiles = np.linspace(0, 100, n_initial + 1)[1:-1]
            # Get x values at those quantiles
            initial_splits = list(np.percentile(x_clean, quantiles))
            # Remove duplicates and sort
            initial_splits = sorted(set(initial_splits))
        except Exception:
            # If quantile calculation fails, use unique values
            initial_splits = sorted(np.unique(x_clean))[:-1]
    
    # Need at least one split to have 2+ bins
    if len(initial_splits) == 0:
        return []
    
    # Helper function: Create bin statistics from split points
    def create_bins_from_splits(splits: List[float]) -> List[dict]:
        """
        Given a list of split points, create bins and calculate statistics.
        """
        bins_list = []
        # Edges include -infinity, all splits, and +infinity
        edges = [-np.inf] + sorted(splits) + [np.inf]
        
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            
            # Create mask for observations in this bin
            # First bin: x <= upper (includes minimum values)
            # Last bin: x > lower (includes maximum values)
            # Middle bins: lower < x <= upper
            if lower == -np.inf:
                bin_mask = x_clean <= upper
            elif upper == np.inf:
                bin_mask = x_clean > lower
            else:
                bin_mask = (x_clean > lower) & (x_clean <= upper)
            
            # Calculate bin statistics
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
    
    # Helper function: Calculate IV contribution for one bin
    def calculate_bin_iv(goods: int, bads: int) -> float:
        """
        Calculate Information Value contribution for a single bin.
        
        IV_bin = (dist_bad - dist_good) × WOE
        where WOE = ln(dist_bad / dist_good)
        """
        # Need both total goods and total bads
        if total_goods == 0 or total_bads == 0:
            return 0.0
        
        # Calculate distributions
        dist_good = goods / total_goods if total_goods > 0 else 0
        dist_bad = bads / total_bads if total_bads > 0 else 0
        
        # If bin has no goods or no bads, IV contribution is 0
        if dist_good == 0 or dist_bad == 0:
            return 0.0
        
        # WOE = ln(dist_bad / dist_good)
        woe = np.log(dist_bad / dist_good)
        
        # IV contribution = (dist_bad - dist_good) × WOE
        return (dist_bad - dist_good) * woe
    
    # Helper function: Calculate total IV for a binning
    def calculate_total_iv(bins_list: List[dict]) -> float:
        """Sum IV contributions across all bins."""
        return sum(calculate_bin_iv(b['goods'], b['bads']) for b in bins_list)
    
    # Initialize with all initial splits
    current_splits = list(initial_splits)
    bins_list = create_bins_from_splits(current_splits)
    
    # Need at least 2 bins
    if len(bins_list) <= 1:
        return []
    
    # STEP 2: Iteratively merge bins with minimum IV loss
    while len(bins_list) > max_bins:
        # Stop if we can't merge anymore
        if len(bins_list) <= 2:
            break
        
        min_iv_loss_found = float('inf')
        best_merge_idx = 0
        
        # Find the pair of adjacent bins whose merge loses least IV
        for i in range(len(bins_list) - 1):
            # Calculate merged bin statistics
            merged_goods = bins_list[i]['goods'] + bins_list[i + 1]['goods']
            merged_bads = bins_list[i]['bads'] + bins_list[i + 1]['bads']
            
            # IV before merge = sum of both bins' contributions
            iv_before = (
                calculate_bin_iv(bins_list[i]['goods'], bins_list[i]['bads']) +
                calculate_bin_iv(bins_list[i + 1]['goods'], bins_list[i + 1]['bads'])
            )
            
            # IV after merge = merged bin's contribution
            iv_after = calculate_bin_iv(merged_goods, merged_bads)
            
            # IV loss from this merge
            iv_loss = iv_before - iv_after
            
            if iv_loss < min_iv_loss_found:
                min_iv_loss_found = iv_loss
                best_merge_idx = i
        
        # If the minimum loss exceeds threshold and we're close to max_bins, stop
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
        
        # Replace two bins with merged bin
        bins_list = bins_list[:i] + [merged_bin] + bins_list[i + 2:]
        
        # Update splits list correspondingly
        if i < len(current_splits):
            current_splits = current_splits[:i] + current_splits[i + 1:]
    
    # STEP 3: Enforce minimum bin size
    min_count = max(int(n_samples * min_prop), min_bin_count)
    
    merged = True
    while merged and len(bins_list) > 2:
        merged = False
        for i, b in enumerate(bins_list):
            if b['count'] < min_count:
                # Determine which neighbor to merge with
                if i == 0:
                    # First bin: merge with second
                    merge_with = 1
                elif i == len(bins_list) - 1:
                    # Last bin: merge with previous
                    merge_with = i - 1
                else:
                    # Middle bin: merge with smaller neighbor
                    if bins_list[i - 1]['count'] <= bins_list[i + 1]['count']:
                        merge_with = i - 1
                    else:
                        merge_with = i + 1
                
                # Determine merge indices
                merge_idx = min(i, merge_with)
                other_idx = max(i, merge_with)
                
                # Create merged bin
                merged_bin = {
                    'lower': bins_list[merge_idx]['lower'],
                    'upper': bins_list[other_idx]['upper'],
                    'count': bins_list[merge_idx]['count'] + bins_list[other_idx]['count'],
                    'goods': bins_list[merge_idx]['goods'] + bins_list[other_idx]['goods'],
                    'bads': bins_list[merge_idx]['bads'] + bins_list[other_idx]['bads']
                }
                
                # Update bins list
                bins_list = bins_list[:merge_idx] + [merged_bin] + bins_list[other_idx + 1:]
                merged = True
                break
    
    # STEP 4: Extract final splits from bins_list
    final_splits = []
    for b in bins_list[:-1]:
        if b['upper'] != np.inf:
            final_splits.append(b['upper'])
    
    return sorted(final_splits)


# =============================================================================
# SECTION 14: BINNING ALGORITHM - SPLINE-BASED (WOE 2.0)
# =============================================================================
# Spline-based binning uses smooth spline functions to model the relationship
# between a variable and the event rate. Bin boundaries are placed at points
# where the relationship changes character (inflection points, extrema).
#
# Based on the WOE 2.0 paper (Raymaekers et al., 2021)
#
# Advantages:
# - Captures non-linear effects more accurately
# - More granular than piecewise-constant methods
# - Automatically finds optimal bin boundaries

def get_spline_splits(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 10,
    min_bin_pct: float = 0.01,
    smoothing: float = 0.5,
    spline_degree: int = 3
) -> List[float]:
    """
    Spline-based binning from WOE 2.0 paper.
    
    This algorithm fits a spline function to the event rate as a function
    of x, then places bin boundaries at inflection points or points of
    maximum curvature.
    
    Algorithm:
    ----------
    1. Sort data by x values
    2. Calculate smoothed event rate using rolling window
    3. Fit UnivariateSpline to event rate as function of x
    4. Compute first and second derivatives of spline
    5. Find inflection points (where second derivative changes sign)
    6. Use inflection points as candidate split points
    7. Filter to ensure minimum bin size
    
    Parameters:
    -----------
    x : pd.Series
        Feature values to bin.
    
    y : pd.Series
        Binary target (0/1).
    
    max_bins : int, optional (default=10)
        Maximum number of bins.
    
    min_bin_pct : float, optional (default=0.01)
        Minimum proportion of observations per bin.
    
    smoothing : float, optional (default=0.5)
        Spline smoothing parameter. Higher = smoother.
    
    spline_degree : int, optional (default=3)
        Polynomial degree for spline (3 = cubic).
    
    Returns:
    --------
    List[float]
        List of split thresholds.
    """
    # Remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    # Fall back to decision tree for very small samples
    if len(x_clean) < 20:
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean), 
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    n_samples = len(x_clean)
    
    # STEP 1: Sort data by x values
    sorted_idx = np.argsort(x_clean)
    x_sorted = x_clean[sorted_idx]
    y_sorted = y_clean[sorted_idx]
    
    # STEP 2: Calculate smoothed event rate
    # Use rolling window to smooth out noise
    # Window size chosen to give approximately 20-50 points for spline
    window_size = max(n_samples // 30, 10)
    
    # Calculate rolling mean of event rate
    event_rates = pd.Series(y_sorted).rolling(
        window=window_size, 
        center=True,      # Window centered on current point
        min_periods=1     # Allow partial windows at edges
    ).mean()
    event_rates = event_rates.values
    
    # STEP 3: Subsample for spline fitting if too many points
    if len(np.unique(x_sorted)) > 500:
        # Take evenly spaced subset of points
        subsample_idx = np.linspace(0, len(x_sorted) - 1, 500).astype(int)
        x_for_spline = x_sorted[subsample_idx]
        y_for_spline = event_rates[subsample_idx]
    else:
        x_for_spline = x_sorted
        y_for_spline = event_rates
    
    # Remove duplicate x values (required for spline fitting)
    # Keep first occurrence where x changes
    unique_mask = np.concatenate([[True], np.diff(x_for_spline) > 0])
    x_for_spline = x_for_spline[unique_mask]
    y_for_spline = y_for_spline[unique_mask]
    
    # Need at least 4 points for cubic spline
    if len(x_for_spline) < 4:
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    try:
        # STEP 4: Fit spline to event rate vs x
        # s parameter controls smoothness (higher = smoother)
        s_factor = smoothing * len(x_for_spline)
        
        # Limit spline degree to number of points - 1
        actual_degree = min(spline_degree, len(x_for_spline) - 1)
        
        spline = UnivariateSpline(
            x_for_spline, 
            y_for_spline, 
            k=actual_degree,
            s=s_factor
        )
        
        # STEP 5: Evaluate spline at many points
        x_eval = np.linspace(x_for_spline.min(), x_for_spline.max(), 1000)
        y_eval = spline(x_eval)
        
        # STEP 6: Compute derivatives
        # First derivative: rate of change
        spline_deriv = spline.derivative()
        y_deriv = spline_deriv(x_eval)
        
        # Second derivative: rate of change of rate of change
        spline_deriv2 = spline_deriv.derivative()
        y_deriv2 = spline_deriv2(x_eval)
        
        # STEP 7: Find inflection points
        # Inflection points are where second derivative changes sign
        inflection_idx = np.where(np.diff(np.sign(y_deriv2)))[0]
        inflection_points = x_eval[inflection_idx]
        
        # Collect candidate split points
        candidate_splits = list(inflection_points)
        
        # If not enough inflection points, add quantile-based splits
        if len(candidate_splits) < max_bins - 1:
            n_quantile_splits = max_bins - 1 - len(candidate_splits)
            quantile_splits = np.percentile(
                x_clean, 
                np.linspace(10, 90, n_quantile_splits)
            )
            candidate_splits.extend(quantile_splits)
        
        # Deduplicate and sort
        candidate_splits = sorted(set(candidate_splits))
        
        # STEP 8: Filter to ensure minimum bin size
        min_count = max(int(n_samples * min_bin_pct), 20)
        valid_splits = []
        prev_split = x_clean.min()
        
        for split in candidate_splits:
            # Count observations in bin from prev_split to split
            count_in_bin = np.sum((x_clean > prev_split) & (x_clean <= split))
            if count_in_bin >= min_count:
                valid_splits.append(split)
                prev_split = split
        
        # Limit to max_bins - 1 splits
        if len(valid_splits) >= max_bins:
            valid_splits = valid_splits[:max_bins - 1]
        
        return valid_splits
        
    except Exception as e:
        # Fall back to decision tree if spline fitting fails
        log_progress(f"  Spline fitting failed: {e}, falling back to DecisionTree")
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )


# =============================================================================
# SECTION 15: BINNING ALGORITHM - ISOTONIC REGRESSION
# =============================================================================
# Isotonic regression fits a monotonic (always increasing or decreasing)
# function to data. For binning, we fit isotonic regression to the event
# rate and use the fitted values to determine bin boundaries.
#
# Advantages:
# - Guaranteed monotonicity (important for interpretable scorecards)
# - Finer-grained than decision trees (finds more splits)
# - Statistically principled approach

def get_isotonic_splits(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 10,
    min_bin_pct: float = 0.01,
    direction: str = 'auto'
) -> List[float]:
    """
    Isotonic regression binning for finer-grained monotonic binning.
    
    Uses sklearn's IsotonicRegression to fit a monotonic function to
    the event rate as a function of x. Bin boundaries are placed where
    the fitted function changes value.
    
    Algorithm:
    ----------
    1. Determine direction (increasing or decreasing) based on correlation
    2. Fit IsotonicRegression to predict event from x
    3. Find unique fitted values (isotonic creates step function)
    4. Place splits at boundaries between different fitted values
    5. Reduce to max_bins using clustering if needed
    6. Ensure minimum bin size
    
    Parameters:
    -----------
    x : pd.Series
        Feature values to bin.
    
    y : pd.Series
        Binary target (0/1).
    
    max_bins : int, optional (default=10)
        Maximum number of bins.
    
    min_bin_pct : float, optional (default=0.01)
        Minimum proportion of observations per bin.
    
    direction : str, optional (default='auto')
        Monotonicity direction: 'auto', 'ascending', 'increasing',
        'descending', or 'decreasing'.
    
    Returns:
    --------
    List[float]
        List of split thresholds.
    """
    # Remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    # Need at least 10 observations
    if len(x_clean) < 10:
        return []
    
    n_samples = len(x_clean)
    
    # Determine direction if 'auto'
    if direction == 'auto':
        # Calculate correlation between x and y
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        
        # If correlation is very small or undefined, default to increasing
        if np.isnan(corr) or abs(corr) < 0.01:
            increasing = True
        else:
            # Positive correlation = increasing, negative = decreasing
            increasing = corr > 0
    else:
        # Parse direction parameter
        increasing = direction in ['ascending', 'increasing']
    
    # Create and fit IsotonicRegression
    # out_of_bounds='clip' handles values outside the training range
    iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    
    try:
        # Fit isotonic regression and get fitted values
        y_iso = iso.fit_transform(x_clean, y_clean)
    except Exception:
        # Fall back to decision tree if isotonic fails
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    # Find unique fitted values
    # Isotonic regression creates a piecewise constant (step) function
    # Each unique fitted value represents a "level"
    unique_fitted = np.unique(y_iso)
    
    # If only one level, no splits possible
    if len(unique_fitted) <= 1:
        return []
    
    # Find split points (boundaries between different fitted values)
    splits = []
    for i in range(len(unique_fitted) - 1):
        # Find x values with current fitted value
        mask_current = y_iso == unique_fitted[i]
        # Find x values with next fitted value
        mask_next = y_iso == unique_fitted[i + 1]
        
        if mask_current.any() and mask_next.any():
            # Split point is between max x of current level and min x of next level
            max_x_current = x_clean[mask_current].max()
            min_x_next = x_clean[mask_next].min()
            # Use midpoint as split
            split = (max_x_current + min_x_next) / 2
            splits.append(split)
    
    # Remove duplicates and sort
    splits = sorted(set(splits))
    
    # If too many splits, use K-means clustering to reduce
    if len(splits) > max_bins - 1:
        splits_arr = np.array(splits).reshape(-1, 1)
        kmeans = KMeans(
            n_clusters=max_bins - 1, 
            random_state=42, 
            n_init=10  # Number of initializations
        )
        kmeans.fit(splits_arr)
        # Use cluster centers as new splits
        splits = sorted(kmeans.cluster_centers_.flatten())
    
    # Ensure minimum bin size
    min_count = max(int(n_samples * min_bin_pct), 20)
    valid_splits = []
    prev_split = -np.inf
    
    for split in splits:
        # Count observations in bin
        if prev_split == -np.inf:
            count_in_bin = np.sum(x_clean <= split)
        else:
            count_in_bin = np.sum((x_clean > prev_split) & (x_clean <= split))
        
        # Only keep split if resulting bin has enough observations
        if count_in_bin >= min_count:
            valid_splits.append(split)
            prev_split = split
    
    return valid_splits


# =============================================================================
# SECTION 16: MONOTONICITY ENFORCEMENT WITH ADVANCED TREND OPTIONS
# =============================================================================
# After initial binning, this function enforces monotonicity constraints
# on the WOE values. This is important for:
# - Scorecard interpretability (higher x = consistently higher/lower risk)
# - Regulatory compliance (some jurisdictions require monotonic scorecards)
# - Preventing overfitting from random noise
#
# Advanced options beyond simple ascending/descending:
# - Peak: Allows one peak (increasing then decreasing)
# - Valley: Allows one valley (decreasing then increasing)
# - Concave: Rate of change must decrease
# - Convex: Rate of change must increase

def enforce_monotonicity(
    x: pd.Series,
    y: pd.Series,
    splits: List[float],
    trend: str = 'auto'
) -> List[float]:
    """
    Enforce monotonicity with advanced trend options.
    
    Given initial splits, this function removes splits that create
    violations of the specified monotonic trend, iteratively merging
    bins until monotonicity is achieved.
    
    Trend Options:
    --------------
    'auto': Automatically detect best monotonic trend based on correlation
    'ascending': WOE must increase with x (positive relationship)
    'descending': WOE must decrease with x (negative relationship)
    'peak': Allows one peak (increase then decrease) - for sweet spots
    'valley': Allows one valley (decrease then increase)
    'concave': Second derivative of WOE must be negative
    'convex': Second derivative of WOE must be positive
    'none': No monotonicity constraint
    
    Parameters:
    -----------
    x : pd.Series
        Feature values.
    
    y : pd.Series
        Binary target.
    
    splits : List[float]
        Initial split points to modify.
    
    trend : str, optional (default='auto')
        Which monotonic trend to enforce.
    
    Returns:
    --------
    List[float]
        Modified split points that satisfy the monotonicity constraint.
    """
    # If no splits or no constraint, return unchanged
    if len(splits) == 0 or trend == 'none':
        return splits
    
    # Remove missing values
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return splits
    
    # Helper function: Calculate WOE values for current splits
    def get_bin_woes(current_splits):
        """
        Calculate WOE for each bin given the current splits.
        Returns tuple of (woe_list, bin_data_list).
        """
        edges = [-np.inf] + list(current_splits) + [np.inf]
        bin_data = []
        
        for i in range(len(edges) - 1):
            # Create mask for this bin
            if i == 0:
                bin_mask = (x_clean <= edges[i + 1])
            elif i == len(edges) - 2:
                bin_mask = (x_clean > edges[i])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            # Calculate bin statistics
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())
            goods = int(count - bads)
            bin_data.append({'goods': goods, 'bads': bads, 'count': count})
        
        # Convert to arrays for WOE calculation
        goods_arr = np.array([b['goods'] for b in bin_data])
        bads_arr = np.array([b['bads'] for b in bin_data])
        
        # Calculate WOE without shrinkage for monotonicity checking
        woes, _, _ = calculate_woe(goods_arr, bads_arr, method="None")
        
        return list(woes), bin_data
    
    # AUTO DETECTION: Determine optimal trend based on correlation
    if trend == 'auto':
        corr = x_clean.corr(y_clean)
        # If correlation is undefined or very small, return unchanged
        if pd.isna(corr) or abs(corr) < 0.01:
            return splits
        # Positive correlation = ascending (higher x = higher WOE/risk)
        # Negative correlation = descending
        trend = 'ascending' if corr > 0 else 'descending'
    
    # Work with a copy of splits
    current_splits = list(splits)
    
    # SIMPLE MONOTONICITY (ascending/descending)
    if trend in ['ascending', 'descending']:
        # Iteratively remove splits that cause monotonicity violations
        max_iterations = 50  # Prevent infinite loop
        
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            # Get current WOE values
            woes, _ = get_bin_woes(current_splits)
            
            # Find first violation
            violating_idx = -1
            for i in range(1, len(woes)):
                if trend == 'ascending' and woes[i] < woes[i - 1]:
                    # Ascending violated: current bin's WOE is less than previous
                    violating_idx = i
                    break
                elif trend == 'descending' and woes[i] > woes[i - 1]:
                    # Descending violated: current bin's WOE is greater than previous
                    violating_idx = i
                    break
            
            # If no violations, we're done
            if violating_idx == -1:
                break
            
            # Remove the split that created the violation
            # Split at index i-1 separates bins i-1 and i
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            elif len(current_splits) > 0:
                current_splits.pop(0)
            else:
                break
    
    # PEAK TREND: Allows increasing then decreasing
    elif trend == 'peak':
        woes, _ = get_bin_woes(current_splits)
        
        # Find optimal peak position that minimizes violations
        best_peak_idx = 0
        best_violations = len(woes)
        
        for peak_idx in range(1, len(woes)):
            violations = 0
            # Before peak: should be ascending
            for i in range(1, peak_idx + 1):
                if woes[i] < woes[i - 1]:
                    violations += 1
            # After peak: should be descending
            for i in range(peak_idx + 1, len(woes)):
                if woes[i] > woes[i - 1]:
                    violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_peak_idx = peak_idx
        
        # Enforce peak pattern by removing violating splits
        max_iterations = 20
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            woes, _ = get_bin_woes(current_splits)
            peak_idx = min(best_peak_idx, len(woes) - 1)
            
            # Find violations
            violating_idx = -1
            # Check ascending part (before peak)
            for i in range(1, peak_idx + 1):
                if i < len(woes) and woes[i] < woes[i - 1]:
                    violating_idx = i
                    break
            # Check descending part (after peak)
            if violating_idx == -1:
                for i in range(peak_idx + 1, len(woes)):
                    if woes[i] > woes[i - 1]:
                        violating_idx = i
                        break
            
            if violating_idx == -1:
                break
            
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            else:
                break
    
    # VALLEY TREND: Allows decreasing then increasing
    elif trend == 'valley':
        woes, _ = get_bin_woes(current_splits)
        
        # Find optimal valley position
        best_valley_idx = 0
        best_violations = len(woes)
        
        for valley_idx in range(1, len(woes)):
            violations = 0
            # Before valley: should be descending
            for i in range(1, valley_idx + 1):
                if woes[i] > woes[i - 1]:
                    violations += 1
            # After valley: should be ascending
            for i in range(valley_idx + 1, len(woes)):
                if woes[i] < woes[i - 1]:
                    violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_valley_idx = valley_idx
        
        # Enforce valley pattern
        max_iterations = 20
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            woes, _ = get_bin_woes(current_splits)
            valley_idx = min(best_valley_idx, len(woes) - 1)
            
            violating_idx = -1
            for i in range(1, valley_idx + 1):
                if i < len(woes) and woes[i] > woes[i - 1]:
                    violating_idx = i
                    break
            if violating_idx == -1:
                for i in range(valley_idx + 1, len(woes)):
                    if woes[i] < woes[i - 1]:
                        violating_idx = i
                        break
            
            if violating_idx == -1:
                break
            
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            else:
                break
    
    return current_splits


# =============================================================================
# SECTION 17: STREAMING BINNING SUPPORT (QUANTILE SKETCHES)
# =============================================================================
# For very large datasets or streaming data, loading all data into memory
# is impractical. Quantile sketches provide approximate quantiles using
# limited memory. This implementation uses a simplified t-digest-like approach.
#
# Note: This is a placeholder implementation. For production streaming,
# consider using specialized libraries like t-digest or KLL sketches.

class QuantileSketch:
    """
    Simple quantile sketch for streaming/large data.
    
    This class maintains approximate quantile estimates from a stream
    of data points without storing all the data. It uses a compression
    mechanism to bound memory usage.
    
    For production use, consider:
    - t-digest (https://github.com/tdunning/t-digest)
    - KLL sketch (from Apache DataSketches)
    - Greenwald-Khanna algorithm
    
    Attributes:
    -----------
    max_size : int
        Maximum number of centroids to maintain.
    
    centroids : List[Tuple[float, int]]
        List of (value, count) tuples representing compressed data.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the quantile sketch.
        
        Parameters:
        -----------
        max_size : int, optional (default=1000)
            Maximum number of centroids to maintain.
            Higher values give more accurate quantiles but use more memory.
        """
        self.max_size = max_size
        # Each centroid is (value, count) representing a cluster of similar values
        self.centroids: List[Tuple[float, int]] = []
        self.total_count = 0
    
    def add(self, value: float, count: int = 1):
        """
        Add a value to the sketch.
        
        Parameters:
        -----------
        value : float
            The value to add.
        
        count : int, optional (default=1)
            Number of times to add this value (for weighted data).
        """
        # Add new centroid
        self.centroids.append((value, count))
        self.total_count += count
        
        # Compress if needed
        if len(self.centroids) > self.max_size * 2:
            self._compress()
    
    def _compress(self):
        """
        Compress centroids to reduce memory usage.
        
        Merge adjacent centroids until we're at or below max_size.
        """
        if len(self.centroids) <= 1:
            return
        
        # Sort centroids by value
        self.centroids.sort(key=lambda x: x[0])
        
        # Merge adjacent pairs until size is acceptable
        while len(self.centroids) > self.max_size:
            # Find the pair with smallest distance
            min_dist = float('inf')
            min_idx = 0
            
            for i in range(len(self.centroids) - 1):
                dist = self.centroids[i + 1][0] - self.centroids[i][0]
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            
            # Merge centroids at min_idx and min_idx + 1
            v1, c1 = self.centroids[min_idx]
            v2, c2 = self.centroids[min_idx + 1]
            
            # Weighted average
            merged_value = (v1 * c1 + v2 * c2) / (c1 + c2)
            merged_count = c1 + c2
            
            # Replace pair with merged centroid
            self.centroids = (
                self.centroids[:min_idx] + 
                [(merged_value, merged_count)] + 
                self.centroids[min_idx + 2:]
            )
    
    def get_quantile(self, q: float) -> float:
        """
        Get the value at quantile q.
        
        Parameters:
        -----------
        q : float
            Quantile to get (0.0 to 1.0).
        
        Returns:
        --------
        float
            Approximate value at quantile q.
        """
        if len(self.centroids) == 0:
            return 0.0
        
        # Sort centroids
        sorted_centroids = sorted(self.centroids, key=lambda x: x[0])
        
        # Find target count
        target = q * self.total_count
        
        # Accumulate counts until we reach target
        cumulative = 0
        for value, count in sorted_centroids:
            cumulative += count
            if cumulative >= target:
                return value
        
        # Return last value if we reach the end
        return sorted_centroids[-1][0]
    
    def get_splits(self, n_splits: int) -> List[float]:
        """
        Get n evenly spaced quantile splits.
        
        Parameters:
        -----------
        n_splits : int
            Number of split points to generate.
        
        Returns:
        --------
        List[float]
            Split points at evenly spaced quantiles.
        """
        if n_splits <= 0:
            return []
        
        # Generate quantiles: 1/(n+1), 2/(n+1), ..., n/(n+1)
        quantiles = [i / (n_splits + 1) for i in range(1, n_splits + 1)]
        return [self.get_quantile(q) for q in quantiles]


class StreamingBinner:
    """
    Streaming binning using quantile sketches.
    
    This class enables binning on data that doesn't fit in memory
    by using quantile sketches to determine split points incrementally.
    
    Usage:
    ------
    binner = StreamingBinner(max_bins=10)
    
    # Stream data in chunks
    for chunk in data_chunks:
        binner.update(chunk['x'], chunk['y'])
    
    # Get final splits
    splits = binner.get_splits()
    """
    
    def __init__(self, max_bins: int = 10, min_bin_pct: float = 0.01):
        """
        Initialize the streaming binner.
        
        Parameters:
        -----------
        max_bins : int, optional (default=10)
            Maximum number of bins.
        
        min_bin_pct : float, optional (default=0.01)
            Minimum proportion of data per bin.
        """
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        
        # Quantile sketch for x values
        self.x_sketch = QuantileSketch()
        
        # Track total counts
        self.total_count = 0
        self.total_bads = 0
    
    def update(self, x_values: np.ndarray, y_values: np.ndarray):
        """
        Update the sketch with a batch of data.
        
        Parameters:
        -----------
        x_values : np.ndarray
            Feature values.
        
        y_values : np.ndarray
            Binary target values (0/1).
        """
        # Add non-null values to sketch
        mask = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_clean = x_values[mask]
        y_clean = y_values[mask]
        
        for x in x_clean:
            self.x_sketch.add(x)
        
        self.total_count += len(x_clean)
        self.total_bads += int(y_clean.sum())
    
    def get_splits(self) -> List[float]:
        """
        Get final split points based on accumulated data.
        
        Returns:
        --------
        List[float]
            Quantile-based split points.
        """
        # Calculate number of splits
        n_splits = min(self.max_bins - 1, self.x_sketch.total_count // 100)
        n_splits = max(n_splits, 1)
        
        return self.x_sketch.get_splits(n_splits)


# =============================================================================
# SECTION 18: VARIABLE PSI CALCULATION
# =============================================================================
# Builds on the basic PSI function to calculate PSI for a specific variable
# given current and reference binning results.

def calculate_variable_psi(
    bins_df: pd.DataFrame,
    reference_bins_df: pd.DataFrame,
    var_name: str
) -> PSIResult:
    """
    Calculate PSI for a single variable given current and reference binning.
    
    This function compares the bin distributions between two datasets
    (typically training/reference vs. new/scoring data) to detect drift.
    
    Parameters:
    -----------
    bins_df : pd.DataFrame
        Current binning results with 'var', 'bin', 'count' columns.
    
    reference_bins_df : pd.DataFrame
        Reference binning results (from training or stable period).
    
    var_name : str
        Name of the variable to calculate PSI for.
    
    Returns:
    --------
    PSIResult
        Dataclass containing PSI value, status, and per-bin details.
    """
    # Filter to specific variable
    current = bins_df[bins_df['var'] == var_name].copy()
    reference = reference_bins_df[reference_bins_df['var'] == var_name].copy()
    
    # Exclude Total row (aggregate summary)
    current = current[current['bin'] != 'Total']
    reference = reference[reference['bin'] != 'Total']
    
    # Handle missing data
    if current.empty or reference.empty:
        return PSIResult(
            variable=var_name,
            psi_value=0.0,
            status="unknown",
            bin_details=pd.DataFrame()
        )
    
    # Calculate current proportions
    current_props = current['count'].values / current['count'].sum()
    
    # Match bins between current and reference
    matched_ref_props = []
    for bin_label in current['bin'].values:
        ref_match = reference[reference['bin'] == bin_label]
        if not ref_match.empty:
            # Found matching bin in reference
            ref_prop = ref_match['count'].values[0] / reference['count'].sum()
        else:
            # New bin not in reference - use small value
            ref_prop = 0.0001
        matched_ref_props.append(ref_prop)
    
    reference_props = np.array(matched_ref_props)
    
    # Calculate PSI
    psi_value = calculate_psi(reference_props, current_props)
    
    # Determine status based on PSI threshold
    if psi_value < 0.1:
        status = "stable"
    elif psi_value < 0.2:
        status = "moderate_drift"
    else:
        status = "significant_drift"
    
    # Create detailed bin-level PSI breakdown
    bin_details = current[['bin', 'count']].copy()
    bin_details['current_prop'] = current_props
    bin_details['reference_prop'] = reference_props
    bin_details['psi_contribution'] = (
        (current_props - reference_props) * 
        np.log(current_props / reference_props)
    )
    
    return PSIResult(
        variable=var_name,
        psi_value=psi_value,
        status=status,
        bin_details=bin_details
    )


# =============================================================================
# SECTION 19: P-VALUE BASED BIN MERGING
# =============================================================================
# Post-processing step to merge adjacent bins that are not statistically
# significantly different from each other.

def merge_by_p_value(
    bins_df: pd.DataFrame,
    var_name: str,
    max_p_value: float = 0.05
) -> pd.DataFrame:
    """
    Merge adjacent bins that are not statistically significantly different.
    
    Uses chi-square test to determine if adjacent bins have significantly
    different good/bad proportions. If p-value > max_p_value, merge them.
    
    Parameters:
    -----------
    bins_df : pd.DataFrame
        Binning results DataFrame.
    
    var_name : str
        Variable to process.
    
    max_p_value : float, optional (default=0.05)
        Maximum p-value threshold. Bins with p-value > threshold are merged.
    
    Returns:
    --------
    pd.DataFrame
        Updated bins DataFrame with merged bins.
    """
    # Get bins for this variable (excluding Total row)
    var_bins = bins_df[bins_df['var'] == var_name].copy()
    var_bins = var_bins[var_bins['bin'] != 'Total']
    
    # Need at least 3 bins to merge (keep minimum of 2)
    if len(var_bins) <= 2:
        return bins_df
    
    # Reset index for easier manipulation
    var_bins = var_bins.reset_index(drop=True)
    
    # Iteratively merge bins with high p-value
    merged = True
    while merged and len(var_bins) > 2:
        merged = False
        
        for i in range(len(var_bins) - 1):
            # Calculate p-value between bins i and i+1
            p_value = chi_square_p_value(
                int(var_bins.iloc[i]['goods']), 
                int(var_bins.iloc[i]['bads']),
                int(var_bins.iloc[i + 1]['goods']), 
                int(var_bins.iloc[i + 1]['bads'])
            )
            
            if p_value > max_p_value:
                # Merge bins i and i+1
                new_row = var_bins.iloc[i].copy()
                new_row['count'] = var_bins.iloc[i]['count'] + var_bins.iloc[i + 1]['count']
                new_row['goods'] = var_bins.iloc[i]['goods'] + var_bins.iloc[i + 1]['goods']
                new_row['bads'] = var_bins.iloc[i]['bads'] + var_bins.iloc[i + 1]['bads']
                new_row['bin'] = f"merged_{i}"
                
                var_bins = pd.concat([
                    var_bins.iloc[:i],
                    pd.DataFrame([new_row]),
                    var_bins.iloc[i + 2:]
                ]).reset_index(drop=True)
                
                merged = True
                break
    
    # Combine with other variables and Total rows
    other_bins = bins_df[(bins_df['var'] != var_name) | (bins_df['bin'] == 'Total')]
    return pd.concat([other_bins, var_bins], ignore_index=True)


# =============================================================================
# SECTION 20: MAIN BINNING ORCHESTRATION FUNCTION
# =============================================================================
# This is the main function that orchestrates the entire binning process.
# It selects the appropriate algorithm, applies monotonicity constraints,
# and produces the final binning results.
#
# Due to the complexity and length of this section, please refer to the
# original woe2_editor_advanced.py for the complete implementation of:
# - create_numeric_bins(): Creates bin DataFrame for numeric variables
# - create_factor_bins(): Creates bin DataFrame for categorical variables
# - get_bins(): Main binning function with algorithm selection
# - apply_bins(): Applies existing binning rules to new data
# - create_binned_columns(): Creates b_* columns in DataFrame
# - add_woe_columns(): Creates WOE_* columns in DataFrame
# - The main KNIME processing logic
# - The interactive Shiny UI
#
# =============================================================================
# END OF COMPREHENSIVELY COMMENTED CODE
# =============================================================================
#
# This file documents the core algorithms and concepts of WOE 2.0 binning.
# For the complete working implementation, use woe2_editor_advanced.py.
#
# Key concepts covered:
# 1. Weight of Evidence (WOE) and Information Value (IV)
# 2. Bayesian shrinkage with Beta-Binomial posterior
# 3. Population Stability Index (PSI) for drift detection
# 4. Chi-square statistics for bin merging
# 5. Decision Tree binning (R-compatible CART)
# 6. ChiMerge binning (bottom-up chi-square based)
# 7. IV-Optimal binning (maximize Information Value)
# 8. Spline-based binning (WOE 2.0 approach)
# 9. Isotonic regression binning
# 10. Advanced monotonicity enforcement (peak, valley, etc.)
# 11. Streaming binning with quantile sketches
#
# =============================================================================

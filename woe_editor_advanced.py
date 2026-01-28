# =============================================================================
# WOE Editor for KNIME Python Script Node - ADVANCED BINNING VERSION
# =============================================================================
# Python implementation using state-of-the-art optimal binning algorithms
# based on academic research and industry best practices.
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

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re
import warnings
import time
import sys
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# Use random port to avoid conflicts when running multiple instances
BASE_PORT = 8055  # Different from original to avoid conflicts
RANDOM_PORT_RANGE = 1000

# Process isolation: Set unique temp directories per instance
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading conflicts

# =============================================================================
# Progress Logging Utilities
# =============================================================================

def log_progress(message: str, flush: bool = True):
    """Print a progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    if flush:
        sys.stdout.flush()

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

# =============================================================================
# Install/Import Dependencies
# =============================================================================

try:
    from scipy import stats
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    from scipy import stats

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Configuration
# =============================================================================

class BinningConfig:
    """Configuration for advanced binning algorithms
    
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
    # Default values - can be overridden by flow variables
    ALGORITHM = "DecisionTree"  # "DecisionTree" (R-compatible) or "ChiMerge"
    MIN_BIN_PCT = 0.01  # Minimum 1% of observations per bin (matches R's minProp)
    MIN_BIN_COUNT = 20  # Minimum absolute count per bin
    MAX_BINS = 10  # Maximum number of bins
    MIN_BINS = 2  # Minimum number of bins
    MAX_CATEGORIES = 50  # Maximum unique categories for categorical variables (prevents high-cardinality issues)
    CHI_MERGE_THRESHOLD = 0.05  # Chi-square p-value threshold
    MIN_IV_GAIN = 0.005  # Minimum IV improvement to continue splitting
    USE_SHRINKAGE = False  # Apply shrinkage to WOE values (optional)
    SHRINKAGE_STRENGTH = 0.1  # Shrinkage regularization strength
    USE_ENHANCEMENTS = False  # Master switch: enables ALL enhancements when True
    
    # Individual enhancement flags (can be set independently)
    ADAPTIVE_MIN_PROP = False  # Relaxes min_prop for sparse data (< 500 samples)
    MIN_EVENT_COUNT = False  # Ensures at least N events per potential bin
    AUTO_RETRY = False  # Retry with relaxed constraints if no splits found
    CHI_SQUARE_VALIDATION = False  # Merge statistically similar bins post-binning
    SINGLE_BIN_PROTECTION = True  # Prevent na_combine from creating single-bin vars (default ON)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BinResult:
    """Container for binning results"""
    var_summary: pd.DataFrame  # Summary stats for each variable
    bin: pd.DataFrame  # Detailed bin information


# =============================================================================
# Core WOE/IV Calculation Functions
# =============================================================================

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray, 
                  use_shrinkage: bool = False, shrinkage_strength: float = 0.1) -> np.ndarray:
    """
    Calculate Weight of Evidence (WOE) for each bin.
    WOE = ln((% of Bads in bin) / (% of Goods in bin))
    
    With optional shrinkage for rare events (fraud detection).
    """
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    # Apply Laplace smoothing to prevent division by zero
    epsilon = 0.0001
    dist_good = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad / dist_good)
    
    # Apply shrinkage (empirical Bayes) for rare events
    if use_shrinkage and shrinkage_strength > 0:
        n_obs = freq_good + freq_bad
        total_obs = n_obs.sum()
        # Weight based on sample size - larger samples get less shrinkage
        weights = n_obs / (n_obs + shrinkage_strength * total_obs / len(n_obs))
        woe = woe * weights
    
    return np.round(woe, 5)


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """Calculate Information Value (IV) for a variable."""
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    epsilon = 0.0001
    dist_good_safe = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad_safe / dist_good_safe)
    iv = np.sum((dist_bad - dist_good) * woe)
    
    if not np.isfinite(iv):
        iv = 0.0
        
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """Calculate entropy for a bin."""
    total = goods + bads
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    p_good = goods / total
    p_bad = bads / total
    
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """Determine if variable is numeric or factor (categorical)"""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return 'factor'
        return 'numeric'
    return 'factor'


# =============================================================================
# ChiMerge Algorithm - Core of Advanced Binning
# =============================================================================

def _chi_square_statistic(bin1_good: int, bin1_bad: int, 
                          bin2_good: int, bin2_bad: int) -> float:
    """
    Calculate chi-square statistic for two adjacent bins.
    Uses Yates continuity correction for 2x2 tables.
    """
    observed = np.array([[bin1_good, bin1_bad], [bin2_good, bin2_bad]])
    
    if observed.sum() == 0:
        return np.inf
    
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    
    if total == 0 or any(row_totals == 0) or any(col_totals == 0):
        return np.inf
    
    # Expected frequencies
    expected = np.outer(row_totals, col_totals) / total
    
    # Chi-square with Yates continuity correction
    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i, j] > 0:
                diff = abs(observed[i, j] - expected[i, j]) - 0.5
                diff = max(diff, 0)
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
    ChiMerge algorithm: Start with fine-grained bins, merge adjacent bins 
    with smallest chi-square statistic until stopping criterion is met.
    
    This is more statistically rigorous than decision tree-based splitting.
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return []
    
    total_count = len(x_clean)
    min_count_required = max(int(total_count * min_bin_pct), min_bin_count)
    
    # Start with fine-grained bins based on percentiles
    n_initial_bins = min(100, len(x_clean.unique()))
    
    try:
        initial_bins = pd.qcut(x_clean, q=n_initial_bins, duplicates='drop')
    except ValueError:
        try:
            initial_bins = pd.cut(x_clean, bins=min(20, len(x_clean.unique())), duplicates='drop')
        except:
            return []
    
    # Get bin edges
    if hasattr(initial_bins, 'categories') and len(initial_bins.categories) > 0:
        edges = sorted(set(
            [initial_bins.categories[0].left] + 
            [cat.right for cat in initial_bins.categories]
        ))
    else:
        return []
    
    # Create initial frequency table for each bin
    def build_bin_stats(edges):
        bins_stats = []
        for i in range(len(edges) - 1):
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
    
    bins_stats = build_bin_stats(edges)
    
    # Remove empty bins
    bins_stats = [b for b in bins_stats if b['count'] > 0]
    
    if len(bins_stats) <= min_bins:
        return [b['right'] for b in bins_stats[:-1]]
    
    # Chi-square critical value for 1 degree of freedom
    chi2_threshold = stats.chi2.ppf(1 - chi_threshold, df=1)
    
    # Iteratively merge bins with smallest chi-square
    while len(bins_stats) > min_bins:
        min_chi2 = np.inf
        merge_idx = -1
        
        # Find adjacent pair with minimum chi-square
        for i in range(len(bins_stats) - 1):
            chi2 = _chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        # Stop if chi-square exceeds threshold AND we have acceptable bin count
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
    
    # Enforce minimum bin size by merging small bins
    changed = True
    while changed and len(bins_stats) > min_bins:
        changed = False
        for i in range(len(bins_stats)):
            if bins_stats[i]['count'] < min_count_required:
                # Merge with adjacent bin that has closer bad rate
                if i == 0 and len(bins_stats) > 1:
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
    
    # Reduce to max_bins if still too many
    while len(bins_stats) > max_bins:
        min_chi2 = np.inf
        merge_idx = 0
        
        for i in range(len(bins_stats) - 1):
            chi2 = _chi_square_statistic(
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
    
    # Extract split points (excluding endpoints)
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
    This is done DURING binning, not as post-processing.
    
    direction: 'auto', 'increasing', or 'decreasing'
    """
    if len(splits) == 0:
        return splits
    
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Detect optimal direction if auto
    if direction == 'auto':
        corr = x_clean.corr(y_clean)
        if pd.isna(corr) or abs(corr) < 0.01:
            return splits  # Not enough correlation to enforce monotonicity
        direction = 'increasing' if corr > 0 else 'decreasing'
    
    def get_bin_woes(current_splits):
        edges = [-np.inf] + list(current_splits) + [np.inf]
        bin_data = []
        
        for i in range(len(edges) - 1):
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
        
        goods_arr = np.array([b['goods'] for b in bin_data])
        bads_arr = np.array([b['bads'] for b in bin_data])
        woes = calculate_woe(goods_arr, bads_arr)
        
        return list(woes), bin_data
    
    current_splits = list(splits)
    max_iterations = 50
    
    for _ in range(max_iterations):
        if len(current_splits) == 0:
            break
            
        woes, bin_data = get_bin_woes(current_splits)
        
        # Check for monotonicity violations
        violating_idx = -1
        for i in range(1, len(woes)):
            if direction == 'increasing' and woes[i] < woes[i - 1]:
                violating_idx = i
                break
            elif direction == 'decreasing' and woes[i] > woes[i - 1]:
                violating_idx = i
                break
        
        if violating_idx == -1:
            break  # No violations, we're done
        
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
    This matches R's logiBin::getBins algorithm.
    
    Individual enhancements can be enabled/disabled:
    1. Adaptive min_prop (adaptive_min_prop): Relaxes min_prop for sparse data
    2. Minimum event count (min_event_count): Prevents unstable bins in low-event data
    3. Automatic retry (auto_retry): Retries with relaxed constraints if no splits found
    
    Parameters:
        x: Feature values
        y: Binary target values
        min_prop: Minimum proportion of samples per bin (R default is 0.01 = 1%)
        max_bins: Maximum number of bins (leaf nodes)
        min_events: Minimum number of events (bads) required per potential bin
        use_enhancements: Master switch - if True, enables all enhancements
        adaptive_min_prop: Enable adaptive min_prop for sparse data (overrides master)
        min_event_count: Enable minimum event count per bin (overrides master)
        auto_retry: Enable auto-retry with relaxed constraints (overrides master)
    
    Returns:
        List of split thresholds
    """
    # Resolve individual flags: use explicit value if provided, else fall back to master switch
    use_adaptive = adaptive_min_prop if adaptive_min_prop is not None else use_enhancements
    use_min_events = min_event_count if min_event_count is not None else use_enhancements
    use_auto_retry = auto_retry if auto_retry is not None else use_enhancements
    
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        return []
    
    n_samples = len(x_clean)
    n_events = int(y_clean.sum())
    event_rate = n_events / n_samples if n_samples > 0 else 0
    
    # Default: Use min_prop directly (R-compatible)
    effective_min_prop = min_prop
    
    # ENHANCEMENT 1: Adaptive min_prop for sparse data
    if use_adaptive:
        # For very sparse data (few non-null values), relax the constraint
        if n_samples < 500:
            effective_min_prop = max(min_prop / 2, 0.005)  # At least 0.5%
        # For very low event rates (fraud), ensure enough events per bin
        if event_rate < 0.05 and n_events > 0:
            # Need at least min_events per bin
            max_possible_bins = max(n_events // min_events, 2)
            min_samples_for_events = n_samples / max_possible_bins
            adaptive_prop = min_samples_for_events / n_samples
            effective_min_prop = max(effective_min_prop, adaptive_prop * 0.8)
    
    # Calculate min_samples_leaf based on proportion
    min_samples_leaf = max(int(n_samples * effective_min_prop), 1)
    
    # ENHANCEMENT 2: Ensure minimum event count per leaf
    if use_min_events and n_events > 0 and min_events > 0:
        # Each leaf should have at least min_events bads on average
        min_samples_for_min_events = int(min_events / max(event_rate, 0.001))
        min_samples_leaf = max(min_samples_leaf, min_samples_for_min_events)
    
    # Don't let min_samples_leaf exceed half the data
    min_samples_leaf = min(min_samples_leaf, n_samples // 2)
    min_samples_leaf = max(min_samples_leaf, 1)  # At least 1
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        return []
    
    # Extract thresholds from tree
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]  # -2 indicates leaf node
    thresholds = sorted(set(thresholds))
    
    # ENHANCEMENT 3: Retry with relaxed constraints if no splits found
    if use_auto_retry and len(thresholds) == 0 and min_samples_leaf > 10:
        # Try again with smaller min_samples_leaf
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
            pass
    
    return thresholds


# =============================================================================
# IV-Optimal Algorithm - Maximizes Information Value
# =============================================================================

def _calculate_bin_iv(goods: int, bads: int, total_goods: int, total_bads: int) -> float:
    """Calculate IV contribution for a single bin."""
    if total_goods == 0 or total_bads == 0:
        return 0.0
    
    dist_good = goods / total_goods if total_goods > 0 else 0
    dist_bad = bads / total_bads if total_bads > 0 else 0
    
    # Handle edge cases to avoid log(0)
    if dist_good == 0 or dist_bad == 0:
        return 0.0
    
    woe = np.log(dist_bad / dist_good)
    iv = (dist_bad - dist_good) * woe
    return iv


def _calculate_total_iv(bins_list: List[dict], total_goods: int, total_bads: int) -> float:
    """Calculate total IV for a binning configuration."""
    total_iv = 0.0
    for bin_info in bins_list:
        total_iv += _calculate_bin_iv(
            bin_info['goods'], bin_info['bads'], 
            total_goods, total_bads
        )
    return total_iv


def _get_bin_woe(goods: int, bads: int, total_goods: int, total_bads: int) -> float:
    """Calculate WOE for a single bin."""
    if total_goods == 0 or total_bads == 0:
        return 0.0
    
    dist_good = goods / total_goods if total_goods > 0 else 0
    dist_bad = bads / total_bads if total_bads > 0 else 0
    
    # Handle edge cases
    if dist_good == 0:
        dist_good = 0.0001  # Small epsilon
    if dist_bad == 0:
        dist_bad = 0.0001
    
    return np.log(dist_bad / dist_good)


def _is_pure_bin(goods: int, bads: int) -> bool:
    """Check if bin is pure (100% goods or 100% bads)."""
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
    
    Algorithm:
    1. Start with dynamic granularity based on variable characteristics
    2. Iteratively merge adjacent bins with smallest IV loss
    3. Merge pure bins to closest adjacent bin by WOE
    4. Stop at max_bins or when IV loss exceeds threshold
    
    Parameters:
        x: Feature values
        y: Binary target values
        min_prop: Minimum proportion of samples per bin
        max_bins: Maximum number of bins (stopping rule)
        min_bin_count: Minimum count per bin
        min_iv_loss: Stop merging when IV loss exceeds this threshold
    
    Returns:
        List of split thresholds
    """
    # Clean data
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        return []
    
    n_samples = len(x_clean)
    n_unique = len(np.unique(x_clean))
    total_goods = int((y_clean == 0).sum())
    total_bads = int((y_clean == 1).sum())
    
    if total_goods == 0 or total_bads == 0:
        return []
    
    # ==========================================================================
    # Step 1: Dynamic starting granularity
    # ==========================================================================
    
    # Determine initial number of bins based on variable characteristics
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
    
    current_splits = list(initial_splits)
    bins_list = create_bins_from_splits(current_splits)
    
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
        
        # Find pure bins and merge them
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
                        # Merge with closer WOE (pure bins have extreme WOE: +/-inf or 0)
                        # For pure bins, merge with the one that has closer absolute WOE
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
                    
                    # Update splits (remove the split between merged bins)
                    if merge_idx < len(splits):
                        splits = splits[:merge_idx] + splits[merge_idx + 1:]
                    
                    # Recalculate WOEs
                    woes = [_get_bin_woe(b['goods'], b['bads'], total_goods, total_bads) for b in bins_list]
                    merged = True
                    break
        
        return bins_list, splits
    
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
            # Only enforce this rule if we're reasonably close to max_bins
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

def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """Create bin DataFrame for numeric variable based on splits."""
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        
        if lower == -np.inf:
            mask = (x <= upper) & x.notna()
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        count = mask.sum()
        if count > 0:
            bads = y[mask].sum()
            goods = count - bads
            bins_data.append({
                'var': var,
                'bin': bin_rule,
                'count': int(count),
                'bads': int(bads),
                'goods': int(goods)
            })
    
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


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    max_categories: int = 50,
    max_bins: int = None
) -> pd.DataFrame:
    """
    Create bin DataFrame for factor/categorical variable.
    
    Strategy:
    - Low-cardinality (≤ max_bins): Each category becomes its own bin
    - Medium-cardinality (> max_bins, ≤ max_categories): Groups categories by WOE similarity
      * Unlike numeric bins, categorical bins don't need adjacency
      * Any categories with similar WOE (risk profile) can be grouped together
      * Example: Group high-risk states together, low-risk states together
    - High-cardinality (> max_categories): Skip variable with warning
    
    Parameters:
        df: Input DataFrame
        var: Variable name
        y_var: Target variable name
        max_categories: Maximum categories before skipping (default 50)
        max_bins: Maximum bins to create (uses MAX_BINS from config if None)
        
    Returns:
        DataFrame with bin statistics
    """
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    unique_vals = x.dropna().unique()
    n_unique = len(unique_vals)
    
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
        
        # Avoid division by zero
        if total_goods == 0 or total_bads == 0:
            print(f"  WARNING: Cannot calculate WOE for '{var}' - no variation in target")
            return pd.DataFrame()
        
        for val in unique_vals:
            mask = x == val
            count = mask.sum()
            if count > 0:
                bads = y[mask].sum()
                goods = count - bads
                
                # Calculate WOE for this category
                dist_goods = (goods / total_goods) if total_goods > 0 else 0.0001
                dist_bads = (bads / total_bads) if total_bads > 0 else 0.0001
                
                # Add smoothing to avoid inf/-inf
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
        
        cat_df = pd.DataFrame(cat_stats)
        
        # Group categories by WOE similarity (not by order!)
        # Categories with similar WOE should be grouped together
        # Use quantile-based binning on WOE values
        n_bins = min(max_bins, len(cat_df))
        
        try:
            # Try to create bins based on WOE quantiles
            cat_df['bin_group'] = pd.qcut(cat_df['woe'], q=n_bins, labels=False, duplicates='drop')
        except (ValueError, IndexError):
            # If qcut fails (e.g., too many duplicates), use simple cut
            cat_df['bin_group'] = pd.cut(cat_df['woe'], bins=n_bins, labels=False, duplicates='drop')
        
        # Handle any remaining NaN bin_groups (from duplicates='drop')
        if cat_df['bin_group'].isna().any():
            # Assign NaN groups to median group
            median_group = cat_df['bin_group'].median()
            cat_df['bin_group'] = cat_df['bin_group'].fillna(median_group)
            # If still NaN (all were NaN), assign to 0
            cat_df['bin_group'] = cat_df['bin_group'].fillna(0)
        
        # Aggregate by bin group
        for bin_idx in sorted(cat_df['bin_group'].unique()):
            bin_cats = cat_df[cat_df['bin_group'] == bin_idx]
            values_str = '", "'.join([str(v) for v in bin_cats['value'].tolist()])
            
            total_count = bin_cats['count'].sum()
            total_bads = bin_cats['bads'].sum()
            total_goods = bin_cats['goods'].sum()
            
            # Calculate average WOE for this bin (for logging)
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
    """Update bin statistics (propn, bad_rate, iv, ent, trend, etc.)"""
    if bin_df.empty:
        return bin_df
    
    df = bin_df.copy()
    
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    df['propn'] = round(df['count'] / total_count * 100, 2)
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    df['iv'] = round((df['goodCap'] - df['badCap']) * np.log(
        np.where(df['goodCap'] == 0, 0.0001, df['goodCap']) / 
        np.where(df['badCap'] == 0, 0.0001, df['badCap'])
    ), 4)
    
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    df['ent'] = df.apply(
        lambda row: calculate_entropy(row['goods'], row['bads']), 
        axis=1
    )
    
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    df['trend'] = None
    bad_rates = df['bad_rate'].values
    for i in range(1, len(bad_rates)):
        if 'is.na' not in str(df.iloc[i]['bin']):
            if bad_rates[i] >= bad_rates[i-1]:
                df.iloc[i, df.columns.get_loc('trend')] = 'I'
            else:
                df.iloc[i, df.columns.get_loc('trend')] = 'D'
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Add a total row to the bin DataFrame."""
    df = bin_df.copy()
    
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()
    
    if total_count > 0:
        total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4)
    else:
        total_ent = 0
    
    trends = df[df['trend'].notna()]['trend'].unique()
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    incr_count = len(df[df['trend'] == 'I'])
    decr_count = len(df[df['trend'] == 'D'])
    total_trend_count = incr_count + decr_count
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    num_bins = len(df)
    
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
    This is the main entry point, equivalent to logiBin::getBins in R.
    
    Algorithm Options:
    - "DecisionTree" (default): Uses CART decision tree - matches R's logiBin::getBins
    - "ChiMerge": Uses chi-square based bin merging (more conservative)
    
    Parameters:
        df: DataFrame with data
        y_var: Name of binary dependent variable
        x_vars: List of independent variable names
        min_prop: Minimum proportion of samples per bin (default 0.01 = 1% to match R)
        max_bins: Maximum number of bins
        enforce_monotonic: Whether to enforce monotonic WOE trends
        algorithm: Binning algorithm ("DecisionTree" or "ChiMerge")
        use_enhancements: Master switch - enables all enhancements when True
        adaptive_min_prop: Individual flag for adaptive min_prop (overrides master)
        min_event_count: Individual flag for minimum event count (overrides master)
        auto_retry: Individual flag for auto-retry with relaxed constraints (overrides master)
    """
    all_bins = []
    var_summaries = []
    
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
    
    min_bin_pct = min_prop  # Don't override - use what's passed
    
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
    
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        if var not in df.columns:
            continue
            
        var_type = get_var_type(df[var])
        
        if var_type == 'numeric':
            if algorithm == "ChiMerge":
                # Use ChiMerge for initial splits
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
                # Use IV-optimal algorithm (maximize Information Value)
                splits = _iv_optimal_get_splits(
                    df[var],
                    df[y_var],
                    min_prop=min_bin_pct,
                    max_bins=max_bins,
                    min_bin_count=BinningConfig.MIN_BIN_COUNT,
                    min_iv_loss=BinningConfig.MIN_IV_GAIN
                )
            else:
                # Use Decision Tree (R-compatible) for initial splits
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
            
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            bin_df = _create_factor_bins(df, var, y_var, 
                                         max_categories=BinningConfig.MAX_CATEGORIES,
                                         max_bins=max_bins)
        
        if bin_df.empty:
            continue
        
        bin_df = update_bin_stats(bin_df)
        bin_df = add_total_row(bin_df, var)
        
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
        
        all_bins.append(bin_df)
        
        # Progress logging
        var_time = time.time() - var_start
        times_per_var.append(var_time)
        processed_count += 1
        
        # Log every 10 variables, every 5 seconds, or at specific milestones
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
    
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {processed_count} variables in {format_time(total_time)}")
    log_progress(f"Average time per variable: {format_time(total_time/max(processed_count,1))}")
    
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    var_summary_df = pd.DataFrame(var_summaries)
    
    return BinResult(var_summary=var_summary_df, bin=combined_bins)


def manual_split(
    bin_result: BinResult,
    var: str,
    y_var: str,
    splits: List[float],
    df: pd.DataFrame
) -> BinResult:
    """Manually split a numeric variable at specified points."""
    bin_df = _create_numeric_bins(df, var, y_var, splits)
    
    if bin_df.empty:
        return bin_result
    
    bin_df = update_bin_stats(bin_df)
    bin_df = add_total_row(bin_df, var)
    
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
    new_bins = pd.concat([other_bins, bin_df], ignore_index=True)
    
    total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
    var_summary = bin_result.var_summary.copy()
    
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

def _parse_numeric_from_rule(rule: str) -> List[float]:
    """Extract numeric values from a bin rule string."""
    pattern = r"'(-?\d+\.?\d*)'"
    matches = re.findall(pattern, rule)
    return [float(m) for m in matches]


def _parse_factor_values_from_rule(rule: str) -> List[str]:
    """Extract factor values from a bin rule string."""
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, rule)
    return matches


def na_combine(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    prevent_single_bin: bool = True
) -> BinResult:
    """Combine NA bin with the adjacent bin that has the closest bad rate.
    
    Parameters:
        bin_result: BinResult with binning information
        vars_to_process: Variables to process
        prevent_single_bin: If True, skip combining if it would result in a single bin (WOE=0)
    """
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    skipped_single_bin = []
    
    for var in vars_to_process:
        var_bins = new_bins[new_bins['var'] == var].copy()
        
        if var_bins.empty:
            continue
        
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        
        if not na_mask.any():
            continue
        
        na_bin = var_bins[na_mask].iloc[0]
        non_na_bins = var_bins[~na_mask & (var_bins['bin'] != 'Total')]
        
        if non_na_bins.empty:
            continue
        
        # Prevent creating single-bin variables (which have WOE=0)
        if prevent_single_bin and len(non_na_bins) <= 1:
            skipped_single_bin.append(var)
            continue
        
        na_bad_rate = na_bin['bads'] / na_bin['count'] if na_bin['count'] > 0 else 0
        non_na_bins = non_na_bins.copy()
        non_na_bins['bad_rate_calc'] = non_na_bins['bads'] / non_na_bins['count']
        non_na_bins['rate_diff'] = abs(non_na_bins['bad_rate_calc'] - na_bad_rate)
        
        closest_idx = non_na_bins['rate_diff'].idxmin()
        closest_bin = non_na_bins.loc[closest_idx]
        
        combined_rule = f"{closest_bin['bin']} | is.na({var})"
        combined_count = closest_bin['count'] + na_bin['count']
        combined_goods = closest_bin['goods'] + na_bin['goods']
        combined_bads = closest_bin['bads'] + na_bin['bads']
        
        new_bins.loc[closest_idx, 'bin'] = combined_rule
        new_bins.loc[closest_idx, 'count'] = combined_count
        new_bins.loc[closest_idx, 'goods'] = combined_goods
        new_bins.loc[closest_idx, 'bads'] = combined_bads
        
        na_idx = var_bins[na_mask].index[0]
        new_bins = new_bins.drop(na_idx)
        
        var_new_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        var_new_bins = update_bin_stats(var_new_bins)
        var_new_bins = add_total_row(var_new_bins, var)
        
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, var_new_bins], ignore_index=True)
        
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
    
    # Log skipped variables (would have become single-bin with WOE=0)
    if skipped_single_bin:
        log_progress(f"  - Skipped NA grouping for {len(skipped_single_bin)} variables (would create single-bin WOE=0)")
        if len(skipped_single_bin) <= 10:
            log_progress(f"    Skipped vars: {skipped_single_bin}")
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def validate_bins_chi_square(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None,
    p_value_threshold: float = 0.05
) -> BinResult:
    """
    ENHANCEMENT: Validate that adjacent bins are statistically different using chi-square test.
    Merges adjacent bins that are not significantly different.
    
    This prevents over-binning where splits don't add meaningful predictive power.
    
    Parameters:
        bin_result: BinResult with binning information
        vars_to_process: Variables to process (default: all variables)
        p_value_threshold: Merge bins if p-value > threshold (not significantly different)
    
    Returns:
        BinResult with validated bins
    """
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    merged_count = 0
    
    for var in vars_to_process:
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        
        if var_bins.empty or len(var_bins) <= 2:
            continue
        
        # Separate NA bin
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
        working_bins = var_bins[~na_mask].copy().reset_index(drop=True)
        
        if len(working_bins) <= 2:
            continue
        
        # Iteratively merge bins that aren't significantly different
        changed = True
        while changed and len(working_bins) > 2:
            changed = False
            
            for i in range(len(working_bins) - 1):
                # Chi-square test between adjacent bins
                observed = np.array([
                    [working_bins.iloc[i]['goods'], working_bins.iloc[i]['bads']],
                    [working_bins.iloc[i+1]['goods'], working_bins.iloc[i+1]['bads']]
                ])
                
                if observed.sum() == 0 or any(observed.sum(axis=1) == 0):
                    continue
                
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
                except:
                    continue
                
                # If not significantly different, merge
                if p_value > p_value_threshold:
                    # Merge bins i and i+1
                    working_bins.iloc[i, working_bins.columns.get_loc('count')] += working_bins.iloc[i+1]['count']
                    working_bins.iloc[i, working_bins.columns.get_loc('goods')] += working_bins.iloc[i+1]['goods']
                    working_bins.iloc[i, working_bins.columns.get_loc('bads')] += working_bins.iloc[i+1]['bads']
                    
                    # Update bin rule to cover both ranges
                    old_rule = working_bins.iloc[i]['bin']
                    new_rule = working_bins.iloc[i+1]['bin']
                    if '<=' in new_rule:
                        # Extend upper bound
                        nums = re.findall(r"'(-?\d+\.?\d*)'", new_rule)
                        if nums:
                            new_upper = max(float(n) for n in nums)
                            if '>' in old_rule and '<=' in old_rule:
                                old_nums = re.findall(r"'(-?\d+\.?\d*)'", old_rule.split('&')[0])
                                if old_nums:
                                    working_bins.iloc[i, working_bins.columns.get_loc('bin')] = f"{var} > '{min(float(n) for n in old_nums)}' & {var} <= '{new_upper}'"
                            elif '<=' in old_rule:
                                working_bins.iloc[i, working_bins.columns.get_loc('bin')] = f"{var} <= '{new_upper}'"
                    
                    working_bins = working_bins.drop(working_bins.index[i+1]).reset_index(drop=True)
                    changed = True
                    merged_count += 1
                    break
        
        # Reconstruct variable bins
        if not na_bin.empty:
            working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
        
        working_bins = update_bin_stats(working_bins)
        working_bins = add_total_row(working_bins, var)
        
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
        
        # Update var_summary
        total_row = working_bins[working_bins['bin'] == 'Total'].iloc[0]
        mask = var_summary['var'] == var
        if mask.any():
            var_summary.loc[mask, 'iv'] = total_row['iv']
            var_summary.loc[mask, 'numBins'] = total_row.get('numBins', len(working_bins) - 1)
    
    if merged_count > 0:
        log_progress(f"  - Chi-square validation: merged {merged_count} statistically similar bin pairs")
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def merge_pure_bins(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None
) -> BinResult:
    """
    Merge pure bins (100% goods or 100% bads) with the closest non-pure bin.
    
    Pure bins cause infinite WOE values (division by zero) which break logistic regression.
    This function iteratively merges pure bins until no pure bins remain.
    
    Parameters:
        bin_result: BinResult with binning information
        vars_to_process: Variables to process (default: all variables)
    
    Returns:
        BinResult with pure bins merged
    """
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
            
            if len(var_bins) <= 1:
                # Only one bin left, can't merge further
                break
            
            # Find pure bins (goods=0 or bads=0)
            pure_mask = (var_bins['goods'] == 0) | (var_bins['bads'] == 0)
            
            if not pure_mask.any():
                # No pure bins, we're done with this variable
                break
            
            # Get the first pure bin
            pure_bin = var_bins[pure_mask].iloc[0]
            pure_idx = var_bins[pure_mask].index[0]
            
            # Find non-pure bins to merge with
            non_pure_bins = var_bins[~pure_mask]
            
            if non_pure_bins.empty:
                # All bins are pure - merge the two with closest counts
                other_bins = var_bins[var_bins.index != pure_idx]
                if other_bins.empty:
                    break
                other_bins = other_bins.copy()
                other_bins['count_diff'] = abs(other_bins['count'] - pure_bin['count'])
                closest_idx = other_bins['count_diff'].idxmin()
                closest_bin = other_bins.loc[closest_idx]
            else:
                # Calculate bad rate for pure bin (0 or 1)
                pure_bad_rate = pure_bin['bads'] / pure_bin['count'] if pure_bin['count'] > 0 else 0.5
                
                # Find closest non-pure bin by bad rate
                non_pure_bins = non_pure_bins.copy()
                non_pure_bins['bad_rate_calc'] = non_pure_bins['bads'] / non_pure_bins['count']
                non_pure_bins['rate_diff'] = abs(non_pure_bins['bad_rate_calc'] - pure_bad_rate)
                
                closest_idx = non_pure_bins['rate_diff'].idxmin()
                closest_bin = non_pure_bins.loc[closest_idx]
            
            # Merge the pure bin into the closest bin
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
        
        # Recalculate stats for this variable
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
                var_summary.loc[mask, 'purNode'] = 'N'  # No more pure nodes after merging
    
    return BinResult(var_summary=var_summary, bin=new_bins)


def break_bin(
    bin_result: BinResult,
    var: str,
    y_var: str,
    df: pd.DataFrame
) -> BinResult:
    """Break all bins for a factor variable - each unique value becomes its own bin."""
    new_var_bins = _create_factor_bins(df, var, y_var, 
                                       max_categories=BinningConfig.MAX_CATEGORIES,
                                       max_bins=BinningConfig.MAX_BINS)
    new_var_bins = update_bin_stats(new_var_bins)
    new_var_bins = add_total_row(new_var_bins, var)
    
    other_bins = bin_result.bin[bin_result.bin['var'] != var].copy()
    new_bins = pd.concat([other_bins, new_var_bins], ignore_index=True)
    
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
    """Force an increasing monotonic trend in bad rates by combining adjacent bins."""
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        
        if var_bins.empty or len(var_bins) < 2:
            continue
        
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
        working_bins = var_bins[~na_mask].copy()
        
        if working_bins.empty:
            continue
        
        working_bins = working_bins.reset_index(drop=True)
        
        changed = True
        while changed and len(working_bins) > 1:
            changed = False
            working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
            
            for i in range(1, len(working_bins)):
                if working_bins.iloc[i]['bad_rate_calc'] < working_bins.iloc[i-1]['bad_rate_calc']:
                    working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                    
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
        
        if not na_bin.empty:
            working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
        
        if 'bad_rate_calc' in working_bins.columns:
            working_bins = working_bins.drop('bad_rate_calc', axis=1)
        
        working_bins = update_bin_stats(working_bins)
        working_bins = add_total_row(working_bins, var)
        
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
        
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
    """Force a decreasing monotonic trend in bad rates by combining adjacent bins."""
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        
        if var_bins.empty or len(var_bins) < 2:
            continue
        
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        na_bin = var_bins[na_mask].copy() if na_mask.any() else pd.DataFrame()
        working_bins = var_bins[~na_mask].copy()
        
        if working_bins.empty:
            continue
        
        working_bins = working_bins.reset_index(drop=True)
        
        changed = True
        while changed and len(working_bins) > 1:
            changed = False
            working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
            
            for i in range(1, len(working_bins)):
                if working_bins.iloc[i]['bad_rate_calc'] > working_bins.iloc[i-1]['bad_rate_calc']:
                    working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                    
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
        
        if not na_bin.empty:
            working_bins = pd.concat([working_bins, na_bin], ignore_index=True)
        
        if 'bad_rate_calc' in working_bins.columns:
            working_bins = working_bins.drop('bad_rate_calc', axis=1)
        
        working_bins = update_bin_stats(working_bins)
        working_bins = add_total_row(working_bins, var)
        
        new_bins = new_bins[new_bins['var'] != var]
        new_bins = pd.concat([new_bins, working_bins], ignore_index=True)
        
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
    """Create binned columns in the DataFrame based on binning rules."""
    result_df = df.copy()
    
    for var in x_vars:
        var_bins = bin_result.bin[(bin_result.bin['var'] == var) & 
                                   (bin_result.bin['bin'] != 'Total')]
        
        if var_bins.empty:
            continue
        
        new_col = prefix + var
        result_df[new_col] = None
        
        na_rule = None
        
        for _, row in var_bins.iterrows():
            rule = row['bin']
            bin_value = rule.replace(var, '').replace(' %in% c', '').strip()
            
            if '| is.na' in rule:
                na_rule = bin_value
                main_rule = rule.split('|')[0].strip()
            else:
                main_rule = rule
            
            try:
                is_na_bin = False
                if 'is.na' in main_rule and '|' not in main_rule:
                    # Standalone NA bin - handle NA rows
                    mask = result_df[var].isna()
                    is_na_bin = True
                elif '%in%' in main_rule:
                    values = _parse_factor_values_from_rule(main_rule)
                    mask = result_df[var].isin(values)
                elif '<=' in main_rule and '>' in main_rule:
                    nums = _parse_numeric_from_rule(main_rule)
                    if len(nums) >= 2:
                        lower, upper = min(nums), max(nums)
                        mask = (result_df[var] > lower) & (result_df[var] <= upper)
                    else:
                        continue
                elif '<=' in main_rule:
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        upper = max(nums)
                        mask = result_df[var] <= upper
                    else:
                        continue
                elif '>' in main_rule:
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        lower = min(nums)
                        mask = result_df[var] > lower
                    else:
                        continue
                elif '==' in main_rule:
                    nums = _parse_numeric_from_rule(main_rule)
                    if nums:
                        result_df.loc[result_df[var] == nums[0], new_col] = bin_value
                    continue
                else:
                    continue
                
                # For NA bins, apply mask directly; for other bins, exclude NA values
                if is_na_bin:
                    result_df.loc[mask, new_col] = bin_value
                else:
                    result_df.loc[mask & result_df[var].notna(), new_col] = bin_value
                
            except Exception:
                continue
        
        if na_rule is not None:
            result_df.loc[result_df[var].isna(), new_col] = na_rule
        elif result_df[var].isna().any():
            na_bins = var_bins[var_bins['bin'].str.match(r'^is\.na\(', na=False)]
            if not na_bins.empty:
                bin_value = na_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
                result_df.loc[result_df[var].isna(), new_col] = bin_value
        
        # Handle any remaining unassigned rows (edge cases that didn't match any rule)
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
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
    """Add WOE columns to the DataFrame by joining with binning rules.
    
    Missing/unmatched bin values are assigned WOE=0 (neutral - no information).
    """
    result_df = df.copy()
    
    for var in x_vars:
        var_bins = bins_df[(bins_df['var'] == var) & (bins_df['bin'] != 'Total')].copy()
        
        if var_bins.empty:
            continue
        
        if 'woe' not in var_bins.columns:
            var_bins['woe'] = calculate_woe(var_bins['goods'].values, var_bins['bads'].values)
        
        var_bins['binValue'] = var_bins['bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
        
        bin_col = prefix + var
        woe_col = woe_prefix + var
        
        if bin_col in result_df.columns:
            woe_map = dict(zip(var_bins['binValue'], var_bins['woe']))
            result_df[woe_col] = result_df[bin_col].map(woe_map)
            
            # Check for any unmapped bin values - this indicates a bug that must be fixed
            missing_woe_count = result_df[woe_col].isna().sum()
            if missing_woe_count > 0:
                unmapped_bins = result_df.loc[result_df[woe_col].isna(), bin_col].unique()
                print(f"[ERROR] {var}: {missing_woe_count} rows have unmapped bin values!")
                print(f"        Unmapped bins: {list(unmapped_bins)}")
                print(f"        Available bins in woe_map: {list(woe_map.keys())}")
                # Try to find and assign the correct WOE for unmapped bins
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

def create_woe_editor_app(df: pd.DataFrame, min_prop: float = 0.05):
    """Create the WOE Editor Shiny application."""
    
    app_results = {
        'df_with_woe': None,
        'df_only_woe': None,
        'bins': None,
        'dv': None,
        'completed': False
    }
    
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css?family=Raleway');
                body { font-family: 'Raleway', sans-serif; background-color: #f5f5f5; }
                .card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .btn-primary { background-color: #2E86AB; border-color: #2E86AB; }
                .btn-success { background-color: #28A745; border-color: #28A745; }
                .btn-danger { background-color: #DC3545; border-color: #DC3545; }
                .btn-secondary { background-color: #6C757D; border-color: #6C757D; }
                .btn-dark { background-color: #343A40; border-color: #343A40; }
                .btn-info { background-color: #17A2B8; border-color: #17A2B8; }
                h4 { font-weight: bold; text-align: center; margin: 20px 0; color: #2E86AB; }
                .divider { width: 10px; display: inline-block; }
                .algorithm-badge { 
                    background: linear-gradient(135deg, #2E86AB, #1A5276);
                    color: white; 
                    padding: 5px 15px; 
                    border-radius: 20px; 
                    font-size: 12px;
                    margin-left: 10px;
                }
            """)
        ),
        
        ui.h4(
            "WOE Editor",
            ui.span("ChiMerge + Monotonic", class_="algorithm-badge")
        ),
        
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=list(df.columns),
                                   selected=df.columns[0] if len(df.columns) > 0 else None)
                ),
                ui.column(6,
                    ui.input_select("tc", "Target Category", choices=[])
                )
            )
        ),
        
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(6,
                    ui.input_select("iv", "Independent Variable", choices=[]),
                    ui.div(
                        ui.input_action_button("prev_btn", "← Previous", class_="btn btn-secondary"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("next_btn", "Next →", class_="btn btn-success"),
                    )
                ),
                ui.column(6,
                    ui.div(
                        ui.input_action_button("group_na_btn", "Group NA", class_="btn btn-primary"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("break_btn", "Break Bin", class_="btn btn-danger"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("reset_btn", "Reset", class_="btn btn-danger"),
                    ),
                    ui.br(),
                    ui.div(
                        ui.input_action_button("optimize_btn", "Optimize", class_="btn btn-dark"),
                        ui.span(" ", class_="divider"),
                        ui.input_action_button("optimize_all_btn", "Optimize All", class_="btn btn-dark"),
                    ),
                )
            )
        ),
        
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px; overflow-y: auto;"},
                    ui.h5("Bin Details"),
                    ui.output_data_frame("woe_table")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 450px;"},
                    ui.h5("WOE & Bad Rate"),
                    output_widget("woe_graph")
                )
            )
        ),
        
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
        
        ui.div(
            {"class": "card"},
            ui.h5("Measurements"),
            ui.output_data_frame("measurements_table")
        ),
        
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run_btn", "Run & Close", class_="btn btn-success btn-lg"),
        ),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        bins_rv = reactive.Value(None)
        all_bins_rv = reactive.Value(None)
        all_bins_mod_rv = reactive.Value(None)
        modified_action_rv = reactive.Value(False)
        initial_bins_rv = reactive.Value(None)
        
        @reactive.Effect
        @reactive.event(input.dv)
        def update_tc():
            dv = input.dv()
            if dv and dv in df.columns:
                unique_vals = df[dv].dropna().unique().tolist()
                ui.update_select("tc", choices=unique_vals, 
                               selected=max(unique_vals) if unique_vals else None)
                
                iv_list = [col for col in df.columns if col != dv]
                ui.update_select("iv", choices=iv_list, 
                               selected=iv_list[0] if iv_list else None)
        
        @reactive.Effect
        @reactive.event(input.tc)
        def init_bins():
            dv = input.dv()
            tc = input.tc()
            
            if not dv or dv not in df.columns:
                return
            
            iv_list = [col for col in df.columns if col != dv]
            
            # Use ChiMerge algorithm with stricter constraints
            result = get_bins(df, dv, iv_list, min_prop=min_prop, enforce_monotonic=True)
            all_bins_rv.set(result)
            all_bins_mod_rv.set(result)
            initial_bins_rv.set(result)
            
            if input.iv() and input.iv() in iv_list:
                var = input.iv()
                var_bin = result.bin[result.bin['var'] == var].copy()
                bins_rv.set(var_bin)
        
        @reactive.Effect
        @reactive.event(input.iv)
        def update_var_bins():
            var = input.iv()
            all_bins = all_bins_mod_rv.get()
            
            if var and all_bins is not None:
                var_bin = all_bins.bin[all_bins.bin['var'] == var].copy()
                bins_rv.set(var_bin)
        
        @reactive.Effect
        @reactive.event(input.prev_btn)
        def go_prev():
            dv = input.dv()
            current = input.iv()
            iv_list = [col for col in df.columns if col != dv]
            
            if current in iv_list:
                idx = iv_list.index(current)
                new_idx = (idx - 1) % len(iv_list)
                ui.update_select("iv", selected=iv_list[new_idx])
        
        @reactive.Effect
        @reactive.event(input.next_btn)
        def go_next():
            dv = input.dv()
            current = input.iv()
            iv_list = [col for col in df.columns if col != dv]
            
            if current in iv_list:
                idx = iv_list.index(current)
                new_idx = (idx + 1) % len(iv_list)
                ui.update_select("iv", selected=iv_list[new_idx])
        
        @reactive.Effect
        @reactive.event(input.group_na_btn)
        def do_group_na():
            var = input.iv()
            all_bins = all_bins_mod_rv.get()
            
            if var and all_bins is not None:
                new_bins = na_combine(all_bins, var)
                all_bins_mod_rv.set(new_bins)
                var_bin = new_bins.bin[new_bins.bin['var'] == var].copy()
                bins_rv.set(var_bin)
                modified_action_rv.set(True)
        
        @reactive.Effect
        @reactive.event(input.break_btn)
        def do_break():
            var = input.iv()
            dv = input.dv()
            all_bins = all_bins_mod_rv.get()
            
            if var and dv and all_bins is not None:
                new_bins = break_bin(all_bins, var, dv, df)
                all_bins_mod_rv.set(new_bins)
                var_bin = new_bins.bin[new_bins.bin['var'] == var].copy()
                bins_rv.set(var_bin)
                modified_action_rv.set(True)
        
        @reactive.Effect
        @reactive.event(input.reset_btn)
        def do_reset():
            initial = initial_bins_rv.get()
            if initial is not None:
                all_bins_mod_rv.set(initial)
                var = input.iv()
                if var:
                    var_bin = initial.bin[initial.bin['var'] == var].copy()
                    bins_rv.set(var_bin)
                modified_action_rv.set(False)
        
        @reactive.Effect
        @reactive.event(input.optimize_btn)
        def do_optimize():
            var = input.iv()
            all_bins = all_bins_mod_rv.get()
            
            if var and all_bins is not None:
                var_summary = all_bins.var_summary
                var_row = var_summary[var_summary['var'] == var]
                
                if not var_row.empty:
                    trend = var_row.iloc[0]['trend']
                    if trend == 'D':
                        new_bins = force_decr_trend(all_bins, var)
                    else:
                        new_bins = force_incr_trend(all_bins, var)
                    
                    all_bins_mod_rv.set(new_bins)
                    var_bin = new_bins.bin[new_bins.bin['var'] == var].copy()
                    bins_rv.set(var_bin)
                    modified_action_rv.set(True)
        
        @reactive.Effect
        @reactive.event(input.optimize_all_btn)
        def do_optimize_all():
            all_bins = all_bins_mod_rv.get()
            
            if all_bins is not None:
                # First combine NAs
                bins_mod = na_combine(all_bins, all_bins.var_summary['var'].tolist())
                
                # Force decreasing trend for D variables
                decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
                if decr_vars:
                    bins_mod = force_decr_trend(bins_mod, decr_vars)
                
                # Force increasing trend for I variables
                incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
                if incr_vars:
                    bins_mod = force_incr_trend(bins_mod, incr_vars)
                
                all_bins_mod_rv.set(bins_mod)
                
                var = input.iv()
                if var:
                    var_bin = bins_mod.bin[bins_mod.bin['var'] == var].copy()
                    bins_rv.set(var_bin)
                
                modified_action_rv.set(True)
        
        @reactive.Effect
        @reactive.event(input.run_btn)
        def do_run():
            all_bins = all_bins_mod_rv.get()
            dv = input.dv()
            
            if all_bins is not None and dv:
                rules = all_bins.bin[all_bins.bin['bin'] != 'Total'].copy()
                rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values,
                                            use_shrinkage=BinningConfig.USE_SHRINKAGE,
                                            shrinkage_strength=BinningConfig.SHRINKAGE_STRENGTH)
                
                for var in all_bins.var_summary['var'].tolist():
                    var_mask = rules['var'] == var
                    rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
                        lambda x: x.replace(var, '').replace(' %in% c', '').strip()
                    )
                
                all_vars = all_bins.var_summary['var'].tolist()
                df_with_bins = create_binned_columns(all_bins, df, all_vars)
                df_woe = add_woe_columns(df_with_bins, rules, all_vars)
                
                woe_cols = [col for col in df_woe.columns if col.startswith('WOE_')]
                df_only = df_woe[woe_cols + [dv]].copy()
                
                app_results['df_with_woe'] = df_woe
                app_results['df_only_woe'] = df_only
                app_results['bins'] = rules
                app_results['dv'] = dv
                app_results['completed'] = True
                
                import asyncio
                asyncio.get_event_loop().call_soon(session.close)
        
        @output
        @render.data_frame
        def woe_table():
            var_bin = bins_rv.get()
            if var_bin is not None and not var_bin.empty:
                display_df = var_bin[['bin', 'count', 'bads', 'goods', 'propn', 'bad_rate']].copy()
                display_df.columns = ['Bin', 'Count', 'Bads', 'Goods', 'Propn%', 'BadRate%']
                return render.DataGrid(display_df, width="100%")
            return render.DataGrid(pd.DataFrame())
        
        @output
        @render_plotly
        def woe_graph():
            var_bin = bins_rv.get()
            if var_bin is None or var_bin.empty:
                return go.Figure()
            
            plot_data = var_bin[var_bin['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            woe_vals = calculate_woe(plot_data['goods'].values, plot_data['bads'].values,
                                    use_shrinkage=BinningConfig.USE_SHRINKAGE)
            plot_data['woe'] = woe_vals
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(len(plot_data))),
                y=plot_data['bad_rate'],
                name='Bad Rate %',
                marker_color='#2E86AB',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(plot_data))),
                y=plot_data['woe'],
                name='WOE',
                mode='lines+markers',
                marker_color='#E74C3C',
                yaxis='y2'
            ))
            
            fig.update_layout(
                yaxis=dict(title='Bad Rate %', side='left'),
                yaxis2=dict(title='WOE', side='right', overlaying='y'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=50, r=50, t=30, b=30),
                height=380
            )
            
            return fig
        
        @output
        @render_plotly
        def count_bar():
            var_bin = bins_rv.get()
            if var_bin is None or var_bin.empty:
                return go.Figure()
            
            plot_data = var_bin[var_bin['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(len(plot_data))),
                y=plot_data['goods'],
                name='Goods',
                marker_color='#28A745'
            ))
            
            fig.add_trace(go.Bar(
                x=list(range(len(plot_data))),
                y=plot_data['bads'],
                name='Bads',
                marker_color='#DC3545'
            ))
            
            fig.update_layout(
                barmode='stack',
                title='Good/Bad Counts',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=50, r=50, t=50, b=30),
                height=300
            )
            
            return fig
        
        @output
        @render_plotly
        def prop_bar():
            var_bin = bins_rv.get()
            if var_bin is None or var_bin.empty:
                return go.Figure()
            
            plot_data = var_bin[var_bin['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            total_goods = plot_data['goods'].sum()
            total_bads = plot_data['bads'].sum()
            
            plot_data['goodCap'] = plot_data['goods'] / total_goods * 100 if total_goods > 0 else 0
            plot_data['badCap'] = plot_data['bads'] / total_bads * 100 if total_bads > 0 else 0
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(len(plot_data))),
                y=plot_data['goodCap'],
                name='Good %',
                marker_color='#28A745'
            ))
            
            fig.add_trace(go.Bar(
                x=list(range(len(plot_data))),
                y=plot_data['badCap'],
                name='Bad %',
                marker_color='#DC3545'
            ))
            
            fig.update_layout(
                barmode='group',
                title='Good/Bad Distribution %',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=50, r=50, t=50, b=30),
                height=300
            )
            
            return fig
        
        @output
        @render.data_frame
        def measurements_table():
            all_bins = all_bins_mod_rv.get()
            if all_bins is not None and not all_bins.var_summary.empty:
                display_df = all_bins.var_summary[['var', 'iv', 'ent', 'trend', 'monTrend', 'numBins']].copy()
                display_df.columns = ['Variable', 'IV', 'Entropy', 'Trend', 'Monotonic', 'Bins']
                display_df = display_df.sort_values('IV', ascending=False)
                return render.DataGrid(display_df, width="100%")
            return render.DataGrid(pd.DataFrame())
    
    return App(app_ui, server), app_results


def run_woe_editor(df: pd.DataFrame, min_prop: float = 0.05) -> Dict[str, Any]:
    """Run the WOE editor interactively and return results."""
    import socket
    import webbrowser
    
    # Find available port (avoid conflicts with parallel instances)
    port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
    
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            break
        except OSError:
            port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
    
    log_progress(f"Starting Shiny UI on port {port} (Instance: {INSTANCE_ID})")
    
    app, results = create_woe_editor_app(df, min_prop)
    
    # Open browser
    webbrowser.open(f'http://127.0.0.1:{port}')
    
    # Run app
    app.run(host='127.0.0.1', port=port)
    
    return results


# =============================================================================
# Read Input Data
# =============================================================================

df = knio.input_tables[0].to_pandas()

# Default min_prop to BinningConfig value (fraud-friendly 2%)
min_prop = BinningConfig.MIN_BIN_PCT

# =============================================================================
# Check for Flow Variables
# =============================================================================
contains_dv = False
dv = None
target = None
optimize_all = False
group_na = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
except:
    pass

try:
    group_na = knio.flow_variables.get("GroupNA", False)
except:
    pass

# New flow variables for fraud model tuning
min_bin_pct = BinningConfig.MIN_BIN_PCT
min_bin_count = BinningConfig.MIN_BIN_COUNT

try:
    # MinBinPct: Minimum percentage of observations per bin (0.01 = 1%, 0.05 = 5%)
    min_bin_pct_fv = knio.flow_variables.get("MinBinPct", None)
    if min_bin_pct_fv is not None:
        min_bin_pct = float(min_bin_pct_fv)
        BinningConfig.MIN_BIN_PCT = min_bin_pct
except:
    pass

try:
    # MinBinCount: Minimum absolute count per bin (default 30)
    min_bin_count_fv = knio.flow_variables.get("MinBinCount", None)
    if min_bin_count_fv is not None:
        min_bin_count = int(min_bin_count_fv)
        BinningConfig.MIN_BIN_COUNT = min_bin_count
except:
    pass

try:
    # UseShrinkage: Apply shrinkage to WOE values (recommended for fraud)
    use_shrinkage_fv = knio.flow_variables.get("UseShrinkage", None)
    if use_shrinkage_fv is not None:
        BinningConfig.USE_SHRINKAGE = bool(use_shrinkage_fv)
except:
    pass

try:
    # Algorithm: "DecisionTree" (R-compatible), "ChiMerge", or "IVOptimal"
    algorithm_fv = knio.flow_variables.get("Algorithm", None)
    if algorithm_fv is not None and algorithm_fv in ["DecisionTree", "ChiMerge", "IVOptimal"]:
        BinningConfig.ALGORITHM = algorithm_fv
except:
    pass

try:
    # UseEnhancements: Master switch - enables ALL advanced enhancements (default False for R-compatibility)
    use_enhancements_fv = knio.flow_variables.get("UseEnhancements", None)
    if use_enhancements_fv is not None:
        BinningConfig.USE_ENHANCEMENTS = bool(use_enhancements_fv)
        # When master switch is enabled, enable all individual enhancements
        if BinningConfig.USE_ENHANCEMENTS:
            BinningConfig.ADAPTIVE_MIN_PROP = True
            BinningConfig.MIN_EVENT_COUNT = True
            BinningConfig.AUTO_RETRY = True
            BinningConfig.CHI_SQUARE_VALIDATION = True
            BinningConfig.SINGLE_BIN_PROTECTION = True
except:
    pass

# =============================================================================
# Individual Enhancement Flow Variables (can override master switch)
# =============================================================================

try:
    # AdaptiveMinProp: Relaxes min_prop for sparse data (< 500 samples)
    adaptive_min_prop_fv = knio.flow_variables.get("AdaptiveMinProp", None)
    if adaptive_min_prop_fv is not None:
        BinningConfig.ADAPTIVE_MIN_PROP = bool(adaptive_min_prop_fv)
except:
    pass

try:
    # MinEventCount: Ensures at least N events per potential bin
    min_event_count_fv = knio.flow_variables.get("MinEventCount", None)
    if min_event_count_fv is not None:
        BinningConfig.MIN_EVENT_COUNT = bool(min_event_count_fv)
except:
    pass

try:
    # AutoRetry: Retry with relaxed constraints if no splits found
    auto_retry_fv = knio.flow_variables.get("AutoRetry", None)
    if auto_retry_fv is not None:
        BinningConfig.AUTO_RETRY = bool(auto_retry_fv)
except:
    pass

try:
    # ChiSquareValidation: Merge statistically similar bins post-binning
    chi_square_validation_fv = knio.flow_variables.get("ChiSquareValidation", None)
    if chi_square_validation_fv is not None:
        BinningConfig.CHI_SQUARE_VALIDATION = bool(chi_square_validation_fv)
except:
    pass

try:
    # SingleBinProtection: Prevent na_combine from creating single-bin vars (default ON)
    single_bin_protection_fv = knio.flow_variables.get("SingleBinProtection", None)
    if single_bin_protection_fv is not None:
        BinningConfig.SINGLE_BIN_PROTECTION = bool(single_bin_protection_fv)
except:
    pass

try:
    # MaxCategories: Maximum unique categories for categorical variables (prevents high-cardinality issues)
    max_categories_fv = knio.flow_variables.get("MaxCategories", None)
    if max_categories_fv is not None:
        BinningConfig.MAX_CATEGORIES = int(max_categories_fv)
except:
    pass

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
    log_progress("=" * 60)
    
    # Determine if any enhancements are active
    any_enhancement = (BinningConfig.USE_ENHANCEMENTS or 
                       BinningConfig.ADAPTIVE_MIN_PROP or 
                       BinningConfig.MIN_EVENT_COUNT or 
                       BinningConfig.AUTO_RETRY or 
                       BinningConfig.CHI_SQUARE_VALIDATION)
    
    if BinningConfig.ALGORITHM == "DecisionTree":
        algo_name = "DecisionTree (R-compatible)" if not any_enhancement else "DecisionTree (Enhanced)"
    else:
        algo_name = "ChiMerge"
    log_progress(f"WOE EDITOR - HEADLESS MODE ({algo_name})")
    log_progress("=" * 60)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"Algorithm: {BinningConfig.ALGORITHM}")
    log_progress(f"OptimizeAll: {optimize_all}, GroupNA: {group_na}")
    log_progress(f"MinBinPct: {BinningConfig.MIN_BIN_PCT:.1%}, MaxBins: {BinningConfig.MAX_BINS}")
    log_progress(f"MaxCategories: {BinningConfig.MAX_CATEGORIES} (categorical variables with more unique values will be skipped)")
    if BinningConfig.ALGORITHM == "ChiMerge":
        log_progress(f"ChiMergeThreshold: {BinningConfig.CHI_MERGE_THRESHOLD}, MinBinCount: {BinningConfig.MIN_BIN_COUNT}")
    log_progress(f"UseShrinkage: {BinningConfig.USE_SHRINKAGE}")
    
    # Log individual enhancement settings
    log_progress("Enhancement Settings:")
    if BinningConfig.USE_ENHANCEMENTS:
        log_progress(f"  UseEnhancements: True (master switch - all enhancements enabled)")
    else:
        log_progress(f"  UseEnhancements: False (individual settings below apply)")
    log_progress(f"  AdaptiveMinProp: {BinningConfig.ADAPTIVE_MIN_PROP}")
    log_progress(f"  MinEventCount: {BinningConfig.MIN_EVENT_COUNT}")
    log_progress(f"  AutoRetry: {BinningConfig.AUTO_RETRY}")
    log_progress(f"  ChiSquareValidation: {BinningConfig.CHI_SQUARE_VALIDATION}")
    log_progress(f"  SingleBinProtection: {BinningConfig.SINGLE_BIN_PROTECTION}")
    
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter out constant/zero-variance variables (only 1 unique value)
    constant_vars = []
    valid_vars = []
    for col in iv_list:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_vars.append(col)
        else:
            valid_vars.append(col)
    
    if constant_vars:
        log_progress(f"Removed {len(constant_vars)} constant variables (only 1 unique value)")
        if len(constant_vars) <= 10:
            log_progress(f"  Constant vars: {constant_vars}")
        else:
            log_progress(f"  First 10: {constant_vars[:10]}...")
    
    iv_list = valid_vars
    log_progress(f"Variables to process: {len(iv_list)}")
    
    # Step 1: Initial binning
    step_start = time.time()
    if BinningConfig.ALGORITHM == "DecisionTree":
        algo_desc = "DecisionTree (R-compatible)" if not any_enhancement else "DecisionTree (Enhanced)"
    else:
        algo_desc = "ChiMerge"
    log_progress(f"STEP 1/6: Computing initial bins ({algo_desc})...")
    bins_result = get_bins(
        df, dv, iv_list, 
        min_prop=BinningConfig.MIN_BIN_PCT, 
        max_bins=BinningConfig.MAX_BINS,
        enforce_monotonic=True, 
        algorithm=BinningConfig.ALGORITHM,
        use_enhancements=BinningConfig.USE_ENHANCEMENTS,
        adaptive_min_prop=BinningConfig.ADAPTIVE_MIN_PROP,
        min_event_count=BinningConfig.MIN_EVENT_COUNT,
        auto_retry=BinningConfig.AUTO_RETRY
    )
    log_progress(f"STEP 1/6 complete in {format_time(time.time() - step_start)}")
    
    # Step 2: Merge pure bins (always - prevents infinite WOE)
    step_start = time.time()
    if 'purNode' in bins_result.var_summary.columns:
        # purNode is 'Y' or 'N' - count variables with pure bins
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    if pure_count > 0:
        log_progress(f"STEP 2/6: Merging {int(pure_count)} pure bins (prevents infinite WOE)...")
        bins_result = merge_pure_bins(bins_result)
        log_progress(f"STEP 2/6 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 2/6: Skipped (no pure bins found)")
    
    # Step 3: Chi-square validation (ENHANCEMENT - only if enabled)
    if BinningConfig.CHI_SQUARE_VALIDATION:
        step_start = time.time()
        log_progress("STEP 3/6: Validating bins with chi-square test (merge similar bins)...")
        bins_result = validate_bins_chi_square(bins_result, p_value_threshold=0.10)
        log_progress(f"STEP 3/6 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 3/6: Skipped (ChiSquareValidation=False)")
    
    # Step 4: Group NA (optional)
    if group_na:
        step_start = time.time()
        log_progress("STEP 4/6: Grouping NA values...")
        if BinningConfig.SINGLE_BIN_PROTECTION:
            log_progress("  (SingleBinProtection enabled - will skip vars that would become single-bin)")
        bins_result = na_combine(
            bins_result, 
            bins_result.var_summary['var'].tolist(),
            prevent_single_bin=BinningConfig.SINGLE_BIN_PROTECTION
        )
        log_progress(f"STEP 4/6 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 4/6: Skipped (GroupNA=False)")
    
    # Step 5: Optimize All (optional)
    if optimize_all:
        step_start = time.time()
        log_progress("STEP 5/6: Optimizing monotonicity for all variables...")
        bins_mod = na_combine(
            bins_result, 
            bins_result.var_summary['var'].tolist(),
            prevent_single_bin=BinningConfig.SINGLE_BIN_PROTECTION
        )
        
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            log_progress(f"  - Forcing decreasing trend on {len(decr_vars)} variables...")
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            log_progress(f"  - Forcing increasing trend on {len(incr_vars)} variables...")
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
        log_progress(f"STEP 5/6 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 5/6: Skipped (OptimizeAll=False)")
    
    # Step 6: Apply WOE transformation
    step_start = time.time()
    log_progress("STEP 6/6: Applying WOE transformation to data...")
    
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values,
                                use_shrinkage=BinningConfig.USE_SHRINKAGE,
                                shrinkage_strength=BinningConfig.SHRINKAGE_STRENGTH)
    
    for var in bins_result.var_summary['var'].tolist():
        var_mask = rules['var'] == var
        rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
    
    all_vars = bins_result.var_summary['var'].tolist()
    log_progress(f"  - Creating binned columns for {len(all_vars)} variables...")
    df_with_bins = create_binned_columns(bins_result, df, all_vars)
    log_progress(f"  - Adding WOE columns...")
    df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
    
    woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
    df_only_woe = df_with_woe[woe_cols + [dv]].copy()
    
    bins = rules
    
    log_progress(f"STEP 6/6 complete in {format_time(time.time() - step_start)}")
    
    # Diagnostic: Identify problematic variables
    log_progress("=" * 60)
    log_progress("DIAGNOSTICS:")
    
    # Variables with single bin (WOE will always be 0)
    single_bin_vars = []
    for var in all_vars:
        var_bins = bins[bins['var'] == var]
        if len(var_bins) <= 1:
            single_bin_vars.append(var)
    
    if single_bin_vars:
        log_progress(f"  [WARN] {len(single_bin_vars)} variables have SINGLE BIN (WOE=0):")
        if len(single_bin_vars) <= 20:
            for v in single_bin_vars[:20]:
                # Get null rate for this variable
                null_rate = df[v].isna().mean() * 100
                log_progress(f"    - {v} (null rate: {null_rate:.1f}%)")
        else:
            log_progress(f"    First 10: {single_bin_vars[:10]}")
            log_progress(f"    ... and {len(single_bin_vars) - 10} more")
    
    # Variables with all WOE=0
    zero_woe_vars = []
    for var in all_vars:
        var_bins = bins[bins['var'] == var]
        if 'woe' in var_bins.columns and (var_bins['woe'].abs() < 0.0001).all():
            if var not in single_bin_vars:  # Don't double-count
                zero_woe_vars.append(var)
    
    if zero_woe_vars:
        log_progress(f"  [WARN] {len(zero_woe_vars)} additional variables have ALL WOE ≈ 0:")
        if len(zero_woe_vars) <= 10:
            log_progress(f"    {zero_woe_vars}")
        else:
            log_progress(f"    First 10: {zero_woe_vars[:10]}")
    
    # Variables with very low IV (< 0.02)
    low_iv_count = (bins_result.var_summary['iv'] < 0.02).sum()
    if low_iv_count > 0:
        log_progress(f"  [INFO] {low_iv_count} variables have IV < 0.02 (weak predictive power)")
    
    # High null rate variables
    high_null_vars = []
    for var in all_vars:
        null_rate = df[var].isna().mean()
        if null_rate > 0.80:  # More than 80% null
            high_null_vars.append((var, null_rate))
    
    if high_null_vars:
        log_progress(f"  [INFO] {len(high_null_vars)} variables have >80% null values")
        if len(high_null_vars) <= 10:
            for v, nr in high_null_vars[:10]:
                log_progress(f"    - {v}: {nr:.1%} null")
    
    # Summary stats
    log_progress("=" * 60)
    mono_count = bins_result.var_summary['monTrend'].value_counts().get('Y', 0)
    log_progress(f"COMPLETE: Processed {len(all_vars)} variables")
    log_progress(f"Monotonic variables: {mono_count}/{len(all_vars)} ({100*mono_count/len(all_vars):.1f}%)")
    if single_bin_vars or zero_woe_vars:
        log_progress(f"[!] Potential issues: {len(single_bin_vars)} single-bin, {len(zero_woe_vars)} zero-WOE vars")
        log_progress(f"    Suggestions: Try lower MinBinPct (e.g., 0.01) or MinBinCount (e.g., 20)")
    log_progress("=" * 60)

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    print("Running in interactive mode - launching Shiny UI...")
    print("Algorithm: ChiMerge + Monotonic Optimization (Advanced)")
    
    results = run_woe_editor(df, min_prop=min_prop)
    
    if results['completed']:
        df_with_woe = results['df_with_woe']
        df_only_woe = results['df_only_woe']
        bins = results['bins']
        dv = results['dv']
        print("Interactive session completed successfully")
    else:
        print("Interactive session cancelled - returning empty results")
        df_with_woe = df.copy()
        df_only_woe = pd.DataFrame()
        bins = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

# Output 1: Original input DataFrame (unchanged)
knio.output_tables[0] = knio.Table.from_pandas(df)

# Output 2: df_with_woe - Original data + binned columns + WOE columns
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)

# Output 3: df_only_woe - Only WOE columns + dependent variable
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)

# Output 4: df_only_bins - ONLY binned columns (b_*) for scorecard scoring
# Extract only b_* columns from df_with_woe for lean scorecard input
b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
df_only_bins = df_with_woe[b_columns].copy()
knio.output_tables[3] = knio.Table.from_pandas(df_only_bins)

# Output 5: bins - Binning rules with WOE values (metadata)
knio.output_tables[4] = knio.Table.from_pandas(bins)

log_progress("=" * 60)
log_progress(f"OUTPUT SUMMARY:")
log_progress(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
log_progress(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
log_progress(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
log_progress(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols) ** USE FOR SCORECARD **")
log_progress(f"  Port 5: Bin Rules ({len(bins)} rows - metadata)")
log_progress("=" * 60)

print("WOE Editor (Advanced) completed successfully")

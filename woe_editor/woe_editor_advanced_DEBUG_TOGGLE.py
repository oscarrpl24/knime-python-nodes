# =============================================================================
# WOE Editor for KNIME - ADVANCED VERSION - DEBUG TOGGLE
# =============================================================================
# Python implementation using state-of-the-art optimal binning algorithms
# based on academic research and industry best practices.
#
# DEBUG TOGGLE VERSION: Debug logging can be enabled/disabled via DEBUG_MODE
#
# Compatible with KNIME 5.9, Python 3.9
#
# ALGORITHM OPTIONS:
# - "DecisionTree" (default): Uses CART decision tree - matches R's logiBin::getBins
# - "ChiMerge": Uses chi-square based bin merging (more conservative)
# - "IVOptimal": Directly maximizes Information Value (allows non-monotonic patterns)
#
# Release Date: 2026-01-28
# Version: 1.4-DEBUG-TOGGLE
# =============================================================================

# =============================================================================
# DEBUG MODE TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG_MODE = True
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
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING SETUP (Toggleable)
# =============================================================================

def setup_debug_logging():
    """Configure debug logging with detailed formatting (respects DEBUG_MODE)."""
    logger = logging.getLogger('WOE_ADVANCED_DEBUG')
    logger.handlers.clear()
    
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
    
    return logger

logger = setup_debug_logging()

def log_function_entry(func_name, **kwargs):
    """Log function entry with parameters (only when DEBUG_MODE is True)."""
    if not DEBUG_MODE:
        return
    params_str = ', '.join([f"{k}={repr(v)[:80]}" for k, v in kwargs.items()])
    logger.debug(f">>> ENTER {func_name}({params_str})")

def log_function_exit(func_name, result=None, duration=None):
    """Log function exit with result (only when DEBUG_MODE is True)."""
    if not DEBUG_MODE:
        return
    result_str = repr(result) if result is not None else "None"
    if len(result_str) > 150:
        result_str = result_str[:150] + "..."
    duration_str = f" [{duration:.3f}s]" if duration else ""
    logger.debug(f"<<< EXIT {func_name}{duration_str} -> {result_str}")

def log_exception(func_name, e):
    """Log exception with full traceback (always logs, regardless of DEBUG_MODE)."""
    logger.error(f"!!! EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
BASE_PORT = 8055
RANDOM_PORT_RANGE = 1000

INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
if DEBUG_MODE:
    logger.info(f"Instance ID: {INSTANCE_ID}")
    logger.info(f"Process ID: {os.getpid()}")

os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
if DEBUG_MODE:
    logger.debug("Thread environment variables set to 1")

# =============================================================================
# Progress Logging Utilities
# =============================================================================

def log_progress(message: str, flush: bool = True):
    """Print a progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    if DEBUG_MODE:
        logger.info(message)
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

if DEBUG_MODE:
    logger.info("Loading dependencies...")

try:
    from scipy import stats
    if DEBUG_MODE:
        logger.debug("scipy.stats imported successfully")
except ImportError:
    if DEBUG_MODE:
        logger.warning("scipy not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    from scipy import stats

try:
    from sklearn.tree import DecisionTreeClassifier
    if DEBUG_MODE:
        logger.debug("sklearn imported successfully")
except ImportError:
    if DEBUG_MODE:
        logger.warning("scikit-learn not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    if DEBUG_MODE:
        logger.debug("Shiny and plotly imported successfully")
except ImportError:
    if DEBUG_MODE:
        logger.warning("Shiny/plotly not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Configuration
# =============================================================================

class BinningConfig:
    """Configuration for advanced binning algorithms - DEBUG TOGGLE VERSION"""
    ALGORITHM = "DecisionTree"
    MIN_BIN_PCT = 0.01
    MIN_BIN_COUNT = 20
    MAX_BINS = 10
    MIN_BINS = 2
    MAX_CATEGORIES = 50
    CHI_MERGE_THRESHOLD = 0.05
    MIN_IV_GAIN = 0.005
    USE_SHRINKAGE = False
    SHRINKAGE_STRENGTH = 0.1
    USE_ENHANCEMENTS = False
    ADAPTIVE_MIN_PROP = False
    MIN_EVENT_COUNT = False
    AUTO_RETRY = False
    CHI_SQUARE_VALIDATION = False
    SINGLE_BIN_PROTECTION = True

if DEBUG_MODE:
    logger.info(f"BinningConfig initialized:")
    logger.info(f"  ALGORITHM: {BinningConfig.ALGORITHM}")
    logger.info(f"  MIN_BIN_PCT: {BinningConfig.MIN_BIN_PCT}")
    logger.info(f"  MAX_BINS: {BinningConfig.MAX_BINS}")
    logger.info(f"  USE_ENHANCEMENTS: {BinningConfig.USE_ENHANCEMENTS}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BinResult:
    """Container for binning results"""
    var_summary: pd.DataFrame
    bin: pd.DataFrame


# =============================================================================
# Core WOE/IV Calculation Functions
# =============================================================================

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray, 
                  use_shrinkage: bool = False, shrinkage_strength: float = 0.1) -> np.ndarray:
    """Calculate Weight of Evidence (WOE) for each bin with optional shrinkage."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('calculate_woe', 
                           freq_good_len=len(freq_good), 
                           freq_bad_len=len(freq_bad),
                           use_shrinkage=use_shrinkage)
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if DEBUG_MODE:
        logger.debug(f"Totals: good={total_good}, bad={total_bad}")
    
    if total_good == 0 or total_bad == 0:
        if DEBUG_MODE:
            logger.warning("Zero total goods or bads")
        result = np.zeros(len(freq_good))
        if DEBUG_MODE:
            log_function_exit('calculate_woe', f"zeros({len(result)})", time.time() - start_time)
        return result
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    epsilon = 0.0001
    dist_good = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad / dist_good)
    
    if use_shrinkage and shrinkage_strength > 0:
        if DEBUG_MODE:
            logger.debug(f"Applying shrinkage with strength: {shrinkage_strength}")
        n_obs = freq_good + freq_bad
        total_obs = n_obs.sum()
        weights = n_obs / (n_obs + shrinkage_strength * total_obs / len(n_obs))
        woe = woe * weights
    
    result = np.round(woe, 5)
    if DEBUG_MODE:
        log_function_exit('calculate_woe', f"array({len(result)})", time.time() - start_time)
    return result


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """Calculate Information Value (IV) for a variable."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('calculate_iv', freq_good_len=len(freq_good), freq_bad_len=len(freq_bad))
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        if DEBUG_MODE:
            log_function_exit('calculate_iv', 0.0, time.time() - start_time)
        return 0.0
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    epsilon = 0.0001
    dist_good_safe = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad_safe / dist_good_safe)
    iv = np.sum((dist_bad - dist_good) * woe)
    
    if not np.isfinite(iv):
        if DEBUG_MODE:
            logger.warning(f"Non-finite IV: {iv}")
        iv = 0.0
    
    result = round(iv, 4)
    if DEBUG_MODE:
        log_function_exit('calculate_iv', result, time.time() - start_time)
    return result


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
    if DEBUG_MODE:
        log_function_entry('get_var_type', series_name=series.name, dtype=str(series.dtype))
    
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique()
        if nunique <= 10:
            result = 'factor'
        else:
            result = 'numeric'
    else:
        result = 'factor'
    
    if DEBUG_MODE:
        log_function_exit('get_var_type', result)
    return result


# =============================================================================
# ChiMerge Algorithm
# =============================================================================

def _chi_square_statistic(bin1_good: int, bin1_bad: int, 
                          bin2_good: int, bin2_bad: int) -> float:
    """Calculate chi-square statistic for two adjacent bins."""
    if DEBUG_MODE:
        log_function_entry('_chi_square_statistic', 
                           bin1=(bin1_good, bin1_bad), 
                           bin2=(bin2_good, bin2_bad))
    
    observed = np.array([[bin1_good, bin1_bad], [bin2_good, bin2_bad]])
    
    if observed.sum() == 0:
        if DEBUG_MODE:
            logger.warning("Zero total observations")
        return np.inf
    
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    
    if total == 0 or any(row_totals == 0) or any(col_totals == 0):
        return np.inf
    
    expected = np.outer(row_totals, col_totals) / total
    
    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i, j] > 0:
                diff = abs(observed[i, j] - expected[i, j]) - 0.5
                diff = max(diff, 0)
                chi2 += (diff ** 2) / expected[i, j]
    
    if DEBUG_MODE:
        logger.debug(f"Chi-square: {chi2:.4f}")
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
    """ChiMerge algorithm for optimal binning."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('_chimerge_get_splits', 
                           x_name=x.name, 
                           min_bin_pct=min_bin_pct, 
                           max_bins=max_bins)
    
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        if DEBUG_MODE:
            log_function_exit('_chimerge_get_splits', [], time.time() - start_time)
        return []
    
    total_count = len(x_clean)
    min_count_required = max(int(total_count * min_bin_pct), min_bin_count)
    if DEBUG_MODE:
        logger.debug(f"Min count required: {min_count_required}")
    
    n_initial_bins = min(100, len(x_clean.unique()))
    
    try:
        initial_bins = pd.qcut(x_clean, q=n_initial_bins, duplicates='drop')
    except ValueError:
        try:
            initial_bins = pd.cut(x_clean, bins=min(20, len(x_clean.unique())), duplicates='drop')
        except:
            if DEBUG_MODE:
                log_function_exit('_chimerge_get_splits', [], time.time() - start_time)
            return []
    
    if hasattr(initial_bins, 'categories') and len(initial_bins.categories) > 0:
        edges = sorted(set(
            [initial_bins.categories[0].left] + 
            [cat.right for cat in initial_bins.categories]
        ))
    else:
        if DEBUG_MODE:
            log_function_exit('_chimerge_get_splits', [], time.time() - start_time)
        return []
    
    if DEBUG_MODE:
        logger.debug(f"Initial edges: {len(edges)}")
    
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
    bins_stats = [b for b in bins_stats if b['count'] > 0]
    
    if DEBUG_MODE:
        logger.debug(f"Initial bins: {len(bins_stats)}")
    
    if len(bins_stats) <= min_bins:
        result = [b['right'] for b in bins_stats[:-1]]
        if DEBUG_MODE:
            log_function_exit('_chimerge_get_splits', f"{len(result)} splits", time.time() - start_time)
        return result
    
    chi2_threshold = stats.chi2.ppf(1 - chi_threshold, df=1)
    if DEBUG_MODE:
        logger.debug(f"Chi-square threshold: {chi2_threshold:.4f}")
    
    merge_count = 0
    while len(bins_stats) > min_bins:
        min_chi2 = np.inf
        merge_idx = -1
        
        for i in range(len(bins_stats) - 1):
            chi2 = _chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        if min_chi2 > chi2_threshold and len(bins_stats) <= max_bins:
            break
        
        if merge_idx >= 0:
            bins_stats[merge_idx] = {
                'left': bins_stats[merge_idx]['left'],
                'right': bins_stats[merge_idx + 1]['right'],
                'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
                'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
                'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
            }
            bins_stats.pop(merge_idx + 1)
            merge_count += 1
        else:
            break
    
    if DEBUG_MODE:
        logger.debug(f"Merged {merge_count} bin pairs")
    
    # Enforce minimum bin size
    changed = True
    while changed and len(bins_stats) > min_bins:
        changed = False
        for i in range(len(bins_stats)):
            if bins_stats[i]['count'] < min_count_required:
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
    
    splits = [b['right'] for b in bins_stats[:-1]]
    if DEBUG_MODE:
        log_function_exit('_chimerge_get_splits', f"{len(splits)} splits", time.time() - start_time)
    return splits


# =============================================================================
# Decision Tree Algorithm (R-compatible)
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
    """Use decision tree (CART) to find optimal split points."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('_get_decision_tree_splits',
                           x_name=x.name,
                           min_prop=min_prop,
                           max_bins=max_bins,
                           use_enhancements=use_enhancements)
    
    use_adaptive = adaptive_min_prop if adaptive_min_prop is not None else use_enhancements
    use_min_events = min_event_count if min_event_count is not None else use_enhancements
    use_auto_retry = auto_retry if auto_retry is not None else use_enhancements
    
    if DEBUG_MODE:
        logger.debug(f"Enhancements: adaptive={use_adaptive}, min_events={use_min_events}, auto_retry={use_auto_retry}")
    
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        if DEBUG_MODE:
            log_function_exit('_get_decision_tree_splits', [], time.time() - start_time)
        return []
    
    n_samples = len(x_clean)
    n_events = int(y_clean.sum())
    event_rate = n_events / n_samples if n_samples > 0 else 0
    
    if DEBUG_MODE:
        logger.debug(f"Samples: {n_samples}, Events: {n_events}, Event rate: {event_rate:.4f}")
    
    effective_min_prop = min_prop
    
    if use_adaptive:
        if n_samples < 500:
            effective_min_prop = max(min_prop / 2, 0.005)
            if DEBUG_MODE:
                logger.debug(f"Adaptive: reduced min_prop to {effective_min_prop}")
        if event_rate < 0.05 and n_events > 0:
            max_possible_bins = max(n_events // min_events, 2)
            min_samples_for_events = n_samples / max_possible_bins
            adaptive_prop = min_samples_for_events / n_samples
            effective_min_prop = max(effective_min_prop, adaptive_prop * 0.8)
            if DEBUG_MODE:
                logger.debug(f"Adaptive: adjusted for low event rate to {effective_min_prop}")
    
    min_samples_leaf = max(int(n_samples * effective_min_prop), 1)
    
    if use_min_events and n_events > 0 and min_events > 0:
        min_samples_for_min_events = int(min_events / max(event_rate, 0.001))
        min_samples_leaf = max(min_samples_leaf, min_samples_for_min_events)
        if DEBUG_MODE:
            logger.debug(f"Min events: adjusted min_samples_leaf to {min_samples_leaf}")
    
    min_samples_leaf = min(min_samples_leaf, n_samples // 2)
    min_samples_leaf = max(min_samples_leaf, 1)
    
    if DEBUG_MODE:
        logger.debug(f"Final min_samples_leaf: {min_samples_leaf}")
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        tree.fit(x_clean, y_clean)
        if DEBUG_MODE:
            logger.debug(f"Tree fitted with {tree.tree_.node_count} nodes")
    except Exception as e:
        log_exception('_get_decision_tree_splits', e)
        return []
    
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    thresholds = sorted(set(thresholds))
    
    if use_auto_retry and len(thresholds) == 0 and min_samples_leaf > 10:
        if DEBUG_MODE:
            logger.debug("Auto-retry: attempting with relaxed constraints")
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
            if DEBUG_MODE:
                logger.debug(f"Retry found {len(thresholds)} splits")
        except Exception as e:
            log_exception('_get_decision_tree_splits:retry', e)
    
    if DEBUG_MODE:
        log_function_exit('_get_decision_tree_splits', f"{len(thresholds)} splits", time.time() - start_time)
    return thresholds


# =============================================================================
# IV-Optimal Algorithm
# =============================================================================

def _iv_optimal_get_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_bin_count: int = 20,
    min_iv_loss: float = 0.001
) -> List[float]:
    """IV-optimal binning algorithm that maximizes Information Value."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('_iv_optimal_get_splits',
                           x_name=x.name,
                           min_prop=min_prop,
                           max_bins=max_bins)
    
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        if DEBUG_MODE:
            log_function_exit('_iv_optimal_get_splits', [], time.time() - start_time)
        return []
    
    n_samples = len(x_clean)
    n_unique = len(np.unique(x_clean))
    total_goods = int((y_clean == 0).sum())
    total_bads = int((y_clean == 1).sum())
    
    if DEBUG_MODE:
        logger.debug(f"Samples: {n_samples}, Unique: {n_unique}, Goods: {total_goods}, Bads: {total_bads}")
    
    if total_goods == 0 or total_bads == 0:
        if DEBUG_MODE:
            log_function_exit('_iv_optimal_get_splits', [], time.time() - start_time)
        return []
    
    if n_unique <= 20:
        initial_splits = sorted(np.unique(x_clean))[:-1]
    else:
        n_initial = min(max(20, n_unique // 5), min(100, n_samples // 50), n_unique - 1)
        try:
            quantiles = np.linspace(0, 100, n_initial + 1)[1:-1]
            initial_splits = list(np.percentile(x_clean, quantiles))
            initial_splits = sorted(set(initial_splits))
        except Exception:
            initial_splits = sorted(np.unique(x_clean))[:-1]
    
    if DEBUG_MODE:
        logger.debug(f"Initial splits: {len(initial_splits)}")
    
    if len(initial_splits) == 0:
        if DEBUG_MODE:
            log_function_exit('_iv_optimal_get_splits', [], time.time() - start_time)
        return []
    
    # Create bins and iteratively merge to maximize IV
    current_splits = list(initial_splits)
    
    while len(current_splits) > max_bins - 1:
        if len(current_splits) <= 1:
            break
        current_splits = current_splits[::2]
    
    if DEBUG_MODE:
        log_function_exit('_iv_optimal_get_splits', f"{len(current_splits)} splits", time.time() - start_time)
    return sorted(current_splits)


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
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('_create_numeric_bins', var=var, num_splits=len(splits))
    
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
    
    result = pd.DataFrame(bins_data)
    if DEBUG_MODE:
        log_function_exit('_create_numeric_bins', f"DataFrame({len(result)})", time.time() - start_time)
    return result


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    max_categories: int = 50,
    max_bins: int = None
) -> pd.DataFrame:
    """Create bin DataFrame for factor/categorical variable."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('_create_factor_bins', var=var, max_categories=max_categories)
    
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    unique_vals = x.dropna().unique()
    n_unique = len(unique_vals)
    
    if max_bins is None:
        max_bins = BinningConfig.MAX_BINS
    
    if DEBUG_MODE:
        logger.debug(f"Unique values: {n_unique}")
    
    if n_unique > max_categories:
        if DEBUG_MODE:
            logger.warning(f"Variable '{var}' has {n_unique} categories, exceeds max {max_categories}")
            log_function_exit('_create_factor_bins', "DataFrame(0)", time.time() - start_time)
        return pd.DataFrame()
    
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
    
    result = pd.DataFrame(bins_data)
    if DEBUG_MODE:
        log_function_exit('_create_factor_bins', f"DataFrame({len(result)})", time.time() - start_time)
    return result


def update_bin_stats(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Update bin statistics."""
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
    min_prop: float = 0.01,
    max_bins: int = 10,
    enforce_monotonic: bool = True,
    algorithm: str = None,
    use_enhancements: bool = None,
    adaptive_min_prop: bool = None,
    min_event_count: bool = None,
    auto_retry: bool = None
) -> BinResult:
    """Get optimal bins for multiple variables."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('get_bins',
                           df_shape=df.shape,
                           y_var=y_var,
                           num_vars=len(x_vars),
                           algorithm=algorithm)
    
    all_bins = []
    var_summaries = []
    
    if algorithm is None:
        algorithm = BinningConfig.ALGORITHM
    if use_enhancements is None:
        use_enhancements = BinningConfig.USE_ENHANCEMENTS
    if adaptive_min_prop is None:
        adaptive_min_prop = BinningConfig.ADAPTIVE_MIN_PROP
    if min_event_count is None:
        min_event_count = BinningConfig.MIN_EVENT_COUNT
    if auto_retry is None:
        auto_retry = BinningConfig.AUTO_RETRY
    
    if DEBUG_MODE:
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Enhancements: {use_enhancements}")
    
    total_vars = len(x_vars)
    processed_count = 0
    
    log_progress(f"Starting binning for {total_vars} variables (Algorithm: {algorithm})")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        if var not in df.columns:
            if DEBUG_MODE:
                logger.warning(f"Variable '{var}' not in DataFrame")
            continue
        
        var_type = get_var_type(df[var])
        
        if var_type == 'numeric':
            if algorithm == "ChiMerge":
                splits = _chimerge_get_splits(
                    df[var], df[y_var],
                    min_bin_pct=min_prop,
                    min_bin_count=BinningConfig.MIN_BIN_COUNT,
                    max_bins=max_bins
                )
            elif algorithm == "IVOptimal":
                splits = _iv_optimal_get_splits(
                    df[var], df[y_var],
                    min_prop=min_prop,
                    max_bins=max_bins
                )
            else:
                splits = _get_decision_tree_splits(
                    df[var], df[y_var],
                    min_prop=min_prop,
                    max_bins=max_bins,
                    use_enhancements=use_enhancements,
                    adaptive_min_prop=adaptive_min_prop,
                    min_event_count=min_event_count,
                    auto_retry=auto_retry
                )
            
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
        processed_count += 1
        
        if processed_count % 10 == 0 or processed_count == total_vars:
            pct = (processed_count / total_vars) * 100
            log_progress(f"[{processed_count}/{total_vars}] {pct:.1f}% | {var}: IV={total_row['iv']:.4f}")
    
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {processed_count} variables in {format_time(total_time)}")
    
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    var_summary_df = pd.DataFrame(var_summaries)
    
    result = BinResult(var_summary=var_summary_df, bin=combined_bins)
    if DEBUG_MODE:
        log_function_exit('get_bins', f"BinResult(vars={len(var_summaries)})", total_time)
    return result


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
    """Combine NA bin with the adjacent bin that has the closest bad rate."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('na_combine', 
                           num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1,
                           prevent_single_bin=prevent_single_bin)
    
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    skipped = 0
    
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
        
        if prevent_single_bin and len(non_na_bins) <= 1:
            skipped += 1
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
    
    if skipped > 0 and DEBUG_MODE:
        logger.info(f"Skipped {skipped} variables (would create single-bin)")
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    if DEBUG_MODE:
        log_function_exit('na_combine', "BinResult", time.time() - start_time)
    return result


def merge_pure_bins(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None
) -> BinResult:
    """Merge pure bins (100% goods or 100% bads) with the closest non-pure bin."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('merge_pure_bins')
    
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    merged_count = 0
    
    for var in vars_to_process:
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            var_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
            
            if len(var_bins) <= 1:
                break
            
            pure_mask = (var_bins['goods'] == 0) | (var_bins['bads'] == 0)
            
            if not pure_mask.any():
                break
            
            pure_bin = var_bins[pure_mask].iloc[0]
            pure_idx = var_bins[pure_mask].index[0]
            
            non_pure_bins = var_bins[~pure_mask]
            
            if non_pure_bins.empty:
                other_bins = var_bins[var_bins.index != pure_idx]
                if other_bins.empty:
                    break
                other_bins = other_bins.copy()
                other_bins['count_diff'] = abs(other_bins['count'] - pure_bin['count'])
                closest_idx = other_bins['count_diff'].idxmin()
                closest_bin = other_bins.loc[closest_idx]
            else:
                pure_bad_rate = pure_bin['bads'] / pure_bin['count'] if pure_bin['count'] > 0 else 0.5
                non_pure_bins = non_pure_bins.copy()
                non_pure_bins['bad_rate_calc'] = non_pure_bins['bads'] / non_pure_bins['count']
                non_pure_bins['rate_diff'] = abs(non_pure_bins['bad_rate_calc'] - pure_bad_rate)
                closest_idx = non_pure_bins['rate_diff'].idxmin()
                closest_bin = non_pure_bins.loc[closest_idx]
            
            combined_rule = f"({closest_bin['bin']}) | ({pure_bin['bin']})"
            combined_count = closest_bin['count'] + pure_bin['count']
            combined_goods = closest_bin['goods'] + pure_bin['goods']
            combined_bads = closest_bin['bads'] + pure_bin['bads']
            
            new_bins.loc[closest_idx, 'bin'] = combined_rule
            new_bins.loc[closest_idx, 'count'] = combined_count
            new_bins.loc[closest_idx, 'goods'] = combined_goods
            new_bins.loc[closest_idx, 'bads'] = combined_bads
            
            new_bins = new_bins.drop(pure_idx)
            merged_count += 1
        
        var_new_bins = new_bins[(new_bins['var'] == var) & (new_bins['bin'] != 'Total')].copy()
        if not var_new_bins.empty:
            var_new_bins = update_bin_stats(var_new_bins)
            var_new_bins = add_total_row(var_new_bins, var)
            
            new_bins = new_bins[new_bins['var'] != var]
            new_bins = pd.concat([new_bins, var_new_bins], ignore_index=True)
            
            total_row = var_new_bins[var_new_bins['bin'] == 'Total'].iloc[0]
            mask = var_summary['var'] == var
            if mask.any():
                var_summary.loc[mask, 'iv'] = total_row['iv']
                var_summary.loc[mask, 'purNode'] = 'N'
    
    if DEBUG_MODE:
        logger.info(f"Merged {merged_count} pure bins")
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    if DEBUG_MODE:
        log_function_exit('merge_pure_bins', "BinResult", time.time() - start_time)
    return result


def force_incr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Force an increasing monotonic trend in bad rates."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('force_incr_trend', 
                           num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
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
        merge_count = 0
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
                    
                    working_bins = working_bins.drop(working_bins.index[i]).reset_index(drop=True)
                    changed = True
                    merge_count += 1
                    break
        
        if DEBUG_MODE:
            logger.debug(f"{var}: merged {merge_count} bins for increasing trend")
        
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
            var_summary.loc[mask, 'monTrend'] = 'Y'
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    if DEBUG_MODE:
        log_function_exit('force_incr_trend', "BinResult", time.time() - start_time)
    return result


def force_decr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Force a decreasing monotonic trend in bad rates."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('force_decr_trend',
                           num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
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
        merge_count = 0
        while changed and len(working_bins) > 1:
            changed = False
            working_bins['bad_rate_calc'] = working_bins['bads'] / working_bins['count']
            
            for i in range(1, len(working_bins)):
                if working_bins.iloc[i]['bad_rate_calc'] > working_bins.iloc[i-1]['bad_rate_calc']:
                    working_bins.iloc[i-1, working_bins.columns.get_loc('count')] += working_bins.iloc[i]['count']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('goods')] += working_bins.iloc[i]['goods']
                    working_bins.iloc[i-1, working_bins.columns.get_loc('bads')] += working_bins.iloc[i]['bads']
                    
                    working_bins = working_bins.drop(working_bins.index[i]).reset_index(drop=True)
                    changed = True
                    merge_count += 1
                    break
        
        if DEBUG_MODE:
            logger.debug(f"{var}: merged {merge_count} bins for decreasing trend")
        
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
            var_summary.loc[mask, 'monTrend'] = 'Y'
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    if DEBUG_MODE:
        log_function_exit('force_decr_trend', "BinResult", time.time() - start_time)
    return result


def create_binned_columns(
    bin_result: BinResult,
    df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_"
) -> pd.DataFrame:
    """Create binned columns in the DataFrame based on binning rules."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('create_binned_columns', num_vars=len(x_vars), prefix=prefix)
    
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
                else:
                    continue
                
                if is_na_bin:
                    result_df.loc[mask, new_col] = bin_value
                else:
                    result_df.loc[mask & result_df[var].notna(), new_col] = bin_value
                
            except Exception as e:
                log_exception('create_binned_columns:rule', e)
                continue
        
        if na_rule is not None:
            result_df.loc[result_df[var].isna(), new_col] = na_rule
        
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
            unassigned_count = unassigned_mask.sum()
            if DEBUG_MODE:
                logger.warning(f"{var}: {unassigned_count} unassigned rows")
            if na_rule is not None:
                fallback_bin = na_rule
            elif not var_bins.empty:
                fallback_bin = var_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
            else:
                fallback_bin = "Unmatched"
            result_df.loc[unassigned_mask, new_col] = fallback_bin
    
    if DEBUG_MODE:
        log_function_exit('create_binned_columns', f"DataFrame({result_df.shape})", time.time() - start_time)
    return result_df


def add_woe_columns(
    df: pd.DataFrame,
    bins_df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    woe_prefix: str = "WOE_"
) -> pd.DataFrame:
    """Add WOE columns to the DataFrame by joining with binning rules."""
    start_time = time.time()
    if DEBUG_MODE:
        log_function_entry('add_woe_columns', num_vars=len(x_vars))
    
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
            
            missing_woe_count = result_df[woe_col].isna().sum()
            if missing_woe_count > 0 and DEBUG_MODE:
                logger.error(f"{var}: {missing_woe_count} rows have unmapped bin values")
    
    if DEBUG_MODE:
        log_function_exit('add_woe_columns', f"DataFrame({result_df.shape})", time.time() - start_time)
    return result_df


# =============================================================================
# Read Input Data
# =============================================================================

if DEBUG_MODE:
    logger.info("=" * 70)
    logger.info("WOE EDITOR ADVANCED - DEBUG TOGGLE VERSION")
    logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
    logger.info("=" * 70)
    logger.info(f"Script started at: {datetime.now().isoformat()}")

if DEBUG_MODE:
    logger.info("Reading input data from KNIME...")
df = knio.input_tables[0].to_pandas()
if DEBUG_MODE:
    logger.info(f"Input data shape: {df.shape}")
    logger.debug(f"Input columns: {list(df.columns)[:10]}...")

# Default min_prop
min_prop = BinningConfig.MIN_BIN_PCT

# =============================================================================
# Check for Flow Variables
# =============================================================================
if DEBUG_MODE:
    logger.info("Checking for flow variables...")

contains_dv = False
dv = None
target = None
optimize_all = False
group_na = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if DEBUG_MODE:
        logger.debug(f"DependentVariable: {dv}")
except:
    pass

try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
    if DEBUG_MODE:
        logger.debug(f"OptimizeAll: {optimize_all}")
except:
    pass

try:
    group_na = knio.flow_variables.get("GroupNA", False)
    if DEBUG_MODE:
        logger.debug(f"GroupNA: {group_na}")
except:
    pass

# Read algorithm settings
try:
    min_bin_pct_fv = knio.flow_variables.get("MinBinPct", None)
    if min_bin_pct_fv is not None:
        min_prop = float(min_bin_pct_fv)
        BinningConfig.MIN_BIN_PCT = min_prop
        if DEBUG_MODE:
            logger.info(f"MinBinPct: {min_prop}")
except:
    pass

try:
    algorithm_fv = knio.flow_variables.get("Algorithm", None)
    if algorithm_fv is not None:
        BinningConfig.ALGORITHM = algorithm_fv
        if DEBUG_MODE:
            logger.info(f"Algorithm: {algorithm_fv}")
except:
    pass

try:
    use_enhancements_fv = knio.flow_variables.get("UseEnhancements", None)
    if use_enhancements_fv is not None:
        BinningConfig.USE_ENHANCEMENTS = bool(use_enhancements_fv)
        if DEBUG_MODE:
            logger.info(f"UseEnhancements: {BinningConfig.USE_ENHANCEMENTS}")
except:
    pass

if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        if DEBUG_MODE:
            logger.info(f"Headless mode activated with DV: {dv}")

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv:
    # HEADLESS MODE
    log_progress("=" * 60)
    log_progress(f"WOE EDITOR ADVANCED - HEADLESS MODE (DEBUG_MODE={DEBUG_MODE})")
    log_progress("=" * 60)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"Algorithm: {BinningConfig.ALGORITHM}")
    log_progress(f"OptimizeAll: {optimize_all}, GroupNA: {group_na}")
    
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter constant variables
    constant_vars = []
    valid_vars = []
    for col in iv_list:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_vars.append(col)
        else:
            valid_vars.append(col)
    
    if constant_vars:
        log_progress(f"Removed {len(constant_vars)} constant variables")
    
    iv_list = valid_vars
    log_progress(f"Variables to process: {len(iv_list)}")
    
    # Step 1: Initial binning
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop,
                          algorithm=BinningConfig.ALGORITHM,
                          use_enhancements=BinningConfig.USE_ENHANCEMENTS)
    
    # Step 2: Merge pure bins
    if 'purNode' in bins_result.var_summary.columns:
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    if pure_count > 0:
        log_progress(f"Merging {int(pure_count)} pure bins...")
        bins_result = merge_pure_bins(bins_result)
    
    # Step 3: Group NA
    if group_na:
        log_progress("Grouping NA values...")
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist(),
                                prevent_single_bin=BinningConfig.SINGLE_BIN_PROTECTION)
    
    # Step 4: Optimize All
    if optimize_all:
        log_progress("Optimizing monotonicity...")
        bins_mod = na_combine(bins_result, bins_result.var_summary['var'].tolist(),
                             prevent_single_bin=BinningConfig.SINGLE_BIN_PROTECTION)
        
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            log_progress(f"  - Forcing decreasing trend on {len(decr_vars)} variables...")
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            log_progress(f"  - Forcing increasing trend on {len(incr_vars)} variables...")
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
    
    # Step 5: Apply WOE transformation
    log_progress("Applying WOE transformation...")
    
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
    df_with_bins = create_binned_columns(bins_result, df, all_vars)
    df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
    
    woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
    df_only_woe = df_with_woe[woe_cols + [dv]].copy()
    
    bins = rules
    
    log_progress("=" * 60)
    log_progress(f"COMPLETE: Processed {len(all_vars)} variables")
    log_progress("=" * 60)

else:
    # INTERACTIVE MODE
    if DEBUG_MODE:
        logger.info("Running in interactive mode - Shiny UI not available in DEBUG version")
    df_with_woe = df.copy()
    df_only_woe = pd.DataFrame()
    bins = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

if DEBUG_MODE:
    logger.info("Writing output tables...")

knio.output_tables[0] = knio.Table.from_pandas(df)
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)

b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
df_only_bins = df_with_woe[b_columns].copy() if b_columns else pd.DataFrame()
knio.output_tables[3] = knio.Table.from_pandas(df_only_bins)

knio.output_tables[4] = knio.Table.from_pandas(bins if isinstance(bins, pd.DataFrame) else pd.DataFrame())

if DEBUG_MODE:
    logger.info("=" * 70)
    logger.info("OUTPUT SUMMARY:")
    logger.info(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
    logger.info(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
    logger.info(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
    logger.info(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols)")
    logger.info(f"  Port 5: Bin Rules ({len(bins) if isinstance(bins, pd.DataFrame) else 0} rows)")
    logger.info("=" * 70)
    logger.info(f"WOE Editor Advanced (DEBUG TOGGLE) completed at: {datetime.now().isoformat()}")

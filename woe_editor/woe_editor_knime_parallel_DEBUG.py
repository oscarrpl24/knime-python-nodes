# =============================================================================
# WOE Editor for KNIME Python Script Node - PARALLEL VERSION - DEBUG
# =============================================================================
# Python implementation matching R's WOE Editor functionality
# Compatible with KNIME 5.9, Python 3.9
#
# DEBUG VERSION: Includes extensive debug logging on every function
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
# Version: 2.0-DEBUG (Parallel)
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re
import warnings
import os
import gc
import sys
import random
import multiprocessing
import logging
import traceback
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================

def setup_debug_logging():
    """Configure debug logging with detailed formatting."""
    logger = logging.getLogger('WOE_PARALLEL_DEBUG')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_debug_logging()

def log_function_entry(func_name, **kwargs):
    """Log function entry with parameters."""
    params_str = ', '.join([f"{k}={repr(v)[:80]}" for k, v in kwargs.items()])
    logger.debug(f">>> ENTER {func_name}({params_str})")

def log_function_exit(func_name, result=None, duration=None):
    """Log function exit with result."""
    result_str = repr(result) if result is not None else "None"
    if len(result_str) > 150:
        result_str = result_str[:150] + "..."
    duration_str = f" [{duration:.3f}s]" if duration else ""
    logger.debug(f"<<< EXIT {func_name}{duration_str} -> {result_str}")

def log_exception(func_name, e):
    """Log exception with full traceback."""
    logger.error(f"!!! EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
BASE_PORT = 8054
RANDOM_PORT_RANGE = 1000

INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
logger.info(f"Instance ID: {INSTANCE_ID}")
logger.info(f"Process ID: {os.getpid()}")

os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
logger.debug("Thread environment variables set to 1")

# =============================================================================
# Install/Import Dependencies
# =============================================================================

logger.info("Loading dependencies...")

try:
    from sklearn.tree import DecisionTreeClassifier
    logger.debug("sklearn imported successfully")
except ImportError:
    logger.warning("scikit-learn not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier

try:
    from joblib import Parallel, delayed
    logger.debug("joblib imported successfully")
except ImportError:
    logger.warning("joblib not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'joblib'])
    from joblib import Parallel, delayed

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    logger.debug("Shiny and plotly imported successfully")
except ImportError:
    logger.warning("Shiny/plotly not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Resource Detection
# =============================================================================

def get_optimal_n_jobs(reserve_cores: int = 1, max_usage_percent: float = 0.75) -> int:
    """Determine optimal number of parallel jobs based on available system resources."""
    start_time = time.time()
    log_function_entry('get_optimal_n_jobs', reserve_cores=reserve_cores, max_usage_percent=max_usage_percent)
    
    try:
        logical_cores = os.cpu_count() or 1
        logger.debug(f"Logical cores detected: {logical_cores}")
        
        try:
            physical_cores = multiprocessing.cpu_count()
            logger.debug(f"Physical cores detected: {physical_cores}")
        except:
            physical_cores = logical_cores
        
        effective_cores = min(logical_cores, int(physical_cores * 1.5))
        logger.debug(f"Effective cores: {effective_cores}")
        
        max_cores = max(1, int(effective_cores * max_usage_percent))
        available_cores = max(1, max_cores - reserve_cores)
        
        logger.info(f"[Parallel Config] Detected: {logical_cores} logical, {physical_cores} physical")
        logger.info(f"[Parallel Config] Using: {available_cores} parallel workers")
        
        log_function_exit('get_optimal_n_jobs', available_cores, time.time() - start_time)
        return available_cores
        
    except Exception as e:
        log_exception('get_optimal_n_jobs', e)
        return 4


N_JOBS = get_optimal_n_jobs(reserve_cores=1, max_usage_percent=0.75)
logger.info(f"Global N_JOBS set to: {N_JOBS}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BinResult:
    """Container for binning results"""
    var_summary: pd.DataFrame
    bin: pd.DataFrame


# =============================================================================
# Core Binning Functions
# =============================================================================

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray) -> np.ndarray:
    """Calculate Weight of Evidence (WOE) for each bin."""
    start_time = time.time()
    log_function_entry('calculate_woe', freq_good_len=len(freq_good), freq_bad_len=len(freq_bad))
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    logger.debug(f"Totals: good={total_good}, bad={total_bad}")
    
    if total_good == 0 or total_bad == 0:
        logger.warning("Zero total goods or bads")
        result = np.zeros(len(freq_good))
        log_function_exit('calculate_woe', f"zeros({len(result)})", time.time() - start_time)
        return result
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    dist_good = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    woe = np.round(np.log(dist_bad / dist_good), 5)
    
    log_function_exit('calculate_woe', f"array({len(woe)})", time.time() - start_time)
    return woe


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """Calculate Information Value (IV) for a variable."""
    start_time = time.time()
    log_function_entry('calculate_iv', freq_good_len=len(freq_good), freq_bad_len=len(freq_bad))
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        log_function_exit('calculate_iv', 0.0, time.time() - start_time)
        return 0.0
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    dist_good_safe = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    woe = np.log(dist_bad_safe / dist_good_safe)
    iv = np.sum((dist_bad - dist_good) * woe)
    
    if not np.isfinite(iv):
        iv = 0.0
    
    result = round(iv, 4)
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
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return 'factor'
        return 'numeric'
    return 'factor'


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.01,
    max_bins: int = 10
) -> List[float]:
    """Use decision tree to find optimal split points for numeric variables."""
    start_time = time.time()
    log_function_entry('_get_decision_tree_splits', x_name=x.name, min_prop=min_prop, max_bins=max_bins)
    
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        log_function_exit('_get_decision_tree_splits', [], time.time() - start_time)
        return []
    
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    logger.debug(f"min_samples_leaf: {min_samples_leaf}, n_samples: {len(x_clean)}")
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        tree.fit(x_clean, y_clean)
    except Exception as e:
        log_exception('_get_decision_tree_splits', e)
        return []
    
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    thresholds = sorted(set(thresholds))
    
    log_function_exit('_get_decision_tree_splits', f"{len(thresholds)} splits", time.time() - start_time)
    return thresholds


def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """Create bin DataFrame for numeric variable based on splits."""
    start_time = time.time()
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
                'count': count,
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
    log_function_exit('_create_numeric_bins', f"DataFrame({len(result)})", time.time() - start_time)
    return result


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """Create bin DataFrame for factor/categorical variable."""
    start_time = time.time()
    log_function_entry('_create_factor_bins', var=var)
    
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    unique_vals = x.dropna().unique()
    
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
    log_function_exit('_create_factor_bins', f"DataFrame({len(result)})", time.time() - start_time)
    return result


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


# =============================================================================
# Parallel Processing Functions
# =============================================================================

def _process_single_var(
    df: pd.DataFrame, 
    var: str, 
    y_var: str, 
    min_prop: float, 
    max_bins: int
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Process a single variable for binning - designed for parallel execution."""
    start_time = time.time()
    # Note: Using print instead of logger in parallel workers (logger may not work in subprocesses)
    print(f"[PARALLEL] Processing variable: {var}")
    
    try:
        if var not in df.columns:
            print(f"  [SKIP] {var}: Column not found")
            return None, None
        
        non_na_count = df[var].notna().sum()
        if non_na_count == 0:
            print(f"  [SKIP] {var}: All values NaN")
            return None, None
        
        if df[var].nunique() <= 1:
            print(f"  [SKIP] {var}: Constant column")
            return None, None
        
        var_type = get_var_type(df[var])
        
        if var_type == 'numeric':
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            bin_df = _create_factor_bins(df, var, y_var)
        
        if bin_df.empty:
            print(f"  [SKIP] {var}: No valid bins")
            return None, None
        
        bin_df = update_bin_stats(bin_df)
        bin_df = add_total_row(bin_df, var)
        
        total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
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
        
        duration = time.time() - start_time
        print(f"  [DONE] {var}: IV={total_row['iv']:.4f}, {len(bin_df)-1} bins [{duration:.2f}s]")
        
        return var_summary, bin_df
        
    except Exception as e:
        print(f"  [ERROR] {var}: {type(e).__name__}: {str(e)[:80]}")
        return None, None


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    min_prop: float = 0.01,
    max_bins: int = 10,
    n_jobs: Optional[int] = None
) -> BinResult:
    """Get optimal bins for multiple variables using parallel processing."""
    start_time = time.time()
    log_function_entry('get_bins', df_shape=df.shape, y_var=y_var, 
                       num_vars=len(x_vars), min_prop=min_prop, n_jobs=n_jobs)
    
    if n_jobs is None:
        n_jobs = N_JOBS
    
    logger.info(f"[Parallel] Processing {len(x_vars)} variables using {n_jobs} workers...")
    
    if len(x_vars) > 1 and n_jobs > 1:
        try:
            logger.debug("Starting parallel processing with joblib...")
            results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_process_single_var)(df, var, y_var, min_prop, max_bins)
                for var in x_vars
            )
            logger.debug("Parallel processing completed")
        except Exception as e:
            log_exception('get_bins:parallel', e)
            logger.warning("Falling back to sequential processing")
            results = [_process_single_var(df, var, y_var, min_prop, max_bins) for var in x_vars]
    else:
        logger.debug("Using sequential processing (n_jobs=1 or single variable)")
        results = [_process_single_var(df, var, y_var, min_prop, max_bins) for var in x_vars]
    
    var_summaries = []
    all_bins = []
    skipped_count = 0
    
    for var_summary, bin_df in results:
        if var_summary is not None:
            var_summaries.append(var_summary)
            all_bins.append(bin_df)
        else:
            skipped_count += 1
    
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    var_summary_df = pd.DataFrame(var_summaries)
    
    duration = time.time() - start_time
    logger.info(f"[Parallel] Completed: {len(var_summaries)}/{len(x_vars)} variables in {duration:.2f}s")
    if skipped_count > 0:
        logger.warning(f"[Parallel] {skipped_count} variables skipped")
    
    result = BinResult(var_summary=var_summary_df, bin=combined_bins)
    log_function_exit('get_bins', f"BinResult(vars={len(var_summaries)})", duration)
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
    n_jobs: Optional[int] = None
) -> BinResult:
    """Combine NA bin with the adjacent bin that has the closest bad rate."""
    start_time = time.time()
    log_function_entry('na_combine', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
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
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    log_function_exit('na_combine', "BinResult", time.time() - start_time)
    return result


def merge_pure_bins(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]] = None
) -> BinResult:
    """Merge pure bins (100% goods or 100% bads) with the closest non-pure bin."""
    start_time = time.time()
    log_function_entry('merge_pure_bins')
    
    if vars_to_process is None:
        vars_to_process = bin_result.var_summary['var'].tolist()
    elif isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
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
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    log_function_exit('merge_pure_bins', "BinResult", time.time() - start_time)
    return result


def force_incr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    n_jobs: Optional[int] = None
) -> BinResult:
    """Force an increasing monotonic trend in bad rates."""
    start_time = time.time()
    log_function_entry('force_incr_trend', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
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
            var_summary.loc[mask, 'monTrend'] = 'Y'
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    log_function_exit('force_incr_trend', "BinResult", time.time() - start_time)
    return result


def force_decr_trend(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    n_jobs: Optional[int] = None
) -> BinResult:
    """Force a decreasing monotonic trend in bad rates."""
    start_time = time.time()
    log_function_entry('force_decr_trend', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
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
            var_summary.loc[mask, 'monTrend'] = 'Y'
    
    result = BinResult(var_summary=var_summary, bin=new_bins)
    log_function_exit('force_decr_trend', "BinResult", time.time() - start_time)
    return result


def create_binned_columns(
    bin_result: BinResult,
    df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """Create binned columns in the DataFrame based on binning rules."""
    start_time = time.time()
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
                continue
        
        if na_rule is not None:
            result_df.loc[result_df[var].isna(), new_col] = na_rule
        
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
            if na_rule is not None:
                fallback_bin = na_rule
            elif not var_bins.empty:
                fallback_bin = var_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
            else:
                fallback_bin = "Unmatched"
            result_df.loc[unassigned_mask, new_col] = fallback_bin
    
    log_function_exit('create_binned_columns', f"DataFrame({result_df.shape})", time.time() - start_time)
    return result_df


def add_woe_columns(
    df: pd.DataFrame,
    bins_df: pd.DataFrame,
    x_vars: List[str],
    prefix: str = "b_",
    woe_prefix: str = "WOE_",
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """Add WOE columns to the DataFrame by joining with binning rules."""
    start_time = time.time()
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
            if missing_woe_count > 0:
                logger.error(f"{var}: {missing_woe_count} rows have unmapped bin values!")
    
    log_function_exit('add_woe_columns', f"DataFrame({result_df.shape})", time.time() - start_time)
    return result_df


# =============================================================================
# Configuration
# =============================================================================
min_prop = 0.01
logger.info(f"Configuration: min_prop={min_prop}")

# =============================================================================
# Read Input Data
# =============================================================================
logger.info("=" * 70)
logger.info("WOE EDITOR (PARALLEL) - DEBUG VERSION")
logger.info("=" * 70)
logger.info(f"Script started at: {datetime.now().isoformat()}")
logger.info(f"Parallel workers: {N_JOBS}")

logger.info("Reading input data from KNIME...")
df = knio.input_tables[0].to_pandas()
logger.info(f"Input data shape: {df.shape}")
logger.debug(f"Input columns: {list(df.columns)[:10]}...")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
logger.info("Checking for flow variables...")

contains_dv = False
dv = None
target = None
optimize_all = False
group_na = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
    logger.debug(f"DependentVariable: {dv}")
except:
    pass

try:
    target = knio.flow_variables.get("TargetCategory", None)
    logger.debug(f"TargetCategory: {target}")
except:
    pass

try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
    logger.debug(f"OptimizeAll: {optimize_all}")
except:
    pass

try:
    group_na = knio.flow_variables.get("GroupNA", False)
    logger.debug(f"GroupNA: {group_na}")
except:
    pass

if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        logger.info(f"Headless mode with DV: {dv}")

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv:
    # =========================================================================
    # HEADLESS MODE (with parallel processing)
    # =========================================================================
    logger.info(f"Running in headless mode with DV: {dv}")
    logger.info(f"[Parallel] Using {N_JOBS} workers")
    
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter out constant variables
    constant_vars = []
    valid_vars = []
    for col in iv_list:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 1:
            constant_vars.append(col)
        else:
            valid_vars.append(col)
    
    if constant_vars:
        logger.info(f"Removed {len(constant_vars)} constant variables")
    
    iv_list = valid_vars
    logger.info(f"Variables to process: {len(iv_list)}")
    
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    
    # Merge pure bins
    if 'purNode' in bins_result.var_summary.columns:
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    if pure_count > 0:
        logger.info(f"Merging {int(pure_count)} pure bins...")
        bins_result = merge_pure_bins(bins_result)
    
    if group_na:
        logger.info("Grouping NA values...")
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist())
    
    if optimize_all:
        logger.info("Optimizing monotonicity...")
        bins_mod = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
    
    logger.info("Applying WOE transformation...")
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
    
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
    
    logger.info(f"Processed {len(all_vars)} variables")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    logger.info("Running in interactive mode - Shiny UI not available in DEBUG version")
    df_with_woe = df.copy()
    df_only_woe = pd.DataFrame()
    bins = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

logger.info("Writing output tables...")

knio.output_tables[0] = knio.Table.from_pandas(df)
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)
knio.output_tables[3] = knio.Table.from_pandas(bins if isinstance(bins, pd.DataFrame) else pd.DataFrame())

logger.info("=" * 70)
logger.info("OUTPUT SUMMARY:")
logger.info(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
logger.info(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
logger.info(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
logger.info(f"  Port 4: Bin Rules ({len(bins) if isinstance(bins, pd.DataFrame) else 0} rows)")
logger.info("=" * 70)
logger.info(f"WOE Editor (Parallel DEBUG) completed at: {datetime.now().isoformat()}")

# =============================================================================
# Cleanup
# =============================================================================
sys.stdout.flush()
gc.collect()
logger.debug("Garbage collection completed")

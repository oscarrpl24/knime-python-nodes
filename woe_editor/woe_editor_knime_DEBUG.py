# =============================================================================
# WOE Editor for KNIME Python Script Node - DEBUG VERSION
# =============================================================================
# Python implementation matching R's WOE Editor functionality
# Compatible with KNIME 5.9, Python 3.9
#
# DEBUG VERSION: Includes extensive debug logging on every function
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When DependentVariable flow variable is provided
#
# Outputs:
# 1. Original input DataFrame (unchanged)
# 2. df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# 3. df_only_woe - Only WOE columns + dependent variable (for logistic regression)
# 4. df_only_bins - ONLY binned columns (b_*) for scorecard scoring (LEAN!)
# 5. bins - Binning rules with WOE values (metadata)
#
# Release Date: 2026-01-15
# Version: 1.0-DEBUG
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
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================

def setup_debug_logging():
    """Configure debug logging with detailed formatting."""
    logger = logging.getLogger('WOE_EDITOR_DEBUG')
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
    params_str = ', '.join([f"{k}={repr(v)[:100]}" for k, v in kwargs.items()])
    logger.debug(f">>> ENTER {func_name}({params_str})")

def log_function_exit(func_name, result=None, duration=None):
    """Log function exit with result."""
    result_str = repr(result) if result is not None else "None"
    if len(result_str) > 200:
        result_str = result_str[:200] + "..."
    duration_str = f" [{duration:.3f}s]" if duration else ""
    logger.debug(f"<<< EXIT {func_name}{duration_str} -> {result_str}")

def log_exception(func_name, e):
    """Log exception with full traceback."""
    logger.error(f"!!! EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
BASE_PORT = 8050
RANDOM_PORT_RANGE = 1000

INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"
logger.info(f"Instance ID: {INSTANCE_ID}")

os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
logger.debug("Thread environment variables set to 1")

# =============================================================================
# Progress Logging Utilities
# =============================================================================

def log_progress(message: str, flush: bool = True):
    """Print a progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    logger.info(message)
    if flush:
        sys.stdout.flush()

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    log_function_entry('format_time', seconds=seconds)
    if seconds < 60:
        result = f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        result = f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        result = f"{hours}h {mins}m"
    log_function_exit('format_time', result)
    return result

# =============================================================================
# Install/Import Dependencies
# =============================================================================

logger.info("Loading dependencies...")

try:
    from sklearn.tree import DecisionTreeClassifier
    logger.debug("sklearn.tree.DecisionTreeClassifier imported successfully")
except ImportError:
    logger.warning("scikit-learn not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier
    logger.debug("scikit-learn installed and imported")

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
    logger.debug("Shiny and plotly installed and imported")


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
    log_function_entry('calculate_woe', 
                       freq_good_shape=np.array(freq_good).shape,
                       freq_bad_shape=np.array(freq_bad).shape)
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    logger.debug(f"Total good: {total_good}, Total bad: {total_bad}")
    
    if total_good == 0 or total_bad == 0:
        logger.warning("Zero total goods or bads, returning zeros")
        result = np.zeros(len(freq_good))
        log_function_exit('calculate_woe', result, time.time() - start_time)
        return result
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    dist_good = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    woe = np.round(np.log(dist_bad / dist_good), 5)
    logger.debug(f"WOE values: min={woe.min():.4f}, max={woe.max():.4f}, mean={woe.mean():.4f}")
    
    log_function_exit('calculate_woe', f"array({len(woe)})", time.time() - start_time)
    return woe


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """Calculate Information Value (IV) for a variable."""
    start_time = time.time()
    log_function_entry('calculate_iv',
                       freq_good_len=len(freq_good),
                       freq_bad_len=len(freq_bad))
    
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        logger.warning("Zero total goods or bads in IV calculation")
        log_function_exit('calculate_iv', 0.0, time.time() - start_time)
        return 0.0
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    dist_good_safe = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    woe = np.log(dist_bad_safe / dist_good_safe)
    iv = np.sum((dist_bad - dist_good) * woe)
    
    if not np.isfinite(iv):
        logger.warning(f"Non-finite IV detected: {iv}, setting to 0")
        iv = 0.0
    
    result = round(iv, 4)
    log_function_exit('calculate_iv', result, time.time() - start_time)
    return result


def calculate_entropy(goods: int, bads: int) -> float:
    """Calculate entropy for a bin."""
    log_function_entry('calculate_entropy', goods=goods, bads=bads)
    
    total = goods + bads
    if total == 0 or goods == 0 or bads == 0:
        logger.debug("Edge case: returning 0.0 for entropy")
        return 0.0
    
    p_good = goods / total
    p_bad = bads / total
    
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    result = round(entropy, 4)
    
    log_function_exit('calculate_entropy', result)
    return result


def get_var_type(series: pd.Series) -> str:
    """Determine if variable is numeric or factor (categorical)"""
    log_function_entry('get_var_type', series_name=series.name, dtype=str(series.dtype))
    
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique()
        logger.debug(f"Numeric dtype detected, nunique={nunique}")
        if nunique <= 10:
            result = 'factor'
        else:
            result = 'numeric'
    else:
        result = 'factor'
    
    log_function_exit('get_var_type', result)
    return result


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.05,
    max_bins: int = 10
) -> List[float]:
    """Use decision tree to find optimal split points for numeric variables."""
    start_time = time.time()
    log_function_entry('_get_decision_tree_splits',
                       x_name=x.name, min_prop=min_prop, max_bins=max_bins)
    
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    logger.debug(f"Clean data: {len(x_clean)} samples")
    
    if len(x_clean) == 0:
        logger.warning("No valid data after cleaning")
        log_function_exit('_get_decision_tree_splits', [], time.time() - start_time)
        return []
    
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    logger.debug(f"min_samples_leaf: {min_samples_leaf}")
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        logger.debug("Fitting decision tree...")
        tree.fit(x_clean, y_clean)
        logger.debug(f"Tree fitted with {tree.tree_.node_count} nodes")
    except Exception as e:
        log_exception('_get_decision_tree_splits', e)
        return []
    
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    thresholds = sorted(set(thresholds))
    
    logger.debug(f"Found {len(thresholds)} unique thresholds: {thresholds[:5]}{'...' if len(thresholds) > 5 else ''}")
    
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
    log_function_entry('_create_numeric_bins', var=var, y_var=y_var, num_splits=len(splits))
    
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    logger.debug(f"Creating bins with edges: {len(edges)} edges")
    
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
            logger.debug(f"Bin {i}: count={count}, goods={goods}, bads={bads}")
    
    # Handle NA
    na_mask = x.isna()
    na_count = na_mask.sum()
    if na_count > 0:
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
        logger.debug(f"NA bin: count={na_count}, goods={na_goods}, bads={na_bads}")
    
    result = pd.DataFrame(bins_data)
    log_function_exit('_create_numeric_bins', f"DataFrame({len(result)} rows)", time.time() - start_time)
    return result


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """Create bin DataFrame for factor/categorical variable."""
    start_time = time.time()
    log_function_entry('_create_factor_bins', var=var, y_var=y_var)
    
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    unique_vals = x.dropna().unique()
    logger.debug(f"Found {len(unique_vals)} unique values")
    
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
    
    # Handle NA
    na_mask = x.isna()
    na_count = na_mask.sum()
    if na_count > 0:
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
        logger.debug(f"NA bin: count={na_count}")
    
    result = pd.DataFrame(bins_data)
    log_function_exit('_create_factor_bins', f"DataFrame({len(result)} rows)", time.time() - start_time)
    return result


def update_bin_stats(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Update bin statistics (propn, bad_rate, iv, ent, trend, etc.)"""
    start_time = time.time()
    log_function_entry('update_bin_stats', input_shape=bin_df.shape)
    
    if bin_df.empty:
        logger.warning("Empty bin_df received")
        return bin_df
    
    df = bin_df.copy()
    
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    logger.debug(f"Totals: count={total_count}, goods={total_goods}, bads={total_bads}")
    
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
    
    log_function_exit('update_bin_stats', f"DataFrame({df.shape})", time.time() - start_time)
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Add a total row to the bin DataFrame."""
    start_time = time.time()
    log_function_entry('add_total_row', var=var, input_shape=bin_df.shape)
    
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
    
    logger.debug(f"Total row: IV={total_iv:.4f}, trend={overall_trend}, monTrend={mon_trend}")
    
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
    
    result = pd.concat([df, total_row], ignore_index=True)
    log_function_exit('add_total_row', f"DataFrame({result.shape})", time.time() - start_time)
    return result


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    min_prop: float = 0.05,
    max_bins: int = 10
) -> BinResult:
    """Get optimal bins for multiple variables."""
    start_time = time.time()
    log_function_entry('get_bins', 
                       df_shape=df.shape, 
                       y_var=y_var, 
                       num_vars=len(x_vars),
                       min_prop=min_prop, 
                       max_bins=max_bins)
    
    all_bins = []
    var_summaries = []
    
    total_vars = len(x_vars)
    last_log_time = start_time
    processed_count = 0
    times_per_var = []
    
    log_progress(f"Starting binning for {total_vars} variables (Algorithm: DecisionTree)")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        if var not in df.columns:
            logger.warning(f"Variable '{var}' not found in DataFrame columns")
            continue
            
        var_type = get_var_type(df[var])
        logger.debug(f"Processing variable: {var} (type: {var_type})")
        
        if var_type == 'numeric':
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            bin_df = _create_factor_bins(df, var, y_var)
        
        if bin_df.empty:
            logger.warning(f"Empty bin result for variable: {var}")
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
        
        var_time = time.time() - var_start
        times_per_var.append(var_time)
        processed_count += 1
        
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
            
            log_progress(
                f"[{processed_count}/{total_vars}] {pct:.1f}% | "
                f"Variable: {var[:30]:30} | "
                f"IV: {total_row['iv']:.4f} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(remaining)}"
            )
            last_log_time = current_time
    
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {processed_count} variables in {format_time(total_time)}")
    
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    var_summary_df = pd.DataFrame(var_summaries)
    
    result = BinResult(var_summary=var_summary_df, bin=combined_bins)
    log_function_exit('get_bins', f"BinResult(vars={len(var_summaries)})", time.time() - start_time)
    return result


def _parse_numeric_from_rule(rule: str) -> List[float]:
    """Extract numeric values from a bin rule string."""
    log_function_entry('_parse_numeric_from_rule', rule=rule[:50])
    pattern = r"'(-?\d+\.?\d*)'"
    matches = re.findall(pattern, rule)
    result = [float(m) for m in matches]
    log_function_exit('_parse_numeric_from_rule', result)
    return result


def _parse_factor_values_from_rule(rule: str) -> List[str]:
    """Extract factor values from a bin rule string."""
    log_function_entry('_parse_factor_values_from_rule', rule=rule[:50])
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, rule)
    log_function_exit('_parse_factor_values_from_rule', matches)
    return matches


def na_combine(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Combine NA bin with the adjacent bin that has the closest bad rate."""
    start_time = time.time()
    log_function_entry('na_combine', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        logger.debug(f"Processing NA combine for variable: {var}")
        var_bins = new_bins[new_bins['var'] == var].copy()
        
        if var_bins.empty:
            logger.debug(f"No bins found for {var}")
            continue
        
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        
        if not na_mask.any():
            logger.debug(f"No NA bin found for {var}")
            continue
        
        na_bin = var_bins[na_mask].iloc[0]
        non_na_bins = var_bins[~na_mask & (var_bins['bin'] != 'Total')]
        
        if non_na_bins.empty:
            logger.debug(f"No non-NA bins for {var}")
            continue
        
        na_bad_rate = na_bin['bads'] / na_bin['count'] if na_bin['count'] > 0 else 0
        logger.debug(f"NA bin bad_rate: {na_bad_rate:.4f}")
        
        non_na_bins = non_na_bins.copy()
        non_na_bins['bad_rate_calc'] = non_na_bins['bads'] / non_na_bins['count']
        non_na_bins['rate_diff'] = abs(non_na_bins['bad_rate_calc'] - na_bad_rate)
        
        closest_idx = non_na_bins['rate_diff'].idxmin()
        closest_bin = non_na_bins.loc[closest_idx]
        
        logger.debug(f"Combining NA with bin: {closest_bin['bin'][:50]}")
        
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
    
    logger.debug(f"Processing {len(vars_to_process)} variables for pure bin merging")
    
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
            
            logger.debug(f"Found {pure_mask.sum()} pure bins for {var} (iteration {iteration})")
            
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
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Force an increasing monotonic trend in bad rates by combining adjacent bins."""
    start_time = time.time()
    log_function_entry('force_incr_trend', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        logger.debug(f"Forcing increasing trend for: {var}")
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
                    elif '>' in new_rule and '<=' not in new_rule:
                        if '>' in old_rule:
                            old_lower = _parse_numeric_from_rule(old_rule.split('&')[0]) if '&' in old_rule else _parse_numeric_from_rule(old_rule)
                            if old_lower:
                                working_bins.iloc[i-1, working_bins.columns.get_loc('bin')] = f"{var} > '{min(old_lower)}'"
                    
                    working_bins = working_bins.drop(working_bins.index[i]).reset_index(drop=True)
                    changed = True
                    merge_count += 1
                    break
        
        logger.debug(f"Merged {merge_count} bins for {var}")
        
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
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Force a decreasing monotonic trend in bad rates by combining adjacent bins."""
    start_time = time.time()
    log_function_entry('force_decr_trend', num_vars=len(vars_to_process) if isinstance(vars_to_process, list) else 1)
    
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    var_summary = bin_result.var_summary.copy()
    
    for var in vars_to_process:
        logger.debug(f"Forcing decreasing trend for: {var}")
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
        
        logger.debug(f"Merged {merge_count} bins for {var}")
        
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
    prefix: str = "b_"
) -> pd.DataFrame:
    """Create binned columns in the DataFrame based on binning rules."""
    start_time = time.time()
    log_function_entry('create_binned_columns', num_vars=len(x_vars), prefix=prefix)
    
    result_df = df.copy()
    
    for var in x_vars:
        logger.debug(f"Creating binned column for: {var}")
        var_bins = bin_result.bin[(bin_result.bin['var'] == var) & 
                                   (bin_result.bin['bin'] != 'Total')]
        
        if var_bins.empty:
            logger.warning(f"No bins found for {var}")
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
                log_exception('create_binned_columns:rule_parsing', e)
                continue
        
        if na_rule is not None:
            result_df.loc[result_df[var].isna(), new_col] = na_rule
        elif result_df[var].isna().any():
            na_bins = var_bins[var_bins['bin'].str.match(r'^is\.na\(', na=False)]
            if not na_bins.empty:
                bin_value = na_bins.iloc[0]['bin'].replace(var, '').replace(' %in% c', '').strip()
                result_df.loc[result_df[var].isna(), new_col] = bin_value
        
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
            unassigned_count = unassigned_mask.sum()
            logger.warning(f"{var}: {unassigned_count} rows unassigned, using fallback")
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
    woe_prefix: str = "WOE_"
) -> pd.DataFrame:
    """Add WOE columns to the DataFrame by joining with binning rules."""
    start_time = time.time()
    log_function_entry('add_woe_columns', num_vars=len(x_vars), prefix=prefix, woe_prefix=woe_prefix)
    
    result_df = df.copy()
    
    for var in x_vars:
        logger.debug(f"Adding WOE column for: {var}")
        var_bins = bins_df[(bins_df['var'] == var) & (bins_df['bin'] != 'Total')].copy()
        
        if var_bins.empty:
            logger.warning(f"No bins found for WOE mapping: {var}")
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
                unmapped_bins = result_df.loc[result_df[woe_col].isna(), bin_col].unique()
                logger.error(f"Unmapped bins: {list(unmapped_bins)}")
                
                for unmapped_bin in unmapped_bins:
                    if unmapped_bin is None or pd.isna(unmapped_bin):
                        na_woe_bins = var_bins[var_bins['bin'].str.contains('is.na', na=False)]
                        if not na_woe_bins.empty:
                            na_woe = na_woe_bins.iloc[0]['woe']
                            result_df.loc[result_df[bin_col].isna(), woe_col] = na_woe
                            logger.info(f"Assigned NA bin WOE: {na_woe}")
    
    log_function_exit('add_woe_columns', f"DataFrame({result_df.shape})", time.time() - start_time)
    return result_df


# =============================================================================
# Configuration
# =============================================================================
min_prop = 0.05
logger.info(f"Configuration: min_prop={min_prop}")

# =============================================================================
# Read Input Data
# =============================================================================
logger.info("=" * 70)
logger.info("WOE EDITOR - DEBUG VERSION")
logger.info("=" * 70)
logger.info(f"Script started at: {datetime.now().isoformat()}")

logger.info("Reading input data from KNIME...")
df = knio.input_tables[0].to_pandas()
logger.info(f"Input data shape: {df.shape}")
logger.info(f"Input columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
logger.debug(f"Input dtypes:\n{df.dtypes}")

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
    logger.debug(f"DependentVariable flow variable: {dv}")
except Exception as e:
    logger.debug(f"Error reading DependentVariable: {e}")

try:
    target = knio.flow_variables.get("TargetCategory", None)
    logger.debug(f"TargetCategory flow variable: {target}")
except Exception as e:
    logger.debug(f"Error reading TargetCategory: {e}")

try:
    optimize_all = knio.flow_variables.get("OptimizeAll", False)
    logger.debug(f"OptimizeAll flow variable: {optimize_all}")
except Exception as e:
    logger.debug(f"Error reading OptimizeAll: {e}")

try:
    group_na = knio.flow_variables.get("GroupNA", False)
    logger.debug(f"GroupNA flow variable: {group_na}")
except Exception as e:
    logger.debug(f"Error reading GroupNA: {e}")

if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        logger.info(f"Headless mode activated with DV: {dv}")
    else:
        logger.error(f"DependentVariable '{dv}' not found in columns!")

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    log_progress("=" * 60)
    log_progress("WOE EDITOR - HEADLESS MODE (DEBUG VERSION)")
    log_progress("=" * 60)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"OptimizeAll: {optimize_all}, GroupNA: {group_na}")
    
    iv_list = [col for col in df.columns if col != dv]
    logger.debug(f"Initial IV list: {len(iv_list)} variables")
    
    # Filter out constant/zero-variance variables
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
        logger.debug(f"Constant vars: {constant_vars[:10]}{'...' if len(constant_vars) > 10 else ''}")
    
    iv_list = valid_vars
    log_progress(f"Variables to process: {len(iv_list)}")
    
    # Step 1: Initial binning
    step_start = time.time()
    log_progress("STEP 1/5: Computing initial bins...")
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    log_progress(f"STEP 1/5 complete in {format_time(time.time() - step_start)}")
    
    # Step 2: Merge pure bins
    step_start = time.time()
    if 'purNode' in bins_result.var_summary.columns:
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    if pure_count > 0:
        log_progress(f"STEP 2/5: Merging {int(pure_count)} pure bins...")
        bins_result = merge_pure_bins(bins_result)
        log_progress(f"STEP 2/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 2/5: Skipped (no pure bins found)")
    
    # Step 3: Group NA
    if group_na:
        step_start = time.time()
        log_progress("STEP 3/5: Grouping NA values...")
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        log_progress(f"STEP 3/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 3/5: Skipped (GroupNA=False)")
    
    # Step 4: Optimize All
    if optimize_all:
        step_start = time.time()
        log_progress("STEP 4/5: Optimizing monotonicity...")
        bins_mod = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        
        decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
        if decr_vars:
            log_progress(f"  - Forcing decreasing trend on {len(decr_vars)} variables...")
            bins_mod = force_decr_trend(bins_mod, decr_vars)
        
        incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
        if incr_vars:
            log_progress(f"  - Forcing increasing trend on {len(incr_vars)} variables...")
            bins_mod = force_incr_trend(bins_mod, incr_vars)
        
        bins_result = bins_mod
        log_progress(f"STEP 4/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 4/5: Skipped (OptimizeAll=False)")
    
    # Step 5: Apply WOE transformation
    step_start = time.time()
    log_progress("STEP 5/5: Applying WOE transformation...")
    
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
    
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
    
    log_progress(f"STEP 5/5 complete in {format_time(time.time() - step_start)}")
    log_progress("=" * 60)
    log_progress(f"COMPLETE: Processed {len(all_vars)} variables")
    log_progress("=" * 60)

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    logger.info("Running in interactive mode - Shiny UI not available in DEBUG version")
    logger.info("Please provide DependentVariable flow variable for headless mode")
    
    # Create empty outputs for interactive mode
    df_with_woe = df.copy()
    df_only_woe = pd.DataFrame()
    bins = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

logger.info("Writing output tables...")

# Output 1: Original input DataFrame
logger.debug("Writing output table 0: Original data")
knio.output_tables[0] = knio.Table.from_pandas(df)

# Output 2: df_with_woe
logger.debug(f"Writing output table 1: df_with_woe ({df_with_woe.shape})")
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)

# Output 3: df_only_woe
logger.debug(f"Writing output table 2: df_only_woe ({df_only_woe.shape})")
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)

# Output 4: df_only_bins
b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
df_only_bins = df_with_woe[b_columns].copy() if b_columns else pd.DataFrame()
logger.debug(f"Writing output table 3: df_only_bins ({df_only_bins.shape})")
knio.output_tables[3] = knio.Table.from_pandas(df_only_bins)

# Output 5: bins
logger.debug(f"Writing output table 4: bins ({bins.shape if isinstance(bins, pd.DataFrame) else 'empty'})")
knio.output_tables[4] = knio.Table.from_pandas(bins if isinstance(bins, pd.DataFrame) else pd.DataFrame())

logger.info("=" * 70)
logger.info("OUTPUT SUMMARY:")
logger.info(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
logger.info(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
logger.info(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
logger.info(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols)")
logger.info(f"  Port 5: Bin Rules ({len(bins) if isinstance(bins, pd.DataFrame) else 0} rows)")
logger.info("=" * 70)
logger.info(f"WOE Editor (DEBUG) completed at: {datetime.now().isoformat()}")

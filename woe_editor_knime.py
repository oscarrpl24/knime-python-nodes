# =============================================================================
# WOE Editor for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's WOE Editor functionality
# Compatible with KNIME 5.9, Python 3.9
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
# Version: 1.0
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

warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# Use random port to avoid conflicts when running multiple instances
BASE_PORT = 8050
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
# Data Classes
# =============================================================================

@dataclass
class BinResult:
    """Container for binning results"""
    var_summary: pd.DataFrame  # Summary stats for each variable
    bin: pd.DataFrame  # Detailed bin information


# =============================================================================
# Core Binning Functions
# =============================================================================

def calculate_woe(freq_good: np.ndarray, freq_bad: np.ndarray) -> np.ndarray:
    """
    Calculate Weight of Evidence (WOE) for each bin.
    WOE = ln((% of Bads in bin) / (% of Goods in bin))
    """
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    dist_good = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad = np.where(dist_bad == 0, 0.0001, dist_bad)
    
    woe = np.round(np.log(dist_bad / dist_good), 5)
    return woe


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
    
    dist_good_safe = np.where(dist_good == 0, 0.0001, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, 0.0001, dist_bad)
    
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


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.05,
    max_bins: int = 10
) -> List[float]:
    """Use decision tree to find optimal split points for numeric variables."""
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        return []
    
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        return []
    
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    thresholds = sorted(set(thresholds))
    
    return thresholds


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
    
    return pd.DataFrame(bins_data)


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """Create bin DataFrame for factor/categorical variable."""
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
    min_prop: float = 0.05,
    max_bins: int = 10
) -> BinResult:
    """
    Get optimal bins for multiple variables.
    This is the main entry point, equivalent to logiBin::getBins in R.
    """
    all_bins = []
    var_summaries = []
    
    total_vars = len(x_vars)
    start_time = time.time()
    last_log_time = start_time
    processed_count = 0
    times_per_var = []
    
    log_progress(f"Starting binning for {total_vars} variables (Algorithm: DecisionTree)")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        if var not in df.columns:
            continue
            
        var_type = get_var_type(df[var])
        
        if var_type == 'numeric':
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            bin_df = _create_factor_bins(df, var, y_var)
        
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
    vars_to_process: Union[str, List[str]]
) -> BinResult:
    """Combine NA bin with the adjacent bin that has the closest bad rate."""
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
                # This shouldn't happen often, but handle it
                other_bins = var_bins[var_bins.index != pure_idx]
                if other_bins.empty:
                    break
                # Pick the one with smallest count difference
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
    new_var_bins = _create_factor_bins(df, var, y_var)
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
        # Assign them to the most common bin or a special "Unmatched" bin
        unassigned_mask = result_df[new_col].isna() | (result_df[new_col] == None)
        if unassigned_mask.any():
            # Try to assign to NA bin if it exists, otherwise use first bin as fallback
            if na_rule is not None:
                fallback_bin = na_rule
            elif not var_bins.empty:
                # Use the bin with highest count as fallback
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
                .btn-primary { background-color: #75AFD7; border-color: #75AFD7; }
                .btn-success { background-color: #9ECC53; border-color: #9ECC53; }
                .btn-danger { background-color: #B5202E; border-color: #B5202E; }
                .btn-secondary { background-color: #8A9399; border-color: #8A9399; }
                .btn-dark { background-color: #525E66; border-color: #525E66; }
                h4 { font-weight: bold; text-align: center; margin: 20px 0; }
                .divider { width: 10px; display: inline-block; }
            """)
        ),
        
        ui.h4("WOE Editor"),
        
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
                
                if df[dv].isna().sum() <= 0:
                    try:
                        all_bins = get_bins(df, dv, iv_list, min_prop=min_prop)
                        all_bins_rv.set(all_bins)
                        
                        bin_vars = all_bins.var_summary['var'].tolist()
                        ui.update_select("iv", choices=bin_vars, 
                                       selected=bin_vars[0] if bin_vars else None)
                    except Exception as e:
                        print(f"Error calculating bins: {e}")
        
        @reactive.Effect
        @reactive.event(input.iv)
        def update_iv_bins():
            iv = input.iv()
            dv = input.dv()
            if iv and dv and not modified_action_rv.get():
                try:
                    bins = get_bins(df, dv, [iv], min_prop=min_prop)
                    bins_rv.set(bins)
                    initial_bins_rv.set(bins)
                except Exception as e:
                    print(f"Error getting bins for {iv}: {e}")
        
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
        
        @reactive.Effect
        @reactive.event(input.group_na_btn)
        def group_na():
            bins = bins_rv.get()
            iv = input.iv()
            if bins is not None and iv:
                new_bins = na_combine(bins, iv)
                bins_rv.set(new_bins)
                modified_action_rv.set(False)
        
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
        
        @reactive.Effect
        @reactive.event(input.reset_btn)
        def reset_bins():
            iv = input.iv()
            dv = input.dv()
            if iv and dv:
                modified_action_rv.set(False)
                bins = get_bins(df, dv, [iv], min_prop=min_prop)
                bins_rv.set(bins)
        
        @reactive.Effect
        @reactive.event(input.optimize_btn)
        def optimize_var():
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
            all_bins = all_bins_rv.get()
            if all_bins is not None:
                modified_action_rv.set(True)
                
                bins_mod = na_combine(all_bins, all_bins.var_summary['var'].tolist())
                
                decr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'D']['var'].tolist()
                if decr_vars:
                    bins_mod = force_decr_trend(bins_mod, decr_vars)
                
                incr_vars = bins_mod.var_summary[bins_mod.var_summary['trend'] == 'I']['var'].tolist()
                if incr_vars:
                    bins_mod = force_incr_trend(bins_mod, incr_vars)
                
                all_bins_mod_rv.set(bins_mod)
        
        @reactive.Calc
        def get_display_bins():
            if modified_action_rv.get():
                all_mod = all_bins_mod_rv.get()
                iv = input.iv()
                if all_mod is not None and iv:
                    var_bins = all_mod.bin[all_mod.bin['var'] == iv].copy()
                    return var_bins
            else:
                bins = bins_rv.get()
                if bins is not None:
                    return bins.bin.copy()
            return pd.DataFrame()
        
        @output
        @render.data_frame
        def woe_table():
            display_bins = get_display_bins()
            if display_bins.empty:
                return render.DataGrid(pd.DataFrame())
            
            non_total = display_bins[display_bins['bin'] != 'Total'].copy()
            if not non_total.empty:
                non_total['woe'] = calculate_woe(non_total['goods'].values, non_total['bads'].values)
            
            total_row = display_bins[display_bins['bin'] == 'Total'].copy()
            if not total_row.empty:
                total_row['woe'] = np.nan
            
            result = pd.concat([non_total, total_row], ignore_index=True)
            
            display_cols = ['bin', 'count', 'goods', 'bads', 'propn', 'bad_rate', 'woe', 'iv']
            display_cols = [c for c in display_cols if c in result.columns]
            
            return render.DataGrid(result[display_cols], selection_mode="rows", height="350px")
        
        @output
        @render_plotly
        def woe_graph():
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=plot_data['bad_rate'] / 100,
                name='Bad Rate', mode='lines+markers', line=dict(color='#3498db')
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=(100 - plot_data['bad_rate']) / 100,
                name='Good Rate', mode='lines+markers', line=dict(color='#2ecc71')
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_data['bin'], y=plot_data['woe'],
                name='WOE', mode='lines+markers', line=dict(color='#e74c3c'), yaxis='y2'
            ))
            
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
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            plot_data['woe'] = calculate_woe(plot_data['goods'].values, plot_data['bads'].values)
            
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
            display_bins = get_display_bins()
            if display_bins.empty:
                return go.Figure()
            
            plot_data = display_bins[display_bins['bin'] != 'Total'].copy()
            if plot_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=plot_data['bin'], x=100 - plot_data['bad_rate'],
                name='Good', orientation='h', marker_color='#9ECC53',
                text=100 - plot_data['bad_rate'], textposition='inside'
            ))
            
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
            display_bins = get_display_bins()
            initial = initial_bins_rv.get()
            
            if display_bins.empty:
                return render.DataGrid(pd.DataFrame({
                    'Initial IV': [0], 'Final IV': [0],
                    'Initial Entropy': [0], 'Final Entropy': [0]
                }))
            
            total_row = display_bins[display_bins['bin'] == 'Total']
            final_iv = total_row['iv'].iloc[0] if not total_row.empty else 0
            final_ent = total_row['ent'].iloc[0] if not total_row.empty else 0
            
            initial_iv = 0
            initial_ent = 0
            if initial is not None:
                init_total = initial.bin[initial.bin['bin'] == 'Total']
                if not init_total.empty:
                    initial_iv = init_total['iv'].iloc[0]
                    initial_ent = init_total['ent'].iloc[0]
            
            measurements = pd.DataFrame({
                'Initial IV': [round(initial_iv, 4)],
                'Final IV': [round(final_iv, 4)],
                'Initial Entropy': [round(initial_ent, 4)],
                'Final Entropy': [round(final_ent, 4)]
            })
            
            return render.DataGrid(measurements)
        
        @reactive.Effect
        @reactive.event(input.run_btn)
        async def run_and_close():
            dv = input.dv()
            
            if modified_action_rv.get():
                final_bins = all_bins_mod_rv.get()
            else:
                final_bins = all_bins_rv.get()
            
            if final_bins is None or dv is None:
                return
            
            all_vars = final_bins.var_summary['var'].tolist()
            
            rules = final_bins.bin[final_bins.bin['bin'] != 'Total'].copy()
            rules['woe'] = calculate_woe(rules['goods'].values, rules['bads'].values)
            
            for var in all_vars:
                var_mask = rules['var'] == var
                rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
                    lambda x: x.replace(var, '').replace(' %in% c', '').strip()
                )
            
            df_with_bins = create_binned_columns(final_bins, df, all_vars)
            df_with_woe = add_woe_columns(df_with_bins, rules, all_vars)
            
            woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
            df_only_woe = df_with_woe[woe_cols + [dv]].copy()
            
            app_results['df_with_woe'] = df_with_woe
            app_results['df_only_woe'] = df_only_woe
            app_results['bins'] = rules
            app_results['dv'] = dv
            app_results['completed'] = True
            
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def run_woe_editor(df: pd.DataFrame, min_prop: float = 0.05) -> Dict[str, Any]:
    """Run the WOE Editor application and return results."""
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
    
    app = create_woe_editor_app(df, min_prop)
    webbrowser.open(f'http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port)
    return app.results


# =============================================================================
# Configuration
# =============================================================================
min_prop = 0.05

# =============================================================================
# Read Input Data
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Check for Flow Variables (Headless Mode)
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
    log_progress("WOE EDITOR - HEADLESS MODE (Original/DecisionTree)")
    log_progress("=" * 60)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"OptimizeAll: {optimize_all}, GroupNA: {group_na}")
    
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
    log_progress("STEP 1/5: Computing initial bins...")
    bins_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    log_progress(f"STEP 1/5 complete in {format_time(time.time() - step_start)}")
    
    # Step 2: Merge pure bins (always - prevents infinite WOE)
    step_start = time.time()
    if 'purNode' in bins_result.var_summary.columns:
        # purNode is 'Y' or 'N' - count variables with pure bins
        pure_count = (bins_result.var_summary['purNode'] == 'Y').sum()
    else:
        pure_count = 0
    if pure_count > 0:
        log_progress(f"STEP 2/5: Merging {int(pure_count)} pure bins (prevents infinite WOE)...")
        bins_result = merge_pure_bins(bins_result)
        log_progress(f"STEP 2/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 2/5: Skipped (no pure bins found)")
    
    # Step 3: Group NA (optional)
    if group_na:
        step_start = time.time()
        log_progress("STEP 3/5: Grouping NA values...")
        bins_result = na_combine(bins_result, bins_result.var_summary['var'].tolist())
        log_progress(f"STEP 3/5 complete in {format_time(time.time() - step_start)}")
    else:
        log_progress("STEP 3/5: Skipped (GroupNA=False)")
    
    # Step 4: Optimize All (optional)
    if optimize_all:
        step_start = time.time()
        log_progress("STEP 4/5: Optimizing monotonicity for all variables...")
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
    log_progress("STEP 5/5: Applying WOE transformation to data...")
    
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
    
    log_progress(f"STEP 4/4 complete in {format_time(time.time() - step_start)}")
    log_progress("=" * 60)
    log_progress(f"COMPLETE: Processed {len(all_vars)} variables")
    log_progress("=" * 60)

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    print("Running in interactive mode - launching Shiny UI...")
    
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

print("=" * 70)
print("OUTPUT SUMMARY:")
print(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
print(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
print(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
print(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols) ** USE FOR SCORECARD **")
print(f"  Port 5: Bin Rules ({len(bins)} rows - metadata)")
print("=" * 70)
print("WOE Editor completed successfully")

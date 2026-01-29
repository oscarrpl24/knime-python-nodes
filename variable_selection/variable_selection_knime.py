# =============================================================================
# Variable Selection with EBM Interaction Discovery for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's variable selection functionality
# with added EBM-based interaction discovery
# Compatible with KNIME 5.9, Python 3.9
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When all required flow variables are provided
#
# Inputs:
# 1. df_with_woe - DataFrame with WOE columns (from WOE Editor output 2)
# 2. bins - Binning rules (from WOE Editor output 4)
#
# Outputs:
# 1. measures - Predictive power measures for all variables (with ranks and EBM importance)
# 2. selected_data - Selected WOE variables + EBM-missed vars + interaction columns + DV (ready for stepwise)
# 3. ebm_report - EBM interaction discovery report
# 4. correlation_matrix - Correlation matrix for selected variables
# 5. vif_report - Variance Inflation Factor report for multicollinearity detection
#
# Release Date: 2026-01-16
# Version: 1.2
#
# Flow Variables (for headless mode):
# - DependentVariable (string): Binary target variable name
# - MeasuresOfPredictivePower (string): Comma-separated measures
#     Options: EntropyExplained, InformationValue, OddsRatio, LikelihoodRatio, PearsonChiSquare, Gini
# - NumberOfVariables (integer): Top N variables for each measure
# - Criteria (string): "Union" or "Intersection"
# - Degree (integer): For Intersection, minimum measures that must agree
# - MaxInteractions (integer, default 20): Max EBM interactions to detect
# - TopInteractions (integer, default 10): Top interactions to include in output
# - AutoAddMissed (boolean, default True): Auto-add EBM/XGBoost-missed variables
# - MaxMissedToAdd (integer, default 0): Max missed variables to auto-add (0 = add ALL)
# - VIFThreshold (float, default 0): Variables with VIF >= threshold are removed (0 = no filtering)
# - UseXGBoost (boolean, default True): Use XGBoost GPU for feature discovery
# - XGBEstimators (integer, default 1000): Number of XGBoost boosting rounds (~1 min GPU)
# - XGBMaxDepth (integer, default 6): Maximum tree depth for XGBoost
# - XGBLearningRate (float, default 0.02): XGBoost learning rate
# - XGBColsampleByTree (float, default 0.7): Fraction of columns to sample per tree
# - XGBSubsample (float, default 0.8): Fraction of rows to sample per tree
# - XGBNumGPUs (integer, default 2): Number of GPUs to use for parallel XGBoost training
# =============================================================================

# =============================================================================
# FIX NUMPY COMPATIBILITY (run once, then set to False)
# =============================================================================
FIX_PACKAGES = False  # Set to True to reinstall packages, then set back to False

if FIX_PACKAGES:
    import subprocess
    import sys
    print("Reinstalling packages to fix compatibility...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'numpy'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'scikit-learn'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'interpret'])
    print("Done! Set FIX_PACKAGES = False and run again.")
    raise SystemExit("Packages reinstalled. Please run the node again with FIX_PACKAGES = False")

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import gc
import sys
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
# Use random port to avoid conflicts when running multiple instances
BASE_PORT = 8052  # Different from model_analyzer to avoid conflicts
RANDOM_PORT_RANGE = 1000  # Will pick random port between BASE_PORT and BASE_PORT + RANDOM_PORT_RANGE

# Process isolation: Set unique temp directories per instance
import os as _os
INSTANCE_ID = f"{_os.getpid()}_{random.randint(10000, 99999)}"
_os.environ['NUMEXPR_MAX_THREADS'] = '1'  # Prevent numexpr threading conflicts
_os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts
_os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading conflicts
_os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading conflicts

# =============================================================================
# Install/Import Dependencies
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install package if not available."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package])

def fix_numpy_compatibility():
    """Fix numpy binary compatibility issues."""
    import subprocess
    print("Attempting to fix NumPy compatibility issue...")
    try:
        subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall', 'numpy'])
        subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall', 'scikit-learn'])
        print("Packages reinstalled. Please restart the KNIME workflow.")
    except Exception as e:
        print(f"Could not fix automatically: {e}")
        print("Please run these commands in your KNIME Python environment:")
        print("  pip install --upgrade --force-reinstall numpy")
        print("  pip install --upgrade --force-reinstall scikit-learn")

# Try to import sklearn - if it fails with numpy error, try to fix
try:
    install_if_missing('scikit-learn', 'sklearn')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        fix_numpy_compatibility()
        raise RuntimeError(
            "NumPy binary incompatibility detected. "
            "Please restart your KNIME workflow after the packages are reinstalled."
        )
    raise

install_if_missing('interpret')
install_if_missing('shiny')
install_if_missing('shinywidgets')
install_if_missing('plotly')

# Try interpret - this may also have numpy issues
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    EBM_AVAILABLE = True
except (ValueError, ImportError) as e:
    print(f"WARNING: EBM not available ({e})")
    print("EBM-based interaction discovery will be disabled.")
    EBM_AVAILABLE = False
    ExplainableBoostingClassifier = None

# Try XGBoost with GPU support
install_if_missing('xgboost')
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    
    # Check for GPU availability
    try:
        # Try to create a small GPU-enabled booster to check if CUDA works
        test_params = {'device': 'cuda', 'tree_method': 'hist'}
        test_dmat = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        test_booster = xgb.train(test_params, test_dmat, num_boost_round=1, verbose_eval=False)
        XGBOOST_GPU_AVAILABLE = True
        print("XGBoost GPU (CUDA) available - will use GPU acceleration")
    except Exception as gpu_err:
        XGBOOST_GPU_AVAILABLE = False
        print(f"XGBoost GPU not available ({gpu_err}), will use CPU")
except ImportError as e:
    print(f"WARNING: XGBoost not available ({e})")
    XGBOOST_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False
    xgb = None

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
    import plotly.express as px
    SHINY_AVAILABLE = True
except ImportError:
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MeasuresResult:
    """Container for predictive measures results"""
    measures: pd.DataFrame  # Measures for each variable
    selected_vars: List[str]  # List of selected variable names


@dataclass
class EBMReport:
    """Container for EBM interaction discovery results"""
    feature_importances: pd.DataFrame  # EBM feature importances
    interactions: pd.DataFrame  # Detected interactions with magnitudes
    missed_by_traditional: List[str]  # Variables important in EBM but not in traditional selection
    ebm_model: Any  # The trained EBM model


@dataclass
class XGBoostReport:
    """Container for XGBoost feature discovery results"""
    feature_importances: pd.DataFrame  # XGBoost feature importances (gain, cover, weight)
    interactions: pd.DataFrame  # Top feature pairs based on SHAP interaction or tree structure
    missed_by_traditional: List[str]  # Variables important in XGBoost but not in traditional selection
    xgb_model: Any  # The trained XGBoost model
    gpu_used: bool  # Whether GPU was used for training


@dataclass
class BinResult:
    """Container for binning results"""
    var_summary: pd.DataFrame  # Summary stats for each variable
    bin: pd.DataFrame  # Detailed bin information


# =============================================================================
# Binning Functions (equivalent to R's logiBin::getBins)
# =============================================================================

def calculate_bin_entropy(goods: int, bads: int) -> float:
    """Calculate entropy for a bin."""
    total = goods + bads
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    p_good = goods / total
    p_bad = bads / total
    
    entropy_val = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    return round(entropy_val, 4)


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
    from sklearn.tree import DecisionTreeClassifier
    
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
        lambda row: calculate_bin_entropy(row['goods'], row['bads']), 
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
    max_bins: int = 10
) -> BinResult:
    """
    Get optimal bins for multiple variables.
    This is the main entry point, equivalent to logiBin::getBins in R.
    """
    all_bins = []
    var_summaries = []
    
    for var in x_vars:
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
    
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    var_summary_df = pd.DataFrame(var_summaries)
    
    return BinResult(var_summary=var_summary_df, bin=combined_bins)


# =============================================================================
# Predictive Measures Functions (from R implementation)
# =============================================================================

def entropy(probs: np.ndarray) -> float:
    """Core entropy calculation."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def input_entropy(bins_df: pd.DataFrame) -> float:
    """Calculate input entropy for a variable's bins."""
    # Get totals from the last row (Total row)
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = total_goods + total_bads
    
    if total == 0:
        return 0.0
    
    probs = np.array([total_goods / total, total_bads / total])
    return round(entropy(probs), 5)


def output_entropy(bins_df: pd.DataFrame) -> float:
    """Calculate output (conditional) entropy for a variable's bins."""
    # Exclude the Total row
    bins_only = bins_df.iloc[:-1]
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    weighted_entropy = 0.0
    for _, row in bins_only.iterrows():
        count = row['count']
        if count == 0:
            continue
        goods = row['goods']
        bads = row['bads']
        
        probs = []
        if goods > 0:
            probs.append(goods / count)
        if bads > 0:
            probs.append(bads / count)
        
        if len(probs) > 0:
            bin_entropy = entropy(np.array(probs))
            weighted_entropy += (count / total) * bin_entropy
    
    return round(weighted_entropy, 5)


def gini_impurity(totals: np.ndarray, overall_total: float) -> float:
    """Core Gini impurity calculation."""
    if overall_total == 0:
        return 0.0
    return 1 - np.sum((totals / overall_total) ** 2)


def input_gini(bins_df: pd.DataFrame) -> float:
    """Calculate input Gini for a variable."""
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = bins_df['count'].iloc[-1]
    
    totals = np.array([total_goods, total_bads])
    return round(gini_impurity(totals, total), 5)


def output_gini(bins_df: pd.DataFrame) -> float:
    """Calculate output (weighted) Gini for a variable."""
    bins_only = bins_df.iloc[:-1]
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    weighted_gini = 0.0
    for _, row in bins_only.iterrows():
        count = row['count']
        if count == 0:
            continue
        goods = row['goods']
        bads = row['bads']
        
        bin_totals = np.array([goods, bads])
        bin_gini = gini_impurity(bin_totals, count)
        weighted_gini += (count / total) * bin_gini
    
    return round(weighted_gini, 5)


def chi_square(observed: np.ndarray, expected: np.ndarray) -> float:
    """Calculate Pearson Chi-Square statistic."""
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    
    # Avoid division by zero
    mask = expected > 0
    if not np.any(mask):
        return 0.0
    
    chi_sq = np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask])
    return chi_sq


def likelihood_ratio(observed: np.ndarray, expected: np.ndarray) -> float:
    """Calculate Likelihood Ratio (G-test) statistic."""
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    
    # Avoid log(0)
    mask = (observed > 0) & (expected > 0)
    if not np.any(mask):
        return 0.0
    
    g_stat = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
    return g_stat


def chi_mls_calc(bins_df: pd.DataFrame, method: str = 'chisquare') -> float:
    """Calculate Chi-Square or Likelihood Ratio for a variable's bins."""
    bins_only = bins_df.iloc[:-1]
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    prop_goods = total_goods / total
    prop_bads = total_bads / total
    
    # Calculate expected values
    exp_goods = (bins_only['count'] / total) * prop_goods * total
    exp_bads = (bins_only['count'] / total) * prop_bads * total
    
    expected = np.concatenate([exp_goods.values, exp_bads.values])
    observed = np.concatenate([bins_only['goods'].values, bins_only['bads'].values])
    
    if method == 'chisquare':
        return round(chi_square(observed, expected), 5)
    elif method == 'mls':
        return round(likelihood_ratio(observed, expected), 5)
    else:
        return 0.0


def odds_ratio(bins_df: pd.DataFrame) -> Optional[float]:
    """Calculate Odds Ratio for binary factor variables only."""
    bins_only = bins_df.iloc[:-1]
    
    # Only calculate for binary factors (exactly 2 bins)
    if len(bins_only) != 2:
        return None
    
    goods1 = bins_only['goods'].iloc[0]
    goods2 = bins_only['goods'].iloc[1]
    bads1 = bins_only['bads'].iloc[0]
    bads2 = bins_only['bads'].iloc[1]
    
    # Avoid division by zero
    if goods2 == 0 or bads2 == 0 or bads1 == 0:
        return None
    
    prop_good = goods1 / goods2
    prop_bad = bads1 / bads2
    
    if prop_bad == 0:
        return None
    
    return round(prop_good / prop_bad, 5)


def calculate_all_measures(
    bins_df: pd.DataFrame,
    var_summary: pd.DataFrame,
    measures_to_calc: List[str]
) -> pd.DataFrame:
    """Calculate all selected predictive measures for all variables."""
    
    variables = var_summary['var'].unique().tolist()
    results = []
    
    for var in variables:
        var_bins = bins_df[bins_df['var'] == var].copy()
        var_info = var_summary[var_summary['var'] == var]
        
        if var_bins.empty:
            continue
        
        # Check if binary factor (for odds ratio)
        is_binary = len(var_bins[var_bins['bin'] != 'Total']) == 2
        if '%in%' in str(var_bins['bin'].values):
            is_binary = True
        
        row = {'Variable': var}
        
        # Calculate each measure
        if 'EntropyExplained' in measures_to_calc:
            in_ent = input_entropy(var_bins)
            out_ent = output_entropy(var_bins)
            if in_ent > 0:
                row['Entropy'] = round(1 - (out_ent / in_ent), 5)
            else:
                row['Entropy'] = 0.0
        
        if 'InformationValue' in measures_to_calc:
            if not var_info.empty and 'iv' in var_info.columns:
                row['Information Value'] = var_info['iv'].iloc[0]
            else:
                # Calculate IV from bins
                total_row = var_bins[var_bins['bin'] == 'Total']
                if not total_row.empty and 'iv' in total_row.columns:
                    row['Information Value'] = total_row['iv'].iloc[0]
                else:
                    row['Information Value'] = 0.0
        
        if 'OddsRatio' in measures_to_calc:
            if is_binary:
                # Use raw OR value (matches R behavior: higher OR = better)
                row['Odds Ratio'] = odds_ratio(var_bins)
            else:
                row['Odds Ratio'] = None
        
        if 'LikelihoodRatio' in measures_to_calc:
            row['Likelihood Ratio'] = chi_mls_calc(var_bins, method='mls')
        
        if 'PearsonChiSquare' in measures_to_calc:
            row['Chi-Square'] = chi_mls_calc(var_bins, method='chisquare')
        
        if 'Gini' in measures_to_calc:
            in_gini = input_gini(var_bins)
            out_gini = output_gini(var_bins)
            if in_gini > 0:
                row['Gini'] = round(1 - (out_gini / in_gini), 5)
            else:
                row['Gini'] = 0.0
        
        results.append(row)
    
    return pd.DataFrame(results)


def filter_variables(
    measures_df: pd.DataFrame,
    criteria: str,
    num_of_variables: int,
    degree: int
) -> pd.DataFrame:
    """
    Filter variables based on criteria.
    
    criteria: 'Union' or 'Intersection'
    num_of_variables: Top N variables to consider FOR EACH MEASURE
    degree: For Intersection, minimum number of measures a variable must be in top N
    
    Returns: All variables that meet the criteria (can be more or less than num_of_variables)
    """
    df = measures_df.copy()
    
    # Define measure columns and their sort order (True = higher is better)
    # All measures: HIGHER = better predictor
    measure_cols = {
        'Entropy': True,  # Higher Entropy Explained = more predictive
        'Information Value': True,  # Higher is better
        'Odds Ratio': True,  # Higher is better
        'Likelihood Ratio': True,  # Higher is better
        'Chi-Square': True,  # Higher is better
        'Gini': True  # Higher is better
    }
    
    # For each measure, identify which variables are in the top N
    top_n_sets = {}
    
    for col, higher_is_better in measure_cols.items():
        if col not in df.columns:
            continue
        
        # Get non-null values and sort
        valid_df = df[df[col].notna()].copy()
        if len(valid_df) == 0:
            continue
        
        # Sort and get top N variable names
        sorted_df = valid_df.sort_values(col, ascending=not higher_is_better)
        top_n = min(num_of_variables, len(sorted_df))
        top_vars = sorted_df.head(top_n)['Variable'].tolist()
        top_n_sets[col] = set(top_vars)
        
        # Show the cutoff value and top 5 variables for this measure
        cutoff_val = sorted_df.iloc[top_n - 1][col] if top_n > 0 else None
        cutoff_str = f"{cutoff_val:.4f}" if cutoff_val is not None else "N/A"
        top5_vars = sorted_df.head(5)['Variable'].tolist()
        top5_vals = sorted_df.head(5)[col].tolist()
        print(f"  Top {top_n} for {col}: cutoff={cutoff_str}")
        print(f"    Top 5: {[(v, round(val, 4)) for v, val in zip(top5_vars, top5_vals)]}")
    
    # Count how many top-N lists each variable appears in AND track which measures
    df['ListCount'] = 0
    df['InMeasures'] = ''
    
    measure_names = list(top_n_sets.keys())
    
    for idx, row in df.iterrows():
        var_name = row['Variable']
        in_measures = []
        for measure_name in measure_names:
            if var_name in top_n_sets[measure_name]:
                in_measures.append(measure_name[:3])  # Abbreviate
        df.at[idx, 'ListCount'] = len(in_measures)
        df.at[idx, 'InMeasures'] = ','.join(in_measures)
    
    # Show overlap between measures
    print(f"\n  Measure overlap analysis:")
    for i, m1 in enumerate(measure_names):
        for m2 in measure_names[i+1:]:
            overlap = len(top_n_sets[m1] & top_n_sets[m2])
            print(f"    {m1[:3]} & {m2[:3]}: {overlap} common variables")
    
    # Show cumulative distribution of list counts (at least N lists)
    print(f"\n  List count distribution (cumulative):")
    total_vars = len(df)
    for count in range(len(top_n_sets) + 1):
        n_vars = len(df[df['ListCount'] >= count])
        print(f"    In {count}+ lists: {n_vars} variables")
    
    # Apply selection criteria
    if criteria == 'Union':
        # Select if in ANY list (ListCount >= 1)
        df['Selected'] = df['ListCount'] >= 1
        print(f"\n  Union: selecting variables in at least 1 list")
    elif criteria == 'Intersection':
        # Select if in at least 'degree' lists
        df['Selected'] = df['ListCount'] >= degree
        print(f"\n  Intersection degree {degree}: selecting variables in at least {degree} lists")
    else:
        df['Selected'] = False
    
    selected_count = df['Selected'].sum()
    print(f"  Result: {selected_count} variables selected")
    
    return df


# =============================================================================
# EBM Interaction Discovery Functions
# =============================================================================

def train_ebm_for_discovery(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    max_interactions: int = 20
) -> Optional[EBMReport]:
    """
    Train an EBM model to discover important interactions.
    Returns None if EBM is not available.
    """
    if not EBM_AVAILABLE or ExplainableBoostingClassifier is None:
        print("EBM not available - skipping interaction discovery")
        return None
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train EBM
    ebm = ExplainableBoostingClassifier(
        max_bins=32,
        interactions=max_interactions,
        max_interaction_bins=16,
        outer_bags=8,
        inner_bags=4,
        random_state=42
    )
    
    ebm.fit(X, y)
    
    # Get term names and importances (EBM uses term_importances(), not feature_importances_)
    term_names = ebm.term_names_
    term_importances_vals = ebm.term_importances()
    
    # Extract feature importances (main effects only, not interactions)
    # Main effects are terms that don't contain ' x ' in their name
    feature_importance_list = []
    for i, name in enumerate(term_names):
        if ' x ' not in name:  # Main effect (single feature)
            feature_importance_list.append({
                'Variable': name,
                'EBM_Importance': term_importances_vals[i]
            })
    
    importances = pd.DataFrame(feature_importance_list)
    if not importances.empty:
        importances = importances.sort_values('EBM_Importance', ascending=False)
    
    # Extract interactions
    interactions_list = []
    
    for i, name in enumerate(term_names):
        if ' x ' in name:  # Interaction term
            vars_in_interaction = name.split(' x ')
            if len(vars_in_interaction) == 2:
                interactions_list.append({
                    'Variable_1': vars_in_interaction[0],
                    'Variable_2': vars_in_interaction[1],
                    'Interaction_Name': name,
                    'Magnitude': term_importances_vals[i]
                })
    
    interactions_df = pd.DataFrame(interactions_list)
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values('Magnitude', ascending=False)
    
    return EBMReport(
        feature_importances=importances,
        interactions=interactions_df,
        missed_by_traditional=[],  # Will be filled later
        ebm_model=ebm
    )


def compare_selections(
    traditional_vars: List[str],
    ebm_importances: pd.DataFrame,
    top_n: int = 50
) -> List[str]:
    """
    Find variables important in EBM but missed by traditional selection.
    """
    # Get top N from EBM
    ebm_top = ebm_importances.nlargest(top_n, 'EBM_Importance')['Variable'].tolist()
    
    # Find variables in EBM top but not in traditional
    # Need to handle WOE_ prefix
    traditional_base = [v.replace('WOE_', '') for v in traditional_vars]
    
    missed = []
    for var in ebm_top:
        base_var = var.replace('WOE_', '')
        if base_var not in traditional_base and var not in traditional_vars:
            missed.append(var)
    
    return missed


# =============================================================================
# XGBoost GPU Feature Discovery
# =============================================================================

def train_xgboost_on_single_gpu(
    X: pd.DataFrame,  # Changed: pass raw data instead of DMatrix
    y: pd.Series,     # Changed: pass raw labels
    feature_cols: List[str],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    colsample_bytree: float,
    subsample: float,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    gpu_id: int = 0,
    seed: int = 42
) -> Tuple[Any, pd.DataFrame, bool]:
    """Train XGBoost on a specific GPU. Returns (model, importance_df, gpu_used).
    
    CRITICAL: Creates its own DMatrix to avoid GPU device conflicts.
    Each thread creates a DMatrix on its assigned GPU.
    """
    import os
    
    # Set CUDA_VISIBLE_DEVICES to isolate this thread to a single GPU
    # This ensures the DMatrix is created on the correct GPU
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if XGBOOST_GPU_AVAILABLE and gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # Create DMatrix on this thread's assigned GPU
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'min_child_weight': 3,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': 0,
            'seed': seed
        }
        
        gpu_used = False
        if XGBOOST_GPU_AVAILABLE:
            # Use cuda:0 because CUDA_VISIBLE_DEVICES remaps the GPU
            params['device'] = 'cuda:0'  
            params['tree_method'] = 'hist'
            gpu_used = True
        else:
            params['tree_method'] = 'hist'
        
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, verbose_eval=False)
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Get feature importances
    importance_gain = model.get_score(importance_type='gain')
    importance_cover = model.get_score(importance_type='cover')
    importance_weight = model.get_score(importance_type='weight')
    
    importance_data = []
    for feat in feature_cols:
        importance_data.append({
            'Variable': feat,
            'XGB_Gain': importance_gain.get(feat, 0),
            'XGB_Cover': importance_cover.get(feat, 0),
            'XGB_Weight': importance_weight.get(feat, 0),
            'XGB_Importance': importance_gain.get(feat, 0)
        })
    
    return model, pd.DataFrame(importance_data), gpu_used


def train_xgboost_for_discovery(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    use_gpu: bool = True,
    n_estimators: int = 3000,
    max_depth: int = 8,
    learning_rate: float = 0.01,
    discover_interactions: bool = True,
    top_interactions: int = 20,
    colsample_bytree: float = 0.5,
    subsample: float = 0.8,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    num_gpus: int = 2
) -> Optional[XGBoostReport]:
    """
    Train XGBoost models on MULTIPLE GPUs in parallel for robust feature discovery.
    
    With 2 GPUs: Trains 2 models simultaneously (different seeds), then
    averages feature importances for more robust rankings.
    
    OPTIMIZED SETTINGS for high-quality feature selection (less noise):
        - n_estimators: 3000 per GPU (~3-5 minutes training for stable importance)
        - max_depth: 8 (deeper trees for better feature discrimination)
        - learning_rate: 0.01 (slower learning = more reliable feature rankings)
        - colsample_bytree: 0.5 (50% column sampling - stronger regularization)
        - subsample: 0.8 (sample 80% rows per tree)
        - reg_alpha: 0.5 (L1 regularization to zero out weak features)
        - reg_lambda: 2.0 (L2 regularization to shrink marginal features)
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        feature_cols: List of feature column names
        use_gpu: Whether to use GPU acceleration (if available)
        n_estimators: Number of boosting rounds (higher = more stable rankings)
        max_depth: Maximum tree depth (higher = better discrimination)
        learning_rate: Learning rate (lower = more reliable rankings)
        discover_interactions: Whether to discover interactions from tree structure
        top_interactions: Number of top interactions to return
        colsample_bytree: Fraction of columns to sample per tree (lower = more regularization)
        subsample: Fraction of rows to sample per tree
        reg_alpha: L1 regularization (higher = zeros out weak features)
        reg_lambda: L2 regularization (higher = shrinks marginal feature importance)
    
    Returns:
        XGBoostReport with feature importances and interactions, or None if failed
    """
    if not XGBOOST_AVAILABLE or xgb is None:
        print("XGBoost not available - skipping XGBoost discovery")
        return None
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Determine number of GPUs to use
        actual_gpus = num_gpus if (use_gpu and XGBOOST_GPU_AVAILABLE) else 0
        
        if actual_gpus >= 2:
            # PARALLEL training on multiple GPUs
            print(f"  Training XGBoost on {actual_gpus} GPUs in PARALLEL: {n_estimators} rounds each")
            print(f"    depth={max_depth}, lr={learning_rate}, colsample={colsample_bytree}, L1={reg_alpha}, L2={reg_lambda}")
            
            start_time = time.time()
            models = []
            importance_dfs = []
            gpu_used = True
            
            with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                futures = {}
                for gpu_id in range(actual_gpus):
                    seed = 42 + gpu_id * 1000  # Different seed per GPU for diversity
                    futures[gpu_id] = executor.submit(
                        train_xgboost_on_single_gpu,
                        X, y, feature_cols, n_estimators, max_depth,  # Pass raw data, not DMatrix
                        learning_rate, colsample_bytree, subsample,
                        reg_alpha, reg_lambda, gpu_id, seed
                    )
                
                for gpu_id, future in futures.items():
                    try:
                        model, imp_df, _ = future.result()
                        models.append(model)
                        importance_dfs.append(imp_df)
                        print(f"    [GPU {gpu_id}] Completed {n_estimators} rounds")
                    except Exception as e:
                        print(f"    [GPU {gpu_id}] Failed: {str(e)}")
            
            elapsed = time.time() - start_time
            print(f"  Parallel GPU training completed in {elapsed:.2f}s ({len(models)} models)")
            
            # Average feature importances across models
            if importance_dfs:
                combined_imp = importance_dfs[0].copy()
                for col in ['XGB_Gain', 'XGB_Cover', 'XGB_Weight', 'XGB_Importance']:
                    combined_imp[col] = sum(df[col] for df in importance_dfs) / len(importance_dfs)
                feature_importances = combined_imp
            else:
                feature_importances = pd.DataFrame()
            
            # Use first model for interaction discovery
            model = models[0] if models else None
            
        else:
            # Single GPU or CPU training
            device_str = "GPU" if actual_gpus == 1 else "CPU"
            print(f"  Training XGBoost on {device_str}: {n_estimators} rounds, depth={max_depth}, lr={learning_rate}")
            
            model, feature_importances, gpu_used = train_xgboost_on_single_gpu(
                X, y, feature_cols, n_estimators, max_depth,  # Pass raw data, not DMatrix
                learning_rate, colsample_bytree, subsample,
                reg_alpha, reg_lambda, gpu_id=0, seed=42
            )
        
        # Sort and normalize
        feature_importances = feature_importances.sort_values('XGB_Importance', ascending=False)
        max_imp = feature_importances['XGB_Importance'].max()
        if max_imp > 0:
            feature_importances['XGB_Importance_Normalized'] = feature_importances['XGB_Importance'] / max_imp
        else:
            feature_importances['XGB_Importance_Normalized'] = 0
        
        print(f"  XGBoost found {len(feature_importances[feature_importances['XGB_Importance'] > 0])} important features")
        
        # Discover interactions from tree structure
        interactions_df = pd.DataFrame()
        
        if discover_interactions and model is not None:
            print(f"  Discovering interactions from tree structure...")
            interactions = discover_xgb_interactions(model, feature_cols, top_n=top_interactions)
            if interactions:
                interactions_df = pd.DataFrame(interactions)
                print(f"  XGBoost discovered {len(interactions_df)} potential interactions")
        
        return XGBoostReport(
            feature_importances=feature_importances,
            interactions=interactions_df,
            missed_by_traditional=[],
            xgb_model=model,
            gpu_used=gpu_used
        )
        
    except Exception as e:
        print(f"  XGBoost training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def discover_xgb_interactions(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> List[dict]:
    """
    Discover feature interactions from XGBoost tree structure.
    
    Analyzes the tree structure to find features that frequently appear
    together in parent-child relationships (indicating potential interactions).
    """
    if xgb is None:
        return []
    
    try:
        # Get all trees as DataFrames
        trees_df = model.trees_to_dataframe()
        
        if trees_df.empty:
            return []
        
        # Find parent-child feature pairs in trees
        interaction_counts = {}
        
        # Group by tree
        for tree_id in trees_df['Tree'].unique():
            tree_data = trees_df[trees_df['Tree'] == tree_id]
            
            # Build node ID to feature mapping
            node_features = {}
            for _, node in tree_data.iterrows():
                if pd.notna(node['Feature']) and node['Feature'] != 'Leaf':
                    node_features[node['ID']] = node['Feature']
            
            # Find parent-child pairs
            for _, node in tree_data.iterrows():
                if pd.notna(node['Feature']) and node['Feature'] != 'Leaf':
                    parent_feat = node['Feature']
                    
                    # Check left child
                    if pd.notna(node.get('Yes')):
                        left_id = node['Yes']
                        if left_id in node_features:
                            child_feat = node_features[left_id]
                            if parent_feat != child_feat:
                                pair = tuple(sorted([parent_feat, child_feat]))
                                interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
                    
                    # Check right child
                    if pd.notna(node.get('No')):
                        right_id = node['No']
                        if right_id in node_features:
                            child_feat = node_features[right_id]
                            if parent_feat != child_feat:
                                pair = tuple(sorted([parent_feat, child_feat]))
                                interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
        
        # Sort by count and take top N
        sorted_interactions = sorted(
            interaction_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Format as list of dicts
        interactions = []
        for (feat1, feat2), count in sorted_interactions:
            interactions.append({
                'Variable_1': feat1,
                'Variable_2': feat2,
                'Interaction_Name': f"{feat1}_x_{feat2}",
                'Magnitude': count,
                'Source': 'XGBoost'
            })
        
        return interactions
        
    except Exception as e:
        print(f"  Interaction discovery failed: {str(e)}")
        return []


def compare_xgb_selections(
    traditional_vars: List[str],
    xgb_importances: pd.DataFrame,
    top_n: int = 25,
    min_importance_threshold: float = 0.05
) -> List[str]:
    """
    Find variables important in XGBoost but missed by traditional selection.
    
    Uses TWO filters to reduce noise:
    1. top_n: Only consider top N features by importance
    2. min_importance_threshold: Only features with normalized importance >= threshold
       (e.g., 0.05 = must have at least 5% of the max feature's importance)
    
    Args:
        traditional_vars: Variables selected by traditional methods
        xgb_importances: DataFrame with XGBoost feature importances
        top_n: Maximum number of top features to consider (default: 25)
        min_importance_threshold: Minimum normalized importance threshold (default: 0.05 = 5%)
    
    Returns:
        List of variables missed by traditional selection but important in XGBoost
    """
    if xgb_importances.empty or 'XGB_Importance' not in xgb_importances.columns:
        return []
    
    # Calculate normalized importance if not already present
    if 'XGB_Importance_Normalized' not in xgb_importances.columns:
        max_imp = xgb_importances['XGB_Importance'].max()
        if max_imp > 0:
            xgb_importances = xgb_importances.copy()
            xgb_importances['XGB_Importance_Normalized'] = xgb_importances['XGB_Importance'] / max_imp
        else:
            return []
    
    # Filter by minimum importance threshold FIRST
    filtered = xgb_importances[xgb_importances['XGB_Importance_Normalized'] >= min_importance_threshold]
    
    # Then take top N from the filtered set
    xgb_top = filtered.nlargest(top_n, 'XGB_Importance')['Variable'].tolist()
    
    print(f"  XGBoost filtering: {len(xgb_importances)} total -> {len(filtered)} above {min_importance_threshold:.0%} threshold -> top {len(xgb_top)} considered")
    
    # Find variables in XGBoost top but not in traditional
    traditional_base = [v.replace('WOE_', '') for v in traditional_vars]
    
    missed = []
    for var in xgb_top:
        base_var = var.replace('WOE_', '')
        if base_var not in traditional_base and var not in traditional_vars:
            missed.append(var)
    
    return missed


def create_interaction_columns(
    df: pd.DataFrame,
    interactions: pd.DataFrame,
    top_n: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create interaction term columns in the DataFrame.
    Returns the modified DataFrame and list of new column names.
    """
    result_df = df.copy()
    new_cols = []
    
    if interactions.empty:
        return result_df, new_cols
    
    # Take top N interactions
    top_interactions = interactions.head(top_n)
    
    for _, row in top_interactions.iterrows():
        var1 = row['Variable_1']
        var2 = row['Variable_2']
        
        # Check if both variables exist in DataFrame
        if var1 in df.columns and var2 in df.columns:
            new_col_name = f"{var1}_x_{var2}"
            result_df[new_col_name] = df[var1] * df[var2]
            new_cols.append(new_col_name)
    
    return result_df, new_cols


def calculate_correlation_matrix(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """Calculate correlation matrix for selected columns."""
    available_cols = [c for c in cols if c in df.columns]
    if len(available_cols) < 2:
        return pd.DataFrame()
    
    return df[available_cols].corr().round(4)


def calculate_vif(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    VIF > 5: Moderate multicollinearity (consider removing)
    VIF > 11: High multicollinearity (should remove)
    VIF = 999.99: Perfect multicollinearity (R² ≈ 1.0, can be perfectly predicted)
    """
    from sklearn.linear_model import LinearRegression
    
    available_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(available_cols) < 2:
        return pd.DataFrame({'Variable': [], 'VIF': [], 'Status': [], 'Reason': []})
    
    # Prepare data - fill NaN with median
    X = df[available_cols].copy()
    X = X.fillna(X.median())
    
    # Check for constant columns (no variance after filling NaN)
    non_constant_cols = []
    constant_cols = []
    for col in available_cols:
        col_std = X[col].std()
        if col_std == 0 or pd.isna(col_std) or X[col].nunique() <= 1:
            constant_cols.append(col)
        else:
            non_constant_cols.append(col)
    
    if constant_cols:
        print(f"  [VIF] Found {len(constant_cols)} constant columns (no variance): {constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}")
    
    # Skip duplicate detection - it was causing false positives due to hash collisions
    # Let the VIF calculation naturally identify perfect collinearity instead
    # This is safer and more accurate
    unique_cols = non_constant_cols
    duplicate_cols = []  # Disabled - rely on VIF=999.99 detection instead
    
    vif_data = []
    
    # Add constant columns with status
    for col in constant_cols:
        vif_data.append({
            'Variable': col,
            'VIF': 999.99,
            'R_Squared': 1.0,
            'Status': 'CONSTANT - Remove',
            'Reason': 'Column has no variance'
        })
    
    # Add duplicate columns with status
    for dup, orig in duplicate_cols:
        vif_data.append({
            'Variable': dup,
            'VIF': 999.99,
            'R_Squared': 1.0,
            'Status': 'DUPLICATE - Remove',
            'Reason': f'Identical to {orig}'
        })
    
    # Calculate VIF for remaining unique, non-constant columns
    for i, col in enumerate(unique_cols):
        # Get all other unique columns as predictors
        other_cols = [c for c in unique_cols if c != col]
        
        if len(other_cols) == 0:
            vif_data.append({'Variable': col, 'VIF': 1.0, 'R_Squared': 0.0, 'Status': 'OK', 'Reason': ''})
            continue
        
        try:
            # Fit regression of col on all other columns
            X_others = X[other_cols].values
            y = X[col].values
            
            model = LinearRegression()
            model.fit(X_others, y)
            
            # Calculate R-squared
            r_squared = model.score(X_others, y)
            
            # VIF = 1 / (1 - R²)
            # Calculate raw VIF first, then cap it
            if r_squared >= 1.0:
                vif = 999.99
            else:
                vif = 1 / (1 - r_squared)
            
            # Cap VIF at 999.99 for display purposes
            # Any VIF > 100 indicates severe multicollinearity anyway
            VIF_CAP = 999.99
            if vif > VIF_CAP:
                vif = VIF_CAP
                status = 'PERFECT COLLINEAR - Remove'
                reason = f'VIF exceeded {VIF_CAP} (R²={r_squared:.6f})'
            elif vif > 10:
                status = 'HIGH - Remove'
                reason = 'Strong multicollinearity'
            elif vif > 5:
                status = 'MODERATE - Review'
                reason = 'Moderate multicollinearity'
            else:
                status = 'OK'
                reason = ''
            
            vif_data.append({
                'Variable': col,
                'VIF': round(vif, 2),
                'R_Squared': round(r_squared, 4),
                'Status': status,
                'Reason': reason
            })
            
        except Exception as e:
            vif_data.append({
                'Variable': col,
                'VIF': None,
                'R_Squared': None,
                'Status': 'Error',
                'Reason': str(e)[:50]
            })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False, na_position='last')
    
    return vif_df


def remove_high_vif_iteratively(
    df: pd.DataFrame, 
    cols: List[str], 
    vif_threshold: float = 11.0,
    max_iterations: int = 100
) -> Tuple[List[str], pd.DataFrame, List[str], List[dict]]:
    """
    Iteratively remove variables with VIF >= threshold.
    
    Since removing one high VIF variable can change VIF of others,
    we remove one at a time (highest VIF first) and recalculate.
    
    Optimization: First removes all CONSTANT and DUPLICATE columns in batch,
    then iteratively removes high VIF columns one by one.
    
    Returns:
        - remaining_cols: List of columns after removal
        - final_vif: VIF DataFrame after all removals
        - removed_cols: List of columns that were removed
        - removed_vif_info: List of dicts with Variable and VIF for removed variables
    """
    remaining_cols = [c for c in cols if c in df.columns]
    removed_cols = []
    removed_vif_info = []  # Track VIF values of removed variables
    
    # First pass: Remove only CONSTANT columns in batch (no variance = can't compute VIF properly)
    # Note: Duplicate detection was disabled due to hash collision false positives
    # Perfect collinearity is now detected via VIF=999.99 in the iterative pass
    vif_df = calculate_vif(df, remaining_cols)
    
    if not vif_df.empty and 'Status' in vif_df.columns:
        # Find only constant columns for batch removal
        batch_remove = vif_df[vif_df['Status'] == 'CONSTANT - Remove']
        
        if len(batch_remove) > 0:
            print(f"  [VIF] Batch removing {len(batch_remove)} constant columns (no variance)")
            for _, row in batch_remove.iterrows():
                var = row['Variable']
                if var in remaining_cols:
                    remaining_cols.remove(var)
                    removed_cols.append(var)
                    removed_vif_info.append({
                        'Variable': var,
                        'VIF': float(row['VIF']) if pd.notna(row['VIF']) else 999.99,
                        'R_Squared': float(row.get('R_Squared', 1.0)) if pd.notna(row.get('R_Squared')) else 1.0,
                        'Reason': row.get('Reason', row['Status'])
                    })
    
    # Second pass: Iteratively remove high VIF columns one by one
    for iteration in range(max_iterations):
        if len(remaining_cols) < 2:
            break
        
        # Calculate VIF for current columns
        vif_df = calculate_vif(df, remaining_cols)
        
        if vif_df.empty:
            break
        
        # Find highest VIF (excluding constant columns which have no valid VIF)
        valid_vif = vif_df[vif_df['Status'] != 'CONSTANT - Remove']
        if valid_vif.empty:
            break
            
        max_vif_row = valid_vif.iloc[0]  # Already sorted descending
        max_vif = max_vif_row['VIF']
        max_vif_var = max_vif_row['Variable']
        
        # If highest VIF is below threshold, we're done (remove when >= threshold)
        if max_vif is None or pd.isna(max_vif) or max_vif < vif_threshold:
            break
        
        # Remove the variable with highest VIF
        remaining_cols = [c for c in remaining_cols if c != max_vif_var]
        removed_cols.append(max_vif_var)
        removed_vif_info.append({
            'Variable': max_vif_var, 
            'VIF': float(max_vif),
            'R_Squared': float(max_vif_row.get('R_Squared', 0)) if pd.notna(max_vif_row.get('R_Squared')) else None,
            'Reason': max_vif_row.get('Reason', max_vif_row['Status'])
        })
        print(f"  Removed {max_vif_var} (VIF={max_vif:.2f}) - {max_vif_row.get('Reason', '')}")
    
    # Calculate final VIF
    final_vif = calculate_vif(df, remaining_cols)
    
    return remaining_cols, final_vif, removed_cols, removed_vif_info


def add_ranks_to_measures(measures_df: pd.DataFrame) -> pd.DataFrame:
    """Add rank columns for each measure to help with variable comparison."""
    df = measures_df.copy()
    
    # Define measure columns and their rank order
    measure_rank_configs = {
        'Entropy': ('Entropy_Rank', True),  # Lower is better, so ascending=True for rank
        'Information Value': ('IV_Rank', False),  # Higher is better
        'Odds Ratio': ('OR_Rank', False),
        'Likelihood Ratio': ('LR_Rank', False),
        'Chi-Square': ('ChiSq_Rank', False),
        'Gini': ('Gini_Rank', False)
    }
    
    for col, (rank_col, ascending) in measure_rank_configs.items():
        if col in df.columns:
            # Use float for ranks to handle NaN values properly
            df[rank_col] = df[col].rank(ascending=ascending, na_option='bottom')
    
    # Calculate average rank across all measures
    rank_cols = [c for c in df.columns if c.endswith('_Rank')]
    if rank_cols:
        df['Avg_Rank'] = df[rank_cols].mean(axis=1).round(2)
        df['Rank_Agreement'] = df[rank_cols].std(axis=1).round(2)  # Lower = more agreement
    
    return df


def add_ebm_importance_to_measures(
    measures_df: pd.DataFrame, 
    ebm_importances: pd.DataFrame
) -> pd.DataFrame:
    """Add EBM importance scores to the measures DataFrame."""
    df = measures_df.copy()
    
    if ebm_importances.empty:
        return df
    
    # Create mapping from variable name to EBM importance
    ebm_map = {}
    for _, row in ebm_importances.iterrows():
        var = row['Variable']
        importance = row['EBM_Importance']
        
        # Map both with and without WOE_ prefix
        ebm_map[var] = importance
        if var.startswith('WOE_'):
            ebm_map[var[4:]] = importance
        else:
            ebm_map[f'WOE_{var}'] = importance
    
    # Add EBM columns
    df['EBM_Importance'] = df['Variable'].map(ebm_map)
    df['EBM_Rank'] = df['EBM_Importance'].rank(ascending=False, na_option='bottom')
    
    # Flag variables where EBM and traditional selection disagree significantly
    if 'Avg_Rank' in df.columns:
        df['Rank_Diff'] = abs(df['EBM_Rank'] - df['Avg_Rank'])
        df['EBM_Disagrees'] = df['Rank_Diff'] > 20  # Flag if rank differs by more than 20
    
    return df


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_variable_selection_app(
    df: pd.DataFrame,
    min_prop: float = 0.01
):
    """Create the Variable Selection Shiny application."""
    
    # Find binary target variable candidates
    binary_vars = [col for col in df.columns 
                   if df[col].nunique() == 2 and not col.startswith(('WOE_', 'b_'))]
    
    # Get WOE columns
    woe_cols = [col for col in df.columns if col.startswith('WOE_')]
    
    app_results = {
        'measures': None,
        'selected_data': None,
        'ebm_report': None,
        'correlation_matrix': None,
        'vif_report': None,
        'removed_for_vif': [],
        'completed': False
    }
    
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
                body { 
                    font-family: 'Space Grotesk', sans-serif; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: #eee;
                    min-height: 100vh;
                }
                .card { 
                    background: rgba(255,255,255,0.05); 
                    border-radius: 12px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    border: 1px solid rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                }
                .btn-primary { background: linear-gradient(45deg, #667eea, #764ba2); border: none; }
                .btn-success { background: linear-gradient(45deg, #11998e, #38ef7d); border: none; }
                .btn-danger { background: linear-gradient(45deg, #eb3349, #f45c43); border: none; }
                .btn-warning { background: linear-gradient(45deg, #f7971e, #ffd200); border: none; color: #333; }
                .btn { border-radius: 25px; padding: 10px 25px; font-weight: 500; }
                h4, h5 { font-weight: 700; text-align: center; margin: 20px 0; color: #fff; }
                .form-control, .form-select { 
                    background: rgba(255,255,255,0.1); 
                    border: 1px solid rgba(255,255,255,0.2);
                    color: #fff;
                }
                .form-control:focus, .form-select:focus { 
                    background: rgba(255,255,255,0.15);
                    border-color: #667eea;
                    color: #fff;
                }
                .form-check-input:checked { background-color: #667eea; border-color: #667eea; }
                label { color: #ccc; }
                .highlight-box {
                    background: linear-gradient(45deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
                    border-left: 4px solid #667eea;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 0 8px 8px 0;
                }
            """)
        ),
        
        ui.h4("🎯 Variable Selection with EBM Interaction Discovery"),
        
        # Configuration Section
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(4,
                    ui.input_select("dv", "Dependent Variable", 
                                   choices=binary_vars,
                                   selected=binary_vars[0] if binary_vars else None)
                ),
                ui.column(4,
                    ui.input_checkbox_group(
                        "measures", "Measures of Predictive Power",
                        choices={
                            'EntropyExplained': 'Entropy Explained',
                            'InformationValue': 'Information Value',
                            'OddsRatio': 'Odds Ratio',
                            'LikelihoodRatio': 'Likelihood Ratio',
                            'PearsonChiSquare': 'Pearson Chi-Square',
                            'Gini': 'Gini'
                        },
                        selected=['EntropyExplained', 'InformationValue', 'LikelihoodRatio', 
                                 'PearsonChiSquare', 'Gini']
                    )
                ),
                ui.column(4,
                    ui.input_action_button("analyze_btn", "📊 Analyze", 
                                          class_="btn btn-primary btn-lg", 
                                          style="width: 100%; margin-top: 30px;")
                )
            )
        ),
        
        # Selection Criteria Section
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(3,
                    ui.input_numeric("num_vars", "Number of Variables", 
                                    value=50, min=1, max=500)
                ),
                ui.column(3,
                    ui.input_select("criteria", "Criteria", 
                                   choices={'Union': 'Union (ANY measure)', 
                                           'Intersection': 'Intersection (MULTIPLE measures)'},
                                   selected='Intersection')
                ),
                ui.column(3,
                    ui.input_numeric("degree", "Degree (for Intersection)", 
                                    value=2, min=1, max=6)
                ),
                ui.column(3,
                    ui.input_action_button("select_btn", "✅ Select Variables", 
                                          class_="btn btn-success", 
                                          style="width: 100%; margin-top: 30px;")
                )
            )
        ),
        
        # EBM Configuration Section
        ui.div(
            {"class": "card highlight-box"},
            ui.h5("🤖 EBM Interaction Discovery"),
            ui.row(
                ui.column(3,
                    ui.input_numeric("max_interactions", "Max Interactions to Detect", 
                                    value=20, min=5, max=50)
                ),
                ui.column(3,
                    ui.input_numeric("top_interactions", "Top Interactions to Include", 
                                    value=10, min=1, max=30)
                ),
                ui.column(3,
                    ui.input_checkbox("auto_add_missed", "Auto-add EBM-missed variables", 
                                     value=True),
                    ui.input_numeric("max_missed_to_add", "Max missed vars to add (0=ALL)", 
                                    value=0, min=0, max=1000)
                ),
                ui.column(3,
                    ui.input_action_button("ebm_btn", "🔍 Discover Interactions", 
                                          class_="btn btn-warning", 
                                          style="width: 100%; margin-top: 30px;")
                )
            )
        ),
        
        # Results Tables
        ui.row(
            ui.column(12,
                ui.div(
                    {"class": "card", "style": "max-height: 400px; overflow-y: auto;"},
                    ui.h5("📈 Predictive Measures"),
                    ui.output_data_frame("measures_table")
                )
            )
        ),
        
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "max-height: 350px; overflow-y: auto;"},
                    ui.h5("🔗 EBM Detected Interactions"),
                    ui.output_data_frame("interactions_table")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "max-height: 350px; overflow-y: auto;"},
                    ui.h5("⚠️ Variables Missed by Traditional Selection"),
                    ui.output_data_frame("missed_table")
                )
            )
        ),
        
        # Visualization
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    output_widget("importance_chart")
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    output_widget("interaction_chart")
                )
            )
        ),
        
        # VIF Table
        ui.row(
            ui.column(12,
                ui.div(
                    {"class": "card", "style": "max-height: 300px; overflow-y: auto;"},
                    ui.h5("📊 VIF - Multicollinearity Check"),
                    ui.output_data_frame("vif_table")
                )
            )
        ),
        
        # Summary Stats
        ui.div(
            {"class": "card"},
            ui.output_ui("summary_stats")
        ),
        
        # Submit Button
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("submit_btn", "🚀 Generate Output & Close", 
                                  class_="btn btn-success btn-lg"),
        ),
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        measures_rv = reactive.Value(pd.DataFrame())
        ebm_report_rv = reactive.Value(None)
        selected_vars_rv = reactive.Value([])
        interaction_cols_rv = reactive.Value([])
        missed_vars_to_add_rv = reactive.Value([])
        vif_rv = reactive.Value(pd.DataFrame())
        
        @reactive.Effect
        @reactive.event(input.analyze_btn)
        def analyze():
            dv = input.dv()
            if not input.measures() or not dv:
                return
            
            # Calculate bins internally (like R's logiBin::getBins)
            iv_list = [col for col in df.columns if col != dv]
            bin_result = get_bins(df, dv, iv_list, min_prop=min_prop)
            bins_df_calc = bin_result.bin
            var_summary_calc = bin_result.var_summary
            
            measures = calculate_all_measures(
                bins_df_calc, 
                var_summary_calc, 
                list(input.measures())
            )
            measures_rv.set(measures)
        
        @reactive.Effect
        @reactive.event(input.select_btn)
        def select_variables():
            measures = measures_rv.get()
            if measures.empty:
                return
            
            criteria = input.criteria()
            num_vars = input.num_vars()
            degree = input.degree() if criteria == 'Intersection' else 1
            
            filtered = filter_variables(measures, criteria, num_vars, degree)
            measures_rv.set(filtered)
            
            selected = filtered[filtered['Selected'] == True]['Variable'].tolist()
            selected_vars_rv.set(selected)
        
        @reactive.Effect
        @reactive.event(input.ebm_btn)
        def run_ebm():
            dv = input.dv()
            if not dv or dv not in df.columns:
                return
            
            # Use WOE columns for EBM
            feature_cols = woe_cols.copy()
            if not feature_cols:
                return
            
            if not EBM_AVAILABLE:
                print("EBM not available")
                return
            
            # Train EBM
            report = train_ebm_for_discovery(
                df, dv, feature_cols, 
                max_interactions=input.max_interactions()
            )
            
            if report is None:
                print("EBM training failed")
                return
            
            # Compare with traditional selection
            selected = selected_vars_rv.get()
            if selected:
                missed = compare_selections(selected, report.feature_importances, top_n=50)
                report.missed_by_traditional = missed
                
                # Auto-add missed variables if enabled
                if input.auto_add_missed():
                    max_to_add = input.max_missed_to_add()
                    if max_to_add == 0:
                        # 0 means add ALL
                        missed_to_add = missed
                    else:
                        missed_to_add = missed[:max_to_add]
                    missed_vars_to_add_rv.set(missed_to_add)
            
            ebm_report_rv.set(report)
            
            # Add EBM importance to measures
            measures = measures_rv.get()
            if not measures.empty:
                measures = add_ranks_to_measures(measures)
                measures = add_ebm_importance_to_measures(measures, report.feature_importances)
                measures_rv.set(measures)
            
            # Create interaction columns
            if not report.interactions.empty:
                _, int_cols = create_interaction_columns(
                    df, report.interactions, 
                    top_n=input.top_interactions()
                )
                interaction_cols_rv.set(int_cols)
            
            # Calculate VIF for selected + missed + interaction columns
            all_selected = selected_vars_rv.get().copy()
            for var in missed_vars_to_add_rv.get():
                if var not in all_selected:
                    all_selected.append(var)
            
            # Map to WOE columns
            vif_cols = []
            for var in all_selected:
                woe_col = f"WOE_{var}" if not var.startswith('WOE_') else var
                if woe_col in df.columns:
                    vif_cols.append(woe_col)
                elif var in df.columns:
                    vif_cols.append(var)
            
            if vif_cols:
                vif_result = calculate_vif(df, vif_cols)
                vif_rv.set(vif_result)
        
        @output
        @render.data_frame
        def measures_table():
            measures = measures_rv.get()
            if measures.empty:
                return render.DataGrid(pd.DataFrame({'Message': ['Click "Analyze" to calculate measures']}))
            return render.DataGrid(measures, selection_mode="rows", height="350px")
        
        @output
        @render.data_frame
        def interactions_table():
            report = ebm_report_rv.get()
            if report is None or report.interactions.empty:
                return render.DataGrid(pd.DataFrame({'Message': ['Click "Discover Interactions" to run EBM']}))
            return render.DataGrid(
                report.interactions[['Variable_1', 'Variable_2', 'Magnitude']].head(20),
                height="300px"
            )
        
        @output
        @render.data_frame
        def missed_table():
            report = ebm_report_rv.get()
            if report is None or not report.missed_by_traditional:
                return render.DataGrid(pd.DataFrame({'Message': ['No variables missed or EBM not run yet']}))
            
            # Get importance for missed variables
            missed_df = report.feature_importances[
                report.feature_importances['Variable'].isin(report.missed_by_traditional)
            ].copy()
            missed_df['Status'] = 'Consider Adding'
            
            return render.DataGrid(missed_df.head(20), height="300px")
        
        @output
        @render_plotly
        def importance_chart():
            report = ebm_report_rv.get()
            if report is None:
                return go.Figure().add_annotation(
                    text="Run EBM to see feature importances",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            top_20 = report.feature_importances.head(20)
            
            fig = go.Figure(go.Bar(
                x=top_20['EBM_Importance'],
                y=top_20['Variable'],
                orientation='h',
                marker=dict(
                    color=top_20['EBM_Importance'],
                    colorscale='Viridis'
                )
            ))
            
            fig.update_layout(
                title='Top 20 Variables by EBM Importance',
                xaxis_title='Importance',
                yaxis_title='Variable',
                height=350,
                yaxis=dict(autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        @output
        @render_plotly
        def interaction_chart():
            report = ebm_report_rv.get()
            if report is None or report.interactions.empty:
                return go.Figure().add_annotation(
                    text="Run EBM to see interactions",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            top_10 = report.interactions.head(10)
            
            fig = go.Figure(go.Bar(
                x=top_10['Magnitude'],
                y=top_10['Interaction_Name'],
                orientation='h',
                marker=dict(
                    color=top_10['Magnitude'],
                    colorscale='Plasma'
                )
            ))
            
            fig.update_layout(
                title='Top 10 Detected Interactions',
                xaxis_title='Interaction Magnitude',
                yaxis_title='Interaction',
                height=350,
                yaxis=dict(autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        @output
        @render.data_frame
        def vif_table():
            vif = vif_rv.get()
            if vif.empty:
                return render.DataGrid(pd.DataFrame({'Message': ['Run EBM to calculate VIF']}))
            return render.DataGrid(vif, height="250px")
        
        @output
        @render.ui
        def summary_stats():
            measures = measures_rv.get()
            selected = selected_vars_rv.get()
            int_cols = interaction_cols_rv.get()
            missed_to_add = missed_vars_to_add_rv.get()
            vif = vif_rv.get()
            
            total_vars = len(measures) if not measures.empty else 0
            selected_count = len(selected)
            missed_count = len(missed_to_add)
            int_count = len(int_cols)
            total_features = selected_count + missed_count + int_count
            
            # Count high VIF variables
            high_vif_count = 0
            if not vif.empty and 'VIF' in vif.columns:
                high_vif_count = len(vif[vif['VIF'] > 5])
            
            return ui.div(
                ui.h5("📊 Selection Summary"),
                ui.row(
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(total_vars), style="color: #667eea; margin: 0;"),
                        ui.p("Total Variables", style="font-size: 12px;")
                    )),
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(selected_count), style="color: #38ef7d; margin: 0;"),
                        ui.p("Selected (Traditional)", style="font-size: 12px;")
                    )),
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(missed_count), style="color: #f7971e; margin: 0;"),
                        ui.p("EBM-Missed Added", style="font-size: 12px;")
                    )),
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(int_count), style="color: #ffd200; margin: 0;"),
                        ui.p("Interaction Terms", style="font-size: 12px;")
                    )),
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(total_features), style="color: #eb3349; margin: 0;"),
                        ui.p("Total for Stepwise", style="font-size: 12px;")
                    )),
                    ui.column(2, ui.div(
                        {"style": "text-align: center; padding: 15px;"},
                        ui.h3(str(high_vif_count), style="color: #ff6b6b; margin: 0;"),
                        ui.p("High VIF (>5)", style="font-size: 12px;")
                    )),
                )
            )
        
        @reactive.Effect
        @reactive.event(input.submit_btn)
        async def submit():
            dv = input.dv()
            measures = measures_rv.get()
            selected = selected_vars_rv.get()
            missed_to_add = missed_vars_to_add_rv.get()
            report = ebm_report_rv.get()
            vif = vif_rv.get()
            
            # Prepare output data
            output_cols = [dv] if dv and dv in df.columns else []
            
            # Add selected WOE variables
            for var in selected:
                woe_col = f"WOE_{var}" if not var.startswith('WOE_') else var
                if woe_col in df.columns:
                    output_cols.append(woe_col)
                elif var in df.columns:
                    output_cols.append(var)
            
            # Add EBM-missed variables (auto-added)
            for var in missed_to_add:
                woe_col = var if var.startswith('WOE_') else f"WOE_{var}"
                if woe_col in df.columns and woe_col not in output_cols:
                    output_cols.append(woe_col)
                elif var in df.columns and var not in output_cols:
                    output_cols.append(var)
            
            # Create output DataFrame with interaction columns
            output_df = df[output_cols].copy() if output_cols else df.copy()
            
            # Add interaction columns from EBM
            if report is not None and not report.interactions.empty:
                output_df, int_cols = create_interaction_columns(
                    output_df, 
                    report.interactions, 
                    top_n=input.top_interactions()
                )
            
            # Prepare EBM report DataFrame
            if report is not None:
                ebm_report_df = report.interactions.copy()
                if not ebm_report_df.empty:
                    ebm_report_df['Status'] = 'Detected Interaction'
                    ebm_report_df['Included'] = True
                
                # Add missed variables as a separate section
                missed_df = pd.DataFrame({
                    'Variable_1': report.missed_by_traditional,
                    'Variable_2': ['(single variable)'] * len(report.missed_by_traditional),
                    'Interaction_Name': report.missed_by_traditional,
                    'Magnitude': [None] * len(report.missed_by_traditional),
                    'Status': ['Missed by Traditional'] * len(report.missed_by_traditional),
                    'Included': [var in missed_to_add for var in report.missed_by_traditional]
                })
                if not missed_df.empty:
                    ebm_report_df = pd.concat([ebm_report_df, missed_df], ignore_index=True)
            else:
                ebm_report_df = pd.DataFrame()
            
            # Calculate correlation matrix (before VIF removal)
            numeric_cols = [c for c in output_df.columns if c != dv and pd.api.types.is_numeric_dtype(output_df[c])]
            corr_matrix = calculate_correlation_matrix(output_df, numeric_cols)
            
            # Iteratively remove high VIF variables (VIF > 11)
            remaining_cols, final_vif, removed_cols, _ = remove_high_vif_iteratively(
                output_df, numeric_cols, vif_threshold=0.0  # No VIF filtering by default
            )
            
            if removed_cols:
                # Keep only DV + remaining numeric columns
                final_cols = [dv] + remaining_cols
                output_df = output_df[final_cols].copy()
                
                # Add removed columns info to VIF report
                final_vif['Removed'] = False
                removed_vif_df = pd.DataFrame({
                    'Variable': removed_cols,
                    'VIF': ['>11 (removed)'] * len(removed_cols),
                    'R_Squared': [None] * len(removed_cols),
                    'Status': ['REMOVED (VIF>11)'] * len(removed_cols),
                    'Removed': [True] * len(removed_cols)
                })
                final_vif = pd.concat([removed_vif_df, final_vif], ignore_index=True)
            else:
                if not final_vif.empty:
                    final_vif['Removed'] = False
            
            app_results['measures'] = measures
            app_results['selected_data'] = output_df
            app_results['ebm_report'] = ebm_report_df
            app_results['correlation_matrix'] = corr_matrix
            app_results['vif_report'] = final_vif
            app_results['removed_for_vif'] = removed_cols
            app_results['completed'] = True
            
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def find_free_port(start_port: int = 8052, max_attempts: int = 50) -> int:
    """Find an available port starting from start_port."""
    import socket
    
    for offset in range(max_attempts):
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    # Fallback: let OS assign a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_variable_selection(
    df: pd.DataFrame,
    port: int = None
):
    """Run the Variable Selection application and return results."""
    # Find a free port to avoid conflicts with multiple instances
    if port is None:
        port = find_free_port(BASE_PORT)
    
    print(f"Starting Shiny app on port {port}")
    sys.stdout.flush()
    
    app = create_variable_selection_app(df)
    
    try:
        app.run(port=port, launch_browser=True)
    except Exception as e:
        print(f"Error running Shiny app: {e}")
        sys.stdout.flush()
        # Try with a different port
        try:
            fallback_port = find_free_port(port + 100)
            print(f"Retrying on port {fallback_port}")
            app.run(port=fallback_port, launch_browser=True)
        except Exception as e2:
            print(f"Failed on fallback port: {e2}")
            app.results['completed'] = False
    
    # Cleanup
    gc.collect()
    sys.stdout.flush()
    
    return app.results


# =============================================================================
# Headless Mode Processing
# =============================================================================

def run_headless_selection(
    df: pd.DataFrame,
    dv: str,
    measures_to_calc: List[str],
    num_of_variables: int,
    criteria: str,
    degree: int,
    max_interactions: int = 20,
    top_interactions: int = 10,
    auto_add_missed: bool = True,
    max_missed_to_add: int = 0,  # 0 = add ALL missed variables
    vif_threshold: float = 0.0,  # 0 = no VIF filtering
    min_prop: float = 0.01,
    use_xgboost: bool = True,  # Use XGBoost for feature discovery
    xgb_n_estimators: int = 3000,  # More rounds for stable importance
    xgb_max_depth: int = 8,  # Deeper trees for better discrimination
    xgb_learning_rate: float = 0.01,  # Lower LR for reliable rankings
    xgb_colsample: float = 0.5,  # 50% column sampling (more regularization)
    xgb_subsample: float = 0.8,
    xgb_reg_alpha: float = 0.5,  # L1 regularization (zeros out weak features)
    xgb_reg_lambda: float = 2.0,  # L2 regularization (shrinks marginal importance)
    xgb_importance_threshold: float = 0.05,  # Min 5% of max importance to be considered
    xgb_top_n: int = 25,  # Only consider top 25 features from XGBoost
    xgb_num_gpus: int = 2  # Number of GPUs for parallel training
) -> Dict[str, Any]:
    """Run variable selection in headless mode with improved XGBoost feature filtering."""
    
    print(f"Running headless variable selection with DV: {dv}")
    print(f"Criteria: {criteria}, NumVars: {num_of_variables}, Degree: {degree}")
    max_missed_str = "ALL" if max_missed_to_add == 0 else str(max_missed_to_add)
    vif_str = "disabled" if vif_threshold == 0 else f">= {vif_threshold}"
    print(f"Auto-add missed: {auto_add_missed}, Max missed to add: {max_missed_str}")
    print(f"VIF filtering: {vif_str}")
    
    # Get all independent variables (exclude DV)
    iv_list = [col for col in df.columns if col != dv]
    print(f"Found {len(iv_list)} potential independent variables")
    
    # Calculate bins internally using get_bins (like R's logiBin::getBins)
    print("Calculating bins for all variables...")
    bin_result = get_bins(df, dv, iv_list, min_prop=min_prop)
    bins_df = bin_result.bin
    var_summary = bin_result.var_summary
    print(f"Generated bins for {len(var_summary)} variables")
    
    # Calculate measures
    if len(measures_to_calc) > 0:
        measures = calculate_all_measures(bins_df, var_summary, measures_to_calc)
        print(f"Calculated measures for {len(measures)} variables")
        
        # Add ranks to measures
        measures = add_ranks_to_measures(measures)
        
        # Filter variables
        measures = filter_variables(measures, criteria, num_of_variables, degree)
        selected_vars = measures[measures['Selected'] == True]['Variable'].tolist()
    else:
        # No measures specified - select all variables
        selected_vars = var_summary['var'].tolist()
        measures = pd.DataFrame({'Variable': selected_vars, 'Selected': True})
    
    print(f"Selected {len(selected_vars)} variables by traditional method")
    
    # Get WOE columns
    woe_cols = [col for col in df.columns if col.startswith('WOE_')]
    
    # Train EBM and XGBoost in PARALLEL for faster discovery
    ebm_report = None
    xgb_report = None
    interaction_cols = []
    missed_to_add = []
    xgb_missed = []
    
    # Check what models we can run
    can_run_ebm = woe_cols and dv in df.columns and EBM_AVAILABLE
    can_run_xgb = use_xgboost and woe_cols and dv in df.columns and XGBOOST_AVAILABLE
    
    if can_run_ebm or can_run_xgb:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        futures = {}
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"PARALLEL ML DISCOVERY on {len(woe_cols)} WOE columns")
        print(f"  EBM: {'enabled' if can_run_ebm else 'disabled'}")
        print(f"  XGBoost: {'enabled (GPU)' if can_run_xgb and XGBOOST_GPU_AVAILABLE else 'enabled (CPU)' if can_run_xgb else 'disabled'}")
        print(f"{'='*50}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit EBM training
            if can_run_ebm:
                futures['EBM'] = executor.submit(
                    train_ebm_for_discovery, 
                    df, dv, woe_cols, max_interactions
                )
            
            # Submit XGBoost training (uses multiple GPUs internally)
            if can_run_xgb:
                futures['XGBoost'] = executor.submit(
                    train_xgboost_for_discovery,
                    df, dv, woe_cols,
                    XGBOOST_GPU_AVAILABLE,
                    xgb_n_estimators,
                    xgb_max_depth,
                    xgb_learning_rate,
                    True,  # discover_interactions
                    max_interactions,
                    xgb_colsample,
                    xgb_subsample,
                    xgb_reg_alpha,
                    xgb_reg_lambda,
                    xgb_num_gpus
                )
            
            # Collect results as they complete
            for future in as_completed(futures.values()):
                model_name = [k for k, v in futures.items() if v == future][0]
                try:
                    result = future.result()
                    if model_name == 'EBM':
                        ebm_report = result
                        print(f"  [DONE] EBM completed")
                    else:
                        xgb_report = result
                        gpu_str = "(GPU)" if xgb_report and xgb_report.gpu_used else "(CPU)"
                        print(f"  [DONE] XGBoost completed {gpu_str}")
                except Exception as e:
                    print(f"  [ERROR] {model_name} failed: {str(e)}")
        
        elapsed = time.time() - start_time
        print(f"Parallel training completed in {elapsed:.2f} seconds")
        print(f"{'='*50}\n")
    else:
        if not EBM_AVAILABLE:
            print("EBM not available")
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
    
    # Process EBM results
    if ebm_report is not None:
        missed = compare_selections(selected_vars, ebm_report.feature_importances, top_n=50)
        ebm_report.missed_by_traditional = missed
        print(f"EBM found {len(ebm_report.interactions)} interactions")
        print(f"Variables missed by traditional selection (EBM): {len(missed)}")
        
        # Auto-add missed variables from EBM
        if auto_add_missed and missed:
            if max_missed_to_add == 0:
                missed_to_add = missed
            else:
                missed_to_add = missed[:max_missed_to_add]
            print(f"Auto-adding {len(missed_to_add)} EBM-missed variables")
        
        # Add EBM importance to measures
        measures = add_ebm_importance_to_measures(measures, ebm_report.feature_importances)
    
    # Process XGBoost results
    if xgb_report is not None:
        xgb_missed = compare_xgb_selections(
            selected_vars, 
            xgb_report.feature_importances, 
            top_n=xgb_top_n,
            min_importance_threshold=xgb_importance_threshold
        )
        xgb_report.missed_by_traditional = xgb_missed
        print(f"XGBoost found {len(xgb_report.interactions)} potential interactions")
        print(f"Variables missed by traditional selection (XGBoost): {len(xgb_missed)}")
        
        # Auto-add XGBoost missed variables (that aren't already in missed_to_add)
        if auto_add_missed and xgb_missed:
            xgb_new_missed = [v for v in xgb_missed if v not in missed_to_add]
            if max_missed_to_add == 0:
                missed_to_add.extend(xgb_new_missed)
            else:
                # Add up to remaining quota
                remaining = max(0, max_missed_to_add - len(missed_to_add))
                missed_to_add.extend(xgb_new_missed[:remaining])
            if xgb_new_missed:
                print(f"Auto-adding {len(xgb_new_missed)} additional XGBoost-missed variables")
        
        # Add XGBoost importance to measures
        if not xgb_report.feature_importances.empty:
            xgb_imp = xgb_report.feature_importances[['Variable', 'XGB_Importance', 'XGB_Gain', 'XGB_Cover']].copy()
            measures = measures.merge(xgb_imp, on='Variable', how='left')
    
    # Prepare output DataFrame
    output_cols = [dv]
    
    # Add selected WOE variables
    added_selected = 0
    for var in selected_vars:
        woe_col = f"WOE_{var}" if not var.startswith('WOE_') else var
        if woe_col in df.columns:
            output_cols.append(woe_col)
            added_selected += 1
        elif var in df.columns:
            output_cols.append(var)
            added_selected += 1
    print(f"  Added {added_selected}/{len(selected_vars)} selected variables to output")
    
    # Add EBM-missed variables
    added_missed = 0
    for var in missed_to_add:
        woe_col = var if var.startswith('WOE_') else f"WOE_{var}"
        if woe_col in df.columns and woe_col not in output_cols:
            output_cols.append(woe_col)
            added_missed += 1
        elif var in df.columns and var not in output_cols:
            output_cols.append(var)
            added_missed += 1
    print(f"  Added {added_missed}/{len(missed_to_add)} EBM-missed variables to output")
    print(f"  Total columns before VIF: {len(output_cols)} (1 DV + {added_selected} selected + {added_missed} EBM-missed)")
    
    output_df = df[output_cols].copy()
    
    # Add interaction columns from EBM
    if ebm_report is not None and not ebm_report.interactions.empty:
        output_df, ebm_int_cols = create_interaction_columns(
            output_df, 
            ebm_report.interactions, 
            top_n=top_interactions
        )
        interaction_cols.extend(ebm_int_cols)
        print(f"Added {len(ebm_int_cols)} EBM interaction columns")
    
    # Add interaction columns from XGBoost
    if xgb_report is not None and not xgb_report.interactions.empty:
        output_df, xgb_int_cols = create_interaction_columns(
            output_df, 
            xgb_report.interactions, 
            top_n=top_interactions
        )
        # Only add unique interaction columns (not already added by EBM)
        new_xgb_cols = [c for c in xgb_int_cols if c not in interaction_cols]
        interaction_cols.extend(new_xgb_cols)
        if new_xgb_cols:
            print(f"Added {len(new_xgb_cols)} XGBoost interaction columns")
    
    # Prepare ML Discovery report DataFrame (EBM + XGBoost)
    ml_report_df = pd.DataFrame()
    
    # Add EBM interactions
    if ebm_report is not None:
        ebm_interactions = ebm_report.interactions.copy()
        if not ebm_interactions.empty:
            ebm_interactions['Source'] = 'EBM'
            ebm_interactions['Status'] = 'Detected Interaction'
            ebm_interactions['Included'] = True
            ml_report_df = pd.concat([ml_report_df, ebm_interactions], ignore_index=True)
        
        # Add EBM missed variables
        if ebm_report.missed_by_traditional:
            missed_df = pd.DataFrame({
                'Variable_1': ebm_report.missed_by_traditional,
                'Variable_2': ['(single variable)'] * len(ebm_report.missed_by_traditional),
                'Interaction_Name': ebm_report.missed_by_traditional,
                'Magnitude': [None] * len(ebm_report.missed_by_traditional),
                'Source': ['EBM'] * len(ebm_report.missed_by_traditional),
                'Status': ['Missed by Traditional'] * len(ebm_report.missed_by_traditional),
                'Included': [var in missed_to_add for var in ebm_report.missed_by_traditional]
            })
            ml_report_df = pd.concat([ml_report_df, missed_df], ignore_index=True)
    
    # Add XGBoost interactions
    if xgb_report is not None:
        xgb_interactions = xgb_report.interactions.copy()
        if not xgb_interactions.empty:
            xgb_interactions['Status'] = 'Detected Interaction'
            xgb_interactions['Included'] = True
            ml_report_df = pd.concat([ml_report_df, xgb_interactions], ignore_index=True)
        
        # Add XGBoost missed variables
        if xgb_report.missed_by_traditional:
            xgb_missed_df = pd.DataFrame({
                'Variable_1': xgb_report.missed_by_traditional,
                'Variable_2': ['(single variable)'] * len(xgb_report.missed_by_traditional),
                'Interaction_Name': xgb_report.missed_by_traditional,
                'Magnitude': [None] * len(xgb_report.missed_by_traditional),
                'Source': ['XGBoost'] * len(xgb_report.missed_by_traditional),
                'Status': ['Missed by Traditional'] * len(xgb_report.missed_by_traditional),
                'Included': [var in missed_to_add for var in xgb_report.missed_by_traditional]
            })
            ml_report_df = pd.concat([ml_report_df, xgb_missed_df], ignore_index=True)
    
    # Use combined report (for backward compatibility, variable is still named ebm_report_df)
    ebm_report_df = ml_report_df
    
    # Calculate correlation matrix (before VIF removal)
    numeric_cols = [c for c in output_df.columns if c != dv and pd.api.types.is_numeric_dtype(output_df[c])]
    corr_matrix = calculate_correlation_matrix(output_df, numeric_cols)
    
    # Iteratively remove high VIF variables
    # VIF filtering (only if threshold > 0)
    removed_cols = []
    removed_vif_info = []
    
    if vif_threshold > 0:
        print(f"Checking for multicollinearity (VIF >= {vif_threshold})...")
        remaining_cols, vif_report, removed_cols, removed_vif_info = remove_high_vif_iteratively(
            output_df, numeric_cols, vif_threshold=vif_threshold
        )
        
        if removed_cols:
            print(f"Removed {len(removed_cols)} variables with VIF >= {vif_threshold}: {removed_cols}")
            # Keep only DV + remaining numeric columns
            final_cols = [dv] + remaining_cols
            output_df = output_df[final_cols].copy()
            print(f"  Final columns after VIF: {len(output_df.columns)} (1 DV + {len(remaining_cols)} features)")
            
            # Add removed columns info to VIF report (using actual numeric VIF values)
            vif_report['Removed'] = False
            vif_report['Status'] = 'OK'
            
            removed_vif_df = pd.DataFrame(removed_vif_info)
            removed_vif_df['Status'] = f'REMOVED (VIF>={vif_threshold})'
            removed_vif_df['Removed'] = True
            
            vif_report = pd.concat([removed_vif_df, vif_report], ignore_index=True)
        else:
            print("No variables with VIF >= threshold found")
            if not vif_report.empty:
                vif_report['Removed'] = False
                vif_report['Status'] = 'OK'
    else:
        print("VIF filtering disabled (threshold = 0)")
        # Still calculate VIF for reporting, but don't remove anything
        vif_report = calculate_vif(output_df, numeric_cols)
        if not vif_report.empty:
            vif_report['Removed'] = False
            vif_report['Status'] = 'OK'
    
    # Ensure VIF column is numeric
    if not vif_report.empty and 'VIF' in vif_report.columns:
        vif_report['VIF'] = pd.to_numeric(vif_report['VIF'], errors='coerce')
    
    moderate_vif = len(vif_report[
        (vif_report['Removed'] == False) & 
        (vif_report['VIF'] > 5)
    ]) if not vif_report.empty else 0
    print(f"VIF summary: {len(removed_cols)} removed, {moderate_vif} moderate (5-11)")
    
    return {
        'measures': measures,
        'selected_data': output_df,
        'ebm_report': ebm_report_df,
        'correlation_matrix': corr_matrix,
        'vif_report': vif_report,
        'removed_for_vif': removed_cols
    }


# =============================================================================
# Read Input Data
# =============================================================================
print("Variable Selection Node - Starting...")
print("=" * 70)

# Single input: the data table
df = knio.input_tables[0].to_pandas()
print(f"Input data: {len(df)} rows, {len(df.columns)} columns")

print("=" * 70)

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
contains_all_vars = False
dv = None
measures_of_power = None
num_of_variables = None
criteria = None
degree = None
max_interactions = 20
top_interactions = 10
auto_add_missed = True
max_missed_to_add = 0  # 0 = add ALL missed variables

try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

try:
    measures_of_power = knio.flow_variables.get("MeasuresOfPredictivePower", None)
except:
    pass

try:
    num_of_variables = knio.flow_variables.get("NumberOfVariables", None)
except:
    pass

try:
    criteria = knio.flow_variables.get("Criteria", None)
except:
    pass

try:
    degree = knio.flow_variables.get("Degree", None)
except:
    pass

try:
    max_interactions = knio.flow_variables.get("MaxInteractions", 20)
except:
    pass

try:
    top_interactions = knio.flow_variables.get("TopInteractions", 10)
except:
    pass

try:
    auto_add_missed = knio.flow_variables.get("AutoAddMissed", True)
except:
    pass

try:
    max_missed_to_add = knio.flow_variables.get("MaxMissedToAdd", 0)  # 0 = add ALL
except:
    pass

try:
    vif_threshold = knio.flow_variables.get("VIFThreshold", 0.0)  # 0 = no filtering
except:
    vif_threshold = 0.0  # No VIF filtering by default

# XGBoost parameters - OPTIMIZED for robust feature selection (less noise)
# These settings take ~3-5 minutes but produce much better feature rankings
use_xgboost = True
xgb_n_estimators = 3000  # More rounds = more stable importance estimates
xgb_max_depth = 8  # Deeper trees for better feature discrimination
xgb_learning_rate = 0.01  # Lower rate + more trees = most reliable rankings
xgb_colsample = 0.5  # Sample 50% columns per tree (stronger regularization)
xgb_subsample = 0.8  # Sample 80% rows per tree
xgb_reg_alpha = 0.5  # L1 regularization - zeros out weak features
xgb_reg_lambda = 2.0  # L2 regularization - shrinks marginal feature importance
xgb_importance_threshold = 0.05  # Only features >= 5% of max importance
xgb_top_n = 25  # Only consider top 25 features from XGBoost

try:
    use_xgboost = knio.flow_variables.get("UseXGBoost", True)
except:
    pass

try:
    xgb_n_estimators = knio.flow_variables.get("XGBEstimators", 3000)
except:
    pass

try:
    xgb_max_depth = knio.flow_variables.get("XGBMaxDepth", 8)
except:
    pass

try:
    xgb_learning_rate = knio.flow_variables.get("XGBLearningRate", 0.01)
except:
    pass

try:
    xgb_colsample = knio.flow_variables.get("XGBColsampleByTree", 0.5)
except:
    pass

try:
    xgb_subsample = knio.flow_variables.get("XGBSubsample", 0.8)
except:
    pass

try:
    xgb_reg_alpha = knio.flow_variables.get("XGBRegAlpha", 0.5)
except:
    pass

try:
    xgb_reg_lambda = knio.flow_variables.get("XGBRegLambda", 2.0)
except:
    pass

try:
    xgb_importance_threshold = knio.flow_variables.get("XGBImportanceThreshold", 0.05)
except:
    pass

try:
    xgb_top_n = knio.flow_variables.get("XGBTopN", 25)
except:
    pass

# Number of GPUs for parallel XGBoost
xgb_num_gpus = 2

try:
    xgb_num_gpus = knio.flow_variables.get("XGBNumGPUs", 2)
except:
    pass

# Debug: Print detected flow variables
print("=" * 70)
print("FLOW VARIABLES DETECTED:")
print(f"  DependentVariable: {dv} (type: {type(dv).__name__})")
print(f"  MeasuresOfPredictivePower: {measures_of_power} (type: {type(measures_of_power).__name__})")
print(f"  NumberOfVariables: {num_of_variables} (type: {type(num_of_variables).__name__})")
print(f"  Criteria: {criteria} (type: {type(criteria).__name__})")
print(f"  Degree: {degree} (type: {type(degree).__name__})")
print(f"  UseXGBoost: {use_xgboost} (GPU available: {XGBOOST_GPU_AVAILABLE}, using {xgb_num_gpus} GPUs)")
if use_xgboost:
    print(f"    XGBEstimators: {xgb_n_estimators}, MaxDepth: {xgb_max_depth}, LR: {xgb_learning_rate}")
    print(f"    ColSample: {xgb_colsample}, SubSample: {xgb_subsample}")
    print(f"    L1 (reg_alpha): {xgb_reg_alpha}, L2 (reg_lambda): {xgb_reg_lambda}")
    print(f"    ImportanceThreshold: {xgb_importance_threshold:.0%}, TopN: {xgb_top_n}")
print("=" * 70)

# Validate flow variables
if (dv is not None and isinstance(dv, str) and dv != "missing" and
    measures_of_power is not None and isinstance(measures_of_power, str) and measures_of_power != "missing" and
    num_of_variables is not None and isinstance(num_of_variables, int) and num_of_variables > 0 and
    criteria is not None and criteria in ['Union', 'Intersection']):
    
    # Degree is only required for Intersection
    if criteria == 'Union':
        degree = 1
        contains_all_vars = True
        print("[OK] All flow variables valid - HEADLESS mode enabled")
    elif degree is not None and isinstance(degree, int) and degree > 0:
        contains_all_vars = True
        print("[OK] All flow variables valid - HEADLESS mode enabled")
    else:
        print("[ERROR] Degree not valid for Intersection criteria")
else:
    print("[ERROR] Flow variables incomplete or invalid - would use INTERACTIVE mode")
    print("  Required: DependentVariable, MeasuresOfPredictivePower, NumberOfVariables, Criteria")

# =============================================================================
# Main Processing Logic
# =============================================================================

# Initialize default outputs in case of errors
measures_out = pd.DataFrame()
selected_data = df.copy()
ebm_report_out = pd.DataFrame()
corr_matrix = pd.DataFrame()
vif_report = pd.DataFrame()
removed_for_vif = []

try:
    if contains_all_vars:
        # =========================================================================
        # HEADLESS MODE
        # =========================================================================
        print("Running in HEADLESS mode")
        
        # Parse measures
        measures_list = [m.strip() for m in measures_of_power.split(',')]
        
        results = run_headless_selection(
            df=df,
            dv=dv,
            measures_to_calc=measures_list,
            num_of_variables=num_of_variables,
            criteria=criteria,
            degree=degree,
            max_interactions=max_interactions,
            top_interactions=top_interactions,
            auto_add_missed=auto_add_missed,
            max_missed_to_add=max_missed_to_add,
            vif_threshold=vif_threshold,
            use_xgboost=use_xgboost,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
            xgb_colsample=xgb_colsample,
            xgb_subsample=xgb_subsample,
            xgb_reg_alpha=xgb_reg_alpha,
            xgb_reg_lambda=xgb_reg_lambda,
            xgb_importance_threshold=xgb_importance_threshold,
            xgb_top_n=xgb_top_n,
            xgb_num_gpus=xgb_num_gpus
        )
        
        measures_out = results['measures']
        selected_data = results['selected_data']
        ebm_report_out = results['ebm_report']
        corr_matrix = results['correlation_matrix']
        vif_report = results['vif_report']
        removed_for_vif = results.get('removed_for_vif', [])
        
    else:
        # =========================================================================
        # INTERACTIVE MODE or FALLBACK
        # =========================================================================
        if SHINY_AVAILABLE:
            print("Running in INTERACTIVE mode - launching Shiny UI...")
            results = run_variable_selection(df)
        else:
            print("=" * 70)
            print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
            print("Please provide flow variables for headless mode:")
            print("  - DependentVariable (string): e.g., 'IsFPD'")
            print("  - MeasuresOfPredictivePower (string): e.g., 'EntropyExplained, InformationValue, Gini'")
            print("  - NumberOfVariables (integer): e.g., 50")
            print("  - Criteria (string): 'Union' or 'Intersection'")
            print("  - Degree (integer): e.g., 2 (only needed for Intersection)")
            print("=" * 70)
            results = {'completed': False}
        
        if results['completed']:
            measures_out = results['measures']
            selected_data = results['selected_data']
            ebm_report_out = results['ebm_report']
            corr_matrix = results['correlation_matrix']
            vif_report = results['vif_report'] if results['vif_report'] is not None else pd.DataFrame()
            removed_for_vif = results.get('removed_for_vif', [])
            print("Interactive session completed successfully")
        else:
            print("Interactive session cancelled - returning empty results")
            measures_out = pd.DataFrame()
            selected_data = df.copy()
            ebm_report_out = pd.DataFrame()
            corr_matrix = pd.DataFrame()
            vif_report = pd.DataFrame()
            removed_for_vif = []

except Exception as e:
    import traceback
    print("=" * 70)
    print(f"ERROR during processing: {str(e)}")
    print("=" * 70)
    print(traceback.format_exc())
    print("=" * 70)
    print("Returning input data as fallback output")
    # Keep default values initialized above

# =============================================================================
# Output Tables
# =============================================================================

# Ensure all outputs are valid DataFrames
if measures_out is None:
    measures_out = pd.DataFrame()
if selected_data is None:
    selected_data = df.copy()
if ebm_report_out is None:
    ebm_report_out = pd.DataFrame()
if corr_matrix is None:
    corr_matrix = pd.DataFrame()
if vif_report is None:
    vif_report = pd.DataFrame()
if removed_for_vif is None:
    removed_for_vif = []

# Ensure VIF report has proper column types for KNIME Arrow conversion
if not vif_report.empty:
    if 'VIF' in vif_report.columns:
        vif_report['VIF'] = pd.to_numeric(vif_report['VIF'], errors='coerce').astype('Float64')
    if 'R_Squared' in vif_report.columns:
        vif_report['R_Squared'] = pd.to_numeric(vif_report['R_Squared'], errors='coerce').astype('Float64')
    if 'Removed' in vif_report.columns:
        vif_report['Removed'] = vif_report['Removed'].astype(bool)
    if 'Variable' in vif_report.columns:
        vif_report['Variable'] = vif_report['Variable'].astype(str)
    if 'Status' in vif_report.columns:
        vif_report['Status'] = vif_report['Status'].astype(str)

# Output 1: Measures table with selection flags, ranks, and EBM importance
knio.output_tables[0] = knio.Table.from_pandas(measures_out)

# Output 2: Selected data with WOE variables + EBM-missed + interaction columns + DV (ready for stepwise)
knio.output_tables[1] = knio.Table.from_pandas(selected_data)

# Output 3: EBM report (interactions + missed variables with inclusion status)
knio.output_tables[2] = knio.Table.from_pandas(ebm_report_out)

# Output 4: Correlation matrix for selected variables
knio.output_tables[3] = knio.Table.from_pandas(corr_matrix)

# Output 5: VIF report for multicollinearity detection
knio.output_tables[4] = knio.Table.from_pandas(vif_report)

print("=" * 70)
print("Variable Selection completed successfully")
print("=" * 70)
print(f"Output 1 (Measures): {len(measures_out)} variables with ranks and ML importance")
print(f"Output 2 (Selected Data): {len(selected_data.columns)} columns ready for stepwise regression")
print(f"Output 3 (ML Discovery Report): {len(ebm_report_out)} entries (EBM + XGBoost interactions + missed vars)")
print(f"Output 4 (Correlation Matrix): {corr_matrix.shape}")
print(f"Output 5 (VIF Report): {len(vif_report)} variables checked for multicollinearity")

# VIF removal summary
if removed_for_vif:
    print(f"\n[VIF REMOVAL] {len(removed_for_vif)} variables with VIF >= {vif_threshold} automatically removed:")
    for var in removed_for_vif:
        print(f"   - {var}")

# Remaining VIF summary
if not vif_report.empty and 'VIF' in vif_report.columns:
    # Count remaining moderate VIF (exclude removed ones)
    try:
        remaining_vif = vif_report[vif_report.get('Removed', False) == False] if 'Removed' in vif_report.columns else vif_report
        if not remaining_vif.empty:
            moderate_vif = len(remaining_vif[remaining_vif['VIF'] > 5])
            if moderate_vif > 0:
                print(f"\n[MODERATE VIF] {moderate_vif} variables with VIF 5-11 (acceptable but monitor)")
                print("   These are acceptable but monitor for stability")
    except:
        pass

print("=" * 70)

# =============================================================================
# Cleanup for Stability
# =============================================================================
sys.stdout.flush()

# Delete large objects to free memory
try:
    del df
except:
    pass

try:
    del df_bins
except:
    pass

try:
    del measures_out
except:
    pass

try:
    del selected_data
except:
    pass

try:
    del corr_matrix
except:
    pass

# Force garbage collection
gc.collect()

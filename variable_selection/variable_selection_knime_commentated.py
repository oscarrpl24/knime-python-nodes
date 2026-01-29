# =============================================================================
# VARIABLE SELECTION WITH EBM INTERACTION DISCOVERY FOR KNIME PYTHON SCRIPT NODE
# =============================================================================
# 
# PURPOSE:
# This script implements a comprehensive variable selection system for credit risk
# modeling workflows in KNIME 5.9. It selects the most predictive variables from
# a dataset using multiple statistical measures and machine learning techniques.
#
# KEY FEATURES:
# 1. Traditional statistical measures (Entropy, Information Value, Gini, Chi-Square, etc.)
# 2. EBM (Explainable Boosting Machine) for interaction discovery
# 3. XGBoost with GPU acceleration for robust feature importance
# 4. VIF (Variance Inflation Factor) for multicollinearity detection
# 5. Both interactive (Shiny UI) and headless (automated) modes
#
# INPUTS:
# 1. df_with_woe - DataFrame containing WOE-transformed columns (from WOE Editor)
#
# OUTPUTS:
# 1. measures - Predictive power measures for all variables with ranks
# 2. selected_data - Selected WOE variables + interactions + DV (ready for modeling)
# 3. ebm_report - EBM/XGBoost interaction discovery report
# 4. correlation_matrix - Correlation matrix for selected variables
# 5. vif_report - VIF report for multicollinearity detection
#
# COMPATIBLE WITH: KNIME 5.9, Python 3.9
# VERSION: 1.2
# RELEASE DATE: 2026-01-16
# =============================================================================

# =============================================================================
# SECTION 1: PACKAGE REINSTALLATION FLAG
# =============================================================================
# This flag controls whether to force-reinstall packages to fix compatibility issues.
# Set to True ONLY if you encounter numpy binary compatibility errors, then set back to False.

FIX_PACKAGES = False  # Boolean flag: True = reinstall packages, False = normal operation

# If FIX_PACKAGES is True, this block runs package reinstallation
if FIX_PACKAGES:
    # Import subprocess module to run system commands from Python
    import subprocess
    # Import sys module to access Python interpreter path
    import sys
    # Print status message to console
    print("Reinstalling packages to fix compatibility...")
    # Reinstall numpy using pip with --upgrade and --force-reinstall flags
    # sys.executable gets the path to the current Python interpreter
    # This ensures we use the same Python that KNIME is using
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'numpy'])
    # Reinstall scikit-learn (sklearn) which depends on numpy
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'scikit-learn'])
    # Reinstall interpret package (provides EBM functionality)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'interpret'])
    # Print completion message
    print("Done! Set FIX_PACKAGES = False and run again.")
    # Raise SystemExit exception to stop execution and prompt user to rerun
    raise SystemExit("Packages reinstalled. Please run the node again with FIX_PACKAGES = False")

# =============================================================================
# SECTION 2: CORE LIBRARY IMPORTS
# =============================================================================
# These are the essential libraries needed for the script to function.

# knime.scripting.io - KNIME's Python scripting interface
# This module provides access to input tables, output tables, and flow variables
import knime.scripting.io as knio

# pandas - Data manipulation library, the backbone of data processing
# Used for DataFrames, data transformations, and statistical operations
import pandas as pd

# numpy - Numerical computing library
# Provides fast array operations, mathematical functions, and linear algebra
import numpy as np

# warnings - Python's warning control system
# Used to suppress non-critical warnings during execution
import warnings

# gc - Garbage collector interface
# Used to manually trigger memory cleanup after large operations
import gc

# sys - System-specific parameters and functions
# Provides access to Python interpreter, stdout flushing, etc.
import sys

# random - Random number generation
# Used for generating random ports to avoid conflicts between instances
import random

# typing - Type hints for better code documentation
# Dict, List, Tuple, Optional, Any, Union are type hint classes
from typing import Dict, List, Tuple, Optional, Any, Union

# dataclasses - Decorator for creating data container classes
# @dataclass automatically generates __init__, __repr__, etc.
from dataclasses import dataclass

# Suppress all warnings to keep console output clean
# This prevents non-critical warnings from cluttering the output
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 3: STABILITY SETTINGS FOR MULTIPLE INSTANCE EXECUTION
# =============================================================================
# These settings ensure the script runs reliably when multiple instances
# are executed simultaneously (e.g., multiple KNIME workflows running in parallel).

# BASE_PORT - Starting port number for the Shiny web UI
# Different from model_analyzer (8051) to prevent port conflicts
BASE_PORT = 8052

# RANDOM_PORT_RANGE - Range for random port selection
# The actual port will be BASE_PORT + random(0, RANDOM_PORT_RANGE)
# This randomization prevents port collisions when multiple instances start
RANDOM_PORT_RANGE = 1000

# Import os module with underscore prefix (convention for "internal use")
import os as _os

# INSTANCE_ID - Unique identifier for this script instance
# Combines process ID (unique per running program) with random number
# Format: "12345_67890" where 12345 is PID and 67890 is random
INSTANCE_ID = f"{_os.getpid()}_{random.randint(10000, 99999)}"

# Threading environment variables - CRITICAL for stability
# These prevent conflicts when numpy/scipy use multi-threaded BLAS libraries

# NUMEXPR_MAX_THREADS - Limits numexpr (used by pandas) to single thread
_os.environ['NUMEXPR_MAX_THREADS'] = '1'

# OMP_NUM_THREADS - Limits OpenMP (parallel computing library) to single thread
_os.environ['OMP_NUM_THREADS'] = '1'

# OPENBLAS_NUM_THREADS - Limits OpenBLAS (linear algebra library) to single thread
_os.environ['OPENBLAS_NUM_THREADS'] = '1'

# MKL_NUM_THREADS - Limits Intel MKL (math kernel library) to single thread
_os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# SECTION 4: DEPENDENCY INSTALLATION AND IMPORT FUNCTIONS
# =============================================================================

def install_if_missing(package, import_name=None):
    """
    Install a Python package if it's not already installed.
    
    PURPOSE:
    Ensures required packages are available without manual pip install commands.
    This makes the script more portable across different KNIME installations.
    
    PARAMETERS:
    - package (str): The pip package name (e.g., 'scikit-learn')
    - import_name (str, optional): The Python import name if different from package
                                   (e.g., 'sklearn' for 'scikit-learn')
    
    HOW IT WORKS:
    1. Try to import the package
    2. If ImportError occurs, run pip install
    
    RETURNS: None (side effect: package installed if missing)
    """
    # If no import_name provided, assume it's the same as package name
    if import_name is None:
        import_name = package
    
    try:
        # Attempt to dynamically import the package
        # __import__ is a built-in function that imports modules by name string
        __import__(import_name)
    except ImportError:
        # Package not found - install it using pip
        import subprocess
        # Run pip install command as subprocess
        subprocess.check_call(['pip', 'install', package])


def fix_numpy_compatibility():
    """
    Attempt to fix numpy binary compatibility issues.
    
    PURPOSE:
    NumPy sometimes has binary incompatibility with other packages (especially sklearn)
    when packages are installed at different times or with different numpy versions.
    This function reinstalls numpy and sklearn to fix these issues.
    
    COMMON ERROR THIS FIXES:
    "numpy.dtype size changed, may indicate binary incompatibility"
    
    HOW IT WORKS:
    1. Reinstalls numpy with --force-reinstall
    2. Reinstalls scikit-learn with --force-reinstall
    3. Prints instructions if automatic fix fails
    
    RETURNS: None (side effect: packages reinstalled)
    """
    import subprocess
    # Print status message
    print("Attempting to fix NumPy compatibility issue...")
    try:
        # Force reinstall numpy to get fresh binaries
        subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall', 'numpy'])
        # Force reinstall scikit-learn which depends on numpy
        subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall', 'scikit-learn'])
        # Print success message
        print("Packages reinstalled. Please restart the KNIME workflow.")
    except Exception as e:
        # If automatic fix fails, print manual instructions
        print(f"Could not fix automatically: {e}")
        print("Please run these commands in your KNIME Python environment:")
        print("  pip install --upgrade --force-reinstall numpy")
        print("  pip install --upgrade --force-reinstall scikit-learn")


# =============================================================================
# SECTION 5: SKLEARN IMPORT WITH ERROR HANDLING
# =============================================================================
# This block handles the import of scikit-learn with special error handling
# for numpy binary compatibility issues.

try:
    # First, ensure scikit-learn is installed
    # 'scikit-learn' is the pip package name, 'sklearn' is the import name
    install_if_missing('scikit-learn', 'sklearn')
    
    # Import LogisticRegression - used for VIF calculation
    # VIF requires fitting regression models
    from sklearn.linear_model import LogisticRegression
    
    # Import StandardScaler - used for feature scaling (not heavily used here)
    from sklearn.preprocessing import StandardScaler
    
except ValueError as e:
    # ValueError with "numpy.dtype size changed" indicates binary incompatibility
    if "numpy.dtype size changed" in str(e):
        # Attempt automatic fix
        fix_numpy_compatibility()
        # Raise RuntimeError to stop execution and prompt restart
        raise RuntimeError(
            "NumPy binary incompatibility detected. "
            "Please restart your KNIME workflow after the packages are reinstalled."
        )
    # If it's a different ValueError, re-raise it
    raise

# =============================================================================
# SECTION 6: ADDITIONAL PACKAGE INSTALLATIONS
# =============================================================================
# Install additional packages required for EBM, Shiny UI, and visualization.

# interpret - Microsoft's interpretable ML library, provides EBM
install_if_missing('interpret')

# shiny - Python port of R's Shiny framework for interactive web apps
install_if_missing('shiny')

# shinywidgets - Widget support for Shiny, needed for plotly integration
install_if_missing('shinywidgets')

# plotly - Interactive visualization library
install_if_missing('plotly')

# =============================================================================
# SECTION 7: EBM (EXPLAINABLE BOOSTING MACHINE) IMPORT
# =============================================================================
# EBM is a glass-box model that automatically detects feature interactions.
# It's particularly valuable for credit risk modeling where interpretability matters.

try:
    # Import the ExplainableBoostingClassifier from interpret
    from interpret.glassbox import ExplainableBoostingClassifier
    # Set flag indicating EBM is available for use
    EBM_AVAILABLE = True
except (ValueError, ImportError) as e:
    # If import fails (numpy issues or missing package), disable EBM
    print(f"WARNING: EBM not available ({e})")
    print("EBM-based interaction discovery will be disabled.")
    # Set flag to False to skip EBM-related code later
    EBM_AVAILABLE = False
    # Set placeholder to None so code doesn't crash on references
    ExplainableBoostingClassifier = None

# =============================================================================
# SECTION 8: XGBOOST IMPORT WITH GPU DETECTION
# =============================================================================
# XGBoost is a gradient boosting library known for speed and accuracy.
# This section imports XGBoost and checks if GPU (CUDA) is available.

# Install xgboost if not present
install_if_missing('xgboost')

try:
    # Import xgboost library
    import xgboost as xgb
    # Set flag indicating XGBoost is available
    XGBOOST_AVAILABLE = True
    
    # GPU availability check
    try:
        # Create a test configuration with CUDA device
        test_params = {'device': 'cuda', 'tree_method': 'hist'}
        # Create a tiny test dataset (2 rows, 2 features)
        test_dmat = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        # Try to train a minimal model on GPU
        test_booster = xgb.train(test_params, test_dmat, num_boost_round=1, verbose_eval=False)
        # If we get here without exception, GPU is available
        XGBOOST_GPU_AVAILABLE = True
        print("XGBoost GPU (CUDA) available - will use GPU acceleration")
    except Exception as gpu_err:
        # GPU test failed - fall back to CPU
        XGBOOST_GPU_AVAILABLE = False
        print(f"XGBoost GPU not available ({gpu_err}), will use CPU")
        
except ImportError as e:
    # XGBoost not installed and installation failed
    print(f"WARNING: XGBoost not available ({e})")
    XGBOOST_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False
    xgb = None  # Set to None to prevent NameError later

# =============================================================================
# SECTION 9: SHINY UI IMPORT
# =============================================================================
# Shiny provides the interactive web interface for variable selection.
# This is only used in interactive mode, not headless mode.

try:
    # Import Shiny core components
    # App - Main application class
    # Inputs - Handles user input values
    # Outputs - Handles rendered outputs
    # Session - Manages user session
    # reactive - Creates reactive values and computations
    # render - Decorators for rendering outputs
    # ui - UI building functions
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Import shinywidgets for Plotly integration
    # render_plotly - Decorator for rendering Plotly charts
    # output_widget - UI placeholder for widgets
    from shinywidgets import render_plotly, output_widget
    
    # Import Plotly for interactive charts
    # graph_objects - Low-level Plotly interface
    # express - High-level Plotly interface
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Set flag indicating Shiny is available
    SHINY_AVAILABLE = True
    
except ImportError:
    # Shiny not available - interactive mode will be disabled
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# SECTION 10: DATA CLASSES
# =============================================================================
# Data classes are simple container classes that hold related data together.
# The @dataclass decorator automatically generates __init__, __repr__, etc.
# These classes define the structure of results returned by various functions.

@dataclass
class MeasuresResult:
    """
    Container for predictive measures calculation results.
    
    PURPOSE:
    Holds the output of predictive power measure calculations, including
    the DataFrame of measures and the list of selected variables.
    
    ATTRIBUTES:
    - measures (pd.DataFrame): DataFrame containing calculated measures for each variable
                               Columns include: Variable, Entropy, Information Value, etc.
    - selected_vars (List[str]): List of variable names that passed the selection criteria
    """
    measures: pd.DataFrame  # DataFrame with measure values per variable
    selected_vars: List[str]  # List of variable names meeting selection criteria


@dataclass
class EBMReport:
    """
    Container for EBM (Explainable Boosting Machine) interaction discovery results.
    
    PURPOSE:
    EBM automatically discovers feature interactions during training.
    This class holds the EBM results including feature importances and interactions.
    
    ATTRIBUTES:
    - feature_importances (pd.DataFrame): EBM importance scores for each feature
                                          Columns: Variable, EBM_Importance
    - interactions (pd.DataFrame): Detected interactions with magnitudes
                                   Columns: Variable_1, Variable_2, Interaction_Name, Magnitude
    - missed_by_traditional (List[str]): Variables important in EBM but not in traditional selection
                                         These are candidates to add back
    - ebm_model (Any): The trained EBM model object (for potential further use)
    """
    feature_importances: pd.DataFrame  # EBM feature importance scores
    interactions: pd.DataFrame  # Detected two-way interactions
    missed_by_traditional: List[str]  # Variables EBM found important but traditional missed
    ebm_model: Any  # The trained EBM model itself


@dataclass
class XGBoostReport:
    """
    Container for XGBoost feature discovery results.
    
    PURPOSE:
    XGBoost provides robust feature importance through gradient boosting.
    This class holds XGBoost results including multiple importance metrics.
    
    ATTRIBUTES:
    - feature_importances (pd.DataFrame): XGBoost importance scores (gain, cover, weight)
                                          Columns: Variable, XGB_Gain, XGB_Cover, XGB_Weight, XGB_Importance
    - interactions (pd.DataFrame): Top feature pairs from tree structure analysis
                                   Columns: Variable_1, Variable_2, Interaction_Name, Magnitude, Source
    - missed_by_traditional (List[str]): Variables important in XGBoost but not in traditional
    - xgb_model (Any): The trained XGBoost model object
    - gpu_used (bool): Whether GPU acceleration was used for training
    """
    feature_importances: pd.DataFrame  # Multiple importance metrics from XGBoost
    interactions: pd.DataFrame  # Interactions inferred from tree structure
    missed_by_traditional: List[str]  # Variables XGBoost found important but traditional missed
    xgb_model: Any  # The trained XGBoost booster
    gpu_used: bool  # Flag indicating if GPU was used


@dataclass
class BinResult:
    """
    Container for binning operation results.
    
    PURPOSE:
    Binning is the process of grouping continuous variables into discrete intervals.
    This class holds both summary statistics and detailed bin information.
    
    ATTRIBUTES:
    - var_summary (pd.DataFrame): Summary stats for each variable
                                  Columns: var, varType, iv, ent, trend, monTrend, flipRatio, numBins, purNode
    - bin (pd.DataFrame): Detailed information for each bin of each variable
                          Columns: var, bin, count, bads, goods, propn, bad_rate, iv, ent, etc.
    """
    var_summary: pd.DataFrame  # One row per variable with overall stats
    bin: pd.DataFrame  # Multiple rows per variable, one per bin


# =============================================================================
# SECTION 11: BINNING FUNCTIONS
# =============================================================================
# These functions implement optimal binning for variables, equivalent to R's logiBin::getBins.
# Binning transforms continuous variables into discrete intervals optimized for prediction.

def calculate_bin_entropy(goods: int, bads: int) -> float:
    """
    Calculate entropy for a single bin.
    
    PURPOSE:
    Entropy measures the impurity/uncertainty in a bin. A bin with all goods (0% bad)
    or all bads (100% bad) has entropy 0 (pure). A 50/50 split has maximum entropy.
    
    PARAMETERS:
    - goods (int): Number of "good" outcomes (non-default/non-event) in the bin
    - bads (int): Number of "bad" outcomes (default/event) in the bin
    
    FORMULA:
    Entropy = -sum(p * log2(p)) for each class probability p
    For binary: E = -(p_bad * log2(p_bad)) - (p_good * log2(p_good))
    
    RETURNS:
    float: Entropy value between 0 (pure bin) and 1 (50/50 split), rounded to 4 decimals
    """
    # Calculate total count in this bin
    total = goods + bads
    
    # Handle edge cases: empty bin or pure bin (all one class)
    # Pure bins have 0 entropy (no uncertainty)
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    # Calculate probability of each class
    p_good = goods / total  # Proportion of goods
    p_bad = bads / total    # Proportion of bads
    
    # Apply entropy formula: E = -sum(p * log2(p))
    # Using log base 2 means max entropy for binary classification is 1.0
    entropy_val = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    
    # Round to 4 decimal places for cleaner output
    return round(entropy_val, 4)


def get_var_type(series: pd.Series) -> str:
    """
    Determine if a variable is numeric or factor (categorical).
    
    PURPOSE:
    Different binning strategies are used for numeric vs categorical variables:
    - Numeric: Use decision tree splits to find optimal cut points
    - Factor: Each unique value becomes its own bin
    
    PARAMETERS:
    - series (pd.Series): The variable's data column
    
    LOGIC:
    - If numeric AND has more than 10 unique values -> 'numeric' (continuous)
    - If numeric AND has 10 or fewer unique values -> 'factor' (treat as categorical)
    - If non-numeric (string, object) -> 'factor'
    
    RETURNS:
    str: Either 'numeric' or 'factor'
    """
    # Check if the series has a numeric dtype (int, float, etc.)
    if pd.api.types.is_numeric_dtype(series):
        # For numeric types, check cardinality (number of unique values)
        if series.nunique() <= 10:
            # Low cardinality numeric -> treat as factor
            # Example: rating scores 1-5
            return 'factor'
        # High cardinality numeric -> treat as continuous
        return 'numeric'
    # Non-numeric types (strings, objects) are always factors
    return 'factor'


def _get_decision_tree_splits(
    x: pd.Series, 
    y: pd.Series, 
    min_prop: float = 0.01,
    max_bins: int = 10
) -> List[float]:
    """
    Use a decision tree to find optimal split points for numeric variables.
    
    PURPOSE:
    Decision trees naturally find the best split points that maximize information gain.
    This function extracts those split points to use for binning.
    
    PARAMETERS:
    - x (pd.Series): The independent variable (feature) values
    - y (pd.Series): The dependent variable (target) values (0/1)
    - min_prop (float): Minimum proportion of data in each leaf (default 1%)
                        Prevents tiny bins with unstable statistics
    - max_bins (int): Maximum number of bins to create (default 10)
    
    HOW IT WORKS:
    1. Remove missing values from both x and y
    2. Fit a DecisionTreeClassifier with constrained complexity
    3. Extract threshold values from the tree nodes
    4. Return sorted unique thresholds as split points
    
    RETURNS:
    List[float]: Sorted list of split points (thresholds)
    """
    # Import DecisionTreeClassifier here to avoid import overhead if not used
    from sklearn.tree import DecisionTreeClassifier
    
    # Create mask for non-null values in both x and y
    # Both must be non-null for a valid observation
    mask = x.notna() & y.notna()
    
    # Extract clean data as numpy arrays
    # Reshape x to 2D array (required by sklearn): shape (n_samples, 1)
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    # If no valid data, return empty list
    if len(x_clean) == 0:
        return []
    
    # Calculate minimum samples per leaf
    # This ensures each bin has at least min_prop * n samples
    min_samples_leaf = max(int(len(x_clean) * min_prop), 1)
    
    # Create decision tree with complexity constraints
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,        # Limits total number of leaves (bins)
        min_samples_leaf=min_samples_leaf,  # Minimum samples per bin
        random_state=42                  # For reproducibility
    )
    
    try:
        # Fit the tree to the data
        tree.fit(x_clean, y_clean)
    except Exception:
        # If fitting fails (e.g., constant column), return empty list
        return []
    
    # Extract thresholds from the tree structure
    # tree.tree_.threshold contains the split value at each node
    # -2 is used for leaf nodes (no split)
    thresholds = tree.tree_.threshold
    
    # Filter out leaf nodes (threshold = -2)
    thresholds = thresholds[thresholds != -2]
    
    # Sort and remove duplicates
    thresholds = sorted(set(thresholds))
    
    return thresholds


def _create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """
    Create bin DataFrame for a numeric variable based on split points.
    
    PURPOSE:
    Given split points from the decision tree, this function counts goods and bads
    in each resulting bin and formats the results as a DataFrame.
    
    PARAMETERS:
    - df (pd.DataFrame): The full dataset
    - var (str): Name of the numeric variable being binned
    - y_var (str): Name of the target/dependent variable
    - splits (List[float]): Split points from decision tree
    
    BIN CREATION LOGIC:
    For splits [10, 20, 30], creates bins:
    - (-inf, 10]: var <= 10
    - (10, 20]: var > 10 AND var <= 20
    - (20, 30]: var > 20 AND var <= 30
    - (30, inf): var > 30
    - Special bin for NA values
    
    RETURNS:
    pd.DataFrame: Bin information with columns: var, bin, count, bads, goods
    """
    # Extract the variable column and target column
    x = df[var]
    y = df[y_var]
    
    # Initialize list to collect bin data
    bins_data = []
    
    # Sort splits and create edges with -inf and +inf boundaries
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    # Iterate through consecutive edge pairs to define bins
    for i in range(len(edges) - 1):
        lower = edges[i]      # Lower bound of this bin
        upper = edges[i + 1]  # Upper bound of this bin
        
        # Create mask for observations falling in this bin
        # Handle first bin (no lower bound) specially
        if lower == -np.inf:
            # First bin: x <= upper (excluding NAs)
            mask = (x <= upper) & x.notna()
            # Create human-readable bin rule
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            # Last bin: x > lower (excluding NAs)
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            # Middle bins: lower < x <= upper
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        # Count observations in this bin
        count = mask.sum()
        
        # Only create bin if it has observations
        if count > 0:
            # Count bads (target = 1) in this bin
            bads = y[mask].sum()
            # Goods = total - bads
            goods = count - bads
            # Add bin to list
            bins_data.append({
                'var': var,           # Variable name
                'bin': bin_rule,      # Human-readable bin definition
                'count': count,       # Total observations in bin
                'bads': int(bads),    # Number of bad outcomes
                'goods': int(goods)   # Number of good outcomes
            })
    
    # Handle NA values as a separate bin
    na_mask = x.isna()
    if na_mask.sum() > 0:
        na_count = na_mask.sum()
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",  # R-style NA notation for compatibility
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
    
    # Convert list of dicts to DataFrame
    return pd.DataFrame(bins_data)


def _create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str
) -> pd.DataFrame:
    """
    Create bin DataFrame for a factor (categorical) variable.
    
    PURPOSE:
    For categorical variables, each unique value becomes its own bin.
    This function counts goods and bads for each category.
    
    PARAMETERS:
    - df (pd.DataFrame): The full dataset
    - var (str): Name of the categorical variable being binned
    - y_var (str): Name of the target/dependent variable
    
    RETURNS:
    pd.DataFrame: Bin information with columns: var, bin, count, bads, goods
    """
    # Extract the variable column and target column
    x = df[var]
    y = df[y_var]
    
    # Initialize list to collect bin data
    bins_data = []
    
    # Get unique non-null values
    unique_vals = x.dropna().unique()
    
    # Create a bin for each unique value
    for val in unique_vals:
        # Create mask for this category
        mask = x == val
        count = mask.sum()
        
        if count > 0:
            # Count bads and goods
            bads = y[mask].sum()
            goods = count - bads
            # Add bin with R-style %in% notation for compatibility
            bins_data.append({
                'var': var,
                'bin': f'{var} %in% c("{val}")',  # R-style category matching
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
    Update bin statistics including proportions, bad rates, IV, entropy, and trend.
    
    PURPOSE:
    After basic bin counts are calculated, this function adds derived statistics
    that are used for predictive power measurement and analysis.
    
    PARAMETERS:
    - bin_df (pd.DataFrame): DataFrame with columns: var, bin, count, bads, goods
    
    CALCULATED STATISTICS:
    - propn: Percentage of total observations in this bin
    - bad_rate: Bad rate (percentage of bads) in this bin
    - goodCap: Proportion of total goods captured by this bin
    - badCap: Proportion of total bads captured by this bin
    - iv: Information Value contribution from this bin
    - ent: Entropy of this bin
    - purNode: 'Y' if bin is pure (all goods or all bads), 'N' otherwise
    - trend: 'I' (increasing) or 'D' (decreasing) bad rate vs previous bin
    
    RETURNS:
    pd.DataFrame: Input DataFrame with additional statistical columns
    """
    # Return empty DataFrame if input is empty
    if bin_df.empty:
        return bin_df
    
    # Create a copy to avoid modifying the original
    df = bin_df.copy()
    
    # Calculate totals across all bins
    total_count = df['count'].sum()   # Total observations
    total_goods = df['goods'].sum()   # Total good outcomes
    total_bads = df['bads'].sum()     # Total bad outcomes
    
    # Calculate proportion of observations in each bin (as percentage)
    df['propn'] = round(df['count'] / total_count * 100, 2)
    
    # Calculate bad rate in each bin (as percentage)
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    
    # Calculate goods capture: what % of total goods are in this bin
    # Used for IV calculation and distribution analysis
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    
    # Calculate bads capture: what % of total bads are in this bin
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    # Calculate Information Value (IV) contribution for each bin
    # IV = (goodCap - badCap) * ln(goodCap / badCap)
    # Using 0.0001 instead of 0 to avoid log(0) and division by zero
    df['iv'] = round((df['goodCap'] - df['badCap']) * np.log(
        np.where(df['goodCap'] == 0, 0.0001, df['goodCap']) / 
        np.where(df['badCap'] == 0, 0.0001, df['badCap'])
    ), 4)
    
    # Replace infinite values with 0 (can occur with extreme distributions)
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    # Calculate entropy for each bin using the helper function
    df['ent'] = df.apply(
        lambda row: calculate_bin_entropy(row['goods'], row['bads']), 
        axis=1
    )
    
    # Mark pure nodes (bins with only goods or only bads)
    # Pure nodes indicate potential overfitting or special populations
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    # Calculate trend: is bad rate increasing or decreasing vs previous bin?
    # This helps assess monotonicity of the variable's relationship with target
    df['trend'] = None
    bad_rates = df['bad_rate'].values
    
    for i in range(1, len(bad_rates)):
        # Skip NA bins when calculating trend
        if 'is.na' not in str(df.iloc[i]['bin']):
            if bad_rates[i] >= bad_rates[i-1]:
                # Bad rate increased -> Increasing trend
                df.iloc[i, df.columns.get_loc('trend')] = 'I'
            else:
                # Bad rate decreased -> Decreasing trend
                df.iloc[i, df.columns.get_loc('trend')] = 'D'
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Add a summary (Total) row to the bin DataFrame.
    
    PURPOSE:
    The Total row provides aggregate statistics for the entire variable,
    including overall IV, weighted entropy, trend analysis, and bin count.
    
    PARAMETERS:
    - bin_df (pd.DataFrame): DataFrame with bin statistics
    - var (str): Name of the variable (for the Total row)
    
    TOTAL ROW STATISTICS:
    - iv: Sum of all bin IVs (total Information Value)
    - ent: Weighted average entropy across bins
    - monTrend: 'Y' if monotonic (all I or all D trends), 'N' otherwise
    - flipRatio: Proportion of minority trend direction (measures monotonicity violation)
    - numBins: Number of bins (excluding Total row)
    - purNode: 'Y' if any bin is pure
    
    RETURNS:
    pd.DataFrame: Input DataFrame with Total row appended
    """
    df = bin_df.copy()
    
    # Calculate totals from all bins
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    # Sum of IV across all bins (replace infinites with 0 first)
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()
    
    # Weighted average entropy (entropy * proportion of observations)
    if total_count > 0:
        total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4)
    else:
        total_ent = 0
    
    # Check for monotonic trend
    # Get unique trend values (excluding None)
    trends = df[df['trend'].notna()]['trend'].unique()
    # Monotonic if only one trend direction (all I or all D)
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    # Calculate flip ratio: measures how much the trend violates monotonicity
    # 0.0 = perfectly monotonic, 0.5 = maximum non-monotonicity
    incr_count = len(df[df['trend'] == 'I'])  # Count of increasing transitions
    decr_count = len(df[df['trend'] == 'D'])  # Count of decreasing transitions
    total_trend_count = incr_count + decr_count
    # Flip ratio is the proportion of the minority direction
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    # Determine overall trend direction (majority wins)
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    
    # Check if any bin is pure (all goods or all bads)
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    
    # Number of bins (current df length, before adding Total)
    num_bins = len(df)
    
    # Create Total row as a single-row DataFrame
    total_row = pd.DataFrame([{
        'var': var,
        'bin': 'Total',
        'count': total_count,
        'bads': total_bads,
        'goods': total_goods,
        'propn': 100.0,  # Total is always 100%
        'bad_rate': round(total_bads / total_count * 100, 2) if total_count > 0 else 0,
        'goodCap': 1.0,  # Total captures 100% of goods
        'badCap': 1.0,   # Total captures 100% of bads
        'iv': round(total_iv, 4),
        'ent': total_ent,
        'purNode': has_pure_node,
        'trend': overall_trend,
        'monTrend': mon_trend,
        'flipRatio': round(flip_ratio, 4),
        'numBins': num_bins
    }])
    
    # Concatenate bins with Total row
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
    
    PURPOSE:
    This is the main entry point for binning, equivalent to R's logiBin::getBins.
    It processes multiple variables, determines their types, creates bins,
    and calculates all statistics.
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset to analyze
    - y_var (str): Name of the binary target variable (0/1)
    - x_vars (List[str]): List of independent variable names to bin
    - min_prop (float): Minimum proportion of data per bin (default 1%)
    - max_bins (int): Maximum number of bins per variable (default 10)
    
    PROCESS:
    For each variable:
    1. Determine type (numeric or factor)
    2. For numeric: use decision tree to find split points
    3. For factor: use each unique value as a bin
    4. Calculate bin statistics (IV, entropy, etc.)
    5. Add Total row with aggregate statistics
    
    RETURNS:
    BinResult: Container with var_summary (one row per variable) and bin (detailed bins)
    """
    # Initialize lists to collect results
    all_bins = []        # Will hold bin DataFrames for all variables
    var_summaries = []   # Will hold summary dicts for all variables
    
    # Process each variable
    for var in x_vars:
        # Skip if variable not in DataFrame
        if var not in df.columns:
            continue
        
        # Determine variable type (numeric or factor)
        var_type = get_var_type(df[var])
        
        # Create bins based on variable type
        if var_type == 'numeric':
            # Use decision tree to find optimal split points
            splits = _get_decision_tree_splits(df[var], df[y_var], min_prop, max_bins)
            # Create bins using the split points
            bin_df = _create_numeric_bins(df, var, y_var, splits)
        else:
            # For factors, each unique value is a bin
            bin_df = _create_factor_bins(df, var, y_var)
        
        # Skip if no bins were created
        if bin_df.empty:
            continue
        
        # Calculate bin statistics (IV, entropy, trend, etc.)
        bin_df = update_bin_stats(bin_df)
        
        # Add Total row with aggregate statistics
        bin_df = add_total_row(bin_df, var)
        
        # Extract summary from Total row for variable summary table
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
        
        # Add this variable's bins to the collection
        all_bins.append(bin_df)
    
    # Combine all bin DataFrames into one
    if all_bins:
        combined_bins = pd.concat(all_bins, ignore_index=True)
    else:
        combined_bins = pd.DataFrame()
    
    # Create variable summary DataFrame
    var_summary_df = pd.DataFrame(var_summaries)
    
    # Return as BinResult container
    return BinResult(var_summary=var_summary_df, bin=combined_bins)


# =============================================================================
# SECTION 12: PREDICTIVE MEASURES FUNCTIONS
# =============================================================================
# These functions calculate various measures of predictive power for variables.
# Each measure quantifies how well a variable can predict the target variable.

def entropy(probs: np.ndarray) -> float:
    """
    Core entropy calculation for an array of probabilities.
    
    PURPOSE:
    Entropy measures uncertainty/disorder in a probability distribution.
    Used as a building block for Entropy Explained measure.
    
    FORMULA:
    H = -sum(p * log2(p)) for all probabilities p
    
    PARAMETERS:
    - probs (np.ndarray): Array of probabilities (must sum to 1)
    
    RETURNS:
    float: Entropy value (0 = no uncertainty, higher = more uncertainty)
    """
    # Convert to float array
    probs = np.array(probs, dtype=float)
    # Remove zeros to avoid log(0) which is undefined
    probs = probs[probs > 0]
    # If all probabilities were zero, entropy is 0
    if len(probs) == 0:
        return 0.0
    # Apply entropy formula: -sum(p * log2(p))
    return -np.sum(probs * np.log2(probs))


def input_entropy(bins_df: pd.DataFrame) -> float:
    """
    Calculate input (marginal) entropy for a variable's bins.
    
    PURPOSE:
    Input entropy is the baseline entropy of the target variable
    before considering the feature. It represents the uncertainty
    we're trying to reduce.
    
    FORMULA:
    H(Y) = -p(good)*log2(p(good)) - p(bad)*log2(p(bad))
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    
    RETURNS:
    float: Input entropy of the target variable
    """
    # Get totals from the last row (Total row)
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = total_goods + total_bads
    
    # Handle edge case of no data
    if total == 0:
        return 0.0
    
    # Calculate class probabilities
    probs = np.array([total_goods / total, total_bads / total])
    # Calculate and round entropy
    return round(entropy(probs), 5)


def output_entropy(bins_df: pd.DataFrame) -> float:
    """
    Calculate output (conditional) entropy for a variable's bins.
    
    PURPOSE:
    Output entropy is the weighted average entropy of the target variable
    within each bin. It represents the remaining uncertainty after
    knowing the bin assignment.
    
    FORMULA:
    H(Y|X) = sum over bins: (bin_count / total) * H(Y in bin)
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    
    RETURNS:
    float: Conditional entropy (lower = more information gained)
    """
    # Exclude the Total row (we want individual bins only)
    bins_only = bins_df.iloc[:-1]
    # Get total count from the last row
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    # Calculate weighted entropy across bins
    weighted_entropy = 0.0
    
    for _, row in bins_only.iterrows():
        count = row['count']
        if count == 0:
            continue  # Skip empty bins
        
        goods = row['goods']
        bads = row['bads']
        
        # Build probability array for this bin
        probs = []
        if goods > 0:
            probs.append(goods / count)
        if bads > 0:
            probs.append(bads / count)
        
        # Calculate bin entropy and weight by bin size
        if len(probs) > 0:
            bin_entropy = entropy(np.array(probs))
            # Weight by proportion of observations in this bin
            weighted_entropy += (count / total) * bin_entropy
    
    return round(weighted_entropy, 5)


def gini_impurity(totals: np.ndarray, overall_total: float) -> float:
    """
    Core Gini impurity calculation.
    
    PURPOSE:
    Gini impurity measures how often a randomly chosen element would be
    incorrectly classified. Used for Gini predictive measure.
    
    FORMULA:
    Gini = 1 - sum(p_i^2) for all classes i
    
    PARAMETERS:
    - totals (np.ndarray): Array of class counts [goods, bads]
    - overall_total (float): Total count across all classes
    
    RETURNS:
    float: Gini impurity (0 = pure, 0.5 = maximum impurity for binary)
    """
    if overall_total == 0:
        return 0.0
    # Gini = 1 - sum of squared proportions
    return 1 - np.sum((totals / overall_total) ** 2)


def input_gini(bins_df: pd.DataFrame) -> float:
    """
    Calculate input (marginal) Gini impurity for a variable.
    
    PURPOSE:
    Baseline Gini impurity before considering the feature.
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    
    RETURNS:
    float: Input Gini impurity
    """
    # Get totals from the last (Total) row
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = bins_df['count'].iloc[-1]
    
    # Create array of class counts
    totals = np.array([total_goods, total_bads])
    return round(gini_impurity(totals, total), 5)


def output_gini(bins_df: pd.DataFrame) -> float:
    """
    Calculate output (weighted) Gini impurity for a variable.
    
    PURPOSE:
    Weighted average Gini impurity within each bin.
    Lower output Gini means the variable better separates classes.
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    
    RETURNS:
    float: Weighted Gini impurity (lower = more separation)
    """
    # Exclude Total row
    bins_only = bins_df.iloc[:-1]
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    # Calculate weighted Gini across bins
    weighted_gini = 0.0
    
    for _, row in bins_only.iterrows():
        count = row['count']
        if count == 0:
            continue
        
        goods = row['goods']
        bads = row['bads']
        
        # Calculate Gini for this bin
        bin_totals = np.array([goods, bads])
        bin_gini = gini_impurity(bin_totals, count)
        # Weight by proportion in this bin
        weighted_gini += (count / total) * bin_gini
    
    return round(weighted_gini, 5)


def chi_square(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate Pearson Chi-Square statistic.
    
    PURPOSE:
    Chi-Square tests whether observed counts differ significantly from
    expected counts under independence assumption.
    
    FORMULA:
    X^2 = sum((O - E)^2 / E) for all cells
    
    PARAMETERS:
    - observed (np.ndarray): Array of observed counts
    - expected (np.ndarray): Array of expected counts under null hypothesis
    
    RETURNS:
    float: Chi-Square statistic (higher = more departure from independence)
    """
    # Convert to float arrays
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    
    # Avoid division by zero - only use cells with positive expected
    mask = expected > 0
    if not np.any(mask):
        return 0.0
    
    # Apply Chi-Square formula
    chi_sq = np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask])
    return chi_sq


def likelihood_ratio(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate Likelihood Ratio (G-test) statistic.
    
    PURPOSE:
    Alternative to Chi-Square that uses log-likelihood ratio.
    Often preferred for small sample sizes.
    
    FORMULA:
    G = 2 * sum(O * ln(O / E)) for all cells
    
    PARAMETERS:
    - observed (np.ndarray): Array of observed counts
    - expected (np.ndarray): Array of expected counts
    
    RETURNS:
    float: G-test statistic (higher = more departure from independence)
    """
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    
    # Need both observed and expected to be positive for log
    mask = (observed > 0) & (expected > 0)
    if not np.any(mask):
        return 0.0
    
    # Apply G-test formula
    g_stat = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
    return g_stat


def chi_mls_calc(bins_df: pd.DataFrame, method: str = 'chisquare') -> float:
    """
    Calculate Chi-Square or Likelihood Ratio for a variable's bins.
    
    PURPOSE:
    Computes expected counts under independence and calculates
    either Chi-Square or G-test statistic.
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    - method (str): 'chisquare' for Pearson Chi-Square, 'mls' for G-test
    
    RETURNS:
    float: The calculated statistic
    """
    # Exclude Total row
    bins_only = bins_df.iloc[:-1]
    # Get overall totals
    total_goods = bins_df['goods'].iloc[-1]
    total_bads = bins_df['bads'].iloc[-1]
    total = bins_df['count'].iloc[-1]
    
    if total == 0:
        return 0.0
    
    # Calculate overall proportions
    prop_goods = total_goods / total
    prop_bads = total_bads / total
    
    # Calculate expected values under independence
    # Expected = (bin_count / total) * proportion * total = bin_count * proportion
    exp_goods = (bins_only['count'] / total) * prop_goods * total
    exp_bads = (bins_only['count'] / total) * prop_bads * total
    
    # Combine expected and observed into arrays
    expected = np.concatenate([exp_goods.values, exp_bads.values])
    observed = np.concatenate([bins_only['goods'].values, bins_only['bads'].values])
    
    # Calculate requested statistic
    if method == 'chisquare':
        return round(chi_square(observed, expected), 5)
    elif method == 'mls':
        return round(likelihood_ratio(observed, expected), 5)
    else:
        return 0.0


def odds_ratio(bins_df: pd.DataFrame) -> Optional[float]:
    """
    Calculate Odds Ratio for binary factor variables only.
    
    PURPOSE:
    Odds Ratio measures the association between a binary predictor and
    binary outcome. Only applicable when variable has exactly 2 levels.
    
    FORMULA:
    OR = (goods1/goods2) / (bads1/bads2)
       = (goods1 * bads2) / (goods2 * bads1)
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Bin DataFrame including Total row
    
    RETURNS:
    Optional[float]: Odds Ratio if binary, None otherwise
    """
    # Exclude Total row
    bins_only = bins_df.iloc[:-1]
    
    # Only calculate for exactly 2 bins (binary factor)
    if len(bins_only) != 2:
        return None
    
    # Extract counts from both bins
    goods1 = bins_only['goods'].iloc[0]
    goods2 = bins_only['goods'].iloc[1]
    bads1 = bins_only['bads'].iloc[0]
    bads2 = bins_only['bads'].iloc[1]
    
    # Avoid division by zero
    if goods2 == 0 or bads2 == 0 or bads1 == 0:
        return None
    
    # Calculate proportions
    prop_good = goods1 / goods2
    prop_bad = bads1 / bads2
    
    if prop_bad == 0:
        return None
    
    # Calculate Odds Ratio
    return round(prop_good / prop_bad, 5)


def calculate_all_measures(
    bins_df: pd.DataFrame,
    var_summary: pd.DataFrame,
    measures_to_calc: List[str]
) -> pd.DataFrame:
    """
    Calculate all selected predictive measures for all variables.
    
    PURPOSE:
    Main function that computes multiple predictive power measures
    for all variables based on their binning results.
    
    PARAMETERS:
    - bins_df (pd.DataFrame): Combined bin data for all variables
    - var_summary (pd.DataFrame): Variable summary from get_bins
    - measures_to_calc (List[str]): List of measures to calculate
      Options: 'EntropyExplained', 'InformationValue', 'OddsRatio',
               'LikelihoodRatio', 'PearsonChiSquare', 'Gini'
    
    RETURNS:
    pd.DataFrame: One row per variable with calculated measure values
    """
    # Get list of unique variables
    variables = var_summary['var'].unique().tolist()
    results = []
    
    # Process each variable
    for var in variables:
        # Get bins for this variable
        var_bins = bins_df[bins_df['var'] == var].copy()
        # Get summary info for this variable
        var_info = var_summary[var_summary['var'] == var]
        
        if var_bins.empty:
            continue
        
        # Check if this is a binary factor (for Odds Ratio)
        is_binary = len(var_bins[var_bins['bin'] != 'Total']) == 2
        # Also check for %in% syntax which indicates categorical
        if '%in%' in str(var_bins['bin'].values):
            is_binary = True
        
        # Initialize result row
        row = {'Variable': var}
        
        # Calculate each requested measure
        
        # ENTROPY EXPLAINED: How much uncertainty is reduced by knowing this variable
        # Formula: 1 - (output_entropy / input_entropy)
        # Range: 0 (no reduction) to 1 (complete reduction)
        if 'EntropyExplained' in measures_to_calc:
            in_ent = input_entropy(var_bins)
            out_ent = output_entropy(var_bins)
            if in_ent > 0:
                row['Entropy'] = round(1 - (out_ent / in_ent), 5)
            else:
                row['Entropy'] = 0.0
        
        # INFORMATION VALUE: Standard measure in credit scoring
        # Sum of (goodCap - badCap) * ln(goodCap / badCap) across bins
        # Interpretation: <0.02 weak, 0.02-0.1 medium, 0.1-0.3 strong, >0.3 very strong
        if 'InformationValue' in measures_to_calc:
            if not var_info.empty and 'iv' in var_info.columns:
                row['Information Value'] = var_info['iv'].iloc[0]
            else:
                # Calculate from bins if not in summary
                total_row = var_bins[var_bins['bin'] == 'Total']
                if not total_row.empty and 'iv' in total_row.columns:
                    row['Information Value'] = total_row['iv'].iloc[0]
                else:
                    row['Information Value'] = 0.0
        
        # ODDS RATIO: Only for binary predictors
        # Higher OR = stronger association
        if 'OddsRatio' in measures_to_calc:
            if is_binary:
                row['Odds Ratio'] = odds_ratio(var_bins)
            else:
                row['Odds Ratio'] = None  # Not applicable for non-binary
        
        # LIKELIHOOD RATIO (G-test): Alternative to Chi-Square
        if 'LikelihoodRatio' in measures_to_calc:
            row['Likelihood Ratio'] = chi_mls_calc(var_bins, method='mls')
        
        # PEARSON CHI-SQUARE: Tests independence between variable and target
        if 'PearsonChiSquare' in measures_to_calc:
            row['Chi-Square'] = chi_mls_calc(var_bins, method='chisquare')
        
        # GINI: How much Gini impurity is reduced by knowing this variable
        # Formula: 1 - (output_gini / input_gini)
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
    Filter variables based on selection criteria.
    
    PURPOSE:
    Selects the most predictive variables based on their rankings across
    multiple measures. Two strategies are available:
    - Union: Select if variable is in top N for ANY measure
    - Intersection: Select if variable is in top N for at least 'degree' measures
    
    PARAMETERS:
    - measures_df (pd.DataFrame): DataFrame with calculated measures
    - criteria (str): 'Union' or 'Intersection'
    - num_of_variables (int): Top N variables to consider FOR EACH MEASURE
    - degree (int): For Intersection, minimum measures a variable must be in top N
    
    HOW IT WORKS:
    1. For each measure, identify the top N variables by that measure
    2. Count how many top-N lists each variable appears in
    3. Apply selection criteria (Union = in at least 1, Intersection = in at least 'degree')
    
    RETURNS:
    pd.DataFrame: Input DataFrame with added columns: ListCount, InMeasures, Selected
    """
    df = measures_df.copy()
    
    # Define measure columns and their ranking direction
    # True = higher is better (most measures work this way)
    measure_cols = {
        'Entropy': True,           # Higher Entropy Explained = more predictive
        'Information Value': True, # Higher IV = more predictive
        'Odds Ratio': True,        # Higher OR = stronger association
        'Likelihood Ratio': True,  # Higher LR = more departure from independence
        'Chi-Square': True,        # Higher Chi-Sq = more association
        'Gini': True               # Higher Gini = more predictive
    }
    
    # For each measure, identify which variables are in the top N
    top_n_sets = {}  # Dictionary: measure name -> set of top N variable names
    
    for col, higher_is_better in measure_cols.items():
        # Skip if this measure wasn't calculated
        if col not in df.columns:
            continue
        
        # Get rows with non-null values for this measure
        valid_df = df[df[col].notna()].copy()
        if len(valid_df) == 0:
            continue
        
        # Sort by measure value (descending if higher is better)
        sorted_df = valid_df.sort_values(col, ascending=not higher_is_better)
        
        # Get top N (or fewer if not enough variables)
        top_n = min(num_of_variables, len(sorted_df))
        top_vars = sorted_df.head(top_n)['Variable'].tolist()
        top_n_sets[col] = set(top_vars)
        
        # Debug output: show cutoff value and top 5 variables
        cutoff_val = sorted_df.iloc[top_n - 1][col] if top_n > 0 else None
        cutoff_str = f"{cutoff_val:.4f}" if cutoff_val is not None else "N/A"
        top5_vars = sorted_df.head(5)['Variable'].tolist()
        top5_vals = sorted_df.head(5)[col].tolist()
        print(f"  Top {top_n} for {col}: cutoff={cutoff_str}")
        print(f"    Top 5: {[(v, round(val, 4)) for v, val in zip(top5_vars, top5_vals)]}")
    
    # Count how many top-N lists each variable appears in
    df['ListCount'] = 0      # Number of measures where variable is in top N
    df['InMeasures'] = ''    # Which measures (abbreviated)
    
    measure_names = list(top_n_sets.keys())
    
    for idx, row in df.iterrows():
        var_name = row['Variable']
        in_measures = []
        # Check each measure's top-N set
        for measure_name in measure_names:
            if var_name in top_n_sets[measure_name]:
                in_measures.append(measure_name[:3])  # Abbreviate to 3 chars
        df.at[idx, 'ListCount'] = len(in_measures)
        df.at[idx, 'InMeasures'] = ','.join(in_measures)
    
    # Print overlap analysis between measures
    print(f"\n  Measure overlap analysis:")
    for i, m1 in enumerate(measure_names):
        for m2 in measure_names[i+1:]:
            overlap = len(top_n_sets[m1] & top_n_sets[m2])
            print(f"    {m1[:3]} & {m2[:3]}: {overlap} common variables")
    
    # Print cumulative distribution of list counts
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
# SECTION 13: EBM INTERACTION DISCOVERY FUNCTIONS
# =============================================================================
# EBM (Explainable Boosting Machine) is a glass-box ML model that automatically
# discovers feature interactions during training. These functions wrap EBM
# training and extract the discovered interactions.

def train_ebm_for_discovery(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    max_interactions: int = 20
) -> Optional[EBMReport]:
    """
    Train an EBM model to discover important feature interactions.
    
    PURPOSE:
    EBM automatically detects pairwise interactions between features.
    This is valuable because interactions can significantly improve model
    performance but are often missed by traditional selection methods.
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset to analyze
    - target_col (str): Name of the binary target variable
    - feature_cols (List[str]): List of feature names to include
    - max_interactions (int): Maximum number of interactions to detect (default 20)
    
    EBM CONFIGURATION:
    - max_bins=32: Number of bins for continuous features
    - interactions=max_interactions: Number of interaction terms to discover
    - max_interaction_bins=16: Bins for interaction visualization
    - outer_bags=8, inner_bags=4: Bagging for stability
    
    RETURNS:
    Optional[EBMReport]: Container with importances and interactions, or None if failed
    """
    # Check if EBM is available
    if not EBM_AVAILABLE or ExplainableBoostingClassifier is None:
        print("EBM not available - skipping interaction discovery")
        return None
    
    # Prepare feature matrix and target vector
    X = df[feature_cols].copy()  # Feature matrix
    y = df[target_col].copy()    # Target vector
    
    # Handle missing values by filling with median
    # EBM can't handle NaN values directly
    X = X.fillna(X.median())
    
    # Create and configure EBM classifier
    ebm = ExplainableBoostingClassifier(
        max_bins=32,              # Number of bins for main effects
        interactions=max_interactions,  # Number of interactions to discover
        max_interaction_bins=16,  # Bins for interaction terms
        outer_bags=8,             # Number of outer bagging iterations
        inner_bags=4,             # Number of inner bagging iterations
        random_state=42           # For reproducibility
    )
    
    # Train the EBM model
    ebm.fit(X, y)
    
    # Extract term names and importances from the trained model
    # EBM uses term_importances() method (not feature_importances_ attribute)
    term_names = ebm.term_names_              # List of term names
    term_importances_vals = ebm.term_importances()  # Importance scores
    
    # Separate main effects from interactions
    # Main effects don't contain ' x ' in their name
    feature_importance_list = []
    for i, name in enumerate(term_names):
        if ' x ' not in name:  # This is a main effect (single feature)
            feature_importance_list.append({
                'Variable': name,
                'EBM_Importance': term_importances_vals[i]
            })
    
    # Create DataFrame and sort by importance
    importances = pd.DataFrame(feature_importance_list)
    if not importances.empty:
        importances = importances.sort_values('EBM_Importance', ascending=False)
    
    # Extract interaction terms (contain ' x ' in name)
    interactions_list = []
    for i, name in enumerate(term_names):
        if ' x ' in name:  # This is an interaction term
            # Split to get the two variables
            vars_in_interaction = name.split(' x ')
            if len(vars_in_interaction) == 2:
                interactions_list.append({
                    'Variable_1': vars_in_interaction[0],
                    'Variable_2': vars_in_interaction[1],
                    'Interaction_Name': name,
                    'Magnitude': term_importances_vals[i]  # Importance of interaction
                })
    
    # Create interactions DataFrame and sort by magnitude
    interactions_df = pd.DataFrame(interactions_list)
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values('Magnitude', ascending=False)
    
    # Return EBMReport container
    return EBMReport(
        feature_importances=importances,
        interactions=interactions_df,
        missed_by_traditional=[],  # Will be filled by compare_selections later
        ebm_model=ebm
    )


def compare_selections(
    traditional_vars: List[str],
    ebm_importances: pd.DataFrame,
    top_n: int = 50
) -> List[str]:
    """
    Find variables important in EBM but missed by traditional selection.
    
    PURPOSE:
    EBM may find variables that traditional statistical measures missed.
    These "missed" variables are candidates to add back to the selection.
    
    PARAMETERS:
    - traditional_vars (List[str]): Variables selected by traditional methods
    - ebm_importances (pd.DataFrame): EBM feature importances
    - top_n (int): Number of top EBM features to consider (default 50)
    
    RETURNS:
    List[str]: Variables in EBM top N but not in traditional selection
    """
    # Get top N variables from EBM by importance
    ebm_top = ebm_importances.nlargest(top_n, 'EBM_Importance')['Variable'].tolist()
    
    # Normalize variable names (handle WOE_ prefix)
    # Traditional selection might use "Age" while EBM uses "WOE_Age"
    traditional_base = [v.replace('WOE_', '') for v in traditional_vars]
    
    # Find variables in EBM top that aren't in traditional selection
    missed = []
    for var in ebm_top:
        base_var = var.replace('WOE_', '')
        # Check if variable (with or without prefix) is in traditional
        if base_var not in traditional_base and var not in traditional_vars:
            missed.append(var)
    
    return missed


# =============================================================================
# SECTION 14: XGBOOST GPU FEATURE DISCOVERY
# =============================================================================
# XGBoost provides robust feature importance through gradient boosting.
# These functions train XGBoost models (optionally on GPU) and extract
# feature importances and potential interactions.

def train_xgboost_on_single_gpu(
    X: pd.DataFrame,
    y: pd.Series,
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
    """
    Train XGBoost on a specific GPU.
    
    PURPOSE:
    Trains a single XGBoost model on a specified GPU (or CPU if GPU unavailable).
    This function is called by train_xgboost_for_discovery, possibly in parallel.
    
    CRITICAL: Creates its own DMatrix to avoid GPU device conflicts.
    Each thread creates a DMatrix on its assigned GPU.
    
    PARAMETERS:
    - X (pd.DataFrame): Feature data
    - y (pd.Series): Target data
    - feature_cols (List[str]): List of feature column names
    - n_estimators (int): Number of boosting rounds
    - max_depth (int): Maximum tree depth
    - learning_rate (float): Learning rate (eta)
    - colsample_bytree (float): Fraction of columns to sample per tree
    - subsample (float): Fraction of rows to sample per tree
    - reg_alpha (float): L1 regularization term
    - reg_lambda (float): L2 regularization term
    - gpu_id (int): GPU device ID to use
    - seed (int): Random seed for reproducibility
    
    RETURNS:
    Tuple[model, importance_df, gpu_used]: Trained model, importance DataFrame, GPU flag
    """
    import os
    
    # Isolate this thread to a single GPU using CUDA_VISIBLE_DEVICES
    # This prevents GPU memory conflicts between parallel threads
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if XGBOOST_GPU_AVAILABLE and gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # Create DMatrix on this thread's assigned GPU
        # DMatrix is XGBoost's internal data structure
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        
        # Configure XGBoost parameters
        params = {
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'auc',             # Use AUC for evaluation
            'max_depth': max_depth,           # Tree depth limit
            'learning_rate': learning_rate,   # Step size shrinkage
            'colsample_bytree': colsample_bytree,  # Column sampling
            'subsample': subsample,           # Row sampling
            'min_child_weight': 3,            # Minimum sum of weights in child
            'reg_alpha': reg_alpha,           # L1 regularization
            'reg_lambda': reg_lambda,         # L2 regularization
            'verbosity': 0,                   # Suppress output
            'seed': seed                      # Random seed
        }
        
        gpu_used = False
        if XGBOOST_GPU_AVAILABLE:
            # Use cuda:0 because CUDA_VISIBLE_DEVICES remaps the GPU
            params['device'] = 'cuda:0'
            params['tree_method'] = 'hist'  # GPU-accelerated histogram method
            gpu_used = True
        else:
            params['tree_method'] = 'hist'  # CPU histogram method
        
        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, verbose_eval=False)
        
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Extract feature importances using different metrics
    # Gain: Average gain of splits using this feature
    importance_gain = model.get_score(importance_type='gain')
    # Cover: Average number of samples affected by splits on this feature
    importance_cover = model.get_score(importance_type='cover')
    # Weight: Number of times feature is used for splits
    importance_weight = model.get_score(importance_type='weight')
    
    # Build importance DataFrame
    importance_data = []
    for feat in feature_cols:
        importance_data.append({
            'Variable': feat,
            'XGB_Gain': importance_gain.get(feat, 0),
            'XGB_Cover': importance_cover.get(feat, 0),
            'XGB_Weight': importance_weight.get(feat, 0),
            'XGB_Importance': importance_gain.get(feat, 0)  # Use gain as primary
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
    Train XGBoost models on multiple GPUs in parallel for robust feature discovery.
    
    PURPOSE:
    XGBoost provides stable feature importance rankings through gradient boosting.
    Training on multiple GPUs with different seeds and averaging provides
    more robust importance estimates.
    
    OPTIMIZED SETTINGS for high-quality feature selection:
    - n_estimators=3000: More rounds = more stable importance
    - max_depth=8: Deeper trees for better feature discrimination
    - learning_rate=0.01: Lower rate = more reliable rankings
    - colsample_bytree=0.5: 50% column sampling for regularization
    - reg_alpha=0.5, reg_lambda=2.0: Regularization to reduce noise
    
    PARAMETERS:
    - df (pd.DataFrame): Input dataset
    - target_col (str): Name of target column
    - feature_cols (List[str]): Feature column names
    - use_gpu (bool): Whether to use GPU acceleration
    - n_estimators (int): Number of boosting rounds
    - max_depth (int): Maximum tree depth
    - learning_rate (float): Learning rate
    - discover_interactions (bool): Whether to discover interactions from trees
    - top_interactions (int): Number of top interactions to return
    - colsample_bytree (float): Column sampling fraction
    - subsample (float): Row sampling fraction
    - reg_alpha (float): L1 regularization
    - reg_lambda (float): L2 regularization
    - num_gpus (int): Number of GPUs for parallel training
    
    RETURNS:
    Optional[XGBoostReport]: Feature importances and interactions, or None if failed
    """
    # Check if XGBoost is available
    if not XGBOOST_AVAILABLE or xgb is None:
        print("XGBoost not available - skipping XGBoost discovery")
        return None
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Prepare feature matrix and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values with median imputation
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
            
            # Use ThreadPoolExecutor for parallel GPU training
            with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                futures = {}
                for gpu_id in range(actual_gpus):
                    # Use different seed per GPU for diversity
                    seed = 42 + gpu_id * 1000
                    futures[gpu_id] = executor.submit(
                        train_xgboost_on_single_gpu,
                        X, y, feature_cols, n_estimators, max_depth,
                        learning_rate, colsample_bytree, subsample,
                        reg_alpha, reg_lambda, gpu_id, seed
                    )
                
                # Collect results from all GPUs
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
            
            # Average feature importances across all models
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
                X, y, feature_cols, n_estimators, max_depth,
                learning_rate, colsample_bytree, subsample,
                reg_alpha, reg_lambda, gpu_id=0, seed=42
            )
        
        # Sort by importance and add normalized scores
        feature_importances = feature_importances.sort_values('XGB_Importance', ascending=False)
        max_imp = feature_importances['XGB_Importance'].max()
        if max_imp > 0:
            # Normalize to [0, 1] range
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
            missed_by_traditional=[],  # Filled later by compare_xgb_selections
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
    
    PURPOSE:
    Analyzes the tree structure to find features that frequently appear
    together in parent-child relationships. When feature A is the parent
    and feature B is the child in a tree node, it suggests an interaction.
    
    PARAMETERS:
    - model (Any): Trained XGBoost booster
    - feature_names (List[str]): List of feature names
    - top_n (int): Number of top interactions to return
    
    HOW IT WORKS:
    1. Convert all trees to a DataFrame format
    2. For each tree, build a mapping of node ID -> feature
    3. Find parent-child pairs where both are features (not leaves)
    4. Count how often each feature pair appears together
    5. Return top N pairs by count
    
    RETURNS:
    List[dict]: List of interaction dictionaries with Variable_1, Variable_2, etc.
    """
    if xgb is None:
        return []
    
    try:
        # Convert all trees to DataFrame format
        trees_df = model.trees_to_dataframe()
        
        if trees_df.empty:
            return []
        
        # Count parent-child feature pairs across all trees
        interaction_counts = {}
        
        # Process each tree separately
        for tree_id in trees_df['Tree'].unique():
            tree_data = trees_df[trees_df['Tree'] == tree_id]
            
            # Build mapping from node ID to feature name
            node_features = {}
            for _, node in tree_data.iterrows():
                # Only include internal nodes (not leaves)
                if pd.notna(node['Feature']) and node['Feature'] != 'Leaf':
                    node_features[node['ID']] = node['Feature']
            
            # Find parent-child feature pairs
            for _, node in tree_data.iterrows():
                if pd.notna(node['Feature']) and node['Feature'] != 'Leaf':
                    parent_feat = node['Feature']
                    
                    # Check left child (Yes branch)
                    if pd.notna(node.get('Yes')):
                        left_id = node['Yes']
                        if left_id in node_features:
                            child_feat = node_features[left_id]
                            if parent_feat != child_feat:
                                # Create sorted pair to avoid duplicates
                                pair = tuple(sorted([parent_feat, child_feat]))
                                interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
                    
                    # Check right child (No branch)
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
        
        # Format as list of dictionaries
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
    
    PURPOSE:
    Uses two filters to identify missed variables while reducing noise:
    1. Top N filter: Only consider top N features
    2. Importance threshold: Only features above minimum importance
    
    PARAMETERS:
    - traditional_vars (List[str]): Variables from traditional selection
    - xgb_importances (pd.DataFrame): XGBoost feature importances
    - top_n (int): Maximum features to consider (default 25)
    - min_importance_threshold (float): Minimum normalized importance (default 5%)
    
    RETURNS:
    List[str]: Variables missed by traditional selection
    """
    if xgb_importances.empty or 'XGB_Importance' not in xgb_importances.columns:
        return []
    
    # Calculate normalized importance if not present
    if 'XGB_Importance_Normalized' not in xgb_importances.columns:
        max_imp = xgb_importances['XGB_Importance'].max()
        if max_imp > 0:
            xgb_importances = xgb_importances.copy()
            xgb_importances['XGB_Importance_Normalized'] = xgb_importances['XGB_Importance'] / max_imp
        else:
            return []
    
    # Filter by minimum importance threshold first
    filtered = xgb_importances[xgb_importances['XGB_Importance_Normalized'] >= min_importance_threshold]
    
    # Then take top N from filtered set
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


# =============================================================================
# SECTION 15: UTILITY FUNCTIONS
# =============================================================================
# Helper functions for creating interaction columns, calculating correlations,
# computing VIF, and other utility operations.

def create_interaction_columns(
    df: pd.DataFrame,
    interactions: pd.DataFrame,
    top_n: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create interaction term columns in the DataFrame.
    
    PURPOSE:
    Interactions detected by EBM/XGBoost are represented as new columns
    computed by multiplying the two interacting variables together.
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset to add interaction columns to
    - interactions (pd.DataFrame): DataFrame with Variable_1, Variable_2 columns
    - top_n (int): Number of top interactions to create (default 10)
    
    INTERACTION FORMULA:
    new_col = Variable_1 * Variable_2
    
    COLUMN NAMING:
    "Variable1_x_Variable2" (e.g., "WOE_Age_x_WOE_Income")
    
    RETURNS:
    Tuple[pd.DataFrame, List[str]]: Modified DataFrame and list of new column names
    """
    result_df = df.copy()
    new_cols = []
    
    # Return unchanged if no interactions
    if interactions.empty:
        return result_df, new_cols
    
    # Take only top N interactions
    top_interactions = interactions.head(top_n)
    
    # Create each interaction column
    for _, row in top_interactions.iterrows():
        var1 = row['Variable_1']
        var2 = row['Variable_2']
        
        # Check if both variables exist in DataFrame
        if var1 in df.columns and var2 in df.columns:
            # Create interaction column name
            new_col_name = f"{var1}_x_{var2}"
            # Calculate interaction as product
            result_df[new_col_name] = df[var1] * df[var2]
            new_cols.append(new_col_name)
    
    return result_df, new_cols


def calculate_correlation_matrix(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for selected columns.
    
    PURPOSE:
    Correlation matrix helps identify redundant variables (high correlation)
    and understand relationships between selected features.
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset
    - cols (List[str]): List of column names to include
    
    RETURNS:
    pd.DataFrame: Square correlation matrix (cols x cols)
    """
    # Filter to only columns that exist in the DataFrame
    available_cols = [c for c in cols if c in df.columns]
    
    # Need at least 2 columns for correlation
    if len(available_cols) < 2:
        return pd.DataFrame()
    
    # Calculate Pearson correlation and round to 4 decimals
    return df[available_cols].corr().round(4)


def calculate_vif(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    PURPOSE:
    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity. High VIF indicates redundant information.
    
    VIF INTERPRETATION:
    - VIF = 1: No multicollinearity
    - VIF = 1-5: Low multicollinearity (OK)
    - VIF = 5-10: Moderate multicollinearity (consider removing)
    - VIF > 10: High multicollinearity (should remove)
    - VIF = 999.99: Perfect multicollinearity (can be perfectly predicted)
    
    FORMULA:
    VIF = 1 / (1 - R²) where R² is from regressing the variable on all others
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset
    - cols (List[str]): List of numeric column names to check
    
    RETURNS:
    pd.DataFrame: VIF report with columns: Variable, VIF, R_Squared, Status, Reason
    """
    from sklearn.linear_model import LinearRegression
    
    # Filter to numeric columns that exist
    available_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(available_cols) < 2:
        return pd.DataFrame({'Variable': [], 'VIF': [], 'Status': [], 'Reason': []})
    
    # Prepare data - fill NaN with median
    X = df[available_cols].copy()
    X = X.fillna(X.median())
    
    # Identify constant columns (zero variance)
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
    
    # Use non-constant columns for VIF calculation
    unique_cols = non_constant_cols
    duplicate_cols = []  # Disabled - rely on VIF=999.99 detection instead
    
    vif_data = []
    
    # Add constant columns with special status
    for col in constant_cols:
        vif_data.append({
            'Variable': col,
            'VIF': 999.99,
            'R_Squared': 1.0,
            'Status': 'CONSTANT - Remove',
            'Reason': 'Column has no variance'
        })
    
    # Add duplicate columns with special status
    for dup, orig in duplicate_cols:
        vif_data.append({
            'Variable': dup,
            'VIF': 999.99,
            'R_Squared': 1.0,
            'Status': 'DUPLICATE - Remove',
            'Reason': f'Identical to {orig}'
        })
    
    # Calculate VIF for remaining columns
    for i, col in enumerate(unique_cols):
        # Get all other columns as predictors
        other_cols = [c for c in unique_cols if c != col]
        
        if len(other_cols) == 0:
            vif_data.append({'Variable': col, 'VIF': 1.0, 'R_Squared': 0.0, 'Status': 'OK', 'Reason': ''})
            continue
        
        try:
            # Fit regression of this column on all others
            X_others = X[other_cols].values
            y = X[col].values
            
            model = LinearRegression()
            model.fit(X_others, y)
            
            # Calculate R-squared
            r_squared = model.score(X_others, y)
            
            # VIF = 1 / (1 - R²)
            if r_squared >= 1.0:
                vif = 999.99  # Perfect multicollinearity
            else:
                vif = 1 / (1 - r_squared)
            
            # Cap VIF at 999.99 for display
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
    
    # Create DataFrame and sort by VIF descending
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
    
    PURPOSE:
    Since removing one high VIF variable changes the VIF of others,
    we must remove variables one at a time and recalculate.
    
    ALGORITHM:
    1. Remove all constant columns in batch (no variance)
    2. Calculate VIF for remaining columns
    3. If highest VIF >= threshold, remove that variable
    4. Repeat steps 2-3 until no variable exceeds threshold
    
    PARAMETERS:
    - df (pd.DataFrame): The dataset
    - cols (List[str]): Columns to check
    - vif_threshold (float): Remove variables with VIF >= this (default 11.0)
    - max_iterations (int): Maximum removal iterations (default 100)
    
    RETURNS:
    Tuple containing:
    - remaining_cols: Columns after removal
    - final_vif: Final VIF DataFrame
    - removed_cols: List of removed column names
    - removed_vif_info: List of dicts with removal details
    """
    remaining_cols = [c for c in cols if c in df.columns]
    removed_cols = []
    removed_vif_info = []
    
    # First pass: Remove constant columns in batch
    vif_df = calculate_vif(df, remaining_cols)
    
    if not vif_df.empty and 'Status' in vif_df.columns:
        # Find constant columns for batch removal
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
        
        # Find highest VIF (excluding constant columns)
        valid_vif = vif_df[vif_df['Status'] != 'CONSTANT - Remove']
        if valid_vif.empty:
            break
        
        max_vif_row = valid_vif.iloc[0]  # Already sorted descending
        max_vif = max_vif_row['VIF']
        max_vif_var = max_vif_row['Variable']
        
        # Stop if highest VIF is below threshold
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
    """
    Add rank columns for each measure to help with variable comparison.
    
    PURPOSE:
    Rankings allow easy comparison of variables across different measures.
    Also calculates average rank and rank agreement (standard deviation).
    
    PARAMETERS:
    - measures_df (pd.DataFrame): DataFrame with calculated measures
    
    ADDED COLUMNS:
    - Entropy_Rank, IV_Rank, OR_Rank, LR_Rank, ChiSq_Rank, Gini_Rank: Individual ranks
    - Avg_Rank: Average of all ranks
    - Rank_Agreement: Standard deviation of ranks (lower = more agreement)
    
    RETURNS:
    pd.DataFrame: Input DataFrame with added rank columns
    """
    df = measures_df.copy()
    
    # Define measure columns and their rank configurations
    # (rank_column_name, ascending): ascending=False means higher is better
    measure_rank_configs = {
        'Entropy': ('Entropy_Rank', False),  # Higher entropy explained = better
        'Information Value': ('IV_Rank', False),
        'Odds Ratio': ('OR_Rank', False),
        'Likelihood Ratio': ('LR_Rank', False),
        'Chi-Square': ('ChiSq_Rank', False),
        'Gini': ('Gini_Rank', False)
    }
    
    for col, (rank_col, ascending) in measure_rank_configs.items():
        if col in df.columns:
            # Calculate rank (1 = best)
            # na_option='bottom' puts NaN values at the worst rank
            df[rank_col] = df[col].rank(ascending=ascending, na_option='bottom')
    
    # Calculate average rank across all measures
    rank_cols = [c for c in df.columns if c.endswith('_Rank')]
    if rank_cols:
        df['Avg_Rank'] = df[rank_cols].mean(axis=1).round(2)
        # Rank agreement: lower std dev = more consistent ranking
        df['Rank_Agreement'] = df[rank_cols].std(axis=1).round(2)
    
    return df


def add_ebm_importance_to_measures(
    measures_df: pd.DataFrame,
    ebm_importances: pd.DataFrame
) -> pd.DataFrame:
    """
    Add EBM importance scores to the measures DataFrame.
    
    PURPOSE:
    Enriches the measures table with EBM importance so users can compare
    traditional measures with ML-based importance.
    
    PARAMETERS:
    - measures_df (pd.DataFrame): DataFrame with calculated measures
    - ebm_importances (pd.DataFrame): EBM feature importances
    
    ADDED COLUMNS:
    - EBM_Importance: Raw EBM importance score
    - EBM_Rank: Rank by EBM importance
    - Rank_Diff: Absolute difference between EBM rank and average traditional rank
    - EBM_Disagrees: Flag if rank difference > 20
    
    RETURNS:
    pd.DataFrame: Input DataFrame with EBM columns added
    """
    df = measures_df.copy()
    
    if ebm_importances.empty:
        return df
    
    # Create mapping from variable name to EBM importance
    ebm_map = {}
    for _, row in ebm_importances.iterrows():
        var = row['Variable']
        importance = row['EBM_Importance']
        
        # Map both with and without WOE_ prefix for flexibility
        ebm_map[var] = importance
        if var.startswith('WOE_'):
            ebm_map[var[4:]] = importance  # Remove WOE_ prefix
        else:
            ebm_map[f'WOE_{var}'] = importance  # Add WOE_ prefix
    
    # Add EBM columns
    df['EBM_Importance'] = df['Variable'].map(ebm_map)
    df['EBM_Rank'] = df['EBM_Importance'].rank(ascending=False, na_option='bottom')
    
    # Flag significant disagreements between EBM and traditional selection
    if 'Avg_Rank' in df.columns:
        df['Rank_Diff'] = abs(df['EBM_Rank'] - df['Avg_Rank'])
        df['EBM_Disagrees'] = df['Rank_Diff'] > 20  # Flag if ranks differ by more than 20
    
    return df


# =============================================================================
# SECTION 16: SHINY UI APPLICATION
# =============================================================================
# This section defines the interactive web-based user interface using Shiny.
# The UI allows users to configure variable selection parameters, run analyses,
# and view results before generating outputs.

def create_variable_selection_app(
    df: pd.DataFrame,
    min_prop: float = 0.01
):
    """
    Create the Variable Selection Shiny application.
    
    PURPOSE:
    Builds and returns a Shiny app for interactive variable selection.
    Users can select measures, configure criteria, run EBM, and submit results.
    
    PARAMETERS:
    - df (pd.DataFrame): The input dataset with WOE columns
    - min_prop (float): Minimum proportion per bin (default 1%)
    
    RETURNS:
    App: Configured Shiny application object with results dictionary attached
    """
    
    # Find binary target variable candidates
    # Binary vars have exactly 2 unique values and don't start with WOE_ or b_
    binary_vars = [col for col in df.columns
                   if df[col].nunique() == 2 and not col.startswith(('WOE_', 'b_'))]
    
    # Get WOE columns (columns starting with WOE_)
    woe_cols = [col for col in df.columns if col.startswith('WOE_')]
    
    # Dictionary to store results from the UI session
    # This is modified by the server function and accessed after app closes
    app_results = {
        'measures': None,           # Calculated measures DataFrame
        'selected_data': None,      # Selected data for output
        'ebm_report': None,         # EBM interaction report
        'correlation_matrix': None, # Correlation matrix
        'vif_report': None,         # VIF report
        'removed_for_vif': [],      # Variables removed for high VIF
        'completed': False          # Flag indicating successful submission
    }
    
    # ==========================================================================
    # UI DEFINITION
    # ==========================================================================
    # The UI is defined using Shiny's page_fluid layout with custom CSS styling
    
    app_ui = ui.page_fluid(
        # Custom CSS for dark theme with gradient background
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
        
        # Page title
        ui.h4("🎯 Variable Selection with EBM Interaction Discovery"),
        
        # Configuration Section - DV selection, measures, analyze button
        ui.div(
            {"class": "card"},
            ui.row(
                ui.column(4,
                    # Dropdown to select dependent variable
                    ui.input_select("dv", "Dependent Variable",
                                   choices=binary_vars,
                                   selected=binary_vars[0] if binary_vars else None)
                ),
                ui.column(4,
                    # Checkboxes to select which measures to calculate
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
                    # Button to trigger analysis
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
                    # Number of top variables to consider per measure
                    ui.input_numeric("num_vars", "Number of Variables",
                                    value=50, min=1, max=500)
                ),
                ui.column(3,
                    # Union vs Intersection selection
                    ui.input_select("criteria", "Criteria",
                                   choices={'Union': 'Union (ANY measure)',
                                           'Intersection': 'Intersection (MULTIPLE measures)'},
                                   selected='Intersection')
                ),
                ui.column(3,
                    # Degree parameter for Intersection criteria
                    ui.input_numeric("degree", "Degree (for Intersection)",
                                    value=2, min=1, max=6)
                ),
                ui.column(3,
                    # Button to apply selection criteria
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
        
        # Results Tables - Measures table
        ui.row(
            ui.column(12,
                ui.div(
                    {"class": "card", "style": "max-height: 400px; overflow-y: auto;"},
                    ui.h5("📈 Predictive Measures"),
                    ui.output_data_frame("measures_table")
                )
            )
        ),
        
        # Interactions and Missed Variables tables side by side
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
        
        # Visualization charts
        ui.row(
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    output_widget("importance_chart")  # Plotly chart
                )
            ),
            ui.column(6,
                ui.div(
                    {"class": "card", "style": "height: 400px;"},
                    output_widget("interaction_chart")  # Plotly chart
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
        
        # Summary Statistics
        ui.div(
            {"class": "card"},
            ui.output_ui("summary_stats")
        ),
        
        # Submit Button - generates output and closes the app
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("submit_btn", "🚀 Generate Output & Close",
                                  class_="btn btn-success btn-lg"),
        ),
    )
    
    # ==========================================================================
    # SERVER FUNCTION
    # ==========================================================================
    # The server function contains all the reactive logic for the UI
    
    def server(input: Inputs, output: Outputs, session: Session):
        # Reactive values to store intermediate results
        measures_rv = reactive.Value(pd.DataFrame())      # Calculated measures
        ebm_report_rv = reactive.Value(None)              # EBM report object
        selected_vars_rv = reactive.Value([])             # Selected variable names
        interaction_cols_rv = reactive.Value([])          # Created interaction columns
        missed_vars_to_add_rv = reactive.Value([])        # Missed vars to auto-add
        vif_rv = reactive.Value(pd.DataFrame())           # VIF results
        
        # ------------------------------------------------------------------
        # ANALYZE BUTTON - Calculate predictive measures
        # ------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.analyze_btn)
        def analyze():
            dv = input.dv()  # Get selected dependent variable
            if not input.measures() or not dv:
                return
            
            # Calculate bins internally (equivalent to R's logiBin::getBins)
            iv_list = [col for col in df.columns if col != dv]
            bin_result = get_bins(df, dv, iv_list, min_prop=min_prop)
            bins_df_calc = bin_result.bin
            var_summary_calc = bin_result.var_summary
            
            # Calculate selected measures
            measures = calculate_all_measures(
                bins_df_calc,
                var_summary_calc,
                list(input.measures())
            )
            measures_rv.set(measures)  # Update reactive value
        
        # ------------------------------------------------------------------
        # SELECT BUTTON - Apply selection criteria
        # ------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.select_btn)
        def select_variables():
            measures = measures_rv.get()
            if measures.empty:
                return
            
            criteria = input.criteria()
            num_vars = input.num_vars()
            degree = input.degree() if criteria == 'Intersection' else 1
            
            # Apply filtering criteria
            filtered = filter_variables(measures, criteria, num_vars, degree)
            measures_rv.set(filtered)
            
            # Extract selected variable names
            selected = filtered[filtered['Selected'] == True]['Variable'].tolist()
            selected_vars_rv.set(selected)
        
        # ------------------------------------------------------------------
        # EBM BUTTON - Run EBM interaction discovery
        # ------------------------------------------------------------------
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
                        missed_to_add = missed  # 0 means add ALL
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
        
        # ------------------------------------------------------------------
        # OUTPUT RENDERERS
        # ------------------------------------------------------------------
        
        @output
        @render.data_frame
        def measures_table():
            """Render the measures DataGrid."""
            measures = measures_rv.get()
            if measures.empty:
                return render.DataGrid(pd.DataFrame({'Message': ['Click "Analyze" to calculate measures']}))
            return render.DataGrid(measures, selection_mode="rows", height="350px")
        
        @output
        @render.data_frame
        def interactions_table():
            """Render the interactions DataGrid."""
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
            """Render the missed variables DataGrid."""
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
            """Render the EBM importance bar chart."""
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
            """Render the interaction magnitude bar chart."""
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
            """Render the VIF DataGrid."""
            vif = vif_rv.get()
            if vif.empty:
                return render.DataGrid(pd.DataFrame({'Message': ['Run EBM to calculate VIF']}))
            return render.DataGrid(vif, height="250px")
        
        @output
        @render.ui
        def summary_stats():
            """Render the summary statistics panel."""
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
        
        # ------------------------------------------------------------------
        # SUBMIT BUTTON - Generate output and close app
        # ------------------------------------------------------------------
        @reactive.Effect
        @reactive.event(input.submit_btn)
        async def submit():
            """Handle submit button - prepare outputs and close session."""
            dv = input.dv()
            measures = measures_rv.get()
            selected = selected_vars_rv.get()
            missed_to_add = missed_vars_to_add_rv.get()
            report = ebm_report_rv.get()
            vif = vif_rv.get()
            
            # Prepare output columns starting with DV
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
                
                # Add missed variables section
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
            
            # Iteratively remove high VIF variables (default: no removal, threshold=0)
            remaining_cols, final_vif, removed_cols, _ = remove_high_vif_iteratively(
                output_df, numeric_cols, vif_threshold=0.0
            )
            
            if removed_cols:
                final_cols = [dv] + remaining_cols
                output_df = output_df[final_cols].copy()
                
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
            
            # Store results in app_results dictionary
            app_results['measures'] = measures
            app_results['selected_data'] = output_df
            app_results['ebm_report'] = ebm_report_df
            app_results['correlation_matrix'] = corr_matrix
            app_results['vif_report'] = final_vif
            app_results['removed_for_vif'] = removed_cols
            app_results['completed'] = True
            
            # Close the session (ends the Shiny app)
            await session.close()
    
    # Create and return the Shiny app
    app = App(app_ui, server)
    app.results = app_results  # Attach results dict to app for external access
    return app


def find_free_port(start_port: int = 8052, max_attempts: int = 50) -> int:
    """
    Find an available port starting from start_port.
    
    PURPOSE:
    When running multiple instances of the Shiny app, ports may be occupied.
    This function finds an available port to bind to.
    
    PARAMETERS:
    - start_port (int): Starting port number (default 8052)
    - max_attempts (int): Maximum number of ports to try (default 50)
    
    RETURNS:
    int: An available port number
    """
    import socket
    
    for offset in range(max_attempts):
        # Pick a random port in the range
        port = start_port + random.randint(0, RANDOM_PORT_RANGE)
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port  # Port is available
        except OSError:
            continue  # Port in use, try another
    
    # Fallback: let OS assign a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_variable_selection(
    df: pd.DataFrame,
    port: int = None
):
    """
    Run the Variable Selection application and return results.
    
    PURPOSE:
    Entry point for interactive mode. Creates the Shiny app, finds a port,
    runs the app, and returns the results after the user closes it.
    
    PARAMETERS:
    - df (pd.DataFrame): The input dataset
    - port (int, optional): Specific port to use (auto-detected if None)
    
    RETURNS:
    dict: Results dictionary from the app (measures, selected_data, etc.)
    """
    # Find a free port if not specified
    if port is None:
        port = find_free_port(BASE_PORT)
    
    print(f"Starting Shiny app on port {port}")
    sys.stdout.flush()
    
    # Create the Shiny app
    app = create_variable_selection_app(df)
    
    try:
        # Run the app (blocks until user closes it)
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

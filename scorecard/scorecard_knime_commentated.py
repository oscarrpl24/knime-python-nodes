# =============================================================================
# Scorecard Generator for KNIME Python Script Node
# =============================================================================
# This is the COMMENTATED VERSION of scorecard_knime.py
# Every line of code is explained in plain English for educational purposes.
# =============================================================================
#
# WHAT IS A SCORECARD?
# --------------------
# A scorecard is a points-based system used in credit risk modeling to convert
# a logistic regression model's predictions into an easy-to-understand score.
# Instead of dealing with probabilities and log-odds, users see simple integer
# points that sum up to a final credit score (like FICO scores).
#
# SCORECARD FORMULA EXPLAINED:
# ----------------------------
# The core formula converts logistic regression coefficients to points:
#
#   b = PDO / log(2)
#       - PDO = "Points to Double the Odds" (e.g., 50 points)
#       - This determines how many points it takes to double the odds of default
#       - log(2) ≈ 0.693, so b ≈ 72.13 when PDO=50
#
#   a = Points + b * log(1/(Odds-1))
#       - Points = base score at target odds (e.g., 600)
#       - Odds = target odds ratio (e.g., 1:19 means 1 bad for every 19 goods)
#       - This anchors the score scale
#
#   basepoints = a - b * intercept
#       - Distributes the intercept term into a fixed base score
#
#   bin_points = round(-b * coefficient * WOE)
#       - Converts each bin's WOE value into points using its coefficient
#       - Negative sign: higher WOE (riskier) = lower points
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided, opens a
#    browser-based interface where users can configure parameters
# 2. Headless - When Points, Odds, and PDO are provided via KNIME flow
#    variables, runs automatically without user interaction
#
# INPUTS (from KNIME workflow):
# 1. Coefficients table from Logistic Regression node (Output 2)
#    - Row ID = variable name (e.g., "(Intercept)", "WOE_Age")
#    - Column "model$coefficients" = coefficient value
# 2. Bins table from WOE Editor node (Output 4)
#    - var, bin, binValue, woe columns required
#
# OUTPUTS:
# 1. Scorecard table - all bins with points (var, bin, woe, points columns)
#
# Flow Variables (for headless mode):
# - Points (int, default 600): Base score at target odds
# - Odds (int, default 20): Target odds ratio (1:Odds, e.g., 20 means 1:19)
# - PDO (int, default 50): Points to Double the Odds
#
# Release Date: 2026-01-19
# Version: 1.0
# =============================================================================

# =============================================================================
# IMPORT SECTION
# =============================================================================
# This section imports all the libraries needed for the script to work.

# Import the KNIME scripting interface - this is how we read data from KNIME
# and write results back. 'knio' is the standard abbreviation used in KNIME
# Python nodes. Without this, the script cannot interact with the KNIME workflow.
import knime.scripting.io as knio

# Import pandas - the main library for working with tabular data in Python.
# 'pd' is the conventional abbreviation. We use it to:
# - Read and manipulate DataFrames (tables)
# - Filter rows, select columns, group data
# - Handle missing values (NA/NaN)
import pandas as pd

# Import numpy - the fundamental library for numerical computing in Python.
# 'np' is the conventional abbreviation. We use it for:
# - Mathematical functions like log() and round()
# - Array operations
# - Handling NaN values
import numpy as np

# Import warnings module - allows us to control warning messages.
# Some operations generate warnings that aren't actually problems,
# so we can suppress them to keep the output clean.
import warnings

# Import type hints from the typing module - these don't affect runtime
# but help developers understand what types of data functions expect/return.
# Dict = dictionary (key-value pairs), List = list of items
# Tuple = fixed-length ordered collection, Optional = can be None
# Any = any type of data
from typing import Dict, List, Tuple, Optional, Any

# Suppress all warning messages. This keeps the KNIME console clean by hiding
# non-critical warnings from pandas, numpy, and other libraries.
# Note: In production, you might want to be more selective about which
# warnings to ignore, but for this script it's safe to ignore all.
warnings.filterwarnings('ignore')

# =============================================================================
# Install/Import Dependencies Section
# =============================================================================
# This section handles optional dependencies that might not be installed.
# The Shiny library is used for the interactive web UI, but it's not always
# available or needed (headless mode doesn't require it).

def install_if_missing(package, import_name=None):
    """
    Attempt to install a Python package if it's not already installed.
    
    This function tries to import a package, and if that fails (ImportError),
    it uses pip to install the package automatically. This is useful for
    KNIME environments where packages might not be pre-installed.
    
    Parameters:
        package: The pip package name to install (e.g., 'shiny')
        import_name: The Python import name if different from package name
                     (e.g., package='PIL' but import_name='pillow')
                     If None, uses the package name for both.
    
    WARNING: Auto-installing packages can be a security risk in some
    environments. In production, packages should be pre-installed.
    """
    # If no separate import name is provided, use the package name for both
    # pip installation and Python import (they're usually the same)
    if import_name is None:
        import_name = package
    
    # Try to import the package to check if it's already installed
    try:
        # __import__ is Python's built-in function to import by string name
        # This is equivalent to "import shiny" but allows dynamic names
        __import__(import_name)
        # If we get here, the import succeeded - package is installed
    except ImportError:
        # ImportError means the package is not installed
        # Use subprocess to run pip install command
        import subprocess
        # subprocess.check_call runs a command and raises error if it fails
        # ['pip', 'install', package] is equivalent to: pip install package
        subprocess.check_call(['pip', 'install', package])

# Try to install the 'shiny' package if not present
# Shiny is a web application framework that creates interactive UIs
# Originally from R, now available for Python too
install_if_missing('shiny')

# Try to install 'shinywidgets' if not present
# shinywidgets provides additional UI components for Shiny apps
install_if_missing('shinywidgets')

# Now try to actually import the Shiny components we need
# This is wrapped in try/except because even after installation,
# imports can sometimes fail due to environment issues
try:
    # Import specific components from the shiny package:
    # - App: The main application class that combines UI and server logic
    # - Inputs: Gives access to user input values from the UI
    # - Outputs: Allows setting output values to display in the UI
    # - Session: Represents a single user's browser session
    # - reactive: Decorators for reactive programming (automatic updates)
    # - render: Decorators for rendering outputs (tables, text, etc.)
    # - ui: Functions to build the user interface (buttons, inputs, etc.)
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    
    # Set flag to True indicating Shiny is available for use
    SHINY_AVAILABLE = True
    
except ImportError:
    # If import fails, print a warning and set flag to False
    # The script will still work but only in headless mode
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Scorecard Creation Functions
# =============================================================================
# These are the core mathematical functions that convert logistic regression
# coefficients and WOE bins into scorecard points.

def calculate_ab(points0: float = 600, odds0: float = 1/19, pdo: float = 50) -> Tuple[float, float]:
    """
    Calculate the 'a' and 'b' scaling parameters for the scorecard formula.
    
    These parameters translate the logistic regression's log-odds scale
    into a user-friendly point scale. The formulas are standard in credit
    scoring and come from solving two equations:
    
    1. At odds = odds0, score should equal points0
    2. When odds double, score should change by PDO points
    
    The mathematical derivation:
    - Score = a + b * log(odds)  [linear in log-odds]
    - At odds0: points0 = a + b * log(odds0)  ... equation 1
    - At 2*odds0: points0 + PDO = a + b * log(2*odds0)  ... equation 2
    
    Subtracting equation 1 from 2:
    - PDO = b * log(2*odds0) - b * log(odds0)
    - PDO = b * [log(2) + log(odds0) - log(odds0)]
    - PDO = b * log(2)
    - Therefore: b = PDO / log(2)
    
    Substituting back:
    - a = points0 - b * log(odds0)
    - But we use: a = points0 + b * log(odds0) because of sign convention
    
    Parameters:
        points0: The base score when odds equal the target odds (default 600)
                 This is the "anchor point" of the scorecard scale.
                 Common values: 600, 660, 700
        odds0: The target odds ratio as a decimal (default 1/19 ≈ 0.0526)
               1/19 means "1 bad outcome for every 19 good outcomes"
               Also expressed as "1:19 odds" or "20:1 goods to bads"
        pdo: Points to Double the Odds (default 50)
             How many points it takes to double the odds of being "good"
             Higher PDO = more spread in scores, finer granularity
             Common values: 20, 50, 100
        
    Returns:
        Tuple containing:
        - a (float): The intercept scaling parameter
        - b (float): The slope scaling parameter (coefficient multiplier)
        
    Example:
        With default values:
        - b = 50 / log(2) ≈ 72.13
        - a = 600 + 72.13 * log(1/19) ≈ 600 + 72.13 * (-2.944) ≈ 387.6
    """
    # Calculate b: the scaling factor for the slope
    # np.log() is the natural logarithm (base e)
    # np.log(2) ≈ 0.693, so if PDO=50, then b ≈ 72.13
    # This means each unit of log-odds converts to about 72 points
    b = pdo / np.log(2)
    
    # Calculate a: the intercept that anchors the scale
    # This ensures that at odds = odds0, the score equals points0
    # Note: We add (not subtract) because odds0 < 1 gives negative log
    a = points0 + b * np.log(odds0)
    
    # Return both values as a tuple for the caller to use
    return a, b


def is_interaction_term(var_name: str) -> bool:
    """
    Determine if a variable name represents an interaction term.
    
    In logistic regression, interaction terms capture how two variables
    jointly affect the outcome beyond their individual effects.
    For example, if Age and Income have an interaction, the effect of
    Age on default risk depends on the Income level (and vice versa).
    
    In WOE-based scorecards, interaction terms are created by multiplying
    the WOE values of two variables: WOE_interaction = WOE_var1 × WOE_var2
    
    Interaction term naming conventions:
    - "WOE_Age_x_WOE_Income" - both variables have WOE_ prefix
    - "Age_x_WOE_Income" - first variable might not have prefix
    - "Age_x_Income" - neither has prefix (less common)
    
    Parameters:
        var_name: The variable name to check (string)
        
    Returns:
        True if the name contains interaction pattern, False otherwise
        
    Examples:
        is_interaction_term("WOE_Age") -> False (regular variable)
        is_interaction_term("WOE_Age_x_WOE_Income") -> True (interaction)
        is_interaction_term("Age_x_Income") -> True (interaction)
    """
    # Check for the most specific pattern first: '_x_WOE_'
    # This appears in names like "WOE_Age_x_WOE_Income"
    # OR check for the more general pattern: '_x_'
    # This appears in names like "Age_x_Income"
    # The '_x_' separator is a common convention for denoting interactions
    return '_x_WOE_' in var_name or '_x_' in var_name


def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term name into its two component variable names.
    
    This function takes a combined interaction name and extracts the
    individual variable names that make up the interaction. It handles
    various naming formats that might come from the logistic regression.
    
    The function must handle these input formats:
    - "WOE_var1_x_WOE_var2" -> returns ("var1", "var2")
    - "var1_x_WOE_var2" -> returns ("var1", "var2")
    - "var1_x_var2" -> returns ("var1", "var2")
    
    Note: Variable names themselves might contain underscores, but we
    rely on the '_x_' or '_x_WOE_' patterns to identify the split point.
    
    Parameters:
        var_name: The full interaction term name (string)
        
    Returns:
        Tuple of (var1_name, var2_name) as strings
        
    Raises:
        ValueError: If the name doesn't match any known interaction pattern
        
    Example:
        parse_interaction_term("WOE_Age_x_WOE_Income")
        -> Returns ("Age", "Income")
    """
    # Start with the full name - we'll clean it up step by step
    clean_name = var_name
    
    # First, remove the leading 'WOE_' prefix if present
    # Many interaction terms start with "WOE_" from the logistic regression
    # but we need the base variable name to look up in the bins table
    if clean_name.startswith('WOE_'):
        # Slice the string starting at position 4 to remove 'WOE_'
        # 'WOE_Age' becomes 'Age'
        clean_name = clean_name[4:]
    
    # Try splitting on '_x_WOE_' first - this is the most specific pattern
    # This handles cases like "Age_x_WOE_Income" after WOE_ removal
    # (original: "WOE_Age_x_WOE_Income")
    if '_x_WOE_' in clean_name:
        # split('_x_WOE_', 1) splits only on the first occurrence
        # and returns at most 2 parts
        # "Age_x_WOE_Income" -> ["Age", "Income"]
        parts = clean_name.split('_x_WOE_', 1)
        # Return first part and second part as the two variable names
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_' - more general pattern
    # This handles cases like "Age_x_Income"
    if '_x_' in clean_name:
        # Split on '_x_' to get the two parts
        parts = clean_name.split('_x_', 1)
        var2 = parts[1]
        
        # The second variable might still have 'WOE_' prefix
        # (e.g., if original was "Age_x_WOE_Income" but we didn't match above)
        if var2.startswith('WOE_'):
            # Remove the WOE_ prefix from var2 as well
            var2 = var2[4:]
        
        # Return both cleaned variable names
        return parts[0], var2
    
    # If we get here, the name doesn't match any known pattern
    # This is an error case - raise an exception with helpful message
    raise ValueError(f"Cannot parse interaction term: {var_name}")


def create_interaction_bins(
    bins: pd.DataFrame,
    var1: str,
    var2: str,
    interaction_name: str,
    coef: float,
    b: float,
    digits: int = 0
) -> List[Dict]:
    """
    Create scorecard entries for an interaction term.
    
    For an interaction between two variables, we need to create scorecard
    rows for every combination of their bins. If var1 has 5 bins and var2
    has 4 bins, we create 5 × 4 = 20 interaction bin combinations.
    
    The interaction WOE is calculated as: woe_interaction = woe1 × woe2
    This captures how the combined bin effects multiply together.
    
    The points formula is the same as regular variables:
    points = round(-b × coefficient × WOE)
    But WOE is now the product of two WOE values.
    
    Example:
        If Age bin "[25,35)" has WOE = 0.3
        And Income bin "[50k,75k)" has WOE = 0.2
        Then interaction WOE = 0.3 × 0.2 = 0.06
        If coefficient = 0.5 and b = 72.13
        Then points = round(-72.13 × 0.5 × 0.06) = round(-2.16) = -2
    
    Parameters:
        bins: DataFrame containing the binning rules for all variables
              Must have 'var', 'bin'/'binValue', and 'woe' columns
        var1: Name of the first component variable (without WOE_ prefix)
        var2: Name of the second component variable (without WOE_ prefix)
        interaction_name: Display name for the interaction (e.g., "Age_x_Income")
        coef: The logistic regression coefficient for this interaction term
        b: The scaling parameter from calculate_ab()
        digits: Number of decimal places for rounding points (default 0 = integers)
        
    Returns:
        List of dictionaries, each representing one row in the scorecard.
        Each dict has keys: 'var', 'bin', 'binValue', 'woe', 'points'
        
    Note:
        If either component variable is not found in the bins table,
        returns an empty list and prints a warning.
    """
    # Initialize empty list to collect scorecard rows
    rows = []
    
    # Get the bins for the first variable from the bins DataFrame
    # Filter to only rows where 'var' column equals var1
    # .copy() creates a new DataFrame to avoid modifying the original
    var1_bins = bins[bins['var'] == var1].copy()
    
    # Same for the second variable
    var2_bins = bins[bins['var'] == var2].copy()
    
    # Check if we found bins for both variables
    # .empty is True if the DataFrame has no rows
    if var1_bins.empty or var2_bins.empty:
        # Print warning with details about what's missing
        print(f"WARNING: Cannot create interaction bins for {interaction_name}")
        print(f"  var1 '{var1}' has {len(var1_bins)} bins")
        print(f"  var2 '{var2}' has {len(var2_bins)} bins")
        # Return empty list - no bins to create
        return rows
    
    # Calculate total number of combinations (for informational logging)
    # This is the Cartesian product: n1 × n2
    total_combinations = len(var1_bins) * len(var2_bins)
    
    # If there are many combinations, print an info message
    # 1000 is arbitrary threshold - just to warn about potentially slow processing
    if total_combinations > 1000:
        print(f"INFO: Interaction {interaction_name} will create {total_combinations:,} combinations")
        # :, in the f-string adds thousands separators (e.g., 1,234)
        print(f"  var1 '{var1}': {len(var1_bins)} bins")
        print(f"  var2 '{var2}': {len(var2_bins)} bins")
        print(f"  This may take some time to process...")
    
    # Create all combinations using nested loops
    # Outer loop: iterate over each bin of variable 1
    for _, row1 in var1_bins.iterrows():
        # iterrows() returns index and row as a tuple
        # We use _ for the index because we don't need it
        
        # Get the WOE value for this bin
        # .get() with default 0 handles the case where 'woe' column might not exist
        woe1 = row1.get('woe', 0)
        
        # Handle missing WOE values (NaN/NA)
        # pd.isna() returns True if the value is NaN, None, or NA
        if pd.isna(woe1):
            woe1 = 0
        
        # Get the bin label for display
        # Try 'binValue' first (preferred display name), fall back to 'bin'
        bin1 = row1.get('binValue', row1.get('bin', ''))
        
        # Inner loop: iterate over each bin of variable 2
        for _, row2 in var2_bins.iterrows():
            # Get WOE for var2's bin
            woe2 = row2.get('woe', 0)
            if pd.isna(woe2):
                woe2 = 0
            
            # Get bin label for var2
            bin2 = row2.get('binValue', row2.get('bin', ''))
            
            # Calculate the interaction WOE as the product of individual WOEs
            # This is the standard approach for WOE interactions
            # Mathematically: log-odds contribution = coef × woe1 × woe2
            interaction_woe = woe1 * woe2
            
            # Calculate points using the standard formula
            # Negative sign: higher WOE (riskier) = fewer points
            # b: scaling factor from calculate_ab()
            # coef: logistic regression coefficient
            # round(..., digits): round to specified decimal places
            points = round(-b * coef * interaction_woe, digits)
            
            # Create a combined bin label that shows both components
            # Format: "Age:[25,35) × Income:[50k,75k)"
            # This makes it clear which combination of bins this row represents
            combined_bin = f"{var1}:{bin1} × {var2}:{bin2}"
            
            # Append a dictionary with all the scorecard columns
            # This will become one row in the final scorecard DataFrame
            rows.append({
                'var': interaction_name,      # The interaction name (e.g., "Age_x_Income")
                'bin': combined_bin,          # Combined bin label
                'binValue': combined_bin,     # Same as bin (for consistency)
                'woe': round(interaction_woe, 6),  # WOE rounded to 6 decimal places
                'points': points              # The calculated points (integer if digits=0)
            })
    
    # Return the list of all combination rows
    return rows


def create_scorecard(
    bins: pd.DataFrame,
    coefficients: pd.DataFrame,
    points0: float = 600,
    odds0: float = 1/19,
    pdo: float = 50,
    basepoints_eq0: bool = False,
    digits: int = 0
) -> pd.DataFrame:
    """
    Create a complete scorecard from binning rules and logistic regression coefficients.
    
    This is the main function that orchestrates the entire scorecard creation process.
    It handles both regular variables and interaction terms.
    
    The process:
    1. Calculate scaling parameters (a, b) from the scorecard settings
    2. Parse coefficients and identify the intercept
    3. Calculate base points from the intercept
    4. For each regular variable: calculate points for each bin
    5. For each interaction term: create all bin combinations with points
    6. Combine everything into a single scorecard DataFrame
    
    Parameters:
        bins: DataFrame with binning rules from WOE Editor
              Required columns: 'var', 'woe'
              Optional columns: 'bin', 'binValue'
        coefficients: DataFrame with model coefficients from Logistic Regression
                      Index = variable name (e.g., "(Intercept)", "WOE_Age")
                      First column = coefficient value
        points0: Base score at target odds (default 600)
        odds0: Target odds ratio as decimal (default 1/19)
        pdo: Points to Double the Odds (default 50)
        basepoints_eq0: If True, force basepoints to 0 (default False)
                        Used when you want to distribute all points to variables
        digits: Number of decimal places for rounding points (default 0)
        
    Returns:
        DataFrame with columns: ['var', 'bin', 'binValue', 'woe', 'points']
        First row is always 'basepoints' with the base score
        Remaining rows are the bins for each variable with their points
        
    Example output:
        var         bin           woe     points
        basepoints  None          None    450
        Age         [18,25)       -0.5    25
        Age         [25,35)       0.0     0
        Age         [35,50)       0.3     -15
        Income      [0,30k)       -0.8    40
        ...
    """
    # Step 1: Calculate the scaling parameters a and b
    # These convert log-odds to points using the scorecard formula
    a, b = calculate_ab(points0, odds0, pdo)
    
    # Step 2: Prepare coefficients
    # The coefficients DataFrame has variable names as index and values in columns
    # Get the name of the first column (usually "model$coefficients" or similar)
    coef_col = coefficients.columns[0] if len(coefficients.columns) > 0 else 'coefficients'
    
    # Create dictionaries to look up coefficients by variable name
    # coef_dict: keeps original names (with WOE_ prefix) for matching
    # coef_dict_clean: strips WOE_ prefix for matching with bins table
    coef_dict = {}         # Full variable names -> coefficient values
    coef_dict_clean = {}   # Cleaned variable names -> coefficient values
    intercept = 0.0        # Will store the intercept coefficient
    
    # Iterate through each row of the coefficients DataFrame
    # .iterrows() yields (index, row) pairs
    # The index is the variable name (e.g., "(Intercept)", "WOE_Age")
    for var_name, row in coefficients.iterrows():
        # Get the coefficient value from the first column
        # row.iloc[0] gets the first value in the row (by position)
        # Fall back to row[coef_col] if iloc fails
        coef_value = row.iloc[0] if len(row) > 0 else row[coef_col]
        
        # Check if this is the intercept term
        # R uses "(Intercept)" with parentheses, so we check for both formats
        if var_name == '(Intercept)' or var_name.lower() == 'intercept':
            # Store the intercept separately - it gets special treatment
            intercept = coef_value
        else:
            # Regular variable - store in both dictionaries
            coef_dict[var_name] = coef_value
            
            # Create a "clean" version with WOE_ prefix removed
            # This helps match with the bins table which uses base names
            # "WOE_Age" -> "Age"
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            coef_dict_clean[clean_var] = coef_value
    
    # Step 3: Calculate base points
    # The intercept from logistic regression gets converted to base points
    if basepoints_eq0:
        # If user wants basepoints = 0, set it directly
        # (All points will be distributed to the variable bins)
        basepoints = 0
    else:
        # Standard formula: basepoints = a - b × intercept
        # This anchors the score scale based on the model's intercept
        basepoints = round(a - b * intercept, digits)
    
    # Step 4: Initialize the list that will hold all scorecard rows
    scorecard_rows = []
    
    # Add the basepoints row first
    # This is a special row that represents the starting score before
    # any variable contributions are added
    scorecard_rows.append({
        'var': 'basepoints',  # Special variable name
        'bin': None,          # No bin for basepoints
        'binValue': None,     # No bin value for basepoints
        'woe': None,          # No WOE for basepoints
        'points': basepoints  # The base score value
    })
    
    # Step 5: Process each variable in the bins table
    # Make a copy to avoid modifying the original DataFrame
    bins_copy = bins.copy()
    
    # Validate that required columns exist in the bins table
    # These columns are essential for scorecard creation
    if 'var' not in bins_copy.columns:
        raise ValueError("Bins table must have 'var' column")
    if 'woe' not in bins_copy.columns:
        raise ValueError("Bins table must have 'woe' column")
    
    # Step 6: Separate regular variables from interaction terms
    # They need different processing logic
    regular_vars = []      # Will hold tuples of (full_name, clean_name)
    interaction_vars = []  # Will hold interaction term names
    
    # Iterate through all variables that have coefficients
    for var_name in coef_dict.keys():
        # Check if this is an interaction term
        if is_interaction_term(var_name):
            # Add to interaction list for later processing
            interaction_vars.append(var_name)
        else:
            # Regular variable - strip WOE_ prefix for matching with bins
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            # Store both names as a tuple: (original_for_coefficient, cleaned_for_bins)
            regular_vars.append((var_name, clean_var))
    
    # Step 7: Process regular (non-interaction) variables
    for full_var, clean_var in regular_vars:
        # Check if this variable exists in the bins table
        # .unique() gets distinct values in the 'var' column
        if clean_var not in bins_copy['var'].unique():
            # Variable not found - print warning and skip
            print(f"WARNING: Variable '{clean_var}' not found in bins table")
            continue
        
        # Get all bins for this variable from the bins table
        var_bins = bins_copy[bins_copy['var'] == clean_var].copy()
        
        # Get the coefficient for this variable
        coef = coef_dict[full_var]
        
        # Iterate through each bin of this variable
        for _, row in var_bins.iterrows():
            # Get the WOE value for this bin
            woe = row.get('woe', 0)
            
            # Handle missing WOE values
            if pd.isna(woe):
                woe = 0
            
            # Calculate points for this bin
            # Formula: points = round(-b × coefficient × WOE)
            # Negative sign: higher WOE (riskier) = fewer points
            points = round(-b * coef * woe, digits)
            
            # Add this bin as a row in the scorecard
            scorecard_rows.append({
                'var': clean_var,                      # Variable name (cleaned)
                'bin': row.get('bin', None),           # Bin label (e.g., "[25,35)")
                'binValue': row.get('binValue', None), # Bin display value
                'woe': woe,                            # WOE value for this bin
                'points': points                       # Calculated points
            })
    
    # Step 8: Process interaction terms
    for interaction_name in interaction_vars:
        try:
            # Parse the interaction name to get component variable names
            # e.g., "WOE_Age_x_WOE_Income" -> ("Age", "Income")
            var1, var2 = parse_interaction_term(interaction_name)
            
            # Get the coefficient for this interaction term
            coef = coef_dict[interaction_name]
            
            # Create all combination bins for this interaction
            # This function handles the nested loop logic
            interaction_rows = create_interaction_bins(
                bins=bins_copy,
                var1=var1,
                var2=var2,
                interaction_name=f"{var1}_x_{var2}",  # Clean display name
                coef=coef,
                b=b,
                digits=digits
            )
            
            # Add all interaction rows to the main list
            # extend() adds each item from the list individually
            scorecard_rows.extend(interaction_rows)
            
            # Print info about how many bins were created
            print(f"Created {len(interaction_rows)} bins for interaction: {var1} × {var2}")
            
        except Exception as e:
            # If anything goes wrong, print error and continue with other variables
            print(f"ERROR processing interaction '{interaction_name}': {e}")
    
    # Step 9: Convert the list of dictionaries to a DataFrame
    scorecard_df = pd.DataFrame(scorecard_rows)
    
    # Step 10: Reorder columns to a standard order
    # This ensures consistent output regardless of how rows were added
    col_order = ['var', 'bin', 'binValue', 'woe', 'points']
    
    # Only include columns that actually exist in the DataFrame
    # (in case some optional columns weren't created)
    col_order = [c for c in col_order if c in scorecard_df.columns]
    
    # Reorder the DataFrame columns
    scorecard_df = scorecard_df[col_order]
    
    # Return the complete scorecard DataFrame
    return scorecard_df


def create_scorecard_list(scorecard_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert a scorecard DataFrame to a list format (dictionary of DataFrames).
    
    This is an alternative output format where instead of one big table,
    you get a separate DataFrame for each variable. This can be useful for:
    - Displaying scorecards in a more compact format
    - Processing variables independently
    - Creating per-variable reports
    
    Parameters:
        scorecard_df: The scorecard DataFrame from create_scorecard()
        
    Returns:
        Dictionary where:
        - Keys are variable names (strings)
        - Values are DataFrames containing just that variable's bins
        
    Example:
        Input scorecard_df:
            var         bin           woe    points
            basepoints  None          None   450
            Age         [18,25)       -0.5   25
            Age         [25,35)       0.0    0
            Income      [0,30k)       -0.8   40
            
        Output:
            {
                'basepoints': DataFrame with 1 row (basepoints),
                'Age': DataFrame with 2 rows (Age bins),
                'Income': DataFrame with 1 row (Income bins)
            }
    """
    # Initialize empty dictionary to hold the results
    card_list = {}
    
    # Iterate through each unique variable name in the scorecard
    # .unique() returns array of distinct values
    for var in scorecard_df['var'].unique():
        # Filter to get only rows for this variable
        var_df = scorecard_df[scorecard_df['var'] == var].copy()
        
        # Reset the index so each mini-DataFrame has clean 0-based indices
        # drop=True means don't keep the old index as a column
        card_list[var] = var_df.reset_index(drop=True)
    
    # Return the dictionary of DataFrames
    return card_list


# =============================================================================
# Shiny UI Application
# =============================================================================
# This section defines the interactive web-based user interface using Shiny.
# Shiny is a reactive web framework that automatically updates outputs when
# inputs change. Originally from R, now available for Python.

def create_scorecard_app(coefficients: pd.DataFrame, bins: pd.DataFrame):
    """
    Create the Scorecard Generator Shiny application.
    
    This function defines both the UI layout and the server logic for an
    interactive scorecard generation tool. Users can:
    1. Adjust scorecard parameters (Points, Odds, PDO)
    2. Generate a preview of the scorecard
    3. View summary statistics
    4. Save results and close the app
    
    The app uses reactive programming: when users change inputs, the
    dependent outputs automatically update.
    
    Parameters:
        coefficients: DataFrame with model coefficients (passed to create_scorecard)
        bins: DataFrame with binning rules (passed to create_scorecard)
        
    Returns:
        A Shiny App object that can be run with app.run()
        The app has a 'results' attribute to retrieve the final scorecard
    """
    # Dictionary to store results from the app
    # This is how we get data back after the user closes the app
    app_results = {
        'scorecard': None,    # Will hold the final scorecard DataFrame
        'completed': False    # Flag to indicate if user clicked "Run & Close"
    }
    
    # ==========================================================================
    # Define the User Interface (UI) Layout
    # ==========================================================================
    # ui.page_fluid() creates a responsive page that adjusts to screen width
    app_ui = ui.page_fluid(
        # Add custom CSS styles in the page header
        ui.tags.head(
            ui.tags.style("""
                /* Import Google Font for a clean, professional look */
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                
                /* Base body styles - light gray background, dark text */
                body { 
                    font-family: 'Source Sans Pro', sans-serif; 
                    background: #f5f7fa;
                    min-height: 100vh;
                    color: #2c3e50;
                }
                
                /* Card component styles - white background with shadow */
                .card { 
                    background: #ffffff;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                /* Card header with blue underline */
                .card-header {
                    color: #2c3e50;
                    font-weight: 700;
                    font-size: 1.1rem;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }
                
                /* Main heading styles */
                h3 { 
                    color: #2c3e50; 
                    text-align: center; 
                    font-weight: 700;
                    margin-bottom: 24px;
                }
                
                /* Primary button (blue) */
                .btn-primary { 
                    background: #3498db;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 10px 24px;
                    border-radius: 6px;
                }
                .btn-primary:hover {
                    background: #2980b9;
                }
                
                /* Success button (green) - for "Run & Close" */
                .btn-success { 
                    background: #27ae60;
                    border: none;
                    color: white;
                    font-weight: 700;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-success:hover {
                    background: #219a52;
                }
                
                /* Secondary button (gray) - for "Close" */
                .btn-secondary { 
                    background: #95a5a6;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-secondary:hover {
                    background: #7f8c8d;
                }
                
                /* Form input styling */
                .form-control, .form-select {
                    background: #ffffff;
                    border: 1px solid #ced4da;
                    color: #2c3e50;
                    border-radius: 6px;
                }
                .form-control:focus, .form-select:focus {
                    background: #ffffff;
                    border-color: #3498db;
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
                    color: #2c3e50;
                }
                
                /* Form labels */
                .form-label {
                    color: #2c3e50;
                    font-weight: 600;
                }
                
                /* Parameter input boxes */
                .param-box {
                    background: #f8f9fa;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                }
                
                /* Metric display boxes */
                .metric-box {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 14px;
                    text-align: center;
                    border: 1px solid #e1e8ed;
                }
                .metric-value {
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #2c3e50;
                    text-align: center;
                }
                .metric-label {
                    color: #7f8c8d;
                    text-align: center;
                    font-size: 0.85rem;
                    margin-top: 4px;
                }
                
                /* Grid layout for metrics */
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                    margin-top: 12px;
                }
                
                /* Scorecard table container with scroll */
                .scorecard-table-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: auto;
                    width: 100%;
                }
                
                /* Table width fixes to prevent resizing during scroll */
                .scorecard-table-container > div {
                    width: 100% !important;
                    min-width: 100% !important;
                }
                .scorecard-table-container table {
                    width: 100% !important;
                    min-width: 600px;
                    table-layout: fixed !important;
                }
                .scorecard-table-container th,
                .scorecard-table-container td {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    padding: 8px 12px;
                }
                
                /* Fixed column widths for scorecard table */
                /* Column 1: Variable name (150px) */
                .scorecard-table-container th:nth-child(1),
                .scorecard-table-container td:nth-child(1) {
                    width: 150px !important;
                    min-width: 150px !important;
                    max-width: 150px !important;
                }
                /* Column 2: Bin (250px) */
                .scorecard-table-container th:nth-child(2),
                .scorecard-table-container td:nth-child(2) {
                    width: 250px !important;
                    min-width: 250px !important;
                    max-width: 250px !important;
                }
                /* Column 3: WOE (100px) */
                .scorecard-table-container th:nth-child(3),
                .scorecard-table-container td:nth-child(3) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
                /* Column 4: Points (100px) */
                .scorecard-table-container th:nth-child(4),
                .scorecard-table-container td:nth-child(4) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
            """)
        ),
        
        # Page title - centered heading
        ui.h3("Scorecard Generator"),
        
        # ==========================================================================
        # Configuration Card - Parameter Inputs
        # ==========================================================================
        ui.div(
            {"class": "card"},  # Apply card styling
            ui.div({"class": "card-header"}, "Scorecard Parameters"),  # Card title
            
            # Row with 4 columns for input parameters
            ui.row(
                # Column 1: Base Points input
                ui.column(3,  # 3 out of 12 grid units (25% width)
                    ui.div(
                        {"class": "param-box"},  # Styling box
                        # Numeric input for base points
                        ui.input_numeric("points", "Base Points", value=600, min=0, step=50),
                        # Helper text explaining the field
                        ui.tags.small("Score at target odds", style="color: #7f8c8d;")
                    )
                ),
                
                # Column 2: Odds Ratio input
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        # Numeric input for odds ratio
                        ui.input_numeric("odds", "Odds Ratio (1:X)", value=20, min=2, step=1),
                        ui.tags.small("Target odds (e.g., 20 = 1:19)", style="color: #7f8c8d;")
                    )
                ),
                
                # Column 3: PDO input
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        # Numeric input for PDO
                        ui.input_numeric("pdo", "Points to Double Odds", value=50, min=10, step=10),
                        ui.tags.small("PDO scaling factor", style="color: #7f8c8d;")
                    )
                ),
                
                # Column 4: Output Format selector
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        # Dropdown select for output format
                        ui.input_select("output_format", "Output Format", 
                                       choices=["Table", "List"],
                                       selected="Table"),
                        ui.tags.small("Scorecard output style", style="color: #7f8c8d;")
                    )
                )
            ),
            
            # Row for the Generate button (centered)
            ui.row(
                ui.column(12,  # Full width column
                    ui.div(
                        {"style": "text-align: center; margin-top: 15px;"},
                        # Action button that triggers scorecard generation
                        ui.input_action_button("analyze", "Generate Scorecard", class_="btn btn-primary btn-lg")
                    )
                )
            )
        ),
        
        # ==========================================================================
        # Summary Stats Card - Model Summary Display
        # ==========================================================================
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Summary"),
            # This output will be rendered by the server function
            ui.output_ui("summary_stats")
        ),
        
        # ==========================================================================
        # Scorecard Table Card - Main Results Display
        # ==========================================================================
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard"),
            ui.div(
                {"class": "scorecard-table-container"},
                # DataGrid output for the scorecard table
                ui.output_data_frame("scorecard_table")
            )
        ),
        
        # ==========================================================================
        # Action Buttons Card - Run & Close buttons
        # ==========================================================================
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            # Green button to save results and close
            ui.input_action_button("run", "Run & Close", class_="btn btn-success btn-lg"),
            # Gray button to close without saving
            ui.input_action_button("close", "Close", class_="btn btn-secondary btn-lg")
        )
    )
    
    # ==========================================================================
    # Define the Server Logic
    # ==========================================================================
    # This function contains all the reactive logic that responds to user input
    def server(input: Inputs, output: Outputs, session: Session):
        # Create a reactive value to hold the current scorecard
        # Reactive values automatically trigger updates when they change
        scorecard_rv = reactive.Value(None)
        
        # ==========================
        # Generate Scorecard Handler
        # ==========================
        @reactive.Effect           # This is a reactive effect (side effect)
        @reactive.event(input.analyze)  # Triggers when "analyze" button is clicked
        def generate_scorecard():
            """Generate the scorecard when button is clicked."""
            # Get current values from inputs, with defaults if empty
            points = input.points() or 600  # Get points input value
            odds = input.odds() or 20       # Get odds input value
            pdo = input.pdo() or 50         # Get PDO input value
            
            # Convert odds input (1:X format) to decimal
            # User enters 20, meaning 1:19 odds, so decimal = 1/(20-1) = 1/19
            odds_decimal = 1 / (odds - 1)
            
            try:
                # Call the scorecard creation function
                card = create_scorecard(
                    bins=bins,                   # Bins DataFrame from closure
                    coefficients=coefficients,  # Coefficients DataFrame from closure
                    points0=points,             # User-specified base points
                    odds0=odds_decimal,         # Converted odds value
                    pdo=pdo,                    # User-specified PDO
                    basepoints_eq0=False,       # Use normal basepoints
                    digits=0                    # Round to integers
                )
                
                # Use binValue for display instead of bin column
                # binValue typically has nicer formatting
                if 'binValue' in card.columns:
                    card['bin'] = card['binValue']        # Copy binValue to bin
                    card = card.drop(columns=['binValue'])  # Remove binValue column
                
                # Update the reactive value - this triggers UI updates
                scorecard_rv.set(card)
                
            except Exception as e:
                # If an error occurs, print it for debugging
                print(f"Error generating scorecard: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace
        
        # ==========================
        # Summary Statistics Renderer
        # ==========================
        @output                    # This is an output renderer
        @render.ui                 # Renders UI components
        def summary_stats():
            """Render the summary statistics panel."""
            # Get the current scorecard from reactive value
            card = scorecard_rv.get()
            
            # If no scorecard yet, show placeholder message
            if card is None:
                return ui.p("Click 'Generate Scorecard' to view summary", 
                           style="text-align: center; color: #7f8c8d;")
            
            # Calculate summary statistics
            # Count variables (excluding basepoints)
            num_vars = len([v for v in card['var'].unique() if v != 'basepoints'])
            
            # Count total bins (excluding basepoints row)
            total_bins = len(card) - 1
            
            # Get the basepoints value
            basepoints_row = card[card['var'] == 'basepoints']
            basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
            
            # Calculate min and max possible scores
            # Start with basepoints
            min_score = basepoints
            max_score = basepoints
            
            # Add min/max points from each variable
            for var in card['var'].unique():
                if var == 'basepoints':
                    continue  # Skip basepoints
                var_points = card[card['var'] == var]['points']
                if not var_points.empty:
                    min_score += var_points.min()  # Add minimum points
                    max_score += var_points.max()  # Add maximum points
            
            # Return the UI with metrics grid
            return ui.div(
                {"class": "metrics-grid"},
                # Box 1: Number of variables
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{num_vars}"),
                    ui.div({"class": "metric-label"}, "Variables")
                ),
                # Box 2: Total bins
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{total_bins}"),
                    ui.div({"class": "metric-label"}, "Total Bins")
                ),
                # Box 3: Minimum score
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(min_score)}"),
                    ui.div({"class": "metric-label"}, "Min Score")
                ),
                # Box 4: Maximum score
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(max_score)}"),
                    ui.div({"class": "metric-label"}, "Max Score")
                )
            )
        
        # ==========================
        # Scorecard Table Renderer
        # ==========================
        @output                    # This is an output renderer
        @render.data_frame         # Renders a DataGrid component
        def scorecard_table():
            """Render the scorecard table."""
            # Get the current scorecard
            card = scorecard_rv.get()
            
            # If no scorecard yet, show empty grid
            if card is None:
                return render.DataGrid(pd.DataFrame())
            
            # Create a copy for display formatting
            display_df = card.copy()
            
            # Round WOE values for cleaner display
            if 'woe' in display_df.columns:
                display_df['woe'] = display_df['woe'].round(4)
            
            # Return a DataGrid with specified dimensions
            return render.DataGrid(display_df, height="450px", width="100%")
        
        # ==========================
        # Run & Close Button Handler
        # ==========================
        @reactive.Effect           # This is a reactive effect
        @reactive.event(input.run)  # Triggers when "run" button is clicked
        async def run_and_close():
            """Save results and close the application."""
            # Get the current scorecard
            card = scorecard_rv.get()
            
            if card is not None:
                # Store the scorecard in the results dictionary
                app_results['scorecard'] = card
                # Mark as completed (signals to main script that we have results)
                app_results['completed'] = True
            
            # Close the session - this stops the Shiny app
            await session.close()
        
        # ==========================
        # Close Button Handler
        # ==========================
        @reactive.Effect           # This is a reactive effect
        @reactive.event(input.close)  # Triggers when "close" button is clicked
        async def close_app():
            """Close the application without saving."""
            # Just close the session - don't save results
            await session.close()
    
    # Create the Shiny App object by combining UI and server
    app = App(app_ui, server)
    
    # Attach the results dictionary to the app object
    # This allows the calling code to access results after the app closes
    app.results = app_results
    
    # Return the configured app
    return app


def run_scorecard_ui(coefficients: pd.DataFrame, bins: pd.DataFrame, port: int = 8052):
    """
    Run the Scorecard Generator application and wait for user to complete.
    
    This function:
    1. Creates the Shiny app
    2. Starts it in a background thread
    3. Opens the user's browser
    4. Waits for the user to click "Run & Close"
    5. Returns the results
    
    Parameters:
        coefficients: DataFrame with model coefficients
        bins: DataFrame with binning rules
        port: HTTP port to run the app on (default 8052)
        
    Returns:
        Dictionary with:
        - 'scorecard': The generated scorecard DataFrame (or None)
        - 'completed': Boolean indicating if user completed the workflow
    """
    # Import additional modules needed for running the server
    import threading  # For running server in background thread
    import time       # For sleep/wait operations
    import socket     # For checking if port is available
    
    def is_port_available(port):
        """
        Check if a TCP port is available for use.
        
        Tries to bind to the port - if successful, port is available.
        If bind fails, something else is using the port.
        """
        # Create a TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try to bind to localhost on the specified port
                s.bind(('127.0.0.1', port))
                # If we get here, bind succeeded - port is available
                return True
            except socket.error:
                # Bind failed - port is already in use
                return False
    
    # Check if our desired port is available
    if not is_port_available(port):
        # Port is in use - warn the user and try next port
        print(f"WARNING: Port {port} is already in use!")
        print(f"Trying to use port {port+1} instead...")
        port = port + 1
        
        # Check if the alternate port is available
        if not is_port_available(port):
            # Both ports in use - warn but continue anyway
            print(f"ERROR: Port {port} is also in use. Please close other applications.")
            print("Continuing anyway - the app may not work correctly...")
    
    # Create the Shiny app
    app = create_scorecard_app(coefficients, bins)
    
    def run_server():
        """
        Function to run the Shiny server.
        This runs in a separate thread so the main thread can wait.
        """
        try:
            # Print instructions for the user
            print("=" * 70)
            print(f"Starting Shiny UI on http://127.0.0.1:{port}")
            print("=" * 70)
            print("IMPORTANT: A browser window should open automatically.")
            print("If it doesn't, manually open: http://127.0.0.1:{port}")
            print("")
            print("STEPS TO COMPLETE:")
            print("  1. Configure parameters in the browser UI")
            print("  2. Click 'Generate Scorecard' button")
            print("  3. Review the scorecard table")
            print("  4. Click 'Run & Close' button (green button at bottom)")
            print("")
            print("Waiting for you to complete the UI workflow...")
            print("=" * 70)
            
            # Start the app - this blocks until the app is closed
            # launch_browser=True automatically opens user's default browser
            app.run(port=port, launch_browser=True)
            
        except Exception as e:
            # If server stops with error, print it
            print(f"Server stopped: {e}")
    
    # Create a thread to run the server
    # daemon=True means the thread will be killed when main thread exits
    server_thread = threading.Thread(target=run_server, daemon=True)
    
    # Start the server thread
    server_thread.start()
    
    # Give the server time to start up before checking for completion
    time.sleep(2)
    
    # Wait loop - check if user has completed the workflow
    wait_count = 0
    
    # Keep waiting until user clicks "Run & Close"
    # app.results is the dictionary we attached to the app
    while not app.results.get('completed', False):
        # Sleep for 1 second between checks
        time.sleep(1)
        wait_count += 1
        
        # Every 10 seconds, print a reminder message
        if wait_count % 10 == 0:
            print(f"Still waiting... ({wait_count} seconds elapsed)")
            print(f"Make sure browser is open at: http://127.0.0.1:{port}")
    
    # Give a moment for cleanup after user closes
    time.sleep(0.5)
    
    # Print completion message
    print("=" * 70)
    print("Scorecard generation complete - returning results")
    print("=" * 70)
    
    # Return the results dictionary
    return app.results


# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================
# This section runs when the script is executed by KNIME.
# It reads input data, determines the mode, and generates the scorecard.

# Print startup message to KNIME console
print("Scorecard Generator Node - Starting...")
print("=" * 70)

# =============================================================================
# Read Input Data from KNIME
# =============================================================================
# KNIME passes data through input ports. This script expects 2 inputs:
# - Input 1: Coefficients table from Logistic Regression node
# - Input 2: Bins table from WOE Editor node

# Read Input 1: Coefficients table
# knio.input_tables[0] is the first input port (0-indexed)
# .to_pandas() converts KNIME's table format to a pandas DataFrame
coefficients = knio.input_tables[0].to_pandas()

# Print info about the coefficients table
print(f"Input 1 (Coefficients): {len(coefficients)} terms")

# Read Input 2: Bins table
# knio.input_tables[1] is the second input port
bins = knio.input_tables[1].to_pandas()

# Print info about the bins table
print(f"Input 2 (Bins): {len(bins)} rows")

# Calculate and display bins per variable statistics
if 'var' in bins.columns:
    # Group by variable and count bins for each
    bins_per_var = bins.groupby('var').size()
    
    # Get statistics
    max_bins = bins_per_var.max()   # Maximum bins for any variable
    avg_bins = bins_per_var.mean()  # Average bins per variable
    
    # Print summary
    print(f"\nBins per variable: min={bins_per_var.min()}, avg={avg_bins:.1f}, max={max_bins}")
    
    # If any variable has many bins, show the top 5
    # This is informational to help user understand their data
    if max_bins > 20:
        print(f"\nVariables with most bins:")
        # .nlargest(5) returns the 5 variables with most bins
        for var, count in bins_per_var.nlargest(5).items():
            print(f"  - {var}: {count} bins")

# Print the coefficient variable names for debugging
print("\nCoefficients:")
for var_name in coefficients.index:
    print(f"  - {var_name}")

# =============================================================================
# Check for Flow Variables (Headless Mode Detection)
# =============================================================================
# KNIME can pass parameters via flow variables. If the user provides
# Points, Odds, and PDO as flow variables, we run in headless mode
# (no UI). Otherwise, we launch the interactive Shiny UI.

# Initialize flag and default values
has_flow_vars = False  # Will be True if any flow variable is provided
points = 600           # Default base points
odds = 20              # Default odds ratio (1:19)
pdo = 50               # Default PDO
output_format = "Table"  # Default output format

# Try to read "Points" flow variable
try:
    # knio.flow_variables.get() retrieves a flow variable by name
    # Returns None if not set (second argument is default)
    points_fv = knio.flow_variables.get("Points", None)
    if points_fv is not None:
        # Convert to integer (flow variables might be strings or floats)
        points = int(points_fv)
        # Mark that we found a flow variable
        has_flow_vars = True
except:
    # If any error occurs, silently continue with default
    # This handles cases where flow_variables isn't available
    pass

# Try to read "Odds" flow variable
try:
    odds_fv = knio.flow_variables.get("Odds", None)
    if odds_fv is not None:
        odds = int(odds_fv)
        has_flow_vars = True
except:
    pass

# Try to read "PDO" flow variable
try:
    pdo_fv = knio.flow_variables.get("PDO", None)
    if pdo_fv is not None:
        pdo = int(pdo_fv)
        has_flow_vars = True
except:
    pass

# Try to read "OutputFormat" flow variable
try:
    output_format = knio.flow_variables.get("OutputFormat", "Table")
except:
    pass

# Print the parameters that will be used
print(f"\nParameters: Points={points}, Odds={odds}, PDO={pdo}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================
# Based on whether flow variables were provided, run in headless or
# interactive mode.

# Initialize empty DataFrame to hold the scorecard result
scorecard = pd.DataFrame()

# Check if we should run in headless mode (flow variables provided)
if has_flow_vars:
    # =========================================================================
    # HEADLESS MODE - No UI, process automatically with provided parameters
    # =========================================================================
    print("Running in HEADLESS mode")
    
    # Convert odds input (1:X format) to decimal
    # User specifies 20, meaning 1:19 odds
    # Decimal = 1 / (20 - 1) = 1/19 ≈ 0.0526
    odds_decimal = 1 / (odds - 1)
    
    try:
        # Generate the scorecard using the core function
        scorecard = create_scorecard(
            bins=bins,                   # Bins table from WOE Editor
            coefficients=coefficients,  # Coefficients from Logistic Regression
            points0=points,             # Base points from flow variable
            odds0=odds_decimal,         # Calculated odds decimal
            pdo=pdo,                    # PDO from flow variable
            basepoints_eq0=False,       # Use normal basepoints calculation
            digits=0                    # Round to integers
        )
        
        # Use binValue for display instead of bin
        # This matches R's behavior where binValue has the display format
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])
        
        # Print confirmation
        print(f"\nScorecard created with {len(scorecard)} rows")
        
    except Exception as e:
        # If scorecard creation fails, print error details
        print(f"ERROR creating scorecard: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

else:
    # =========================================================================
    # INTERACTIVE MODE - Launch Shiny UI for user to configure parameters
    # =========================================================================
    
    # Check if Shiny is available
    if SHINY_AVAILABLE:
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        
        # Run the Shiny application and wait for user to complete
        results = run_scorecard_ui(coefficients, bins)
        
        # Check if user completed the workflow (clicked "Run & Close")
        if results['completed']:
            # User completed - get the scorecard from results
            scorecard = results['scorecard']
            print("Interactive session completed successfully")
        else:
            # User cancelled or closed without completing
            print("Interactive session cancelled - returning empty results")
            
    else:
        # Shiny not available - can't run interactive mode
        print("=" * 70)
        print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
        print("Please provide flow variables for headless mode:")
        print("  - Points (int): Base score at target odds, default 600")
        print("  - Odds (int): Odds ratio (1:X), default 20")
        print("  - PDO (int): Points to Double the Odds, default 50")
        print("=" * 70)
        
        # Fall back to running with default parameters anyway
        # This ensures the node produces output even without UI
        odds_decimal = 1 / (odds - 1)
        
        scorecard = create_scorecard(
            bins=bins,
            coefficients=coefficients,
            points0=points,
            odds0=odds_decimal,
            pdo=pdo,
            basepoints_eq0=False,
            digits=0
        )
        
        # Clean up binValue column
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])

# =============================================================================
# Output Table to KNIME
# =============================================================================
# Send the scorecard back to KNIME through the output port.

# Ensure scorecard is a valid DataFrame with expected columns
# This handles edge cases where processing might have failed
if scorecard is None or scorecard.empty:
    # Create empty DataFrame with correct column structure
    scorecard = pd.DataFrame(columns=['var', 'bin', 'woe', 'points'])

# Write the scorecard to KNIME's first output port
# knio.output_tables[0] is the first output port (0-indexed)
# knio.Table.from_pandas() converts pandas DataFrame to KNIME table format
knio.output_tables[0] = knio.Table.from_pandas(scorecard)

# =============================================================================
# Print Summary
# =============================================================================
# Print final summary for the KNIME console.

print("=" * 70)
print("Scorecard Generator completed successfully")
print("=" * 70)

# Only print statistics if we have a non-empty scorecard
if not scorecard.empty:
    # Get the basepoints value
    basepoints_row = scorecard[scorecard['var'] == 'basepoints']
    basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
    
    # Calculate score range by summing min/max from each variable
    min_score = basepoints
    max_score = basepoints
    
    for var in scorecard['var'].unique():
        if var == 'basepoints':
            continue  # Skip basepoints row
        var_points = scorecard[scorecard['var'] == var]['points']
        if not var_points.empty:
            min_score += var_points.min()  # Add worst possible points
            max_score += var_points.max()  # Add best possible points
    
    # Count variables (excluding basepoints)
    num_vars = len([v for v in scorecard['var'].unique() if v != 'basepoints'])
    
    # Print the summary statistics
    print(f"Variables in scorecard: {num_vars}")
    print(f"Base points: {int(basepoints)}")
    print(f"Score range: {int(min_score)} to {int(max_score)}")

# Print final output info
print(f"\nOutput (Scorecard): {len(scorecard)} rows")
print("=" * 70)

# =============================================================================
# END OF SCRIPT
# =============================================================================
# The scorecard is now available in KNIME for further processing or export.
# Connect this node's output to:
# - A CSV Writer to save the scorecard
# - The Scorecard Apply node to score new data
# - A Table View to inspect the results
# =============================================================================

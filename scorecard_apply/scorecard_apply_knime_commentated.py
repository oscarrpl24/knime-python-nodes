# =============================================================================
# Scorecard Apply (Scoring) for KNIME Python Script Node
# =============================================================================
# 
# PURPOSE:
# This script is a Python implementation of R's scorecard scoring functionality.
# It applies a pre-built scorecard to data containing binned columns to calculate
# credit risk scores for each row (typically representing loan applications or customers).
#
# WHAT IS A SCORECARD?
# In credit risk modeling, a scorecard is a points-based system that converts
# statistical model outputs into a simple additive score. Each variable contributes
# points based on which "bin" (range/category) a customer falls into.
# Higher scores typically indicate lower risk (better creditworthiness).
#
# HOW IT WORKS:
# 1. Takes a scorecard table that defines points for each variable's bins
# 2. Takes data with binned columns (b_* prefix) from WOE Editor
# 3. Matches each row's bin values to the scorecard to look up points
# 4. Sums all points + base score to get final Score
#
# COMPATIBILITY:
# - KNIME Version: 5.9
# - Python Version: 3.9
#
# INPUTS (Two input ports):
# 1. Scorecard table (from Scorecard Generator node)
#    Required columns:
#    - var: variable name (e.g., "Age") or interaction name (e.g., "Age_x_Income")
#    - bin: bin label that matches the b_* column values (e.g., "[18,25)", "25+")
#    - points: integer score points assigned to each bin
#
# 2. Data table with binned columns (b_* prefix)
#    - Best to connect WOE Editor Port 4 (df_only_bins - lean output with ONLY b_* columns)
#    - Alternative: WOE Editor Port 2 also works (includes original + WOE columns)
#    - Each b_* column contains bin labels that correspond to scorecard bin values
#
# OUTPUTS (Two output ports):
# 1. Points only DataFrame containing:
#    - {var}_points for each variable (e.g., Age_points, Income_points)
#    - basescore column (the constant base score from the scorecard)
#    - Score column (total sum of all points)
#
# 2. Full data with scores:
#    - All original data columns
#    - All points columns from Output 1
#    - Useful for analysis and joining back to source data
#
# FLOW VARIABLES (optional configuration):
# - WithOriginalData (boolean, default True):
#   Controls whether Output 2 includes original data columns or just points
#
# SCORING LOGIC EXPLAINED:
#   For regular variables (e.g., "Age"):
#     - Find the b_Age column in the data
#     - For each row, look up the bin value (e.g., "[25,35)")
#     - Find matching bin in scorecard for variable "Age"
#     - Get the points value for that bin
#     - Store in Age_points column
#
#   For interaction terms (e.g., "Age_x_Income"):
#     - These represent the combined effect of two variables
#     - Get bin values from both b_Age and b_Income columns
#     - Create combined label: "Age:[25,35) × Income:[50000,75000)"
#     - Match against scorecard interaction bins
#     - Store in Age_x_Income_points column
#
#   Total Score calculation:
#     Score = basescore + Age_points + Income_points + ... (all variable points)
#
# VERSION HISTORY:
# 1.2 (2026-01-28): Fixed RowID preservation - output tables now maintain original
#                   RowIDs from input table, enabling joins back to source data
# 1.1: Added support for interaction terms (var1_x_var2)
# 1.0: Initial release
# =============================================================================

# =============================================================================
# IMPORT SECTION
# =============================================================================

# Import the KNIME Python scripting interface
# This module provides access to input/output tables and flow variables
# knio.input_tables[n] - read input port n
# knio.output_tables[n] - write to output port n
# knio.flow_variables - read flow variables passed from KNIME workflow
import knime.scripting.io as knio

# Import pandas - the primary data manipulation library
# Used for DataFrame operations (reading, filtering, merging, transforming data)
# pd is the conventional alias for pandas
import pandas as pd

# Import numpy - numerical computing library
# Provides efficient array operations and mathematical functions
# np is the conventional alias for numpy
import numpy as np

# Import warnings module to control Python warning messages
# Allows us to suppress or customize how warnings are displayed
import warnings

# Import re (regular expressions) module
# Used for pattern matching and string manipulation
# We use this to parse bin value expressions like "x >= 0 & x < 10"
import re

# Import type hints for better code documentation and IDE support
# Dict - dictionary type hint
# List - list type hint
# Tuple - tuple type hint (fixed-length, typed sequence)
# Optional - indicates a value can be the specified type or None
# Any - indicates any type is acceptable
from typing import Dict, List, Tuple, Optional, Any

# Suppress all warning messages globally
# This prevents pandas/numpy warnings from cluttering KNIME's console output
# Common warnings suppressed: SettingWithCopyWarning, FutureWarning, etc.
warnings.filterwarnings('ignore')

# =============================================================================
# INTERACTION TERM FUNCTIONS
# =============================================================================
# 
# These functions handle "interaction terms" - variables that represent the
# combined effect of two other variables. In credit risk modeling, interactions
# capture relationships like "high age AND high income together have a different
# risk profile than either alone."
#
# Interaction term naming convention: "var1_x_var2" (the _x_ separator)
# Examples: "Age_x_Income", "Employment_x_Housing"
# =============================================================================

def is_interaction_term(var_name: str) -> bool:
    """
    Check if a variable name represents an interaction term.
    
    This function determines whether a variable is a simple variable (like "Age")
    or an interaction between two variables (like "Age_x_Income").
    
    Interaction terms are identified by the presence of '_x_' in the name.
    This is a convention used throughout the credit risk modeling pipeline.
    
    Recognized patterns include:
    - var1_x_var2 (basic interaction)
    - var1_x_WOE_var2 (interaction with WOE prefix on second variable)
    
    Parameters:
        var_name: The variable name string to check
        
    Returns:
        True if the variable name contains '_x_' (is an interaction term)
        False if it's a regular single variable
    
    Examples:
        is_interaction_term("Age") -> False
        is_interaction_term("Age_x_Income") -> True
        is_interaction_term("WOE_Age_x_WOE_Income") -> True
    """
    # Simply check if the interaction separator '_x_' exists in the variable name
    # The 'in' operator performs a substring search
    return '_x_' in var_name


def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term into its two component variable names.
    
    Given an interaction term like "Age_x_Income", this function extracts
    the two individual variable names ("Age" and "Income").
    
    This is needed because to score an interaction, we need to:
    1. Look up the bin value for variable 1 (from b_Age column)
    2. Look up the bin value for variable 2 (from b_Income column)
    3. Combine them to match against the scorecard
    
    The function handles various naming conventions that might appear:
    - "var1_x_var2" -> ("var1", "var2")
    - "var1_x_WOE_var2" -> ("var1", "var2")  [removes WOE_ prefix]
    - "WOE_var1_x_WOE_var2" -> ("var1", "var2")  [removes both WOE_ prefixes]
    
    Parameters:
        var_name: The interaction term name to parse
        
    Returns:
        A tuple of (first_variable_name, second_variable_name)
        
    Raises:
        ValueError: If the variable name cannot be parsed as an interaction
                   (doesn't contain '_x_' separator)
    
    Examples:
        parse_interaction_term("Age_x_Income") -> ("Age", "Income")
        parse_interaction_term("WOE_Age_x_WOE_Income") -> ("Age", "Income")
    """
    # Start with the original variable name
    # We'll progressively clean it up
    clean_name = var_name
    
    # Check if the name starts with 'WOE_' prefix and remove it
    # WOE = Weight of Evidence, a common prefix in credit risk modeling
    # The [4:] slice takes everything after the first 4 characters
    if clean_name.startswith('WOE_'):
        clean_name = clean_name[4:]  # Remove 'WOE_' (4 characters)
    
    # First, try to split on the more specific pattern '_x_WOE_'
    # This handles cases like "Age_x_WOE_Income"
    # We use split with maxsplit=1 to only split on first occurrence
    if '_x_WOE_' in clean_name:
        # Split into exactly 2 parts on first occurrence of '_x_WOE_'
        parts = clean_name.split('_x_WOE_', 1)
        # parts[0] = first variable name
        # parts[1] = second variable name (WOE_ already removed by split pattern)
        return parts[0], parts[1]
    
    # If the more specific pattern wasn't found, try the basic '_x_' pattern
    # This handles cases like "Age_x_Income"
    if '_x_' in clean_name:
        # Split into exactly 2 parts on first occurrence of '_x_'
        parts = clean_name.split('_x_', 1)
        # parts[0] = first variable name
        var2 = parts[1]  # Second part might still have WOE_ prefix
        
        # Check if the second variable still has a WOE_ prefix and remove it
        if var2.startswith('WOE_'):
            var2 = var2[4:]  # Remove 'WOE_' prefix
        
        return parts[0], var2
    
    # If we get here, the variable name doesn't contain '_x_' at all
    # This means it's not a valid interaction term, so raise an error
    raise ValueError(f"Cannot parse interaction term: {var_name}")


def create_interaction_bin_label(var1: str, bin1: str, var2: str, bin2: str) -> str:
    """
    Create the combined bin label for an interaction term.
    
    When scoring an interaction term, we need to match against the scorecard's
    interaction bins. This function creates the label format that the scorecard
    generator uses.
    
    The format is: "var1:bin1 × var2:bin2"
    Note: The × is a multiplication sign (Unicode ×, not the letter x)
    
    This must EXACTLY match the format created by the scorecard generator node,
    otherwise the lookup will fail and points will be NaN.
    
    Parameters:
        var1: Name of the first variable (e.g., "Age")
        bin1: Bin value from the first variable (e.g., "[25,35)")
        var2: Name of the second variable (e.g., "Income")
        bin2: Bin value from the second variable (e.g., "[50000,75000)")
        
    Returns:
        Combined bin label string in the format "var1:bin1 × var2:bin2"
    
    Example:
        create_interaction_bin_label("Age", "[25,35)", "Income", "[50000,75000)")
        Returns: "Age:[25,35) × Income:[50000,75000)"
    """
    # Use an f-string (formatted string literal) to construct the label
    # The × symbol (Unicode multiplication sign) is used as the separator
    # This format must match what the scorecard generator produces
    return f"{var1}:{bin1} × {var2}:{bin2}"


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================
#
# These are the core functions that perform the actual scorecard application.
# They take the scorecard and data, match bin values, and calculate points.
# =============================================================================

def bin_value_num(cardx: pd.DataFrame) -> pd.DataFrame:
    """
    Parse binValue for numeric variables to extract min/max values.
    
    NOTE: This function is currently not used in the main scoring logic,
    but is kept for potential future use or debugging. It was part of an
    earlier implementation that matched numeric ranges directly.
    
    The current implementation uses direct string matching of bin labels,
    which is simpler and more robust.
    
    This function would convert bin expressions like "x >= 0 & x < 10" 
    into structured min/max columns for range-based matching.
    
    Parameters:
        cardx: DataFrame with a 'bin' column containing bin expressions
               for a single variable
        
    Returns:
        DataFrame with additional columns:
        - bins_value: cleaned version of the bin expression
        - min: minimum value of the bin range
        - max: maximum value of the bin range
        - is_na: indicator for NA/missing value bins
    """
    # Create a copy to avoid modifying the original DataFrame
    # This prevents SettingWithCopyWarning and unintended side effects
    cardx = cardx.copy()
    
    # Convert the 'bin' column to string type and store in 'bins_value'
    # This ensures we can perform string operations even if bin values are numeric
    cardx['bins_value'] = cardx['bin'].astype(str)
    
    # Define a regex pattern to remove comparison operators and parentheses
    # This pattern matches: < > = ( ) and spaces
    # We'll use this to clean up expressions like "x >= 0" to just "0"
    pattern = r'[<>=() ]'
    
    # Process each binValue string:
    # 1. Replace 'is.na' with '1' (marks NA bins)
    # 2. Replace ' & ' and ' and ' with '|' as a separator
    # 3. Remove all comparison operators and parentheses using regex
    cardx['bins_value'] = cardx['bins_value'].apply(
        lambda x: re.sub(pattern, '',  # Remove operators/parens using regex
                         x.replace('is.na', '1')  # Mark NA bins with '1'
                          .replace(' & ', '|')     # Use | as separator
                          .replace(' and ', '|'))  # Handle 'and' keyword too
    )
    
    # Split the cleaned bins_value on '|' separator to get individual values
    # expand=True returns a DataFrame with one column per split part
    # Example: "0|10|1" splits into columns [0, 1, 2] with values ["0", "10", "1"]
    split_values = cardx['bins_value'].str.split(r'\|', expand=True)
    
    # Assign the first split value as 'min' (if it exists)
    # split_values.shape[1] gives the number of columns (split parts)
    if split_values.shape[1] >= 1:
        cardx['min'] = split_values[0]  # First value is minimum
    else:
        cardx['min'] = None  # No values to split
        
    # Assign the second split value as 'max' (if it exists)
    if split_values.shape[1] >= 2:
        cardx['max'] = split_values[1]  # Second value is maximum
    else:
        cardx['max'] = None  # Only one value was present
        
    # Assign the third split value as 'is_na' indicator (if it exists)
    if split_values.shape[1] >= 3:
        cardx['is_na'] = split_values[2]  # Third value indicates NA bin
    else:
        cardx['is_na'] = None  # No NA indicator present
    
    # Special handling for the first row (lowest bin)
    # The first bin typically has no lower bound, so we set min to -99999
    if len(cardx) > 0:
        # Move the current min value to max position for the first row
        cardx.loc[cardx.index[0], 'max'] = cardx.loc[cardx.index[0], 'min']
        # Set min to a very low number (effectively -infinity)
        cardx.loc[cardx.index[0], 'min'] = -99999
    
    # Special handling for the last row (highest bin)
    # The last bin typically has no upper bound, so we set max to 99999
    if len(cardx) > 1:
        # Set max to a very high number (effectively +infinity)
        cardx.loc[cardx.index[-1], 'max'] = 99999
    
    # Return the DataFrame with the new columns added
    return cardx


def scorepoint_ply(card: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scorecard to calculate score points for each variable.
    
    This is the MAIN SCORING FUNCTION that does the core work of the node.
    It takes a scorecard and data, and produces a DataFrame with points columns.
    
    The function:
    1. Identifies all b_* (binned) columns in the input data
    2. For each variable in the scorecard:
       - Regular variables: matches b_{var} column values to scorecard bins
       - Interaction terms: combines bin values from two columns
    3. Creates {var}_points columns with the looked-up point values
    4. Adds the basescore as a constant column
    5. Calculates total Score as sum of all columns
    
    Parameters:
        card: Scorecard DataFrame containing:
              - var: variable name or 'basepoints'
              - bin: bin label (what to match against)
              - points: score points for that bin
        df: Data DataFrame containing b_* columns with bin labels
        
    Returns:
        DataFrame with columns:
        - {var}_points for each variable in the scorecard
        - basescore: the base score (constant for all rows)
        - Score: total score (sum of basescore + all points)
        
        The DataFrame preserves the original index from df (important for RowIDs)
    """
    # STEP 1: Extract only the binned columns (b_* prefix) from the input data
    # List comprehension creates a list of column names starting with 'b_'
    # Example: ['b_Age', 'b_Income', 'b_Employment']
    b_columns = [col for col in df.columns if col.startswith('b_')]
    
    # Create a copy of just the binned columns to work with
    # This prevents modifications to the original df
    dt = df[b_columns].copy()
    
    # STEP 2: Initialize the result DataFrame
    # We preserve the original index from df - this is CRITICAL for RowID preservation
    # When KNIME converts back to a KNIME table, the pandas index becomes the RowID
    # This allows users to join the scores back to their original data
    data = pd.DataFrame(index=df.index)
    
    # STEP 3: Get unique variable names from scorecard, excluding 'basepoints'
    # 'basepoints' is a special row that contains the base score constant
    # .unique() returns an array of distinct values
    xs = card[card['var'] != 'basepoints']['var'].unique()
    
    # STEP 4: Separate variables into regular and interaction categories
    # We process them differently, so it's cleaner to separate first
    regular_vars = []      # Will contain simple variables like "Age", "Income"
    interaction_vars = []  # Will contain interactions like "Age_x_Income"
    
    # Loop through each variable and categorize it
    for x_i in xs:
        if is_interaction_term(x_i):
            interaction_vars.append(x_i)
        else:
            regular_vars.append(x_i)
    
    # STEP 5: Process regular (non-interaction) variables
    # For each variable, we look up points by matching bin labels
    for x_i in regular_vars:
        # Construct the expected binned column name
        # Convention: variable "Age" has binned column "b_Age"
        b_x_i = f'b_{x_i}'
        
        # Check if the required binned column exists in the data
        # If not, skip this variable (but print a warning)
        if b_x_i not in dt.columns:
            print(f"WARNING: Column '{b_x_i}' not found in data, skipping variable '{x_i}'")
            continue  # Skip to the next variable
        
        # Get the scorecard rows for just this variable
        # Filter: keep only rows where var == current variable name
        # Select only 'bin' and 'points' columns (all we need for matching)
        # .copy() prevents SettingWithCopyWarning
        cardx = card[card['var'] == x_i][['bin', 'points']].copy()
        
        # Get the binned column values from the data
        # This will be used as the left side of the join
        dtx = dt[[b_x_i]].copy()
        
        # Prepare the output column name for points
        # Convention: variable "Age" gets points column "Age_points"
        point_col_name = f'{x_i}_points'
        
        # PERFORM THE SCORE LOOKUP via merge (join) operation
        # Left join: keep ALL rows from data, add matching points from scorecard
        # - left_on: the column in data to match on (b_Age)
        # - right_on: the column in scorecard to match on (bin)
        # - how='left': keep all data rows even if no match (will get NaN)
        merged = dtx.merge(
            cardx,          # Right side of join (scorecard)
            left_on=b_x_i,  # Match data's binned column...
            right_on='bin', # ...to scorecard's bin column
            how='left'      # Keep all data rows
        )
        
        # Extract the points values and add to result DataFrame
        # .values converts the Series to a numpy array (drops the index)
        # This ensures the values align with data's original index
        data[point_col_name] = merged['points'].values
    
    # STEP 6: Process interaction variables
    # These require combining bin values from two separate columns
    for x_i in interaction_vars:
        try:
            # Parse the interaction term to get the two component variable names
            # Example: "Age_x_Income" -> ("Age", "Income")
            var1, var2 = parse_interaction_term(x_i)
            
            # Construct the binned column names for both component variables
            b_var1 = f'b_{var1}'  # e.g., "b_Age"
            b_var2 = f'b_{var2}'  # e.g., "b_Income"
            
            # Check if BOTH required binned columns exist in the data
            # Both must be present to score an interaction
            if b_var1 not in dt.columns:
                print(f"WARNING: Column '{b_var1}' not found for interaction '{x_i}', skipping")
                continue  # Skip this interaction
            if b_var2 not in dt.columns:
                print(f"WARNING: Column '{b_var2}' not found for interaction '{x_i}', skipping")
                continue  # Skip this interaction
            
            # Get the scorecard rows for this interaction variable
            cardx = card[card['var'] == x_i][['bin', 'points']].copy()
            
            # CREATE COMBINED BIN LABELS for each row
            # We need to match against labels like "Age:[25,35) × Income:[50000,75000)"
            # Use .apply() with axis=1 to process each row
            # lambda row: ... applies the function to each row (as a Series)
            combined_bins = dt.apply(
                lambda row: create_interaction_bin_label(
                    var1,              # First variable name
                    str(row[b_var1]),  # First variable's bin value (as string)
                    var2,              # Second variable name
                    str(row[b_var2])   # Second variable's bin value (as string)
                ),
                axis=1  # axis=1 means apply function to each row
            )
            
            # Create a temporary DataFrame for the merge operation
            # Just contains the combined bin labels
            dtx = pd.DataFrame({'combined_bin': combined_bins})
            
            # Prepare the output column name
            point_col_name = f'{x_i}_points'
            
            # PERFORM THE SCORE LOOKUP via merge
            # Same logic as regular variables, but matching on combined bins
            merged = dtx.merge(
                cardx,
                left_on='combined_bin',  # Match our combined labels...
                right_on='bin',          # ...to scorecard's bin column
                how='left'               # Keep all rows
            )
            
            # Add the points to the result DataFrame
            data[point_col_name] = merged['points'].values
            
            # Print confirmation message (useful for debugging)
            print(f"Processed interaction: {x_i} ({var1} × {var2})")
            
        except Exception as e:
            # If anything goes wrong processing this interaction, log and skip it
            # This prevents one bad interaction from crashing the entire scoring
            print(f"ERROR processing interaction '{x_i}': {e}")
            continue  # Move on to the next interaction
    
    # STEP 7: Get the base score from the scorecard
    # The basepoints row contains a constant score added to everyone
    # This shifts the entire score distribution up or down
    basepoints_row = card[card['var'] == 'basepoints']
    
    # Check if basepoints row exists
    if not basepoints_row.empty:
        # Get the points value from the first (and should be only) basepoints row
        # .iloc[0] gets the first row's value
        basescore = basepoints_row['points'].iloc[0]
    else:
        # No basepoints found - use 0 as default (and warn)
        basescore = 0
        print("WARNING: No basepoints found in scorecard, using 0")
    
    # Add basescore as a constant column (same value for all rows)
    data['basescore'] = basescore
    
    # STEP 8: Calculate the total Score
    # Sum ALL columns (all {var}_points columns + basescore)
    # axis=1 means sum across columns (horizontally) for each row
    data['Score'] = data.sum(axis=1)
    
    # Return the completed points DataFrame
    return data


def scorepoint_ply_with_data(
    card: pd.DataFrame, 
    df: pd.DataFrame, 
    with_org_data: bool = True
) -> pd.DataFrame:
    """
    Apply scorecard and optionally append original data.
    
    This is a wrapper around scorepoint_ply() that adds the option to
    include the original data columns alongside the score points.
    
    This is useful when you want to:
    - Analyze scores alongside the original features
    - Export a complete dataset with scores
    - Debug by comparing bin values with points
    
    Parameters:
        card: Scorecard DataFrame with var, bin, points columns
              (passed through to scorepoint_ply)
        df: Data DataFrame with b_* binned columns
            (passed through to scorepoint_ply)
        with_org_data: Boolean flag controlling output format
                      True: Combine original data + score columns (default)
                      False: Return only score columns
        
    Returns:
        If with_org_data is True:
            DataFrame with all original columns first, then all score columns
        If with_org_data is False:
            DataFrame with only score columns (same as scorepoint_ply output)
    """
    # First, apply the scorecard to get the points DataFrame
    # This contains: {var}_points columns, basescore, and Score
    data = scorepoint_ply(card, df)
    
    # Check if we should append the original data
    if with_org_data:
        # Combine the original data with the score columns
        # pd.concat with axis=1 joins DataFrames side-by-side (column-wise)
        # The order [df, data] means original columns come first
        # Both DataFrames must have the same index for proper alignment
        # (This is why we preserved df.index in scorepoint_ply)
        data = pd.concat([df, data], axis=1)
    
    # Return the combined (or unchanged) DataFrame
    return data


def validate_scorecard(card: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that scorecard has the required structure.
    
    Before attempting to score, we check that the scorecard is properly
    formatted. This prevents cryptic errors later and gives clear feedback.
    
    Validation checks:
    1. All required columns exist: 'var', 'bin', 'points'
    2. Scorecard is not empty
    3. There is a 'basepoints' row (required for scoring formula)
    
    Parameters:
        card: The scorecard DataFrame to validate
        
    Returns:
        A tuple of (is_valid, error_message):
        - is_valid: True if scorecard passes all checks, False otherwise
        - error_message: Empty string if valid, description of problem if invalid
        
    Example:
        is_valid, error_msg = validate_scorecard(card)
        if not is_valid:
            raise ValueError(error_msg)
    """
    # Define the columns that must be present in the scorecard
    required_cols = ['var', 'bin', 'points']
    
    # Check for each required column
    for col in required_cols:
        if col not in card.columns:
            # Return failure with specific message about which column is missing
            return False, f"Scorecard missing required column: '{col}'"
    
    # Check that scorecard is not empty
    if card.empty:
        return False, "Scorecard is empty"
    
    # Check that basepoints row exists
    # The 'var' column should contain at least one row with value 'basepoints'
    if 'basepoints' not in card['var'].values:
        return False, "Scorecard missing 'basepoints' row"
    
    # All checks passed - return success with empty error message
    return True, ""


def validate_data(df: pd.DataFrame, card: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that data has the required binned columns for scoring.
    
    This function checks whether the input data has the necessary b_*
    columns to score against the provided scorecard. It's a pre-flight
    check that helps catch data mismatches before scoring fails.
    
    For regular variables:
        - Checks if b_{var} column exists
    
    For interaction terms:
        - Checks if BOTH component columns exist (b_{var1} and b_{var2})
        - If either is missing, the interaction cannot be scored
    
    Parameters:
        df: The data DataFrame to validate (should have b_* columns)
        card: The scorecard DataFrame (defines which columns are needed)
        
    Returns:
        A tuple of (has_any_columns, list_of_missing_columns):
        - has_any_columns: True if at least one required column exists
        - list_of_missing_columns: List of column names that are missing
                                   (empty list if all columns present)
        
    Note:
        The function returns True (has_any) even if some columns are missing,
        as long as at least one can be scored. This allows partial scoring.
    """
    # Get the list of variables from scorecard (excluding basepoints)
    # These are the variables we need to find columns for
    required_vars = card[card['var'] != 'basepoints']['var'].unique()
    
    # Initialize tracking variables
    missing = []     # List of missing column descriptions
    found_count = 0  # Count of variables we CAN score
    
    # Check each required variable
    for var in required_vars:
        # Handle interaction terms differently from regular variables
        if is_interaction_term(var):
            # For interactions, we need BOTH component columns
            try:
                # Parse the interaction to get component variable names
                var1, var2 = parse_interaction_term(var)
                b_col1 = f'b_{var1}'  # First component column
                b_col2 = f'b_{var2}'  # Second component column
                
                # Check which component columns are missing (if any)
                missing_components = []
                if b_col1 not in df.columns:
                    missing_components.append(b_col1)
                if b_col2 not in df.columns:
                    missing_components.append(b_col2)
                
                # If any components are missing, record the interaction as problematic
                if missing_components:
                    # Create a descriptive message about what's needed
                    missing.append(f"{var} (needs: {', '.join(missing_components)})")
                else:
                    # Both components found - we can score this interaction
                    found_count += 1
                    
            except ValueError:
                # Couldn't parse the interaction term (shouldn't happen normally)
                missing.append(f"b_{var} (unparseable interaction)")
        else:
            # Regular variable - just need the single b_{var} column
            b_col = f'b_{var}'
            
            if b_col not in df.columns:
                # Column not found - add to missing list
                missing.append(b_col)
            else:
                # Column found - we can score this variable
                found_count += 1
    
    # Determine if we have at least one scoreable variable
    # We can proceed with partial scoring if some columns exist
    has_any = found_count > 0
    
    return has_any, missing


# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================
#
# This is where the actual script execution begins when run in KNIME.
# Everything above defined functions; here we use them to process data.
# =============================================================================

# =============================================================================
# Read Input Data
# =============================================================================

# Print startup banner to KNIME console
# This helps users confirm the node is running and see progress
print("Scorecard Apply Node - Starting...")
print("=" * 70)  # Print a line of 70 '=' characters as a visual separator

# INPUT 1: Read the scorecard from the first input port (index 0)
# knio.input_tables[0] returns a KNIME table object
# .to_pandas() converts it to a pandas DataFrame for processing
# The scorecard contains: var, bin, points columns
card = knio.input_tables[0].to_pandas()

# Print info about what we received
# len(card) gives the number of rows in the DataFrame
print(f"Input 1 (Scorecard): {len(card)} rows")

# INPUT 2: Read the data to be scored from the second input port (index 1)
# This should contain b_* columns from the WOE Editor node
df = knio.input_tables[1].to_pandas()

# Print data dimensions
# len(df) = number of rows, len(df.columns) = number of columns
print(f"Input 2 (Data): {len(df)} rows, {len(df.columns)} columns")

# DEBUG OUTPUT: Show scorecard variables
# This helps users verify the correct scorecard is connected
# We separate regular variables from interactions for clarity
regular_vars_list = []      # Will hold names of simple variables
interaction_vars_list = []  # Will hold names of interaction terms

# Loop through unique variable names in the scorecard
for var in card['var'].unique():
    # Skip the basepoints row (it's not a variable)
    if var != 'basepoints':
        # Categorize as interaction or regular
        if is_interaction_term(var):
            interaction_vars_list.append(var)
        else:
            regular_vars_list.append(var)

# Print summary counts
print(f"\nScorecard variables: {len(regular_vars_list)} regular, {len(interaction_vars_list)} interactions")

# Print details of regular variables
print("\nRegular variables:")
for var in regular_vars_list:
    # Count how many bins (rows) exist for this variable
    var_rows = len(card[card['var'] == var])
    print(f"  - {var} ({var_rows} bins)")

# Print details of interaction variables (if any exist)
if interaction_vars_list:
    print("\nInteraction variables:")
    for var in interaction_vars_list:
        # Count bins for this interaction
        var_rows = len(card[card['var'] == var])
        try:
            # Parse and display the component variables
            var1, var2 = parse_interaction_term(var)
            print(f"  - {var} = {var1} × {var2} ({var_rows} bins)")
        except:
            # If parsing fails, just show the name
            print(f"  - {var} ({var_rows} bins)")

# Print the base points value
basepoints_row = card[card['var'] == 'basepoints']
if not basepoints_row.empty:
    # .iloc[0] gets the first row's 'points' value
    print(f"\nBase points: {basepoints_row['points'].iloc[0]}")

# DEBUG OUTPUT: Show available b_* columns in data
# This helps users verify the correct data is connected
b_columns = [col for col in df.columns if col.startswith('b_')]
print(f"\nBinned columns in data: {len(b_columns)}")

# Print the first 10 binned column names (don't overwhelm the console)
for col in b_columns[:10]:  # Slice to first 10 elements
    print(f"  - {col}")

# If there are more than 10, indicate how many are hidden
if len(b_columns) > 10:
    print(f"  ... and {len(b_columns) - 10} more")

# =============================================================================
# Check for Flow Variables
# =============================================================================
#
# Flow variables allow KNIME workflow configuration to control script behavior.
# Here we check for the "WithOriginalData" flow variable.
# =============================================================================

# Set default value for the with_original_data flag
# True = include original data columns in output 2
with_original_data = True

# Try to read the flow variable from KNIME
# We use try/except because the flow variable might not exist
try:
    # .get() returns the value if it exists, or the default (None) if not
    with_org_fv = knio.flow_variables.get("WithOriginalData", None)
    
    # If the flow variable was found, use its value
    if with_org_fv is not None:
        # Convert to boolean (handles various input types)
        with_original_data = bool(with_org_fv)
except:
    # If anything goes wrong (missing, wrong type, etc.), use the default
    # The 'pass' statement means "do nothing" - we already have the default
    pass

# Print the configuration for confirmation
print(f"\nWith original data: {with_original_data}")
print("=" * 70)  # Visual separator

# =============================================================================
# Validation
# =============================================================================
#
# Before processing, we validate that both inputs are properly formatted.
# This catches problems early with clear error messages.
# =============================================================================

# VALIDATE SCORECARD STRUCTURE
# Check that scorecard has required columns and basepoints
is_valid, error_msg = validate_scorecard(card)

if not is_valid:
    # Scorecard is invalid - print error and raise exception
    # raise ValueError(...) stops execution and shows error in KNIME
    print(f"ERROR: {error_msg}")
    raise ValueError(error_msg)

# VALIDATE DATA HAS REQUIRED COLUMNS
# Check that data has at least some of the required b_* columns
has_columns, missing_cols = validate_data(df, card)

# Print warnings about any missing columns
# We don't fail on missing columns - just warn and skip them during scoring
if missing_cols:
    print(f"WARNING: Missing binned columns: {missing_cols}")

# If NO matching columns exist, we can't score anything - fail
if not has_columns:
    print("ERROR: No matching binned columns found in data")
    raise ValueError("Data does not have any matching b_* columns for scorecard variables")

# =============================================================================
# Apply Scorecard
# =============================================================================
#
# Now we do the actual scoring work.
# We create two outputs:
# 1. Just the points columns (lightweight)
# 2. Full data with points (complete dataset)
# =============================================================================

print("\nApplying scorecard...")

# OUTPUT 1: Calculate points only (no original data columns)
# This is a lightweight output useful for quick analysis or joins
df_points = scorepoint_ply(card, df)

# Print the dimensions of the output
print(f"Points-only output: {len(df_points)} rows, {len(df_points.columns)} columns")

# OUTPUT 2: Calculate points with original data appended
# This provides a complete dataset with all original columns plus scores
df_points_dat = scorepoint_ply_with_data(card, df, with_org_data=with_original_data)

# Print the dimensions of this larger output
print(f"Full output: {len(df_points_dat)} rows, {len(df_points_dat.columns)} columns")

# =============================================================================
# Output Tables
# =============================================================================
#
# Write the results to KNIME output ports.
# KNIME nodes can have multiple output ports; this node has 2.
# =============================================================================

# OUTPUT PORT 0: Points-only DataFrame
# knio.Table.from_pandas() converts a pandas DataFrame to a KNIME table
# This output contains: {var}_points, basescore, Score
knio.output_tables[0] = knio.Table.from_pandas(df_points)

# OUTPUT PORT 1: Full data with scores
# This output contains: all original columns + all score columns
knio.output_tables[1] = knio.Table.from_pandas(df_points_dat)

# =============================================================================
# Print Summary
# =============================================================================
#
# After processing, we print useful statistics about the results.
# This helps users quickly verify the scoring worked correctly.
# =============================================================================

print("=" * 70)
print("Scorecard Apply completed successfully")
print("=" * 70)

# Calculate and display score statistics
# Only proceed if we have data and a Score column
if not df_points.empty and 'Score' in df_points.columns:
    # Extract the Score column as a Series for statistical calculations
    scores = df_points['Score']
    
    # Print basic descriptive statistics
    print(f"\nScore Statistics:")
    print(f"  Count:    {len(scores)}")              # Total number of scores
    print(f"  Min:      {scores.min():.0f}")         # Lowest score (no decimals)
    print(f"  Max:      {scores.max():.0f}")         # Highest score (no decimals)
    print(f"  Mean:     {scores.mean():.1f}")        # Average score (1 decimal)
    print(f"  Median:   {scores.median():.0f}")      # Middle score (no decimals)
    print(f"  Std Dev:  {scores.std():.1f}")         # Standard deviation (1 decimal)
    
    # Create and display a score distribution histogram
    # This shows how scores are distributed across ranges
    print(f"\nScore Distribution:")
    
    # Get the score range
    score_min = int(scores.min())  # Convert to int for clean bucket boundaries
    score_max = int(scores.max())
    
    # Determine bucket size
    # Use at least 50 points per bucket, or divide range into ~10 buckets
    bucket_size = max(50, (score_max - score_min) // 10)
    
    # Create bucket boundaries
    # Start from a clean multiple of bucket_size below the minimum
    # End after the maximum
    buckets = range(
        score_min - (score_min % bucket_size),  # Round down to clean start
        score_max + bucket_size,                 # Ensure we include max
        bucket_size                              # Step by bucket size
    )
    
    # Print each bucket with count and visual bar
    # list(buckets)[:-1] excludes the last boundary (it's just the endpoint)
    for i, bucket_start in enumerate(list(buckets)[:-1]):
        bucket_end = bucket_start + bucket_size
        
        # Count scores in this bucket (>= start and < end)
        count = ((scores >= bucket_start) & (scores < bucket_end)).sum()
        
        # Calculate percentage
        pct = count / len(scores) * 100
        
        # Create a visual bar using '#' characters
        # Each '#' represents 2 percentage points
        bar = '#' * int(pct / 2)
        
        # Print the bucket info with formatting
        # :4d = integer padded to 4 characters
        # :5d = integer padded to 5 characters
        # :5.1f = float with 1 decimal padded to 5 characters
        print(f"  {bucket_start:4d}-{bucket_end:4d}: {count:5d} ({pct:5.1f}%) {bar}")

# List all the points columns that were created
# This confirms which variables were successfully scored
points_cols = [col for col in df_points.columns if col.endswith('_points')]

print(f"\nPoints columns created: {len(points_cols)}")
for col in points_cols:
    print(f"  - {col}")

# Final summary of output sizes
print(f"\nOutput 1 (Points only): {len(df_points)} rows")
print(f"Output 2 (With original data): {len(df_points_dat)} rows")
print("=" * 70)

# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# At this point, the script has:
# 1. Read the scorecard and data from KNIME input ports
# 2. Validated both inputs for proper structure
# 3. Applied the scorecard to calculate points for each variable
# 4. Handled both regular variables and interaction terms
# 5. Calculated total scores as sum of all points + basescore
# 6. Written results to both KNIME output ports
# 7. Printed comprehensive statistics and diagnostics
#
# The user can now:
# - Use Output 1 (points only) for lightweight analysis
# - Use Output 2 (with original data) for complete datasets
# - Join scores back to original data using preserved RowIDs
# =============================================================================

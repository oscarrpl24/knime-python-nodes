# =============================================================================
# Scorecard Apply (Scoring) for KNIME Python Script Node - TOGGLE DEBUG VERSION
# =============================================================================
# Python implementation of R's scorecard scoring functionality
# Applies a scorecard to data with binned columns to calculate credit scores
# Compatible with KNIME 5.9, Python 3.9
#
# TOGGLE DEBUG VERSION: Debug logging can be enabled/disabled via DEBUG_MODE flag
# Set DEBUG_MODE = True to enable extensive logging, False for production use.
#
# Inputs:
# 1. Scorecard table (from Scorecard Generator node)
#    - var: variable name (or interaction name like "var1_x_var2")
#    - bin: bin label (used for matching with b_* columns)
#    - points: score points for each bin
# 2. Data table with binned columns (b_* prefix)
#    - Connect WOE Editor Port 4 (df_only_bins - lean output with ONLY b_* columns)
#    - Alternative: WOE Editor Port 2 also works (includes original + WOE columns)
#
# Outputs:
# 1. Points only - columns: {var}_points for each variable + basescore + Score
# 2. Full data with scores - original data + all points columns + Score
#
# Flow Variables (optional):
# - WithOriginalData (boolean, default True): Include original data in output 2
#
# Scoring Logic:
#   For regular variables:
#     - Match b_{var} column values with scorecard bin labels
#     - Assign corresponding points to {var}_points column
#   For interaction terms (var1_x_var2):
#     - Combine bin values from b_{var1} and b_{var2}
#     - Create combined label: "var1:bin1 × var2:bin2"
#     - Match against scorecard interaction bins
#     - Assign corresponding points to {var1_x_var2}_points column
#   Total Score = basescore + sum of all {var}_points
#
# Release Date: 2026-01-28
# Version: 1.2-TOGGLE
# 
# Version History:
# 1.2-TOGGLE (2026-01-28): Toggle debug version with DEBUG_MODE flag
# 1.2-DEBUG (2026-01-28): Debug version with extensive logging
# 1.2 (2026-01-28): Fixed RowID preservation - output tables now maintain original
#                   RowIDs from input table, enabling joins back to source data
# 1.1: Interaction term support added
# =============================================================================

# =============================================================================
# DEBUG MODE TOGGLE
# =============================================================================
# Set to True to enable extensive debug logging
# Set to False for production use (minimal console output)
DEBUG_MODE = False
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import re
import logging
import traceback
import time
import functools
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# DEBUG LOGGING SETUP (Conditional)
# =============================================================================

# Configure logging based on DEBUG_MODE
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('ScorecardApply_TOGGLE')
    logger.setLevel(logging.DEBUG)
else:
    # Create a null logger that does nothing
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger('ScorecardApply_TOGGLE')
    logger.setLevel(logging.CRITICAL)


def debug_print(*args, **kwargs):
    """Print only if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)


def log_function_call(func):
    """
    Decorator that logs function entry, exit, parameters, return values, and timing.
    Only performs logging if DEBUG_MODE is True.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If debug mode is off, just run the function without any logging
        if not DEBUG_MODE:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        
        # Log function entry with parameters
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        
        # Log positional arguments
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                logger.debug(f"  arg[{i}] (DataFrame): shape={arg.shape}, columns={list(arg.columns)[:10]}{'...' if len(arg.columns) > 10 else ''}")
                logger.debug(f"    dtypes: {dict(list(arg.dtypes.items())[:5])}{'...' if len(arg.dtypes) > 5 else ''}")
                if not arg.empty:
                    logger.debug(f"    head(3):\n{arg.head(3).to_string()}")
            elif isinstance(arg, (list, tuple)):
                logger.debug(f"  arg[{i}] ({type(arg).__name__}): len={len(arg)}, values={arg[:5]}{'...' if len(arg) > 5 else ''}")
            elif isinstance(arg, dict):
                logger.debug(f"  arg[{i}] (dict): keys={list(arg.keys())[:5]}{'...' if len(arg) > 5 else ''}")
            else:
                logger.debug(f"  arg[{i}] ({type(arg).__name__}): {repr(arg)[:200]}")
        
        # Log keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                logger.debug(f"  {key} (DataFrame): shape={value.shape}")
            else:
                logger.debug(f"  {key}: {repr(value)[:200]}")
        
        # Execute function with timing
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            
            # Log return value
            if isinstance(result, pd.DataFrame):
                logger.debug(f"EXITING: {func_name} (elapsed: {elapsed_time:.4f}s)")
                logger.debug(f"  return (DataFrame): shape={result.shape}, columns={list(result.columns)[:10]}{'...' if len(result.columns) > 10 else ''}")
                if not result.empty:
                    logger.debug(f"    head(3):\n{result.head(3).to_string()}")
            elif isinstance(result, tuple):
                logger.debug(f"EXITING: {func_name} (elapsed: {elapsed_time:.4f}s)")
                logger.debug(f"  return (tuple): len={len(result)}, values={result}")
            elif isinstance(result, (list, dict)):
                logger.debug(f"EXITING: {func_name} (elapsed: {elapsed_time:.4f}s)")
                logger.debug(f"  return ({type(result).__name__}): {repr(result)[:300]}")
            else:
                logger.debug(f"EXITING: {func_name} (elapsed: {elapsed_time:.4f}s)")
                logger.debug(f"  return: {repr(result)[:200]}")
            
            logger.debug(f"{'='*60}")
            return result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"EXCEPTION in {func_name} (elapsed: {elapsed_time:.4f}s)")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {str(e)}")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            logger.debug(f"{'='*60}")
            raise
    
    return wrapper


# =============================================================================
# Interaction Term Functions
# =============================================================================

@log_function_call
def is_interaction_term(var_name: str) -> bool:
    """
    Check if a variable name represents an interaction term.
    
    Interaction terms have patterns like:
    - var1_x_var2
    - var1_x_WOE_var2
    
    Parameters:
        var_name: Variable name to check
        
    Returns:
        True if this is an interaction term
    """
    if DEBUG_MODE:
        logger.debug(f"Checking if '{var_name}' contains '_x_'")
    result = '_x_' in var_name
    if DEBUG_MODE:
        logger.debug(f"  Contains '_x_': {result}")
    return result


@log_function_call
def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term into its two component variable names.
    
    Input formats:
        - "var1_x_var2" -> ("var1", "var2")
        - "var1_x_WOE_var2" -> ("var1", "var2")
        - "WOE_var1_x_WOE_var2" -> ("var1", "var2")
    
    Parameters:
        var_name: Interaction term name
        
    Returns:
        Tuple of (var1_name, var2_name)
    """
    if DEBUG_MODE:
        logger.debug(f"Parsing interaction term: '{var_name}'")
    
    # Remove leading WOE_ if present
    clean_name = var_name
    if clean_name.startswith('WOE_'):
        clean_name = clean_name[4:]  # Remove 'WOE_'
        if DEBUG_MODE:
            logger.debug(f"  Removed leading 'WOE_': '{clean_name}'")
    
    # Split on '_x_WOE_' first (most specific pattern)
    if '_x_WOE_' in clean_name:
        parts = clean_name.split('_x_WOE_', 1)
        if DEBUG_MODE:
            logger.debug(f"  Split on '_x_WOE_': {parts}")
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_'
    if '_x_' in clean_name:
        parts = clean_name.split('_x_', 1)
        if DEBUG_MODE:
            logger.debug(f"  Split on '_x_': {parts}")
        var2 = parts[1]
        # Remove WOE_ prefix from var2 if present
        if var2.startswith('WOE_'):
            var2 = var2[4:]
            if DEBUG_MODE:
                logger.debug(f"  Removed 'WOE_' from var2: '{var2}'")
        return parts[0], var2
    
    logger.error(f"Cannot parse interaction term: {var_name}")
    raise ValueError(f"Cannot parse interaction term: {var_name}")


@log_function_call
def create_interaction_bin_label(var1: str, bin1: str, var2: str, bin2: str) -> str:
    """
    Create the combined bin label for an interaction term.
    
    Must match the format created by scorecard generator:
    "var1:bin1 × var2:bin2"
    
    Parameters:
        var1, bin1: First variable name and bin value
        var2, bin2: Second variable name and bin value
        
    Returns:
        Combined bin label string
    """
    if DEBUG_MODE:
        logger.debug(f"Creating interaction bin label: var1='{var1}', bin1='{bin1}', var2='{var2}', bin2='{bin2}'")
    result = f"{var1}:{bin1} × {var2}:{bin2}"
    if DEBUG_MODE:
        logger.debug(f"  Created label: '{result}'")
    return result


# =============================================================================
# Scoring Functions
# =============================================================================

@log_function_call
def bin_value_num(cardx: pd.DataFrame) -> pd.DataFrame:
    """
    Parse binValue for numeric variables to extract min/max values.
    
    Converts binValue expressions like "x >= 0 & x < 10" into min/max columns.
    Handles is.na() expressions by marking is_na column.
    
    Parameters:
        cardx: DataFrame with binValue column for a single variable
        
    Returns:
        DataFrame with added bins_value, min, max, is_na columns
    """
    if DEBUG_MODE:
        logger.debug(f"bin_value_num called with cardx shape: {cardx.shape}")
        logger.debug(f"  Input columns: {list(cardx.columns)}")
    
    cardx = cardx.copy()
    cardx['bins_value'] = cardx['bin'].astype(str)
    if DEBUG_MODE:
        logger.debug(f"  bins_value column created from 'bin': {cardx['bins_value'].tolist()[:5]}")
    
    # Pattern to remove: < > = ( ) and spaces
    pattern = r'[<>=() ]'
    
    # Process each binValue
    cardx['bins_value'] = cardx['bins_value'].apply(
        lambda x: re.sub(pattern, '', 
                         x.replace('is.na', '1').replace(' & ', '|').replace(' and ', '|'))
    )
    if DEBUG_MODE:
        logger.debug(f"  bins_value after regex cleaning: {cardx['bins_value'].tolist()[:5]}")
    
    # Split into min, max, is_na on | separator
    split_values = cardx['bins_value'].str.split(r'\|', expand=True)
    if DEBUG_MODE:
        logger.debug(f"  Split values shape: {split_values.shape}")
    
    # Handle varying number of splits
    if split_values.shape[1] >= 1:
        cardx['min'] = split_values[0]
    else:
        cardx['min'] = None
        
    if split_values.shape[1] >= 2:
        cardx['max'] = split_values[1]
    else:
        cardx['max'] = None
        
    if split_values.shape[1] >= 3:
        cardx['is_na'] = split_values[2]
    else:
        cardx['is_na'] = None
    
    if DEBUG_MODE:
        logger.debug(f"  min values: {cardx['min'].tolist()[:5]}")
        logger.debug(f"  max values: {cardx['max'].tolist()[:5]}")
    
    # Adjust first and last row boundaries
    if len(cardx) > 0:
        if DEBUG_MODE:
            old_first_max = cardx.loc[cardx.index[0], 'max']
            old_first_min = cardx.loc[cardx.index[0], 'min']
        cardx.loc[cardx.index[0], 'max'] = cardx.loc[cardx.index[0], 'min']
        cardx.loc[cardx.index[0], 'min'] = -99999
        if DEBUG_MODE:
            logger.debug(f"  First row adjusted: min {old_first_min}->{-99999}, max {old_first_max}->{cardx.loc[cardx.index[0], 'max']}")
        
    if len(cardx) > 1:
        if DEBUG_MODE:
            old_last_max = cardx.loc[cardx.index[-1], 'max']
        cardx.loc[cardx.index[-1], 'max'] = 99999
        if DEBUG_MODE:
            logger.debug(f"  Last row adjusted: max {old_last_max}->99999")
    
    if DEBUG_MODE:
        logger.debug(f"  Final cardx shape: {cardx.shape}")
    return cardx


@log_function_call
def scorepoint_ply(card: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scorecard to calculate score points for each variable.
    
    Main scoring function that:
    1. Filters data to only b_* columns (binned columns)
    2. For each variable in scorecard, joins with bin labels to get points
    3. Handles interaction terms by combining bin values from component variables
    4. Creates {var}_points columns
    5. Calculates total Score as sum of all points + basescore
    
    Parameters:
        card: Scorecard DataFrame with var, bin, points columns
        df: Data DataFrame with b_* binned columns
        
    Returns:
        DataFrame with {var}_points columns, basescore, and Score
    """
    if DEBUG_MODE:
        logger.debug(f"scorepoint_ply called")
        logger.debug(f"  card shape: {card.shape}, columns: {list(card.columns)}")
        logger.debug(f"  df shape: {df.shape}, columns: {list(df.columns)[:10]}...")
    
    # Get only b_* columns from the data
    b_columns = [col for col in df.columns if col.startswith('b_')]
    if DEBUG_MODE:
        logger.debug(f"  Found {len(b_columns)} b_* columns: {b_columns[:10]}...")
    
    dt = df[b_columns].copy()
    if DEBUG_MODE:
        logger.debug(f"  dt (binned columns only) shape: {dt.shape}")
    
    # Initialize result DataFrame - preserve the original index for RowID consistency
    data = pd.DataFrame(index=df.index)
    if DEBUG_MODE:
        logger.debug(f"  Initialized result DataFrame with index length: {len(data.index)}")
    
    # Get unique variables from scorecard (excluding basepoints)
    xs = card[card['var'] != 'basepoints']['var'].unique()
    if DEBUG_MODE:
        logger.debug(f"  Found {len(xs)} unique variables in scorecard (excluding basepoints)")
        logger.debug(f"  Variables: {list(xs)}")
    
    # Separate regular variables from interaction terms
    regular_vars = []
    interaction_vars = []
    
    for x_i in xs:
        if is_interaction_term(x_i):
            interaction_vars.append(x_i)
        else:
            regular_vars.append(x_i)
    
    if DEBUG_MODE:
        logger.debug(f"  Regular variables ({len(regular_vars)}): {regular_vars}")
        logger.debug(f"  Interaction variables ({len(interaction_vars)}): {interaction_vars}")
    
    # Process regular variables
    for x_i in regular_vars:
        if DEBUG_MODE:
            logger.debug(f"  Processing regular variable: '{x_i}'")
        b_x_i = f'b_{x_i}'
        
        # Check if binned column exists in data
        if b_x_i not in dt.columns:
            if DEBUG_MODE:
                logger.warning(f"    Column '{b_x_i}' not found in data, skipping variable '{x_i}'")
            print(f"WARNING: Column '{b_x_i}' not found in data, skipping variable '{x_i}'")
            continue
        
        # Get scorecard rows for this variable
        cardx = card[card['var'] == x_i][['bin', 'points']].copy()
        if DEBUG_MODE:
            logger.debug(f"    Scorecard rows for '{x_i}': {len(cardx)} rows")
            logger.debug(f"    Bins: {cardx['bin'].tolist()}")
            logger.debug(f"    Points: {cardx['points'].tolist()}")
        
        # Get binned column from data
        dtx = dt[[b_x_i]].copy()
        if DEBUG_MODE:
            logger.debug(f"    Data binned column unique values: {dtx[b_x_i].unique()[:10]}...")
        
        # Join to get points: match b_{var} values with scorecard bin labels
        point_col_name = f'{x_i}_points'
        
        # Merge on bin value
        if DEBUG_MODE:
            logger.debug(f"    Merging on left='{b_x_i}', right='bin'")
        merged = dtx.merge(
            cardx,
            left_on=b_x_i,
            right_on='bin',
            how='left'
        )
        if DEBUG_MODE:
            logger.debug(f"    Merged shape: {merged.shape}")
            logger.debug(f"    Null points after merge: {merged['points'].isna().sum()}")
        
        # Add points column to result
        data[point_col_name] = merged['points'].values
        if DEBUG_MODE:
            logger.debug(f"    Added column '{point_col_name}' with {len(merged['points'].values)} values")
            logger.debug(f"    Points stats: min={merged['points'].min()}, max={merged['points'].max()}, mean={merged['points'].mean():.2f}")
    
    # Process interaction variables
    for x_i in interaction_vars:
        if DEBUG_MODE:
            logger.debug(f"  Processing interaction variable: '{x_i}'")
        try:
            # Parse interaction term to get component variables
            var1, var2 = parse_interaction_term(x_i)
            b_var1 = f'b_{var1}'
            b_var2 = f'b_{var2}'
            if DEBUG_MODE:
                logger.debug(f"    Parsed components: var1='{var1}' (col='{b_var1}'), var2='{var2}' (col='{b_var2}')")
            
            # Check if both binned columns exist
            if b_var1 not in dt.columns:
                if DEBUG_MODE:
                    logger.warning(f"    Column '{b_var1}' not found for interaction '{x_i}', skipping")
                print(f"WARNING: Column '{b_var1}' not found for interaction '{x_i}', skipping")
                continue
            if b_var2 not in dt.columns:
                if DEBUG_MODE:
                    logger.warning(f"    Column '{b_var2}' not found for interaction '{x_i}', skipping")
                print(f"WARNING: Column '{b_var2}' not found for interaction '{x_i}', skipping")
                continue
            
            # Get scorecard rows for this interaction
            cardx = card[card['var'] == x_i][['bin', 'points']].copy()
            if DEBUG_MODE:
                logger.debug(f"    Scorecard rows for interaction: {len(cardx)}")
                logger.debug(f"    Interaction bins: {cardx['bin'].tolist()[:5]}...")
            
            # Create combined bin labels for each row in the data
            if DEBUG_MODE:
                logger.debug(f"    Creating combined bin labels...")
            combined_bins = dt.apply(
                lambda row: create_interaction_bin_label(var1, str(row[b_var1]), var2, str(row[b_var2])),
                axis=1
            )
            if DEBUG_MODE:
                logger.debug(f"    Created {len(combined_bins)} combined bin labels")
                logger.debug(f"    Sample combined bins: {combined_bins.head().tolist()}")
            
            # Create temp DataFrame for merging
            dtx = pd.DataFrame({'combined_bin': combined_bins})
            
            # Merge to get points
            point_col_name = f'{x_i}_points'
            
            merged = dtx.merge(
                cardx,
                left_on='combined_bin',
                right_on='bin',
                how='left'
            )
            if DEBUG_MODE:
                logger.debug(f"    Merged shape: {merged.shape}")
                logger.debug(f"    Null points after merge: {merged['points'].isna().sum()}")
            
            # Add points column to result
            data[point_col_name] = merged['points'].values
            
            if DEBUG_MODE:
                logger.info(f"    Processed interaction: {x_i} ({var1} × {var2})")
            print(f"Processed interaction: {x_i} ({var1} × {var2})")
            
        except Exception as e:
            logger.error(f"    ERROR processing interaction '{x_i}': {e}")
            if DEBUG_MODE:
                logger.error(f"    Traceback:\n{traceback.format_exc()}")
            print(f"ERROR processing interaction '{x_i}': {e}")
            continue
    
    # Get basescore
    if DEBUG_MODE:
        logger.debug(f"  Looking for basepoints row...")
    basepoints_row = card[card['var'] == 'basepoints']
    if not basepoints_row.empty:
        basescore = basepoints_row['points'].iloc[0]
        if DEBUG_MODE:
            logger.debug(f"    Found basescore: {basescore}")
    else:
        basescore = 0
        if DEBUG_MODE:
            logger.warning(f"    No basepoints found in scorecard, using 0")
        print("WARNING: No basepoints found in scorecard, using 0")
    
    # Add basescore column
    data['basescore'] = basescore
    if DEBUG_MODE:
        logger.debug(f"  Added basescore column with value: {basescore}")
    
    # Calculate total score: sum of all columns (including basescore)
    if DEBUG_MODE:
        logger.debug(f"  Calculating total Score...")
        logger.debug(f"  Columns to sum: {list(data.columns)}")
    data['Score'] = data.sum(axis=1)
    if DEBUG_MODE:
        logger.debug(f"  Score stats: min={data['Score'].min()}, max={data['Score'].max()}, mean={data['Score'].mean():.2f}")
    
    if DEBUG_MODE:
        logger.debug(f"  Final result shape: {data.shape}")
        logger.debug(f"  Final columns: {list(data.columns)}")
    return data


@log_function_call
def scorepoint_ply_with_data(
    card: pd.DataFrame, 
    df: pd.DataFrame, 
    with_org_data: bool = True
) -> pd.DataFrame:
    """
    Apply scorecard and optionally append original data.
    
    Parameters:
        card: Scorecard DataFrame with var, bin, points columns
        df: Data DataFrame with b_* binned columns
        with_org_data: If True, prepend original data columns to result
        
    Returns:
        DataFrame with score points (and optionally original data)
    """
    if DEBUG_MODE:
        logger.debug(f"scorepoint_ply_with_data called")
        logger.debug(f"  card shape: {card.shape}")
        logger.debug(f"  df shape: {df.shape}")
        logger.debug(f"  with_org_data: {with_org_data}")
    
    # Apply score points
    data = scorepoint_ply(card, df)
    if DEBUG_MODE:
        logger.debug(f"  Score points calculated, shape: {data.shape}")
    
    # Append original dataset if requested
    if with_org_data:
        if DEBUG_MODE:
            logger.debug(f"  Concatenating original data with score points...")
        # Combine: original data first, then score columns
        # Preserve original index (RowIDs) for joining back to source data
        data = pd.concat([df, data], axis=1)
        if DEBUG_MODE:
            logger.debug(f"  Combined shape: {data.shape}")
    else:
        if DEBUG_MODE:
            logger.debug(f"  Returning score points only (with_org_data=False)")
    
    return data


@log_function_call
def validate_scorecard(card: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that scorecard has required structure.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if DEBUG_MODE:
        logger.debug(f"validate_scorecard called")
        logger.debug(f"  card shape: {card.shape}")
        logger.debug(f"  card columns: {list(card.columns)}")
    
    required_cols = ['var', 'bin', 'points']
    
    for col in required_cols:
        if col not in card.columns:
            logger.error(f"  Missing required column: '{col}'")
            return False, f"Scorecard missing required column: '{col}'"
    
    if DEBUG_MODE:
        logger.debug(f"  All required columns present")
    
    if card.empty:
        logger.error(f"  Scorecard is empty")
        return False, "Scorecard is empty"
    
    if DEBUG_MODE:
        logger.debug(f"  Scorecard is not empty ({len(card)} rows)")
    
    # Check for basepoints
    if 'basepoints' not in card['var'].values:
        logger.error(f"  Missing 'basepoints' row")
        return False, "Scorecard missing 'basepoints' row"
    
    if DEBUG_MODE:
        logger.debug(f"  'basepoints' row found")
        logger.info(f"  Scorecard validation PASSED")
    return True, ""


@log_function_call
def validate_data(df: pd.DataFrame, card: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that data has required binned columns.
    
    For interaction terms, checks that both component columns exist.
    
    Returns:
        Tuple of (has_any_columns, list_of_missing_columns)
    """
    if DEBUG_MODE:
        logger.debug(f"validate_data called")
        logger.debug(f"  df shape: {df.shape}")
        logger.debug(f"  df columns: {list(df.columns)[:20]}...")
        logger.debug(f"  card shape: {card.shape}")
    
    # Get required variables from scorecard
    required_vars = card[card['var'] != 'basepoints']['var'].unique()
    if DEBUG_MODE:
        logger.debug(f"  Required variables: {list(required_vars)}")
    
    missing = []
    found_count = 0
    
    for var in required_vars:
        if is_interaction_term(var):
            # For interaction terms, check both component columns
            try:
                var1, var2 = parse_interaction_term(var)
                b_col1 = f'b_{var1}'
                b_col2 = f'b_{var2}'
                
                missing_components = []
                if b_col1 not in df.columns:
                    missing_components.append(b_col1)
                    if DEBUG_MODE:
                        logger.debug(f"    Interaction '{var}': missing '{b_col1}'")
                if b_col2 not in df.columns:
                    missing_components.append(b_col2)
                    if DEBUG_MODE:
                        logger.debug(f"    Interaction '{var}': missing '{b_col2}'")
                    
                if missing_components:
                    missing.append(f"{var} (needs: {', '.join(missing_components)})")
                else:
                    found_count += 1
                    if DEBUG_MODE:
                        logger.debug(f"    Interaction '{var}': both columns found")
                    
            except ValueError:
                missing.append(f"b_{var} (unparseable interaction)")
                logger.error(f"    Could not parse interaction: '{var}'")
        else:
            # Regular variable
            b_col = f'b_{var}'
            if b_col not in df.columns:
                missing.append(b_col)
                if DEBUG_MODE:
                    logger.debug(f"    Variable '{var}': missing '{b_col}'")
            else:
                found_count += 1
                if DEBUG_MODE:
                    logger.debug(f"    Variable '{var}': column '{b_col}' found")
    
    has_any = found_count > 0
    if DEBUG_MODE:
        logger.debug(f"  Summary: found={found_count}, missing={len(missing)}, has_any={has_any}")
    if missing and DEBUG_MODE:
        logger.warning(f"  Missing columns: {missing}")
    
    return has_any, missing


# =============================================================================
# Read Input Data
# =============================================================================
if DEBUG_MODE:
    logger.info("=" * 80)
    logger.info("SCORECARD APPLY NODE - TOGGLE DEBUG VERSION - STARTING")
    logger.info("=" * 80)

print("Scorecard Apply Node - Starting...")
if DEBUG_MODE:
    print("DEBUG MODE: ENABLED")
print("=" * 70)

# Input 1: Scorecard from Scorecard Generator
if DEBUG_MODE:
    logger.debug("Reading Input 1 (Scorecard)...")
card = knio.input_tables[0].to_pandas()
if DEBUG_MODE:
    logger.debug(f"  Scorecard loaded: {len(card)} rows, {len(card.columns)} columns")
    logger.debug(f"  Scorecard columns: {list(card.columns)}")
    logger.debug(f"  Scorecard head:\n{card.head(10).to_string()}")
print(f"Input 1 (Scorecard): {len(card)} rows")

# Input 2: Data with binned columns
if DEBUG_MODE:
    logger.debug("Reading Input 2 (Data)...")
df = knio.input_tables[1].to_pandas()
if DEBUG_MODE:
    logger.debug(f"  Data loaded: {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"  Data columns: {list(df.columns)[:30]}{'...' if len(df.columns) > 30 else ''}")
    logger.debug(f"  Data dtypes: {dict(list(df.dtypes.items())[:10])}...")
    logger.debug(f"  Data index: {df.index[:5].tolist()}...")
print(f"Input 2 (Data): {len(df)} rows, {len(df.columns)} columns")

# Debug: Show scorecard variables (separate regular and interaction)
regular_vars_list = []
interaction_vars_list = []

if DEBUG_MODE:
    logger.debug("Categorizing scorecard variables...")
for var in card['var'].unique():
    if var != 'basepoints':
        if is_interaction_term(var):
            interaction_vars_list.append(var)
        else:
            regular_vars_list.append(var)

if DEBUG_MODE:
    logger.debug(f"  Regular variables: {regular_vars_list}")
    logger.debug(f"  Interaction variables: {interaction_vars_list}")
print(f"\nScorecard variables: {len(regular_vars_list)} regular, {len(interaction_vars_list)} interactions")

print("\nRegular variables:")
for var in regular_vars_list:
    var_rows = len(card[card['var'] == var])
    if DEBUG_MODE:
        logger.debug(f"    {var}: {var_rows} bins")
    print(f"  - {var} ({var_rows} bins)")

if interaction_vars_list:
    print("\nInteraction variables:")
    for var in interaction_vars_list:
        var_rows = len(card[card['var'] == var])
        try:
            var1, var2 = parse_interaction_term(var)
            if DEBUG_MODE:
                logger.debug(f"    {var} = {var1} × {var2}: {var_rows} bins")
            print(f"  - {var} = {var1} × {var2} ({var_rows} bins)")
        except:
            if DEBUG_MODE:
                logger.debug(f"    {var}: {var_rows} bins (could not parse)")
            print(f"  - {var} ({var_rows} bins)")

basepoints_row = card[card['var'] == 'basepoints']
if not basepoints_row.empty:
    bp_value = basepoints_row['points'].iloc[0]
    if DEBUG_MODE:
        logger.debug(f"  Base points: {bp_value}")
    print(f"\nBase points: {bp_value}")

# Debug: Show available b_* columns in data
b_columns = [col for col in df.columns if col.startswith('b_')]
if DEBUG_MODE:
    logger.debug(f"  Binned columns in data: {b_columns}")
print(f"\nBinned columns in data: {len(b_columns)}")
for col in b_columns[:10]:  # Show first 10
    print(f"  - {col}")
if len(b_columns) > 10:
    print(f"  ... and {len(b_columns) - 10} more")

# =============================================================================
# Check for Flow Variables
# =============================================================================
if DEBUG_MODE:
    logger.debug("Checking flow variables...")
with_original_data = True

try:
    with_org_fv = knio.flow_variables.get("WithOriginalData", None)
    if DEBUG_MODE:
        logger.debug(f"  WithOriginalData flow variable: {with_org_fv}")
    if with_org_fv is not None:
        with_original_data = bool(with_org_fv)
        if DEBUG_MODE:
            logger.debug(f"  Using flow variable value: {with_original_data}")
    else:
        if DEBUG_MODE:
            logger.debug(f"  Flow variable not set, using default: {with_original_data}")
except Exception as e:
    if DEBUG_MODE:
        logger.debug(f"  Error reading flow variable: {e}")
    pass

print(f"\nWith original data: {with_original_data}")
print("=" * 70)

# =============================================================================
# Validation
# =============================================================================
if DEBUG_MODE:
    logger.info("Starting validation...")

# Validate scorecard structure
is_valid, error_msg = validate_scorecard(card)
if not is_valid:
    logger.error(f"Scorecard validation FAILED: {error_msg}")
    print(f"ERROR: {error_msg}")
    raise ValueError(error_msg)

if DEBUG_MODE:
    logger.info("Scorecard validation PASSED")

# Validate data has required columns
has_columns, missing_cols = validate_data(df, card)
if missing_cols:
    if DEBUG_MODE:
        logger.warning(f"Missing binned columns: {missing_cols}")
    print(f"WARNING: Missing binned columns: {missing_cols}")
if not has_columns:
    logger.error("No matching binned columns found in data")
    print("ERROR: No matching binned columns found in data")
    raise ValueError("Data does not have any matching b_* columns for scorecard variables")

if DEBUG_MODE:
    logger.info("Data validation PASSED")

# =============================================================================
# Apply Scorecard
# =============================================================================
if DEBUG_MODE:
    logger.info("Starting scorecard application...")
print("\nApplying scorecard...")

# Output 1: Points only (no original data)
if DEBUG_MODE:
    logger.debug("Calculating points-only output...")
df_points = scorepoint_ply(card, df)
if DEBUG_MODE:
    logger.debug(f"  Points-only output shape: {df_points.shape}")
    logger.debug(f"  Points-only columns: {list(df_points.columns)}")
print(f"Points-only output: {len(df_points)} rows, {len(df_points.columns)} columns")

# Output 2: Points with original data
if DEBUG_MODE:
    logger.debug("Calculating full output with original data...")
df_points_dat = scorepoint_ply_with_data(card, df, with_org_data=with_original_data)
if DEBUG_MODE:
    logger.debug(f"  Full output shape: {df_points_dat.shape}")
    logger.debug(f"  Full output columns: {list(df_points_dat.columns)[:20]}...")
print(f"Full output: {len(df_points_dat)} rows, {len(df_points_dat.columns)} columns")

# =============================================================================
# Output Tables
# =============================================================================
if DEBUG_MODE:
    logger.info("Writing output tables...")
    logger.debug(f"  Output 0 (Points only): shape={df_points.shape}")
    logger.debug(f"  Output 1 (Full data): shape={df_points_dat.shape}")

knio.output_tables[0] = knio.Table.from_pandas(df_points)
knio.output_tables[1] = knio.Table.from_pandas(df_points_dat)

if DEBUG_MODE:
    logger.info("Output tables written successfully")

# =============================================================================
# Print Summary
# =============================================================================
print("=" * 70)
print("Scorecard Apply completed successfully")
print("=" * 70)

if DEBUG_MODE:
    logger.info("=" * 80)
    logger.info("SCORECARD APPLY - COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

# Calculate and display score statistics
if not df_points.empty and 'Score' in df_points.columns:
    scores = df_points['Score']
    
    if DEBUG_MODE:
        logger.debug("Score statistics:")
        logger.debug(f"  Count: {len(scores)}")
        logger.debug(f"  Min: {scores.min()}")
        logger.debug(f"  Max: {scores.max()}")
        logger.debug(f"  Mean: {scores.mean()}")
        logger.debug(f"  Median: {scores.median()}")
        logger.debug(f"  Std Dev: {scores.std()}")
    
    print(f"\nScore Statistics:")
    print(f"  Count:    {len(scores)}")
    print(f"  Min:      {scores.min():.0f}")
    print(f"  Max:      {scores.max():.0f}")
    print(f"  Mean:     {scores.mean():.1f}")
    print(f"  Median:   {scores.median():.0f}")
    print(f"  Std Dev:  {scores.std():.1f}")
    
    # Show score distribution buckets
    print(f"\nScore Distribution:")
    score_min = int(scores.min())
    score_max = int(scores.max())
    bucket_size = max(50, (score_max - score_min) // 10)
    
    if DEBUG_MODE:
        logger.debug(f"Score distribution: min={score_min}, max={score_max}, bucket_size={bucket_size}")
    
    buckets = range(score_min - (score_min % bucket_size), 
                    score_max + bucket_size, 
                    bucket_size)
    
    for i, bucket_start in enumerate(list(buckets)[:-1]):
        bucket_end = bucket_start + bucket_size
        count = ((scores >= bucket_start) & (scores < bucket_end)).sum()
        pct = count / len(scores) * 100
        bar = '#' * int(pct / 2)
        if DEBUG_MODE:
            logger.debug(f"  Bucket {bucket_start}-{bucket_end}: count={count}, pct={pct:.1f}%")
        print(f"  {bucket_start:4d}-{bucket_end:4d}: {count:5d} ({pct:5.1f}%) {bar}")

# Show points columns
points_cols = [col for col in df_points.columns if col.endswith('_points')]
if DEBUG_MODE:
    logger.debug(f"Points columns created: {points_cols}")
print(f"\nPoints columns created: {len(points_cols)}")
for col in points_cols:
    print(f"  - {col}")

print(f"\nOutput 1 (Points only): {len(df_points)} rows")
print(f"Output 2 (With original data): {len(df_points_dat)} rows")
print("=" * 70)

if DEBUG_MODE:
    logger.info("Script execution complete")

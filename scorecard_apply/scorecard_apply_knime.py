# =============================================================================
# Scorecard Apply (Scoring) for KNIME Python Script Node
# =============================================================================
# Python implementation of R's scorecard scoring functionality
# Applies a scorecard to data with binned columns to calculate credit scores
# Compatible with KNIME 5.9, Python 3.9
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
# Release Date: 2026-01-19
# Version: 1.2
# 
# Version History:
# 1.2 (2026-01-28): Fixed RowID preservation - output tables now maintain original
#                   RowIDs from input table, enabling joins back to source data
# 1.1: Interaction term support added
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import re
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# Interaction Term Functions
# =============================================================================

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
    return '_x_' in var_name


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
    # Remove leading WOE_ if present
    clean_name = var_name
    if clean_name.startswith('WOE_'):
        clean_name = clean_name[4:]  # Remove 'WOE_'
    
    # Split on '_x_WOE_' first (most specific pattern)
    if '_x_WOE_' in clean_name:
        parts = clean_name.split('_x_WOE_', 1)
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_'
    if '_x_' in clean_name:
        parts = clean_name.split('_x_', 1)
        var2 = parts[1]
        # Remove WOE_ prefix from var2 if present
        if var2.startswith('WOE_'):
            var2 = var2[4:]
        return parts[0], var2
    
    raise ValueError(f"Cannot parse interaction term: {var_name}")


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
    return f"{var1}:{bin1} × {var2}:{bin2}"


# =============================================================================
# Scoring Functions
# =============================================================================

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
    cardx = cardx.copy()
    cardx['bins_value'] = cardx['bin'].astype(str)
    
    # Pattern to remove: < > = ( ) and spaces
    pattern = r'[<>=() ]'
    
    # Process each binValue
    cardx['bins_value'] = cardx['bins_value'].apply(
        lambda x: re.sub(pattern, '', 
                         x.replace('is.na', '1').replace(' & ', '|').replace(' and ', '|'))
    )
    
    # Split into min, max, is_na on | separator
    split_values = cardx['bins_value'].str.split(r'\|', expand=True)
    
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
    
    # Adjust first and last row boundaries
    if len(cardx) > 0:
        cardx.loc[cardx.index[0], 'max'] = cardx.loc[cardx.index[0], 'min']
        cardx.loc[cardx.index[0], 'min'] = -99999
        
    if len(cardx) > 1:
        cardx.loc[cardx.index[-1], 'max'] = 99999
    
    return cardx


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
    # Get only b_* columns from the data
    b_columns = [col for col in df.columns if col.startswith('b_')]
    dt = df[b_columns].copy()
    
    # Initialize result DataFrame - preserve the original index for RowID consistency
    data = pd.DataFrame(index=df.index)
    
    # Get unique variables from scorecard (excluding basepoints)
    xs = card[card['var'] != 'basepoints']['var'].unique()
    
    # Separate regular variables from interaction terms
    regular_vars = []
    interaction_vars = []
    
    for x_i in xs:
        if is_interaction_term(x_i):
            interaction_vars.append(x_i)
        else:
            regular_vars.append(x_i)
    
    # Process regular variables
    for x_i in regular_vars:
        b_x_i = f'b_{x_i}'
        
        # Check if binned column exists in data
        if b_x_i not in dt.columns:
            print(f"WARNING: Column '{b_x_i}' not found in data, skipping variable '{x_i}'")
            continue
        
        # Get scorecard rows for this variable
        cardx = card[card['var'] == x_i][['bin', 'points']].copy()
        
        # Get binned column from data
        dtx = dt[[b_x_i]].copy()
        
        # Join to get points: match b_{var} values with scorecard bin labels
        # Left join: keep all rows from data, add points from scorecard
        point_col_name = f'{x_i}_points'
        
        # Merge on bin value
        merged = dtx.merge(
            cardx,
            left_on=b_x_i,
            right_on='bin',
            how='left'
        )
        
        # Add points column to result
        data[point_col_name] = merged['points'].values
    
    # Process interaction variables
    for x_i in interaction_vars:
        try:
            # Parse interaction term to get component variables
            var1, var2 = parse_interaction_term(x_i)
            b_var1 = f'b_{var1}'
            b_var2 = f'b_{var2}'
            
            # Check if both binned columns exist
            if b_var1 not in dt.columns:
                print(f"WARNING: Column '{b_var1}' not found for interaction '{x_i}', skipping")
                continue
            if b_var2 not in dt.columns:
                print(f"WARNING: Column '{b_var2}' not found for interaction '{x_i}', skipping")
                continue
            
            # Get scorecard rows for this interaction
            cardx = card[card['var'] == x_i][['bin', 'points']].copy()
            
            # Create combined bin labels for each row in the data
            # Format: "var1:bin1 × var2:bin2"
            combined_bins = dt.apply(
                lambda row: create_interaction_bin_label(var1, str(row[b_var1]), var2, str(row[b_var2])),
                axis=1
            )
            
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
            
            # Add points column to result
            data[point_col_name] = merged['points'].values
            
            print(f"Processed interaction: {x_i} ({var1} × {var2})")
            
        except Exception as e:
            print(f"ERROR processing interaction '{x_i}': {e}")
            continue
    
    # Get basescore
    basepoints_row = card[card['var'] == 'basepoints']
    if not basepoints_row.empty:
        basescore = basepoints_row['points'].iloc[0]
    else:
        basescore = 0
        print("WARNING: No basepoints found in scorecard, using 0")
    
    # Add basescore column
    data['basescore'] = basescore
    
    # Calculate total score: sum of all columns (including basescore)
    data['Score'] = data.sum(axis=1)
    
    return data


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
    # Apply score points
    data = scorepoint_ply(card, df)
    
    # Append original dataset if requested
    if with_org_data:
        # Combine: original data first, then score columns
        # Preserve original index (RowIDs) for joining back to source data
        data = pd.concat([df, data], axis=1)
    
    return data


def validate_scorecard(card: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that scorecard has required structure.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ['var', 'bin', 'points']
    
    for col in required_cols:
        if col not in card.columns:
            return False, f"Scorecard missing required column: '{col}'"
    
    if card.empty:
        return False, "Scorecard is empty"
    
    # Check for basepoints
    if 'basepoints' not in card['var'].values:
        return False, "Scorecard missing 'basepoints' row"
    
    return True, ""


def validate_data(df: pd.DataFrame, card: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that data has required binned columns.
    
    For interaction terms, checks that both component columns exist.
    
    Returns:
        Tuple of (has_any_columns, list_of_missing_columns)
    """
    # Get required variables from scorecard
    required_vars = card[card['var'] != 'basepoints']['var'].unique()
    
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
                if b_col2 not in df.columns:
                    missing_components.append(b_col2)
                    
                if missing_components:
                    missing.append(f"{var} (needs: {', '.join(missing_components)})")
                else:
                    found_count += 1
                    
            except ValueError:
                missing.append(f"b_{var} (unparseable interaction)")
        else:
            # Regular variable
            b_col = f'b_{var}'
            if b_col not in df.columns:
                missing.append(b_col)
            else:
                found_count += 1
    
    has_any = found_count > 0
    return has_any, missing


# =============================================================================
# Read Input Data
# =============================================================================
print("Scorecard Apply Node - Starting...")
print("=" * 70)

# Input 1: Scorecard from Scorecard Generator
card = knio.input_tables[0].to_pandas()
print(f"Input 1 (Scorecard): {len(card)} rows")

# Input 2: Data with binned columns
df = knio.input_tables[1].to_pandas()
print(f"Input 2 (Data): {len(df)} rows, {len(df.columns)} columns")

# Debug: Show scorecard variables (separate regular and interaction)
regular_vars_list = []
interaction_vars_list = []

for var in card['var'].unique():
    if var != 'basepoints':
        if is_interaction_term(var):
            interaction_vars_list.append(var)
        else:
            regular_vars_list.append(var)

print(f"\nScorecard variables: {len(regular_vars_list)} regular, {len(interaction_vars_list)} interactions")

print("\nRegular variables:")
for var in regular_vars_list:
    var_rows = len(card[card['var'] == var])
    print(f"  - {var} ({var_rows} bins)")

if interaction_vars_list:
    print("\nInteraction variables:")
    for var in interaction_vars_list:
        var_rows = len(card[card['var'] == var])
        try:
            var1, var2 = parse_interaction_term(var)
            print(f"  - {var} = {var1} × {var2} ({var_rows} bins)")
        except:
            print(f"  - {var} ({var_rows} bins)")

basepoints_row = card[card['var'] == 'basepoints']
if not basepoints_row.empty:
    print(f"\nBase points: {basepoints_row['points'].iloc[0]}")

# Debug: Show available b_* columns in data
b_columns = [col for col in df.columns if col.startswith('b_')]
print(f"\nBinned columns in data: {len(b_columns)}")
for col in b_columns[:10]:  # Show first 10
    print(f"  - {col}")
if len(b_columns) > 10:
    print(f"  ... and {len(b_columns) - 10} more")

# =============================================================================
# Check for Flow Variables
# =============================================================================
with_original_data = True

try:
    with_org_fv = knio.flow_variables.get("WithOriginalData", None)
    if with_org_fv is not None:
        with_original_data = bool(with_org_fv)
except:
    pass

print(f"\nWith original data: {with_original_data}")
print("=" * 70)

# =============================================================================
# Validation
# =============================================================================

# Validate scorecard structure
is_valid, error_msg = validate_scorecard(card)
if not is_valid:
    print(f"ERROR: {error_msg}")
    raise ValueError(error_msg)

# Validate data has required columns
has_columns, missing_cols = validate_data(df, card)
if missing_cols:
    print(f"WARNING: Missing binned columns: {missing_cols}")
if not has_columns:
    print("ERROR: No matching binned columns found in data")
    raise ValueError("Data does not have any matching b_* columns for scorecard variables")

# =============================================================================
# Apply Scorecard
# =============================================================================

print("\nApplying scorecard...")

# Output 1: Points only (no original data)
df_points = scorepoint_ply(card, df)
print(f"Points-only output: {len(df_points)} rows, {len(df_points.columns)} columns")

# Output 2: Points with original data
df_points_dat = scorepoint_ply_with_data(card, df, with_org_data=with_original_data)
print(f"Full output: {len(df_points_dat)} rows, {len(df_points_dat.columns)} columns")

# =============================================================================
# Output Tables
# =============================================================================

knio.output_tables[0] = knio.Table.from_pandas(df_points)
knio.output_tables[1] = knio.Table.from_pandas(df_points_dat)

# =============================================================================
# Print Summary
# =============================================================================
print("=" * 70)
print("Scorecard Apply completed successfully")
print("=" * 70)

# Calculate and display score statistics
if not df_points.empty and 'Score' in df_points.columns:
    scores = df_points['Score']
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
    
    buckets = range(score_min - (score_min % bucket_size), 
                    score_max + bucket_size, 
                    bucket_size)
    
    for i, bucket_start in enumerate(list(buckets)[:-1]):
        bucket_end = bucket_start + bucket_size
        count = ((scores >= bucket_start) & (scores < bucket_end)).sum()
        pct = count / len(scores) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bucket_start:4d}-{bucket_end:4d}: {count:5d} ({pct:5.1f}%) {bar}")

# Show points columns
points_cols = [col for col in df_points.columns if col.endswith('_points')]
print(f"\nPoints columns created: {len(points_cols)}")
for col in points_cols:
    print(f"  - {col}")

print(f"\nOutput 1 (Points only): {len(df_points)} rows")
print(f"Output 2 (With original data): {len(df_points_dat)} rows")
print("=" * 70)

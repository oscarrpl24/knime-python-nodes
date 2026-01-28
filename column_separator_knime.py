# =============================================================================
# Column Separator Node for KNIME
# =============================================================================
# Purpose: Separates columns into two outputs based on naming patterns
# 
# Output 1: Columns that do NOT match any criteria (main data)
# Output 2: Columns that match any of the following criteria:
#   - Starts with "b_"
#   - Contains "date" (case insensitive)
#   - Contains "code" (case insensitive)
#   - Starts with "WOE_"
#   - Ends with "_points"
#   - Contains "basescore" (case insensitive)
#   - Contains "unq_id" (case insensitive)
#   - Contains "nodeid" (case insensitive)
#   - Contains "avg_" (case insensitive)
# =============================================================================

import knime.scripting.io as knio

# -----------------------------------------------------------------------------
# Read Input Table
# -----------------------------------------------------------------------------
df = knio.input_tables[0].to_pandas()

# -----------------------------------------------------------------------------
# Define Column Matching Criteria
# -----------------------------------------------------------------------------
def should_go_to_output2(col_name):
    """
    Check if a column matches any criteria for Output 2.
    Returns True if the column should go to Output 2.
    """
    col_lower = col_name.lower()
    
    # Starts with "b_" (case sensitive as per typical WOE naming)
    if col_name.startswith("b_"):
        return True
    
    # Contains "date" (case insensitive)
    if "date" in col_lower:
        return True
    
    # Contains "code" (case insensitive)
    if "code" in col_lower:
        return True
    
    # Starts with "WOE_" (case sensitive as per typical WOE naming)
    if col_name.startswith("WOE_"):
        return True
    
    # Ends with "_points" (case insensitive)
    if col_lower.endswith("_points"):
        return True
    
    # Contains "basescore" (case insensitive)
    if "basescore" in col_lower:
        return True
    
    # Contains "unq_id" (case insensitive)
    if "unq_id" in col_lower:
        return True
    
    # Contains "nodeid" (case insensitive)
    if "nodeid" in col_lower:
        return True
    
    # Contains "avg_" (case insensitive)
    if "avg_" in col_lower:
        return True
    
    return False

# -----------------------------------------------------------------------------
# Separate Columns
# -----------------------------------------------------------------------------
output1_cols = []
output2_cols = []

for col in df.columns:
    if should_go_to_output2(col):
        output2_cols.append(col)
    else:
        output1_cols.append(col)

# Create output DataFrames
df_output1 = df[output1_cols] if output1_cols else df.iloc[:, 0:0]  # Empty DataFrame if no columns
df_output2 = df[output2_cols] if output2_cols else df.iloc[:, 0:0]  # Empty DataFrame if no columns

# -----------------------------------------------------------------------------
# Log Summary
# -----------------------------------------------------------------------------
print(f"Input columns: {len(df.columns)}")
print(f"Output 1 columns (main data): {len(output1_cols)}")
print(f"Output 2 columns (filtered): {len(output2_cols)}")

if output2_cols:
    print(f"\nColumns sent to Output 2:")
    for col in output2_cols:
        print(f"  - {col}")

# -----------------------------------------------------------------------------
# Write Outputs
# -----------------------------------------------------------------------------
knio.output_tables[0] = knio.Table.from_pandas(df_output1)
knio.output_tables[1] = knio.Table.from_pandas(df_output2)

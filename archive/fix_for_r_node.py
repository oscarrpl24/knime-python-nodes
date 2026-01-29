"""
KNIME Python Script Node - Place this BEFORE the Table to R node
This script sanitizes column names for R compatibility

Input: Table with potentially problematic column names
Output: Same table with R-safe column names
"""

import knime.scripting.io as knio
import re

# Read input table
df = knio.input_tables[0].to_pandas()

print(f"Original columns: {len(df.columns)}")

# Create mapping of old names to new names
renamed_cols = {}
for col in df.columns:
    new_col = col
    
    # Replace problematic characters for R
    # R column names should only contain: letters, numbers, dots, underscores
    new_col = re.sub(r'#', '_NUM_', new_col)  # Replace # with _NUM_
    new_col = re.sub(r'\(', '_', new_col)      # Replace ( with _
    new_col = re.sub(r'\)', '_', new_col)      # Replace ) with _
    new_col = re.sub(r'\s+', '_', new_col)     # Replace spaces with _
    new_col = re.sub(r'-', '_', new_col)       # Replace dashes with _
    new_col = re.sub(r'__+', '_', new_col)     # Replace multiple underscores with single
    new_col = re.sub(r'_$', '', new_col)       # Remove trailing underscore
    
    if new_col != col:
        renamed_cols[col] = new_col
        print(f"  Renamed: '{col}' -> '{new_col}'")

# Apply renaming
if renamed_cols:
    df = df.rename(columns=renamed_cols)
    print(f"\nRenamed {len(renamed_cols)} columns")
else:
    print("No columns needed renaming")

# Output the fixed table
knio.output_tables[0] = knio.Table.from_pandas(df)

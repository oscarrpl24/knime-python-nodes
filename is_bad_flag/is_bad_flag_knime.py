# =============================================================================
# Is Bad Flag Node for KNIME
# =============================================================================
# Purpose: Creates a binary "isBad" column based on GRODI26_wRI values
# 
# Logic:
#   - If GRODI26_wRI < 1, then isBad = 1
#   - Otherwise, isBad = 0
#
# Input: Single table with GRODI26_wRI column
# Output: Same table with isBad column added as the first column
# =============================================================================

import knime.scripting.io as knio
import pandas as pd

# -----------------------------------------------------------------------------
# Read Input Table
# -----------------------------------------------------------------------------
df = knio.input_tables[0].to_pandas()

# -----------------------------------------------------------------------------
# Validate Required Column
# -----------------------------------------------------------------------------
if "GRODI26_wRI" not in df.columns:
    raise ValueError("Required column 'GRODI26_wRI' not found in input table")

# -----------------------------------------------------------------------------
# Create isBad Column
# -----------------------------------------------------------------------------
# isBad = 1 if GRODI26_wRI < 1, else 0
df["isBad"] = (df["GRODI26_wRI"] < 1).astype("Int32")

# -----------------------------------------------------------------------------
# Reorder Columns (isBad first)
# -----------------------------------------------------------------------------
cols = df.columns.tolist()
cols.remove("isBad")
cols = ["isBad"] + cols
df = df[cols]

# -----------------------------------------------------------------------------
# Log Summary
# -----------------------------------------------------------------------------
total_rows = len(df)
bad_count = (df["isBad"] == 1).sum()
good_count = (df["isBad"] == 0).sum()

print(f"Total rows: {total_rows}")
print(f"isBad = 1 (GRODI26_wRI < 1): {bad_count} ({100*bad_count/total_rows:.2f}%)")
print(f"isBad = 0 (GRODI26_wRI >= 1): {good_count} ({100*good_count/total_rows:.2f}%)")

# -----------------------------------------------------------------------------
# Write Output
# -----------------------------------------------------------------------------
knio.output_tables[0] = knio.Table.from_pandas(df)

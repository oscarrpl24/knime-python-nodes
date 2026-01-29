# =============================================================================
# CCR Score Filter Node for KNIME
# =============================================================================
# Purpose: Converts CCR.score (string) to numeric and filters by cutoff value
# 
# Logic:
#   1. Creates CCR.score.num column by converting CCR.score string to number
#   2. Filters out rows where CCR.score.num < cutoff OR is null/NA
#
# Input: Single table with CCR.score column (string)
# Output: Filtered table with CCR.score.num column added
# =============================================================================

import knime.scripting.io as knio
import pandas as pd

# =============================================================================
# CONFIGURATION - Edit this value to change the cutoff
# =============================================================================
CCR_SCORE_CUTOFF = 480
# =============================================================================

# -----------------------------------------------------------------------------
# Read Input Table
# -----------------------------------------------------------------------------
df = knio.input_tables[0].to_pandas()

# -----------------------------------------------------------------------------
# Validate Required Column
# -----------------------------------------------------------------------------
if "CCR.score" not in df.columns:
    raise ValueError("Required column 'CCR.score' not found in input table")

# -----------------------------------------------------------------------------
# Convert CCR.score to Numeric
# -----------------------------------------------------------------------------
# Convert string to numeric, invalid values become NaN
df["CCR.score.num"] = pd.to_numeric(df["CCR.score"], errors="coerce")

# -----------------------------------------------------------------------------
# Log Pre-Filter Summary
# -----------------------------------------------------------------------------
total_rows = len(df)
null_count = df["CCR.score.num"].isna().sum()
below_cutoff = (df["CCR.score.num"] < CCR_SCORE_CUTOFF).sum()

print(f"Pre-filter summary:")
print(f"  Total rows: {total_rows}")
print(f"  Null/NA values: {null_count}")
print(f"  Values below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")

# -----------------------------------------------------------------------------
# Filter Rows
# -----------------------------------------------------------------------------
# Keep rows where CCR.score.num >= cutoff AND is not null
df_filtered = df[df["CCR.score.num"] >= CCR_SCORE_CUTOFF].copy()

# -----------------------------------------------------------------------------
# Log Post-Filter Summary
# -----------------------------------------------------------------------------
filtered_rows = len(df_filtered)
removed_rows = total_rows - filtered_rows

print(f"\nPost-filter summary:")
print(f"  Rows kept: {filtered_rows}")
print(f"  Rows removed: {removed_rows}")
print(f"  Cutoff used: CCR.score.num >= {CCR_SCORE_CUTOFF}")

# -----------------------------------------------------------------------------
# Write Output
# -----------------------------------------------------------------------------
knio.output_tables[0] = knio.Table.from_pandas(df_filtered)

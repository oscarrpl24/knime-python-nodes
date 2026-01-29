# =============================================================================
# CCR Score Filter Node for KNIME - TOGGLE DEBUG VERSION
# =============================================================================
# Purpose: Converts CCR.score (string) to numeric and filters by cutoff value
# 
# TOGGLE VERSION: Debug logging can be enabled/disabled via DEBUG_MODE boolean.
#                 Set DEBUG_MODE = True for extensive logging, False for quiet mode.
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
import numpy as np
import sys
import traceback
from datetime import datetime


# =============================================================================
# CONFIGURATION - Edit these values to change behavior
# =============================================================================
DEBUG_MODE = True  # Set to False to disable debug logging
CCR_SCORE_CUTOFF = 480
# =============================================================================


# =============================================================================
# DEBUG LOGGING UTILITIES
# =============================================================================

def debug_log(message, level="INFO"):
    """
    Centralized debug logging function with timestamp and level.
    Only outputs when DEBUG_MODE is True.
    
    Args:
        message: The message to log
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if not DEBUG_MODE:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")


def debug_separator(title=""):
    """Print a visual separator for log readability."""
    if not DEBUG_MODE:
        return
    if title:
        debug_log(f"{'='*20} {title} {'='*20}")
    else:
        debug_log("=" * 60)


def debug_dataframe_info(df, name="DataFrame"):
    """
    Log comprehensive information about a DataFrame.
    Only outputs when DEBUG_MODE is True.
    
    Args:
        df: The pandas DataFrame to inspect
        name: A descriptive name for the DataFrame
    """
    if not DEBUG_MODE:
        return
    debug_log(f"--- {name} Information ---", "DEBUG")
    debug_log(f"  Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})", "DEBUG")
    debug_log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB", "DEBUG")
    debug_log(f"  Columns: {list(df.columns)}", "DEBUG")
    debug_log(f"  Data types:", "DEBUG")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        debug_log(f"    - {col}: {df[col].dtype} (nulls: {null_count}, {null_pct:.2f}%)", "DEBUG")
    
    # Sample data (first 3 rows)
    if len(df) > 0:
        debug_log(f"  First 3 rows sample:", "DEBUG")
        for idx, row in df.head(3).iterrows():
            debug_log(f"    Row {idx}: {dict(row)}", "DEBUG")
    else:
        debug_log(f"  [DataFrame is empty - no rows to sample]", "DEBUG")


def debug_series_info(series, name="Series"):
    """
    Log comprehensive information about a pandas Series.
    Only outputs when DEBUG_MODE is True.
    
    Args:
        series: The pandas Series to inspect
        name: A descriptive name for the Series
    """
    if not DEBUG_MODE:
        return
    debug_log(f"--- {name} Information ---", "DEBUG")
    debug_log(f"  Length: {len(series)}", "DEBUG")
    debug_log(f"  Data type: {series.dtype}", "DEBUG")
    debug_log(f"  Null count: {series.isna().sum()}", "DEBUG")
    debug_log(f"  Non-null count: {series.notna().sum()}", "DEBUG")
    
    if series.dtype in ['int64', 'float64', 'Int32', 'Int64', 'Float64']:
        non_null = series.dropna()
        if len(non_null) > 0:
            debug_log(f"  Min: {non_null.min()}", "DEBUG")
            debug_log(f"  Max: {non_null.max()}", "DEBUG")
            debug_log(f"  Mean: {non_null.mean():.4f}", "DEBUG")
            debug_log(f"  Median: {non_null.median():.4f}", "DEBUG")
            debug_log(f"  Std Dev: {non_null.std():.4f}", "DEBUG")
    
    # Unique values sample
    unique_count = series.nunique()
    debug_log(f"  Unique values: {unique_count}", "DEBUG")
    if unique_count <= 10:
        debug_log(f"  All unique values: {list(series.dropna().unique())}", "DEBUG")
    else:
        debug_log(f"  Sample unique values (first 5): {list(series.dropna().unique()[:5])}", "DEBUG")


def debug_conversion_details(original_series, converted_series, col_name):
    """
    Log detailed information about a type conversion operation.
    Only outputs when DEBUG_MODE is True.
    
    Args:
        original_series: The original Series before conversion
        converted_series: The Series after conversion
        col_name: The name of the column being converted
    """
    if not DEBUG_MODE:
        return
    debug_log(f"--- Conversion Details for '{col_name}' ---", "DEBUG")
    debug_log(f"  Original dtype: {original_series.dtype}", "DEBUG")
    debug_log(f"  Converted dtype: {converted_series.dtype}", "DEBUG")
    
    # Count conversion outcomes
    original_nulls = original_series.isna().sum()
    converted_nulls = converted_series.isna().sum()
    new_nulls = converted_nulls - original_nulls
    
    debug_log(f"  Original null count: {original_nulls}", "DEBUG")
    debug_log(f"  Converted null count: {converted_nulls}", "DEBUG")
    debug_log(f"  New nulls introduced by conversion: {new_nulls}", "DEBUG")
    
    # Show examples of values that became null
    if new_nulls > 0:
        original_not_null = original_series.notna()
        converted_null = converted_series.isna()
        became_null_mask = original_not_null & converted_null
        became_null_values = original_series[became_null_mask]
        debug_log(f"  Sample values that became null (up to 5):", "WARNING")
        for val in became_null_values.head(5):
            debug_log(f"    - '{val}'", "WARNING")


# =============================================================================
# MAIN SCRIPT EXECUTION
# =============================================================================

debug_separator("CCR SCORE FILTER - TOGGLE DEBUG VERSION")
debug_log("Script execution started")
debug_log(f"Python version: {sys.version}")
debug_log(f"Pandas version: {pd.__version__}")
debug_log(f"NumPy version: {np.__version__}")

# -----------------------------------------------------------------------------
# Configuration Logging
# -----------------------------------------------------------------------------
debug_separator("CONFIGURATION")
debug_log(f"DEBUG_MODE = {DEBUG_MODE}")
debug_log(f"CCR_SCORE_CUTOFF = {CCR_SCORE_CUTOFF}")
debug_log(f"CCR_SCORE_CUTOFF type = {type(CCR_SCORE_CUTOFF).__name__}")

# -----------------------------------------------------------------------------
# Read Input Table
# -----------------------------------------------------------------------------
debug_separator("READING INPUT TABLE")
try:
    debug_log("Attempting to read input table from knio.input_tables[0]...")
    debug_log(f"Number of input tables available: {len(knio.input_tables)}")
    
    if len(knio.input_tables) == 0:
        raise ValueError("No input tables connected to this node")
    
    df = knio.input_tables[0].to_pandas()
    debug_log("Successfully converted KNIME table to pandas DataFrame", "INFO")
    debug_dataframe_info(df, "Input DataFrame")
    
except Exception as e:
    debug_log(f"ERROR reading input table: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Validate Required Column
# -----------------------------------------------------------------------------
debug_separator("VALIDATING REQUIRED COLUMNS")
try:
    debug_log("Checking for required column 'CCR.score'...")
    debug_log(f"Available columns: {list(df.columns)}")
    
    if "CCR.score" not in df.columns:
        debug_log("CRITICAL: Required column 'CCR.score' NOT FOUND!", "CRITICAL")
        # Check for similar column names (case-insensitive)
        similar_cols = [col for col in df.columns if 'ccr' in col.lower() or 'score' in col.lower()]
        if similar_cols:
            debug_log(f"  Similar columns found: {similar_cols}", "WARNING")
            debug_log("  Did you mean one of these columns?", "WARNING")
        raise ValueError("Required column 'CCR.score' not found in input table")
    
    debug_log("Column 'CCR.score' found successfully", "INFO")
    debug_series_info(df["CCR.score"], "CCR.score column")
    
except ValueError:
    raise
except Exception as e:
    debug_log(f"ERROR during column validation: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Convert CCR.score to Numeric
# -----------------------------------------------------------------------------
debug_separator("CONVERTING CCR.SCORE TO NUMERIC")
try:
    debug_log("Beginning conversion of CCR.score string to numeric...")
    
    # Store original for comparison
    original_ccr_score = df["CCR.score"].copy()
    
    # Perform conversion
    debug_log("Calling pd.to_numeric() with errors='coerce'...")
    df["CCR.score.num"] = pd.to_numeric(df["CCR.score"], errors="coerce")
    
    debug_log("Conversion completed", "INFO")
    debug_conversion_details(original_ccr_score, df["CCR.score.num"], "CCR.score")
    debug_series_info(df["CCR.score.num"], "CCR.score.num (converted)")
    
except Exception as e:
    debug_log(f"ERROR during numeric conversion: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Pre-Filter Analysis
# -----------------------------------------------------------------------------
debug_separator("PRE-FILTER ANALYSIS")
try:
    total_rows = len(df)
    null_count = df["CCR.score.num"].isna().sum()
    non_null_count = df["CCR.score.num"].notna().sum()
    below_cutoff = (df["CCR.score.num"] < CCR_SCORE_CUTOFF).sum()
    at_or_above_cutoff = (df["CCR.score.num"] >= CCR_SCORE_CUTOFF).sum()
    
    debug_log(f"Total rows in dataset: {total_rows}")
    debug_log(f"Rows with valid numeric score: {non_null_count} ({non_null_count/total_rows*100:.2f}%)")
    debug_log(f"Rows with null/invalid score: {null_count} ({null_count/total_rows*100:.2f}%)")
    debug_log(f"Rows with score below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")
    debug_log(f"Rows with score at or above cutoff ({CCR_SCORE_CUTOFF}): {at_or_above_cutoff}")
    debug_log(f"Expected rows after filter: {at_or_above_cutoff}")
    debug_log(f"Expected rows to be removed: {total_rows - at_or_above_cutoff}")
    
    # Score distribution analysis
    if non_null_count > 0:
        valid_scores = df["CCR.score.num"].dropna()
        debug_log(f"Score distribution (valid scores only):")
        debug_log(f"  Minimum: {valid_scores.min()}")
        debug_log(f"  Maximum: {valid_scores.max()}")
        debug_log(f"  Mean: {valid_scores.mean():.2f}")
        debug_log(f"  Median: {valid_scores.median():.2f}")
        debug_log(f"  Std Dev: {valid_scores.std():.2f}")
        
        # Percentile distribution
        percentiles = [10, 25, 50, 75, 90]
        debug_log(f"  Percentiles:")
        for p in percentiles:
            pval = np.percentile(valid_scores, p)
            debug_log(f"    {p}th percentile: {pval:.2f}")
    
    # Print summary (always shown - user-facing output)
    print(f"Pre-filter summary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Null/NA values: {null_count}")
    print(f"  Values below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")
    
except Exception as e:
    debug_log(f"ERROR during pre-filter analysis: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Filter Rows
# -----------------------------------------------------------------------------
debug_separator("APPLYING FILTER")
try:
    debug_log(f"Applying filter: CCR.score.num >= {CCR_SCORE_CUTOFF}")
    debug_log("Creating boolean mask...")
    
    filter_mask = df["CCR.score.num"] >= CCR_SCORE_CUTOFF
    debug_log(f"Boolean mask created: {filter_mask.sum()} True, {(~filter_mask).sum()} False")
    
    debug_log("Applying mask and creating filtered DataFrame...")
    df_filtered = df[filter_mask].copy()
    
    debug_log(f"Filtered DataFrame created successfully", "INFO")
    debug_dataframe_info(df_filtered, "Filtered DataFrame")
    
except Exception as e:
    debug_log(f"ERROR during filtering: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Post-Filter Analysis
# -----------------------------------------------------------------------------
debug_separator("POST-FILTER ANALYSIS")
try:
    filtered_rows = len(df_filtered)
    removed_rows = total_rows - filtered_rows
    
    debug_log(f"Rows in filtered dataset: {filtered_rows}")
    debug_log(f"Rows removed by filter: {removed_rows}")
    debug_log(f"Retention rate: {filtered_rows/total_rows*100:.2f}%" if total_rows > 0 else "N/A (no input rows)")
    debug_log(f"Removal rate: {removed_rows/total_rows*100:.2f}%" if total_rows > 0 else "N/A (no input rows)")
    
    # Verify filter worked correctly
    if filtered_rows > 0:
        min_score = df_filtered["CCR.score.num"].min()
        max_score = df_filtered["CCR.score.num"].max()
        debug_log(f"Verification - Score range in filtered data: {min_score} to {max_score}")
        
        if min_score < CCR_SCORE_CUTOFF:
            debug_log(f"WARNING: Minimum score ({min_score}) is below cutoff ({CCR_SCORE_CUTOFF})!", "WARNING")
        else:
            debug_log(f"Filter verification PASSED: All scores >= {CCR_SCORE_CUTOFF}", "INFO")
        
        # Check for nulls in filtered data
        null_in_filtered = df_filtered["CCR.score.num"].isna().sum()
        if null_in_filtered > 0:
            debug_log(f"WARNING: {null_in_filtered} null values present in filtered data!", "WARNING")
        else:
            debug_log("No null values in filtered CCR.score.num column", "INFO")
    
    # Print summary (always shown - user-facing output)
    print(f"\nPost-filter summary:")
    print(f"  Rows kept: {filtered_rows}")
    print(f"  Rows removed: {removed_rows}")
    print(f"  Cutoff used: CCR.score.num >= {CCR_SCORE_CUTOFF}")
    
except Exception as e:
    debug_log(f"ERROR during post-filter analysis: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Write Output
# -----------------------------------------------------------------------------
debug_separator("WRITING OUTPUT")
try:
    debug_log("Preparing to write output to knio.output_tables[0]...")
    debug_log(f"Output DataFrame shape: {df_filtered.shape}")
    debug_log(f"Output DataFrame columns: {list(df_filtered.columns)}")
    debug_log(f"Output DataFrame dtypes:")
    for col in df_filtered.columns:
        debug_log(f"  - {col}: {df_filtered[col].dtype}")
    
    # Convert and assign output
    debug_log("Converting pandas DataFrame to KNIME table...")
    knio.output_tables[0] = knio.Table.from_pandas(df_filtered)
    
    debug_log("Output table successfully written to KNIME", "INFO")
    
except Exception as e:
    debug_log(f"ERROR writing output table: {str(e)}", "ERROR")
    debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# Script Completion
# -----------------------------------------------------------------------------
debug_separator("SCRIPT COMPLETED SUCCESSFULLY")
debug_log(f"Total input rows: {total_rows}")
debug_log(f"Total output rows: {filtered_rows}")
debug_log(f"Rows filtered out: {removed_rows}")
debug_log("CCR Score Filter TOGGLE script execution completed")
debug_separator()

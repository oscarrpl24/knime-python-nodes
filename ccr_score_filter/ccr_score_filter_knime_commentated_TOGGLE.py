# =============================================================================
# CCR Score Filter Node for KNIME - Fully Commentated TOGGLE Version
# =============================================================================
# 
# This is the TOGGLE version of the fully commentated CCR Score Filter script.
# Debug logging can be enabled/disabled via the DEBUG_MODE boolean at the top.
# 
# Purpose: This script is designed to run inside a KNIME 5.9 Python Script node.
#          It takes credit score data (specifically "CCR.score" which stands for
#          Consumer Credit Risk score) and performs two main operations:
#          1. Converts the score from a text/string format to a numeric format
#          2. Filters out any records that have scores below a specified cutoff value
#
# DEBUG TOGGLE FEATURE:
#   - Set DEBUG_MODE = True for extensive logging (timestamps, data inspection, etc.)
#   - Set DEBUG_MODE = False for quiet mode (only essential user-facing output)
#
# Why this is needed:
#   - CCR scores often come from external data sources as text strings
#   - For credit risk modeling, we need numeric values we can compare and calculate with
#   - We filter out low scores because they represent lower credit quality applicants
#     that may not meet minimum underwriting criteria
#
# Input: A single KNIME data table containing a column called "CCR.score" (as text)
# Output: A filtered KNIME data table with:
#         - A new column "CCR.score.num" containing the numeric score
#         - Only rows where the score meets or exceeds the cutoff threshold
#
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORT SECTION
# -----------------------------------------------------------------------------
# We import the libraries (pre-written code packages) that we need to use.

import knime.scripting.io as knio
# This line imports the KNIME Scripting Input/Output library and gives it a short name "knio".
# This library is REQUIRED for any Python script running inside KNIME.
# It provides the connection between KNIME's workflow (the visual node interface) and
# this Python script. Without it, we cannot read input data from KNIME or send output
# data back to KNIME.
# The "as knio" part creates an alias (nickname) so we can type "knio" instead of
# the full name "knime.scripting.io" every time we need to use it.

import pandas as pd
# This line imports the pandas library and gives it the short name "pd".
# Pandas is the most important Python library for working with tabular data (rows and columns).
# A pandas DataFrame is like an Excel spreadsheet or database table held in memory.
# We use pandas to manipulate, filter, and transform our data.
# The "as pd" is a standard convention - nearly all Python data scientists use this alias.

import numpy as np
# NumPy is a fundamental package for numerical computing in Python.
# We use it here for statistical calculations in debug logging.

import sys
# The sys module provides access to system-specific parameters and functions.
# We use it to get the Python version for debug logging.

import traceback
# The traceback module is used to extract, format and print stack traces.
# This is essential for detailed error logging when exceptions occur.

from datetime import datetime
# We import datetime to add timestamps to our debug log messages.
# This helps trace the exact timing of operations.


# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# This section contains settings that control how the script behaves.
# By putting configuration values at the top of the script, they are easy to find
# and modify without having to search through the entire code.

DEBUG_MODE = True
# This boolean (True/False) controls whether debug logging is enabled.
# - Set to True: Extensive debug output with timestamps, data inspection, etc.
# - Set to False: Quiet mode with only essential user-facing output (summaries)
#
# When debugging issues, set this to True to see detailed information about
# every step of the script execution. For production use, set to False.

CCR_SCORE_CUTOFF = 480
# This line creates a variable (a named storage container) called CCR_SCORE_CUTOFF.
# We set its value to 480, which is the minimum acceptable credit score threshold.
# 
# What this means for the business:
#   - Any applicant with a CCR score of 480 or higher will be kept in the data
#   - Any applicant with a CCR score below 480 will be filtered out (removed)
#   - Any applicant with a missing/invalid score will also be filtered out
#
# Why 480? This is a business decision based on credit risk policy.
# Scores below this level may represent unacceptable default risk.
#
# Variable naming convention:
#   - We use ALL_CAPS_WITH_UNDERSCORES for configuration constants
#   - This makes it visually obvious that this is a setting that should not
#     change during script execution, only between runs
#
# =============================================================================


# =============================================================================
# DEBUG LOGGING UTILITIES SECTION
# =============================================================================
# This section defines utility functions for debug logging.
# These functions provide consistent, formatted output for troubleshooting.
# All functions check DEBUG_MODE and return early if debugging is disabled.

def debug_log(message, level="INFO"):
    """
    Centralized debug logging function with timestamp and level.
    Only outputs when DEBUG_MODE is True.
    
    This function is the heart of our debug logging system. It ensures all
    log messages have a consistent format with:
    - A timestamp showing when the message was generated
    - A severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - The actual message content
    
    Args:
        message (str): The message to log. This can be any string describing
                      what's happening in the script at this point.
        level (str): Log level indicating severity. Defaults to "INFO".
                    - DEBUG: Detailed information for diagnosing problems
                    - INFO: General information about script progress
                    - WARNING: Something unexpected but not critical
                    - ERROR: A problem that needs attention
                    - CRITICAL: A severe problem that may stop execution
    
    Returns:
        None: This function prints to console but doesn't return anything.
    
    Example:
        debug_log("Processing started", "INFO")
        # Output: [2024-01-15 14:30:22.456] [INFO] Processing started
    """
    # Early return if debug mode is disabled - this is the toggle mechanism
    if not DEBUG_MODE:
        return
    
    # Get the current timestamp with millisecond precision
    # strftime formats the datetime, and we slice off the last 3 microsecond digits
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Print the formatted log message
    # f-strings allow us to embed variables directly in the string
    print(f"[{timestamp}] [{level}] {message}")


def debug_separator(title=""):
    """
    Print a visual separator for log readability.
    Only outputs when DEBUG_MODE is True.
    
    When reading through long debug logs, visual separators help identify
    different sections of the script execution. This function prints a line
    of equals signs, optionally with a title in the middle.
    
    Args:
        title (str): Optional title to display in the separator.
                    If empty, just prints a line of equals signs.
    
    Returns:
        None: This function prints to console but doesn't return anything.
    
    Example:
        debug_separator("READING INPUT")
        # Output: [timestamp] [INFO] ==================== READING INPUT ====================
        
        debug_separator()
        # Output: [timestamp] [INFO] ============================================================
    """
    if not DEBUG_MODE:
        return
    if title:
        # If a title is provided, center it between equals signs
        debug_log(f"{'='*20} {title} {'='*20}")
    else:
        # If no title, just print a line of 60 equals signs
        debug_log("=" * 60)


def debug_dataframe_info(df, name="DataFrame"):
    """
    Log comprehensive information about a DataFrame.
    Only outputs when DEBUG_MODE is True.
    
    This function provides a detailed snapshot of a DataFrame's structure
    and contents. It's invaluable for understanding what data looks like
    at various stages of processing.
    
    The function logs:
    - Shape (rows x columns)
    - Memory usage
    - Column names
    - Data types for each column
    - Null counts for each column
    - Sample of first 3 rows
    
    Args:
        df (pandas.DataFrame): The DataFrame to inspect.
        name (str): A descriptive name for the DataFrame, used in log headers.
                   Defaults to "DataFrame".
    
    Returns:
        None: This function prints to console but doesn't return anything.
    
    Example:
        debug_dataframe_info(my_df, "Input Data")
    """
    if not DEBUG_MODE:
        return
    
    debug_log(f"--- {name} Information ---", "DEBUG")
    
    # Log the shape - number of rows and columns
    # df.shape returns a tuple (rows, cols)
    debug_log(f"  Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})", "DEBUG")
    
    # Log memory usage
    # memory_usage(deep=True) calculates actual memory including string contents
    # We convert bytes to KB by dividing by 1024
    debug_log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB", "DEBUG")
    
    # Log all column names as a list
    debug_log(f"  Columns: {list(df.columns)}", "DEBUG")
    
    # Log data types and null counts for each column
    debug_log(f"  Data types:", "DEBUG")
    for col in df.columns:
        # Count null values in this column
        null_count = df[col].isna().sum()
        # Calculate null percentage
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        debug_log(f"    - {col}: {df[col].dtype} (nulls: {null_count}, {null_pct:.2f}%)", "DEBUG")
    
    # Log sample data (first 3 rows)
    if len(df) > 0:
        debug_log(f"  First 3 rows sample:", "DEBUG")
        # iterrows() returns an iterator of (index, row) pairs
        for idx, row in df.head(3).iterrows():
            # Convert row to dictionary for cleaner display
            debug_log(f"    Row {idx}: {dict(row)}", "DEBUG")
    else:
        debug_log(f"  [DataFrame is empty - no rows to sample]", "DEBUG")


def debug_series_info(series, name="Series"):
    """
    Log comprehensive information about a pandas Series.
    Only outputs when DEBUG_MODE is True.
    
    A Series is a single column of data. This function provides detailed
    information about a Series including its type, null counts, and
    statistical summary for numeric data.
    
    Args:
        series (pandas.Series): The Series to inspect.
        name (str): A descriptive name for the Series. Defaults to "Series".
    
    Returns:
        None: This function prints to console but doesn't return anything.
    
    Example:
        debug_series_info(df["CCR.score"], "CCR Score Column")
    """
    if not DEBUG_MODE:
        return
    
    debug_log(f"--- {name} Information ---", "DEBUG")
    
    # Basic information
    debug_log(f"  Length: {len(series)}", "DEBUG")
    debug_log(f"  Data type: {series.dtype}", "DEBUG")
    debug_log(f"  Null count: {series.isna().sum()}", "DEBUG")
    debug_log(f"  Non-null count: {series.notna().sum()}", "DEBUG")
    
    # For numeric types, show statistical summary
    # We check for both lowercase and capitalized type names
    if series.dtype in ['int64', 'float64', 'Int32', 'Int64', 'Float64']:
        # Get only non-null values for statistics
        non_null = series.dropna()
        if len(non_null) > 0:
            debug_log(f"  Min: {non_null.min()}", "DEBUG")
            debug_log(f"  Max: {non_null.max()}", "DEBUG")
            debug_log(f"  Mean: {non_null.mean():.4f}", "DEBUG")
            debug_log(f"  Median: {non_null.median():.4f}", "DEBUG")
            debug_log(f"  Std Dev: {non_null.std():.4f}", "DEBUG")
    
    # Show unique value information
    unique_count = series.nunique()
    debug_log(f"  Unique values: {unique_count}", "DEBUG")
    
    # If there are few unique values, show them all
    # Otherwise, just show a sample
    if unique_count <= 10:
        debug_log(f"  All unique values: {list(series.dropna().unique())}", "DEBUG")
    else:
        debug_log(f"  Sample unique values (first 5): {list(series.dropna().unique()[:5])}", "DEBUG")


def debug_conversion_details(original_series, converted_series, col_name):
    """
    Log detailed information about a type conversion operation.
    Only outputs when DEBUG_MODE is True.
    
    When converting data types (e.g., string to numeric), some values may
    become null if they can't be converted. This function tracks and reports
    on those conversion outcomes.
    
    Args:
        original_series (pandas.Series): The Series before conversion.
        converted_series (pandas.Series): The Series after conversion.
        col_name (str): The name of the column being converted.
    
    Returns:
        None: This function prints to console but doesn't return anything.
    
    Example:
        debug_conversion_details(df["CCR.score"], df["CCR.score.num"], "CCR.score")
    """
    if not DEBUG_MODE:
        return
    
    debug_log(f"--- Conversion Details for '{col_name}' ---", "DEBUG")
    
    # Show before and after data types
    debug_log(f"  Original dtype: {original_series.dtype}", "DEBUG")
    debug_log(f"  Converted dtype: {converted_series.dtype}", "DEBUG")
    
    # Count how many values became null during conversion
    original_nulls = original_series.isna().sum()
    converted_nulls = converted_series.isna().sum()
    new_nulls = converted_nulls - original_nulls
    
    debug_log(f"  Original null count: {original_nulls}", "DEBUG")
    debug_log(f"  Converted null count: {converted_nulls}", "DEBUG")
    debug_log(f"  New nulls introduced by conversion: {new_nulls}", "DEBUG")
    
    # If some values became null, show examples of what those values were
    # This is very helpful for understanding data quality issues
    if new_nulls > 0:
        # Find rows where original was not null but converted is null
        original_not_null = original_series.notna()
        converted_null = converted_series.isna()
        became_null_mask = original_not_null & converted_null
        became_null_values = original_series[became_null_mask]
        
        debug_log(f"  Sample values that became null (up to 5):", "WARNING")
        for val in became_null_values.head(5):
            debug_log(f"    - '{val}'", "WARNING")


def debug_filter_details(df, mask, cutoff, col_name):
    """
    Log detailed information about a filtering operation.
    Only outputs when DEBUG_MODE is True.
    
    This function provides insights into how the filter is being applied
    and what data is being kept vs removed.
    
    Args:
        df (pandas.DataFrame): The DataFrame being filtered.
        mask (pandas.Series): The boolean mask used for filtering.
        cutoff: The cutoff value being applied.
        col_name (str): The name of the column being filtered on.
    
    Returns:
        None: This function prints to console but doesn't return anything.
    """
    if not DEBUG_MODE:
        return
    
    debug_log(f"--- Filter Details for '{col_name}' ---", "DEBUG")
    debug_log(f"  Filter condition: {col_name} >= {cutoff}", "DEBUG")
    debug_log(f"  Total rows evaluated: {len(df)}", "DEBUG")
    debug_log(f"  Rows passing filter (True): {mask.sum()}", "DEBUG")
    debug_log(f"  Rows failing filter (False): {(~mask).sum()}", "DEBUG")
    
    # Break down the failures
    col = df[col_name]
    below_cutoff = (col < cutoff) & col.notna()
    is_null = col.isna()
    
    debug_log(f"  Breakdown of filtered-out rows:", "DEBUG")
    debug_log(f"    - Below cutoff ({cutoff}): {below_cutoff.sum()}", "DEBUG")
    debug_log(f"    - Null/NaN values: {is_null.sum()}", "DEBUG")


def debug_output_validation(df, col_name, cutoff):
    """
    Validate and log the output DataFrame to ensure filtering worked correctly.
    Only outputs when DEBUG_MODE is True.
    
    This function performs sanity checks on the filtered output to catch
    any potential issues before the data is sent to KNIME.
    
    Args:
        df (pandas.DataFrame): The filtered DataFrame.
        col_name (str): The name of the score column.
        cutoff: The cutoff value that should have been applied.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not DEBUG_MODE:
        return True  # Skip validation in non-debug mode
    
    debug_log(f"--- Output Validation ---", "DEBUG")
    
    if len(df) == 0:
        debug_log("Output DataFrame is empty - all rows were filtered out", "WARNING")
        return True
    
    # Check that all scores meet the cutoff
    min_score = df[col_name].min()
    max_score = df[col_name].max()
    
    debug_log(f"  Score range in output: {min_score} to {max_score}", "DEBUG")
    
    if min_score < cutoff:
        debug_log(f"VALIDATION FAILED: Min score ({min_score}) < cutoff ({cutoff})", "ERROR")
        return False
    
    # Check for nulls
    null_count = df[col_name].isna().sum()
    if null_count > 0:
        debug_log(f"VALIDATION FAILED: {null_count} null values in output", "ERROR")
        return False
    
    debug_log("VALIDATION PASSED: All output rows meet filter criteria", "INFO")
    return True


# =============================================================================
# MAIN SCRIPT EXECUTION BEGINS HERE
# =============================================================================

debug_separator("CCR SCORE FILTER - TOGGLE VERSION (COMMENTATED)")
debug_log("Script execution started")
debug_log(f"DEBUG_MODE = {DEBUG_MODE}")
debug_log(f"Python version: {sys.version}")
debug_log(f"Pandas version: {pd.__version__}")
debug_log(f"NumPy version: {np.__version__}")

# -----------------------------------------------------------------------------
# CONFIGURATION LOGGING
# -----------------------------------------------------------------------------
debug_separator("CONFIGURATION")
debug_log(f"DEBUG_MODE = {DEBUG_MODE}")
debug_log(f"CCR_SCORE_CUTOFF = {CCR_SCORE_CUTOFF}")
debug_log(f"CCR_SCORE_CUTOFF type = {type(CCR_SCORE_CUTOFF).__name__}")

# -----------------------------------------------------------------------------
# READ INPUT DATA FROM KNIME
# -----------------------------------------------------------------------------
# This section retrieves the data that was passed into this Python node from
# the upstream KNIME workflow.

debug_separator("READING INPUT TABLE")
try:
    debug_log("Attempting to read input table from knio.input_tables[0]...")
    debug_log(f"Number of input tables available: {len(knio.input_tables)}")
    
    # Validate that at least one input table exists
    if len(knio.input_tables) == 0:
        debug_log("CRITICAL: No input tables connected to this node!", "CRITICAL")
        raise ValueError("No input tables connected to this node")
    
    # Read the input table and convert to pandas DataFrame
    # knio.input_tables[0] - Access the first (and only) input port
    # .to_pandas() - Convert KNIME's internal format to a pandas DataFrame
    df = knio.input_tables[0].to_pandas()
    # This line does several things in one statement. Let's break it down:
    #
    # knio.input_tables
    #   - This is a list (ordered collection) of all the input tables connected to this node
    #   - KNIME Python Script nodes can have multiple input ports (connection points)
    #   - Each input port feeds data into this list
    #
    # [0]
    #   - This accesses the FIRST table in the list (Python counts from 0, not 1)
    #   - So [0] means "the first input table" which is the top input port in KNIME
    #   - If we had a second input, we would use [1], third would be [2], etc.
    #
    # .to_pandas()
    #   - This is a method (function attached to an object) that converts the KNIME table
    #     format into a pandas DataFrame format
    #   - KNIME has its own internal table format that is not directly usable by pandas
    #   - This conversion allows us to use all of pandas' powerful data manipulation features
    #
    # df =
    #   - We assign the resulting pandas DataFrame to a variable named "df"
    #   - "df" is a very common abbreviation for "DataFrame" in Python data science
    #   - This variable now holds all our input data in memory, ready to be processed
    #
    # At this point, df contains all rows and columns from the KNIME input table,
    # exactly as they appeared in the previous KNIME node's output.
    
    debug_log("Successfully converted KNIME table to pandas DataFrame", "INFO")
    debug_dataframe_info(df, "Input DataFrame")
    
except Exception as e:
    debug_log(f"ERROR reading input table: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise  # Re-raise the exception after logging

# -----------------------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# -----------------------------------------------------------------------------
# Before processing, we check that the data contains the columns we expect.
# This is called "defensive programming" - we anticipate problems and handle them
# gracefully with clear error messages, rather than letting the script crash
# with a confusing error later.

debug_separator("VALIDATING REQUIRED COLUMNS")
try:
    debug_log("Checking for required column 'CCR.score'...")
    debug_log(f"Available columns in input: {list(df.columns)}")
    debug_log(f"Number of columns: {len(df.columns)}")
    
    if "CCR.score" not in df.columns:
        # This is a conditional check (if statement) that validates our input data.
        # Let's break it down:
        #
        # df.columns
        #   - This property of the DataFrame returns a list of all column names
        #   - Example: if df has columns A, B, C, then df.columns returns ["A", "B", "C"]
        #
        # "CCR.score" not in df.columns
        #   - This is a boolean (True/False) expression
        #   - It checks whether the text "CCR.score" is NOT present in the list of column names
        #   - If the column exists, this evaluates to False and we skip the indented code
        #   - If the column is missing, this evaluates to True and we execute the indented code
        
        debug_log("CRITICAL: Required column 'CCR.score' NOT FOUND!", "CRITICAL")
        
        # Try to help the user by finding similar column names
        similar_cols = [col for col in df.columns if 'ccr' in col.lower() or 'score' in col.lower()]
        if similar_cols:
            debug_log(f"  Similar columns found (case-insensitive search): {similar_cols}", "WARNING")
            debug_log("  Did you mean one of these columns? Check for case sensitivity!", "WARNING")
        else:
            debug_log("  No similar columns found. Please verify your upstream data.", "WARNING")
        
        # raise ValueError("...")
        #   - "raise" is a Python keyword that triggers an error (called an "exception")
        #   - ValueError is a type of error indicating that a value is invalid or missing
        #   - The message in quotes explains what went wrong
        #   - When this line runs, the script stops immediately and KNIME shows this error message
        #   - This is much better than letting the script continue and fail with a confusing
        #     "KeyError" or similar when we try to use a column that doesn't exist
        raise ValueError("Required column 'CCR.score' not found in input table")
    
    debug_log("Column 'CCR.score' found successfully", "INFO")
    debug_series_info(df["CCR.score"], "CCR.score column")
    
    # Additional column analysis for debugging
    debug_log("Analyzing CCR.score column contents...", "DEBUG")
    sample_values = df["CCR.score"].head(10).tolist()
    debug_log(f"  First 10 values: {sample_values}", "DEBUG")
    
except ValueError:
    # Re-raise ValueError without additional logging (already logged above)
    raise
except Exception as e:
    debug_log(f"ERROR during column validation: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# CONVERT CCR.SCORE FROM STRING TO NUMERIC
# -----------------------------------------------------------------------------
# The CCR.score column comes in as text (string type) but we need it as a number
# so we can compare it to our cutoff threshold.

debug_separator("CONVERTING CCR.SCORE TO NUMERIC")
try:
    debug_log("Beginning conversion of CCR.score string to numeric...")
    
    # Store original for comparison (this helps us track what changed)
    original_ccr_score = df["CCR.score"].copy()
    debug_log("Created copy of original CCR.score for comparison", "DEBUG")
    
    # Perform the conversion
    # pd.to_numeric(...)
    #   - This is a pandas function that converts values to numeric type
    #   - It can convert strings like "480", "750.5", "-100" to actual numbers
    #   - It works on entire columns (series) at once, which is very fast
    #
    # df["CCR.score"]
    #   - This is the first argument to pd.to_numeric - the data we want to convert
    #   - We're passing the entire CCR.score column
    #
    # errors="coerce"
    #   - This is a named argument that tells pd.to_numeric how to handle conversion errors
    #   - "coerce" means: if a value cannot be converted to a number, replace it with NaN
    #   - NaN stands for "Not a Number" and represents missing/invalid data in pandas
    #   - Other options are:
    #     - errors="raise" - throw an error if any value can't be converted (we don't want this)
    #     - errors="ignore" - return the original column unchanged (not useful for us)
    #
    # Example of what this does:
    #   - "480" becomes 480.0 (a number)
    #   - "575" becomes 575.0 (a number)
    #   - "N/A" becomes NaN (not a number, will be filtered out later)
    #   - "" (empty) becomes NaN
    #   - "INVALID" becomes NaN
    #   - null/None becomes NaN
    
    debug_log("Calling pd.to_numeric() with errors='coerce'...", "DEBUG")
    df["CCR.score.num"] = pd.to_numeric(df["CCR.score"], errors="coerce")
    # df["CCR.score.num"]
    #   - This is how we create a new column in a pandas DataFrame
    #   - We use square brackets with the desired column name as a string
    #   - If the column doesn't exist, pandas creates it
    #   - If it did exist, this would overwrite it (which is fine in our case)
    #   - The name "CCR.score.num" indicates this is the numeric version of CCR.score
    
    debug_log("Conversion completed", "INFO")
    
    # Log detailed conversion information
    debug_conversion_details(original_ccr_score, df["CCR.score.num"], "CCR.score")
    debug_series_info(df["CCR.score.num"], "CCR.score.num (converted)")
    
    # After this line, df has a new column called "CCR.score.num" with numeric values.
    
except Exception as e:
    debug_log(f"ERROR during numeric conversion: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# LOG PRE-FILTER SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# Before we filter the data, we calculate and display summary statistics.
# This helps users understand their data and verify the script is working correctly.
# These print statements output to KNIME's Python console view.

debug_separator("PRE-FILTER ANALYSIS")
try:
    total_rows = len(df)
    # This line counts the total number of rows in the DataFrame.
    #
    # len() is a built-in Python function that returns the "length" of an object.
    # For a DataFrame, this means the number of rows (not columns).
    #
    # We store this in a variable called "total_rows" for two reasons:
    # 1. We'll use it later to calculate how many rows were removed
    # 2. It makes our print statement more readable
    
    null_count = df["CCR.score.num"].isna().sum()
    # This line counts how many values in the new numeric column are null/NaN.
    # Let's break it down:
    #
    # df["CCR.score.num"]
    #   - Access the column we just created
    #
    # .isna()
    #   - This is a pandas method that returns a series of True/False values
    #   - True means the value is NaN/null, False means it's a valid number
    #   - Example: [480, NaN, 575, NaN] becomes [False, True, False, True]
    #
    # .sum()
    #   - When you sum True/False values, Python treats True as 1 and False as 0
    #   - So summing [False, True, False, True] gives us 2
    #   - This effectively counts how many null values exist
    
    non_null_count = df["CCR.score.num"].notna().sum()
    
    below_cutoff = (df["CCR.score.num"] < CCR_SCORE_CUTOFF).sum()
    # This line counts how many scores are BELOW the cutoff threshold.
    # Let's break it down:
    #
    # df["CCR.score.num"] < CCR_SCORE_CUTOFF
    #   - This compares every value in the column to our cutoff (480)
    #   - It returns a series of True/False values
    #   - True means the score is below 480, False means it's 480 or higher
    #   - NaN values result in False for this comparison (NaN < 480 is False)
    #
    # The parentheses around the comparison ensure it's evaluated before .sum()
    #
    # .sum()
    #   - Same as before, counts the True values
    #   - Gives us the count of scores that are below the cutoff
    
    at_or_above_cutoff = (df["CCR.score.num"] >= CCR_SCORE_CUTOFF).sum()
    
    # Log comprehensive pre-filter statistics (only in debug mode)
    debug_log(f"Total rows in dataset: {total_rows}")
    debug_log(f"Rows with valid numeric score: {non_null_count} ({non_null_count/total_rows*100:.2f}%)" if total_rows > 0 else "N/A")
    debug_log(f"Rows with null/invalid score: {null_count} ({null_count/total_rows*100:.2f}%)" if total_rows > 0 else "N/A")
    debug_log(f"Rows with score below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")
    debug_log(f"Rows with score at or above cutoff ({CCR_SCORE_CUTOFF}): {at_or_above_cutoff}")
    debug_log(f"Expected rows after filter: {at_or_above_cutoff}")
    debug_log(f"Expected rows to be removed: {total_rows - at_or_above_cutoff}")
    
    # Score distribution analysis for debugging
    if non_null_count > 0:
        valid_scores = df["CCR.score.num"].dropna()
        debug_log(f"Score distribution (valid scores only):", "DEBUG")
        debug_log(f"  Minimum: {valid_scores.min()}", "DEBUG")
        debug_log(f"  Maximum: {valid_scores.max()}", "DEBUG")
        debug_log(f"  Mean: {valid_scores.mean():.2f}", "DEBUG")
        debug_log(f"  Median: {valid_scores.median():.2f}", "DEBUG")
        debug_log(f"  Std Dev: {valid_scores.std():.2f}", "DEBUG")
        
        # Percentile distribution
        percentiles = [10, 25, 50, 75, 90]
        debug_log(f"  Percentiles:", "DEBUG")
        for p in percentiles:
            pval = np.percentile(valid_scores, p)
            debug_log(f"    {p}th percentile: {pval:.2f}", "DEBUG")
    
    # Print summary (ALWAYS shown - user-facing output, not controlled by DEBUG_MODE)
    # print() is a built-in Python function that outputs text to the console/terminal.
    # In KNIME, this output appears in the Python Script node's console view.
    print(f"Pre-filter summary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Null/NA values: {null_count}")
    print(f"  Values below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")
    
except Exception as e:
    debug_log(f"ERROR during pre-filter analysis: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# FILTER ROWS BASED ON CUTOFF THRESHOLD
# -----------------------------------------------------------------------------
# This is the core filtering operation - we remove rows that don't meet our criteria.

debug_separator("APPLYING FILTER")
try:
    debug_log(f"Applying filter: CCR.score.num >= {CCR_SCORE_CUTOFF}")
    
    # Create boolean mask
    debug_log("Creating boolean mask...", "DEBUG")
    filter_mask = df["CCR.score.num"] >= CCR_SCORE_CUTOFF
    # df["CCR.score.num"] >= CCR_SCORE_CUTOFF
    #   - This creates a boolean (True/False) series
    #   - For each row, it checks: is this score greater than or equal to 480?
    #   - Example: [480, 475, 550, NaN, 400] becomes [True, False, True, False, False]
    #   - IMPORTANT: NaN >= 480 evaluates to False, so null values are automatically excluded
    
    debug_log(f"Boolean mask created: {filter_mask.sum()} True, {(~filter_mask).sum()} False", "DEBUG")
    
    # Log filter details
    debug_filter_details(df, filter_mask, CCR_SCORE_CUTOFF, "CCR.score.num")
    
    # Apply the filter
    debug_log("Applying mask and creating filtered DataFrame...", "DEBUG")
    df_filtered = df[filter_mask].copy()
    # df[...]
    #   - When you put a boolean series inside square brackets after a DataFrame,
    #     pandas returns only the rows where the value is True
    #   - This is called "boolean indexing" or "boolean filtering"
    #   - It's the standard way to filter data in pandas
    #
    # .copy()
    #   - This creates an independent copy of the filtered data
    #   - Without .copy(), df_filtered would be a "view" of df, not a separate object
    #   - Views can cause confusing warnings and bugs when you try to modify them
    #   - Using .copy() is a best practice for creating filtered subsets
    #
    # df_filtered =
    #   - We store the filtered result in a new variable called "df_filtered"
    #   - We use a new variable name (not overwriting 'df') for two reasons:
    #     1. We can still access the original data in 'df' if needed
    #     2. It's clearer to readers that 'df_filtered' is the processed version
    #
    # What gets filtered OUT (removed):
    #   - Rows where CCR.score.num is less than 480
    #   - Rows where CCR.score.num is null/NaN (because NaN >= 480 is False)
    #
    # What stays IN:
    #   - Rows where CCR.score.num is 480 or higher
    
    debug_log(f"Filtered DataFrame created successfully", "INFO")
    debug_dataframe_info(df_filtered, "Filtered DataFrame")
    
except Exception as e:
    debug_log(f"ERROR during filtering: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# LOG POST-FILTER SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# After filtering, we display summary statistics to show what happened.
# This helps users verify the filter worked as expected.

debug_separator("POST-FILTER ANALYSIS")
try:
    filtered_rows = len(df_filtered)
    # Count the number of rows that passed the filter (i.e., rows remaining).
    # We use the same len() function as before, but now on the filtered DataFrame.
    
    removed_rows = total_rows - filtered_rows
    # Calculate how many rows were removed by the filter.
    # This is simply: (original count) minus (remaining count).
    # This includes both rows with low scores AND rows with null scores.
    
    debug_log(f"Rows in filtered dataset: {filtered_rows}")
    debug_log(f"Rows removed by filter: {removed_rows}")
    debug_log(f"Retention rate: {filtered_rows/total_rows*100:.2f}%" if total_rows > 0 else "N/A (no input rows)")
    debug_log(f"Removal rate: {removed_rows/total_rows*100:.2f}%" if total_rows > 0 else "N/A (no input rows)")
    
    # Validate the output (only in debug mode)
    if not debug_output_validation(df_filtered, "CCR.score.num", CCR_SCORE_CUTOFF):
        debug_log("WARNING: Output validation failed! Check the data.", "WARNING")
    
    # Print summary (ALWAYS shown - user-facing output, not controlled by DEBUG_MODE)
    # The \n at the beginning is a "newline character" - it creates a blank line
    print(f"\nPost-filter summary:")
    print(f"  Rows kept: {filtered_rows}")
    print(f"  Rows removed: {removed_rows}")
    print(f"  Cutoff used: CCR.score.num >= {CCR_SCORE_CUTOFF}")
    
except Exception as e:
    debug_log(f"ERROR during post-filter analysis: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# WRITE OUTPUT DATA BACK TO KNIME
# -----------------------------------------------------------------------------
# This final section sends our processed data back to KNIME so it can be
# used by downstream nodes in the workflow.

debug_separator("WRITING OUTPUT")
try:
    debug_log("Preparing to write output to knio.output_tables[0]...")
    debug_log(f"Output DataFrame shape: {df_filtered.shape}")
    debug_log(f"Output DataFrame columns: {list(df_filtered.columns)}")
    
    # Log data types for each column in the output
    debug_log(f"Output DataFrame dtypes:", "DEBUG")
    for col in df_filtered.columns:
        null_count = df_filtered[col].isna().sum()
        debug_log(f"  - {col}: {df_filtered[col].dtype} (nulls: {null_count})", "DEBUG")
    
    # Convert and assign output
    debug_log("Converting pandas DataFrame to KNIME table...", "DEBUG")
    knio.output_tables[0] = knio.Table.from_pandas(df_filtered)
    # This line sends our filtered DataFrame back to KNIME as the output.
    # Let's break it down:
    #
    # knio.output_tables
    #   - This is a list (like input_tables) that holds the output tables for KNIME
    #   - KNIME Python Script nodes can have multiple output ports
    #   - Each index in this list corresponds to one output port
    #
    # [0]
    #   - We're writing to the FIRST (and only) output port
    #   - If this node had multiple output ports, we could write to [1], [2], etc.
    #
    # knio.Table.from_pandas(df_filtered)
    #   - This converts our pandas DataFrame back into KNIME's table format
    #   - It's the reverse of the .to_pandas() we used at the beginning
    #   - KNIME cannot directly use pandas DataFrames, so this conversion is required
    #
    # The = assignment puts our converted table into the output port.
    # When the script finishes, KNIME reads this output and passes it to the next node.
    #
    # The output table will contain:
    #   - All original columns from the input (including CCR.score as a string)
    #   - The new CCR.score.num column (as a numeric type)
    #   - Only rows where CCR.score.num >= 480
    
    debug_log("Output table successfully written to KNIME", "INFO")
    
except Exception as e:
    debug_log(f"ERROR writing output table: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
    raise

# -----------------------------------------------------------------------------
# SCRIPT COMPLETION
# -----------------------------------------------------------------------------
debug_separator("SCRIPT COMPLETED SUCCESSFULLY")
debug_log(f"Total input rows: {total_rows}")
debug_log(f"Total output rows: {filtered_rows}")
debug_log(f"Rows filtered out: {removed_rows}")
debug_log(f"Retention rate: {filtered_rows/total_rows*100:.2f}%" if total_rows > 0 else "N/A")
debug_log("CCR Score Filter TOGGLE (Commentated) script execution completed")
debug_separator()

# =============================================================================
# END OF SCRIPT
# =============================================================================
# 
# Summary of what this TOGGLE script accomplished:
# 1. Read a data table from KNIME's input port (with optional extensive logging)
# 2. Validated that the required CCR.score column exists (with helpful suggestions if not)
# 3. Converted the CCR.score string column to a numeric column (tracking conversion outcomes)
# 4. Logged comprehensive statistics about the data before filtering (when DEBUG_MODE=True)
# 5. Filtered out rows with scores below 480 or with null/invalid scores
# 6. Validated the output meets all filter criteria (when DEBUG_MODE=True)
# 7. Logged comprehensive statistics about the filtered results (when DEBUG_MODE=True)
# 8. Sent the filtered data to KNIME's output port
#
# Debug toggle feature:
# - Set DEBUG_MODE = True at the top of the script for extensive logging
# - Set DEBUG_MODE = False for quiet mode (only pre/post-filter summaries shown)
#
# Common troubleshooting:
# - If you see "Required column 'CCR.score' not found", check that the upstream
#   node provides a column with exactly that name (case-sensitive!)
# - If too many rows are filtered out, check for data quality issues in CCR.score
# - If no rows are filtered, the cutoff might be set too low
# - To change the cutoff, modify the CCR_SCORE_CUTOFF value at the top
# - To enable debug logging, set DEBUG_MODE = True at the top
#
# =============================================================================

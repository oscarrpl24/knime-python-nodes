# ==============================================================================
# KNIME Python Script: Clean b_Score Column (Fully Commentated DEBUG VERSION)
# ==============================================================================
#
# PURPOSE:
# This script is designed to run inside a KNIME 5.9 Python Script node.
# Its sole function is to clean the 'b_Score' column by removing any single
# quote characters (') that may have been inadvertently introduced during
# data import, transformation, or transfer between systems.
#
# DEBUG VERSION:
# This version includes comprehensive logging at every step to help with
# troubleshooting and understanding data flow through the script.
#
# WHY THIS IS NEEDED:
# In credit risk modeling workflows, score values are often passed between
# systems as text. Sometimes single quotes get wrapped around numeric values
# (e.g., "'750'" instead of "750"), which causes issues when:
#   - Converting the score to a numeric type
#   - Performing mathematical operations on the score
#   - Comparing or sorting scores
#   - Exporting data to downstream systems that expect clean numeric strings
#
# INPUT:
# - Port 0: A single KNIME table containing at least a column named 'b_Score'
#   The b_Score column typically contains credit score values that may have
#   unwanted single quote characters embedded in them.
#
# OUTPUT:
# - Port 0: The same table with the 'b_Score' column cleaned (quotes removed)
#   All other columns remain unchanged.
#
# ==============================================================================

# ------------------------------------------------------------------------------
# LOGGING SETUP SECTION
# ------------------------------------------------------------------------------
# 
# We set up logging FIRST, before any other imports, so we can capture all
# subsequent operations in the log output. This is critical for debugging.
# ------------------------------------------------------------------------------

import logging
import sys
from datetime import datetime
from typing import Optional, Any

def setup_comprehensive_logger(
    name: str = "clean_b_score_commentated_DEBUG",
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure and return a comprehensive logger with detailed formatting.
    
    This function creates a logger that outputs to stdout (console), which
    in KNIME's Python Script node will appear in the node's output/console view.
    
    The logger is configured with:
    - DEBUG level (most verbose) - captures everything
    - Timestamp in ISO format for precise timing
    - Log level indicator (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Human-readable message format
    
    Args:
        name: A unique identifier for this logger instance. Using unique names
              prevents log message duplication if the script is run multiple times.
        level: The minimum log level to capture. logging.DEBUG = 10 (most verbose),
               logging.INFO = 20, logging.WARNING = 30, logging.ERROR = 40,
               logging.CRITICAL = 50.
    
    Returns:
        A configured logging.Logger instance ready for use.
    
    Example usage:
        logger = setup_comprehensive_logger()
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
    """
    # Get or create a logger with the specified name
    # Using getLogger() with a name means we can retrieve the same logger
    # from different parts of the code if needed
    logger = logging.getLogger(name)
    
    # Set the minimum level for this logger
    # Messages below this level will be ignored
    logger.setLevel(level)
    
    # Clear any existing handlers to prevent duplicate log messages
    # This is important when running in KNIME, as the script may be
    # executed multiple times in the same Python session
    logger.handlers.clear()
    
    # Create a StreamHandler that writes to stdout (standard output)
    # In KNIME, stdout is captured and displayed in the node's console view
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create a formatter that produces readable log messages
    # Format: [TIMESTAMP] [LEVEL   ] Message
    # The -8s in levelname ensures left-alignment with padding
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] [%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Attach the handler to the logger
    logger.addHandler(console_handler)
    
    return logger


def log_separator(logger: logging.Logger, char: str = "=", length: int = 70) -> None:
    """
    Log a visual separator line for better readability in log output.
    
    This is a simple utility function that creates visual breaks in the log
    output, making it easier to identify different sections of processing.
    
    Args:
        logger: The logger instance to use
        char: The character to repeat for the separator line
        length: The total length of the separator line
    """
    logger.info(char * length)


# Initialize the logger IMMEDIATELY so all subsequent code can use it
logger = setup_comprehensive_logger()

# ==============================================================================
# SCRIPT INITIALIZATION
# ==============================================================================

log_separator(logger, "=")
logger.info("CLEAN B_SCORE SCRIPT - FULLY COMMENTATED DEBUG VERSION")
log_separator(logger, "=")
logger.info(f"Script execution started at: {datetime.now().isoformat()}")
logger.debug(f"Python interpreter version: {sys.version}")
logger.debug(f"Python executable path: {sys.executable}")


# ------------------------------------------------------------------------------
# IMPORT SECTION
# ------------------------------------------------------------------------------
#
# We import the required modules here. The KNIME module is essential for
# reading/writing data. We also import pandas for additional type checking.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 1: Importing required modules")
logger.info("-" * 70)

# Import the KNIME Python scripting I/O module.
# This module is REQUIRED for all KNIME Python Script nodes - it provides
# the interface between your Python code and the KNIME workflow.
#
# The 'knio' alias is a conventional shorthand that makes the code more readable.
# This module provides access to:
#   - knio.input_tables[]  : Read data from KNIME input ports
#   - knio.output_tables[] : Write data to KNIME output ports
#   - knio.flow_variables  : Read/write KNIME flow variables
#   - knio.Table           : Convert between KNIME tables and pandas DataFrames

logger.debug("Attempting to import knime.scripting.io...")
try:
    import knime.scripting.io as knio
    logger.info("SUCCESS: Imported knime.scripting.io module")
    logger.debug(f"knio module location: {knio.__file__ if hasattr(knio, '__file__') else 'built-in'}")
except ImportError as e:
    logger.critical(f"FATAL: Failed to import KNIME scripting module!")
    logger.critical(f"Error details: {type(e).__name__}: {e}")
    logger.critical("This script must be run inside a KNIME Python Script node.")
    raise

# Import pandas for additional data inspection capabilities
logger.debug("Attempting to import pandas...")
try:
    import pandas as pd
    logger.info(f"SUCCESS: Imported pandas version {pd.__version__}")
except ImportError as e:
    logger.warning(f"Could not directly import pandas: {e}")
    logger.warning("Will use KNIME's bundled pandas through the knio module")


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS SECTION
# ------------------------------------------------------------------------------
#
# These utility functions encapsulate common logging patterns to keep the
# main code clean and ensure consistent, detailed logging throughout.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 2: Defining helper functions")
logger.info("-" * 70)


def log_dataframe_summary(df, label: str = "DataFrame") -> None:
    """
    Log a comprehensive summary of a pandas DataFrame.
    
    This function provides a complete overview of the DataFrame's structure,
    including shape, memory usage, column names, data types, and null counts.
    This information is invaluable for debugging data processing issues.
    
    Args:
        df: The pandas DataFrame to summarize
        label: A descriptive label to identify this DataFrame in the logs
    
    The function logs the following information:
    - Shape (rows x columns)
    - Total memory usage
    - List of all column names
    - Data type of each column
    - Count of null/missing values per column
    """
    logger.debug(f"╔{'═' * 60}╗")
    logger.debug(f"║ {label} Summary".ljust(60) + " ║")
    logger.debug(f"╠{'═' * 60}╣")
    logger.debug(f"║ Shape: {df.shape[0]} rows × {df.shape[1]} columns".ljust(60) + " ║")
    logger.debug(f"║ Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB".ljust(60) + " ║")
    logger.debug(f"╠{'═' * 60}╣")
    logger.debug(f"║ Columns and Types:".ljust(60) + " ║")
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        type_str = str(df[col].dtype)
        line = f"║   {col}: {type_str} ({null_count} nulls, {null_pct:.1f}%)"
        logger.debug(line.ljust(60) + " ║")
    
    logger.debug(f"╚{'═' * 60}╝")


def log_column_details(series, column_name: str) -> None:
    """
    Log detailed statistics and sample values for a specific column.
    
    This function performs deep inspection of a pandas Series, including:
    - Basic statistics (length, nulls, unique values)
    - Sample values for visual verification
    - Quote detection for string columns
    
    Args:
        series: The pandas Series (column) to inspect
        column_name: The name of the column for log labeling
    
    This detailed inspection helps identify:
    - Data quality issues (unexpected nulls)
    - Encoding problems (quotes where there shouldn't be any)
    - Value distribution (through unique counts)
    """
    logger.debug(f"┌{'─' * 50}┐")
    logger.debug(f"│ Column Analysis: '{column_name}'".ljust(50) + " │")
    logger.debug(f"├{'─' * 50}┤")
    
    # Basic statistics
    logger.debug(f"│ Data Type: {series.dtype}".ljust(50) + " │")
    logger.debug(f"│ Total Values: {len(series)}".ljust(50) + " │")
    logger.debug(f"│ Null Count: {series.isnull().sum()}".ljust(50) + " │")
    logger.debug(f"│ Non-Null Count: {series.notna().sum()}".ljust(50) + " │")
    logger.debug(f"│ Unique Values: {series.nunique()}".ljust(50) + " │")
    
    # Sample values (first 5 non-null)
    sample_values = series.dropna().head(5).tolist()
    logger.debug(f"├{'─' * 50}┤")
    logger.debug(f"│ Sample Values (first 5 non-null):".ljust(50) + " │")
    for i, val in enumerate(sample_values):
        val_str = repr(val)  # repr() shows quotes around strings
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        logger.debug(f"│   [{i}]: {val_str}".ljust(50) + " │")
    
    # Quote detection for string-like columns
    if series.dtype == 'object' or 'string' in str(series.dtype).lower():
        str_series = series.astype(str)
        single_quote_count = str_series.str.contains("'", na=False).sum()
        double_quote_count = str_series.str.contains('"', na=False).sum()
        logger.debug(f"├{'─' * 50}┤")
        logger.debug(f"│ Quote Analysis:".ljust(50) + " │")
        logger.debug(f"│   Single quotes ('): {single_quote_count} values".ljust(50) + " │")
        logger.debug(f"│   Double quotes (\"): {double_quote_count} values".ljust(50) + " │")
    
    logger.debug(f"└{'─' * 50}┘")


def log_transformation_comparison(
    before_series,
    after_series,
    column_name: str,
    max_examples: int = 10
) -> None:
    """
    Log a detailed comparison of a column before and after transformation.
    
    This function is essential for verifying that transformations work as
    expected. It compares the original and transformed values, showing:
    - Type changes
    - Count of modified values
    - Specific examples of what changed
    
    Args:
        before_series: The column values before transformation
        after_series: The column values after transformation
        column_name: The name of the column for log labeling
        max_examples: Maximum number of change examples to show (default: 10)
    
    This detailed comparison helps:
    - Verify the transformation worked correctly
    - Identify unexpected changes
    - Debug issues with specific data patterns
    """
    logger.debug(f"╔{'═' * 60}╗")
    logger.debug(f"║ Transformation Comparison: '{column_name}'".ljust(60) + " ║")
    logger.debug(f"╠{'═' * 60}╣")
    
    # Type comparison
    logger.debug(f"║ Type Before: {before_series.dtype}".ljust(60) + " ║")
    logger.debug(f"║ Type After:  {after_series.dtype}".ljust(60) + " ║")
    
    # Convert both to strings for comparison
    before_str = before_series.astype(str)
    after_str = after_series.astype(str)
    
    # Count changes
    changed_mask = before_str != after_str
    changed_count = changed_mask.sum()
    unchanged_count = (~changed_mask).sum()
    total_count = len(before_series)
    
    logger.debug(f"╠{'═' * 60}╣")
    logger.debug(f"║ Change Statistics:".ljust(60) + " ║")
    logger.debug(f"║   Total values: {total_count}".ljust(60) + " ║")
    logger.debug(f"║   Changed: {changed_count} ({changed_count/total_count*100:.1f}%)".ljust(60) + " ║")
    logger.debug(f"║   Unchanged: {unchanged_count} ({unchanged_count/total_count*100:.1f}%)".ljust(60) + " ║")
    
    # Show examples of changes
    if changed_count > 0:
        logger.debug(f"╠{'═' * 60}╣")
        logger.debug(f"║ Examples of Changes (up to {max_examples}):".ljust(60) + " ║")
        
        changed_indices = before_str[changed_mask].head(max_examples).index
        for idx in changed_indices:
            before_val = repr(before_str.loc[idx])
            after_val = repr(after_str.loc[idx])
            # Truncate long values
            if len(before_val) > 20:
                before_val = before_val[:17] + "..."
            if len(after_val) > 20:
                after_val = after_val[:17] + "..."
            logger.debug(f"║   Row {idx}: {before_val} → {after_val}".ljust(60) + " ║")
    
    logger.debug(f"╚{'═' * 60}╝")


def validate_column_exists(df, column_name: str, suggestion_threshold: float = 0.7) -> bool:
    """
    Validate that a required column exists in the DataFrame.
    
    If the column is not found, this function provides helpful suggestions
    by checking for similar column names (case-insensitive matches and
    partial matches).
    
    Args:
        df: The pandas DataFrame to check
        column_name: The exact column name to look for
        suggestion_threshold: Not used currently, but could implement fuzzy matching
    
    Returns:
        True if the column exists, False otherwise
    
    Side effects:
        Logs detailed information about the validation result
    """
    logger.debug(f"Validating existence of column: '{column_name}'")
    logger.debug(f"Total columns in DataFrame: {len(df.columns)}")
    logger.debug(f"Column names: {list(df.columns)}")
    
    if column_name in df.columns:
        logger.info(f"✓ Column '{column_name}' found in DataFrame")
        return True
    
    # Column not found - provide helpful suggestions
    logger.error(f"✗ Column '{column_name}' NOT FOUND in DataFrame!")
    
    # Check for case-insensitive matches
    case_matches = [col for col in df.columns if col.lower() == column_name.lower()]
    if case_matches:
        logger.error(f"  ⚠ Possible case mismatch! Found: {case_matches}")
        logger.error(f"    Remember: Column names are CASE-SENSITIVE!")
    
    # Check for partial matches
    partial_matches = [col for col in df.columns if column_name.lower() in col.lower()]
    if partial_matches and partial_matches != case_matches:
        logger.error(f"  ⚠ Columns containing '{column_name}': {partial_matches}")
    
    # Show all available columns
    logger.error(f"  Available columns: {list(df.columns)}")
    
    return False


logger.debug("Helper functions defined successfully")


# ------------------------------------------------------------------------------
# DATA INPUT SECTION
# ------------------------------------------------------------------------------
#
# Read the input table from KNIME and convert it to a pandas DataFrame.
# We log extensive details about the input data to help diagnose issues.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 3: Reading input data from KNIME")
logger.info("-" * 70)

# Read the first (and in this case, only) input table from KNIME.
#
# BREAKDOWN OF THIS OPERATION:
# - knio.input_tables    : This is a list-like object containing all input tables
#                          connected to the Python Script node in KNIME
# - [0]                  : Access the first input table (Python uses 0-based indexing,
#                          so [0] means "the first element")
# - .to_pandas()         : Convert the KNIME table to a pandas DataFrame
#                          This allows us to use all of pandas' powerful data
#                          manipulation functions

logger.debug("Accessing KNIME input tables...")
logger.debug(f"Number of input ports available: {len(knio.input_tables)}")

try:
    logger.debug("Reading from input port 0...")
    input_table_object = knio.input_tables[0]
    logger.debug(f"Input table object type: {type(input_table_object)}")
    logger.debug(f"Input table object repr: {repr(input_table_object)}")
    
    logger.debug("Converting KNIME table to pandas DataFrame...")
    start_time = datetime.now()
    df = input_table_object.to_pandas()
    conversion_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"SUCCESS: Read input table in {conversion_time:.3f} seconds")
    logger.info(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Log comprehensive DataFrame summary
    log_dataframe_summary(df, "Input Data")
    
except IndexError as e:
    logger.critical("FATAL: No input table found at port 0!")
    logger.critical(f"Exception: {type(e).__name__}: {e}")
    logger.critical("Ensure a data table is connected to the Python Script node's input.")
    raise ValueError("No input table connected to port 0") from e

except Exception as e:
    logger.critical(f"FATAL: Failed to read input table!")
    logger.critical(f"Exception type: {type(e).__name__}")
    logger.critical(f"Exception message: {e}")
    logger.critical("Check that the input data is valid and properly formatted.")
    raise


# ------------------------------------------------------------------------------
# VALIDATION SECTION
# ------------------------------------------------------------------------------
#
# Check if the required 'b_Score' column exists in the input DataFrame.
# This is DEFENSIVE PROGRAMMING - we verify our assumptions before proceeding.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 4: Validating input data requirements")
logger.info("-" * 70)

# Define the target column name as a constant for clarity
TARGET_COLUMN = 'b_Score'

logger.debug(f"Target column to clean: '{TARGET_COLUMN}'")

# Use our validation helper function
if not validate_column_exists(df, TARGET_COLUMN):
    error_msg = (
        f"Column '{TARGET_COLUMN}' not found in input table. "
        f"Available columns: {list(df.columns)}"
    )
    logger.critical(f"FATAL: {error_msg}")
    raise ValueError(error_msg)

# Log detailed information about the target column before cleaning
logger.debug("Analyzing target column before cleaning...")
log_column_details(df[TARGET_COLUMN], TARGET_COLUMN)


# ------------------------------------------------------------------------------
# DATA CLEANING SECTION
# ------------------------------------------------------------------------------
#
# Remove all single quote characters (') from the b_Score column.
# This is the core transformation of the script.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 5: Cleaning b_Score column")
logger.info("-" * 70)

# Store a copy of the original column for comparison logging
# This allows us to show exactly what changed after the transformation
logger.debug("Creating backup of original column for comparison...")
original_b_score = df[TARGET_COLUMN].copy()
logger.debug(f"Original column backed up (dtype: {original_b_score.dtype})")

# STEP 1: Convert to string type
# This is a SAFETY measure because:
#   - If some values are already numeric (int/float), str.replace() would fail
#   - If there are None/NaN values, they become the string "nan"
#   - This ensures we can safely perform string operations on every value

logger.debug("STEP 5a: Converting column to string type...")
logger.debug(f"  Current dtype: {df[TARGET_COLUMN].dtype}")

try:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
    logger.debug(f"  New dtype after astype(str): {df[TARGET_COLUMN].dtype}")
except Exception as e:
    logger.error(f"Failed to convert to string type: {type(e).__name__}: {e}")
    raise

# Log statistics about quotes BEFORE removal
quotes_containing_values = df[TARGET_COLUMN].str.contains("'", na=False).sum()
logger.info(f"Values containing single quotes BEFORE cleaning: {quotes_containing_values}")

# STEP 2: Remove single quotes using str.replace()
#
# BREAKDOWN:
# - .str           : Access the string methods of the Series (pandas string accessor)
# - .replace()     : The string replacement method
# - "'"            : The pattern to search for (a single quote character)
# - ""             : The replacement string (empty string = delete the match)
# - regex=False    : Treat the pattern as a literal string, not a regular expression

logger.debug("STEP 5b: Removing single quotes using str.replace()...")
logger.debug("  Pattern: \"'\" (single quote)")
logger.debug("  Replacement: \"\" (empty string)")
logger.debug("  Regex mode: False (literal string matching)")

try:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace("'", "", regex=False)
    logger.debug("  Quote removal completed successfully")
except Exception as e:
    logger.error(f"Failed to remove quotes: {type(e).__name__}: {e}")
    raise

# Log statistics about quotes AFTER removal
quotes_remaining = df[TARGET_COLUMN].str.contains("'", na=False).sum()
logger.info(f"Values containing single quotes AFTER cleaning: {quotes_remaining}")
logger.info(f"Total quotes removed from: {quotes_containing_values - quotes_remaining} values")

# Log detailed transformation comparison
log_transformation_comparison(original_b_score, df[TARGET_COLUMN], TARGET_COLUMN)


# ------------------------------------------------------------------------------
# OUTPUT VALIDATION SECTION
# ------------------------------------------------------------------------------
#
# Before writing output, we verify the data quality and log any potential issues.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 6: Validating output data quality")
logger.info("-" * 70)

# Log final DataFrame summary
log_dataframe_summary(df, "Output Data")

# Log cleaned column details
log_column_details(df[TARGET_COLUMN], f"{TARGET_COLUMN} (cleaned)")

# Check for potential data quality issues
logger.debug("Checking for potential data quality issues...")

# Issue 1: 'nan' strings from NaN conversion
nan_string_count = (df[TARGET_COLUMN] == 'nan').sum()
if nan_string_count > 0:
    logger.warning(f"⚠ Found {nan_string_count} values that are the string 'nan'")
    logger.warning("  These were originally NaN/None values before string conversion")
    logger.warning("  Consider handling NaN values explicitly if needed")

# Issue 2: Empty strings
empty_string_count = (df[TARGET_COLUMN] == '').sum()
if empty_string_count > 0:
    logger.warning(f"⚠ Found {empty_string_count} empty string values")
    logger.warning("  These may have been empty or contained only quotes")

# Issue 3: Remaining quotes (should be zero!)
if quotes_remaining > 0:
    logger.error(f"✗ ERROR: {quotes_remaining} values still contain single quotes!")
    logger.error("  This indicates the cleaning operation may have failed")
    # Show examples of remaining quotes
    examples = df[df[TARGET_COLUMN].str.contains("'", na=False)][TARGET_COLUMN].head(5)
    logger.error(f"  Examples: {examples.tolist()}")
else:
    logger.info("✓ Verification passed: No single quotes remaining in b_Score column")

# Issue 4: Check for other potentially problematic characters
double_quote_count = df[TARGET_COLUMN].str.contains('"', na=False).sum()
if double_quote_count > 0:
    logger.info(f"ℹ Note: {double_quote_count} values contain double quotes (\") - not removed")


# ------------------------------------------------------------------------------
# DATA OUTPUT SECTION
# ------------------------------------------------------------------------------
#
# Write the cleaned DataFrame back to KNIME's output port.
# This is REQUIRED - without this, downstream nodes won't receive any data.
# ------------------------------------------------------------------------------

logger.info("-" * 70)
logger.info("PHASE 7: Writing output to KNIME")
logger.info("-" * 70)

# Write the cleaned DataFrame back to KNIME's first output port.
#
# BREAKDOWN:
# - knio.output_tables[0]     : Access the first output port of the KNIME node
# - knio.Table.from_pandas(df): Convert pandas DataFrame back to KNIME table format

logger.debug("Converting pandas DataFrame to KNIME table format...")
try:
    start_time = datetime.now()
    knime_output_table = knio.Table.from_pandas(df)
    conversion_time = (datetime.now() - start_time).total_seconds()
    logger.debug(f"Conversion completed in {conversion_time:.3f} seconds")
    logger.debug(f"Output table type: {type(knime_output_table)}")
except Exception as e:
    logger.critical(f"FATAL: Failed to convert DataFrame to KNIME table!")
    logger.critical(f"Exception: {type(e).__name__}: {e}")
    raise

logger.debug("Assigning table to output port 0...")
try:
    knio.output_tables[0] = knime_output_table
    logger.info("SUCCESS: Output table assigned to port 0")
except Exception as e:
    logger.critical(f"FATAL: Failed to assign output table!")
    logger.critical(f"Exception: {type(e).__name__}: {e}")
    raise


# ==============================================================================
# SCRIPT COMPLETION
# ==============================================================================

log_separator(logger, "=")
logger.info("SCRIPT EXECUTION COMPLETED SUCCESSFULLY")
log_separator(logger, "=")
logger.info(f"End time: {datetime.now().isoformat()}")
logger.info(f"Final output shape: {df.shape[0]} rows × {df.shape[1]} columns")
logger.info(f"Column '{TARGET_COLUMN}' has been cleaned of single quotes")
log_separator(logger, "=")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
#
# SUMMARY OF WHAT THIS DEBUG VERSION DOES:
# 1. Sets up comprehensive logging to stdout
# 2. Imports the KNIME Python interface module (with logging)
# 3. Defines helper functions for detailed logging
# 4. Reads the input data table from KNIME (with logging)
# 5. Validates that the required 'b_Score' column exists (with suggestions)
# 6. Removes all single quote characters from 'b_Score' (with before/after comparison)
# 7. Validates output data quality (checks for issues)
# 8. Outputs the cleaned DataFrame back to KNIME (with logging)
#
# DEBUGGING WITH THIS SCRIPT:
# - All operations are logged with timestamps
# - Data types and shapes are shown at each step
# - Sample values help verify data content
# - Transformation comparisons show exactly what changed
# - Warnings highlight potential data quality issues
#
# ==============================================================================

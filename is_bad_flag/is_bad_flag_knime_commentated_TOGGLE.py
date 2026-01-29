# =============================================================================
# IS BAD FLAG NODE FOR KNIME - FULLY COMMENTATED TOGGLE DEBUG VERSION
# =============================================================================
#
# OVERVIEW:
# ---------
# This Python script is designed to run inside a KNIME Python Script node.
# Its purpose is to create a binary "isBad" target variable for credit risk 
# modeling. In credit risk, we need to identify which customers are "bad" 
# (defaulted, delinquent, or risky) versus "good" (paid on time, low risk).
#
# TOGGLE DEBUG VERSION:
# ---------------------
# This version includes extensive debug logging on every function that can be
# toggled on or off by setting the DEBUG variable to True or False.
#
# WHAT IS GRODI26_wRI?
# --------------------
# GRODI26_wRI is a credit performance metric. The "26" typically refers to
# 26 months of observation. "GRODI" stands for "Gross Roll-Down Indicator"
# or similar performance metric. Values below 1 indicate poor performance
# (the customer is considered "bad"), while values >= 1 indicate acceptable
# performance (the customer is considered "good").
#
# BUSINESS LOGIC:
# ---------------
#   - If GRODI26_wRI < 1: Customer is "bad" → isBad = 1
#   - If GRODI26_wRI >= 1: Customer is "good" → isBad = 0
#
# INPUT:
# ------
# A single table containing at least the GRODI26_wRI column
#
# OUTPUT:
# -------
# The same table with a new "isBad" column added as the FIRST column
# (This makes it easy to find the target variable in subsequent nodes)
#
# =============================================================================


# =============================================================================
# DEBUG TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG = False
# =============================================================================


# =============================================================================
# SECTION 1: IMPORT STATEMENTS
# =============================================================================

# -----------------------------------------------------------------------------
# KNIME I/O Module
# -----------------------------------------------------------------------------
# This imports the KNIME Python scripting I/O module and gives it the 
# shorter alias "knio" for convenience.
#
# WHAT IS knio?
# - It is the bridge between KNIME and Python
# - It provides access to input tables (data flowing into this node)
# - It provides access to output tables (data flowing out of this node)
# - It also provides access to flow variables (configuration parameters)
# -----------------------------------------------------------------------------
import knime.scripting.io as knio

# -----------------------------------------------------------------------------
# Pandas Library
# -----------------------------------------------------------------------------
# Pandas is Python's most popular data manipulation library. It provides
# DataFrame objects (similar to Excel tables or SQL tables) for filtering,
# transforming, aggregating, and analyzing data.
# -----------------------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# Debug Logging Libraries
# -----------------------------------------------------------------------------
# These libraries are used to provide extensive debug logging throughout
# the script execution:
#
# - logging: Python's built-in logging framework for structured log output
# - sys: Provides access to stdout for log output
# - traceback: Formats exception tracebacks for detailed error reporting
# - datetime: Used for timestamping log entries and measuring execution time
# - functools.wraps: Preserves function metadata when using decorators
# - typing: Type hints for better code documentation
# -----------------------------------------------------------------------------
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Optional, Union


# =============================================================================
# SECTION 2: LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure comprehensive debug logging for the entire script.
    
    This function sets up the Python logging framework with:
    - DEBUG level logging (captures all log messages) when DEBUG=True
    - WARNING level logging (minimal output) when DEBUG=False
    - Custom format with timestamp, level, function name, and line number
    - Output to stdout (visible in KNIME console)
    
    RETURNS:
    --------
    logging.Logger: Configured logger instance
    
    DEBUG LOG FORMAT:
    -----------------
    2024-01-15 14:30:45.123456 | DEBUG    | function_name              | Line 123  | Message
    
    WHY DEBUG LEVEL?
    ----------------
    DEBUG is the most verbose logging level. It captures:
    - DEBUG: Detailed diagnostic information
    - INFO: Confirmation that things are working
    - WARNING: Something unexpected but not breaking
    - ERROR: Something failed
    - CRITICAL: Program cannot continue
    
    When DEBUG=True, we capture everything. When DEBUG=False, we only see warnings and errors.
    """
    
    # Define the log message format
    # %(asctime)s - Timestamp of the log entry
    # %(levelname)-8s - Log level, left-aligned in 8 characters
    # %(funcName)-25s - Function name, left-aligned in 25 characters  
    # Line %(lineno)-4d - Line number, left-aligned in 4 digits
    # %(message)s - The actual log message
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(funcName)-25s | "
        "Line %(lineno)-4d | %(message)s"
    )
    
    # Date format with microseconds for precise timing
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    
    # Set level based on DEBUG toggle
    log_level = logging.DEBUG if DEBUG else logging.WARNING
    
    # Configure the root logger with our settings
    logging.basicConfig(
        level=log_level,                # Capture based on DEBUG toggle
        format=log_format,              # Use our custom format
        datefmt=date_format,            # Use our date format
        handlers=[
            logging.StreamHandler(sys.stdout)  # Output to stdout
        ]
    )
    
    # Get a logger instance for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    return logger


# Initialize the global logger at module load time
# This runs when the script is first loaded
logger = setup_logging()


# =============================================================================
# SECTION 3: DEBUG DECORATOR AND UTILITIES
# =============================================================================

def debug_function(func):
    """
    A decorator that adds comprehensive entry/exit logging to any function.
    Only logs when DEBUG=True.
    
    WHAT IS A DECORATOR?
    --------------------
    A decorator is a function that wraps another function to extend its
    behavior without modifying its code. The @debug_function syntax above
    a function definition automatically applies this wrapper.
    
    WHAT THIS DECORATOR DOES (when DEBUG=True):
    -------------------------------------------
    1. Logs function entry with all arguments
    2. Records the start time
    3. Executes the original function
    4. Logs the return value and execution time
    5. If an exception occurs, logs detailed error information
    
    WHEN DEBUG=False:
    -----------------
    The function executes normally with no additional overhead.
    
    USAGE:
    ------
    @debug_function
    def my_function(arg1, arg2):
        return arg1 + arg2
    
    When my_function is called with DEBUG=True, you'll see:
    - ENTERING: my_function
    - Positional arg[0]: value1
    - Positional arg[1]: value2
    - ... function executes ...
    - Duration: 0.000123 seconds
    - Return value: result
    - EXITING (SUCCESS): my_function
    
    PARAMETERS:
    -----------
    func : callable
        The function to wrap with debug logging
    
    RETURNS:
    --------
    callable
        The wrapped function with debug logging (when enabled)
    """
    
    @wraps(func)  # Preserves original function's name, docstring, etc.
    def wrapper(*args, **kwargs):
        """Inner wrapper function that adds logging."""
        
        # If DEBUG is off, just execute the function without logging
        if not DEBUG:
            return func(*args, **kwargs)
        
        # Get the function name for logging
        func_name = func.__name__
        
        # =====================================================================
        # LOG FUNCTION ENTRY
        # =====================================================================
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        logger.debug(f"{'='*60}")
        
        # Log all positional arguments
        # *args captures any number of positional arguments as a tuple
        if args:
            for i, arg in enumerate(args):
                # Use safe_repr to handle large/complex objects
                arg_repr = _safe_repr(arg)
                logger.debug(f"  Positional arg[{i}]: {arg_repr}")
        
        # Log all keyword arguments
        # **kwargs captures any number of keyword arguments as a dict
        if kwargs:
            for key, value in kwargs.items():
                value_repr = _safe_repr(value)
                logger.debug(f"  Keyword arg '{key}': {value_repr}")
        
        # Record start time for duration calculation
        start_time = datetime.now()
        logger.debug(f"  Start time: {start_time}")
        
        try:
            # =================================================================
            # EXECUTE THE ORIGINAL FUNCTION
            # =================================================================
            result = func(*args, **kwargs)
            
            # =================================================================
            # LOG SUCCESSFUL COMPLETION
            # =================================================================
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"  End time: {end_time}")
            logger.debug(f"  Duration: {duration:.6f} seconds")
            logger.debug(f"  Return value: {_safe_repr(result)}")
            logger.debug(f"EXITING (SUCCESS): {func_name}")
            logger.debug(f"{'='*60}")
            
            return result
            
        except Exception as e:
            # =================================================================
            # LOG EXCEPTION DETAILS
            # =================================================================
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"  EXCEPTION in {func_name}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {str(e)}")
            logger.error(f"  Duration before error: {duration:.6f} seconds")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            logger.error(f"EXITING (FAILURE): {func_name}")
            logger.debug(f"{'='*60}")
            
            # Re-raise the exception so it propagates normally
            raise
    
    return wrapper


def _safe_repr(obj: Any, max_length: int = 500) -> str:
    """
    Safely create a string representation of any object for logging.
    
    WHY IS THIS NEEDED?
    -------------------
    When logging function arguments and return values, we need to convert
    objects to strings. However:
    - DataFrames can be huge - we don't want to log millions of rows
    - Some objects may have broken __repr__ methods
    - We want consistent, informative output
    
    This function handles special cases for pandas objects and provides
    safe fallbacks for everything else.
    
    PARAMETERS:
    -----------
    obj : Any
        The object to convert to a string representation
    max_length : int
        Maximum length of the output string (default 500)
    
    RETURNS:
    --------
    str
        A safe, truncated string representation of the object
    
    EXAMPLES:
    ---------
    DataFrame -> "DataFrame(shape=(1000, 50), columns=['a', 'b', ...])"
    Series -> "Series(name='column', length=1000, dtype=float64)"
    Large list -> "list(length=10000, first_10=[1, 2, 3, ...])"
    """
    
    try:
        # Handle pandas DataFrame specially
        if isinstance(obj, pd.DataFrame):
            return (
                f"DataFrame(shape={obj.shape}, "
                f"columns={list(obj.columns)[:10]}{'...' if len(obj.columns) > 10 else ''}, "
                f"dtypes={dict(list(obj.dtypes.items())[:5])}{'...' if len(obj.dtypes) > 5 else ''})"
            )
        
        # Handle pandas Series specially
        elif isinstance(obj, pd.Series):
            return (
                f"Series(name='{obj.name}', length={len(obj)}, "
                f"dtype={obj.dtype}, "
                f"head={list(obj.head(3).values)})"
            )
        
        # Handle lists and tuples - show first 10 items if large
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 10:
                return f"{type(obj).__name__}(length={len(obj)}, first_10={obj[:10]}...)"
            return repr(obj)
        
        # Handle dictionaries - show first 5 items if large
        elif isinstance(obj, dict):
            if len(obj) > 5:
                sample = dict(list(obj.items())[:5])
                return f"dict(length={len(obj)}, sample={sample}...)"
            return repr(obj)
        
        # Default: use standard repr() with length limit
        else:
            result = repr(obj)
            if len(result) > max_length:
                return result[:max_length] + "..."
            return result
            
    except Exception as e:
        # If repr() fails for any reason, return error info
        return f"<repr failed: {type(e).__name__}: {e}>"


def _log_dataframe_details(df: pd.DataFrame, context: str = "") -> None:
    """
    Log detailed information about a DataFrame.
    Only logs when DEBUG=True.
    
    This helper function logs comprehensive details about a DataFrame,
    which is useful for debugging data transformation issues.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze and log
    context : str
        Optional context string for the log message (e.g., "after merge")
    
    LOGGED INFORMATION:
    -------------------
    - Shape (rows x columns)
    - Column names
    - Data types for each column
    - Memory usage
    - Null value counts per column
    - First few rows as sample
    """
    
    if not DEBUG:
        return
    
    prefix = f"[{context}] " if context else ""
    
    logger.debug(f"{prefix}DataFrame Details:")
    logger.debug(f"  Shape: {df.shape} (rows={df.shape[0]}, columns={df.shape[1]})")
    logger.debug(f"  Columns: {list(df.columns)}")
    
    # Log data types
    logger.debug(f"  Data types:")
    for col, dtype in df.dtypes.items():
        logger.debug(f"    - {col}: {dtype}")
    
    # Log memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.debug(f"  Memory usage: {memory_mb:.4f} MB")
    
    # Log null counts
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.debug(f"  Columns with nulls:")
        for col, count in null_counts[null_counts > 0].items():
            pct = 100 * count / len(df)
            logger.debug(f"    - {col}: {count} ({pct:.2f}%)")
    else:
        logger.debug(f"  No null values in any column")


# =============================================================================
# SECTION 4: CORE FUNCTIONS WITH DEBUG LOGGING
# =============================================================================

@debug_function
def read_input_table() -> pd.DataFrame:
    """
    Read the input table from KNIME and convert to pandas DataFrame.
    
    This function accesses the first input port of the KNIME Python Script
    node and converts the KNIME Table to a pandas DataFrame for processing.
    
    KNIME I/O PATTERN:
    ------------------
    - knio.input_tables is a list of all input tables connected to this node
    - knio.input_tables[0] is the first input table (0-indexed)
    - .to_pandas() converts the KNIME Table to a pandas DataFrame
    
    RETURNS:
    --------
    pd.DataFrame
        The input data as a pandas DataFrame
    
    RAISES:
    -------
    ValueError
        If no input tables are available
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - Number of input tables available
    - DataFrame shape, columns, and data types
    - Memory usage and null value counts
    - First 5 rows for inspection
    """
    
    if DEBUG:
        logger.info("Reading input table from KNIME...")
        logger.debug(f"Number of input tables available: {len(knio.input_tables)}")
    
    # Validate that we have at least one input table
    if len(knio.input_tables) == 0:
        if DEBUG:
            logger.error("No input tables found!")
        raise ValueError("No input tables available. Ensure the node has an input connection.")
    
    # Access the first input table
    if DEBUG:
        logger.debug("Accessing input_tables[0]...")
    knime_table = knio.input_tables[0]
    if DEBUG:
        logger.debug(f"KNIME table object type: {type(knime_table)}")
    
    # Convert to pandas DataFrame
    if DEBUG:
        logger.debug("Converting KNIME table to pandas DataFrame...")
    df = knime_table.to_pandas()
    
    # Log comprehensive DataFrame details
    if DEBUG:
        logger.info(f"DataFrame loaded successfully")
        _log_dataframe_details(df, "Input")
        logger.debug(f"First 5 rows:\n{df.head().to_string()}")
    
    return df


@debug_function
def validate_required_column(df: pd.DataFrame, column_name: str = "GRODI26_wRI") -> bool:
    """
    Validate that the required column exists in the DataFrame.
    
    This is a defensive programming pattern - we explicitly check that our
    required column exists before trying to use it. This provides a clear
    error message if the column is missing, rather than a cryptic KeyError.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame to validate
    column_name : str
        The name of the required column (default: "GRODI26_wRI")
    
    RETURNS:
    --------
    bool
        True if validation passes
    
    RAISES:
    -------
    ValueError
        If the required column is not found in the DataFrame
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - All available column names
    - Whether the required column exists
    - Column statistics if column is found (min, max, mean, etc.)
    - Similar column names (case-insensitive) if column not found
    """
    
    if DEBUG:
        logger.info(f"Validating required column: '{column_name}'")
        logger.debug(f"Available columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            logger.debug(f"  [{i}] '{col}' - dtype: {df[col].dtype}")
    
    # Check if the required column exists
    column_exists = column_name in df.columns
    if DEBUG:
        logger.debug(f"Column '{column_name}' exists: {column_exists}")
    
    if not column_exists:
        # Log detailed error information
        if DEBUG:
            logger.error(f"Required column '{column_name}' NOT FOUND in input table!")
            logger.error(f"Available columns: {list(df.columns)}")
            similar_columns = [col for col in df.columns if column_name.lower() in col.lower()]
            if similar_columns:
                logger.error(f"Similar columns found (case-insensitive): {similar_columns}")
        
        raise ValueError(f"Required column '{column_name}' not found in input table")
    
    # Log column statistics for debugging
    if DEBUG:
        col_data = df[column_name]
        logger.debug(f"Column '{column_name}' statistics:")
        logger.debug(f"  dtype: {col_data.dtype}")
        logger.debug(f"  non-null count: {col_data.count()}")
        logger.debug(f"  null count: {col_data.isnull().sum()}")
        
        if pd.api.types.is_numeric_dtype(col_data):
            logger.debug(f"  min: {col_data.min()}")
            logger.debug(f"  max: {col_data.max()}")
            logger.debug(f"  mean: {col_data.mean():.4f}")
            logger.debug(f"  median: {col_data.median():.4f}")
            logger.debug(f"  std: {col_data.std():.4f}")
            logger.debug(f"  Values < 1 count: {(col_data < 1).sum()}")
            logger.debug(f"  Values >= 1 count: {(col_data >= 1).sum()}")
        
        logger.debug(f"First 10 values: {list(col_data.head(10).values)}")
        logger.info(f"Column '{column_name}' validation PASSED")
    
    return True


@debug_function
def create_is_bad_column(
    df: pd.DataFrame, 
    source_column: str = "GRODI26_wRI", 
    target_column: str = "isBad"
) -> pd.DataFrame:
    """
    Create the binary isBad column based on the source column values.
    
    BUSINESS LOGIC:
    ---------------
    - If source_column < 1: isBad = 1 (customer is "bad")
    - If source_column >= 1: isBad = 0 (customer is "good")
    
    This is the core logic of the script. We perform an element-wise
    comparison on the source column and convert the boolean result to
    a nullable integer (Int32).
    
    WHY Int32 (CAPITAL I)?
    ----------------------
    - "Int32" is a nullable integer type in pandas
    - "int" (lowercase) cannot contain null/missing values
    - If there are nulls in the source column, the comparison produces NaN
    - Using "Int32" allows proper null handling when data goes back to KNIME
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame to modify
    source_column : str
        The column to base the binary flag on (default: "GRODI26_wRI")
    target_column : str
        The name for the new binary column (default: "isBad")
    
    RETURNS:
    --------
    pd.DataFrame
        The DataFrame with the new binary column added
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - Source column values and statistics
    - Comparison result (True/False distribution)
    - Converted integer values
    - Value distribution in the new column
    """
    
    if DEBUG:
        logger.info(f"Creating binary column '{target_column}' from '{source_column}'")
        logger.debug(f"Input DataFrame shape: {df.shape}")
        logger.debug(f"Source column '{source_column}' dtype: {df[source_column].dtype}")
    
    # Get source column values
    source_values = df[source_column]
    if DEBUG:
        logger.debug(f"Source column sample values: {list(source_values.head(10).values)}")
    
    # Perform the comparison: < 1 means "bad"
    if DEBUG:
        logger.debug(f"Performing comparison: {source_column} < 1")
    comparison_result = source_values < 1
    
    # Log comparison results
    if DEBUG:
        logger.debug(f"Comparison result dtype: {comparison_result.dtype}")
        logger.debug(f"Comparison result sample: {list(comparison_result.head(10).values)}")
        logger.debug(f"True count (will be isBad=1): {comparison_result.sum()}")
        logger.debug(f"False count (will be isBad=0): {(~comparison_result).sum()}")
    
    # Convert boolean to nullable integer
    if DEBUG:
        logger.debug("Converting boolean to Int32 (nullable integer)...")
    is_bad_values = comparison_result.astype("Int32")
    if DEBUG:
        logger.debug(f"Converted column dtype: {is_bad_values.dtype}")
        logger.debug(f"Converted column sample values: {list(is_bad_values.head(10).values)}")
    
    # Track null value handling
    if DEBUG:
        null_count_source = source_values.isnull().sum()
        null_count_result = is_bad_values.isnull().sum()
        logger.debug(f"Null values in source: {null_count_source}")
        logger.debug(f"Null values in result: {null_count_result}")
        if null_count_source != null_count_result:
            logger.warning(f"Null count changed during conversion!")
    
    # Assign the new column to the DataFrame
    if DEBUG:
        logger.debug(f"Assigning '{target_column}' column to DataFrame...")
    df[target_column] = is_bad_values
    
    # Verify the assignment worked
    if DEBUG:
        logger.debug(f"Output DataFrame shape: {df.shape}")
        logger.debug(f"'{target_column}' column now in DataFrame: {target_column in df.columns}")
        logger.debug(f"'{target_column}' dtype: {df[target_column].dtype}")
        value_counts = df[target_column].value_counts(dropna=False)
        logger.debug(f"Value distribution:\n{value_counts.to_string()}")
        logger.info(f"Binary column '{target_column}' created successfully")
    
    return df


@debug_function
def reorder_columns(df: pd.DataFrame, first_column: str = "isBad") -> pd.DataFrame:
    """
    Reorder DataFrame columns to put the specified column first.
    
    WHY PUT isBad FIRST?
    --------------------
    - The target variable (dependent variable) is typically placed first
    - This makes it easy to find when viewing the data
    - Many modeling tools expect the target in the first column
    - It's a standard convention in data science workflows
    
    HOW IT WORKS:
    -------------
    1. Get the current column list
    2. Remove the target column from wherever it currently is
    3. Prepend the target column to the beginning
    4. Apply the new column order to the DataFrame
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame to reorder
    first_column : str
        The column to move to the first position (default: "isBad")
    
    RETURNS:
    --------
    pd.DataFrame
        The DataFrame with reordered columns
    
    RAISES:
    -------
    ValueError
        If the specified column does not exist in the DataFrame
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - Current column order
    - Current position of the target column
    - New column order after reordering
    """
    
    if DEBUG:
        logger.info(f"Reordering columns to put '{first_column}' first")
        logger.debug(f"Current column order: {list(df.columns)}")
        logger.debug(f"Current number of columns: {len(df.columns)}")
    
    # Verify the column exists before attempting to reorder
    if first_column not in df.columns:
        if DEBUG:
            logger.error(f"Column '{first_column}' not found for reordering!")
        raise ValueError(f"Cannot reorder: column '{first_column}' not found")
    
    # Get current column list as a mutable Python list
    cols = df.columns.tolist()
    if DEBUG:
        logger.debug(f"Columns as list: {cols}")
    
    # Find current position of the target column
    current_position = cols.index(first_column)
    if DEBUG:
        logger.debug(f"Current position of '{first_column}': {current_position}")
    
    # Remove from current position
    if DEBUG:
        logger.debug(f"Removing '{first_column}' from position {current_position}...")
    cols.remove(first_column)
    if DEBUG:
        logger.debug(f"Columns after removal: {cols}")
    
    # Prepend to beginning using list concatenation
    if DEBUG:
        logger.debug(f"Prepending '{first_column}' to beginning...")
    cols = [first_column] + cols
    if DEBUG:
        logger.debug(f"New column order: {cols}")
    
    # Apply new column order to DataFrame
    if DEBUG:
        logger.debug("Applying new column order to DataFrame...")
    df = df[cols]
    
    # Verify reordering was successful
    if DEBUG:
        logger.debug(f"Verified new column order: {list(df.columns)}")
        logger.debug(f"First column is '{first_column}': {df.columns[0] == first_column}")
        logger.info(f"Columns reordered successfully. First column: '{df.columns[0]}'")
    
    return df


@debug_function
def log_summary_statistics(
    df: pd.DataFrame, 
    target_column: str = "isBad", 
    source_column: str = "GRODI26_wRI"
) -> Dict[str, Any]:
    """
    Calculate and log summary statistics for the binary target.
    
    This function provides a quick summary of the binary target distribution,
    which is crucial for credit risk modeling. The "bad rate" is one of the
    most important metrics in credit scoring.
    
    TYPICAL BAD RATES:
    ------------------
    - Prime credit cards: 1-3%
    - Subprime products: 10-20%
    - If you see 50%+ bad rate, something may be wrong with the data!
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame containing the binary target
    target_column : str
        The name of the binary target column (default: "isBad")
    source_column : str
        The name of the source column for display (default: "GRODI26_wRI")
    
    RETURNS:
    --------
    Dict[str, Any]
        A dictionary containing:
        - total_rows: Total number of rows
        - bad_count: Number of isBad = 1
        - good_count: Number of isBad = 0
        - null_count: Number of null values
        - bad_pct, good_pct, null_pct: Corresponding percentages
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - All counts and percentages
    - Verification that counts sum to total
    - Warning if there are null values
    """
    
    if DEBUG:
        logger.info("Calculating summary statistics...")
    
    # Calculate total rows
    total_rows = len(df)
    if DEBUG:
        logger.debug(f"Total rows in DataFrame: {total_rows}")
    
    # Calculate counts for each value
    bad_count = (df[target_column] == 1).sum()
    good_count = (df[target_column] == 0).sum()
    null_count = df[target_column].isnull().sum()
    
    if DEBUG:
        logger.debug(f"isBad = 1 (bad) count: {bad_count}")
        logger.debug(f"isBad = 0 (good) count: {good_count}")
        logger.debug(f"isBad = null count: {null_count}")
    
    # Calculate percentages (handle division by zero)
    if total_rows > 0:
        bad_pct = 100 * bad_count / total_rows
        good_pct = 100 * good_count / total_rows
        null_pct = 100 * null_count / total_rows
    else:
        bad_pct = good_pct = null_pct = 0.0
        if DEBUG:
            logger.warning("Total rows is 0! Cannot calculate percentages.")
    
    if DEBUG:
        logger.debug(f"isBad = 1 percentage: {bad_pct:.4f}%")
        logger.debug(f"isBad = 0 percentage: {good_pct:.4f}%")
        logger.debug(f"isBad = null percentage: {null_pct:.4f}%")
    
    # Verify counts add up correctly
    if DEBUG:
        sum_of_counts = bad_count + good_count + null_count
        logger.debug(f"Sum of counts: {sum_of_counts} (should equal {total_rows})")
        if sum_of_counts != total_rows:
            logger.warning(f"Count mismatch! Sum={sum_of_counts}, Total={total_rows}")
    
    # Print summary to KNIME console (visible to user)
    print(f"\n{'='*60}")
    print(f"IS BAD FLAG - SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows}")
    print(f"isBad = 1 ({source_column} < 1): {bad_count} ({bad_pct:.2f}%)")
    print(f"isBad = 0 ({source_column} >= 1): {good_count} ({good_pct:.2f}%)")
    if null_count > 0:
        print(f"isBad = null: {null_count} ({null_pct:.2f}%)")
    print(f"{'='*60}\n")
    
    # Log as info for visibility
    if DEBUG:
        logger.info(f"Summary: Total={total_rows}, Bad={bad_count} ({bad_pct:.2f}%), Good={good_count} ({good_pct:.2f}%)")
    
    # Return statistics as a dictionary for potential further use
    return {
        "total_rows": total_rows,
        "bad_count": bad_count,
        "good_count": good_count,
        "null_count": null_count,
        "bad_pct": bad_pct,
        "good_pct": good_pct,
        "null_pct": null_pct
    }


@debug_function
def write_output_table(df: pd.DataFrame) -> bool:
    """
    Write the DataFrame back to KNIME as an output table.
    
    This is the final step - we convert our pandas DataFrame back to a
    KNIME Table and assign it to the first output port. The data will
    then be available to downstream nodes in the KNIME workflow.
    
    KNIME OUTPUT PATTERN:
    ---------------------
    - knio.output_tables is a list of output ports
    - knio.output_tables[0] is the first output port
    - knio.Table.from_pandas(df) converts DataFrame to KNIME Table
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame to write to KNIME
    
    RETURNS:
    --------
    bool
        True if the write was successful
    
    POTENTIAL ISSUES:
    -----------------
    - Writing to output_tables[1] when only 1 port is configured → IndexError
    - Invalid data types may fail to convert
    - Infinite values may cause issues in some KNIME nodes
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - Output DataFrame details (shape, columns, types)
    - First 5 rows for inspection
    - Warning if infinite values are detected
    """
    
    if DEBUG:
        logger.info("Writing output table to KNIME...")
        logger.debug(f"Output DataFrame shape: {df.shape}")
        logger.debug(f"Output columns: {list(df.columns)}")
        _log_dataframe_details(df, "Output")
        logger.debug(f"First 5 rows of output:\n{df.head().to_string()}")
        logger.debug("Checking for potential output issues...")
    
    # Check for infinite values in numeric columns
    if DEBUG:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                inf_count = (non_null_values.abs() == float('inf')).sum()
                if inf_count > 0:
                    logger.warning(f"Column '{col}' contains {inf_count} infinite values!")
    
    # Convert to KNIME table
    if DEBUG:
        logger.debug("Converting pandas DataFrame to KNIME Table...")
    knime_table = knio.Table.from_pandas(df)
    if DEBUG:
        logger.debug(f"KNIME Table created. Type: {type(knime_table)}")
    
    # Assign to output port
    if DEBUG:
        logger.debug("Assigning to output_tables[0]...")
    knio.output_tables[0] = knime_table
    
    if DEBUG:
        logger.info("Output table written successfully to KNIME")
    return True


# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

@debug_function
def main() -> None:
    """
    Main execution function for the Is Bad Flag node.
    
    This function orchestrates the entire processing pipeline:
    1. Read input table from KNIME
    2. Validate that the required GRODI26_wRI column exists
    3. Create the binary isBad column
    4. Reorder columns to put isBad first
    5. Log summary statistics
    6. Write output table to KNIME
    
    All steps are wrapped with debug logging for complete traceability
    (when DEBUG=True).
    
    ERROR HANDLING:
    ---------------
    If any step fails, the exception is caught, logged with full details,
    and then re-raised so KNIME can display the error to the user.
    
    DEBUG LOGGING (when enabled):
    -----------------------------
    - Execution start and end times
    - Total execution duration
    - Step-by-step progress indicators
    - Full traceback on any error
    """
    
    if DEBUG:
        logger.info("="*70)
        logger.info("IS BAD FLAG NODE - COMMENTATED TOGGLE DEBUG VERSION - STARTING EXECUTION")
        logger.info("="*70)
    
    execution_start = datetime.now()
    if DEBUG:
        logger.debug(f"Execution started at: {execution_start}")
    
    try:
        # =====================================================================
        # STEP 1: READ INPUT TABLE
        # =====================================================================
        if DEBUG:
            logger.info("[STEP 1/5] Reading input table...")
        df = read_input_table()
        
        # =====================================================================
        # STEP 2: VALIDATE REQUIRED COLUMN
        # =====================================================================
        if DEBUG:
            logger.info("[STEP 2/5] Validating required column...")
        validate_required_column(df, "GRODI26_wRI")
        
        # =====================================================================
        # STEP 3: CREATE isBad COLUMN
        # =====================================================================
        if DEBUG:
            logger.info("[STEP 3/5] Creating isBad column...")
        df = create_is_bad_column(df, "GRODI26_wRI", "isBad")
        
        # =====================================================================
        # STEP 4: REORDER COLUMNS
        # =====================================================================
        if DEBUG:
            logger.info("[STEP 4/5] Reordering columns...")
        df = reorder_columns(df, "isBad")
        
        # =====================================================================
        # STEP 5: LOG SUMMARY AND WRITE OUTPUT
        # =====================================================================
        if DEBUG:
            logger.info("[STEP 5/5] Writing output...")
        log_summary_statistics(df, "isBad", "GRODI26_wRI")
        write_output_table(df)
        
        # =====================================================================
        # LOG SUCCESSFUL COMPLETION
        # =====================================================================
        if DEBUG:
            execution_end = datetime.now()
            execution_duration = (execution_end - execution_start).total_seconds()
            
            logger.info("="*70)
            logger.info("IS BAD FLAG NODE - EXECUTION COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {execution_duration:.4f} seconds")
            logger.info("="*70)
        
    except Exception as e:
        # =====================================================================
        # LOG EXECUTION FAILURE
        # =====================================================================
        execution_end = datetime.now()
        execution_duration = (execution_end - execution_start).total_seconds()
        
        if DEBUG:
            logger.error("="*70)
            logger.error("IS BAD FLAG NODE - EXECUTION FAILED")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Time before failure: {execution_duration:.4f} seconds")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error("="*70)
        
        # Re-raise the exception so KNIME displays the error
        raise


# =============================================================================
# SECTION 6: SCRIPT ENTRY POINT
# =============================================================================

# Execute main function
# In KNIME, __name__ is not "__main__", but we want to run regardless
if __name__ == "__main__":
    main()
else:
    # When run in KNIME, __name__ is the module name, not "__main__"
    main()


# =============================================================================
# END OF SCRIPT - TOGGLE DEBUG VERSION
# =============================================================================
#
# SUMMARY OF WHAT THIS SCRIPT DOES:
# 1. Reads input data from KNIME (first input port)
# 2. Validates that the required GRODI26_wRI column exists
# 3. Creates a binary isBad column (1 = bad, 0 = good)
# 4. Reorders columns to put isBad first
# 5. Prints a summary of the bad rate to the console
# 6. Writes the result back to KNIME (first output port)
#
# DEBUG TOGGLE FEATURES:
# - Set DEBUG = True at the top to enable:
#   - Comprehensive logging on every function entry/exit
#   - Argument and return value logging
#   - DataFrame details (shape, types, nulls) at each step
#   - Execution timing for performance analysis
#   - Full tracebacks on any error
# - Set DEBUG = False for production use (minimal output)
#
# TYPICAL USE IN A WORKFLOW:
# This node would typically come early in a credit risk modeling pipeline,
# right after data loading. The isBad column created here becomes the
# target variable (dependent variable) for:
# - WOE binning (to calculate Weight of Evidence)
# - Variable selection (to find predictive features)
# - Logistic regression (to build the scorecard model)
# - Model evaluation (to measure AUC, K-S, etc.)
#
# =============================================================================

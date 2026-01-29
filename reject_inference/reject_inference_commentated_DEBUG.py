# =============================================================================
# KNIME Python Script - Reject Inference Column Generation (DEBUG VERSION)
# =============================================================================
# 
# PURPOSE:
# This script performs "Reject Inference" - a credit risk modeling technique used
# to estimate the likely outcomes (default or non-default) for loan applications
# that were REJECTED and therefore have no actual performance data.
#
# In credit risk modeling, we have two populations:
#   1. APPROVED applications - these have actual outcomes (did they default or not?)
#   2. REJECTED applications - these have no outcomes (we never gave them a loan)
#
# Reject Inference attempts to infer what would have happened if we HAD approved
# the rejected applications, using probability scores from a model.
#
# DEBUG VERSION NOTES:
# This version includes extensive debug logging on every function to help
# troubleshoot issues when running inside KNIME Python Script nodes.
#
# COMPATIBILITY:
# - KNIME Version: 5.9
# - Python Version: 3.9.23
# - Platform: Windows
#
# REQUIRED INPUT COLUMNS:
# - LoanID: Identifier for the loan (missing/empty means rejected application)
# - IsFPD: "Is First Payment Default" - actual default flag for approved loans
# - FPD: Alternative default flag column
# - expected_DefaultRate2: Model-predicted probability of default
# - FRODI26: Some fraud/risk indicator metric
# - GRODI26: Another fraud/risk indicator metric
#
# OUTPUT COLUMNS CREATED:
# - isFPD_wRI: Default flag "with Reject Inference" applied
# - FRODI26_wRI: FRODI26 "with Reject Inference" applied  
# - GRODI26_wRI: GRODI26 "with Reject Inference" applied
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------

# Import the KNIME scripting interface - this is the bridge between Python and KNIME.
# The 'knio' module provides access to input/output tables and flow variables.
# This is a KNIME-specific module that only works inside KNIME Python Script nodes.
import knime.scripting.io as knio

# Import pandas - the primary data manipulation library in Python.
# Pandas provides DataFrame objects (similar to Excel spreadsheets or SQL tables)
# that make it easy to work with tabular data.
import pandas as pd

# Import numpy - the fundamental package for numerical computing in Python.
# NumPy provides fast array operations and mathematical functions.
# Here we use it specifically for generating random numbers.
import numpy as np

# Import logging - Python's built-in logging framework for debug output
import logging

# Import sys - for system-specific parameters and functions (stdout access)
import sys

# Import datetime - for timestamping log entries
from datetime import datetime

# Import functools.wraps - for preserving function metadata in decorators
from functools import wraps

# Import traceback - for detailed error stack traces
import traceback

# =============================================================================
# SECTION 1A: DEBUG LOGGING SETUP
# =============================================================================
# 
# Set up a comprehensive logging system that will output debug information
# to the console (which KNIME captures and displays in the node output).
# =============================================================================

def setup_logging():
    """
    Configure logging for debug output.
    
    Creates a logger with:
    - DEBUG level to capture all messages
    - Console handler to output to stdout
    - Formatted output with timestamps, log levels, and function names
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create a custom logger with a unique name
    # Using a unique name prevents conflicts with other loggers in the system
    logger = logging.getLogger('RejectInference_Commentated_DEBUG')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to avoid duplicate log messages
    # This is important when the script might be run multiple times
    logger.handlers = []
    
    # Create console handler that writes to standard output
    # KNIME captures stdout and displays it in the node's console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create a formatter that includes useful context in each log message:
    # - %(asctime)s: Timestamp when the log was created
    # - %(levelname)s: Log level (DEBUG, INFO, WARNING, ERROR)
    # - %(funcName)s: Name of the function where log was called
    # - %(lineno)d: Line number where log was called
    # - %(message)s: The actual log message
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger

# Initialize the logger at module level so it's available to all functions
logger = setup_logging()

# =============================================================================
# SECTION 1B: DEBUG DECORATOR FOR FUNCTION TRACING
# =============================================================================
#
# A decorator is a function that wraps another function to add behavior.
# This decorator adds logging at the entry and exit of every function,
# including timing information and error handling.
# =============================================================================

def debug_trace(func):
    """
    Decorator that logs function entry, exit, and execution time.
    
    When applied to a function, it will:
    1. Log when the function is entered (with argument counts)
    2. Time how long the function takes to execute
    3. Log when the function exits (with success/failure status)
    4. Log full exception details if the function raises an error
    
    Usage:
        @debug_trace
        def my_function(arg1, arg2):
            # function code here
    
    Args:
        func: The function to wrap
        
    Returns:
        function: Wrapped function with logging
    """
    @wraps(func)  # Preserves the original function's metadata (name, docstring)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Log function entry with a visual separator
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        logger.debug(f"Arguments: args count={len(args)}, kwargs keys={list(kwargs.keys())}")
        
        # Record start time for elapsed time calculation
        start_time = datetime.now()
        
        try:
            # Execute the actual function
            result = func(*args, **kwargs)
            
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Log successful completion
            logger.debug(f"EXITING: {func_name} (SUCCESS) - Elapsed: {elapsed:.4f}s")
            
            return result
            
        except Exception as e:
            # Calculate elapsed time even on failure
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Log error details including full traceback
            logger.error(f"EXITING: {func_name} (ERROR) - Elapsed: {elapsed:.4f}s")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # Re-raise the exception so it propagates normally
            raise
            
    return wrapper

# =============================================================================
# SECTION 1C: DEBUG UTILITY FUNCTIONS
# =============================================================================
#
# Utility functions for logging detailed information about DataFrames,
# columns, and boolean masks. These help understand data state at each step.
# =============================================================================

@debug_trace
def log_dataframe_info(df, name="DataFrame"):
    """
    Log detailed information about a DataFrame.
    
    Outputs:
    - Shape (rows x columns)
    - Memory usage
    - Column names and data types
    - Null counts per column
    - First few rows for inspection
    
    Args:
        df: pandas DataFrame to inspect
        name: Descriptive name for the DataFrame (for log messages)
        
    Returns:
        DataFrame: Returns the input DataFrame unchanged (for chaining)
    """
    logger.debug(f"--- {name} Info ---")
    logger.debug(f"  Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})")
    
    # Memory usage with deep introspection (includes object column content sizes)
    memory_kb = df.memory_usage(deep=True).sum() / 1024
    logger.debug(f"  Memory usage: {memory_kb:.2f} KB")
    
    # List all column names
    logger.debug(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Log data type and null count for each column
    logger.debug(f"  Data types and null counts:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        logger.debug(f"    - {col}: {df[col].dtype} (nulls: {null_count}, {null_pct:.2f}%)")
    
    # Show first few rows if DataFrame is not empty
    if len(df) > 0:
        logger.debug(f"  First 3 rows:\n{df.head(3).to_string()}")
    else:
        logger.debug("  DataFrame is empty!")
    
    return df

@debug_trace
def log_column_stats(df, column_name):
    """
    Log detailed statistics for a specific column.
    
    For all columns:
    - Data type
    - Non-null and null counts
    - Number of unique values
    - Value counts (top 10)
    
    For numeric columns additionally:
    - Min, max, mean, median, standard deviation
    
    Args:
        df: pandas DataFrame containing the column
        column_name: Name of the column to analyze
    """
    # Check if column exists
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in DataFrame!")
        logger.warning(f"  Available columns: {list(df.columns)}")
        return
    
    col = df[column_name]
    
    logger.debug(f"--- Column Stats: {column_name} ---")
    logger.debug(f"  Dtype: {col.dtype}")
    logger.debug(f"  Non-null count: {col.notna().sum()}")
    logger.debug(f"  Null count: {col.isna().sum()}")
    logger.debug(f"  Unique values: {col.nunique()}")
    
    # Additional stats for numeric columns
    if pd.api.types.is_numeric_dtype(col):
        # Use .mean(), .min(), etc. which ignore NaN values
        logger.debug(f"  Min: {col.min()}")
        logger.debug(f"  Max: {col.max()}")
        logger.debug(f"  Mean: {col.mean()}")
        logger.debug(f"  Median: {col.median()}")
        logger.debug(f"  Std: {col.std()}")
    
    # Value counts including NA as a category
    # head(10) limits to top 10 most common values
    value_counts = col.value_counts(dropna=False).head(10)
    logger.debug(f"  Top 10 value counts:\n{value_counts.to_string()}")

@debug_trace  
def log_mask_info(mask, name="Mask"):
    """
    Log information about a boolean mask (True/False series).
    
    Boolean masks are used extensively in pandas for row selection.
    This function logs how many rows are True vs False.
    
    Args:
        mask: pandas Series of boolean values
        name: Descriptive name for the mask (for log messages)
        
    Returns:
        Series: Returns the input mask unchanged (for chaining)
    """
    true_count = mask.sum()  # True = 1, False = 0, so sum = count of True
    false_count = (~mask).sum()  # Inverted mask counts False values
    total = len(mask)
    true_pct = (true_count / total) * 100 if total > 0 else 0
    
    logger.debug(f"--- Mask: {name} ---")
    logger.debug(f"  Total elements: {total}")
    logger.debug(f"  True: {true_count} ({true_pct:.2f}%)")
    logger.debug(f"  False: {false_count} ({100-true_pct:.2f}%)")
    
    return mask

# -----------------------------------------------------------------------------
# SECTION 2: READ INPUT DATA FROM KNIME (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def read_input_table():
    """
    Read input table from KNIME and log its contents.
    
    This function:
    1. Reads the first input table from KNIME
    2. Logs detailed DataFrame information
    3. Checks for required columns
    4. Logs statistics for key columns
    
    Returns:
        DataFrame: Input data from KNIME
        
    Raises:
        Exception: If reading the input table fails
    """
    logger.info("Reading input table from KNIME...")
    
    try:
        # Read the first input table (index 0) from KNIME and convert to pandas
        df = knio.input_tables[0].to_pandas()
        logger.info(f"Successfully read input table")
        
        # Log detailed DataFrame info
        log_dataframe_info(df, "Input DataFrame")
        
        # Define the columns required for this script to function
        required_columns = ['LoanID', 'IsFPD', 'FPD', 'expected_DefaultRate2', 'FRODI26', 'GRODI26']
        
        # Check which required columns are missing
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        else:
            logger.info(f"All required columns present: {required_columns}")
            
        # Log detailed statistics for each required column (if present)
        for col in required_columns:
            if col in df.columns:
                log_column_stats(df, col)
                
        return df
        
    except Exception as e:
        logger.error(f"Failed to read input table: {e}")
        raise

# -----------------------------------------------------------------------------
# SECTION 3: DEFINE HELPER FUNCTION FOR MISSING VALUE DETECTION
# -----------------------------------------------------------------------------

@debug_trace
def is_missing(value):
    """
    Check if a single value should be considered "missing".
    
    This is important because missing values can come in many forms:
      - pandas NA (pd.NA)
      - numpy NaN (np.nan)
      - Python None
      - Empty strings ("")
      - Strings with only whitespace ("   ")
    
    This function returns True if the value is any kind of missing, False otherwise.
    
    DEBUG: Logs the value being checked and the result.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if value is considered missing, False otherwise
    """
    logger.debug(f"Checking value: {repr(value)} (type: {type(value).__name__})")
    
    # First, check if the value is a pandas/numpy null value.
    # pd.isna() returns True for: None, pd.NA, np.nan, and pd.NaT (datetime null).
    if pd.isna(value):
        logger.debug(f"  -> pd.isna() returned True - value IS missing")
        return True
    
    # Second, check if the value is a string that is effectively empty.
    # isinstance(value, str) checks if the value is a text string.
    # value.strip() removes all leading and trailing whitespace from the string.
    if isinstance(value, str) and value.strip() == '':
        logger.debug(f"  -> Empty string detected - value IS missing")
        return True
    
    # If neither condition was met, the value is NOT missing.
    logger.debug(f"  -> Value is NOT missing")
    return False

@debug_trace
def is_missing_vectorized(series):
    """
    Vectorized version of is_missing for better performance on Series.
    
    Instead of calling is_missing() on each value individually (slow),
    this uses vectorized pandas operations (fast) to check an entire Series.
    
    DEBUG: Logs counts of each type of missing value found.
    
    Args:
        series: pandas Series to check
        
    Returns:
        Series of bool: True where values are considered missing
    """
    logger.debug(f"Processing series: '{series.name}', length={len(series)}, dtype={series.dtype}")
    
    # Check for pandas NA values using vectorized operation
    na_mask = series.isna()
    na_count = na_mask.sum()
    logger.debug(f"  pd.isna() matches: {na_count}")
    
    # Check for empty strings (only applies to object/string columns)
    if series.dtype == 'object':
        # Apply lambda to check each value for empty string
        empty_str_mask = series.apply(lambda x: isinstance(x, str) and x.strip() == '')
        empty_count = empty_str_mask.sum()
        logger.debug(f"  Empty string matches: {empty_count}")
        
        # Combine both masks with OR - True if EITHER condition is True
        result = na_mask | empty_str_mask
    else:
        # For non-object columns, only check NA (no string handling needed)
        result = na_mask
    
    total_missing = result.sum()
    logger.debug(f"  Total missing: {total_missing} ({(total_missing/len(series))*100:.2f}%)")
    
    return result

# -----------------------------------------------------------------------------
# SECTION 4: CREATE MASKS FOR LOANID PRESENCE (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def create_loan_id_masks(df):
    """
    Create masks for LoanID presence/absence.
    
    A "mask" in pandas is a boolean Series where each row is True or False.
    Masks are used to filter or select specific rows of a DataFrame.
    
    This function creates:
    - loan_id_present: True where LoanID has a valid value (approved loans)
    - loan_id_missing: True where LoanID is missing (rejected applications)
    
    DEBUG: Logs sample values and mask statistics.
    
    Args:
        df: DataFrame containing the LoanID column
        
    Returns:
        tuple: (loan_id_present, loan_id_missing) boolean Series
    """
    logger.info("Creating LoanID presence masks...")
    
    # Log sample values for debugging (helps understand data format)
    logger.debug("Sample LoanID values (first 10):")
    for i, val in enumerate(df['LoanID'].head(10)):
        is_miss = is_missing(val)
        logger.debug(f"  [{i}] {repr(val)} -> missing={is_miss}")
    
    # Use vectorized version for efficiency on full dataset
    loan_id_missing = is_missing_vectorized(df['LoanID'])
    loan_id_present = ~loan_id_missing  # Invert to get "present" mask
    
    # Log mask statistics
    log_mask_info(loan_id_present, "loan_id_present (approved loans)")
    log_mask_info(loan_id_missing, "loan_id_missing (rejected applications)")
    
    return loan_id_present, loan_id_missing

# -----------------------------------------------------------------------------
# SECTION 5: INITIALIZE THE NEW DEFAULT FLAG COLUMN (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def initialize_isfpd_wri(df):
    """
    Initialize the isFPD_wRI column with nullable integer type.
    
    Create a new column called 'isFPD_wRI' (IsFPD with Reject Inference).
    This column will eventually contain:
      - For approved loans (LoanID present): the actual default status
      - For rejected loans (LoanID missing): an inferred default status
    
    We use 'Int32' (capital I) not 'int32' (lowercase i) because:
      - 'Int32' is a nullable integer type that can hold missing values (pd.NA)
      - 'int32' is a regular integer that CANNOT hold missing values
    KNIME requires nullable types when columns might contain missing values.
    
    DEBUG: Logs the column initialization details.
    
    Args:
        df: DataFrame to add the column to
        
    Returns:
        DataFrame: Input DataFrame with isFPD_wRI column added
    """
    logger.info("Initializing isFPD_wRI column...")
    
    # Create array of NA values with nullable Int32 type
    df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')
    
    logger.debug(f"Created isFPD_wRI column with dtype: {df['isFPD_wRI'].dtype}")
    logger.debug(f"All values are NA: {df['isFPD_wRI'].isna().all()}")
    logger.debug(f"Column length: {len(df['isFPD_wRI'])}")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 6: STEP 1 - APPROVED LOANS: COPY ACTUAL DEFAULT STATUS
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_isfpd(df, loan_id_present):
    """
    Step 1: For approved loans, copy actual default status from IsFPD.
    
    For approved loans (where LoanID is present), we use the actual default status.
    These loans have real outcomes - we know if they actually defaulted or not.
    
    DEBUG: Logs value distributions before and after assignment.
    
    Args:
        df: DataFrame to modify
        loan_id_present: Boolean mask where True = LoanID is present (approved)
        
    Returns:
        DataFrame: Modified DataFrame with isFPD_wRI values assigned
    """
    logger.info("Step 1: Assigning isFPD_wRI from IsFPD where LoanID is present...")
    
    rows_to_assign = loan_id_present.sum()
    logger.debug(f"Number of rows to assign: {rows_to_assign}")
    
    if rows_to_assign > 0:
        # Log source value distribution
        source_values = df.loc[loan_id_present, 'IsFPD']
        logger.debug(f"Source IsFPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        # Perform the assignment
        # df.loc[mask, column] selects specific rows and columns
        df.loc[loan_id_present, 'isFPD_wRI'] = source_values
        
        # Log result value distribution
        assigned_values = df.loc[loan_id_present, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
    else:
        logger.warning("No rows to assign in Step 1 - no approved loans found!")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 7: STEP 2 - REJECTED LOANS: TRY USING FPD COLUMN FIRST
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_fpd(df, loan_id_missing):
    """
    Step 2: For rejected loans, try to use values from FPD column.
    
    For rejected loans (where LoanID is missing), first try to use the FPD column.
    The FPD column might have pre-populated values for some rejected applications.
    This could be from previous reject inference runs or external data sources.
    
    DEBUG: Logs value distributions before and after assignment.
    
    Args:
        df: DataFrame to modify
        loan_id_missing: Boolean mask where True = LoanID is missing (rejected)
        
    Returns:
        DataFrame: Modified DataFrame with isFPD_wRI values assigned where FPD had values
    """
    logger.info("Step 2: Assigning isFPD_wRI from FPD where LoanID is missing...")
    
    rows_to_assign = loan_id_missing.sum()
    logger.debug(f"Number of rejected applications: {rows_to_assign}")
    
    if rows_to_assign > 0:
        # Log source value distribution
        source_values = df.loc[loan_id_missing, 'FPD']
        logger.debug(f"Source FPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        # Perform the assignment
        # If FPD is missing for a row, isFPD_wRI will remain missing
        df.loc[loan_id_missing, 'isFPD_wRI'] = source_values
        
        # Log result value distribution
        assigned_values = df.loc[loan_id_missing, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
        
        # Log how many still need values
        still_missing = df.loc[loan_id_missing, 'isFPD_wRI'].isna().sum()
        logger.debug(f"Rows still needing values after FPD assignment: {still_missing}")
    else:
        logger.warning("No rows to assign in Step 2 - no rejected applications found!")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 8: STEP 3 - REJECTED LOANS: PROBABILISTIC INFERENCE FOR REMAINING
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_probability(df, loan_id_missing):
    """
    Step 3: Handle remaining missing values using probabilistic inference.
    
    Some rejected loans might still have missing isFPD_wRI values after Step 2.
    This happens when both LoanID is missing AND FPD is missing.
    For these cases, we use probabilistic reject inference based on model scores.
    
    CORE LOGIC:
    If random_value <= expected_default_rate, assign 1 (default), else 0.
    
    Why this works:
    - If expected_rate = 0.30 (30% default probability)
    - random_value is uniformly distributed from 0 to 1
    - There's a 30% chance random_value will be <= 0.30
    - So 30% of such cases will be assigned "default" (1)
    - This matches the expected default rate!
    
    DEBUG: Logs all intermediate calculations and assignment results.
    
    Args:
        df: DataFrame to modify
        loan_id_missing: Boolean mask where True = LoanID is missing (rejected)
        
    Returns:
        DataFrame: Modified DataFrame with probabilistic isFPD_wRI values assigned
    """
    logger.info("Step 3: Probabilistic inference for remaining missing values...")
    
    # Create mask for rows that STILL need values
    still_missing_mask = loan_id_missing & df['isFPD_wRI'].isna()
    
    # Create mask for rows that have an expected default rate
    has_expected_rate = df['expected_DefaultRate2'].notna()
    
    # Combine: only process rows that need values AND have a probability
    rows_to_process = still_missing_mask & has_expected_rate
    
    # Log mask details
    log_mask_info(still_missing_mask, "still_missing_mask (rejected, no FPD)")
    log_mask_info(has_expected_rate, "has_expected_rate (has probability)")
    log_mask_info(rows_to_process, "rows_to_process (intersection)")
    
    if rows_to_process.any():
        num_rows = rows_to_process.sum()
        logger.info(f"Processing {num_rows} rows with probabilistic inference")
        
        # Generate random values for each row
        random_values = np.random.random(num_rows)
        logger.debug(f"Random values generated: {num_rows}")
        logger.debug(f"  Min: {random_values.min():.4f}")
        logger.debug(f"  Max: {random_values.max():.4f}")
        logger.debug(f"  Mean: {random_values.mean():.4f}")
        
        # Get the expected default rates for these rows
        expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
        logger.debug(f"Expected default rates:")
        logger.debug(f"  Min: {expected_rates.min():.4f}")
        logger.debug(f"  Max: {expected_rates.max():.4f}")
        logger.debug(f"  Mean: {expected_rates.mean():.4f}")
        
        # Apply the probabilistic assignment
        # (random_values <= expected_rates) produces boolean array
        # .astype(int) converts True->1, False->0
        assigned_values = (random_values <= expected_rates).astype(int)
        
        ones_count = assigned_values.sum()
        zeros_count = len(assigned_values) - ones_count
        logger.debug(f"Assignment results:")
        logger.debug(f"  Assigned 1 (default): {ones_count} ({ones_count/num_rows*100:.2f}%)")
        logger.debug(f"  Assigned 0 (non-default): {zeros_count} ({zeros_count/num_rows*100:.2f}%)")
        
        # Write the assigned values back to the DataFrame
        df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values
        
        logger.info(f"Completed probabilistic inference for {num_rows} rows")
    else:
        logger.info("No rows require probabilistic inference")
    
    # Check for any remaining missing values
    remaining_missing = loan_id_missing & df['isFPD_wRI'].isna()
    if remaining_missing.any():
        logger.warning(f"WARNING: {remaining_missing.sum()} rejected rows still have no isFPD_wRI value!")
        logger.warning("  These rows are missing both FPD and expected_DefaultRate2")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 9: ENSURE PROPER DATA TYPE FOR isFPD_wRI
# -----------------------------------------------------------------------------

@debug_trace
def finalize_isfpd_wri(df):
    """
    Ensure isFPD_wRI column is nullable integer type and log final stats.
    
    Even though we initialized it as Int32, operations might have changed it.
    This explicit cast ensures KNIME will receive the expected data type.
    
    DEBUG: Logs final column statistics.
    
    Args:
        df: DataFrame with isFPD_wRI column
        
    Returns:
        DataFrame: DataFrame with properly typed isFPD_wRI column
    """
    logger.info("Finalizing isFPD_wRI column...")
    
    # Log before type conversion
    logger.debug(f"Current dtype: {df['isFPD_wRI'].dtype}")
    
    # Force convert to Int32 (nullable integer)
    df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')
    
    # Log after type conversion
    logger.debug(f"Final dtype: {df['isFPD_wRI'].dtype}")
    
    # Log comprehensive column statistics
    log_column_stats(df, 'isFPD_wRI')
    
    return df

# =============================================================================
# SECTION 10: FRODI26_wRI AND GRODI26_wRI COLUMN GENERATION
# =============================================================================

@debug_trace
def calculate_group_averages(df):
    """
    Calculate average FRODI26 and GRODI26 grouped by IsFPD.
    
    We use the ORIGINAL IsFPD column (not isFPD_wRI) because we want averages
    based on ACTUAL outcomes from approved loans, not inferred ones.
    
    These averages are used to impute missing values in rejected applications.
    
    DEBUG: Logs all calculated averages and warns about NaN values.
    
    Args:
        df: DataFrame containing IsFPD, FRODI26, GRODI26 columns
        
    Returns:
        tuple: (avg_frodi26_fpd1, avg_frodi26_fpd0, avg_grodi26_fpd1, avg_grodi26_fpd0)
    """
    logger.info("Calculating group averages for FRODI26 and GRODI26...")
    
    # Create masks for IsFPD values
    fpd1_mask = df['IsFPD'] == 1  # Defaulters
    fpd0_mask = df['IsFPD'] == 0  # Non-defaulters
    
    logger.debug(f"IsFPD == 1 (defaulters): {fpd1_mask.sum()} rows")
    logger.debug(f"IsFPD == 0 (non-defaulters): {fpd0_mask.sum()} rows")
    
    # Calculate averages for FRODI26
    avg_frodi26_fpd1 = df.loc[fpd1_mask, 'FRODI26'].mean()
    avg_frodi26_fpd0 = df.loc[fpd0_mask, 'FRODI26'].mean()
    
    # Calculate averages for GRODI26
    avg_grodi26_fpd1 = df.loc[fpd1_mask, 'GRODI26'].mean()
    avg_grodi26_fpd0 = df.loc[fpd0_mask, 'GRODI26'].mean()
    
    # Log the calculated averages
    logger.info(f"Calculated averages:")
    logger.info(f"  avg_frodi26_fpd1 (defaulters): {avg_frodi26_fpd1}")
    logger.info(f"  avg_frodi26_fpd0 (non-defaulters): {avg_frodi26_fpd0}")
    logger.info(f"  avg_grodi26_fpd1 (defaulters): {avg_grodi26_fpd1}")
    logger.info(f"  avg_grodi26_fpd0 (non-defaulters): {avg_grodi26_fpd0}")
    
    # Warn about NaN averages (indicates no data in that group)
    if pd.isna(avg_frodi26_fpd1):
        logger.warning("avg_frodi26_fpd1 is NaN - no IsFPD=1 rows with FRODI26 values!")
    if pd.isna(avg_frodi26_fpd0):
        logger.warning("avg_frodi26_fpd0 is NaN - no IsFPD=0 rows with FRODI26 values!")
    if pd.isna(avg_grodi26_fpd1):
        logger.warning("avg_grodi26_fpd1 is NaN - no IsFPD=1 rows with GRODI26 values!")
    if pd.isna(avg_grodi26_fpd0):
        logger.warning("avg_grodi26_fpd0 is NaN - no IsFPD=0 rows with GRODI26 values!")
    
    return avg_frodi26_fpd1, avg_frodi26_fpd0, avg_grodi26_fpd1, avg_grodi26_fpd0

# -----------------------------------------------------------------------------
# SECTION 12: CREATE FRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

@debug_trace
def create_frodi26_wri(df, avg_frodi26_fpd1, avg_frodi26_fpd0):
    """
    Create FRODI26_wRI column with imputation for missing values.
    
    Logic:
    - If FRODI26 has a value: copy it to FRODI26_wRI
    - If FRODI26 is missing AND isFPD_wRI = 1: use avg_frodi26_fpd1
    - If FRODI26 is missing AND isFPD_wRI = 0: use avg_frodi26_fpd0
    
    DEBUG: Logs mask details and imputation counts.
    
    Args:
        df: DataFrame to modify
        avg_frodi26_fpd1: Average FRODI26 for defaulters
        avg_frodi26_fpd0: Average FRODI26 for non-defaulters
        
    Returns:
        DataFrame: Modified DataFrame with FRODI26_wRI column
    """
    logger.info("Creating FRODI26_wRI column...")
    
    # Create masks for missing/present FRODI26 values
    frodi26_missing = is_missing_vectorized(df['FRODI26'])
    frodi26_present = ~frodi26_missing
    
    log_mask_info(frodi26_present, "frodi26_present")
    log_mask_info(frodi26_missing, "frodi26_missing")
    
    # Initialize with NA
    df['FRODI26_wRI'] = pd.NA
    logger.debug("Initialized FRODI26_wRI with NA")
    
    # Copy existing values where present
    present_count = frodi26_present.sum()
    df.loc[frodi26_present, 'FRODI26_wRI'] = df.loc[frodi26_present, 'FRODI26']
    logger.debug(f"Copied {present_count} existing FRODI26 values")
    
    # Impute missing values based on isFPD_wRI
    fpd1_impute_mask = frodi26_missing & (df['isFPD_wRI'] == 1)
    fpd0_impute_mask = frodi26_missing & (df['isFPD_wRI'] == 0)
    
    fpd1_impute_count = fpd1_impute_mask.sum()
    fpd0_impute_count = fpd0_impute_mask.sum()
    
    logger.debug(f"Rows to impute with avg_frodi26_fpd1 ({avg_frodi26_fpd1}): {fpd1_impute_count}")
    logger.debug(f"Rows to impute with avg_frodi26_fpd0 ({avg_frodi26_fpd0}): {fpd0_impute_count}")
    
    df.loc[fpd1_impute_mask, 'FRODI26_wRI'] = avg_frodi26_fpd1
    df.loc[fpd0_impute_mask, 'FRODI26_wRI'] = avg_frodi26_fpd0
    
    # Convert to Float64 (nullable float type for KNIME)
    df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')
    
    # Log final column statistics
    log_column_stats(df, 'FRODI26_wRI')
    
    return df

# -----------------------------------------------------------------------------
# SECTION 13: CREATE GRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

@debug_trace
def create_grodi26_wri(df, avg_grodi26_fpd1, avg_grodi26_fpd0):
    """
    Create GRODI26_wRI column with imputation for missing values.
    
    Logic:
    - If GRODI26 has a value: copy it to GRODI26_wRI
    - If GRODI26 is missing AND isFPD_wRI = 1: use avg_grodi26_fpd1
    - If GRODI26 is missing AND isFPD_wRI = 0: use avg_grodi26_fpd0
    
    DEBUG: Logs mask details and imputation counts.
    
    Args:
        df: DataFrame to modify
        avg_grodi26_fpd1: Average GRODI26 for defaulters
        avg_grodi26_fpd0: Average GRODI26 for non-defaulters
        
    Returns:
        DataFrame: Modified DataFrame with GRODI26_wRI column
    """
    logger.info("Creating GRODI26_wRI column...")
    
    # Create masks for missing/present GRODI26 values
    grodi26_missing = is_missing_vectorized(df['GRODI26'])
    grodi26_present = ~grodi26_missing
    
    log_mask_info(grodi26_present, "grodi26_present")
    log_mask_info(grodi26_missing, "grodi26_missing")
    
    # Initialize with NA
    df['GRODI26_wRI'] = pd.NA
    logger.debug("Initialized GRODI26_wRI with NA")
    
    # Copy existing values where present
    present_count = grodi26_present.sum()
    df.loc[grodi26_present, 'GRODI26_wRI'] = df.loc[grodi26_present, 'GRODI26']
    logger.debug(f"Copied {present_count} existing GRODI26 values")
    
    # Impute missing values based on isFPD_wRI
    fpd1_impute_mask = grodi26_missing & (df['isFPD_wRI'] == 1)
    fpd0_impute_mask = grodi26_missing & (df['isFPD_wRI'] == 0)
    
    fpd1_impute_count = fpd1_impute_mask.sum()
    fpd0_impute_count = fpd0_impute_mask.sum()
    
    logger.debug(f"Rows to impute with avg_grodi26_fpd1 ({avg_grodi26_fpd1}): {fpd1_impute_count}")
    logger.debug(f"Rows to impute with avg_grodi26_fpd0 ({avg_grodi26_fpd0}): {fpd0_impute_count}")
    
    df.loc[fpd1_impute_mask, 'GRODI26_wRI'] = avg_grodi26_fpd1
    df.loc[fpd0_impute_mask, 'GRODI26_wRI'] = avg_grodi26_fpd0
    
    # Convert to Float64 (nullable float type for KNIME)
    df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')
    
    # Log final column statistics
    log_column_stats(df, 'GRODI26_wRI')
    
    return df

# =============================================================================
# SECTION 14: REORDER COLUMNS FOR BETTER ORGANIZATION
# =============================================================================

@debug_trace
def reorder_columns(df):
    """
    Reorder columns to place new _wRI columns after their source columns.
    
    This makes the output easier to read and understand:
      - isFPD_wRI appears right after IsFPD
      - FRODI26_wRI appears right after FRODI26
      - GRODI26_wRI appears right after GRODI26
    
    DEBUG: Logs column order before and after reordering.
    
    Args:
        df: DataFrame to reorder
        
    Returns:
        DataFrame: Reordered DataFrame
    """
    logger.info("Reordering columns...")
    
    # Log original column order
    original_cols = df.columns.tolist()
    logger.debug(f"Original column order ({len(original_cols)} columns): {original_cols}")
    
    # Get mutable copy of column list
    cols = df.columns.tolist()
    
    # Remove new columns from their current positions (at the end)
    new_cols = ['isFPD_wRI', 'FRODI26_wRI', 'GRODI26_wRI']
    for col in new_cols:
        if col in cols:
            cols.remove(col)
            logger.debug(f"Removed '{col}' from position {original_cols.index(col)}")
        else:
            logger.warning(f"Column '{col}' not found for reordering!")
    
    # Insert isFPD_wRI after IsFPD
    try:
        isfpd_index = cols.index('IsFPD')
        cols.insert(isfpd_index + 1, 'isFPD_wRI')
        logger.debug(f"Inserted 'isFPD_wRI' after 'IsFPD' at position {isfpd_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'IsFPD' column: {e}")
        cols.append('isFPD_wRI')
        logger.warning("Appended 'isFPD_wRI' at end instead")
    
    # Insert FRODI26_wRI after FRODI26
    try:
        frodi26_index = cols.index('FRODI26')
        cols.insert(frodi26_index + 1, 'FRODI26_wRI')
        logger.debug(f"Inserted 'FRODI26_wRI' after 'FRODI26' at position {frodi26_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'FRODI26' column: {e}")
        cols.append('FRODI26_wRI')
        logger.warning("Appended 'FRODI26_wRI' at end instead")
    
    # Insert GRODI26_wRI after GRODI26
    try:
        grodi26_index = cols.index('GRODI26')
        cols.insert(grodi26_index + 1, 'GRODI26_wRI')
        logger.debug(f"Inserted 'GRODI26_wRI' after 'GRODI26' at position {grodi26_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'GRODI26' column: {e}")
        cols.append('GRODI26_wRI')
        logger.warning("Appended 'GRODI26_wRI' at end instead")
    
    # Apply the new column order
    df = df[cols]
    
    # Log final column order
    logger.debug(f"Final column order ({len(cols)} columns): {cols}")
    
    return df

# =============================================================================
# SECTION 15: OUTPUT RESULTS TO KNIME
# =============================================================================

@debug_trace
def write_output_table(df):
    """
    Write output table to KNIME and log final state.
    
    This function:
    1. Logs detailed DataFrame information
    2. Validates the output (checks for infinite values, required columns)
    3. Writes the DataFrame to KNIME's output port
    
    DEBUG: Logs comprehensive output validation.
    
    Args:
        df: Final DataFrame to output
    """
    logger.info("Writing output table to KNIME...")
    
    # Log final DataFrame info
    log_dataframe_info(df, "Output DataFrame")
    
    # Validate output
    logger.info("Validating output before writing...")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Convert to float and fill NA to check for inf
        inf_count = np.isinf(df[col].astype(float).fillna(0)).sum()
        if inf_count > 0:
            logger.warning(f"Column '{col}' contains {inf_count} infinite values!")
    
    # Check that all new columns exist
    new_cols = ['isFPD_wRI', 'FRODI26_wRI', 'GRODI26_wRI']
    for col in new_cols:
        if col in df.columns:
            logger.info(f"  OK: Column '{col}' present (dtype: {df[col].dtype})")
        else:
            logger.error(f"  MISSING: Column '{col}' not in output!")
    
    # Write to KNIME
    try:
        knio.output_tables[0] = knio.Table.from_pandas(df)
        logger.info("Successfully wrote output table to KNIME")
    except Exception as e:
        logger.error(f"Failed to write output table: {e}")
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

@debug_trace
def main():
    """
    Main execution function with full debug logging.
    
    Orchestrates all processing steps:
    1. Read input data
    2. Create isFPD_wRI column (with 3 sub-steps)
    3. Create FRODI26_wRI and GRODI26_wRI columns
    4. Reorder columns
    5. Write output
    
    DEBUG: Logs overall progress and phase boundaries.
    """
    # Banner header
    logger.info("=" * 70)
    logger.info("REJECT INFERENCE SCRIPT - DEBUG VERSION (COMMENTATED)")
    logger.info("=" * 70)
    logger.info(f"Execution started at: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info("=" * 70)
    
    # Phase 0: Read input
    df = read_input_table()
    
    # Phase 1: Create isFPD_wRI column
    logger.info("-" * 70)
    logger.info("PHASE 1: Creating isFPD_wRI column")
    logger.info("-" * 70)
    
    loan_id_present, loan_id_missing = create_loan_id_masks(df)
    df = initialize_isfpd_wri(df)
    df = assign_isfpd_from_isfpd(df, loan_id_present)
    df = assign_isfpd_from_fpd(df, loan_id_missing)
    df = assign_isfpd_from_probability(df, loan_id_missing)
    df = finalize_isfpd_wri(df)
    
    # Phase 2: Create FRODI26_wRI and GRODI26_wRI columns
    logger.info("-" * 70)
    logger.info("PHASE 2: Creating FRODI26_wRI and GRODI26_wRI columns")
    logger.info("-" * 70)
    
    avg_frodi26_fpd1, avg_frodi26_fpd0, avg_grodi26_fpd1, avg_grodi26_fpd0 = calculate_group_averages(df)
    df = create_frodi26_wri(df, avg_frodi26_fpd1, avg_frodi26_fpd0)
    df = create_grodi26_wri(df, avg_grodi26_fpd1, avg_grodi26_fpd0)
    
    # Phase 3: Reorder columns
    logger.info("-" * 70)
    logger.info("PHASE 3: Reordering columns")
    logger.info("-" * 70)
    
    df = reorder_columns(df)
    
    # Phase 4: Write output
    logger.info("-" * 70)
    logger.info("PHASE 4: Writing output")
    logger.info("-" * 70)
    
    write_output_table(df)
    
    # Completion banner
    logger.info("=" * 70)
    logger.info(f"Execution completed at: {datetime.now().isoformat()}")
    logger.info("REJECT INFERENCE SCRIPT - COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Execute main function
# This works both when run as a module (__name__ == "__main__") and
# when run directly in KNIME (where __name__ may not be "__main__")
if __name__ == "__main__":
    main()
else:
    # When run directly in KNIME (not as module)
    main()

# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# SUMMARY OF WHAT THIS DEBUG SCRIPT DOES:
#
# 1. Sets up comprehensive logging that outputs to KNIME's console
#
# 2. Wraps all functions with debug_trace decorator for automatic
#    entry/exit logging with timing
#
# 3. Reads input table and logs detailed DataFrame info
#
# 4. Creates isFPD_wRI (Is First Payment Default with Reject Inference):
#    - For approved loans: uses actual default status (IsFPD)
#    - For rejected loans: first tries FPD column, then uses probabilistic
#      inference based on expected_DefaultRate2
#    - Logs value distributions and mask statistics at each step
#
# 5. Creates FRODI26_wRI and GRODI26_wRI:
#    - Preserves original values where they exist
#    - For missing values: imputes with averages from approved loans
#    - Logs imputation counts and final statistics
#
# 6. Reorders columns and logs before/after column order
#
# 7. Validates and writes output, logging any issues found
#
# USE THIS DEBUG VERSION TO:
# - Diagnose data quality issues
# - Understand processing flow
# - Track value transformations
# - Find bugs in edge cases
# - Validate column types for KNIME compatibility
# =============================================================================

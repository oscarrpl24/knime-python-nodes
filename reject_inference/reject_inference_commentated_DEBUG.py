# =============================================================================
# KNIME Python Script - Reject Inference Column Generation (DEBUG VERSION)
# =============================================================================
# 
# PURPOSE:
# This script implements "reject inference" for credit risk modeling.
# In credit scoring, we only observe outcomes (default/no default) for
# applications that were APPROVED. Rejected applications never get loans,
# so we never see their actual outcomes. This creates a biased sample.
#
# Reject inference addresses this by:
# 1. Using actual outcomes for approved applications (those with a LoanID)
# 2. Probabilistically assigning outcomes to rejected applications based
#    on their predicted default probability (expected_DefaultRate2)
#
# This script also imputes missing values for performance metrics (FRODI26,
# GRODI26) using group averages based on the inferred default status.
#
# DEBUG VERSION:
# This version includes extensive debug logging on every function to help
# troubleshoot issues when running inside KNIME Python Script nodes.
# All processing logic is identical to the original - only logging is added.
#
# COMPATIBLE WITH:
# - KNIME Version: 5.9
# - Python Version: 3.9.23
#
# INPUT:
# - Table with columns: LoanID, IsFPD, FPD, expected_DefaultRate2, FRODI26, GRODI26
#
# OUTPUT:
# - Same table with three new columns: isFPD_wRI, FRODI26_wRI, GRODI26_wRI
#
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------

# Import the KNIME scripting interface
# This module provides access to KNIME's input/output tables and flow variables
# It is the bridge between the Python script and the KNIME workflow
import knime.scripting.io as knio

# Import pandas for data manipulation
# pandas is the primary library for working with tabular data in Python
# It provides DataFrame objects (similar to Excel spreadsheets or SQL tables)
import pandas as pd

# Import numpy for numerical operations
# numpy provides efficient array operations and random number generation
# We use it here specifically for generating random values for reject inference
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
# DEBUG LOGGING SETUP
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
    logger = logging.getLogger('RejectInference_Commentated_DEBUG')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to avoid duplicate log messages
    logger.handlers = []
    
    # Create console handler that writes to standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create a formatter with useful context
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger

# Initialize the logger at module level
logger = setup_logging()

# =============================================================================
# DEBUG DECORATOR FOR FUNCTION TRACING
# =============================================================================

def debug_trace(func):
    """
    Decorator that logs function entry, exit, and execution time.
    
    When applied to a function, it will:
    1. Log when the function is entered (with argument counts)
    2. Time how long the function takes to execute
    3. Log when the function exits (with success/failure status)
    4. Log full exception details if the function raises an error
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        logger.debug(f"Arguments: args count={len(args)}, kwargs keys={list(kwargs.keys())}")
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"EXITING: {func_name} (SUCCESS) - Elapsed: {elapsed:.4f}s")
            return result
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"EXITING: {func_name} (ERROR) - Elapsed: {elapsed:.4f}s")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
            
    return wrapper

# =============================================================================
# DEBUG UTILITY FUNCTIONS
# =============================================================================

@debug_trace
def log_dataframe_info(df, name="DataFrame"):
    """
    Log detailed information about a DataFrame.
    """
    logger.debug(f"--- {name} Info ---")
    logger.debug(f"  Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})")
    memory_kb = df.memory_usage(deep=True).sum() / 1024
    logger.debug(f"  Memory usage: {memory_kb:.2f} KB")
    logger.debug(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    
    logger.debug(f"  Data types and null counts:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        logger.debug(f"    - {col}: {df[col].dtype} (nulls: {null_count}, {null_pct:.2f}%)")
    
    if len(df) > 0:
        logger.debug(f"  First 3 rows:\n{df.head(3).to_string()}")
    else:
        logger.debug("  DataFrame is empty!")
    
    return df

@debug_trace
def log_column_stats(df, column_name):
    """
    Log detailed statistics for a specific column.
    """
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in DataFrame!")
        return
    
    col = df[column_name]
    
    logger.debug(f"--- Column Stats: {column_name} ---")
    logger.debug(f"  Dtype: {col.dtype}")
    logger.debug(f"  Non-null count: {col.notna().sum()}")
    logger.debug(f"  Null count: {col.isna().sum()}")
    logger.debug(f"  Unique values: {col.nunique()}")
    
    if pd.api.types.is_numeric_dtype(col):
        logger.debug(f"  Min: {col.min()}")
        logger.debug(f"  Max: {col.max()}")
        logger.debug(f"  Mean: {col.mean()}")
        logger.debug(f"  Median: {col.median()}")
        logger.debug(f"  Std: {col.std()}")
    
    value_counts = col.value_counts(dropna=False).head(10)
    logger.debug(f"  Top 10 value counts:\n{value_counts.to_string()}")

@debug_trace  
def log_mask_info(mask, name="Mask"):
    """
    Log information about a boolean mask.
    """
    true_count = mask.sum()
    false_count = (~mask).sum()
    total = len(mask)
    true_pct = (true_count / total) * 100 if total > 0 else 0
    
    logger.debug(f"--- Mask: {name} ---")
    logger.debug(f"  Total elements: {total}")
    logger.debug(f"  True: {true_count} ({true_pct:.2f}%)")
    logger.debug(f"  False: {false_count} ({100-true_pct:.2f}%)")
    
    return mask

# -----------------------------------------------------------------------------
# SECTION 2: LOAD INPUT DATA (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def read_input_table():
    """
    Read the first input table from KNIME and convert it to a pandas DataFrame.
    
    knio.input_tables is a list of all input ports connected to this Python node
    [0] accesses the first (and in this case, only) input port
    .to_pandas() converts the KNIME table format into a pandas DataFrame
    """
    logger.info("Reading input table from KNIME...")
    
    try:
        df = knio.input_tables[0].to_pandas()
        logger.info(f"Successfully read input table")
        log_dataframe_info(df, "Input DataFrame")
        
        # Check for required columns
        required_columns = ['LoanID', 'IsFPD', 'FPD', 'expected_DefaultRate2', 'FRODI26', 'GRODI26']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        else:
            logger.info(f"All required columns present: {required_columns}")
            
        # Log stats for key columns
        for col in required_columns:
            if col in df.columns:
                log_column_stats(df, col)
                
        return df
        
    except Exception as e:
        logger.error(f"Failed to read input table: {e}")
        raise

# -----------------------------------------------------------------------------
# SECTION 3: HELPER FUNCTION FOR MISSING VALUE DETECTION
# -----------------------------------------------------------------------------

@debug_trace
def is_missing(value):
    """
    Check if a value is null, empty, or missing.
    
    This function handles multiple ways data can be "missing":
      - Standard Python/pandas null values (None, NaN, pd.NA)
      - Empty strings or strings containing only whitespace
    
    Parameters:
        value: Any value to check for missingness
    
    Returns:
        True if the value is considered missing, False otherwise
    """
    logger.debug(f"Checking value: {repr(value)} (type: {type(value).__name__})")
    
    # First check: Use pandas' isna() function to detect standard null values
    if pd.isna(value):
        logger.debug(f"  -> pd.isna() returned True - value IS missing")
        return True
    
    # Second check: Handle empty strings
    if isinstance(value, str) and value.strip() == '':
        logger.debug(f"  -> Empty string detected - value IS missing")
        return True
    
    logger.debug(f"  -> Value is NOT missing")
    return False

@debug_trace
def is_missing_vectorized(series):
    """
    Vectorized version of is_missing for better performance on Series.
    """
    logger.debug(f"Processing series: '{series.name}', length={len(series)}, dtype={series.dtype}")
    
    na_mask = series.isna()
    na_count = na_mask.sum()
    logger.debug(f"  pd.isna() matches: {na_count}")
    
    if series.dtype == 'object':
        empty_str_mask = series.apply(lambda x: isinstance(x, str) and x.strip() == '')
        empty_count = empty_str_mask.sum()
        logger.debug(f"  Empty string matches: {empty_count}")
        result = na_mask | empty_str_mask
    else:
        result = na_mask
    
    total_missing = result.sum()
    logger.debug(f"  Total missing: {total_missing} ({(total_missing/len(series))*100:.2f}%)")
    
    return result

# -----------------------------------------------------------------------------
# SECTION 4: CREATE LOAN ID PRESENCE MASKS (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def create_loan_id_masks(df):
    """
    Create boolean masks indicating where LoanID is present or missing.
    
    Result:
      - loan_id_present: True = has valid LoanID (approved application with a loan)
      - loan_id_missing: True = no LoanID (rejected application)
    """
    logger.info("Creating LoanID presence masks...")
    
    # Log sample values for debugging
    logger.debug("Sample LoanID values (first 10):")
    for i, val in enumerate(df['LoanID'].head(10)):
        is_miss = is_missing(val)
        logger.debug(f"  [{i}] {repr(val)} -> missing={is_miss}")
    
    # Use vectorized version for efficiency
    loan_id_missing = is_missing_vectorized(df['LoanID'])
    loan_id_present = ~loan_id_missing
    
    log_mask_info(loan_id_present, "loan_id_present (approved applications)")
    log_mask_info(loan_id_missing, "loan_id_missing (rejected applications)")
    
    return loan_id_present, loan_id_missing

# -----------------------------------------------------------------------------
# SECTION 5: INITIALIZE THE NEW isFPD_wRI COLUMN (WITH DEBUG LOGGING)
# -----------------------------------------------------------------------------

@debug_trace
def initialize_isfpd_wri(df):
    """
    Create a new column 'isFPD_wRI' (is First Payment Default with Reject Inference)
    initialized with all missing values (pd.NA).
    
    Uses 'Int32' (capital I) - pandas' nullable integer type that can hold pd.NA.
    """
    logger.info("Initializing isFPD_wRI column...")
    
    df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')
    
    logger.debug(f"Created isFPD_wRI column with dtype: {df['isFPD_wRI'].dtype}")
    logger.debug(f"All values are NA: {df['isFPD_wRI'].isna().all()}")
    logger.debug(f"Column length: {len(df['isFPD_wRI'])}")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 6: STEP 1 - ASSIGN VALUES FOR APPROVED APPLICATIONS
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_isfpd(df, loan_id_present):
    """
    Step 1: For approved applications (those with a LoanID), use the actual IsFPD value.
    
    When a loan was actually issued (LoanID is present), we have the real outcome:
      - IsFPD = 1 means the customer defaulted on their first payment (bad)
      - IsFPD = 0 means the customer made their first payment successfully (good)
    """
    logger.info("Step 1: Assigning isFPD_wRI from IsFPD for approved applications...")
    
    rows_to_assign = loan_id_present.sum()
    logger.debug(f"Number of approved applications: {rows_to_assign}")
    
    if rows_to_assign > 0:
        source_values = df.loc[loan_id_present, 'IsFPD']
        logger.debug(f"Source IsFPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        df.loc[loan_id_present, 'isFPD_wRI'] = source_values
        
        assigned_values = df.loc[loan_id_present, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
    else:
        logger.warning("No approved applications found!")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 7: STEP 2 - ASSIGN VALUES FROM FPD FOR REJECTED APPLICATIONS
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_fpd(df, loan_id_missing):
    """
    Step 2: For rejected applications (no LoanID), try to use the FPD column.
    
    The FPD column might contain pre-assigned values for some rejected applications.
    If FPD has a value, we use it; if not, we'll handle it in Step 3.
    """
    logger.info("Step 2: Assigning isFPD_wRI from FPD for rejected applications...")
    
    rows_to_assign = loan_id_missing.sum()
    logger.debug(f"Number of rejected applications: {rows_to_assign}")
    
    if rows_to_assign > 0:
        source_values = df.loc[loan_id_missing, 'FPD']
        logger.debug(f"Source FPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        df.loc[loan_id_missing, 'isFPD_wRI'] = source_values
        
        assigned_values = df.loc[loan_id_missing, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
        
        still_missing = df.loc[loan_id_missing, 'isFPD_wRI'].isna().sum()
        logger.debug(f"Rows still needing values after FPD assignment: {still_missing}")
    else:
        logger.warning("No rejected applications found!")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 8: STEP 3 - PROBABILISTIC INFERENCE FOR REMAINING MISSING VALUES
# -----------------------------------------------------------------------------

@debug_trace
def assign_isfpd_from_probability(df, loan_id_missing):
    """
    Step 3: Handle remaining missing values using probabilistic assignment.
    
    Uses Monte Carlo simulation based on expected_DefaultRate2:
      - If random <= expected_rate: assign 1 (default)
      - If random > expected_rate: assign 0 (no default)
    """
    logger.info("Step 3: Probabilistic inference for remaining missing values...")
    
    # Create mask for rows that STILL have missing isFPD_wRI AND are rejected
    still_missing_mask = loan_id_missing & df['isFPD_wRI'].isna()
    has_expected_rate = df['expected_DefaultRate2'].notna()
    rows_to_process = still_missing_mask & has_expected_rate
    
    log_mask_info(still_missing_mask, "still_missing_mask")
    log_mask_info(has_expected_rate, "has_expected_rate")
    log_mask_info(rows_to_process, "rows_to_process (intersection)")
    
    if rows_to_process.any():
        num_rows = rows_to_process.sum()
        logger.info(f"Processing {num_rows} rows with probabilistic inference")
        
        # Generate random values
        random_values = np.random.random(num_rows)
        logger.debug(f"Random values: min={random_values.min():.4f}, max={random_values.max():.4f}, mean={random_values.mean():.4f}")
        
        # Get expected default rates
        expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
        logger.debug(f"Expected rates: min={expected_rates.min():.4f}, max={expected_rates.max():.4f}, mean={expected_rates.mean():.4f}")
        
        # Probabilistic assignment
        assigned_values = (random_values <= expected_rates).astype(int)
        ones_count = assigned_values.sum()
        zeros_count = len(assigned_values) - ones_count
        logger.debug(f"Assignment results: 1s={ones_count} ({ones_count/num_rows*100:.2f}%), 0s={zeros_count} ({zeros_count/num_rows*100:.2f}%)")
        
        df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values
        logger.info(f"Completed probabilistic inference for {num_rows} rows")
    else:
        logger.info("No rows require probabilistic inference")
    
    # Check for any remaining missing values
    remaining_missing = loan_id_missing & df['isFPD_wRI'].isna()
    if remaining_missing.any():
        logger.warning(f"WARNING: {remaining_missing.sum()} rejected rows still have no isFPD_wRI value!")
    
    return df

# -----------------------------------------------------------------------------
# SECTION 9: ENSURE CORRECT DATA TYPE FOR isFPD_wRI
# -----------------------------------------------------------------------------

@debug_trace
def finalize_isfpd_wri(df):
    """
    Ensure the isFPD_wRI column is stored as nullable Int32.
    """
    logger.info("Finalizing isFPD_wRI column...")
    
    logger.debug(f"Current dtype: {df['isFPD_wRI'].dtype}")
    df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')
    logger.debug(f"Final dtype: {df['isFPD_wRI'].dtype}")
    
    log_column_stats(df, 'isFPD_wRI')
    
    return df

# =============================================================================
# SECTION 10-11: CALCULATE GROUP AVERAGES FOR IMPUTATION
# =============================================================================

@debug_trace
def calculate_group_averages(df):
    """
    Calculate average FRODI26 and GRODI26 values grouped by IsFPD status.
    
    These averages are used to impute missing values for rejected applications.
    """
    logger.info("Calculating group averages for FRODI26 and GRODI26...")
    
    # Count rows by IsFPD status
    fpd1_mask = df['IsFPD'] == 1
    fpd0_mask = df['IsFPD'] == 0
    
    logger.debug(f"IsFPD == 1 (defaulters): {fpd1_mask.sum()} rows")
    logger.debug(f"IsFPD == 0 (non-defaulters): {fpd0_mask.sum()} rows")
    
    # Calculate averages
    avg_frodi26_fpd1 = df.loc[fpd1_mask, 'FRODI26'].mean()
    avg_frodi26_fpd0 = df.loc[fpd0_mask, 'FRODI26'].mean()
    avg_grodi26_fpd1 = df.loc[fpd1_mask, 'GRODI26'].mean()
    avg_grodi26_fpd0 = df.loc[fpd0_mask, 'GRODI26'].mean()
    
    logger.info(f"Calculated averages:")
    logger.info(f"  avg_frodi26_fpd1 (defaulters): {avg_frodi26_fpd1}")
    logger.info(f"  avg_frodi26_fpd0 (non-defaulters): {avg_frodi26_fpd0}")
    logger.info(f"  avg_grodi26_fpd1 (defaulters): {avg_grodi26_fpd1}")
    logger.info(f"  avg_grodi26_fpd0 (non-defaulters): {avg_grodi26_fpd0}")
    
    # Warn about NaN averages
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
    
    - If FRODI26 has a value: copy it
    - If FRODI26 is missing AND isFPD_wRI = 1: use avg_frodi26_fpd1
    - If FRODI26 is missing AND isFPD_wRI = 0: use avg_frodi26_fpd0
    """
    logger.info("Creating FRODI26_wRI column...")
    
    # Create masks
    frodi26_missing = is_missing_vectorized(df['FRODI26'])
    frodi26_present = ~frodi26_missing
    
    log_mask_info(frodi26_present, "frodi26_present")
    log_mask_info(frodi26_missing, "frodi26_missing")
    
    # Initialize with NA
    df['FRODI26_wRI'] = pd.NA
    logger.debug("Initialized FRODI26_wRI with NA")
    
    # Copy existing values
    present_count = frodi26_present.sum()
    df.loc[frodi26_present, 'FRODI26_wRI'] = df.loc[frodi26_present, 'FRODI26']
    logger.debug(f"Copied {present_count} existing FRODI26 values")
    
    # Impute missing values
    fpd1_impute_mask = frodi26_missing & (df['isFPD_wRI'] == 1)
    fpd0_impute_mask = frodi26_missing & (df['isFPD_wRI'] == 0)
    
    fpd1_impute_count = fpd1_impute_mask.sum()
    fpd0_impute_count = fpd0_impute_mask.sum()
    
    logger.debug(f"Rows to impute with avg_frodi26_fpd1 ({avg_frodi26_fpd1}): {fpd1_impute_count}")
    logger.debug(f"Rows to impute with avg_frodi26_fpd0 ({avg_frodi26_fpd0}): {fpd0_impute_count}")
    
    df.loc[fpd1_impute_mask, 'FRODI26_wRI'] = avg_frodi26_fpd1
    df.loc[fpd0_impute_mask, 'FRODI26_wRI'] = avg_frodi26_fpd0
    
    # Convert to Float64
    df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')
    
    log_column_stats(df, 'FRODI26_wRI')
    
    return df

# -----------------------------------------------------------------------------
# SECTION 13: CREATE GRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

@debug_trace
def create_grodi26_wri(df, avg_grodi26_fpd1, avg_grodi26_fpd0):
    """
    Create GRODI26_wRI column with imputation for missing values.
    
    - If GRODI26 has a value: copy it
    - If GRODI26 is missing AND isFPD_wRI = 1: use avg_grodi26_fpd1
    - If GRODI26 is missing AND isFPD_wRI = 0: use avg_grodi26_fpd0
    """
    logger.info("Creating GRODI26_wRI column...")
    
    # Create masks
    grodi26_missing = is_missing_vectorized(df['GRODI26'])
    grodi26_present = ~grodi26_missing
    
    log_mask_info(grodi26_present, "grodi26_present")
    log_mask_info(grodi26_missing, "grodi26_missing")
    
    # Initialize with NA
    df['GRODI26_wRI'] = pd.NA
    logger.debug("Initialized GRODI26_wRI with NA")
    
    # Copy existing values
    present_count = grodi26_present.sum()
    df.loc[grodi26_present, 'GRODI26_wRI'] = df.loc[grodi26_present, 'GRODI26']
    logger.debug(f"Copied {present_count} existing GRODI26 values")
    
    # Impute missing values
    fpd1_impute_mask = grodi26_missing & (df['isFPD_wRI'] == 1)
    fpd0_impute_mask = grodi26_missing & (df['isFPD_wRI'] == 0)
    
    fpd1_impute_count = fpd1_impute_mask.sum()
    fpd0_impute_count = fpd0_impute_mask.sum()
    
    logger.debug(f"Rows to impute with avg_grodi26_fpd1 ({avg_grodi26_fpd1}): {fpd1_impute_count}")
    logger.debug(f"Rows to impute with avg_grodi26_fpd0 ({avg_grodi26_fpd0}): {fpd0_impute_count}")
    
    df.loc[fpd1_impute_mask, 'GRODI26_wRI'] = avg_grodi26_fpd1
    df.loc[fpd0_impute_mask, 'GRODI26_wRI'] = avg_grodi26_fpd0
    
    # Convert to Float64
    df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')
    
    log_column_stats(df, 'GRODI26_wRI')
    
    return df

# =============================================================================
# SECTION 14-19: REORDER COLUMNS FOR BETTER ORGANIZATION
# =============================================================================

@debug_trace
def reorder_columns(df):
    """
    Reorganize DataFrame columns to place new columns after their source columns.
    
    Desired order:
      - IsFPD followed by isFPD_wRI
      - FRODI26 followed by FRODI26_wRI
      - GRODI26 followed by GRODI26_wRI
    """
    logger.info("Reordering columns...")
    
    original_cols = df.columns.tolist()
    logger.debug(f"Original column order ({len(original_cols)} columns): {original_cols}")
    
    cols = df.columns.tolist()
    
    # Remove new columns from current positions
    new_cols = ['isFPD_wRI', 'FRODI26_wRI', 'GRODI26_wRI']
    for col in new_cols:
        if col in cols:
            cols.remove(col)
            logger.debug(f"Removed '{col}' from position")
        else:
            logger.warning(f"Column '{col}' not found!")
    
    # Insert isFPD_wRI after IsFPD
    try:
        isfpd_index = cols.index('IsFPD')
        cols.insert(isfpd_index + 1, 'isFPD_wRI')
        logger.debug(f"Inserted 'isFPD_wRI' after 'IsFPD' at position {isfpd_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'IsFPD' column: {e}")
        cols.append('isFPD_wRI')
    
    # Insert FRODI26_wRI after FRODI26
    try:
        frodi26_index = cols.index('FRODI26')
        cols.insert(frodi26_index + 1, 'FRODI26_wRI')
        logger.debug(f"Inserted 'FRODI26_wRI' after 'FRODI26' at position {frodi26_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'FRODI26' column: {e}")
        cols.append('FRODI26_wRI')
    
    # Insert GRODI26_wRI after GRODI26
    try:
        grodi26_index = cols.index('GRODI26')
        cols.insert(grodi26_index + 1, 'GRODI26_wRI')
        logger.debug(f"Inserted 'GRODI26_wRI' after 'GRODI26' at position {grodi26_index + 1}")
    except ValueError as e:
        logger.error(f"Cannot find 'GRODI26' column: {e}")
        cols.append('GRODI26_wRI')
    
    df = df[cols]
    
    logger.debug(f"Final column order ({len(cols)} columns): {cols}")
    
    return df

# =============================================================================
# SECTION 20: OUTPUT THE RESULTS TO KNIME
# =============================================================================

@debug_trace
def write_output_table(df):
    """
    Write the processed DataFrame to the first output port.
    """
    logger.info("Writing output table to KNIME...")
    
    log_dataframe_info(df, "Output DataFrame")
    
    # Validate output
    logger.info("Validating output...")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col].astype(float).fillna(0)).sum()
        if inf_count > 0:
            logger.warning(f"Column '{col}' contains {inf_count} infinite values!")
    
    # Check new columns exist
    new_cols = ['isFPD_wRI', 'FRODI26_wRI', 'GRODI26_wRI']
    for col in new_cols:
        if col in df.columns:
            logger.info(f"  OK: Column '{col}' present (dtype: {df[col].dtype})")
        else:
            logger.error(f"  MISSING: Column '{col}' not in output!")
    
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
    Main execution function orchestrating all processing steps.
    """
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
    
    logger.info("=" * 70)
    logger.info(f"Execution completed at: {datetime.now().isoformat()}")
    logger.info("REJECT INFERENCE SCRIPT - COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
else:
    main()

# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# SUMMARY OF WHAT THIS DEBUG SCRIPT DOES:
#
# 1. SETS UP comprehensive logging to KNIME's console output
#
# 2. READS input data and logs detailed DataFrame information
#
# 3. CREATES isFPD_wRI (First Payment Default with Reject Inference):
#    - For APPROVED applications (have LoanID): Uses actual IsFPD value
#    - For REJECTED applications (no LoanID): 
#      a. First tries to use the FPD column if available
#      b. Otherwise, uses Monte Carlo simulation based on expected_DefaultRate2
#    - LOGS value distributions at each step
#
# 4. CREATES FRODI26_wRI and GRODI26_wRI:
#    - For applications WITH existing values: Copies the original value
#    - For applications WITHOUT values: Imputes using group averages
#    - LOGS imputation counts and statistics
#
# 5. REORDERS columns and LOGS before/after column order
#
# 6. VALIDATES and OUTPUTS the enriched DataFrame to KNIME
#
# USE THIS DEBUG VERSION TO:
# - Diagnose data quality issues
# - Understand processing flow
# - Track value transformations
# - Find bugs in edge cases
# - Validate column types for KNIME compatibility
# =============================================================================

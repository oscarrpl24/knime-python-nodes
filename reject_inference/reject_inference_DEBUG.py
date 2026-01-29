# KNIME Python Script - Reject Inference Column Generation (DEBUG VERSION)
# Compatible with KNIME 5.9, Python 3.9.23
# This version includes extensive debug logging for troubleshooting

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from functools import wraps
import traceback

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging for debug output."""
    # Create a custom logger
    logger = logging.getLogger('RejectInference_DEBUG')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# DEBUG DECORATOR FOR FUNCTION TRACING
# =============================================================================

def debug_trace(func):
    """Decorator to log function entry, exit, and execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        logger.debug(f"Arguments: args={len(args)}, kwargs={list(kwargs.keys())}")
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"EXITING: {func_name} (SUCCESS) - Elapsed: {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"EXITING: {func_name} (ERROR) - Elapsed: {elapsed:.4f}s")
            logger.error(f"Exception: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
    return wrapper

# =============================================================================
# DEBUG UTILITY FUNCTIONS
# =============================================================================

@debug_trace
def log_dataframe_info(df, name="DataFrame"):
    """Log detailed information about a DataFrame."""
    logger.debug(f"--- {name} Info ---")
    logger.debug(f"  Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})")
    logger.debug(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    logger.debug(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    logger.debug(f"  Data types:")
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
    """Log statistics for a specific column."""
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
    
    # Value counts (top 10)
    value_counts = col.value_counts(dropna=False).head(10)
    logger.debug(f"  Top 10 value counts:\n{value_counts.to_string()}")

@debug_trace  
def log_mask_info(mask, name="Mask"):
    """Log information about a boolean mask."""
    true_count = mask.sum()
    false_count = (~mask).sum()
    total = len(mask)
    true_pct = (true_count / total) * 100 if total > 0 else 0
    
    logger.debug(f"--- Mask: {name} ---")
    logger.debug(f"  Total elements: {total}")
    logger.debug(f"  True: {true_count} ({true_pct:.2f}%)")
    logger.debug(f"  False: {false_count} ({100-true_pct:.2f}%)")
    
    return mask

# =============================================================================
# CORE FUNCTIONS WITH DEBUG LOGGING
# =============================================================================

@debug_trace
def is_missing(value):
    """
    Check if a value is null/empty/missing.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if value is considered missing, False otherwise
    """
    logger.debug(f"Checking value: {repr(value)} (type: {type(value).__name__})")
    
    if pd.isna(value):
        logger.debug(f"  -> pd.isna() returned True")
        return True
    if isinstance(value, str) and value.strip() == '':
        logger.debug(f"  -> Empty string detected")
        return True
    
    logger.debug(f"  -> Value is NOT missing")
    return False

@debug_trace
def is_missing_vectorized(series):
    """
    Vectorized version of is_missing for better performance on Series.
    
    Args:
        series: pandas Series to check
        
    Returns:
        Series of bool: True where values are considered missing
    """
    logger.debug(f"Processing series: {series.name}, length={len(series)}, dtype={series.dtype}")
    
    # Check for pandas NA
    na_mask = series.isna()
    na_count = na_mask.sum()
    logger.debug(f"  pd.isna() matches: {na_count}")
    
    # Check for empty strings
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

@debug_trace
def read_input_table():
    """Read input table from KNIME and log its contents."""
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

@debug_trace
def create_loan_id_masks(df):
    """Create masks for LoanID presence/absence."""
    logger.info("Creating LoanID presence masks...")
    
    # Use vectorized version for efficiency but log sample checks
    logger.debug("Sample LoanID values (first 10):")
    for i, val in enumerate(df['LoanID'].head(10)):
        is_miss = is_missing(val)
        logger.debug(f"  [{i}] {repr(val)} -> missing={is_miss}")
    
    loan_id_missing = is_missing_vectorized(df['LoanID'])
    loan_id_present = ~loan_id_missing
    
    log_mask_info(loan_id_present, "loan_id_present")
    log_mask_info(loan_id_missing, "loan_id_missing")
    
    return loan_id_present, loan_id_missing

@debug_trace
def initialize_isfpd_wri(df):
    """Initialize the isFPD_wRI column with nullable integer type."""
    logger.info("Initializing isFPD_wRI column...")
    
    df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')
    
    logger.debug(f"Created isFPD_wRI column with dtype: {df['isFPD_wRI'].dtype}")
    logger.debug(f"All values are NA: {df['isFPD_wRI'].isna().all()}")
    
    return df

@debug_trace
def assign_isfpd_from_isfpd(df, loan_id_present):
    """Step 1: When LoanID is NOT null/empty/missing, take value from IsFPD."""
    logger.info("Step 1: Assigning isFPD_wRI from IsFPD where LoanID is present...")
    
    rows_to_assign = loan_id_present.sum()
    logger.debug(f"Rows to assign: {rows_to_assign}")
    
    if rows_to_assign > 0:
        source_values = df.loc[loan_id_present, 'IsFPD']
        logger.debug(f"Source IsFPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        df.loc[loan_id_present, 'isFPD_wRI'] = source_values
        
        assigned_values = df.loc[loan_id_present, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
    else:
        logger.warning("No rows to assign in Step 1!")
    
    return df

@debug_trace
def assign_isfpd_from_fpd(df, loan_id_missing):
    """Step 2: When LoanID IS null/empty/missing, take value from FPD."""
    logger.info("Step 2: Assigning isFPD_wRI from FPD where LoanID is missing...")
    
    rows_to_assign = loan_id_missing.sum()
    logger.debug(f"Rows to assign: {rows_to_assign}")
    
    if rows_to_assign > 0:
        source_values = df.loc[loan_id_missing, 'FPD']
        logger.debug(f"Source FPD value distribution:\n{source_values.value_counts(dropna=False).to_string()}")
        
        df.loc[loan_id_missing, 'isFPD_wRI'] = source_values
        
        assigned_values = df.loc[loan_id_missing, 'isFPD_wRI']
        logger.debug(f"Assigned isFPD_wRI value distribution:\n{assigned_values.value_counts(dropna=False).to_string()}")
    else:
        logger.warning("No rows to assign in Step 2!")
    
    return df

@debug_trace
def assign_isfpd_from_probability(df, loan_id_missing):
    """Step 3: Handle remaining missing values using probabilistic inference."""
    logger.info("Step 3: Probabilistic inference for remaining missing values...")
    
    # Identify rows needing inference
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
        logger.debug(f"Random values stats: min={random_values.min():.4f}, max={random_values.max():.4f}, mean={random_values.mean():.4f}")
        
        # Get expected default rates
        expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
        logger.debug(f"Expected rates stats: min={expected_rates.min():.4f}, max={expected_rates.max():.4f}, mean={expected_rates.mean():.4f}")
        
        # Compare random to threshold
        assigned_values = (random_values <= expected_rates).astype(int)
        ones_count = assigned_values.sum()
        zeros_count = len(assigned_values) - ones_count
        logger.debug(f"Assignment results: 1s={ones_count} ({ones_count/num_rows*100:.2f}%), 0s={zeros_count} ({zeros_count/num_rows*100:.2f}%)")
        
        df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values
        
        logger.info(f"Assigned {num_rows} values via probabilistic inference")
    else:
        logger.info("No rows require probabilistic inference")
    
    return df

@debug_trace
def finalize_isfpd_wri(df):
    """Ensure isFPD_wRI column is nullable integer type and log final stats."""
    logger.info("Finalizing isFPD_wRI column...")
    
    df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')
    
    log_column_stats(df, 'isFPD_wRI')
    
    return df

@debug_trace
def calculate_group_averages(df):
    """Calculate average FRODI26 and GRODI26 grouped by IsFPD."""
    logger.info("Calculating group averages for FRODI26 and GRODI26...")
    
    # FRODI26 averages
    fpd1_mask = df['IsFPD'] == 1
    fpd0_mask = df['IsFPD'] == 0
    
    logger.debug(f"IsFPD == 1: {fpd1_mask.sum()} rows")
    logger.debug(f"IsFPD == 0: {fpd0_mask.sum()} rows")
    
    avg_frodi26_fpd1 = df.loc[fpd1_mask, 'FRODI26'].mean()
    avg_frodi26_fpd0 = df.loc[fpd0_mask, 'FRODI26'].mean()
    avg_grodi26_fpd1 = df.loc[fpd1_mask, 'GRODI26'].mean()
    avg_grodi26_fpd0 = df.loc[fpd0_mask, 'GRODI26'].mean()
    
    logger.info(f"Calculated averages:")
    logger.info(f"  avg_frodi26_fpd1 (IsFPD=1): {avg_frodi26_fpd1}")
    logger.info(f"  avg_frodi26_fpd0 (IsFPD=0): {avg_frodi26_fpd0}")
    logger.info(f"  avg_grodi26_fpd1 (IsFPD=1): {avg_grodi26_fpd1}")
    logger.info(f"  avg_grodi26_fpd0 (IsFPD=0): {avg_grodi26_fpd0}")
    
    # Check for NaN averages (indicates no data in group)
    if pd.isna(avg_frodi26_fpd1):
        logger.warning("avg_frodi26_fpd1 is NaN - no IsFPD=1 rows with FRODI26 values!")
    if pd.isna(avg_frodi26_fpd0):
        logger.warning("avg_frodi26_fpd0 is NaN - no IsFPD=0 rows with FRODI26 values!")
    if pd.isna(avg_grodi26_fpd1):
        logger.warning("avg_grodi26_fpd1 is NaN - no IsFPD=1 rows with GRODI26 values!")
    if pd.isna(avg_grodi26_fpd0):
        logger.warning("avg_grodi26_fpd0 is NaN - no IsFPD=0 rows with GRODI26 values!")
    
    return avg_frodi26_fpd1, avg_frodi26_fpd0, avg_grodi26_fpd1, avg_grodi26_fpd0

@debug_trace
def create_frodi26_wri(df, avg_frodi26_fpd1, avg_frodi26_fpd0):
    """Create FRODI26_wRI column with imputation."""
    logger.info("Creating FRODI26_wRI column...")
    
    # Create mask for missing FRODI26 values
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
    
    # Convert to Float64
    df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')
    
    log_column_stats(df, 'FRODI26_wRI')
    
    return df

@debug_trace
def create_grodi26_wri(df, avg_grodi26_fpd1, avg_grodi26_fpd0):
    """Create GRODI26_wRI column with imputation."""
    logger.info("Creating GRODI26_wRI column...")
    
    # Create mask for missing GRODI26 values
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
    
    # Convert to Float64
    df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')
    
    log_column_stats(df, 'GRODI26_wRI')
    
    return df

@debug_trace
def reorder_columns(df):
    """Reorder columns to place new columns after their source columns."""
    logger.info("Reordering columns...")
    
    original_cols = df.columns.tolist()
    logger.debug(f"Original column order: {original_cols}")
    
    cols = df.columns.tolist()
    
    # Remove new columns from their current positions
    new_cols = ['isFPD_wRI', 'FRODI26_wRI', 'GRODI26_wRI']
    for col in new_cols:
        if col in cols:
            cols.remove(col)
            logger.debug(f"Removed '{col}' from position")
        else:
            logger.warning(f"Column '{col}' not found for reordering!")
    
    # Insert isFPD_wRI after IsFPD
    try:
        isfpd_index = cols.index('IsFPD')
        cols.insert(isfpd_index + 1, 'isFPD_wRI')
        logger.debug(f"Inserted 'isFPD_wRI' after 'IsFPD' (index {isfpd_index + 1})")
    except ValueError as e:
        logger.error(f"Cannot find 'IsFPD' column: {e}")
        cols.append('isFPD_wRI')
    
    # Insert FRODI26_wRI after FRODI26
    try:
        frodi26_index = cols.index('FRODI26')
        cols.insert(frodi26_index + 1, 'FRODI26_wRI')
        logger.debug(f"Inserted 'FRODI26_wRI' after 'FRODI26' (index {frodi26_index + 1})")
    except ValueError as e:
        logger.error(f"Cannot find 'FRODI26' column: {e}")
        cols.append('FRODI26_wRI')
    
    # Insert GRODI26_wRI after GRODI26
    try:
        grodi26_index = cols.index('GRODI26')
        cols.insert(grodi26_index + 1, 'GRODI26_wRI')
        logger.debug(f"Inserted 'GRODI26_wRI' after 'GRODI26' (index {grodi26_index + 1})")
    except ValueError as e:
        logger.error(f"Cannot find 'GRODI26' column: {e}")
        cols.append('GRODI26_wRI')
    
    df = df[cols]
    
    logger.debug(f"Final column order: {df.columns.tolist()}")
    
    return df

@debug_trace
def write_output_table(df):
    """Write output table to KNIME and log final state."""
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
            logger.info(f"  ✓ Column '{col}' present (dtype: {df[col].dtype})")
        else:
            logger.error(f"  ✗ Column '{col}' MISSING from output!")
    
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
    """Main execution function with full debug logging."""
    logger.info("=" * 70)
    logger.info("REJECT INFERENCE SCRIPT - DEBUG VERSION")
    logger.info("=" * 70)
    logger.info(f"Execution started at: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info("=" * 70)
    
    # Step 0: Read input
    df = read_input_table()
    
    # Step 1-3: Create isFPD_wRI column
    logger.info("-" * 70)
    logger.info("PHASE 1: Creating isFPD_wRI column")
    logger.info("-" * 70)
    
    loan_id_present, loan_id_missing = create_loan_id_masks(df)
    df = initialize_isfpd_wri(df)
    df = assign_isfpd_from_isfpd(df, loan_id_present)
    df = assign_isfpd_from_fpd(df, loan_id_missing)
    df = assign_isfpd_from_probability(df, loan_id_missing)
    df = finalize_isfpd_wri(df)
    
    # Step 4: Create FRODI26_wRI and GRODI26_wRI columns
    logger.info("-" * 70)
    logger.info("PHASE 2: Creating FRODI26_wRI and GRODI26_wRI columns")
    logger.info("-" * 70)
    
    avg_frodi26_fpd1, avg_frodi26_fpd0, avg_grodi26_fpd1, avg_grodi26_fpd0 = calculate_group_averages(df)
    df = create_frodi26_wri(df, avg_frodi26_fpd1, avg_frodi26_fpd0)
    df = create_grodi26_wri(df, avg_grodi26_fpd1, avg_grodi26_fpd0)
    
    # Step 5: Reorder columns
    logger.info("-" * 70)
    logger.info("PHASE 3: Reordering columns")
    logger.info("-" * 70)
    
    df = reorder_columns(df)
    
    # Step 6: Write output
    logger.info("-" * 70)
    logger.info("PHASE 4: Writing output")
    logger.info("-" * 70)
    
    write_output_table(df)
    
    logger.info("=" * 70)
    logger.info(f"Execution completed at: {datetime.now().isoformat()}")
    logger.info("REJECT INFERENCE SCRIPT - COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

# Execute main function
if __name__ == "__main__":
    main()
else:
    # When run directly in KNIME (not as module)
    main()

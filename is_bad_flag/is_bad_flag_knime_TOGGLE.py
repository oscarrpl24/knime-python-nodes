# =============================================================================
# Is Bad Flag Node for KNIME - TOGGLE DEBUG VERSION
# =============================================================================
# Purpose: Creates a binary "isBad" column based on GRODI26_wRI values
# 
# Logic:
#   - If GRODI26_wRI < 1, then isBad = 1
#   - Otherwise, isBad = 0
#
# Input: Single table with GRODI26_wRI column
# Output: Same table with isBad column added as the first column
#
# TOGGLE DEBUG VERSION: Set DEBUG = True to enable extensive logging
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps

# =============================================================================
# DEBUG TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG = False
# =============================================================================

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure comprehensive debug logging."""
    # Create a custom formatter with detailed information
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(funcName)-25s | "
        "Line %(lineno)-4d | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    
    # Set level based on DEBUG toggle
    log_level = logging.DEBUG if DEBUG else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# DEBUG DECORATOR
# =============================================================================

def debug_function(func):
    """Decorator to add entry/exit logging to functions (only when DEBUG is True)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If DEBUG is off, just execute the function without logging
        if not DEBUG:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        
        # Log function entry
        logger.debug(f"{'='*60}")
        logger.debug(f"ENTERING: {func_name}")
        logger.debug(f"{'='*60}")
        
        # Log arguments (excluding self for methods)
        if args:
            for i, arg in enumerate(args):
                arg_repr = _safe_repr(arg)
                logger.debug(f"  Positional arg[{i}]: {arg_repr}")
        
        if kwargs:
            for key, value in kwargs.items():
                value_repr = _safe_repr(value)
                logger.debug(f"  Keyword arg '{key}': {value_repr}")
        
        start_time = datetime.now()
        logger.debug(f"  Start time: {start_time}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"  End time: {end_time}")
            logger.debug(f"  Duration: {duration:.6f} seconds")
            logger.debug(f"  Return value: {_safe_repr(result)}")
            logger.debug(f"EXITING (SUCCESS): {func_name}")
            logger.debug(f"{'='*60}")
            
            return result
            
        except Exception as e:
            # Log exception details
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"  EXCEPTION in {func_name}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {str(e)}")
            logger.error(f"  Duration before error: {duration:.6f} seconds")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            logger.error(f"EXITING (FAILURE): {func_name}")
            logger.debug(f"{'='*60}")
            
            raise
    
    return wrapper


def _safe_repr(obj, max_length=500):
    """Safely create a string representation of an object for logging."""
    try:
        if isinstance(obj, pd.DataFrame):
            return (
                f"DataFrame(shape={obj.shape}, "
                f"columns={list(obj.columns)[:10]}{'...' if len(obj.columns) > 10 else ''}, "
                f"dtypes={dict(list(obj.dtypes.items())[:5])}{'...' if len(obj.dtypes) > 5 else ''})"
            )
        elif isinstance(obj, pd.Series):
            return (
                f"Series(name='{obj.name}', length={len(obj)}, "
                f"dtype={obj.dtype}, "
                f"head={list(obj.head(3).values)})"
            )
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 10:
                return f"{type(obj).__name__}(length={len(obj)}, first_10={obj[:10]}...)"
            return repr(obj)
        elif isinstance(obj, dict):
            if len(obj) > 5:
                sample = dict(list(obj.items())[:5])
                return f"dict(length={len(obj)}, sample={sample}...)"
            return repr(obj)
        else:
            result = repr(obj)
            if len(result) > max_length:
                return result[:max_length] + "..."
            return result
    except Exception as e:
        return f"<repr failed: {type(e).__name__}: {e}>"


# =============================================================================
# CORE FUNCTIONS WITH DEBUG LOGGING
# =============================================================================

@debug_function
def read_input_table():
    """Read the input table from KNIME and convert to pandas DataFrame."""
    if DEBUG:
        logger.info("Reading input table from KNIME...")
        logger.debug(f"Number of input tables available: {len(knio.input_tables)}")
    
    if len(knio.input_tables) == 0:
        if DEBUG:
            logger.error("No input tables found!")
        raise ValueError("No input tables available. Ensure the node has an input connection.")
    
    # Read the input table
    if DEBUG:
        logger.debug("Accessing input_tables[0]...")
    knime_table = knio.input_tables[0]
    if DEBUG:
        logger.debug(f"KNIME table object type: {type(knime_table)}")
    
    # Convert to pandas
    if DEBUG:
        logger.debug("Converting KNIME table to pandas DataFrame...")
    df = knime_table.to_pandas()
    
    # Log DataFrame details
    if DEBUG:
        logger.info(f"DataFrame loaded successfully")
        logger.debug(f"  Shape: {df.shape} (rows={df.shape[0]}, columns={df.shape[1]})")
        logger.debug(f"  Columns: {list(df.columns)}")
        logger.debug(f"  Data types:\n{df.dtypes.to_string()}")
        logger.debug(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        logger.debug(f"  Null counts:\n{df.isnull().sum().to_string()}")
        logger.debug(f"First 5 rows:\n{df.head().to_string()}")
    
    return df


@debug_function
def validate_required_column(df, column_name="GRODI26_wRI"):
    """Validate that the required column exists in the DataFrame."""
    if DEBUG:
        logger.info(f"Validating required column: '{column_name}'")
        logger.debug(f"Available columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            logger.debug(f"  [{i}] '{col}' - dtype: {df[col].dtype}")
    
    # Check if column exists
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
    
    # Log column statistics
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
def create_is_bad_column(df, source_column="GRODI26_wRI", target_column="isBad"):
    """Create the binary isBad column based on the source column values."""
    if DEBUG:
        logger.info(f"Creating binary column '{target_column}' from '{source_column}'")
        logger.debug(f"Input DataFrame shape: {df.shape}")
        logger.debug(f"Source column '{source_column}' dtype: {df[source_column].dtype}")
    
    # Get source column values
    source_values = df[source_column]
    if DEBUG:
        logger.debug(f"Source column sample values: {list(source_values.head(10).values)}")
    
    # Perform comparison
    if DEBUG:
        logger.debug(f"Performing comparison: {source_column} < 1")
    comparison_result = source_values < 1
    if DEBUG:
        logger.debug(f"Comparison result dtype: {comparison_result.dtype}")
        logger.debug(f"Comparison result sample: {list(comparison_result.head(10).values)}")
        logger.debug(f"True count (will be isBad=1): {comparison_result.sum()}")
        logger.debug(f"False count (will be isBad=0): {(~comparison_result).sum()}")
    
    # Convert to Int32
    if DEBUG:
        logger.debug("Converting boolean to Int32 (nullable integer)...")
    is_bad_values = comparison_result.astype("Int32")
    if DEBUG:
        logger.debug(f"Converted column dtype: {is_bad_values.dtype}")
        logger.debug(f"Converted column sample values: {list(is_bad_values.head(10).values)}")
    
    # Check for null handling
    if DEBUG:
        null_count_source = source_values.isnull().sum()
        null_count_result = is_bad_values.isnull().sum()
        logger.debug(f"Null values in source: {null_count_source}")
        logger.debug(f"Null values in result: {null_count_result}")
    
    # Assign to DataFrame
    if DEBUG:
        logger.debug(f"Assigning '{target_column}' column to DataFrame...")
    df[target_column] = is_bad_values
    
    # Verify assignment
    if DEBUG:
        logger.debug(f"Output DataFrame shape: {df.shape}")
        logger.debug(f"'{target_column}' column now in DataFrame: {target_column in df.columns}")
        logger.debug(f"'{target_column}' dtype: {df[target_column].dtype}")
        value_counts = df[target_column].value_counts(dropna=False)
        logger.debug(f"Value distribution:\n{value_counts.to_string()}")
        logger.info(f"Binary column '{target_column}' created successfully")
    
    return df


@debug_function
def reorder_columns(df, first_column="isBad"):
    """Reorder DataFrame columns to put the specified column first."""
    if DEBUG:
        logger.info(f"Reordering columns to put '{first_column}' first")
        logger.debug(f"Current column order: {list(df.columns)}")
        logger.debug(f"Current number of columns: {len(df.columns)}")
    
    # Verify the column exists
    if first_column not in df.columns:
        if DEBUG:
            logger.error(f"Column '{first_column}' not found for reordering!")
        raise ValueError(f"Cannot reorder: column '{first_column}' not found")
    
    # Get current column list
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
    
    # Prepend to beginning
    if DEBUG:
        logger.debug(f"Prepending '{first_column}' to beginning...")
    cols = [first_column] + cols
    if DEBUG:
        logger.debug(f"New column order: {cols}")
    
    # Apply new column order
    if DEBUG:
        logger.debug("Applying new column order to DataFrame...")
    df = df[cols]
    
    # Verify reordering
    if DEBUG:
        logger.debug(f"Verified new column order: {list(df.columns)}")
        logger.debug(f"First column is '{first_column}': {df.columns[0] == first_column}")
        logger.info(f"Columns reordered successfully. First column: '{df.columns[0]}'")
    
    return df


@debug_function
def log_summary_statistics(df, target_column="isBad", source_column="GRODI26_wRI"):
    """Calculate and log summary statistics for the binary target."""
    if DEBUG:
        logger.info("Calculating summary statistics...")
    
    # Calculate counts
    total_rows = len(df)
    if DEBUG:
        logger.debug(f"Total rows in DataFrame: {total_rows}")
    
    bad_count = (df[target_column] == 1).sum()
    good_count = (df[target_column] == 0).sum()
    null_count = df[target_column].isnull().sum()
    
    if DEBUG:
        logger.debug(f"isBad = 1 (bad) count: {bad_count}")
        logger.debug(f"isBad = 0 (good) count: {good_count}")
        logger.debug(f"isBad = null count: {null_count}")
    
    # Calculate percentages
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
    
    # Verify counts add up
    if DEBUG:
        sum_of_counts = bad_count + good_count + null_count
        logger.debug(f"Sum of counts: {sum_of_counts} (should equal {total_rows})")
        if sum_of_counts != total_rows:
            logger.warning(f"Count mismatch! Sum={sum_of_counts}, Total={total_rows}")
    
    # Print summary (visible in KNIME console)
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
def write_output_table(df):
    """Write the DataFrame back to KNIME as an output table."""
    if DEBUG:
        logger.info("Writing output table to KNIME...")
        logger.debug(f"Output DataFrame shape: {df.shape}")
        logger.debug(f"Output columns: {list(df.columns)}")
        logger.debug(f"Output dtypes:\n{df.dtypes.to_string()}")
        logger.debug(f"Output memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        logger.debug(f"First 5 rows of output:\n{df.head().to_string()}")
        logger.debug("Checking for potential output issues...")
    
    # Check for infinite values in numeric columns
    if DEBUG:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            inf_count = (~df[col].isnull() & (df[col].abs() == float('inf'))).sum()
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
# MAIN EXECUTION
# =============================================================================

@debug_function
def main():
    """Main execution function for the Is Bad Flag node."""
    if DEBUG:
        logger.info("="*70)
        logger.info("IS BAD FLAG NODE - TOGGLE DEBUG VERSION - STARTING EXECUTION")
        logger.info("="*70)
    
    execution_start = datetime.now()
    if DEBUG:
        logger.debug(f"Execution started at: {execution_start}")
    
    try:
        # Step 1: Read input table
        if DEBUG:
            logger.info("[STEP 1/5] Reading input table...")
        df = read_input_table()
        
        # Step 2: Validate required column
        if DEBUG:
            logger.info("[STEP 2/5] Validating required column...")
        validate_required_column(df, "GRODI26_wRI")
        
        # Step 3: Create isBad column
        if DEBUG:
            logger.info("[STEP 3/5] Creating isBad column...")
        df = create_is_bad_column(df, "GRODI26_wRI", "isBad")
        
        # Step 4: Reorder columns
        if DEBUG:
            logger.info("[STEP 4/5] Reordering columns...")
        df = reorder_columns(df, "isBad")
        
        # Step 5: Log summary and write output
        if DEBUG:
            logger.info("[STEP 5/5] Writing output...")
        log_summary_statistics(df, "isBad", "GRODI26_wRI")
        write_output_table(df)
        
        # Log successful completion
        if DEBUG:
            execution_end = datetime.now()
            execution_duration = (execution_end - execution_start).total_seconds()
            
            logger.info("="*70)
            logger.info("IS BAD FLAG NODE - EXECUTION COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {execution_duration:.4f} seconds")
            logger.info("="*70)
        
    except Exception as e:
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
        
        raise


# Execute main function
if __name__ == "__main__":
    main()
else:
    # When run in KNIME, __name__ is not "__main__"
    main()

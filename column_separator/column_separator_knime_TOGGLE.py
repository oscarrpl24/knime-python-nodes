# =============================================================================
# Column Separator Node for KNIME - TOGGLE DEBUG VERSION
# =============================================================================
# Purpose: Separates columns into two outputs based on naming patterns
# 
# TOGGLE VERSION: Debug logging can be enabled/disabled via DEBUG_ENABLED flag
# 
# Output 1: Columns that do NOT match any criteria (main data)
# Output 2: Columns that match any of the following criteria:
#   - Starts with "b_"
#   - Contains "date" (case insensitive)
#   - Contains "code" (case insensitive)
#   - Starts with "WOE_"
#   - Ends with "_points"
#   - Contains "basescore" (case insensitive)
#   - Contains "unq_id" (case insensitive)
#   - Contains "nodeid" (case insensitive)
#   - Contains "avg_" (case insensitive)
# =============================================================================

import knime.scripting.io as knio

# =============================================================================
# DEBUG TOGGLE - SET TO True TO ENABLE DEBUG LOGGING, False TO DISABLE
# =============================================================================
DEBUG_ENABLED = False
# =============================================================================

# Conditional imports for debugging
if DEBUG_ENABLED:
    import logging
    import sys
    import time
    import traceback
    from datetime import datetime
    from functools import wraps

# =============================================================================
# LOGGING CONFIGURATION (only when DEBUG_ENABLED)
# =============================================================================
if DEBUG_ENABLED:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    logger = logging.getLogger("ColumnSeparator_DEBUG")
    logger.setLevel(logging.DEBUG)

    def log_separator(char="=", length=80):
        """Log a visual separator line."""
        logger.debug(char * length)

    def log_section(title):
        """Log a section header."""
        log_separator()
        logger.info(f"SECTION: {title}")
        log_separator("-", 80)
else:
    # Dummy functions when debugging is disabled
    def log_separator(char="=", length=80):
        pass

    def log_section(title):
        pass

# =============================================================================
# FUNCTION DECORATOR FOR AUTOMATIC LOGGING (only when DEBUG_ENABLED)
# =============================================================================
if DEBUG_ENABLED:
    def debug_log(func):
        """
        Decorator that automatically logs function entry, exit, parameters,
        return values, and execution time.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            logger.debug(f">>> ENTERING: {func_name}")
            
            if args:
                for i, arg in enumerate(args):
                    arg_repr = repr(arg)
                    if len(arg_repr) > 100:
                        arg_repr = arg_repr[:100] + "... [truncated]"
                    logger.debug(f"    ARG[{i}] = {arg_repr}")
            
            if kwargs:
                for key, value in kwargs.items():
                    value_repr = repr(value)
                    if len(value_repr) > 100:
                        value_repr = value_repr[:100] + "... [truncated]"
                    logger.debug(f"    KWARG[{key}] = {value_repr}")
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time
                
                result_repr = repr(result)
                if len(result_repr) > 100:
                    result_repr = result_repr[:100] + "... [truncated]"
                logger.debug(f"    RETURN = {result_repr}")
                logger.debug(f"<<< EXITING: {func_name} (took {elapsed_time:.6f}s)")
                
                return result
                
            except Exception as e:
                elapsed_time = time.perf_counter() - start_time
                logger.error(f"!!! EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
                logger.error(f"    Traceback:\n{traceback.format_exc()}")
                logger.debug(f"<<< EXITING (with error): {func_name} (took {elapsed_time:.6f}s)")
                raise
        
        return wrapper
else:
    # No-op decorator when debugging is disabled
    def debug_log(func):
        return func

# =============================================================================
# SCRIPT START
# =============================================================================
if DEBUG_ENABLED:
    script_start_time = time.perf_counter()
    log_separator("=", 80)
    logger.info("COLUMN SEPARATOR DEBUG SCRIPT - STARTING EXECUTION")
    logger.info(f"Execution started at: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    log_separator("=", 80)

# -----------------------------------------------------------------------------
# Read Input Table
# -----------------------------------------------------------------------------
log_section("READING INPUT TABLE")

try:
    if DEBUG_ENABLED:
        logger.debug("Accessing knio.input_tables[0]...")
    input_table = knio.input_tables[0]
    if DEBUG_ENABLED:
        logger.debug(f"Input table object type: {type(input_table)}")
    
    if DEBUG_ENABLED:
        logger.debug("Converting input table to pandas DataFrame...")
    df = input_table.to_pandas()
    
    if DEBUG_ENABLED:
        logger.info(f"DataFrame loaded successfully")
        logger.info(f"  - Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        logger.debug("Column information:")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            logger.debug(f"  [{i:3d}] '{col}' - dtype: {dtype}, nulls: {null_count}")
        
        logger.debug("First 5 rows of data:")
        logger.debug(f"\n{df.head().to_string()}")
    
except Exception as e:
    if DEBUG_ENABLED:
        logger.critical(f"FATAL ERROR reading input table: {type(e).__name__}: {str(e)}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise

# -----------------------------------------------------------------------------
# Define Column Matching Criteria
# -----------------------------------------------------------------------------
log_section("COLUMN MATCHING CRITERIA FUNCTION")

@debug_log
def should_go_to_output2(col_name):
    """
    Check if a column matches any criteria for Output 2.
    Returns True if the column should go to Output 2.
    """
    if DEBUG_ENABLED:
        logger.debug(f"Evaluating column: '{col_name}'")
    
    col_lower = col_name.lower()
    if DEBUG_ENABLED:
        logger.debug(f"  Lowercase version: '{col_lower}'")
    
    # Check: Starts with "b_" (case sensitive)
    check_result = col_name.startswith("b_")
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'startswith(b_)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' starts with 'b_' -> Output 2")
        return True
    
    # Check: Contains "date" (case insensitive)
    check_result = "date" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(date)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'date' -> Output 2")
        return True
    
    # Check: Contains "code" (case insensitive)
    check_result = "code" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(code)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'code' -> Output 2")
        return True
    
    # Check: Starts with "WOE_" (case sensitive)
    check_result = col_name.startswith("WOE_")
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'startswith(WOE_)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' starts with 'WOE_' -> Output 2")
        return True
    
    # Check: Ends with "_points" (case insensitive)
    check_result = col_lower.endswith("_points")
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'endswith(_points)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' ends with '_points' -> Output 2")
        return True
    
    # Check: Contains "basescore" (case insensitive)
    check_result = "basescore" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(basescore)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'basescore' -> Output 2")
        return True
    
    # Check: Contains "unq_id" (case insensitive)
    check_result = "unq_id" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(unq_id)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'unq_id' -> Output 2")
        return True
    
    # Check: Contains "nodeid" (case insensitive)
    check_result = "nodeid" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(nodeid)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'nodeid' -> Output 2")
        return True
    
    # Check: Contains "avg_" (case insensitive)
    check_result = "avg_" in col_lower
    if DEBUG_ENABLED:
        logger.debug(f"  Check 'contains(avg_)': {check_result}")
    if check_result:
        if DEBUG_ENABLED:
            logger.info(f"  MATCH: Column '{col_name}' contains 'avg_' -> Output 2")
        return True
    
    if DEBUG_ENABLED:
        logger.debug(f"  NO MATCH: Column '{col_name}' -> Output 1")
    return False

# -----------------------------------------------------------------------------
# Separate Columns
# -----------------------------------------------------------------------------
log_section("SEPARATING COLUMNS")

output1_cols = []
output2_cols = []

if DEBUG_ENABLED:
    logger.info(f"Processing {len(df.columns)} columns...")
    logger.debug("Beginning column classification loop...")

for idx, col in enumerate(df.columns):
    if DEBUG_ENABLED:
        logger.debug(f"Processing column {idx + 1}/{len(df.columns)}: '{col}'")
    
    if should_go_to_output2(col):
        output2_cols.append(col)
        if DEBUG_ENABLED:
            logger.debug(f"  -> Added to output2_cols (current count: {len(output2_cols)})")
    else:
        output1_cols.append(col)
        if DEBUG_ENABLED:
            logger.debug(f"  -> Added to output1_cols (current count: {len(output1_cols)})")

if DEBUG_ENABLED:
    log_separator("-", 80)
    logger.info(f"Column classification complete:")
    logger.info(f"  - Output 1 columns (main data): {len(output1_cols)}")
    logger.info(f"  - Output 2 columns (filtered): {len(output2_cols)}")
    
    logger.debug("Output 1 columns:")
    for i, col in enumerate(output1_cols):
        logger.debug(f"  [{i:3d}] {col}")
    
    logger.debug("Output 2 columns:")
    for i, col in enumerate(output2_cols):
        logger.debug(f"  [{i:3d}] {col}")

# -----------------------------------------------------------------------------
# Create Output DataFrames
# -----------------------------------------------------------------------------
log_section("CREATING OUTPUT DATAFRAMES")

try:
    if DEBUG_ENABLED:
        logger.debug("Creating df_output1...")
    if output1_cols:
        df_output1 = df[output1_cols]
        if DEBUG_ENABLED:
            logger.info(f"df_output1 created with {len(output1_cols)} columns and {len(df_output1)} rows")
    else:
        df_output1 = df.iloc[:, 0:0]
        if DEBUG_ENABLED:
            logger.warning("df_output1 is EMPTY (no columns matched Output 1 criteria)")
    
    if DEBUG_ENABLED:
        logger.debug(f"df_output1 shape: {df_output1.shape}")
        logger.debug(f"df_output1 columns: {list(df_output1.columns)}")
        logger.debug(f"df_output1 memory usage: {df_output1.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
except Exception as e:
    if DEBUG_ENABLED:
        logger.error(f"Error creating df_output1: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    raise

try:
    if DEBUG_ENABLED:
        logger.debug("Creating df_output2...")
    if output2_cols:
        df_output2 = df[output2_cols]
        if DEBUG_ENABLED:
            logger.info(f"df_output2 created with {len(output2_cols)} columns and {len(df_output2)} rows")
    else:
        df_output2 = df.iloc[:, 0:0]
        if DEBUG_ENABLED:
            logger.warning("df_output2 is EMPTY (no columns matched Output 2 criteria)")
    
    if DEBUG_ENABLED:
        logger.debug(f"df_output2 shape: {df_output2.shape}")
        logger.debug(f"df_output2 columns: {list(df_output2.columns)}")
        logger.debug(f"df_output2 memory usage: {df_output2.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
except Exception as e:
    if DEBUG_ENABLED:
        logger.error(f"Error creating df_output2: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    raise

# -----------------------------------------------------------------------------
# Log Summary (Standard Output)
# -----------------------------------------------------------------------------
log_section("SUMMARY OUTPUT")

print(f"Input columns: {len(df.columns)}")
print(f"Output 1 columns (main data): {len(output1_cols)}")
print(f"Output 2 columns (filtered): {len(output2_cols)}")

if output2_cols:
    print(f"\nColumns sent to Output 2:")
    for col in output2_cols:
        print(f"  - {col}")

# -----------------------------------------------------------------------------
# Write Outputs
# -----------------------------------------------------------------------------
log_section("WRITING OUTPUTS TO KNIME")

try:
    if DEBUG_ENABLED:
        logger.debug("Writing df_output1 to knio.output_tables[0]...")
        logger.debug(f"  DataFrame shape: {df_output1.shape}")
        logger.debug(f"  DataFrame dtypes:\n{df_output1.dtypes.to_string()}")
    
    knio.output_tables[0] = knio.Table.from_pandas(df_output1)
    if DEBUG_ENABLED:
        logger.info("Output 1 written successfully to knio.output_tables[0]")
    
except Exception as e:
    if DEBUG_ENABLED:
        logger.critical(f"FATAL ERROR writing Output 1: {type(e).__name__}: {str(e)}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise

try:
    if DEBUG_ENABLED:
        logger.debug("Writing df_output2 to knio.output_tables[1]...")
        logger.debug(f"  DataFrame shape: {df_output2.shape}")
        logger.debug(f"  DataFrame dtypes:\n{df_output2.dtypes.to_string()}")
    
    knio.output_tables[1] = knio.Table.from_pandas(df_output2)
    if DEBUG_ENABLED:
        logger.info("Output 2 written successfully to knio.output_tables[1]")
    
except Exception as e:
    if DEBUG_ENABLED:
        logger.critical(f"FATAL ERROR writing Output 2: {type(e).__name__}: {str(e)}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise

# =============================================================================
# SCRIPT COMPLETE
# =============================================================================
if DEBUG_ENABLED:
    script_elapsed_time = time.perf_counter() - script_start_time
    
    log_separator("=", 80)
    logger.info("COLUMN SEPARATOR DEBUG SCRIPT - EXECUTION COMPLETE")
    logger.info(f"Total execution time: {script_elapsed_time:.4f} seconds")
    logger.info(f"Execution ended at: {datetime.now().isoformat()}")
    log_separator("=", 80)
    
    logger.info("FINAL STATISTICS:")
    logger.info(f"  Input:  {len(df.columns)} columns, {len(df)} rows")
    logger.info(f"  Output 1: {len(output1_cols)} columns (main data)")
    logger.info(f"  Output 2: {len(output2_cols)} columns (filtered)")
    logger.info(f"  Column separation ratio: {len(output1_cols)}/{len(output2_cols)} = {len(output1_cols)/max(len(output2_cols), 1):.2f}")
    log_separator("=", 80)

# ==============================================================================
# KNIME Python Script: Clean b_Score Column (TOGGLE DEBUG VERSION)
# ==============================================================================
# Purpose: Remove single quotes from the b_Score column
# Input: Single table with b_Score column
# Output: Same table with cleaned b_Score column
#
# TOGGLE VERSION: Debug logging can be enabled/disabled via DEBUG_MODE boolean
# ==============================================================================

# ==============================================================================
# DEBUG TOGGLE - Set to True for verbose logging, False for silent operation
# ==============================================================================
DEBUG_MODE = True
# ==============================================================================

import logging
import sys
from datetime import datetime

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logger(name: str = "clean_b_score_TOGGLE", level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return a logger with console output.
    
    Args:
        name: Logger name identifier
        level: Logging level (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    if DEBUG_MODE:
        logger.setLevel(level)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter with timestamp, level, and message
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)-8s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    else:
        # Disable all logging when DEBUG_MODE is False
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
    
    return logger


# Initialize logger
logger = setup_logger()

# ==============================================================================
# SCRIPT START
# ==============================================================================

logger.info("=" * 70)
logger.info("CLEAN B_SCORE SCRIPT - TOGGLE DEBUG VERSION")
logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
logger.info("=" * 70)
logger.info(f"Script started at: {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")

# ==============================================================================
# IMPORT SECTION
# ==============================================================================

logger.debug("Importing KNIME scripting module...")

try:
    import knime.scripting.io as knio
    logger.info("Successfully imported knime.scripting.io")
except ImportError as e:
    logger.critical(f"Failed to import KNIME scripting module: {e}")
    raise

try:
    import pandas as pd
    logger.debug(f"Pandas version: {pd.__version__}")
except ImportError as e:
    logger.warning(f"Could not import pandas directly (using KNIME's bundled version): {e}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def log_dataframe_info(df, label: str = "DataFrame") -> None:
    """
    Log detailed information about a DataFrame.
    
    Args:
        df: pandas DataFrame to inspect
        label: Descriptive label for the DataFrame
    """
    if not DEBUG_MODE:
        return
    logger.debug(f"--- {label} Info ---")
    logger.debug(f"  Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
    logger.debug(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    logger.debug(f"  Columns: {list(df.columns)}")
    logger.debug(f"  Data types:\n{df.dtypes.to_string()}")
    logger.debug(f"  Null counts:\n{df.isnull().sum().to_string()}")


def log_column_info(series, column_name: str) -> None:
    """
    Log detailed information about a specific column.
    
    Args:
        series: pandas Series to inspect
        column_name: Name of the column for logging
    """
    if not DEBUG_MODE:
        return
    logger.debug(f"--- Column '{column_name}' Info ---")
    logger.debug(f"  Data type: {series.dtype}")
    logger.debug(f"  Length: {len(series)}")
    logger.debug(f"  Null count: {series.isnull().sum()}")
    logger.debug(f"  Unique values: {series.nunique()}")
    
    # Log sample values (first 5 non-null)
    sample_values = series.dropna().head(5).tolist()
    logger.debug(f"  Sample values (first 5): {sample_values}")
    
    # Check for quotes in string representation
    if series.dtype == 'object' or str(series.dtype) == 'string':
        values_with_quotes = series.astype(str).str.contains("'", na=False).sum()
        logger.debug(f"  Values containing single quotes: {values_with_quotes}")


def log_transformation_result(before_series, after_series, column_name: str) -> None:
    """
    Log the results of a transformation by comparing before and after.
    
    Args:
        before_series: Series before transformation
        after_series: Series after transformation
        column_name: Name of the column for logging
    """
    if not DEBUG_MODE:
        return
    logger.debug(f"--- Transformation Results for '{column_name}' ---")
    
    # Compare data types
    logger.debug(f"  Type before: {before_series.dtype} -> Type after: {after_series.dtype}")
    
    # Count changed values
    before_str = before_series.astype(str)
    after_str = after_series.astype(str)
    changed_count = (before_str != after_str).sum()
    logger.debug(f"  Values changed: {changed_count} out of {len(before_series)}")
    
    # Show examples of changed values
    if changed_count > 0:
        changed_mask = before_str != after_str
        examples = list(zip(
            before_str[changed_mask].head(5).tolist(),
            after_str[changed_mask].head(5).tolist()
        ))
        logger.debug(f"  Example changes (before -> after):")
        for before, after in examples:
            logger.debug(f"    '{before}' -> '{after}'")


# ==============================================================================
# DATA INPUT SECTION
# ==============================================================================

logger.info("-" * 70)
logger.info("STEP 1: Reading input table from KNIME")
logger.info("-" * 70)

try:
    logger.debug("Accessing knio.input_tables[0]...")
    input_table = knio.input_tables[0]
    logger.debug(f"Input table object type: {type(input_table)}")
    
    logger.debug("Converting to pandas DataFrame...")
    df = input_table.to_pandas()
    logger.info(f"Successfully read input table with shape: {df.shape}")
    
    # Log detailed DataFrame info
    log_dataframe_info(df, "Input DataFrame")
    
except IndexError as e:
    logger.critical(f"No input table found at port 0: {e}")
    logger.critical("Ensure a table is connected to the Python Script node input port")
    raise
except Exception as e:
    logger.critical(f"Failed to read input table: {type(e).__name__}: {e}")
    raise

# ==============================================================================
# VALIDATION SECTION
# ==============================================================================

logger.info("-" * 70)
logger.info("STEP 2: Validating input data")
logger.info("-" * 70)

# Check column existence
TARGET_COLUMN = 'b_Score'
logger.debug(f"Checking for required column: '{TARGET_COLUMN}'")
logger.debug(f"Available columns: {list(df.columns)}")

if TARGET_COLUMN not in df.columns:
    logger.error(f"Required column '{TARGET_COLUMN}' not found!")
    logger.error(f"Available columns are: {list(df.columns)}")
    
    # Check for similar column names (case-insensitive)
    similar_cols = [col for col in df.columns if col.lower() == TARGET_COLUMN.lower()]
    if similar_cols:
        logger.error(f"Did you mean one of these? {similar_cols} (column names are CASE-SENSITIVE)")
    
    raise ValueError(f"Column '{TARGET_COLUMN}' not found in input table. Available columns: {list(df.columns)}")

logger.info(f"Validation passed: Column '{TARGET_COLUMN}' exists")

# Log detailed info about the target column before cleaning
log_column_info(df[TARGET_COLUMN], TARGET_COLUMN)

# ==============================================================================
# DATA CLEANING SECTION
# ==============================================================================

logger.info("-" * 70)
logger.info("STEP 3: Cleaning b_Score column (removing single quotes)")
logger.info("-" * 70)

# Store original for comparison
logger.debug("Creating copy of original column for comparison...")
original_column = df[TARGET_COLUMN].copy()
logger.debug(f"Original column dtype: {original_column.dtype}")

# Step 3a: Convert to string
logger.debug("Converting column to string type...")
try:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
    logger.debug(f"After astype(str), dtype: {df[TARGET_COLUMN].dtype}")
except Exception as e:
    logger.error(f"Failed to convert column to string: {type(e).__name__}: {e}")
    raise

# Count quotes before removal
quotes_before = df[TARGET_COLUMN].str.contains("'", na=False).sum()
logger.debug(f"Values containing single quotes before removal: {quotes_before}")

# Step 3b: Remove single quotes
logger.debug("Removing single quotes using str.replace()...")
try:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace("'", "", regex=False)
    logger.debug("Quote removal completed successfully")
except Exception as e:
    logger.error(f"Failed to remove quotes: {type(e).__name__}: {e}")
    raise

# Count quotes after removal
quotes_after = df[TARGET_COLUMN].str.contains("'", na=False).sum()
logger.debug(f"Values containing single quotes after removal: {quotes_after}")

# Log transformation results
log_transformation_result(original_column, df[TARGET_COLUMN], TARGET_COLUMN)

logger.info(f"Cleaning complete: Removed quotes from {quotes_before - quotes_after} values")

# ==============================================================================
# OUTPUT VALIDATION
# ==============================================================================

logger.info("-" * 70)
logger.info("STEP 4: Validating output data")
logger.info("-" * 70)

# Log final DataFrame info
log_dataframe_info(df, "Output DataFrame")
log_column_info(df[TARGET_COLUMN], f"{TARGET_COLUMN} (after cleaning)")

# Check for any remaining issues
logger.debug("Checking for potential issues in output...")

# Check for 'nan' strings (from NaN conversion)
nan_strings = (df[TARGET_COLUMN] == 'nan').sum()
if nan_strings > 0:
    logger.warning(f"Found {nan_strings} values that are the string 'nan' (were originally NaN/None)")

# Check for empty strings
empty_strings = (df[TARGET_COLUMN] == '').sum()
if empty_strings > 0:
    logger.warning(f"Found {empty_strings} empty string values")

# Check for remaining quotes
remaining_quotes = df[TARGET_COLUMN].str.contains("'", na=False).sum()
if remaining_quotes > 0:
    logger.error(f"ERROR: {remaining_quotes} values still contain single quotes!")
else:
    logger.info("Verification passed: No single quotes remaining in b_Score column")

# ==============================================================================
# DATA OUTPUT SECTION
# ==============================================================================

logger.info("-" * 70)
logger.info("STEP 5: Writing output to KNIME")
logger.info("-" * 70)

try:
    logger.debug("Converting DataFrame to KNIME table format...")
    output_table = knio.Table.from_pandas(df)
    logger.debug(f"Output table object type: {type(output_table)}")
    
    logger.debug("Assigning to output port 0...")
    knio.output_tables[0] = output_table
    logger.info("Successfully wrote output table to port 0")
    
except Exception as e:
    logger.critical(f"Failed to write output table: {type(e).__name__}: {e}")
    raise

# ==============================================================================
# SCRIPT COMPLETE
# ==============================================================================

logger.info("=" * 70)
logger.info("SCRIPT COMPLETED SUCCESSFULLY")
logger.info("=" * 70)
logger.info(f"Script ended at: {datetime.now().isoformat()}")
logger.info(f"Rows processed: {df.shape[0]}")
logger.info(f"Columns in output: {df.shape[1]}")
logger.info("=" * 70)

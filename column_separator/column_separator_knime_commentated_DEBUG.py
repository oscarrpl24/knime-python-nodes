# =============================================================================
# COLUMN SEPARATOR NODE FOR KNIME - FULLY COMMENTATED DEBUG VERSION
# =============================================================================
#
# FILE NAME: column_separator_knime_commentated_DEBUG.py
#
# DEBUG VERSION: This script includes extensive debug logging on every function,
# operation, and decision point. All logging is output to stdout for KNIME console
# visibility and includes timestamps, function names, and detailed state tracking.
#
# WHAT THIS SCRIPT DOES:
# This script is designed to run inside a KNIME Python Script node. Its job is
# to take a single input table and split its columns into two separate output
# tables based on specific naming patterns. Think of it as a column filter that
# routes columns to different destinations based on what their names look like.
#
# WHY THIS IS USEFUL:
# In credit risk modeling workflows, you often have datasets with many different
# types of columns mixed together:
#   - Raw feature columns (like "age", "income", "loan_amount")
#   - Derived WOE (Weight of Evidence) columns (like "WOE_age", "WOE_income")
#   - Scorecard point columns (like "age_points", "income_points")
#   - Metadata columns (like "application_date", "customer_code", "node_id")
#
# This script helps you separate the "main" analytical columns from the
# metadata/derived columns so you can process them differently downstream.
#
# OUTPUTS:
#   Output 1 (Port 0): Columns that DON'T match any filtering criteria
#                      (typically your main analytical data)
#   Output 2 (Port 1): Columns that DO match filtering criteria
#                      (metadata, WOE columns, score columns, etc.)
#
# FILTERING CRITERIA (columns sent to Output 2):
#   - Starts with "b_"           → Binary indicator columns
#   - Contains "date"            → Date-related columns
#   - Contains "code"            → Code/ID columns
#   - Starts with "WOE_"         → Weight of Evidence transformed columns
#   - Ends with "_points"        → Scorecard point columns
#   - Contains "basescore"       → Scorecard base score columns
#   - Contains "unq_id"          → Unique identifier columns
#   - Contains "nodeid"          → KNIME node tracking columns
#   - Contains "avg_"            → Averaged/aggregated columns
#
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

# This line imports the KNIME scripting input/output module and gives it the
# short alias "knio". This module is the bridge between Python and KNIME.
# It provides access to:
#   - knio.input_tables[]  → Read data from KNIME input ports
#   - knio.output_tables[] → Write data to KNIME output ports
#   - knio.flow_variables  → Read/write KNIME flow variables
# The "knime.scripting.io" module is automatically available in KNIME Python
# Script nodes - you don't need to install it separately.
import knime.scripting.io as knio

# DEBUG IMPORTS:
# These additional imports support comprehensive debug logging
import logging        # Python's built-in logging module for structured log output
import sys           # System module for accessing stdout and version info
import time          # Time module for measuring execution duration
import traceback     # Traceback module for detailed error stack traces
from datetime import datetime  # For human-readable timestamps
from functools import wraps    # For preserving function metadata in decorators


# =============================================================================
# SECTION 2: DEBUG LOGGING CONFIGURATION
# =============================================================================
# This section sets up the logging infrastructure that will capture detailed
# debug information throughout the script execution.
# =============================================================================

# CONFIGURE THE LOGGING SYSTEM:
# logging.basicConfig() sets up the root logger with our desired settings.
#
# Parameters explained:
#   - level=logging.DEBUG: Capture all log messages (DEBUG is the lowest level)
#   - format: Template for how each log message will appear
#     - %(asctime)s: Timestamp when the log was created
#     - %(levelname)-8s: Log level (DEBUG, INFO, etc.), left-aligned in 8 chars
#     - %(funcName)-25s: Function name, left-aligned in 25 chars
#     - %(message)s: The actual log message
#   - datefmt: Format for the timestamp
#   - stream=sys.stdout: Send logs to standard output (visible in KNIME console)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# CREATE A NAMED LOGGER:
# By using getLogger() with a specific name, we can identify our logs
# in the output and potentially configure this logger independently.
logger = logging.getLogger("ColumnSeparator_DEBUG")
logger.setLevel(logging.DEBUG)


# HELPER FUNCTION: Log a visual separator line
# These separator lines make it easier to read the log output by clearly
# delineating different sections of the script execution.
def log_separator(char="=", length=80):
    """
    Log a visual separator line.
    
    DEBUG INFO:
    This function is called to create visual breaks in the log output.
    It helps identify where different processing phases begin and end.
    
    Parameters:
        char (str): The character to use for the separator line
        length (int): How many characters wide the separator should be
    """
    logger.debug(char * length)


# HELPER FUNCTION: Log a section header
# This creates a consistent format for major section headers in the log.
def log_section(title):
    """
    Log a section header with visual separators.
    
    DEBUG INFO:
    This function is called at the start of each major processing phase.
    It makes the log output easier to scan and navigate.
    
    Parameters:
        title (str): The name of the section being logged
    """
    log_separator()
    logger.info(f"SECTION: {title}")
    log_separator("-", 80)


# =============================================================================
# SECTION 3: FUNCTION DECORATOR FOR AUTOMATIC DEBUG LOGGING
# =============================================================================
# This section defines a decorator that can be applied to any function to
# automatically log its entry, exit, parameters, return value, and timing.
# =============================================================================

def debug_log(func):
    """
    Decorator that automatically logs function entry, exit, parameters,
    return values, and execution time.
    
    DEBUG INFO:
    This decorator is applied to functions using the @debug_log syntax.
    When a decorated function is called, the decorator:
    1. Logs that the function was entered
    2. Logs all positional and keyword arguments
    3. Executes the original function
    4. Logs the return value
    5. Logs the execution time
    6. If an exception occurs, logs the full traceback
    
    HOW DECORATORS WORK:
    A decorator is a function that takes another function as input and returns
    a new function that "wraps" the original. The @decorator syntax is just
    shorthand for: func = decorator(func)
    
    Parameters:
        func: The function being decorated
        
    Returns:
        wrapper: A new function that wraps the original with logging
    """
    # @wraps preserves the original function's metadata (name, docstring, etc.)
    # Without this, the decorated function would appear to be named "wrapper"
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function name for logging
        func_name = func.__name__
        
        # LOG FUNCTION ENTRY:
        # The ">>>" prefix visually indicates we're entering a function
        logger.debug(f">>> ENTERING: {func_name}")
        
        # LOG POSITIONAL ARGUMENTS:
        # *args captures all positional arguments as a tuple
        if args:
            for i, arg in enumerate(args):
                # repr() gives us a string representation of the argument
                # that can be used to recreate it (includes quotes for strings, etc.)
                arg_repr = repr(arg)
                # Truncate very long argument representations
                if len(arg_repr) > 100:
                    arg_repr = arg_repr[:100] + "... [truncated]"
                logger.debug(f"    ARG[{i}] = {arg_repr}")
        
        # LOG KEYWORD ARGUMENTS:
        # **kwargs captures all keyword arguments as a dictionary
        if kwargs:
            for key, value in kwargs.items():
                value_repr = repr(value)
                if len(value_repr) > 100:
                    value_repr = value_repr[:100] + "... [truncated]"
                logger.debug(f"    KWARG[{key}] = {value_repr}")
        
        # START TIMING:
        # time.perf_counter() provides high-resolution timing
        start_time = time.perf_counter()
        
        try:
            # EXECUTE THE ORIGINAL FUNCTION:
            # Pass through all arguments to the wrapped function
            result = func(*args, **kwargs)
            
            # CALCULATE EXECUTION TIME:
            elapsed_time = time.perf_counter() - start_time
            
            # LOG RETURN VALUE:
            result_repr = repr(result)
            if len(result_repr) > 100:
                result_repr = result_repr[:100] + "... [truncated]"
            logger.debug(f"    RETURN = {result_repr}")
            
            # LOG FUNCTION EXIT:
            # The "<<<" prefix visually indicates we're exiting a function
            logger.debug(f"<<< EXITING: {func_name} (took {elapsed_time:.6f}s)")
            
            return result
            
        except Exception as e:
            # EXCEPTION HANDLING:
            # If the function raises an exception, log full details before re-raising
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"!!! EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
            # traceback.format_exc() gives us the full stack trace as a string
            logger.error(f"    Traceback:\n{traceback.format_exc()}")
            logger.debug(f"<<< EXITING (with error): {func_name} (took {elapsed_time:.6f}s)")
            raise  # Re-raise the exception so it propagates normally
    
    return wrapper


# =============================================================================
# SECTION 4: SCRIPT EXECUTION START
# =============================================================================
# This section marks the beginning of the actual script execution and logs
# initial environment information.
# =============================================================================

# Record the script start time for total execution timing
script_start_time = time.perf_counter()

# Log the script header
log_separator("=", 80)
logger.info("COLUMN SEPARATOR DEBUG SCRIPT (COMMENTATED VERSION) - STARTING EXECUTION")
logger.info(f"Execution started at: {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")
log_separator("=", 80)


# =============================================================================
# SECTION 5: READ INPUT DATA
# =============================================================================
# This section retrieves the input data from KNIME and converts it to a format
# that Python can work with easily (a pandas DataFrame).
#
# DEBUG ENHANCEMENTS:
# - Logs each step of the data loading process
# - Reports DataFrame shape, memory usage, and column information
# - Logs sample data (first 5 rows) for verification
# - Captures and logs any errors with full traceback
# =============================================================================

log_section("READING INPUT TABLE")

try:
    # LOG ACCESS TO INPUT TABLE:
    # This step retrieves the KNIME table object from the first input port
    logger.debug("Accessing knio.input_tables[0]...")
    input_table = knio.input_tables[0]
    logger.debug(f"Input table object type: {type(input_table)}")
    
    # LOG CONVERSION TO PANDAS:
    # The .to_pandas() method converts KNIME's internal table format to a
    # pandas DataFrame, which is much easier to work with in Python.
    logger.debug("Converting input table to pandas DataFrame...")
    df = input_table.to_pandas()
    
    # LOG SUCCESS AND BASIC METRICS:
    logger.info(f"DataFrame loaded successfully")
    logger.info(f"  - Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
    # Memory usage helps identify if we're dealing with a large dataset
    logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # LOG DETAILED COLUMN INFORMATION:
    # This helps verify that the data was loaded correctly and shows the
    # data types that pandas inferred for each column
    logger.debug("Column information:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        # Count null values to identify potential data quality issues
        null_count = df[col].isna().sum()
        logger.debug(f"  [{i:3d}] '{col}' - dtype: {dtype}, nulls: {null_count}")
    
    # LOG SAMPLE DATA:
    # Seeing the first few rows helps verify the data looks correct
    logger.debug("First 5 rows of data:")
    logger.debug(f"\n{df.head().to_string()}")
    
except Exception as e:
    # CRITICAL ERROR HANDLING:
    # If we can't load the input data, the script cannot proceed
    logger.critical(f"FATAL ERROR reading input table: {type(e).__name__}: {str(e)}")
    logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise


# =============================================================================
# SECTION 6: DEFINE THE COLUMN CLASSIFICATION FUNCTION
# =============================================================================
# This section defines a function that decides whether a given column should
# be routed to Output 1 (main data) or Output 2 (filtered/metadata).
#
# DEBUG ENHANCEMENTS:
# - The @debug_log decorator automatically logs entry/exit and timing
# - Each pattern check is individually logged with its result
# - Match decisions are logged at INFO level for visibility
# =============================================================================

log_section("COLUMN MATCHING CRITERIA FUNCTION")

# Apply the debug logging decorator to this function
@debug_log
def should_go_to_output2(col_name):
    """
    DOCSTRING (documentation inside triple quotes):
    This function examines a column name and determines if it matches any of
    the criteria that would route it to Output 2.
    
    PARAMETERS:
        col_name (str): The name of a column to evaluate
        
    RETURNS:
        bool: True if this column should go to Output 2
              False if this column should go to Output 1
    
    The function uses pattern matching on the column name to make its decision.
    Some patterns are case-sensitive (like "WOE_") while others are
    case-insensitive (like "date").
    
    DEBUG INFO:
    This function logs each pattern check result so you can see exactly
    why a column was routed to a particular output. Look for MATCH log
    entries at INFO level to see which pattern triggered each routing decision.
    """
    
    # LOG THE COLUMN BEING EVALUATED:
    logger.debug(f"Evaluating column: '{col_name}'")
    
    # CREATE A LOWERCASE VERSION OF THE COLUMN NAME:
    # The .lower() method converts all uppercase letters to lowercase.
    # For example: "ApplicationDate" becomes "applicationdate"
    #              "WOE_Age" becomes "woe_age"
    #              "customer_code" stays "customer_code" (already lowercase)
    #
    # We store this in a separate variable (col_lower) because:
    #   1. We need to keep the original name (col_name) for case-sensitive checks
    #   2. Calling .lower() once and reusing is more efficient than calling it
    #      multiple times
    col_lower = col_name.lower()
    logger.debug(f"  Lowercase version: '{col_lower}'")
    
    # -------------------------------------------------------------------------
    # CHECK 1: Does the column name start with "b_"?
    # -------------------------------------------------------------------------
    # The .startswith() method returns True if the string begins with the
    # specified prefix, False otherwise.
    #
    # WHY THIS PATTERN:
    # In credit risk modeling, columns prefixed with "b_" often represent
    # binary (0/1) indicator variables or binned versions of continuous
    # variables. For example:
    #   - "b_is_employed" → Binary flag for employment status
    #   - "b_age_group"   → Binned age category
    #
    # NOTE: This check is CASE-SENSITIVE (uses col_name, not col_lower)
    # because the "b_" prefix is typically lowercase by convention.
    check_result = col_name.startswith("b_")
    logger.debug(f"  Check 'startswith(b_)': {check_result}")
    if check_result:
        # If the condition is True, immediately return True and exit the function.
        # No need to check the remaining conditions.
        logger.info(f"  MATCH: Column '{col_name}' starts with 'b_' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 2: Does the column name contain "date" anywhere?
    # -------------------------------------------------------------------------
    # The "in" operator checks if one string exists inside another string.
    # For example:
    #   "date" in "application_date"     → True
    #   "date" in "ApplicationDate"      → False (case mismatch)
    #   "date" in "applicationdate"      → True (after lowercasing)
    #   "date" in "created_at_timestamp" → False (no "date" substring)
    #
    # WHY THIS PATTERN:
    # Date columns (like "application_date", "DateOfBirth", "created_date")
    # are typically metadata rather than analytical features. They're often
    # used for filtering or joining but not for model building.
    #
    # NOTE: This check is CASE-INSENSITIVE (uses col_lower) because date
    # columns can be named in various styles (camelCase, snake_case, etc.)
    check_result = "date" in col_lower
    logger.debug(f"  Check 'contains(date)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'date' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 3: Does the column name contain "code" anywhere?
    # -------------------------------------------------------------------------
    # Similar to the date check, but looking for "code" substring.
    #
    # WHY THIS PATTERN:
    # Code columns (like "branch_code", "ProductCode", "occupation_code")
    # are typically categorical identifiers. While they might be useful for
    # some analyses, they're often considered metadata rather than features.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    check_result = "code" in col_lower
    logger.debug(f"  Check 'contains(code)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'code' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 4: Does the column name start with "WOE_"?
    # -------------------------------------------------------------------------
    # Checks if the column name begins with the prefix "WOE_".
    #
    # WHY THIS PATTERN:
    # WOE (Weight of Evidence) columns are transformed versions of original
    # features. In a credit risk workflow, you often want to keep the WOE
    # columns separate from the original raw features because:
    #   - They're used for logistic regression modeling
    #   - They shouldn't be confused with the original values
    #
    # TYPICAL NAMING:
    #   Original column: "age"
    #   WOE column: "WOE_age"
    #
    # NOTE: This check is CASE-SENSITIVE (uses col_name, not col_lower)
    # because "WOE_" is a standardized uppercase prefix in this workflow.
    check_result = col_name.startswith("WOE_")
    logger.debug(f"  Check 'startswith(WOE_)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' starts with 'WOE_' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 5: Does the column name end with "_points"?
    # -------------------------------------------------------------------------
    # The .endswith() method returns True if the string ends with the
    # specified suffix, False otherwise.
    #
    # WHY THIS PATTERN:
    # Scorecard point columns (like "age_points", "income_points") are
    # derived columns that represent the points assigned for each binned
    # value in a credit scorecard. These are outputs of the scorecard
    # transformation and should be kept separate from raw data.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower) because different
    # systems might use "Age_Points" or "age_points".
    check_result = col_lower.endswith("_points")
    logger.debug(f"  Check 'endswith(_points)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' ends with '_points' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 6: Does the column name contain "basescore" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "basescore" substring.
    #
    # WHY THIS PATTERN:
    # The base score is a constant offset in a credit scorecard calculation.
    # It's derived from the model intercept and is a derived value, not a
    # raw feature. Columns like "BaseScore" or "basescore_adjustment" would
    # match this pattern.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    check_result = "basescore" in col_lower
    logger.debug(f"  Check 'contains(basescore)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'basescore' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 7: Does the column name contain "unq_id" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "unq_id" substring.
    #
    # WHY THIS PATTERN:
    # "unq_id" typically stands for "unique identifier". These are key
    # columns used to uniquely identify each row (like customer ID,
    # application ID, etc.). They're essential for joining tables but
    # are not analytical features.
    #
    # EXAMPLES:
    #   - "customer_unq_id" → Unique customer identifier
    #   - "unq_id_application" → Application identifier
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    check_result = "unq_id" in col_lower
    logger.debug(f"  Check 'contains(unq_id)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'unq_id' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 8: Does the column name contain "nodeid" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "nodeid" substring.
    #
    # WHY THIS PATTERN:
    # "nodeid" columns are typically internal KNIME tracking columns that
    # record which node processed the data. These are metadata for workflow
    # debugging and not analytical features.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    check_result = "nodeid" in col_lower
    logger.debug(f"  Check 'contains(nodeid)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'nodeid' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 9: Does the column name contain "avg_" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "avg_" substring.
    #
    # WHY THIS PATTERN:
    # Columns prefixed or containing "avg_" are typically aggregated values
    # (averages). These are often summary statistics or derived features
    # that might need to be handled separately from raw features.
    #
    # EXAMPLES:
    #   - "avg_balance" → Average account balance
    #   - "monthly_avg_spend" → Monthly average spending
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    check_result = "avg_" in col_lower
    logger.debug(f"  Check 'contains(avg_)': {check_result}")
    if check_result:
        logger.info(f"  MATCH: Column '{col_name}' contains 'avg_' -> Output 2")
        return True
    
    # -------------------------------------------------------------------------
    # DEFAULT CASE: No patterns matched
    # -------------------------------------------------------------------------
    # If we've reached this point, the column name didn't match ANY of the
    # criteria above. This means it should go to Output 1 (main data).
    # We return False to indicate "No, this column should NOT go to Output 2".
    logger.debug(f"  NO MATCH: Column '{col_name}' -> Output 1")
    return False


# =============================================================================
# SECTION 7: SEPARATE COLUMNS INTO TWO GROUPS
# =============================================================================
# This section loops through all columns in the input DataFrame and decides
# which output each column should go to.
#
# DEBUG ENHANCEMENTS:
# - Logs progress through the column loop
# - Reports running counts of columns assigned to each output
# - Logs complete column lists after classification
# =============================================================================

log_section("SEPARATING COLUMNS")

# CREATE TWO EMPTY LISTS TO STORE COLUMN NAMES:
# In Python, [] creates an empty list. Lists are ordered collections that
# can grow as you add items to them.
#
# output1_cols will store names of columns destined for Output 1 (main data)
output1_cols = []

# output2_cols will store names of columns destined for Output 2 (filtered)
output2_cols = []

# LOG THE START OF COLUMN PROCESSING:
logger.info(f"Processing {len(df.columns)} columns...")
logger.debug("Beginning column classification loop...")

# LOOP THROUGH ALL COLUMNS IN THE DATAFRAME:
# df.columns gives us an Index object containing all column names in the
# DataFrame. The "for col in ..." syntax iterates through each column name
# one at a time, assigning the current name to the variable "col".
#
# We use enumerate() to also get the index (idx) for logging progress.
for idx, col in enumerate(df.columns):
    # LOG PROGRESS:
    logger.debug(f"Processing column {idx + 1}/{len(df.columns)}: '{col}'")
    
    # CALL OUR CLASSIFICATION FUNCTION:
    # For each column, we call the should_go_to_output2() function we defined
    # earlier, passing in the current column name. The function returns True
    # or False based on whether the column matches any filtering criteria.
    if should_go_to_output2(col):
        # IF THE FUNCTION RETURNED TRUE:
        # This column matches one of the filtering criteria, so it should
        # go to Output 2. We add (append) this column name to the output2_cols
        # list using the .append() method.
        output2_cols.append(col)
        logger.debug(f"  -> Added to output2_cols (current count: {len(output2_cols)})")
    else:
        # IF THE FUNCTION RETURNED FALSE:
        # This column doesn't match any criteria, so it stays in the main
        # data (Output 1). We add this column name to output1_cols.
        output1_cols.append(col)
        logger.debug(f"  -> Added to output1_cols (current count: {len(output1_cols)})")

# LOG CLASSIFICATION SUMMARY:
log_separator("-", 80)
logger.info(f"Column classification complete:")
logger.info(f"  - Output 1 columns (main data): {len(output1_cols)}")
logger.info(f"  - Output 2 columns (filtered): {len(output2_cols)}")

# LOG THE COMPLETE COLUMN LISTS:
logger.debug("Output 1 columns:")
for i, col in enumerate(output1_cols):
    logger.debug(f"  [{i:3d}] {col}")

logger.debug("Output 2 columns:")
for i, col in enumerate(output2_cols):
    logger.debug(f"  [{i:3d}] {col}")


# =============================================================================
# SECTION 8: CREATE OUTPUT DATAFRAMES
# =============================================================================
# Now we create the actual DataFrames for each output by selecting only the
# relevant columns from the original DataFrame.
#
# DEBUG ENHANCEMENTS:
# - Logs each DataFrame creation step
# - Reports shape, columns, and memory usage
# - Captures errors with full traceback
# =============================================================================

log_section("CREATING OUTPUT DATAFRAMES")

# CREATE OUTPUT 1 DATAFRAME (Main Data):
# This line has a conditional expression (Python's ternary operator):
#   result = value_if_true if condition else value_if_false
#
# CONDITION: output1_cols (evaluates to True if the list is not empty)
#
# IF TRUE (there are columns for output 1):
#   df[output1_cols] creates a new DataFrame containing only the columns
#   whose names are in the output1_cols list. This is called column selection
#   or column slicing in pandas.
#
# IF FALSE (output1_cols is empty, no columns for output 1):
#   df.iloc[:, 0:0] creates an empty DataFrame with:
#     - All the same rows as the original df (the ":" before the comma)
#     - Zero columns (the "0:0" after the comma selects columns 0 up to 0,
#       which is an empty range)
#   This preserves the DataFrame structure (index, etc.) while having no data.
#
# WHY THE EMPTY CHECK IS IMPORTANT:
# If we tried to do df[[]] (select with an empty list), pandas would give us
# an error or unexpected behavior. The iloc approach safely creates an empty
# DataFrame with the correct structure.
try:
    logger.debug("Creating df_output1...")
    if output1_cols:
        df_output1 = df[output1_cols]
        logger.info(f"df_output1 created with {len(output1_cols)} columns and {len(df_output1)} rows")
    else:
        df_output1 = df.iloc[:, 0:0]
        logger.warning("df_output1 is EMPTY (no columns matched Output 1 criteria)")
    
    logger.debug(f"df_output1 shape: {df_output1.shape}")
    logger.debug(f"df_output1 columns: {list(df_output1.columns)}")
    logger.debug(f"df_output1 memory usage: {df_output1.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
except Exception as e:
    logger.error(f"Error creating df_output1: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")
    raise

# CREATE OUTPUT 2 DATAFRAME (Filtered Columns):
# Same logic as above, but for the filtered columns going to Output 2.
try:
    logger.debug("Creating df_output2...")
    if output2_cols:
        df_output2 = df[output2_cols]
        logger.info(f"df_output2 created with {len(output2_cols)} columns and {len(df_output2)} rows")
    else:
        df_output2 = df.iloc[:, 0:0]
        logger.warning("df_output2 is EMPTY (no columns matched Output 2 criteria)")
    
    logger.debug(f"df_output2 shape: {df_output2.shape}")
    logger.debug(f"df_output2 columns: {list(df_output2.columns)}")
    logger.debug(f"df_output2 memory usage: {df_output2.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
except Exception as e:
    logger.error(f"Error creating df_output2: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")
    raise


# =============================================================================
# SECTION 9: LOGGING / SUMMARY OUTPUT
# =============================================================================
# This section prints a summary of what the script did. These print statements
# appear in KNIME's console output, which is useful for debugging and
# verifying that the column separation worked as expected.
# =============================================================================

log_section("SUMMARY OUTPUT")

# PRINT THE TOTAL NUMBER OF INPUT COLUMNS:
# The f"..." syntax creates a "formatted string" (f-string) in Python.
# Expressions inside curly braces {} are evaluated and inserted into the string.
#
# len(df.columns) returns the number of columns in the original DataFrame.
# For example, if df has 50 columns, this prints: "Input columns: 50"
print(f"Input columns: {len(df.columns)}")

# PRINT THE NUMBER OF COLUMNS GOING TO OUTPUT 1:
# len(output1_cols) counts how many items are in the output1_cols list.
print(f"Output 1 columns (main data): {len(output1_cols)}")

# PRINT THE NUMBER OF COLUMNS GOING TO OUTPUT 2:
# len(output2_cols) counts how many items are in the output2_cols list.
print(f"Output 2 columns (filtered): {len(output2_cols)}")

# OPTIONALLY PRINT THE NAMES OF FILTERED COLUMNS:
# This "if" statement checks if output2_cols is not empty (any non-empty
# list evaluates to True in Python).
if output2_cols:
    # Print a blank line and header for readability
    # The \n creates a newline character, adding vertical spacing
    print(f"\nColumns sent to Output 2:")
    
    # Loop through each column name in output2_cols and print it
    # The "  - " prefix creates a bulleted list appearance in the console
    for col in output2_cols:
        print(f"  - {col}")


# =============================================================================
# SECTION 10: WRITE OUTPUTS TO KNIME
# =============================================================================
# This final section sends our two DataFrames back to KNIME so they can be
# used by downstream nodes in the workflow.
#
# DEBUG ENHANCEMENTS:
# - Logs each write operation with DataFrame details
# - Logs data types being written
# - Captures errors with full traceback
# =============================================================================

log_section("WRITING OUTPUTS TO KNIME")

# WRITE OUTPUT 1 (Main Data) TO KNIME:
# DETAILED BREAKDOWN:
#
# knio.output_tables[0]
#   ├── knio           → The KNIME I/O module
#   ├── output_tables  → A list-like object for all output ports
#   └── [0]            → The FIRST output port (index 0)
#
# = knio.Table.from_pandas(df_output1)
#   ├── knio.Table     → The KNIME Table class
#   ├── .from_pandas() → A class method that converts a pandas DataFrame
#   │                    into a KNIME-compatible table format
#   └── (df_output1)   → The DataFrame we want to convert and send
#
# IMPORTANT: The KNIME node must be configured with the correct number of
# output ports. If this script writes to output_tables[0] and [1], the node
# needs at least 2 output ports configured.
#
# After this line, the main data (columns that didn't match any filter
# criteria) will be available at Output Port 1 in KNIME.
try:
    logger.debug("Writing df_output1 to knio.output_tables[0]...")
    logger.debug(f"  DataFrame shape: {df_output1.shape}")
    logger.debug(f"  DataFrame dtypes:\n{df_output1.dtypes.to_string()}")
    
    knio.output_tables[0] = knio.Table.from_pandas(df_output1)
    logger.info("Output 1 written successfully to knio.output_tables[0]")
    
except Exception as e:
    logger.critical(f"FATAL ERROR writing Output 1: {type(e).__name__}: {str(e)}")
    logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise

# WRITE OUTPUT 2 (Filtered Columns) TO KNIME:
# Same pattern as above, but writing to the second output port (index 1).
# After this line, the filtered columns (WOE columns, date columns, etc.)
# will be available at Output Port 2 in KNIME.
try:
    logger.debug("Writing df_output2 to knio.output_tables[1]...")
    logger.debug(f"  DataFrame shape: {df_output2.shape}")
    logger.debug(f"  DataFrame dtypes:\n{df_output2.dtypes.to_string()}")
    
    knio.output_tables[1] = knio.Table.from_pandas(df_output2)
    logger.info("Output 2 written successfully to knio.output_tables[1]")
    
except Exception as e:
    logger.critical(f"FATAL ERROR writing Output 2: {type(e).__name__}: {str(e)}")
    logger.critical(f"Traceback:\n{traceback.format_exc()}")
    raise


# =============================================================================
# SECTION 11: SCRIPT COMPLETE - FINAL STATISTICS
# =============================================================================
# Log the completion of the script and report final execution statistics.
# =============================================================================

# Calculate total execution time
script_elapsed_time = time.perf_counter() - script_start_time

log_separator("=", 80)
logger.info("COLUMN SEPARATOR DEBUG SCRIPT (COMMENTATED VERSION) - EXECUTION COMPLETE")
logger.info(f"Total execution time: {script_elapsed_time:.4f} seconds")
logger.info(f"Execution ended at: {datetime.now().isoformat()}")
log_separator("=", 80)

# FINAL STATISTICS:
# These summary statistics provide a quick overview of what the script did
logger.info("FINAL STATISTICS:")
logger.info(f"  Input:  {len(df.columns)} columns, {len(df)} rows")
logger.info(f"  Output 1: {len(output1_cols)} columns (main data)")
logger.info(f"  Output 2: {len(output2_cols)} columns (filtered)")
# Calculate ratio (avoiding division by zero)
ratio = len(output1_cols) / max(len(output2_cols), 1)
logger.info(f"  Column separation ratio: {len(output1_cols)}/{len(output2_cols)} = {ratio:.2f}")
log_separator("=", 80)


# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# EXECUTION FLOW SUMMARY:
# 1. Configure logging infrastructure for debug output
# 2. Define helper functions and decorators for automatic logging
# 3. Import the KNIME I/O module
# 4. Read the input table from KNIME's first input port (with logging)
# 5. Define a function to classify columns based on naming patterns (decorated)
# 6. Loop through all columns and categorize them into two groups (with logging)
# 7. Create two separate DataFrames from those groups (with logging)
# 8. Print a summary of the separation for standard output
# 9. Send both DataFrames to KNIME's output ports (with logging)
# 10. Report final execution statistics
#
# DEBUGGING WITH THIS SCRIPT:
# - Run this script instead of the standard version to get full debug output
# - Look for "MATCH" entries at INFO level to see which patterns triggered
# - Look for ">>>" and "<<<" entries to trace function calls
# - Look for "FATAL ERROR" or "EXCEPTION" entries to diagnose failures
# - Check execution times to identify performance bottlenecks
#
# =============================================================================

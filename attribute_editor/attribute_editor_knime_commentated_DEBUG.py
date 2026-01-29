# =============================================================================
# Attribute Editor for KNIME Python Script Node - DEBUG VERSION (Commentated)
# =============================================================================
# This is a Python script designed to run inside a KNIME 5.9 Python Script node.
# It provides functionality equivalent to the R-based Attribute Editor, which is
# used in credit risk modeling workflows to configure variable metadata.
#
# DEBUG VERSION: Contains extensive logging on every function for troubleshooting
#
# The script operates in two distinct modes:
# 1. Headless Mode - Runs automatically without user interaction when:
#    - The DependentVariable flow variable is provided AND
#    - The VarOverride flow variable is NOT set to 1
# 2. Interactive Mode (Shiny UI) - Launches a web-based user interface when:
#    - No DependentVariable is provided, OR
#    - VarOverride is set to 1 (forcing interactive mode)
#
# Flow Variables (inputs from KNIME workflow):
# - DependentVariable (string): The name of the target/dependent variable column
#   that will be predicted by the model (e.g., "is_bad" for credit default)
# - VarOverride (integer): If set to 1, forces the interactive UI to launch
#   even when DependentVariable is already specified
#
# Output Tables:
# 1. Variable metadata DataFrame containing:
#    - VariableName: The column name
#    - Include: Boolean indicating if variable should be used in modeling
#    - Role: Either 'dependent' (target) or 'independent' (feature)
#    - Usage: Variable type (continuous, discrete, nominal, ordinal, no binning)
#    - Various other metadata for downstream processing
# 2. Converted data DataFrame with:
#    - Type conversions applied based on Usage settings
#    - Excluded columns removed (where Include == False)
#
# Version History:
# Release Date: 2026-01-26
# Version: 1.3-DEBUG - Debug version with extensive logging (commentated)
# =============================================================================

# =============================================================================
# Import Statements
# =============================================================================

# Import the KNIME scripting I/O module - this is the interface between
# Python and KNIME, allowing us to read input tables and write output tables
import knime.scripting.io as knio

# Import pandas - the core library for data manipulation in Python
# Pandas provides DataFrame objects similar to R's data frames
import pandas as pd

# Import numpy - provides numerical computing functionality
# Used here for array operations and random number generation
import numpy as np

# Import the warnings module from Python's standard library
# This allows us to control warning message display
import warnings

# Import logging module for debug output
import logging

# Import traceback for detailed exception information
import traceback

# Import sys for stdout access
import sys

# Import datetime for timestamp generation
from datetime import datetime

# Import type hints from the typing module
# List is used for annotating list types in function signatures
# Optional indicates a parameter that can be None
# Any is a special type that accepts any value
from typing import List, Optional, Any

# Suppress all warning messages to keep console output clean
# This prevents pandas deprecation warnings and other non-critical warnings
# from cluttering the output during execution
warnings.filterwarnings('ignore')

# =============================================================================
# Configure Debug Logging
# =============================================================================
# This section sets up a comprehensive logging system for debugging purposes.
# The logger outputs detailed information about function calls, parameters,
# return values, and any errors that occur during execution.

# Create a custom logger with a descriptive name
# This allows us to identify log messages from this specific module
logger = logging.getLogger('AttributeEditorDebug')

# Set the logging level to DEBUG to capture all messages
# DEBUG is the lowest level, so all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL) are captured
logger.setLevel(logging.DEBUG)

# Create a console handler that writes to standard output
# This ensures log messages appear in the KNIME console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a detailed formatter for log messages
# Format: TIME | LEVEL | FUNCTION | MESSAGE
# This makes it easy to trace execution flow and identify issues
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
    datefmt='%H:%M:%S.%f'  # Include microseconds for precise timing
)
console_handler.setFormatter(formatter)

# Add the handler to the logger, but only if no handlers exist
# This prevents duplicate log messages if the module is reloaded
if not logger.handlers:
    logger.addHandler(console_handler)


def log_function_entry(func_name: str, **kwargs):
    """
    Log function entry with parameters.
    
    This helper function creates a visually distinct log block when entering
    a function, showing the function name and all parameter values.
    
    Parameters:
    - func_name: The name of the function being entered
    - **kwargs: Key-value pairs of parameter names and their values
    """
    # Create a visual separator for easy scanning
    logger.debug(f"{'='*60}")
    logger.debug(f"ENTERING: {func_name}")
    
    # Log each parameter with its value
    if kwargs:
        for key, value in kwargs.items():
            # Truncate long values to keep logs readable
            value_repr = repr(value)
            if len(value_repr) > 200:
                value_repr = value_repr[:200] + "..."
            logger.debug(f"  PARAM {key}: {value_repr}")
    
    logger.debug(f"{'='*60}")


def log_function_exit(func_name: str, result=None, success: bool = True):
    """
    Log function exit with result.
    
    This helper function creates a log block when exiting a function,
    showing whether it succeeded and what value was returned.
    
    Parameters:
    - func_name: The name of the function being exited
    - result: The return value (optional, may be None)
    - success: Boolean indicating if the function completed successfully
    """
    logger.debug(f"{'='*60}")
    
    # Indicate success or failure
    if success:
        logger.debug(f"EXITING: {func_name} - SUCCESS")
    else:
        logger.debug(f"EXITING: {func_name} - FAILED")
    
    # Log the result if provided
    if result is not None:
        result_repr = repr(result)
        if len(result_repr) > 300:
            result_repr = result_repr[:300] + "..."
        logger.debug(f"  RESULT: {result_repr}")
    
    logger.debug(f"{'='*60}")


def log_step(step_name: str, details: str = None):
    """
    Log an intermediate step within a function.
    
    This helper function logs progress within a function, showing what
    operation is being performed and optionally additional details.
    
    Parameters:
    - step_name: A brief description of the current step
    - details: Optional additional information about the step
    """
    msg = f"STEP: {step_name}"
    if details:
        # Truncate long details to keep logs readable
        if len(details) > 200:
            details = details[:200] + "..."
        msg += f" | {details}"
    logger.debug(msg)


def log_variable(var_name: str, var_value, context: str = None):
    """
    Log a variable's value.
    
    This helper function logs the current value of a variable, optionally
    with context about where in the code this is happening.
    
    Parameters:
    - var_name: The name of the variable
    - var_value: The current value of the variable
    - context: Optional context string (e.g., "after calculation")
    """
    value_repr = repr(var_value)
    if len(value_repr) > 200:
        value_repr = value_repr[:200] + "..."
    
    msg = f"VAR {var_name} = {value_repr}"
    if context:
        msg = f"[{context}] {msg}"
    logger.debug(msg)


def log_exception(func_name: str, exception: Exception):
    """
    Log an exception with full traceback.
    
    This helper function logs detailed information about an exception,
    including the exception type, message, and full stack trace.
    
    Parameters:
    - func_name: The name of the function where the exception occurred
    - exception: The exception object that was raised
    """
    # Create a distinctive visual marker for exceptions
    logger.error(f"{'!'*60}")
    logger.error(f"EXCEPTION in {func_name}: {type(exception).__name__}")
    logger.error(f"  Message: {str(exception)}")
    logger.error(f"  Traceback:")
    
    # Log each line of the traceback separately for readability
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            logger.error(f"    {line}")
    
    logger.error(f"{'!'*60}")


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log DataFrame summary information.
    
    This helper function logs key characteristics of a DataFrame,
    useful for understanding the data being processed.
    
    Parameters:
    - df: The pandas DataFrame to summarize
    - name: A descriptive name for the DataFrame
    """
    logger.debug(f"DataFrame Info - {name}:")
    logger.debug(f"  Shape: {df.shape}")
    logger.debug(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    logger.debug(f"  Dtypes: {dict(list(df.dtypes.items())[:5])}{'...' if len(df.dtypes) > 5 else ''}")
    logger.debug(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


# =============================================================================
# Install/Import Dependencies
# =============================================================================
# This section handles the Shiny library import with automatic installation
# Shiny is a Python library for building interactive web applications

logger.info("Starting Attribute Editor DEBUG version (commentated)")
logger.debug("Attempting to import Shiny...")

# Try to import the required Shiny components
try:
    # App is the main application class that ties together UI and server logic
    # Inputs represents all user inputs from the UI
    # Outputs represents all outputs to display in the UI
    # Session manages the user session state
    # reactive provides reactivity system for automatic UI updates
    # render provides output rendering functions
    # ui provides UI component building functions
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    logger.debug("Shiny imported successfully")

# If Shiny is not installed, catch the ImportError exception
except ImportError:
    logger.warning("Shiny not found, attempting to install...")
    
    # Import subprocess module to run shell commands from Python
    import subprocess
    
    # Run pip install command to install Shiny
    # check_call raises an exception if the command fails
    subprocess.check_call(['pip', 'install', 'shiny'])
    
    # Now import Shiny components after successful installation
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    logger.debug("Shiny installed and imported successfully")


# =============================================================================
# Helper Functions
# =============================================================================
# This section contains utility functions used throughout the script
# These functions handle type detection, string cleaning, and data analysis

def get_column_class(series: pd.Series) -> str:
    """
    Determine the R-equivalent class of a pandas Series.
    
    In R, columns have classes like 'integer', 'numeric', or 'factor'.
    This function maps pandas data types to their R equivalents for
    compatibility with the R-based workflow.
    
    Parameters:
    - series: A pandas Series (single column from a DataFrame)
    
    Returns:
    - 'integer' if the column contains whole numbers
    - 'numeric' if the column contains floating-point numbers
    - 'factor' if the column contains categorical/string data
    """
    func_name = "get_column_class"
    log_function_entry(func_name, series_name=series.name, dtype=str(series.dtype), length=len(series))
    
    try:
        log_step("Checking pandas dtype", f"dtype={series.dtype}")
        
        # Check if the series contains integer data type
        # pd.api.types.is_integer_dtype handles all integer variants (int32, int64, Int32, Int64)
        if pd.api.types.is_integer_dtype(series):
            result = 'integer'
            log_step("Detected integer dtype", f"using pd.api.types.is_integer_dtype")
        
        # Check if the series contains floating-point data type
        # pd.api.types.is_float_dtype handles float32, float64, Float64, etc.
        elif pd.api.types.is_float_dtype(series):
            result = 'numeric'
            log_step("Detected float dtype", f"using pd.api.types.is_float_dtype")
        
        # Check if the series contains boolean (True/False) values
        # In R, booleans are often treated as integers (0/1)
        elif pd.api.types.is_bool_dtype(series):
            result = 'integer'
            log_step("Detected bool dtype", f"treating as integer (0/1)")
        
        # If none of the above, assume it's categorical/string data
        else:
            result = 'factor'
            log_step("Detected object/other dtype", f"treating as factor (categorical)")
        
        log_function_exit(func_name, result)
        return result
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def clean_string_value(val) -> str:
    """
    Clean a string value by removing extra quotes and whitespace.
    
    Data imported from KNIME or CSV files may have extra quoting like
    '"123"' or '""value""' which need to be stripped for proper processing.
    Additionally, common null indicators (NULL, NA, N/A, etc.) are treated
    as None to ensure consistent missing value handling.
    
    Parameters:
    - val: Any value to clean (will be converted to string if not None)
    
    Returns:
    - None if the value is missing or a null indicator string
    - The cleaned string with extra quotes removed
    """
    func_name = "clean_string_value"
    log_function_entry(func_name, val=val, val_type=type(val).__name__)
    
    try:
        # Check if the value is missing (NaN, None, etc.)
        # pd.isna() handles various missing value representations
        if pd.isna(val):
            log_step("Value is NA/null", f"pd.isna returned True")
            log_function_exit(func_name, None)
            return None
        
        # Convert the value to string and remove leading/trailing whitespace
        # str(val) ensures we can work with the value as a string
        # .strip() removes spaces, tabs, newlines from both ends
        s = str(val).strip()
        log_variable("s", s, "after strip")
        
        # Remove surrounding quotes iteratively
        # This handles cases like '"value"' or '""value""' or "'value'"
        # We keep removing quote pairs until none remain
        quote_removal_count = 0
        while len(s) >= 2:
            # Check if string starts AND ends with matching quotes (double or single)
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                # Remove the first and last character (the quotes)
                # Then strip any whitespace that was inside the quotes
                s = s[1:-1].strip()
                quote_removal_count += 1
            else:
                # No more surrounding quotes found, exit the loop
                break
        
        if quote_removal_count > 0:
            log_step(f"Removed {quote_removal_count} layers of quotes", f"result='{s}'")
        
        # Define a set of strings that represent null/missing values
        # These are common indicators in various data formats and systems
        # Using a set for O(1) lookup performance
        null_indicators = {'null', 'na', 'n/a', 'nan', 'none', '', '.', '-'}
        
        # Check if the lowercase version of the string is a null indicator
        # Using lowercase comparison for case-insensitive matching
        if s.lower() in null_indicators:
            log_step("Value is a null indicator", f"value='{s}'")
            log_function_exit(func_name, None)
            return None
        
        # Return the cleaned string, or None if it's empty after cleaning
        # The 'if s else None' handles the case where s is an empty string
        result = s if s else None
        log_function_exit(func_name, result)
        return result
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def is_numeric_convertible(series: pd.Series) -> bool:
    """
    Check if a factor/object column can be converted to numeric.
    
    This is equivalent to R's: is.numeric(type.convert(unique(df[,i])))
    It determines if a string column actually contains numeric values
    that were just stored as strings (common in CSV imports).
    
    Parameters:
    - series: A pandas Series to check for numeric convertibility
    
    Returns:
    - True if all non-null values can be converted to numbers
    - False if any value cannot be converted or if column is empty
    """
    func_name = "is_numeric_convertible"
    log_function_entry(func_name, series_name=series.name, dtype=str(series.dtype))
    
    try:
        # Get all unique non-null values from the series
        # dropna() removes missing values before getting unique values
        # unique() returns an array of distinct values
        unique_vals = series.dropna().unique()
        log_variable("unique_vals_count", len(unique_vals))
        
        # If there are no non-null values, return False
        # Can't determine numeric convertibility of an empty column
        if len(unique_vals) == 0:
            log_step("No unique values found", "returning False")
            log_function_exit(func_name, False)
            return False
        
        # Clean each unique value by removing extra quotes
        # This handles values like '"123"' that should be numeric
        cleaned_vals = [clean_string_value(v) for v in unique_vals]
        
        # Filter out None values and empty strings from the cleaned list
        # These represent missing values and shouldn't affect numeric determination
        cleaned_vals = [v for v in cleaned_vals if v is not None and v != '']
        log_variable("cleaned_vals_count", len(cleaned_vals))
        
        # If all values were null indicators, return False
        # No actual data to determine numeric convertibility
        if len(cleaned_vals) == 0:
            log_step("No cleaned values remaining", "returning False")
            log_function_exit(func_name, False)
            return False
        
        # Try to convert each cleaned value to a number
        # If any conversion fails, an exception will be raised
        log_step("Testing numeric conversion", f"sample values: {cleaned_vals[:3]}")
        for i, val in enumerate(cleaned_vals):
            # pd.to_numeric will raise ValueError if val can't be converted
            pd.to_numeric(val)
            if i < 3:
                log_step(f"Value '{val}' converted successfully")
        
        # All values converted successfully, return True
        log_function_exit(func_name, True)
        return True
    
    # Catch ValueError (invalid conversion) or TypeError (wrong input type)
    except (ValueError, TypeError) as e:
        log_step(f"Conversion failed", f"error: {str(e)}")
        log_function_exit(func_name, False)
        return False
    except Exception as e:
        log_exception(func_name, e)
        raise


def is_integer_values(series: pd.Series) -> bool:
    """
    Check if all numeric values in a series are integers (no decimal part).
    
    This helps distinguish between discrete variables (whole numbers only)
    and continuous variables (decimal values). For example:
    - Ages like 25, 30, 45 are integers -> discrete
    - Amounts like 25.50, 30.75 have decimals -> continuous
    
    Parameters:
    - series: A pandas Series to check
    
    Returns:
    - True if all numeric values are whole numbers
    - False if any value has a decimal part or conversion fails
    """
    func_name = "is_integer_values"
    log_function_entry(func_name, series_name=series.name)
    
    try:
        # Clean each value in the series by removing extra quotes
        # dropna() first removes missing values, then apply cleaning function
        cleaned = series.dropna().apply(lambda x: clean_string_value(x))
        
        # Filter out values that became None or empty string after cleaning
        # These are null indicators that shouldn't affect the integer check
        # notna() returns True for non-null values
        cleaned = cleaned[cleaned.notna() & (cleaned != '')]
        log_variable("cleaned_count", len(cleaned))
        
        # If no values remain after cleaning, return False
        if len(cleaned) == 0:
            log_step("No valid cleaned values", "returning False")
            log_function_exit(func_name, False)
            return False
        
        # Convert cleaned strings to numeric values
        # errors='coerce' converts unconvertible values to NaN instead of raising error
        # Then dropna() removes any values that couldn't be converted
        numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
        log_variable("numeric_vals_count", len(numeric_vals))
        
        # If no values could be converted to numeric, return False
        if len(numeric_vals) == 0:
            log_step("No numeric values after conversion", "returning False")
            log_function_exit(func_name, False)
            return False
        
        # Check if all numeric values equal their rounded version
        # If value == round(value), then the value has no decimal part
        # .all() returns True only if all comparisons are True
        is_int = (numeric_vals == numeric_vals.round()).all()
        log_step("Integer check complete", f"all values are integers: {is_int}")
        log_variable("sample_values", numeric_vals.head(5).tolist())
        
        log_function_exit(func_name, is_int)
        return is_int
    
    # Catch any exception that might occur during processing
    except Exception as e:
        log_exception(func_name, e)
        log_function_exit(func_name, False, success=False)
        return False


def get_top_samples(series: pd.Series, n: int = 5) -> str:
    """
    Get top n unique samples from a series as a comma-separated string.
    
    This provides a preview of the data in a column, helping users
    understand what values exist without looking at the full dataset.
    
    Parameters:
    - series: A pandas Series to sample from
    - n: Maximum number of unique values to include (default 5)
    
    Returns:
    - A comma-separated string of sample values
    """
    func_name = "get_top_samples"
    log_function_entry(func_name, series_name=series.name, n=n)
    
    try:
        # Get all unique non-null values from the series
        # dropna() removes missing values before extracting unique values
        unique_vals = series.dropna().unique()
        log_variable("unique_vals_count", len(unique_vals))
        
        # If there are fewer unique values than requested, use all of them
        if len(unique_vals) <= n:
            samples = unique_vals
        else:
            # Otherwise, take only the first n values
            # Note: this takes the first n, not a random sample
            samples = unique_vals[:n]
        
        log_variable("samples_selected", len(samples))
        
        # Clean each sample value to remove extra quotes
        # This ensures the preview looks clean to the user
        cleaned_samples = []
        
        # Iterate through each sample value
        for v in samples:
            # Apply the cleaning function to remove quotes
            cleaned = clean_string_value(v)
            
            # If cleaning succeeded (non-null result), use the cleaned value
            if cleaned is not None:
                cleaned_samples.append(cleaned)
            else:
                # If cleaning returned None (null indicator), use string representation
                # This preserves visibility of what the original value was
                cleaned_samples.append(str(v))
        
        # Join all samples with comma and space separator
        # Returns a string like "value1, value2, value3"
        result = ", ".join(cleaned_samples)
        log_variable("result", result)
        log_function_exit(func_name, result)
        return result
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def analyze_variable(df: pd.DataFrame, col_name: str, dv: Optional[str] = None) -> dict:
    """
    Analyze a single variable and return its metadata.
    
    This function examines a column's data type, cardinality, null counts,
    and value range to determine appropriate settings for WOE binning
    and model building.
    
    Parameters:
    - df: The pandas DataFrame containing the data
    - col_name: Name of the column to analyze
    - dv: Name of the dependent variable (if known), used to set role
    
    Returns:
    - Dictionary containing all metadata fields for this variable
    """
    func_name = "analyze_variable"
    log_function_entry(func_name, col_name=col_name, dv=dv, df_shape=df.shape)
    
    try:
        # Extract the column as a pandas Series for analysis
        series = df[col_name]
        
        # Determine the R-equivalent data type class
        # Returns 'integer', 'numeric', or 'factor'
        col_class = get_column_class(series)
        
        # Count the number of unique values (cardinality)
        # Low cardinality often indicates categorical/discrete variables
        cardinality = series.nunique()
        
        # Count the number of null/missing values
        # isna() returns True for each null value, sum() counts the Trues
        null_qty = series.isna().sum()
        
        log_variable("col_class", col_class)
        log_variable("cardinality", cardinality)
        log_variable("null_qty", null_qty)
        
        # Initialize the metadata dictionary with default values
        # These values may be overridden based on analysis below
        meta = {
            'VariableName': col_name,
            'Include': True,
            'Role': 'dependent' if col_name == dv else 'independent',
            'Usage': col_class,
            'UsageOriginal': col_class,
            'UsageProposed': "don't",
            'NullQty': null_qty,
            'min': 69,  # Default placeholder like in R code
            'max': 420,  # Default placeholder like in R code
            'Cardinality': cardinality,
            'Samples': get_top_samples(series),
            'DefaultBins': 1,
            'IntervalsType': 'static',
            'BreakApart': 'yes',
            'MissingValues': 'use',
            'OrderedDisplay': 'present',
            'PValue': 0.05
        }
        
        log_step("Initial metadata created", f"Role={meta['Role']}")
        
        # ==========================================================================
        # Handle integer type columns
        # ==========================================================================
        if col_class == 'integer':
            log_step("Processing integer type column")
            
            # Convert values to numeric, dropping any that can't be converted
            # errors='coerce' replaces unconvertible values with NaN
            numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
            
            # Calculate minimum and maximum values
            meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
            meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
            log_variable("min", meta['min'])
            log_variable("max", meta['max'])
            
            # Determine if this should be discrete or continuous based on cardinality
            # Low cardinality (< 21 unique values) suggests discrete categories
            if cardinality < 21:
                meta['UsageOriginal'] = 'discrete'
                meta['UsageProposed'] = 'discrete'
                meta['Usage'] = 'discrete'
                log_step("Low cardinality integer", f"cardinality={cardinality} < 21, set to discrete")
            else:
                meta['UsageOriginal'] = 'continuous'
                meta['UsageProposed'] = 'continuous'
                meta['Usage'] = 'continuous'
                log_step("High cardinality integer", f"cardinality={cardinality} >= 21, set to continuous")
            
            # Set default number of bins based on cardinality
            if cardinality > 10:
                meta['DefaultBins'] = 10
            else:
                meta['DefaultBins'] = cardinality
            log_variable("DefaultBins", meta['DefaultBins'])
        
        # ==========================================================================
        # Handle factor (object/string) type columns
        # ==========================================================================
        elif col_class == 'factor':
            log_step("Processing factor type column")
            
            meta['UsageOriginal'] = 'nominal'
            meta['min'] = 0
            meta['max'] = 0
            meta['Usage'] = 'nominal'
            
            # Check if the string values can be converted to numbers
            is_convertible = is_numeric_convertible(series)
            log_variable("is_convertible", is_convertible)
            
            # If convertible to numeric, check if values are integers
            has_integer_values = is_integer_values(series) if is_convertible else False
            log_variable("has_integer_values", has_integer_values)
            
            # If values cannot be converted to numeric, keep as nominal
            if not is_convertible:
                meta['UsageProposed'] = 'nominal'
                log_step("Not convertible to numeric", "keeping as nominal")
            else:
                # Values can be converted to numeric - calculate actual min/max
                try:
                    cleaned = series.dropna().apply(lambda x: clean_string_value(x))
                    numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        meta['min'] = float(numeric_vals.min())
                        meta['max'] = float(numeric_vals.max())
                        log_variable("min", meta['min'])
                        log_variable("max", meta['max'])
                except Exception as e:
                    log_step(f"Failed to calculate min/max", str(e))
                
                # Determine usage type based on cardinality and value type
                if cardinality < 20:
                    log_step("Low cardinality factor", f"cardinality={cardinality} < 20")
                    if has_integer_values:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                        log_step("Integer values detected", "set to discrete")
                    else:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                        log_step("Float values with low cardinality", "set to discrete")
                    
                    if cardinality < 10:
                        meta['DefaultBins'] = cardinality
                else:
                    log_step("High cardinality factor", f"cardinality={cardinality} >= 20")
                    meta['DefaultBins'] = 10
                    
                    if has_integer_values:
                        if cardinality > 50:
                            meta['UsageProposed'] = 'continuous'
                            meta['Usage'] = 'continuous'
                            log_step("Integer values with very high cardinality", "set to continuous")
                        else:
                            meta['UsageProposed'] = 'discrete'
                            meta['Usage'] = 'discrete'
                            log_step("Integer values with moderate cardinality", "set to discrete")
                    else:
                        meta['UsageProposed'] = 'continuous'
                        meta['Usage'] = 'continuous'
                        log_step("Float values", "set to continuous")
        
        # ==========================================================================
        # Handle numeric (float) type columns
        # ==========================================================================
        elif col_class == 'numeric':
            log_step("Processing numeric (float) type column")
            
            numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
            meta['UsageOriginal'] = 'continuous'
            meta['UsageProposed'] = 'continuous'
            meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
            meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
            meta['Usage'] = 'continuous'
            meta['DefaultBins'] = 10
            log_variable("min", meta['min'])
            log_variable("max", meta['max'])
        
        # ==========================================================================
        # Special handling for dependent variable
        # ==========================================================================
        if col_name == dv:
            log_step("Setting dependent variable special properties")
            meta['Role'] = 'dependent'
            meta['MissingValues'] = 'float'
            meta['OrderedDisplay'] = 'range'
        
        log_function_exit(func_name, f"Usage={meta['Usage']}, Role={meta['Role']}")
        return meta
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def analyze_all_variables(df: pd.DataFrame, dv: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze all variables in a DataFrame and return metadata DataFrame.
    
    This function iterates through every column in the input DataFrame
    and generates metadata for each, compiling results into a single
    DataFrame suitable for output.
    
    Parameters:
    - df: The pandas DataFrame to analyze
    - dv: Name of the dependent variable (optional)
    
    Returns:
    - DataFrame containing metadata for all variables
    """
    func_name = "analyze_all_variables"
    log_function_entry(func_name, df_shape=df.shape, dv=dv, columns=list(df.columns))
    
    try:
        # Initialize empty list to collect metadata dictionaries
        metadata_list = []
        total_cols = len(df.columns)
        
        # Iterate through each column in the DataFrame
        for i, col in enumerate(df.columns):
            log_step(f"Analyzing column {i+1}/{total_cols}", f"column='{col}'")
            meta = analyze_variable(df, col, dv)
            metadata_list.append(meta)
        
        # Convert list of dictionaries to a DataFrame
        df_var = pd.DataFrame(metadata_list)
        log_step("Created metadata DataFrame", f"shape={df_var.shape}")
        
        # Define the desired column order for the output
        column_order = [
            'VariableName', 'Include', 'Role', 'Usage', 'UsageOriginal', 
            'UsageProposed', 'NullQty', 'min', 'max', 'Cardinality', 
            'Samples', 'DefaultBins', 'IntervalsType', 'BreakApart',
            'MissingValues', 'OrderedDisplay', 'PValue'
        ]
        
        result = df_var[column_order]
        log_dataframe_info(result, "Metadata Result")
        log_function_exit(func_name, f"shape={result.shape}")
        return result
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def apply_type_conversions(df: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type conversions based on the metadata DataFrame.
    
    This function transforms the original data according to the metadata
    settings, performing two main operations:
    1. Removes columns where Include == False
    2. Converts column types based on Usage vs UsageOriginal:
       - nominal -> string (cleaned)
       - continuous -> float
       - discrete -> integer
    
    Parameters:
    - df: The original data DataFrame
    - df_out: The metadata DataFrame with conversion settings
    
    Returns:
    - New DataFrame with conversions applied and excluded columns removed
    """
    func_name = "apply_type_conversions"
    log_function_entry(func_name, df_shape=df.shape, df_out_shape=df_out.shape)
    
    try:
        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Initialize list to track columns that should be removed
        columns_to_remove = []
        conversion_count = 0
        
        # Iterate through each row in the metadata DataFrame
        for idx, row in df_out.iterrows():
            var_name = row['VariableName']
            include = row['Include']
            original_type = row['UsageOriginal']
            target_type = row['Usage']
            
            log_step(f"Processing variable", f"name={var_name}, include={include}, {original_type}->{target_type}")
            
            # Check if this column exists in the result DataFrame
            if var_name not in result_df.columns:
                log_step(f"Variable not found in DataFrame", var_name)
                continue
            
            # Check if the variable should be excluded
            if not include:
                log_step(f"Variable excluded", var_name)
                print(f"Variable {var_name} will not be included")
                columns_to_remove.append(var_name)
                continue
            
            # Check if type conversion is needed
            if original_type != target_type:
                log_step(f"Converting type", f"{var_name}: {original_type} -> {target_type}")
                print(f"Variable {var_name} from {original_type} to {target_type}")
                conversion_count += 1
                
                # =================================================================
                # Convert to nominal (string)
                # =================================================================
                if target_type == 'nominal':
                    log_step("Applying nominal conversion", "cleaning string values")
                    result_df[var_name] = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                
                # =================================================================
                # Convert to continuous (float)
                # =================================================================
                elif target_type == 'continuous':
                    log_step("Applying continuous conversion", "converting to Float64")
                    cleaned = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    result_df[var_name] = pd.to_numeric(cleaned, errors='coerce').astype('Float64')
                
                # =================================================================
                # Convert to discrete (integer)
                # =================================================================
                elif target_type == 'discrete':
                    log_step("Applying discrete conversion", "converting to Int32")
                    cleaned = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    numeric_vals = pd.to_numeric(cleaned, errors='coerce')
                    result_df[var_name] = numeric_vals.round().astype('Int32')
                
                # =================================================================
                # Convert to ordinal (treated same as nominal for now)
                # =================================================================
                elif target_type == 'ordinal':
                    log_step("Applying ordinal conversion", "cleaning string values")
                    result_df[var_name] = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                
                # =================================================================
                # No binning - keep as-is
                # =================================================================
                elif target_type == 'no binning':
                    log_step("No binning requested", "keeping as-is")
                    pass
            
            # Type is the same - no conversion needed
            else:
                # Even when type matches, we may need to clean string values
                if original_type in ['nominal', 'factor']:
                    sample_val = result_df[var_name].dropna().iloc[0] if len(result_df[var_name].dropna()) > 0 else None
                    if sample_val is not None and isinstance(sample_val, str):
                        if sample_val.startswith('"') or sample_val.startswith("'"):
                            log_step("Cleaning quoted string values", var_name)
                            result_df[var_name] = result_df[var_name].apply(
                                lambda x: clean_string_value(x) if pd.notna(x) else None
                            )
                print(f"Variable {var_name} not changed")
        
        # Remove columns that were marked for exclusion
        if columns_to_remove:
            log_step(f"Removing {len(columns_to_remove)} excluded columns", str(columns_to_remove))
            result_df = result_df.drop(columns=columns_to_remove)
        
        log_variable("total_conversions", conversion_count)
        log_variable("columns_removed", len(columns_to_remove))
        log_dataframe_info(result_df, "Converted Result")
        log_function_exit(func_name, f"shape={result_df.shape}")
        return result_df
        
    except Exception as e:
        log_exception(func_name, e)
        raise


# =============================================================================
# Shiny UI Application
# =============================================================================
# This section defines the interactive web-based user interface
# The Shiny framework (originally from R, ported to Python) provides
# reactive programming for building dynamic web applications

def create_attribute_editor_app(df: pd.DataFrame, initial_dv: Optional[str] = None):
    """
    Create the Attribute Editor Shiny application.
    
    This function builds the complete Shiny application with:
    - A styled user interface with gradient background
    - Interactive data table for editing variable attributes
    - Dropdown for selecting the dependent variable
    - Submit button to finalize selections
    
    Parameters:
    - df: The data DataFrame to analyze and configure
    - initial_dv: Optional initial selection for dependent variable
    
    Returns:
    - Shiny App object ready to be run
    """
    func_name = "create_attribute_editor_app"
    log_function_entry(func_name, df_shape=df.shape, initial_dv=initial_dv)
    
    try:
        # Dictionary to store application results
        app_results = {
            'df_var': None,
            'completed': False
        }
        
        # Create list of column names for dropdown choices
        column_choices = list(df.columns)
        log_variable("column_choices_count", len(column_choices))
        
        # Define choices for various dropdown fields
        usage_choices = ['continuous', 'nominal', 'ordinal', 'discrete', 'no binning']
        role_choices = ['dependent', 'independent']
        intervals_choices = ['', 'static']
        break_apart_choices = ['yes', 'no']
        missing_choices = ['use', 'ignore', 'float']
        ordered_display_choices = ['range', 'present']
        
        log_step("Building UI components")
        
        # ==========================================================================
        # Define the User Interface (UI)
        # ==========================================================================
        app_ui = ui.page_fluid(
            ui.tags.head(
                ui.tags.style("""
                    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');
                    body { 
                        font-family: 'Raleway', sans-serif; 
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                        min-height: 100vh;
                        color: #e8e8e8;
                    }
                    .card { 
                        background: rgba(255, 255, 255, 0.05); 
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 16px; 
                        padding: 24px; 
                        margin: 12px 0; 
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    }
                    .btn-primary { 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        border-radius: 25px;
                        padding: 10px 24px;
                        font-weight: 600;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    .btn-primary:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                    }
                    .btn-success { 
                        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                        border: none;
                        border-radius: 25px;
                        padding: 12px 36px;
                        font-weight: 700;
                        font-size: 1.1em;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    .btn-success:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 20px rgba(56, 239, 125, 0.4);
                    }
                    .btn-secondary {
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 25px;
                        padding: 10px 24px;
                        color: #e8e8e8;
                    }
                    h4 { 
                        font-weight: 700; 
                        text-align: center; 
                        margin: 24px 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        font-size: 2em;
                    }
                    h5 {
                        color: #a8a8b8;
                        font-weight: 600;
                        margin-bottom: 16px;
                    }
                    .form-control, .form-select {
                        background: rgba(255, 255, 255, 0.08);
                        border: 1px solid rgba(255, 255, 255, 0.15);
                        border-radius: 8px;
                        color: #e8e8e8;
                        padding: 8px 12px;
                    }
                    .form-control:focus, .form-select:focus {
                        background: rgba(255, 255, 255, 0.12);
                        border-color: #667eea;
                        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
                        color: #ffffff;
                    }
                    .form-select option {
                        background: #1a1a2e;
                        color: #e8e8e8;
                    }
                    .table {
                        color: #e8e8e8;
                    }
                    .table th {
                        background: rgba(102, 126, 234, 0.2);
                        border-color: rgba(255, 255, 255, 0.1);
                        font-weight: 600;
                    }
                    .table td {
                        border-color: rgba(255, 255, 255, 0.05);
                        vertical-align: middle;
                    }
                    .table-striped tbody tr:nth-of-type(odd) {
                        background: rgba(255, 255, 255, 0.03);
                    }
                    .table-hover tbody tr:hover {
                        background: rgba(102, 126, 234, 0.1);
                    }
                    .form-check-input {
                        background: rgba(255, 255, 255, 0.1);
                        border-color: rgba(255, 255, 255, 0.3);
                    }
                    .form-check-input:checked {
                        background-color: #667eea;
                        border-color: #667eea;
                    }
                    .divider { 
                        width: 12px; 
                        display: inline-block; 
                    }
                    label {
                        color: #a8a8b8;
                        font-weight: 500;
                        margin-bottom: 6px;
                    }
                    .debug-banner {
                        background: rgba(255, 165, 0, 0.2);
                        border: 1px solid orange;
                        color: orange;
                        padding: 8px;
                        text-align: center;
                        border-radius: 8px;
                        margin-bottom: 10px;
                    }
                """)
            ),
            
            # Debug mode banner
            ui.div(
                {"class": "debug-banner"},
                "🔧 DEBUG MODE ENABLED - Extensive logging active"
            ),
            
            ui.h4("Attribute Editor (DEBUG MODE - Commentated)"),
            
            # Dependent Variable Selection
            ui.div(
                {"class": "card"},
                ui.row(
                    ui.column(6,
                        ui.input_select("dv", "Dependent Variable", 
                                       choices=column_choices,
                                       selected=initial_dv if initial_dv in column_choices else column_choices[0] if column_choices else None)
                    ),
                    ui.column(6,
                        ui.br(),
                        ui.input_action_button("reroll_btn", "🎲 Reroll Samples", class_="btn btn-secondary")
                    )
                )
            ),
            
            # Data Table
            ui.div(
                {"class": "card"},
                ui.h5("Variable Attributes"),
                ui.output_data_frame("var_table")
            ),
            
            # Submit Button
            ui.div(
                {"class": "card", "style": "text-align: center; padding: 20px;"},
                ui.input_action_button("submit_btn", "✈️ Submit", class_="btn btn-success btn-lg"),
            ),
        )
        
        log_step("UI components built successfully")
        
        # ==========================================================================
        # Define the Server Logic
        # ==========================================================================
        def server(input: Inputs, output: Outputs, session: Session):
            """Server function containing all reactive logic."""
            logger.debug("SERVER: Initializing server function")
            
            # Reactive value to hold the variable metadata DataFrame
            df_var_rv = reactive.Value(None)
            
            @reactive.Effect
            def init_table():
                """Initialize the variable table on startup."""
                logger.debug("SERVER: init_table() called")
                dv = input.dv()
                logger.debug(f"SERVER: Selected DV = {dv}")
                df_var = analyze_all_variables(df, dv)
                df_var_rv.set(df_var)
                logger.debug("SERVER: Variable table initialized successfully")
            
            @reactive.Effect
            @reactive.event(input.dv)
            def update_dv():
                """Update roles when dependent variable changes."""
                logger.debug("SERVER: update_dv() triggered by DV change")
                dv = input.dv()
                current_df = df_var_rv.get()
                
                if current_df is not None:
                    current_df = current_df.copy()
                    # Update Role column
                    current_df['Role'] = current_df['VariableName'].apply(
                        lambda x: 'dependent' if x == dv else 'independent'
                    )
                    # Update MissingValues and OrderedDisplay for DV
                    current_df.loc[current_df['VariableName'] == dv, 'MissingValues'] = 'float'
                    current_df.loc[current_df['VariableName'] == dv, 'OrderedDisplay'] = 'range'
                    df_var_rv.set(current_df)
                    logger.debug(f"SERVER: Updated DV to '{dv}', roles reassigned")
            
            @reactive.Effect
            @reactive.event(input.reroll_btn)
            def reroll_samples():
                """Reroll sample values for variables with high cardinality."""
                logger.debug("SERVER: reroll_samples() triggered by button click")
                current_df = df_var_rv.get()
                if current_df is not None:
                    current_df = current_df.copy()
                    reroll_count = 0
                    for idx, row in current_df.iterrows():
                        if row['Cardinality'] > 5:
                            var_name = row['VariableName']
                            unique_vals = df[var_name].dropna().unique()
                            if len(unique_vals) > 5:
                                sample_indices = np.random.choice(len(unique_vals), min(5, len(unique_vals)), replace=False)
                                samples = unique_vals[sample_indices]
                                current_df.loc[idx, 'Samples'] = ", ".join(str(v) for v in samples)
                                reroll_count += 1
                    df_var_rv.set(current_df)
                    logger.debug(f"SERVER: Rerolled samples for {reroll_count} variables")
            
            @output
            @render.data_frame
            def var_table():
                """Render the variable attributes table."""
                logger.debug("SERVER: var_table() render triggered")
                current_df = df_var_rv.get()
                if current_df is None:
                    logger.debug("SERVER: No data available for rendering")
                    return render.DataGrid(pd.DataFrame())
                
                # Select columns for display
                display_cols = [
                    'VariableName', 'Include', 'Role', 'Usage', 'UsageOriginal',
                    'UsageProposed', 'NullQty', 'min', 'max', 'Cardinality',
                    'Samples', 'DefaultBins', 'IntervalsType', 'BreakApart',
                    'MissingValues', 'OrderedDisplay', 'PValue'
                ]
                
                display_df = current_df[display_cols].copy()
                logger.debug(f"SERVER: Rendering DataGrid with {len(display_df)} rows, {len(display_cols)} columns")
                
                return render.DataGrid(
                    display_df,
                    editable=True,
                    selection_mode="rows",
                    height="500px",
                    width="100%"
                )
            
            @reactive.Effect
            @reactive.event(input.var_table_cell_edit)
            def handle_cell_edit():
                """Handle cell edits in the data table."""
                edit_info = input.var_table_cell_edit()
                if edit_info is not None:
                    logger.debug(f"SERVER: Cell edit detected - row={edit_info['row']}, col={edit_info['col']}, value='{edit_info['value']}'")
                    current_df = df_var_rv.get()
                    if current_df is not None:
                        current_df = current_df.copy()
                        row_idx = edit_info['row']
                        col_idx = edit_info['col']
                        new_value = edit_info['value']
                        
                        col_name = current_df.columns[col_idx]
                        old_value = current_df.iloc[row_idx, col_idx]
                        
                        # Handle type conversion for specific columns
                        if col_name == 'Include':
                            new_value = str(new_value).lower() in ['true', '1', 'yes']
                        elif col_name in ['NullQty', 'Cardinality', 'DefaultBins']:
                            try:
                                new_value = int(new_value)
                            except:
                                pass
                        elif col_name in ['min', 'max', 'PValue']:
                            try:
                                new_value = float(new_value)
                            except:
                                pass
                        
                        current_df.iloc[row_idx, col_idx] = new_value
                        df_var_rv.set(current_df)
                        logger.debug(f"SERVER: Updated cell [{row_idx}, {col_name}]: '{old_value}' -> '{new_value}'")
            
            @reactive.Effect
            @reactive.event(input.submit_btn)
            async def submit():
                """Handle submit button click."""
                logger.debug("SERVER: submit() triggered by button click")
                current_df = df_var_rv.get()
                if current_df is not None:
                    app_results['df_var'] = current_df.copy()
                    app_results['completed'] = True
                    logger.debug(f"SERVER: Results saved - {len(current_df)} variables captured")
                
                logger.debug("SERVER: Closing session")
                await session.close()
        
        log_step("Server function defined successfully")
        
        # Create and return the App
        app = App(app_ui, server)
        app.results = app_results
        
        log_function_exit(func_name, "App created successfully")
        return app
        
    except Exception as e:
        log_exception(func_name, e)
        raise


def run_attribute_editor(df: pd.DataFrame, initial_dv: Optional[str] = None, port: int = 8051):
    """
    Run the Attribute Editor application and return results.
    
    This is the main entry point for launching the interactive UI.
    It creates the Shiny app, runs it on the specified port, and
    returns the results after the user closes the app.
    
    Parameters:
    - df: The data DataFrame to analyze and configure
    - initial_dv: Optional initial selection for dependent variable
    - port: Port number to run the web server on (default 8051)
    
    Returns:
    - Dictionary containing:
        - 'df_var': The final metadata DataFrame (or None if cancelled)
        - 'completed': Boolean indicating if user clicked Submit
    """
    func_name = "run_attribute_editor"
    log_function_entry(func_name, df_shape=df.shape, initial_dv=initial_dv, port=port)
    
    try:
        # Create the Shiny application with the provided data
        app = create_attribute_editor_app(df, initial_dv)
        
        # Run the app on the specified port
        log_step("Starting Shiny app", f"port={port}, launch_browser=True")
        app.run(port=port, launch_browser=True)
        
        log_step("Shiny app closed by user")
        log_function_exit(func_name, f"completed={app.results['completed']}")
        return app.results
        
    except Exception as e:
        log_exception(func_name, e)
        raise


# =============================================================================
# Configuration
# =============================================================================
# This section would typically contain configuration constants
# Currently empty but provides a placeholder for future settings

# =============================================================================
# Main Execution
# =============================================================================
# This is where the script execution begins when run in KNIME

logger.info("="*80)
logger.info("ATTRIBUTE EDITOR DEBUG VERSION (COMMENTATED) - MAIN EXECUTION STARTING")
logger.info("="*80)
logger.info(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")

# =============================================================================
# Read Input Data
# =============================================================================
log_step("Reading input data from KNIME")

# Read the first input table (index 0) from KNIME and convert to pandas DataFrame
df = knio.input_tables[0].to_pandas()
log_dataframe_info(df, "Input Data from KNIME")

# =============================================================================
# Preprocess: Replace common null indicator strings with actual NA
# =============================================================================
log_step("Preprocessing - replacing null indicator strings")

# Define a list of common null indicator strings
null_indicators = [
    'NULL', 'null', 'NA', 'na', 'N/A', 'n/a', 
    'NaN', 'nan', 'None', 'none', '.', '-', ''
]
logger.debug(f"Null indicators to replace: {null_indicators}")

# Track how many replacements are made
null_replacement_count = 0

# Iterate through each column in the DataFrame
for col in df.columns:
    # Only process string (object) columns
    if df[col].dtype == 'object':
        before_null_count = df[col].isna().sum()
        
        # Replace null indicator strings with pd.NA
        df[col] = df[col].apply(
            lambda x: pd.NA if (isinstance(x, str) and x.strip() in null_indicators) else x
        )
        
        after_null_count = df[col].isna().sum()
        
        # Log if any replacements were made
        if after_null_count > before_null_count:
            replacements = after_null_count - before_null_count
            null_replacement_count += replacements
            logger.debug(f"Column '{col}': replaced {replacements} null indicators")

logger.debug(f"Total null indicators replaced across all columns: {null_replacement_count}")

# Create a copy of the preprocessed DataFrame
df_temp = df.copy()

# =============================================================================
# Check for Flow Variables
# =============================================================================
log_step("Checking KNIME flow variables")

# Initialize flags and variables with default values
contains_dv = False
is_var_override = False
dv = None

# Attempt to get DependentVariable from flow variables
try:
    dv = knio.flow_variables.get("DependentVariable", None)
    logger.debug(f"DependentVariable flow variable: {repr(dv)}")
except Exception as e:
    logger.debug(f"Failed to get DependentVariable: {str(e)}")

# Attempt to get VarOverride from flow variables
try:
    var_override = knio.flow_variables.get("VarOverride", None)
    logger.debug(f"VarOverride flow variable: {repr(var_override)}")
except Exception as e:
    logger.debug(f"Failed to get VarOverride: {str(e)}")
    var_override = None

# Validate DependentVariable
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        logger.debug(f"Valid DependentVariable found: '{dv}' (exists in columns)")
    else:
        logger.warning(f"DependentVariable '{dv}' NOT found in DataFrame columns")
        logger.warning(f"Available columns: {list(df.columns)}")

# Validate VarOverride
if var_override is not None and isinstance(var_override, int) and var_override == 1:
    is_var_override = True
    logger.debug("VarOverride is enabled (value=1)")

log_variable("contains_dv", contains_dv)
log_variable("is_var_override", is_var_override)

# =============================================================================
# Main Processing Logic
# =============================================================================

# Determine which mode to run based on flow variable conditions
if contains_dv and not is_var_override:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    logger.info("="*60)
    logger.info("RUNNING IN HEADLESS MODE")
    logger.info(f"Dependent Variable: {dv}")
    logger.info("="*60)
    print(f"Running in headless mode with DV: {dv}")
    
    # Analyze all variables automatically using the provided dependent variable
    df_out = analyze_all_variables(df, dv)
    
    print(f"Analyzed {len(df_out)} variables")
    logger.info(f"Headless analysis complete: {len(df_out)} variables analyzed")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    logger.info("="*60)
    logger.info("RUNNING IN INTERACTIVE MODE")
    if not contains_dv:
        logger.info("Reason: No valid DependentVariable provided")
    if is_var_override:
        logger.info("Reason: VarOverride is set to 1")
    logger.info("="*60)
    print("Running in interactive mode - launching Shiny UI...")
    
    # Determine initial DV selection for the UI dropdown
    initial_dv = dv if dv and dv in df.columns else None
    logger.debug(f"Initial DV for UI: {initial_dv}")
    
    # Launch the Shiny application and wait for user to complete
    results = run_attribute_editor(df, initial_dv=initial_dv)
    
    # Check if the user completed the session (clicked Submit)
    if results['completed']:
        df_out = results['df_var']
        print("Interactive session completed successfully")
        logger.info("Interactive session completed successfully - user submitted configuration")
    else:
        print("Interactive session cancelled - generating default metadata")
        logger.warning("Interactive session cancelled - generating default metadata")
        df_out = analyze_all_variables(df, None)

# =============================================================================
# Apply Type Conversions to Original Data
# =============================================================================
log_step("Applying type conversions to data")

# Apply type conversions based on metadata settings
df_converted = apply_type_conversions(df, df_out)

print(f"Applied type conversions. Output has {len(df_converted.columns)} columns.")
logger.info(f"Type conversions applied successfully")
logger.info(f"  Input columns: {len(df.columns)}")
logger.info(f"  Output columns: {len(df_converted.columns)}")
logger.info(f"  Columns removed: {len(df.columns) - len(df_converted.columns)}")

# =============================================================================
# Output Tables
# =============================================================================
log_step("Preparing output tables for KNIME")

# Ensure correct data types for metadata output
logger.debug("Converting metadata column types for KNIME compatibility")
df_out['Include'] = df_out['Include'].astype(bool)
df_out['NullQty'] = df_out['NullQty'].astype('Int32')
df_out['Cardinality'] = df_out['Cardinality'].astype('Int32')
df_out['DefaultBins'] = df_out['DefaultBins'].astype('Int32')
df_out['min'] = df_out['min'].astype('Float64')
df_out['max'] = df_out['max'].astype('Float64')
df_out['PValue'] = df_out['PValue'].astype('Float64')

log_dataframe_info(df_out, "Metadata Output (Output Port 0)")
log_dataframe_info(df_converted, "Converted Data Output (Output Port 1)")

# Output 1: Variable metadata DataFrame
knio.output_tables[0] = knio.Table.from_pandas(df_out)
logger.debug("Output table 0 (metadata) written to KNIME")

# Output 2: Original data with type conversions applied and excluded columns removed
knio.output_tables[1] = knio.Table.from_pandas(df_converted)
logger.debug("Output table 1 (converted data) written to KNIME")

print("Attribute Editor completed successfully")
logger.info("="*80)
logger.info("ATTRIBUTE EDITOR DEBUG VERSION (COMMENTATED) - EXECUTION COMPLETE")
logger.info(f"Execution ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
logger.info("="*80)

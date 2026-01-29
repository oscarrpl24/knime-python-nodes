# =============================================================================
# Attribute Editor for KNIME Python Script Node
# =============================================================================
# This is a Python script designed to run inside a KNIME 5.9 Python Script node.
# It provides functionality equivalent to the R-based Attribute Editor, which is
# used in credit risk modeling workflows to configure variable metadata.
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
# Version: 1.3 - Fixed handling of "NULL" text as missing values for numeric columns
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
# Install/Import Dependencies
# =============================================================================
# This section handles the Shiny library import with automatic installation
# Shiny is a Python library for building interactive web applications

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

# If Shiny is not installed, catch the ImportError exception
except ImportError:
    # Import subprocess module to run shell commands from Python
    import subprocess
    
    # Run pip install command to install Shiny
    # check_call raises an exception if the command fails
    subprocess.check_call(['pip', 'install', 'shiny'])
    
    # Now import Shiny components after successful installation
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui


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
    
    # Check if the series contains integer data type
    # pd.api.types.is_integer_dtype handles all integer variants (int32, int64, Int32, Int64)
    if pd.api.types.is_integer_dtype(series):
        # Return 'integer' to indicate whole number data
        return 'integer'
    
    # Check if the series contains floating-point data type
    # pd.api.types.is_float_dtype handles float32, float64, Float64, etc.
    elif pd.api.types.is_float_dtype(series):
        # Return 'numeric' which is R's term for floating-point numbers
        return 'numeric'
    
    # Check if the series contains boolean (True/False) values
    # In R, booleans are often treated as integers (0/1)
    elif pd.api.types.is_bool_dtype(series):
        # Return 'integer' since booleans can be treated as 0/1
        return 'integer'
    
    # If none of the above, assume it's categorical/string data
    else:
        # Return 'factor' which is R's term for categorical variables
        return 'factor'


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
    
    # Check if the value is missing (NaN, None, etc.)
    # pd.isna() handles various missing value representations
    if pd.isna(val):
        # Return None for missing values
        return None
    
    # Convert the value to string and remove leading/trailing whitespace
    # str(val) ensures we can work with the value as a string
    # .strip() removes spaces, tabs, newlines from both ends
    s = str(val).strip()
    
    # Remove surrounding quotes iteratively
    # This handles cases like '"value"' or '""value""' or "'value'"
    # We keep removing quote pairs until none remain
    while len(s) >= 2:
        # Check if string starts AND ends with matching quotes (double or single)
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            # Remove the first and last character (the quotes)
            # Then strip any whitespace that was inside the quotes
            s = s[1:-1].strip()
        else:
            # No more surrounding quotes found, exit the loop
            break
    
    # Define a set of strings that represent null/missing values
    # These are common indicators in various data formats and systems
    # Using a set for O(1) lookup performance
    null_indicators = {'null', 'na', 'n/a', 'nan', 'none', '', '.', '-'}
    
    # Check if the lowercase version of the string is a null indicator
    # Using lowercase comparison for case-insensitive matching
    if s.lower() in null_indicators:
        # Return None to indicate this is a missing value
        return None
    
    # Return the cleaned string, or None if it's empty after cleaning
    # The 'if s else None' handles the case where s is an empty string
    return s if s else None


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
    
    # Wrap in try-except to catch any conversion errors
    try:
        # Get all unique non-null values from the series
        # dropna() removes missing values before getting unique values
        # unique() returns an array of distinct values
        unique_vals = series.dropna().unique()
        
        # If there are no non-null values, return False
        # Can't determine numeric convertibility of an empty column
        if len(unique_vals) == 0:
            return False
        
        # Clean each unique value by removing extra quotes
        # This handles values like '"123"' that should be numeric
        cleaned_vals = [clean_string_value(v) for v in unique_vals]
        
        # Filter out None values and empty strings from the cleaned list
        # These represent missing values and shouldn't affect numeric determination
        cleaned_vals = [v for v in cleaned_vals if v is not None and v != '']
        
        # If all values were null indicators, return False
        # No actual data to determine numeric convertibility
        if len(cleaned_vals) == 0:
            return False
        
        # Try to convert each cleaned value to a number
        # If any conversion fails, an exception will be raised
        for val in cleaned_vals:
            # pd.to_numeric will raise ValueError if val can't be converted
            pd.to_numeric(val)
        
        # All values converted successfully, return True
        return True
    
    # Catch ValueError (invalid conversion) or TypeError (wrong input type)
    except (ValueError, TypeError):
        # At least one value couldn't be converted, return False
        return False


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
    
    # Wrap in try-except to handle any errors gracefully
    try:
        # Clean each value in the series by removing extra quotes
        # dropna() first removes missing values, then apply cleaning function
        cleaned = series.dropna().apply(lambda x: clean_string_value(x))
        
        # Filter out values that became None or empty string after cleaning
        # These are null indicators that shouldn't affect the integer check
        # notna() returns True for non-null values
        cleaned = cleaned[cleaned.notna() & (cleaned != '')]
        
        # If no values remain after cleaning, return False
        if len(cleaned) == 0:
            return False
        
        # Convert cleaned strings to numeric values
        # errors='coerce' converts unconvertible values to NaN instead of raising error
        # Then dropna() removes any values that couldn't be converted
        numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
        
        # If no values could be converted to numeric, return False
        if len(numeric_vals) == 0:
            return False
        
        # Check if all numeric values equal their rounded version
        # If value == round(value), then the value has no decimal part
        # .all() returns True only if all comparisons are True
        return (numeric_vals == numeric_vals.round()).all()
    
    # Catch any exception that might occur during processing
    except:
        # Return False as a safe default if anything goes wrong
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
    
    # Get all unique non-null values from the series
    # dropna() removes missing values before extracting unique values
    unique_vals = series.dropna().unique()
    
    # If there are fewer unique values than requested, use all of them
    if len(unique_vals) <= n:
        samples = unique_vals
    else:
        # Otherwise, take only the first n values
        # Note: this takes the first n, not a random sample
        samples = unique_vals[:n]
    
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
    return ", ".join(cleaned_samples)


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
    
    # Initialize the metadata dictionary with default values
    # These values may be overridden based on analysis below
    meta = {
        # The name of the variable/column
        'VariableName': col_name,
        
        # Whether to include this variable in modeling (default True)
        'Include': True,
        
        # Role in modeling: 'dependent' for target, 'independent' for features
        # Determined by comparing column name to the DV parameter
        'Role': 'dependent' if col_name == dv else 'independent',
        
        # The usage type that will be applied (may be changed by user)
        'Usage': col_class,
        
        # The original data type detected from the data
        'UsageOriginal': col_class,
        
        # The system's suggested usage type (will be calculated below)
        'UsageProposed': "don't",
        
        # Count of null/missing values in this column
        'NullQty': null_qty,
        
        # Minimum value (for numeric columns), placeholder value 69
        'min': 69,  # Default placeholder like in R code
        
        # Maximum value (for numeric columns), placeholder value 420
        'max': 420,  # Default placeholder like in R code
        
        # Number of unique values
        'Cardinality': cardinality,
        
        # Sample values from the column for preview
        'Samples': get_top_samples(series),
        
        # Default number of bins for WOE binning
        'DefaultBins': 1,
        
        # Type of interval definition for binning ('static' is default)
        'IntervalsType': 'static',
        
        # Whether to break apart combined bins ('yes' is default)
        'BreakApart': 'yes',
        
        # How to handle missing values in binning ('use' includes them)
        'MissingValues': 'use',
        
        # How to display binned values ('present' shows actual values)
        'OrderedDisplay': 'present',
        
        # P-value threshold for statistical significance (default 0.05)
        'PValue': 0.05
    }
    
    # ==========================================================================
    # Handle integer type columns
    # ==========================================================================
    if col_class == 'integer':
        # Convert values to numeric, dropping any that can't be converted
        # errors='coerce' replaces unconvertible values with NaN
        numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
        
        # Calculate minimum value if there are any numeric values
        # float() ensures consistent type for KNIME output
        meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
        
        # Calculate maximum value if there are any numeric values
        meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
        
        # Determine if this should be discrete or continuous based on cardinality
        # Low cardinality (< 21 unique values) suggests discrete categories
        if cardinality < 21:
            # Treat as discrete - each unique value gets its own bin
            meta['UsageOriginal'] = 'discrete'
            meta['UsageProposed'] = 'discrete'
            meta['Usage'] = 'discrete'
        else:
            # High cardinality suggests continuous - will need range-based binning
            meta['UsageOriginal'] = 'continuous'
            meta['UsageProposed'] = 'continuous'
            meta['Usage'] = 'continuous'
        
        # Set default number of bins based on cardinality
        # Use 10 bins for high cardinality, or cardinality for low
        if cardinality > 10:
            meta['DefaultBins'] = 10
        else:
            meta['DefaultBins'] = cardinality
    
    # ==========================================================================
    # Handle factor (object/string) type columns
    # ==========================================================================
    elif col_class == 'factor':
        # Set original usage type to nominal (categorical string data)
        meta['UsageOriginal'] = 'nominal'
        
        # Set min/max to 0 since nominal variables don't have numeric ranges
        meta['min'] = 0
        meta['max'] = 0
        
        # Set initial usage to nominal
        meta['Usage'] = 'nominal'
        
        # Check if the string values can be converted to numbers
        # This detects numeric values stored as strings (common in CSV imports)
        is_convertible = is_numeric_convertible(series)
        
        # If convertible to numeric, check if values are integers
        # This distinguishes between discrete (integers) and continuous (floats)
        has_integer_values = is_integer_values(series) if is_convertible else False
        
        # If values cannot be converted to numeric, keep as nominal
        if not is_convertible:
            meta['UsageProposed'] = 'nominal'
        else:
            # Values can be converted to numeric - calculate actual min/max
            # Wrap in try-except for safety
            try:
                # Clean the values by removing extra quotes
                cleaned = series.dropna().apply(lambda x: clean_string_value(x))
                
                # Convert cleaned strings to numeric values
                # errors='coerce' handles any remaining unconvertible values
                numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
                
                # Calculate min/max if we have any numeric values
                if len(numeric_vals) > 0:
                    meta['min'] = float(numeric_vals.min())
                    meta['max'] = float(numeric_vals.max())
            except:
                # If anything fails, keep the default 0 values
                pass
            
            # Determine usage type based on cardinality and value type
            if cardinality < 20:
                # Low cardinality - likely a discrete/categorical variable
                if has_integer_values:
                    # Integer values with low cardinality -> discrete
                    meta['UsageProposed'] = 'discrete'
                    meta['Usage'] = 'discrete'
                else:
                    # Float values but low cardinality - still treat as discrete
                    # Few unique float values suggest categories
                    meta['UsageProposed'] = 'discrete'
                    meta['Usage'] = 'discrete'
                
                # Set default bins based on low cardinality
                if cardinality < 10:
                    # Use cardinality as bin count for very low cardinality
                    meta['DefaultBins'] = cardinality
            else:
                # High cardinality (20+ unique values)
                meta['DefaultBins'] = 10
                
                # Check if values are integers or floats
                if has_integer_values:
                    # Integer values with high cardinality
                    # Very high (50+) -> continuous, otherwise discrete
                    if cardinality > 50:
                        meta['UsageProposed'] = 'continuous'
                        meta['Usage'] = 'continuous'
                    else:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                else:
                    # Float values with high cardinality -> continuous
                    meta['UsageProposed'] = 'continuous'
                    meta['Usage'] = 'continuous'
    
    # ==========================================================================
    # Handle numeric (float) type columns
    # ==========================================================================
    elif col_class == 'numeric':
        # Convert to numeric (should already be float, but ensure consistency)
        numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
        
        # Float columns are continuous by default
        meta['UsageOriginal'] = 'continuous'
        meta['UsageProposed'] = 'continuous'
        
        # Calculate min and max values
        meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
        meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
        
        # Set usage to continuous
        meta['Usage'] = 'continuous'
        
        # Default to 10 bins for continuous variables
        meta['DefaultBins'] = 10
    
    # ==========================================================================
    # Special handling for dependent variable
    # ==========================================================================
    if col_name == dv:
        # Ensure role is set to dependent (target variable)
        meta['Role'] = 'dependent'
        
        # Set missing values handling to 'float' for dependent variable
        # This means missing DV values will result in excluding those records
        meta['MissingValues'] = 'float'
        
        # Set ordered display to 'range' for dependent variable
        meta['OrderedDisplay'] = 'range'
    
    # Return the complete metadata dictionary
    return meta


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
    
    # Initialize empty list to collect metadata dictionaries
    metadata_list = []
    
    # Iterate through each column in the DataFrame
    for col in df.columns:
        # Analyze this column and get its metadata dictionary
        meta = analyze_variable(df, col, dv)
        
        # Append the metadata to our collection list
        metadata_list.append(meta)
    
    # Convert list of dictionaries to a DataFrame
    # Each dictionary becomes a row, with keys as column names
    df_var = pd.DataFrame(metadata_list)
    
    # Define the desired column order for the output
    # This ensures consistent output regardless of dictionary key order
    column_order = [
        'VariableName',     # Column name
        'Include',          # Whether to include in modeling
        'Role',             # Dependent or independent
        'Usage',            # Current usage type setting
        'UsageOriginal',    # Original detected type
        'UsageProposed',    # System-recommended type
        'NullQty',          # Count of missing values
        'min',              # Minimum value
        'max',              # Maximum value
        'Cardinality',      # Number of unique values
        'Samples',          # Sample values preview
        'DefaultBins',      # Recommended bin count
        'IntervalsType',    # Binning interval type
        'BreakApart',       # Whether to break combined bins
        'MissingValues',    # Missing value handling strategy
        'OrderedDisplay',   # Display order preference
        'PValue'            # P-value significance threshold
    ]
    
    # Return DataFrame with columns in specified order
    return df_var[column_order]


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
    
    # Create a copy of the input DataFrame to avoid modifying the original
    # This is important because we may be applying multiple transformations
    result_df = df.copy()
    
    # Initialize list to track columns that should be removed
    columns_to_remove = []
    
    # Iterate through each row in the metadata DataFrame
    # Each row represents one variable from the original data
    for idx, row in df_out.iterrows():
        # Extract key fields from the metadata row
        var_name = row['VariableName']      # Column name
        include = row['Include']             # Whether to keep the column
        original_type = row['UsageOriginal'] # Original detected type
        target_type = row['Usage']           # Desired type after conversion
        
        # Check if this column exists in the result DataFrame
        # It might have been removed or renamed
        if var_name not in result_df.columns:
            # Skip this variable if it doesn't exist
            continue
        
        # Check if the variable should be excluded
        if not include:
            # Print message indicating exclusion (for logging purposes)
            print(f"Variable {var_name} will not be included")
            
            # Add to removal list (don't remove yet to avoid index issues)
            columns_to_remove.append(var_name)
            
            # Skip to next variable
            continue
        
        # Check if type conversion is needed
        # Only convert if original type differs from target type
        if original_type != target_type:
            # Print message indicating the conversion (for logging)
            print(f"Variable {var_name} from {original_type} to {target_type}")
            
            # =================================================================
            # Convert to nominal (string)
            # =================================================================
            if target_type == 'nominal':
                # Apply cleaning function to each value in the column
                # This removes extra quotes and normalizes null indicators
                result_df[var_name] = result_df[var_name].apply(
                    # Lambda function: clean if not null, else keep as None
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
            
            # =================================================================
            # Convert to continuous (float)
            # =================================================================
            elif target_type == 'continuous':
                # First clean string values (remove quotes, normalize nulls)
                cleaned = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                
                # Then convert cleaned strings to numeric (float)
                # errors='coerce' converts invalid values to NaN
                # astype('Float64') uses nullable float type (capital F)
                result_df[var_name] = pd.to_numeric(cleaned, errors='coerce').astype('Float64')
            
            # =================================================================
            # Convert to discrete (integer)
            # =================================================================
            elif target_type == 'discrete':
                # First clean string values (remove quotes, normalize nulls)
                cleaned = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                
                # Convert cleaned strings to numeric
                numeric_vals = pd.to_numeric(cleaned, errors='coerce')
                
                # Round to nearest integer and convert to nullable Int32
                # round() handles floats that should be integers (e.g., 5.0 -> 5)
                # Int32 (capital I) is nullable integer type required by KNIME
                result_df[var_name] = numeric_vals.round().astype('Int32')
            
            # =================================================================
            # Convert to ordinal (treated same as nominal for now)
            # =================================================================
            elif target_type == 'ordinal':
                # Ordinal is like nominal but with a natural ordering
                # For now, just clean the string values
                result_df[var_name] = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
            
            # =================================================================
            # No binning - keep as-is
            # =================================================================
            elif target_type == 'no binning':
                # 'no binning' means pass through without transformation
                # Used for IDs, dates, or other non-model variables
                pass
        
        # Type is the same - no conversion needed
        else:
            # Even when type matches, we may need to clean string values
            # that have extra quotes (common in CSV imports)
            if original_type in ['nominal', 'factor']:
                # Get a sample value to check if cleaning is needed
                # Take first non-null value from the column
                sample_val = result_df[var_name].dropna().iloc[0] if len(result_df[var_name].dropna()) > 0 else None
                
                # Check if sample value has extra quotes
                if sample_val is not None and isinstance(sample_val, str):
                    # Check if value starts with a quote character
                    if sample_val.startswith('"') or sample_val.startswith("'"):
                        # Apply cleaning to all values in the column
                        result_df[var_name] = result_df[var_name].apply(
                            lambda x: clean_string_value(x) if pd.notna(x) else None
                        )
            
            # Print message indicating no change needed (for logging)
            print(f"Variable {var_name} not changed")
    
    # Remove columns that were marked for exclusion
    if columns_to_remove:
        # Drop the excluded columns from the result DataFrame
        # axis=1 specifies we're dropping columns (not rows)
        result_df = result_df.drop(columns=columns_to_remove)
    
    # Return the transformed DataFrame
    return result_df


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
    
    # Dictionary to store application results
    # This allows data to persist after the app closes
    app_results = {
        'df_var': None,       # Will hold the final metadata DataFrame
        'completed': False    # Flag indicating if user submitted (vs closed)
    }
    
    # Create list of column names for dropdown choices
    # These are the options for selecting the dependent variable
    column_choices = list(df.columns)
    
    # Define choices for various dropdown fields in the editable table
    # Usage types for variable classification
    usage_choices = ['continuous', 'nominal', 'ordinal', 'discrete', 'no binning']
    
    # Role choices for variable function
    role_choices = ['dependent', 'independent']
    
    # Interval type choices for binning
    intervals_choices = ['', 'static']
    
    # Break apart choices for bin handling
    break_apart_choices = ['yes', 'no']
    
    # Missing value handling choices
    missing_choices = ['use', 'ignore', 'float']
    
    # Display ordering choices
    ordered_display_choices = ['range', 'present']
    
    # ==========================================================================
    # Define the User Interface (UI)
    # ==========================================================================
    # The UI is built using Shiny's declarative UI components
    # Each component is a function that returns HTML elements
    
    app_ui = ui.page_fluid(
        # page_fluid creates a responsive full-width page layout
        
        # =======================================================================
        # Head section with CSS styles
        # =======================================================================
        ui.tags.head(
            # tags.head adds elements to the HTML <head> section
            
            ui.tags.style("""
                /* Import Raleway font from Google Fonts for modern typography */
                @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');
                
                /* Body styles - dark gradient background with light text */
                body { 
                    font-family: 'Raleway', sans-serif;  /* Use Raleway font */
                    /* Gradient from dark purple to blue */
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;  /* Full viewport height minimum */
                    color: #e8e8e8;     /* Light gray text color */
                }
                
                /* Card component styles - glassmorphism effect */
                .card { 
                    background: rgba(255, 255, 255, 0.05);  /* Semi-transparent white */
                    backdrop-filter: blur(10px);  /* Blur effect behind card */
                    border: 1px solid rgba(255, 255, 255, 0.1);  /* Subtle border */
                    border-radius: 16px;   /* Rounded corners */
                    padding: 24px;          /* Internal spacing */
                    margin: 12px 0;         /* Vertical spacing between cards */
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);  /* Drop shadow */
                }
                
                /* Primary button styles - purple gradient */
                .btn-primary { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border: none;           /* Remove default border */
                    border-radius: 25px;    /* Pill shape */
                    padding: 10px 24px;     /* Button padding */
                    font-weight: 600;       /* Semi-bold text */
                    transition: transform 0.2s, box-shadow 0.2s;  /* Smooth animations */
                }
                
                /* Primary button hover effect */
                .btn-primary:hover {
                    transform: translateY(-2px);  /* Lift up effect */
                    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);  /* Glow effect */
                }
                
                /* Success button styles - green gradient */
                .btn-success { 
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    border: none;
                    border-radius: 25px;
                    padding: 12px 36px;     /* Larger padding for submit button */
                    font-weight: 700;       /* Bold text */
                    font-size: 1.1em;       /* Slightly larger text */
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                
                /* Success button hover effect */
                .btn-success:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 20px rgba(56, 239, 125, 0.4);  /* Green glow */
                }
                
                /* Secondary button styles - transparent with border */
                .btn-secondary {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 25px;
                    padding: 10px 24px;
                    color: #e8e8e8;
                }
                
                /* h4 heading styles - gradient text effect */
                h4 { 
                    font-weight: 700;       /* Bold */
                    text-align: center;     /* Centered */
                    margin: 24px 0;         /* Vertical spacing */
                    /* Gradient text using background-clip technique */
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-size: 2em;         /* Large text */
                }
                
                /* h5 subheading styles */
                h5 {
                    color: #a8a8b8;         /* Muted gray */
                    font-weight: 600;       /* Semi-bold */
                    margin-bottom: 16px;    /* Space below */
                }
                
                /* Form input and select styles */
                .form-control, .form-select {
                    background: rgba(255, 255, 255, 0.08);  /* Dark transparent */
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 8px;
                    color: #e8e8e8;         /* Light text */
                    padding: 8px 12px;
                }
                
                /* Form input focus state */
                .form-control:focus, .form-select:focus {
                    background: rgba(255, 255, 255, 0.12);  /* Slightly brighter */
                    border-color: #667eea;   /* Purple border */
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);  /* Glow ring */
                    color: #ffffff;          /* White text */
                }
                
                /* Dropdown option styles */
                .form-select option {
                    background: #1a1a2e;    /* Dark background */
                    color: #e8e8e8;         /* Light text */
                }
                
                /* Table text color */
                .table {
                    color: #e8e8e8;
                }
                
                /* Table header styles */
                .table th {
                    background: rgba(102, 126, 234, 0.2);  /* Purple tint */
                    border-color: rgba(255, 255, 255, 0.1);
                    font-weight: 600;
                }
                
                /* Table cell styles */
                .table td {
                    border-color: rgba(255, 255, 255, 0.05);
                    vertical-align: middle;  /* Center content vertically */
                }
                
                /* Alternating row background (zebra striping) */
                .table-striped tbody tr:nth-of-type(odd) {
                    background: rgba(255, 255, 255, 0.03);
                }
                
                /* Table row hover effect */
                .table-hover tbody tr:hover {
                    background: rgba(102, 126, 234, 0.1);  /* Purple tint on hover */
                }
                
                /* Checkbox styles */
                .form-check-input {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: rgba(255, 255, 255, 0.3);
                }
                
                /* Checked checkbox styles */
                .form-check-input:checked {
                    background-color: #667eea;  /* Purple fill */
                    border-color: #667eea;
                }
                
                /* Divider/spacer utility class */
                .divider { 
                    width: 12px; 
                    display: inline-block; 
                }
                
                /* Label styles */
                label {
                    color: #a8a8b8;         /* Muted gray */
                    font-weight: 500;       /* Medium weight */
                    margin-bottom: 6px;     /* Space below */
                }
            """)
        ),
        
        # =======================================================================
        # Page Title
        # =======================================================================
        ui.h4("Attribute Editor"),
        # h4 creates an <h4> heading element with the title text
        
        # =======================================================================
        # Dependent Variable Selection Card
        # =======================================================================
        ui.div(
            # div creates a container element with specified class
            {"class": "card"},  # Apply card styling
            
            ui.row(
                # row creates a Bootstrap grid row for horizontal layout
                
                # Left column (6 of 12 columns = 50% width)
                ui.column(6,
                    # Dropdown select for choosing the dependent variable
                    ui.input_select(
                        "dv",                    # Input ID for referencing in server
                        "Dependent Variable",    # Label text
                        choices=column_choices,  # List of options
                        # Set initial selection: use provided DV if valid, else first column
                        selected=initial_dv if initial_dv in column_choices else column_choices[0] if column_choices else None
                    )
                ),
                
                # Right column (6 of 12 columns = 50% width)
                ui.column(6,
                    ui.br(),  # Line break for vertical alignment
                    
                    # Button to randomize sample values shown
                    ui.input_action_button(
                        "reroll_btn",           # Button ID for referencing in server
                        "🎲 Reroll Samples",    # Button text with dice emoji
                        class_="btn btn-secondary"  # Apply secondary button styling
                    )
                )
            )
        ),
        
        # =======================================================================
        # Data Table Card
        # =======================================================================
        ui.div(
            {"class": "card"},  # Apply card styling
            
            # Section heading
            ui.h5("Variable Attributes"),
            
            # Output placeholder for the data table
            # The actual table content is rendered by the server function
            ui.output_data_frame("var_table")
        ),
        
        # =======================================================================
        # Submit Button Card
        # =======================================================================
        ui.div(
            # Card with centered content
            {"class": "card", "style": "text-align: center; padding: 20px;"},
            
            # Submit button to finalize selections
            ui.input_action_button(
                "submit_btn",                    # Button ID
                "✈️ Submit",                     # Button text with airplane emoji
                class_="btn btn-success btn-lg"  # Green button with large size
            ),
        ),
    )
    
    # ==========================================================================
    # Define the Server Logic
    # ==========================================================================
    # The server function contains all the reactive logic that responds
    # to user interactions and updates the UI accordingly
    
    def server(input: Inputs, output: Outputs, session: Session):
        """
        Server function containing all reactive logic.
        
        Parameters:
        - input: Object containing all user input values
        - output: Object for registering output renderers
        - session: Session object for managing the user session
        """
        
        # ======================================================================
        # Reactive Values
        # ======================================================================
        # Reactive values are special containers that trigger UI updates
        # when their contents change
        
        # Create a reactive value to hold the variable metadata DataFrame
        # Initialize as None; will be populated when app starts
        df_var_rv = reactive.Value(None)
        
        # ======================================================================
        # Initialize Table on Startup
        # ======================================================================
        @reactive.Effect
        def init_table():
            """
            Initialize the variable table when the application starts.
            
            This reactive effect runs once on startup and creates the
            initial metadata DataFrame based on the selected dependent variable.
            """
            # Get the currently selected dependent variable from the dropdown
            dv = input.dv()
            
            # Analyze all variables in the DataFrame
            # This creates the initial metadata with the selected DV
            df_var = analyze_all_variables(df, dv)
            
            # Store the metadata in the reactive value
            # This triggers any outputs that depend on df_var_rv to update
            df_var_rv.set(df_var)
        
        # ======================================================================
        # Update When Dependent Variable Changes
        # ======================================================================
        @reactive.Effect
        @reactive.event(input.dv)  # Only trigger when dv dropdown changes
        def update_dv():
            """
            Update roles when the dependent variable selection changes.
            
            When the user selects a different dependent variable, this
            function updates the Role column to reflect the new selection.
            """
            # Get the newly selected dependent variable
            dv = input.dv()
            
            # Get the current metadata DataFrame
            current_df = df_var_rv.get()
            
            # Only proceed if we have a DataFrame to update
            if current_df is not None:
                # Create a copy to avoid modifying the original in place
                current_df = current_df.copy()
                
                # Update the Role column for all variables
                # Set to 'dependent' if variable name matches DV, else 'independent'
                current_df['Role'] = current_df['VariableName'].apply(
                    lambda x: 'dependent' if x == dv else 'independent'
                )
                
                # Update MissingValues setting for the dependent variable
                # DV should use 'float' handling (exclude records with missing DV)
                current_df.loc[current_df['VariableName'] == dv, 'MissingValues'] = 'float'
                
                # Update OrderedDisplay setting for the dependent variable
                current_df.loc[current_df['VariableName'] == dv, 'OrderedDisplay'] = 'range'
                
                # Store the updated DataFrame in the reactive value
                df_var_rv.set(current_df)
        
        # ======================================================================
        # Reroll Samples Button Handler
        # ======================================================================
        @reactive.Effect
        @reactive.event(input.reroll_btn)  # Only trigger when button is clicked
        def reroll_samples():
            """
            Reroll sample values for variables with high cardinality.
            
            When the user clicks the Reroll Samples button, this function
            randomly selects new sample values to display in the Samples column.
            """
            # Get the current metadata DataFrame
            current_df = df_var_rv.get()
            
            # Only proceed if we have a DataFrame
            if current_df is not None:
                # Create a copy to modify
                current_df = current_df.copy()
                
                # Iterate through each row (variable) in the metadata
                for idx, row in current_df.iterrows():
                    # Only reroll for variables with more than 5 unique values
                    # Variables with <= 5 unique values already show all samples
                    if row['Cardinality'] > 5:
                        # Get the variable name
                        var_name = row['VariableName']
                        
                        # Get all unique non-null values from the original data
                        unique_vals = df[var_name].dropna().unique()
                        
                        # Only reroll if there are more than 5 unique values
                        if len(unique_vals) > 5:
                            # Randomly select up to 5 indices without replacement
                            sample_indices = np.random.choice(
                                len(unique_vals),              # Range to sample from
                                min(5, len(unique_vals)),      # Number of samples
                                replace=False                   # No duplicates
                            )
                            
                            # Get the values at the selected indices
                            samples = unique_vals[sample_indices]
                            
                            # Update the Samples column with new comma-separated values
                            current_df.loc[idx, 'Samples'] = ", ".join(str(v) for v in samples)
                
                # Store the updated DataFrame
                df_var_rv.set(current_df)
        
        # ======================================================================
        # Render the Data Table
        # ======================================================================
        @output
        @render.data_frame
        def var_table():
            """
            Render the variable attributes table.
            
            This function creates the interactive DataGrid that displays
            the variable metadata and allows editing.
            
            Returns:
            - DataGrid object for display in the UI
            """
            # Get the current metadata DataFrame from reactive value
            current_df = df_var_rv.get()
            
            # If no data yet, return empty grid
            if current_df is None:
                return render.DataGrid(pd.DataFrame())
            
            # Define which columns to display in the table
            # All columns from the metadata DataFrame
            display_cols = [
                'VariableName',    # Column name
                'Include',         # Include in modeling checkbox
                'Role',            # Dependent or independent
                'Usage',           # Usage type setting
                'UsageOriginal',   # Original detected type
                'UsageProposed',   # System recommendation
                'NullQty',         # Missing value count
                'min',             # Minimum value
                'max',             # Maximum value
                'Cardinality',     # Unique value count
                'Samples',         # Sample values
                'DefaultBins',     # Recommended bin count
                'IntervalsType',   # Interval type
                'BreakApart',      # Break apart setting
                'MissingValues',   # Missing handling strategy
                'OrderedDisplay',  # Display order
                'PValue'           # P-value threshold
            ]
            
            # Select only the display columns (creates a copy)
            display_df = current_df[display_cols].copy()
            
            # Create and return the DataGrid widget
            return render.DataGrid(
                display_df,              # Data to display
                editable=True,           # Allow cell editing
                selection_mode="rows",   # Enable row selection
                height="500px",          # Fixed height with scrolling
                width="100%"             # Full width
            )
        
        # ======================================================================
        # Handle Cell Edits
        # ======================================================================
        @reactive.Effect
        @reactive.event(input.var_table_cell_edit)  # Trigger on any cell edit
        def handle_cell_edit():
            """
            Handle cell edits in the data table.
            
            When the user edits a cell in the DataGrid, this function
            captures the edit and updates the underlying DataFrame.
            """
            # Get the edit information (row, column, new value)
            edit_info = input.var_table_cell_edit()
            
            # Only proceed if we have edit information
            if edit_info is not None:
                # Get the current metadata DataFrame
                current_df = df_var_rv.get()
                
                # Only proceed if we have a DataFrame
                if current_df is not None:
                    # Create a copy to modify
                    current_df = current_df.copy()
                    
                    # Extract edit details
                    row_idx = edit_info['row']     # Row index that was edited
                    col_idx = edit_info['col']     # Column index that was edited
                    new_value = edit_info['value'] # New value entered by user
                    
                    # Get the column name from the index
                    col_name = current_df.columns[col_idx]
                    
                    # ==========================================================
                    # Type conversion for specific columns
                    # ==========================================================
                    # Different columns expect different data types
                    # Convert the string input to the appropriate type
                    
                    if col_name == 'Include':
                        # Include is a boolean - convert string to True/False
                        # Accept various truthy string values
                        new_value = str(new_value).lower() in ['true', '1', 'yes']
                    
                    elif col_name in ['NullQty', 'Cardinality', 'DefaultBins']:
                        # These columns expect integer values
                        try:
                            new_value = int(new_value)
                        except:
                            # If conversion fails, keep the string value
                            pass
                    
                    elif col_name in ['min', 'max', 'PValue']:
                        # These columns expect float values
                        try:
                            new_value = float(new_value)
                        except:
                            # If conversion fails, keep the string value
                            pass
                    
                    # Update the cell in the DataFrame
                    # iloc allows integer-based indexing
                    current_df.iloc[row_idx, col_idx] = new_value
                    
                    # Store the updated DataFrame
                    df_var_rv.set(current_df)
        
        # ======================================================================
        # Submit Button Handler
        # ======================================================================
        @reactive.Effect
        @reactive.event(input.submit_btn)  # Trigger when submit button clicked
        async def submit():
            """
            Handle submit button click.
            
            When the user clicks Submit, this function saves the current
            metadata DataFrame to the results dictionary and closes the session.
            
            Note: This is an async function because session.close() is async.
            """
            # Get the current metadata DataFrame
            current_df = df_var_rv.get()
            
            # Only save if we have a DataFrame
            if current_df is not None:
                # Copy the DataFrame to the results dictionary
                # This preserves the data after the app closes
                app_results['df_var'] = current_df.copy()
                
                # Set the completed flag to True
                # This indicates the user submitted (vs just closing the window)
                app_results['completed'] = True
            
            # Close the Shiny session
            # This stops the app and returns control to the calling code
            await session.close()
    
    # ==========================================================================
    # Create and Return the App
    # ==========================================================================
    
    # Create the Shiny App by combining the UI and server
    # The App class ties together the UI definition and server logic
    app = App(app_ui, server)
    
    # Attach the results dictionary to the app object
    # This allows the calling code to access results after the app runs
    app.results = app_results
    
    # Return the configured app object
    return app


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
    
    # Create the Shiny application with the provided data
    app = create_attribute_editor_app(df, initial_dv)
    
    # Run the app on the specified port
    # launch_browser=True automatically opens the default web browser
    # This call blocks until the user closes the app
    app.run(port=port, launch_browser=True)
    
    # Return the results dictionary attached to the app
    # Contains the metadata DataFrame and completion status
    return app.results


# =============================================================================
# Configuration
# =============================================================================
# This section would typically contain configuration constants
# Currently empty but provides a placeholder for future settings

# =============================================================================
# Read Input Data
# =============================================================================
# This section reads the input data from KNIME's input port

# Read the first input table (index 0) from KNIME and convert to pandas DataFrame
# knio.input_tables[0] accesses the first input port of the Python Script node
# to_pandas() converts the KNIME table to a pandas DataFrame
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Preprocess: Replace common null indicator strings with actual NA
# =============================================================================
# KNIME may pass "NULL", "NA", etc. as string values instead of actual
# missing values. This section normalizes these to pandas NA for consistent
# handling throughout the script.

# Define a list of common null indicator strings
# These are values that represent "missing" in various data systems
null_indicators = [
    'NULL',   # Standard SQL null
    'null',   # Lowercase variant
    'NA',     # R's missing value
    'na',     # Lowercase variant
    'N/A',    # Common abbreviation
    'n/a',    # Lowercase variant
    'NaN',    # Python/pandas not-a-number
    'nan',    # Lowercase variant
    'None',   # Python's None as string
    'none',   # Lowercase variant
    '.',      # SAS missing value
    '-',      # Common placeholder
    ''        # Empty string
]

# Iterate through each column in the DataFrame
for col in df.columns:
    # Only process string (object) columns
    # Numeric columns don't need this preprocessing
    if df[col].dtype == 'object':  # 'object' is pandas dtype for strings
        # Replace null indicator strings with pd.NA (pandas missing value)
        # Apply a lambda function to each value in the column
        df[col] = df[col].apply(
            # Lambda function checks:
            # 1. Is it a string? (isinstance check)
            # 2. Is the stripped value in our null_indicators list?
            # If both true, replace with pd.NA; otherwise keep original
            lambda x: pd.NA if (isinstance(x, str) and x.strip() in null_indicators) else x
        )

# Create a copy of the preprocessed DataFrame
# This preserves the original data for reference if needed
df_temp = df.copy()

# =============================================================================
# Check for Flow Variables
# =============================================================================
# Flow variables are KNIME's mechanism for passing parameters between nodes
# This section reads and validates the flow variables that control script behavior

# Initialize flags and variables with default values
contains_dv = False       # Flag: Is DependentVariable provided and valid?
is_var_override = False   # Flag: Should we force interactive mode?
dv = None                 # The dependent variable name (if provided)

# Attempt to get DependentVariable from flow variables
# Wrapped in try-except because flow variable may not exist
try:
    # get() returns the value if the key exists, or the default (None) if not
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    # If any error occurs (shouldn't happen with .get()), dv remains None
    pass

# Attempt to get VarOverride from flow variables
# VarOverride = 1 forces interactive mode even when DV is set
try:
    var_override = knio.flow_variables.get("VarOverride", None)
except:
    # If any error occurs, var_override remains None
    var_override = None

# Validate DependentVariable
# Must be: not None, a string, non-empty, not "missing", and exist in the data
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    # Check if the specified column actually exists in the DataFrame
    if dv in df.columns:
        # Valid DependentVariable found
        contains_dv = True

# Validate VarOverride
# Must be an integer equal to 1 to trigger override
if var_override is not None and isinstance(var_override, int) and var_override == 1:
    # VarOverride is set to 1, enable override
    is_var_override = True

# =============================================================================
# Main Processing Logic
# =============================================================================
# The script runs in one of two modes based on flow variable settings:
# 1. Headless Mode: Automatic processing without user interaction
# 2. Interactive Mode: Launches Shiny UI for user configuration

# Determine which mode to run based on flow variable conditions
if contains_dv and not is_var_override:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    # Conditions for headless mode:
    # - DependentVariable is provided and valid (contains_dv = True)
    # - VarOverride is NOT set to 1 (is_var_override = False)
    
    # Print status message for logging
    print(f"Running in headless mode with DV: {dv}")
    
    # Analyze all variables automatically using the provided dependent variable
    # This creates the metadata DataFrame without user interaction
    df_out = analyze_all_variables(df, dv)
    
    # Print confirmation message with variable count
    print(f"Analyzed {len(df_out)} variables")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    # Conditions for interactive mode:
    # - DependentVariable is NOT provided, OR
    # - VarOverride IS set to 1 (forcing interactive mode)
    
    # Print status message for logging
    print("Running in interactive mode - launching Shiny UI...")
    
    # Determine initial DV selection for the UI dropdown
    # Use the flow variable value if it's valid and exists in the data
    initial_dv = dv if dv and dv in df.columns else None
    
    # Launch the Shiny application and wait for user to complete
    # This call blocks until the user closes the UI
    results = run_attribute_editor(df, initial_dv=initial_dv)
    
    # Check if the user completed the session (clicked Submit)
    if results['completed']:
        # User clicked Submit - use their configuration
        df_out = results['df_var']
        print("Interactive session completed successfully")
    else:
        # User closed the window without submitting
        # Generate default metadata as fallback
        print("Interactive session cancelled - generating default metadata")
        df_out = analyze_all_variables(df, None)

# =============================================================================
# Apply Type Conversions to Original Data
# =============================================================================
# Now that we have the metadata (either from headless or interactive mode),
# apply the type conversions to create the output data

# Apply type conversions based on metadata settings
# This transforms the data according to Usage settings and removes excluded columns
df_converted = apply_type_conversions(df, df_out)

# Print confirmation with column count
# Note: df_converted may have fewer columns than df if some were excluded
print(f"Applied type conversions. Output has {len(df_converted.columns)} columns.")

# =============================================================================
# Output Tables
# =============================================================================
# Prepare the output data for KNIME by ensuring correct data types

# Ensure correct data types for metadata output
# KNIME requires specific types for proper column mapping

# Include column should be boolean type
df_out['Include'] = df_out['Include'].astype(bool)

# Integer columns should use nullable Int32 (capital I for nullable)
# This allows KNIME to properly handle potential null values
df_out['NullQty'] = df_out['NullQty'].astype('Int32')
df_out['Cardinality'] = df_out['Cardinality'].astype('Int32')
df_out['DefaultBins'] = df_out['DefaultBins'].astype('Int32')

# Float columns should use nullable Float64 (capital F for nullable)
df_out['min'] = df_out['min'].astype('Float64')
df_out['max'] = df_out['max'].astype('Float64')
df_out['PValue'] = df_out['PValue'].astype('Float64')

# Output 1: Variable metadata DataFrame
# This goes to the first output port (index 0)
# Contains all the variable configuration settings
# knio.Table.from_pandas() converts the DataFrame back to KNIME table format
knio.output_tables[0] = knio.Table.from_pandas(df_out)

# Output 2: Original data with type conversions applied and excluded columns removed
# This goes to the second output port (index 1)
# Contains the transformed data ready for downstream processing
knio.output_tables[1] = knio.Table.from_pandas(df_converted)

# Print final success message
print("Attribute Editor completed successfully")

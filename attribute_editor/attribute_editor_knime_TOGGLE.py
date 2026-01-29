# =============================================================================
# Attribute Editor for KNIME Python Script Node - TOGGLE DEBUG VERSION
# =============================================================================
# Python implementation matching R's Attribute Editor functionality
# Compatible with KNIME 5.9, Python 3.9
#
# TOGGLE DEBUG VERSION: Debugging can be enabled/disabled via DEBUG_ENABLED flag
#
# This script has two modes:
# 1. Headless - When DependentVariable is provided AND VarOverride != 1
# 2. Interactive (Shiny UI) - Otherwise
#
# Flow Variables:
# - DependentVariable (string): Name of the dependent variable column
# - VarOverride (integer): If 1, launches interactive UI even with DV set
#
# Outputs:
# 1. Variable metadata DataFrame (VariableName, Include, Role, Usage, etc.)
# 2. Converted data DataFrame with:
#    - Type conversions applied (nominal→string, continuous→float, discrete→int)
#    - Excluded columns removed (where Include == False)
#
# Release Date: 2026-01-28
# Version: 1.4-TOGGLE - Toggle debug version with controllable logging
# =============================================================================

# =============================================================================
# DEBUG TOGGLE - Set to True to enable extensive logging, False to disable
# =============================================================================
DEBUG_ENABLED = True
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
import logging
import traceback
import sys
from datetime import datetime
from typing import List, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# Configure Debug Logging (conditional on DEBUG_ENABLED)
# =============================================================================

# Create a custom logger
logger = logging.getLogger('AttributeEditorDebug')

if DEBUG_ENABLED:
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%H:%M:%S.%f'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger (avoid duplicate handlers)
    if not logger.handlers:
        logger.addHandler(console_handler)
else:
    # Disable logging when DEBUG_ENABLED is False
    logger.setLevel(logging.CRITICAL + 1)  # Effectively disables all logging
    logger.addHandler(logging.NullHandler())


def log_function_entry(func_name: str, **kwargs):
    """Log function entry with parameters."""
    if not DEBUG_ENABLED:
        return
    logger.debug(f"{'='*60}")
    logger.debug(f"ENTERING: {func_name}")
    if kwargs:
        for key, value in kwargs.items():
            value_repr = repr(value)
            if len(value_repr) > 200:
                value_repr = value_repr[:200] + "..."
            logger.debug(f"  PARAM {key}: {value_repr}")
    logger.debug(f"{'='*60}")


def log_function_exit(func_name: str, result=None, success: bool = True):
    """Log function exit with result."""
    if not DEBUG_ENABLED:
        return
    logger.debug(f"{'='*60}")
    if success:
        logger.debug(f"EXITING: {func_name} - SUCCESS")
    else:
        logger.debug(f"EXITING: {func_name} - FAILED")
    if result is not None:
        result_repr = repr(result)
        if len(result_repr) > 300:
            result_repr = result_repr[:300] + "..."
        logger.debug(f"  RESULT: {result_repr}")
    logger.debug(f"{'='*60}")


def log_step(step_name: str, details: str = None):
    """Log an intermediate step within a function."""
    if not DEBUG_ENABLED:
        return
    msg = f"STEP: {step_name}"
    if details:
        if len(details) > 200:
            details = details[:200] + "..."
        msg += f" | {details}"
    logger.debug(msg)


def log_variable(var_name: str, var_value, context: str = None):
    """Log a variable's value."""
    if not DEBUG_ENABLED:
        return
    value_repr = repr(var_value)
    if len(value_repr) > 200:
        value_repr = value_repr[:200] + "..."
    msg = f"VAR {var_name} = {value_repr}"
    if context:
        msg = f"[{context}] {msg}"
    logger.debug(msg)


def log_exception(func_name: str, exception: Exception):
    """Log an exception with full traceback."""
    if not DEBUG_ENABLED:
        return
    logger.error(f"{'!'*60}")
    logger.error(f"EXCEPTION in {func_name}: {type(exception).__name__}")
    logger.error(f"  Message: {str(exception)}")
    logger.error(f"  Traceback:")
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            logger.error(f"    {line}")
    logger.error(f"{'!'*60}")


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Log DataFrame summary information."""
    if not DEBUG_ENABLED:
        return
    logger.debug(f"DataFrame Info - {name}:")
    logger.debug(f"  Shape: {df.shape}")
    logger.debug(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    logger.debug(f"  Dtypes: {dict(list(df.dtypes.items())[:5])}{'...' if len(df.dtypes) > 5 else ''}")
    logger.debug(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


# =============================================================================
# Install/Import Dependencies
# =============================================================================

if DEBUG_ENABLED:
    logger.info("Starting Attribute Editor TOGGLE DEBUG version")
    logger.debug("Attempting to import Shiny...")

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    if DEBUG_ENABLED:
        logger.debug("Shiny imported successfully")
except ImportError:
    if DEBUG_ENABLED:
        logger.warning("Shiny not found, attempting to install...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    if DEBUG_ENABLED:
        logger.debug("Shiny installed and imported successfully")


# =============================================================================
# Helper Functions
# =============================================================================

def get_column_class(series: pd.Series) -> str:
    """
    Determine the R-equivalent class of a pandas Series.
    Returns: 'integer', 'numeric', or 'factor'
    """
    func_name = "get_column_class"
    log_function_entry(func_name, series_name=series.name, dtype=str(series.dtype), length=len(series))
    
    try:
        log_step("Checking dtype", f"dtype={series.dtype}")
        
        if pd.api.types.is_integer_dtype(series):
            result = 'integer'
            log_step("Detected integer dtype")
        elif pd.api.types.is_float_dtype(series):
            result = 'numeric'
            log_step("Detected float dtype")
        elif pd.api.types.is_bool_dtype(series):
            result = 'integer'
            log_step("Detected bool dtype (treating as integer)")
        else:
            result = 'factor'
            log_step("Detected object/other dtype (treating as factor)")
        
        log_function_exit(func_name, result)
        return result
    except Exception as e:
        log_exception(func_name, e)
        raise


def clean_string_value(val) -> str:
    """
    Clean a string value by removing extra quotes and whitespace.
    Handles values like '"123"' or '""value""' that may have extra quoting.
    Also treats common null indicators (NULL, NA, N/A, etc.) as None.
    """
    func_name = "clean_string_value"
    log_function_entry(func_name, val=val, val_type=type(val).__name__)
    
    try:
        if pd.isna(val):
            log_step("Value is NA/null")
            log_function_exit(func_name, None)
            return None
        
        s = str(val).strip()
        log_variable("s", s, "after strip")
        
        # Remove surrounding quotes (single or double) iteratively
        quote_removal_count = 0
        while len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
                quote_removal_count += 1
            else:
                break
        
        if quote_removal_count > 0:
            log_step(f"Removed {quote_removal_count} layers of quotes", f"result={s}")
        
        # Treat common null indicators as None
        null_indicators = {'null', 'na', 'n/a', 'nan', 'none', '', '.', '-'}
        if s.lower() in null_indicators:
            log_step("Value is a null indicator", f"value={s}")
            log_function_exit(func_name, None)
            return None
        
        result = s if s else None
        log_function_exit(func_name, result)
        return result
    except Exception as e:
        log_exception(func_name, e)
        raise


def is_numeric_convertible(series: pd.Series) -> bool:
    """
    Check if a factor/object column can be converted to numeric.
    Equivalent to R's is.numeric(type.convert(unique(df[,i])))
    """
    func_name = "is_numeric_convertible"
    log_function_entry(func_name, series_name=series.name, dtype=str(series.dtype))
    
    try:
        unique_vals = series.dropna().unique()
        log_variable("unique_vals_count", len(unique_vals))
        
        if len(unique_vals) == 0:
            log_step("No unique values found")
            log_function_exit(func_name, False)
            return False
        
        # Clean values first (remove extra quotes)
        cleaned_vals = [clean_string_value(v) for v in unique_vals]
        cleaned_vals = [v for v in cleaned_vals if v is not None and v != '']
        log_variable("cleaned_vals_count", len(cleaned_vals))
        
        if len(cleaned_vals) == 0:
            log_step("No cleaned values remaining")
            log_function_exit(func_name, False)
            return False
        
        # Try to convert all cleaned values to numeric
        log_step("Testing numeric conversion", f"sample values: {cleaned_vals[:3]}")
        for i, val in enumerate(cleaned_vals):
            pd.to_numeric(val)
            if i < 3:
                log_step(f"Value '{val}' converted successfully")
        
        log_function_exit(func_name, True)
        return True
    except (ValueError, TypeError) as e:
        log_step(f"Conversion failed: {str(e)}")
        log_function_exit(func_name, False)
        return False
    except Exception as e:
        log_exception(func_name, e)
        raise


def is_integer_values(series: pd.Series) -> bool:
    """
    Check if all numeric values in a series are integers (no decimal part).
    """
    func_name = "is_integer_values"
    log_function_entry(func_name, series_name=series.name)
    
    try:
        cleaned = series.dropna().apply(lambda x: clean_string_value(x))
        cleaned = cleaned[cleaned.notna() & (cleaned != '')]
        log_variable("cleaned_count", len(cleaned))
        
        if len(cleaned) == 0:
            log_step("No valid cleaned values")
            log_function_exit(func_name, False)
            return False
        
        numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
        log_variable("numeric_vals_count", len(numeric_vals))
        
        if len(numeric_vals) == 0:
            log_step("No numeric values after conversion")
            log_function_exit(func_name, False)
            return False
        
        # Check if all values are integers (no decimal part)
        is_int = (numeric_vals == numeric_vals.round()).all()
        log_step("Integer check", f"all values are integers: {is_int}")
        log_variable("sample_values", numeric_vals.head(5).tolist())
        
        log_function_exit(func_name, is_int)
        return is_int
    except Exception as e:
        log_exception(func_name, e)
        log_function_exit(func_name, False, success=False)
        return False


def get_top_samples(series: pd.Series, n: int = 5) -> str:
    """
    Get top n unique samples from a series as a comma-separated string.
    Cleans string values to remove extra quotes.
    """
    func_name = "get_top_samples"
    log_function_entry(func_name, series_name=series.name, n=n)
    
    try:
        unique_vals = series.dropna().unique()
        log_variable("unique_vals_count", len(unique_vals))
        
        if len(unique_vals) <= n:
            samples = unique_vals
        else:
            samples = unique_vals[:n]
        
        log_variable("samples_selected", len(samples))
        
        # Clean each sample value to remove extra quotes
        cleaned_samples = []
        for v in samples:
            cleaned = clean_string_value(v)
            if cleaned is not None:
                cleaned_samples.append(cleaned)
            else:
                cleaned_samples.append(str(v))
        
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
    """
    func_name = "analyze_variable"
    log_function_entry(func_name, col_name=col_name, dv=dv, df_shape=df.shape)
    
    try:
        series = df[col_name]
        col_class = get_column_class(series)
        cardinality = series.nunique()
        null_qty = series.isna().sum()
        
        log_variable("col_class", col_class)
        log_variable("cardinality", cardinality)
        log_variable("null_qty", null_qty)
        
        # Initialize metadata
        meta = {
            'VariableName': col_name,
            'Include': True,
            'Role': 'dependent' if col_name == dv else 'independent',
            'Usage': col_class,
            'UsageOriginal': col_class,
            'UsageProposed': "don't",
            'NullQty': null_qty,
            'min': 69,
            'max': 420,
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
        
        # Handle integer type
        if col_class == 'integer':
            log_step("Processing integer type")
            numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
            meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
            meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
            log_variable("min", meta['min'])
            log_variable("max", meta['max'])
            
            if cardinality < 21:
                meta['UsageOriginal'] = 'discrete'
                meta['UsageProposed'] = 'discrete'
                meta['Usage'] = 'discrete'
                log_step("Low cardinality - set to discrete", f"cardinality={cardinality}")
            else:
                meta['UsageOriginal'] = 'continuous'
                meta['UsageProposed'] = 'continuous'
                meta['Usage'] = 'continuous'
                log_step("High cardinality - set to continuous", f"cardinality={cardinality}")
            
            if cardinality > 10:
                meta['DefaultBins'] = 10
            else:
                meta['DefaultBins'] = cardinality
            log_variable("DefaultBins", meta['DefaultBins'])
        
        # Handle factor (object/string) type
        elif col_class == 'factor':
            log_step("Processing factor type")
            meta['UsageOriginal'] = 'nominal'
            meta['min'] = 0
            meta['max'] = 0
            meta['Usage'] = 'nominal'
            
            # Check if values can be converted to numeric
            is_convertible = is_numeric_convertible(series)
            log_variable("is_convertible", is_convertible)
            
            # Check if all numeric values are integers (for discrete detection)
            has_integer_values = is_integer_values(series) if is_convertible else False
            log_variable("has_integer_values", has_integer_values)
            
            if not is_convertible:
                meta['UsageProposed'] = 'nominal'
                log_step("Not convertible - keeping nominal")
            else:
                # Calculate min/max for numeric-convertible columns
                try:
                    cleaned = series.dropna().apply(lambda x: clean_string_value(x))
                    numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        meta['min'] = float(numeric_vals.min())
                        meta['max'] = float(numeric_vals.max())
                        log_variable("min", meta['min'])
                        log_variable("max", meta['max'])
                except Exception as e:
                    log_step(f"Failed to calculate min/max: {str(e)}")
                
                if cardinality < 20:
                    log_step("Low cardinality factor", f"cardinality={cardinality}")
                    if has_integer_values:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                    else:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                    
                    if cardinality < 10:
                        meta['DefaultBins'] = cardinality
                else:
                    log_step("High cardinality factor", f"cardinality={cardinality}")
                    meta['DefaultBins'] = 10
                    if has_integer_values:
                        if cardinality > 50:
                            meta['UsageProposed'] = 'continuous'
                            meta['Usage'] = 'continuous'
                        else:
                            meta['UsageProposed'] = 'discrete'
                            meta['Usage'] = 'discrete'
                    else:
                        meta['UsageProposed'] = 'continuous'
                        meta['Usage'] = 'continuous'
        
        # Handle numeric (float) type
        elif col_class == 'numeric':
            log_step("Processing numeric type")
            numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
            meta['UsageOriginal'] = 'continuous'
            meta['UsageProposed'] = 'continuous'
            meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
            meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
            meta['Usage'] = 'continuous'
            meta['DefaultBins'] = 10
            log_variable("min", meta['min'])
            log_variable("max", meta['max'])
        
        # Special handling for dependent variable
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
    """
    func_name = "analyze_all_variables"
    log_function_entry(func_name, df_shape=df.shape, dv=dv, columns=list(df.columns))
    
    try:
        metadata_list = []
        total_cols = len(df.columns)
        
        for i, col in enumerate(df.columns):
            log_step(f"Analyzing column {i+1}/{total_cols}", f"column={col}")
            meta = analyze_variable(df, col, dv)
            metadata_list.append(meta)
        
        df_var = pd.DataFrame(metadata_list)
        log_step("Created metadata DataFrame", f"shape={df_var.shape}")
        
        # Ensure correct column order
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
    """
    func_name = "apply_type_conversions"
    log_function_entry(func_name, df_shape=df.shape, df_out_shape=df_out.shape)
    
    try:
        result_df = df.copy()
        columns_to_remove = []
        conversion_count = 0
        
        for idx, row in df_out.iterrows():
            var_name = row['VariableName']
            include = row['Include']
            original_type = row['UsageOriginal']
            target_type = row['Usage']
            
            log_step(f"Processing variable", f"name={var_name}, include={include}, {original_type}->{target_type}")
            
            if var_name not in result_df.columns:
                log_step(f"Variable not found in DataFrame", var_name)
                continue
            
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
                
                if target_type == 'nominal':
                    log_step("Applying nominal conversion")
                    result_df[var_name] = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    
                elif target_type == 'continuous':
                    log_step("Applying continuous conversion")
                    cleaned = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    result_df[var_name] = pd.to_numeric(cleaned, errors='coerce').astype('Float64')
                    
                elif target_type == 'discrete':
                    log_step("Applying discrete conversion")
                    cleaned = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    numeric_vals = pd.to_numeric(cleaned, errors='coerce')
                    result_df[var_name] = numeric_vals.round().astype('Int32')
                    
                elif target_type == 'ordinal':
                    log_step("Applying ordinal conversion")
                    result_df[var_name] = result_df[var_name].apply(
                        lambda x: clean_string_value(x) if pd.notna(x) else None
                    )
                    
                elif target_type == 'no binning':
                    log_step("No binning - keeping as-is")
                    pass
            else:
                # Even if type is the same, clean up string values that might have extra quotes
                if original_type in ['nominal', 'factor']:
                    sample_val = result_df[var_name].dropna().iloc[0] if len(result_df[var_name].dropna()) > 0 else None
                    if sample_val is not None and isinstance(sample_val, str):
                        if sample_val.startswith('"') or sample_val.startswith("'"):
                            log_step("Cleaning quoted string values", var_name)
                            result_df[var_name] = result_df[var_name].apply(
                                lambda x: clean_string_value(x) if pd.notna(x) else None
                            )
                print(f"Variable {var_name} not changed")
        
        # Remove columns that are not included
        if columns_to_remove:
            log_step(f"Removing {len(columns_to_remove)} excluded columns")
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

def create_attribute_editor_app(df: pd.DataFrame, initial_dv: Optional[str] = None):
    """Create the Attribute Editor Shiny application."""
    func_name = "create_attribute_editor_app"
    log_function_entry(func_name, df_shape=df.shape, initial_dv=initial_dv)
    
    try:
        app_results = {
            'df_var': None,
            'completed': False
        }
        
        # Column choices
        column_choices = list(df.columns)
        log_variable("column_choices_count", len(column_choices))
        
        # Usage choices for dropdown
        usage_choices = ['continuous', 'nominal', 'ordinal', 'discrete', 'no binning']
        role_choices = ['dependent', 'independent']
        intervals_choices = ['', 'static']
        break_apart_choices = ['yes', 'no']
        missing_choices = ['use', 'ignore', 'float']
        ordered_display_choices = ['range', 'present']
        
        log_step("Building UI components")
        
        # Determine title based on debug mode
        title_text = "Attribute Editor (DEBUG MODE)" if DEBUG_ENABLED else "Attribute Editor"
        
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
                """)
            ),
            
            ui.h4(title_text),
            
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
        
        log_step("UI components built")
        
        def server(input: Inputs, output: Outputs, session: Session):
            if DEBUG_ENABLED:
                logger.debug("SERVER: Initializing server function")
            
            # Reactive values
            df_var_rv = reactive.Value(None)
            
            @reactive.Effect
            def init_table():
                """Initialize the variable table on startup."""
                if DEBUG_ENABLED:
                    logger.debug("SERVER: init_table() called")
                dv = input.dv()
                if DEBUG_ENABLED:
                    logger.debug(f"SERVER: Selected DV = {dv}")
                df_var = analyze_all_variables(df, dv)
                df_var_rv.set(df_var)
                if DEBUG_ENABLED:
                    logger.debug("SERVER: Variable table initialized")
            
            @reactive.Effect
            @reactive.event(input.dv)
            def update_dv():
                """Update roles when dependent variable changes."""
                if DEBUG_ENABLED:
                    logger.debug("SERVER: update_dv() called")
                dv = input.dv()
                current_df = df_var_rv.get()
                
                if current_df is not None:
                    current_df = current_df.copy()
                    current_df['Role'] = current_df['VariableName'].apply(
                        lambda x: 'dependent' if x == dv else 'independent'
                    )
                    current_df.loc[current_df['VariableName'] == dv, 'MissingValues'] = 'float'
                    current_df.loc[current_df['VariableName'] == dv, 'OrderedDisplay'] = 'range'
                    df_var_rv.set(current_df)
                    if DEBUG_ENABLED:
                        logger.debug(f"SERVER: Updated DV to {dv}")
            
            @reactive.Effect
            @reactive.event(input.reroll_btn)
            def reroll_samples():
                """Reroll sample values for variables with high cardinality."""
                if DEBUG_ENABLED:
                    logger.debug("SERVER: reroll_samples() called")
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
                    if DEBUG_ENABLED:
                        logger.debug(f"SERVER: Rerolled samples for {reroll_count} variables")
            
            @output
            @render.data_frame
            def var_table():
                """Render the variable attributes table."""
                if DEBUG_ENABLED:
                    logger.debug("SERVER: var_table() render called")
                current_df = df_var_rv.get()
                if current_df is None:
                    if DEBUG_ENABLED:
                        logger.debug("SERVER: No data to render")
                    return render.DataGrid(pd.DataFrame())
                
                display_cols = [
                    'VariableName', 'Include', 'Role', 'Usage', 'UsageOriginal',
                    'UsageProposed', 'NullQty', 'min', 'max', 'Cardinality',
                    'Samples', 'DefaultBins', 'IntervalsType', 'BreakApart',
                    'MissingValues', 'OrderedDisplay', 'PValue'
                ]
                
                display_df = current_df[display_cols].copy()
                if DEBUG_ENABLED:
                    logger.debug(f"SERVER: Rendering table with {len(display_df)} rows")
                
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
                    if DEBUG_ENABLED:
                        logger.debug(f"SERVER: Cell edit - row={edit_info['row']}, col={edit_info['col']}, value={edit_info['value']}")
                    current_df = df_var_rv.get()
                    if current_df is not None:
                        current_df = current_df.copy()
                        row_idx = edit_info['row']
                        col_idx = edit_info['col']
                        new_value = edit_info['value']
                        
                        col_name = current_df.columns[col_idx]
                        
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
                        if DEBUG_ENABLED:
                            logger.debug(f"SERVER: Updated cell [{row_idx}, {col_name}] = {new_value}")
            
            @reactive.Effect
            @reactive.event(input.submit_btn)
            async def submit():
                """Handle submit button click."""
                if DEBUG_ENABLED:
                    logger.debug("SERVER: submit() called")
                current_df = df_var_rv.get()
                if current_df is not None:
                    app_results['df_var'] = current_df.copy()
                    app_results['completed'] = True
                    if DEBUG_ENABLED:
                        logger.debug("SERVER: Results saved, closing session")
                
                await session.close()
        
        log_step("Server function defined")
        
        app = App(app_ui, server)
        app.results = app_results
        
        log_function_exit(func_name, "App created successfully")
        return app
    except Exception as e:
        log_exception(func_name, e)
        raise


def run_attribute_editor(df: pd.DataFrame, initial_dv: Optional[str] = None, port: int = 8051):
    """Run the Attribute Editor application and return results."""
    func_name = "run_attribute_editor"
    log_function_entry(func_name, df_shape=df.shape, initial_dv=initial_dv, port=port)
    
    try:
        app = create_attribute_editor_app(df, initial_dv)
        log_step("Starting Shiny app", f"port={port}")
        app.run(port=port, launch_browser=True)
        log_step("Shiny app closed")
        log_function_exit(func_name, f"completed={app.results['completed']}")
        return app.results
    except Exception as e:
        log_exception(func_name, e)
        raise


# =============================================================================
# Main Execution
# =============================================================================

if DEBUG_ENABLED:
    logger.info("="*80)
    logger.info("ATTRIBUTE EDITOR TOGGLE DEBUG VERSION - MAIN EXECUTION STARTING")
    logger.info(f"DEBUG_ENABLED = {DEBUG_ENABLED}")
    logger.info("="*80)

# =============================================================================
# Read Input Data
# =============================================================================
log_step("Reading input data from KNIME")
df = knio.input_tables[0].to_pandas()
log_dataframe_info(df, "Input Data")

# =============================================================================
# Preprocess: Replace common null indicator strings with actual NA
# =============================================================================
log_step("Preprocessing null indicators")
null_indicators = ['NULL', 'null', 'NA', 'na', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '.', '-', '']

null_replacement_count = 0
for col in df.columns:
    if df[col].dtype == 'object':
        before_null_count = df[col].isna().sum()
        df[col] = df[col].apply(
            lambda x: pd.NA if (isinstance(x, str) and x.strip() in null_indicators) else x
        )
        after_null_count = df[col].isna().sum()
        if after_null_count > before_null_count:
            null_replacement_count += (after_null_count - before_null_count)
            if DEBUG_ENABLED:
                logger.debug(f"Column {col}: replaced {after_null_count - before_null_count} null indicators")

if DEBUG_ENABLED:
    logger.debug(f"Total null indicators replaced: {null_replacement_count}")

df_temp = df.copy()

# =============================================================================
# Check for Flow Variables
# =============================================================================
log_step("Checking flow variables")
contains_dv = False
is_var_override = False
dv = None

# Attempt to get DependentVariable
try:
    dv = knio.flow_variables.get("DependentVariable", None)
    if DEBUG_ENABLED:
        logger.debug(f"DependentVariable flow variable: {dv}")
except Exception as e:
    if DEBUG_ENABLED:
        logger.debug(f"Failed to get DependentVariable: {str(e)}")

# Attempt to get VarOverride
try:
    var_override = knio.flow_variables.get("VarOverride", None)
    if DEBUG_ENABLED:
        logger.debug(f"VarOverride flow variable: {var_override}")
except Exception as e:
    if DEBUG_ENABLED:
        logger.debug(f"Failed to get VarOverride: {str(e)}")
    var_override = None

# Validate DependentVariable
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True
        if DEBUG_ENABLED:
            logger.debug(f"Valid DependentVariable found: {dv}")
    else:
        if DEBUG_ENABLED:
            logger.warning(f"DependentVariable '{dv}' not found in columns")

# Validate VarOverride
if var_override is not None and isinstance(var_override, int) and var_override == 1:
    is_var_override = True
    if DEBUG_ENABLED:
        logger.debug("VarOverride is enabled (=1)")

log_variable("contains_dv", contains_dv)
log_variable("is_var_override", is_var_override)

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv and not is_var_override:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    if DEBUG_ENABLED:
        logger.info("="*60)
        logger.info("RUNNING IN HEADLESS MODE")
        logger.info("="*60)
    print(f"Running in headless mode with DV: {dv}")
    
    df_out = analyze_all_variables(df, dv)
    
    print(f"Analyzed {len(df_out)} variables")
    if DEBUG_ENABLED:
        logger.info(f"Headless analysis complete: {len(df_out)} variables")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    if DEBUG_ENABLED:
        logger.info("="*60)
        logger.info("RUNNING IN INTERACTIVE MODE")
        logger.info("="*60)
    print("Running in interactive mode - launching Shiny UI...")
    
    # Get initial DV from flow variable if available
    initial_dv = dv if dv and dv in df.columns else None
    if DEBUG_ENABLED:
        logger.debug(f"Initial DV for UI: {initial_dv}")
    
    results = run_attribute_editor(df, initial_dv=initial_dv)
    
    if results['completed']:
        df_out = results['df_var']
        print("Interactive session completed successfully")
        if DEBUG_ENABLED:
            logger.info("Interactive session completed successfully")
    else:
        print("Interactive session cancelled - generating default metadata")
        if DEBUG_ENABLED:
            logger.warning("Interactive session cancelled - generating default metadata")
        df_out = analyze_all_variables(df, None)

# =============================================================================
# Apply Type Conversions to Original Data
# =============================================================================
log_step("Applying type conversions")
df_converted = apply_type_conversions(df, df_out)

print(f"Applied type conversions. Output has {len(df_converted.columns)} columns.")
if DEBUG_ENABLED:
    logger.info(f"Type conversions applied. Output columns: {len(df_converted.columns)}")

# =============================================================================
# Output Tables
# =============================================================================
log_step("Preparing output tables")

# Ensure correct data types for metadata output
df_out['Include'] = df_out['Include'].astype(bool)
df_out['NullQty'] = df_out['NullQty'].astype('Int32')
df_out['Cardinality'] = df_out['Cardinality'].astype('Int32')
df_out['DefaultBins'] = df_out['DefaultBins'].astype('Int32')
df_out['min'] = df_out['min'].astype('Float64')
df_out['max'] = df_out['max'].astype('Float64')
df_out['PValue'] = df_out['PValue'].astype('Float64')

log_dataframe_info(df_out, "Metadata Output")
log_dataframe_info(df_converted, "Converted Data Output")

# Output 1: Variable metadata DataFrame
knio.output_tables[0] = knio.Table.from_pandas(df_out)
if DEBUG_ENABLED:
    logger.debug("Output table 0 (metadata) written")

# Output 2: Original data with type conversions applied and excluded columns removed
knio.output_tables[1] = knio.Table.from_pandas(df_converted)
if DEBUG_ENABLED:
    logger.debug("Output table 1 (converted data) written")

print("Attribute Editor completed successfully")
if DEBUG_ENABLED:
    logger.info("="*80)
    logger.info("ATTRIBUTE EDITOR TOGGLE DEBUG VERSION - EXECUTION COMPLETE")
    logger.info("="*80)

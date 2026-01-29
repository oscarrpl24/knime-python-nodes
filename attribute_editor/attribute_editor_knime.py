# =============================================================================
# Attribute Editor for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's Attribute Editor functionality
# Compatible with KNIME 5.9, Python 3.9
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
# Release Date: 2026-01-26
# Version: 1.3 - Fixed handling of "NULL" text as missing values for numeric columns
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# Install/Import Dependencies
# =============================================================================

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui


# =============================================================================
# Helper Functions
# =============================================================================

def get_column_class(series: pd.Series) -> str:
    """
    Determine the R-equivalent class of a pandas Series.
    Returns: 'integer', 'numeric', or 'factor'
    """
    if pd.api.types.is_integer_dtype(series):
        return 'integer'
    elif pd.api.types.is_float_dtype(series):
        return 'numeric'
    elif pd.api.types.is_bool_dtype(series):
        return 'integer'
    else:
        return 'factor'


def clean_string_value(val) -> str:
    """
    Clean a string value by removing extra quotes and whitespace.
    Handles values like '"123"' or '""value""' that may have extra quoting.
    Also treats common null indicators (NULL, NA, N/A, etc.) as None.
    """
    if pd.isna(val):
        return None
    
    s = str(val).strip()
    
    # Remove surrounding quotes (single or double) iteratively
    while len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        else:
            break
    
    # Treat common null indicators as None
    null_indicators = {'null', 'na', 'n/a', 'nan', 'none', '', '.', '-'}
    if s.lower() in null_indicators:
        return None
    
    return s if s else None


def is_numeric_convertible(series: pd.Series) -> bool:
    """
    Check if a factor/object column can be converted to numeric.
    Equivalent to R's is.numeric(type.convert(unique(df[,i])))
    """
    try:
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 0:
            return False
        
        # Clean values first (remove extra quotes)
        cleaned_vals = [clean_string_value(v) for v in unique_vals]
        cleaned_vals = [v for v in cleaned_vals if v is not None and v != '']
        
        if len(cleaned_vals) == 0:
            return False
        
        # Try to convert all cleaned values to numeric
        for val in cleaned_vals:
            pd.to_numeric(val)
        
        return True
    except (ValueError, TypeError):
        return False


def is_integer_values(series: pd.Series) -> bool:
    """
    Check if all numeric values in a series are integers (no decimal part).
    """
    try:
        cleaned = series.dropna().apply(lambda x: clean_string_value(x))
        cleaned = cleaned[cleaned.notna() & (cleaned != '')]
        
        if len(cleaned) == 0:
            return False
        
        numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
        
        if len(numeric_vals) == 0:
            return False
        
        # Check if all values are integers (no decimal part)
        return (numeric_vals == numeric_vals.round()).all()
    except:
        return False


def get_top_samples(series: pd.Series, n: int = 5) -> str:
    """
    Get top n unique samples from a series as a comma-separated string.
    Cleans string values to remove extra quotes.
    """
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= n:
        samples = unique_vals
    else:
        samples = unique_vals[:n]
    
    # Clean each sample value to remove extra quotes
    cleaned_samples = []
    for v in samples:
        cleaned = clean_string_value(v)
        if cleaned is not None:
            cleaned_samples.append(cleaned)
        else:
            cleaned_samples.append(str(v))
    
    return ", ".join(cleaned_samples)


def analyze_variable(df: pd.DataFrame, col_name: str, dv: Optional[str] = None) -> dict:
    """
    Analyze a single variable and return its metadata.
    """
    series = df[col_name]
    col_class = get_column_class(series)
    cardinality = series.nunique()
    null_qty = series.isna().sum()
    
    # Initialize metadata
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
    
    # Handle integer type
    if col_class == 'integer':
        numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
        meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
        meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
        
        if cardinality < 21:
            meta['UsageOriginal'] = 'discrete'
            meta['UsageProposed'] = 'discrete'
            meta['Usage'] = 'discrete'
        else:
            meta['UsageOriginal'] = 'continuous'
            meta['UsageProposed'] = 'continuous'
            meta['Usage'] = 'continuous'
        
        if cardinality > 10:
            meta['DefaultBins'] = 10
        else:
            meta['DefaultBins'] = cardinality
    
    # Handle factor (object/string) type
    elif col_class == 'factor':
        meta['UsageOriginal'] = 'nominal'
        meta['min'] = 0
        meta['max'] = 0
        meta['Usage'] = 'nominal'
        
        # Check if values can be converted to numeric
        is_convertible = is_numeric_convertible(series)
        
        # Check if all numeric values are integers (for discrete detection)
        has_integer_values = is_integer_values(series) if is_convertible else False
        
        if not is_convertible:
            meta['UsageProposed'] = 'nominal'
        else:
            # Calculate min/max for numeric-convertible columns
            try:
                cleaned = series.dropna().apply(lambda x: clean_string_value(x))
                numeric_vals = pd.to_numeric(cleaned, errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    meta['min'] = float(numeric_vals.min())
                    meta['max'] = float(numeric_vals.max())
            except:
                pass
            
            if cardinality < 20:
                # Low cardinality - likely discrete
                if has_integer_values:
                    meta['UsageProposed'] = 'discrete'
                    meta['Usage'] = 'discrete'
                else:
                    # Float values but low cardinality - still treat as discrete
                    meta['UsageProposed'] = 'discrete'
                    meta['Usage'] = 'discrete'
                
                if cardinality < 10:
                    meta['DefaultBins'] = cardinality
            else:
                meta['DefaultBins'] = 10
                # High cardinality - check if integer or float
                if has_integer_values:
                    # Integer values with high cardinality - could be discrete or continuous
                    # Use continuous if cardinality is very high, otherwise discrete
                    if cardinality > 50:
                        meta['UsageProposed'] = 'continuous'
                        meta['Usage'] = 'continuous'
                    else:
                        meta['UsageProposed'] = 'discrete'
                        meta['Usage'] = 'discrete'
                else:
                    # Float values - continuous
                    meta['UsageProposed'] = 'continuous'
                    meta['Usage'] = 'continuous'
    
    # Handle numeric (float) type
    elif col_class == 'numeric':
        numeric_vals = pd.to_numeric(series.dropna(), errors='coerce')
        meta['UsageOriginal'] = 'continuous'
        meta['UsageProposed'] = 'continuous'
        meta['min'] = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0
        meta['max'] = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0
        meta['Usage'] = 'continuous'
        meta['DefaultBins'] = 10
    
    # Special handling for dependent variable
    if col_name == dv:
        meta['Role'] = 'dependent'
        meta['MissingValues'] = 'float'
        meta['OrderedDisplay'] = 'range'
    
    return meta


def analyze_all_variables(df: pd.DataFrame, dv: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze all variables in a DataFrame and return metadata DataFrame.
    """
    metadata_list = []
    for col in df.columns:
        meta = analyze_variable(df, col, dv)
        metadata_list.append(meta)
    
    df_var = pd.DataFrame(metadata_list)
    
    # Ensure correct column order
    column_order = [
        'VariableName', 'Include', 'Role', 'Usage', 'UsageOriginal', 
        'UsageProposed', 'NullQty', 'min', 'max', 'Cardinality', 
        'Samples', 'DefaultBins', 'IntervalsType', 'BreakApart',
        'MissingValues', 'OrderedDisplay', 'PValue'
    ]
    
    return df_var[column_order]


def apply_type_conversions(df: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Apply type conversions based on the metadata DataFrame.
    
    This function:
    1. Removes columns where Include == False
    2. Converts column types based on Usage vs UsageOriginal
       - nominal → string
       - continuous → float
       - discrete → integer
    
    Returns a new DataFrame with conversions applied.
    """
    result_df = df.copy()
    columns_to_remove = []
    
    for idx, row in df_out.iterrows():
        var_name = row['VariableName']
        include = row['Include']
        original_type = row['UsageOriginal']
        target_type = row['Usage']
        
        if var_name not in result_df.columns:
            continue
        
        if not include:
            print(f"Variable {var_name} will not be included")
            columns_to_remove.append(var_name)
            continue
        
        # Check if type conversion is needed
        if original_type != target_type:
            print(f"Variable {var_name} from {original_type} to {target_type}")
            
            if target_type == 'nominal':
                # Convert to clean string (remove extra quotes)
                result_df[var_name] = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                
            elif target_type == 'continuous':
                # Clean string values first, then convert to float
                cleaned = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                result_df[var_name] = pd.to_numeric(cleaned, errors='coerce').astype('Float64')
                
            elif target_type == 'discrete':
                # Clean string values first, then convert to integer
                cleaned = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                numeric_vals = pd.to_numeric(cleaned, errors='coerce')
                result_df[var_name] = numeric_vals.round().astype('Int32')
                
            elif target_type == 'ordinal':
                # Convert to clean string (remove extra quotes)
                result_df[var_name] = result_df[var_name].apply(
                    lambda x: clean_string_value(x) if pd.notna(x) else None
                )
                
            elif target_type == 'no binning':
                # Keep as-is, no conversion needed
                pass
        else:
            # Even if type is the same, clean up string values that might have extra quotes
            if original_type in ['nominal', 'factor']:
                # Check if values have extra quotes and clean them
                sample_val = result_df[var_name].dropna().iloc[0] if len(result_df[var_name].dropna()) > 0 else None
                if sample_val is not None and isinstance(sample_val, str):
                    if sample_val.startswith('"') or sample_val.startswith("'"):
                        result_df[var_name] = result_df[var_name].apply(
                            lambda x: clean_string_value(x) if pd.notna(x) else None
                        )
            print(f"Variable {var_name} not changed")
    
    # Remove columns that are not included
    if columns_to_remove:
        result_df = result_df.drop(columns=columns_to_remove)
    
    return result_df


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_attribute_editor_app(df: pd.DataFrame, initial_dv: Optional[str] = None):
    """Create the Attribute Editor Shiny application."""
    
    app_results = {
        'df_var': None,
        'completed': False
    }
    
    # Column choices
    column_choices = list(df.columns)
    
    # Usage choices for dropdown
    usage_choices = ['continuous', 'nominal', 'ordinal', 'discrete', 'no binning']
    role_choices = ['dependent', 'independent']
    intervals_choices = ['', 'static']
    break_apart_choices = ['yes', 'no']
    missing_choices = ['use', 'ignore', 'float']
    ordered_display_choices = ['range', 'present']
    
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
        
        ui.h4("Attribute Editor"),
        
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
    
    def server(input: Inputs, output: Outputs, session: Session):
        # Reactive values
        df_var_rv = reactive.Value(None)
        
        @reactive.Effect
        def init_table():
            """Initialize the variable table on startup."""
            dv = input.dv()
            df_var = analyze_all_variables(df, dv)
            df_var_rv.set(df_var)
        
        @reactive.Effect
        @reactive.event(input.dv)
        def update_dv():
            """Update roles when dependent variable changes."""
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
        
        @reactive.Effect
        @reactive.event(input.reroll_btn)
        def reroll_samples():
            """Reroll sample values for variables with high cardinality."""
            current_df = df_var_rv.get()
            if current_df is not None:
                current_df = current_df.copy()
                for idx, row in current_df.iterrows():
                    if row['Cardinality'] > 5:
                        var_name = row['VariableName']
                        unique_vals = df[var_name].dropna().unique()
                        if len(unique_vals) > 5:
                            sample_indices = np.random.choice(len(unique_vals), min(5, len(unique_vals)), replace=False)
                            samples = unique_vals[sample_indices]
                            current_df.loc[idx, 'Samples'] = ", ".join(str(v) for v in samples)
                df_var_rv.set(current_df)
        
        @output
        @render.data_frame
        def var_table():
            """Render the variable attributes table."""
            current_df = df_var_rv.get()
            if current_df is None:
                return render.DataGrid(pd.DataFrame())
            
            # Select columns for display
            display_cols = [
                'VariableName', 'Include', 'Role', 'Usage', 'UsageOriginal',
                'UsageProposed', 'NullQty', 'min', 'max', 'Cardinality',
                'Samples', 'DefaultBins', 'IntervalsType', 'BreakApart',
                'MissingValues', 'OrderedDisplay', 'PValue'
            ]
            
            display_df = current_df[display_cols].copy()
            
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
        
        @reactive.Effect
        @reactive.event(input.submit_btn)
        async def submit():
            """Handle submit button click."""
            current_df = df_var_rv.get()
            if current_df is not None:
                app_results['df_var'] = current_df.copy()
                app_results['completed'] = True
            
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def run_attribute_editor(df: pd.DataFrame, initial_dv: Optional[str] = None, port: int = 8051):
    """Run the Attribute Editor application and return results."""
    app = create_attribute_editor_app(df, initial_dv)
    app.run(port=port, launch_browser=True)
    return app.results


# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Read Input Data
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Preprocess: Replace common null indicator strings with actual NA
# =============================================================================
# KNIME may pass "NULL", "NA", etc. as string values instead of actual missing values
# Replace these with pd.NA for proper handling
null_indicators = ['NULL', 'null', 'NA', 'na', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '.', '-', '']

for col in df.columns:
    if df[col].dtype == 'object':  # Only process string columns
        # Replace null indicator strings with pd.NA
        df[col] = df[col].apply(
            lambda x: pd.NA if (isinstance(x, str) and x.strip() in null_indicators) else x
        )

df_temp = df.copy()

# =============================================================================
# Check for Flow Variables
# =============================================================================
contains_dv = False
is_var_override = False
dv = None

# Attempt to get DependentVariable
try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

# Attempt to get VarOverride
try:
    var_override = knio.flow_variables.get("VarOverride", None)
except:
    var_override = None

# Validate DependentVariable
if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True

# Validate VarOverride
if var_override is not None and isinstance(var_override, int) and var_override == 1:
    is_var_override = True

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv and not is_var_override:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    print(f"Running in headless mode with DV: {dv}")
    
    df_out = analyze_all_variables(df, dv)
    
    print(f"Analyzed {len(df_out)} variables")

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    print("Running in interactive mode - launching Shiny UI...")
    
    # Get initial DV from flow variable if available
    initial_dv = dv if dv and dv in df.columns else None
    
    results = run_attribute_editor(df, initial_dv=initial_dv)
    
    if results['completed']:
        df_out = results['df_var']
        print("Interactive session completed successfully")
    else:
        print("Interactive session cancelled - generating default metadata")
        df_out = analyze_all_variables(df, None)

# =============================================================================
# Apply Type Conversions to Original Data
# =============================================================================

# Apply type conversions based on metadata
df_converted = apply_type_conversions(df, df_out)

print(f"Applied type conversions. Output has {len(df_converted.columns)} columns.")

# =============================================================================
# Output Tables
# =============================================================================

# Ensure correct data types for metadata output
df_out['Include'] = df_out['Include'].astype(bool)
df_out['NullQty'] = df_out['NullQty'].astype('Int32')
df_out['Cardinality'] = df_out['Cardinality'].astype('Int32')
df_out['DefaultBins'] = df_out['DefaultBins'].astype('Int32')
df_out['min'] = df_out['min'].astype('Float64')
df_out['max'] = df_out['max'].astype('Float64')
df_out['PValue'] = df_out['PValue'].astype('Float64')

# Output 1: Variable metadata DataFrame
knio.output_tables[0] = knio.Table.from_pandas(df_out)

# Output 2: Original data with type conversions applied and excluded columns removed
knio.output_tables[1] = knio.Table.from_pandas(df_converted)

print("Attribute Editor completed successfully")


# =============================================================================
# Scorecard Generator for KNIME Python Script Node
# =============================================================================
# Python implementation matching R's scorecard creation functionality
# with Shiny UI for parameter configuration
# Compatible with KNIME 5.9, Python 3.9
#
# This script has two modes:
# 1. Interactive (Shiny UI) - When no flow variables are provided
# 2. Headless - When Points, Odds, and PDO are provided via flow variables
#
# Inputs:
# 1. Coefficients table from Logistic Regression node (Output 2)
#    - Row ID = variable name (e.g., "(Intercept)", "WOE_Age")
#    - Column "model$coefficients" = coefficient value
# 2. Bins table from WOE Editor node (Output 4)
#    - var, bin, binValue, woe columns required
#
# Outputs:
# 1. Scorecard table - all bins with points (var, bin, woe, points columns)
#
# Flow Variables (for headless mode):
# - Points (int, default 600): Base score at target odds
# - Odds (int, default 20): Target odds ratio (1:Odds, e.g., 20 means 1:19)
# - PDO (int, default 50): Points to Double the Odds
#
# Scorecard Formula:
#   b = PDO / log(2)
#   a = Points + b * log(1/(Odds-1))
#   basepoints = a - b * intercept_coefficient
#   bin_points = round(-b * coefficient * woe, digits)
#
# Release Date: 2026-01-19
# Version: 1.0
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# Install/Import Dependencies
# =============================================================================

def install_if_missing(package, import_name=None):
    """Install package if not available."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', package])

install_if_missing('shiny')
install_if_missing('shinywidgets')

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    SHINY_AVAILABLE = True
except ImportError:
    print("WARNING: Shiny not available. Interactive mode disabled.")
    SHINY_AVAILABLE = False


# =============================================================================
# Scorecard Creation Functions
# =============================================================================

def calculate_ab(points0: float = 600, odds0: float = 1/19, pdo: float = 50) -> Tuple[float, float]:
    """
    Calculate scaling parameters a and b for scorecard.
    
    Parameters:
        points0: Base score at target odds
        odds0: Target odds ratio (as decimal, e.g., 1/19 for 1:19)
        pdo: Points to Double the Odds
        
    Returns:
        Tuple of (a, b) scaling parameters
    """
    b = pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    return a, b


def is_interaction_term(var_name: str) -> bool:
    """Check if a variable name represents an interaction term."""
    # Interaction terms have pattern: var1_x_WOE_var2 or WOE_var1_x_WOE_var2
    return '_x_WOE_' in var_name or '_x_' in var_name


def parse_interaction_term(var_name: str) -> Tuple[str, str]:
    """
    Parse an interaction term into its two component variable names.
    
    Input formats:
        - "WOE_var1_x_WOE_var2" -> ("var1", "var2")
        - "var1_x_WOE_var2" -> ("var1", "var2")
        - "var1_x_var2" -> ("var1", "var2")
    
    Returns:
        Tuple of (var1_name, var2_name)
    """
    # Remove leading WOE_ if present
    clean_name = var_name
    if clean_name.startswith('WOE_'):
        clean_name = clean_name[4:]  # Remove 'WOE_'
    
    # Split on '_x_WOE_' first (most specific pattern)
    if '_x_WOE_' in clean_name:
        parts = clean_name.split('_x_WOE_', 1)
        return parts[0], parts[1]
    
    # Fall back to splitting on '_x_'
    if '_x_' in clean_name:
        parts = clean_name.split('_x_', 1)
        var2 = parts[1]
        # Remove WOE_ prefix from var2 if present
        if var2.startswith('WOE_'):
            var2 = var2[4:]
        return parts[0], var2
    
    raise ValueError(f"Cannot parse interaction term: {var_name}")


def create_interaction_bins(
    bins: pd.DataFrame,
    var1: str,
    var2: str,
    interaction_name: str,
    coef: float,
    b: float,
    digits: int = 0
) -> List[Dict]:
    """
    Create scorecard entries for an interaction term.
    
    For interaction WOE_var1_x_WOE_var2:
    - Creates all combinations of var1 bins × var2 bins
    - Interaction WOE = woe1 × woe2
    - Points = round(-b × coefficient × woe1 × woe2)
    
    Parameters:
        bins: DataFrame with binning rules
        var1, var2: Component variable names
        interaction_name: Full interaction term name (for display)
        coef: Coefficient for this interaction
        b: Scaling parameter
        digits: Rounding precision
        
    Returns:
        List of scorecard row dictionaries
    """
    rows = []
    
    # Get bins for each component variable
    var1_bins = bins[bins['var'] == var1].copy()
    var2_bins = bins[bins['var'] == var2].copy()
    
    if var1_bins.empty or var2_bins.empty:
        print(f"WARNING: Cannot create interaction bins for {interaction_name}")
        print(f"  var1 '{var1}' has {len(var1_bins)} bins")
        print(f"  var2 '{var2}' has {len(var2_bins)} bins")
        return rows
    
    # Log combination count (informational only - will process regardless)
    total_combinations = len(var1_bins) * len(var2_bins)
    if total_combinations > 1000:
        print(f"INFO: Interaction {interaction_name} will create {total_combinations:,} combinations")
        print(f"  var1 '{var1}': {len(var1_bins)} bins")
        print(f"  var2 '{var2}': {len(var2_bins)} bins")
        print(f"  This may take some time to process...")
    
    # Create all combinations
    for _, row1 in var1_bins.iterrows():
        woe1 = row1.get('woe', 0)
        if pd.isna(woe1):
            woe1 = 0
        bin1 = row1.get('binValue', row1.get('bin', ''))
        
        for _, row2 in var2_bins.iterrows():
            woe2 = row2.get('woe', 0)
            if pd.isna(woe2):
                woe2 = 0
            bin2 = row2.get('binValue', row2.get('bin', ''))
            
            # Interaction WOE is the product
            interaction_woe = woe1 * woe2
            
            # Calculate points
            points = round(-b * coef * interaction_woe, digits)
            
            # Create combined bin label
            combined_bin = f"{var1}:{bin1} × {var2}:{bin2}"
            
            rows.append({
                'var': interaction_name,
                'bin': combined_bin,
                'binValue': combined_bin,
                'woe': round(interaction_woe, 6),
                'points': points
            })
    
    return rows


def create_scorecard(
    bins: pd.DataFrame,
    coefficients: pd.DataFrame,
    points0: float = 600,
    odds0: float = 1/19,
    pdo: float = 50,
    basepoints_eq0: bool = False,
    digits: int = 0
) -> pd.DataFrame:
    """
    Create a scorecard from binning rules and logistic regression coefficients.
    
    Handles both regular variables and interaction terms.
    For interaction terms (e.g., WOE_var1_x_WOE_var2), creates all combinations
    of the component variables' bins with WOE = woe1 × woe2.
    
    Parameters:
        bins: DataFrame with binning rules (var, bin, binValue, woe columns)
        coefficients: DataFrame with model coefficients (index = var name, column = coefficient)
        points0: Base score at target odds (default 600)
        odds0: Target odds ratio as decimal (default 1/19)
        pdo: Points to Double the Odds (default 50)
        basepoints_eq0: If True, set basepoints to 0 (default False)
        digits: Number of decimal places for rounding points (default 0)
        
    Returns:
        DataFrame with scorecard (var, bin, woe, points columns)
    """
    # Calculate scaling parameters
    a, b = calculate_ab(points0, odds0, pdo)
    
    # Prepare coefficients
    # Handle both cases: coefficient column might be named differently
    coef_col = coefficients.columns[0] if len(coefficients.columns) > 0 else 'coefficients'
    
    # Create coefficient lookup: variable name -> coefficient value
    # Keep full names for interaction detection
    coef_dict = {}
    coef_dict_clean = {}  # With WOE_ prefix stripped for regular vars
    intercept = 0.0
    
    for var_name, row in coefficients.iterrows():
        coef_value = row.iloc[0] if len(row) > 0 else row[coef_col]
        
        if var_name == '(Intercept)' or var_name.lower() == 'intercept':
            intercept = coef_value
        else:
            coef_dict[var_name] = coef_value
            # Also store with WOE_ stripped for regular variable matching
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            coef_dict_clean[clean_var] = coef_value
    
    # Calculate base points
    if basepoints_eq0:
        basepoints = 0
    else:
        basepoints = round(a - b * intercept, digits)
    
    # Create scorecard entries
    scorecard_rows = []
    
    # Add basepoints row
    scorecard_rows.append({
        'var': 'basepoints',
        'bin': None,
        'binValue': None,
        'woe': None,
        'points': basepoints
    })
    
    # Process each variable in bins
    bins_copy = bins.copy()
    
    # Ensure we have the required columns
    if 'var' not in bins_copy.columns:
        raise ValueError("Bins table must have 'var' column")
    if 'woe' not in bins_copy.columns:
        raise ValueError("Bins table must have 'woe' column")
    
    # Separate regular variables from interaction terms
    regular_vars = []
    interaction_vars = []
    
    for var_name in coef_dict.keys():
        if is_interaction_term(var_name):
            interaction_vars.append(var_name)
        else:
            # Strip WOE_ for matching with bins
            clean_var = var_name.replace('WOE_', '') if var_name.startswith('WOE_') else var_name
            regular_vars.append((var_name, clean_var))
    
    # Process regular variables
    for full_var, clean_var in regular_vars:
        if clean_var not in bins_copy['var'].unique():
            print(f"WARNING: Variable '{clean_var}' not found in bins table")
            continue
            
        var_bins = bins_copy[bins_copy['var'] == clean_var].copy()
        coef = coef_dict[full_var]
        
        for _, row in var_bins.iterrows():
            woe = row.get('woe', 0)
            if pd.isna(woe):
                woe = 0
            
            # Calculate points: -b * coefficient * woe
            points = round(-b * coef * woe, digits)
            
            scorecard_rows.append({
                'var': clean_var,
                'bin': row.get('bin', None),
                'binValue': row.get('binValue', None),
                'woe': woe,
                'points': points
            })
    
    # Process interaction terms
    for interaction_name in interaction_vars:
        try:
            var1, var2 = parse_interaction_term(interaction_name)
            coef = coef_dict[interaction_name]
            
            interaction_rows = create_interaction_bins(
                bins=bins_copy,
                var1=var1,
                var2=var2,
                interaction_name=f"{var1}_x_{var2}",
                coef=coef,
                b=b,
                digits=digits
            )
            
            scorecard_rows.extend(interaction_rows)
            print(f"Created {len(interaction_rows)} bins for interaction: {var1} × {var2}")
            
        except Exception as e:
            print(f"ERROR processing interaction '{interaction_name}': {e}")
    
    scorecard_df = pd.DataFrame(scorecard_rows)
    
    # Reorder columns
    col_order = ['var', 'bin', 'binValue', 'woe', 'points']
    col_order = [c for c in col_order if c in scorecard_df.columns]
    scorecard_df = scorecard_df[col_order]
    
    return scorecard_df


def create_scorecard_list(scorecard_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert scorecard DataFrame to list format (dictionary of DataFrames per variable).
    
    Parameters:
        scorecard_df: Scorecard DataFrame from create_scorecard()
        
    Returns:
        Dictionary with variable names as keys and corresponding bin DataFrames as values
    """
    card_list = {}
    
    for var in scorecard_df['var'].unique():
        var_df = scorecard_df[scorecard_df['var'] == var].copy()
        card_list[var] = var_df.reset_index(drop=True)
    
    return card_list


# =============================================================================
# Shiny UI Application
# =============================================================================

def create_scorecard_app(coefficients: pd.DataFrame, bins: pd.DataFrame):
    """Create the Scorecard Generator Shiny application."""
    
    app_results = {
        'scorecard': None,
        'completed': False
    }
    
    app_ui = ui.page_fluid(
        ui.tags.head(
            ui.tags.style("""
                @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
                body { 
                    font-family: 'Source Sans Pro', sans-serif; 
                    background: #f5f7fa;
                    min-height: 100vh;
                    color: #2c3e50;
                }
                .card { 
                    background: #ffffff;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                .card-header {
                    color: #2c3e50;
                    font-weight: 700;
                    font-size: 1.1rem;
                    margin-bottom: 16px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }
                h3 { 
                    color: #2c3e50; 
                    text-align: center; 
                    font-weight: 700;
                    margin-bottom: 24px;
                }
                .btn-primary { 
                    background: #3498db;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 10px 24px;
                    border-radius: 6px;
                }
                .btn-primary:hover {
                    background: #2980b9;
                }
                .btn-success { 
                    background: #27ae60;
                    border: none;
                    color: white;
                    font-weight: 700;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-success:hover {
                    background: #219a52;
                }
                .btn-secondary { 
                    background: #95a5a6;
                    border: none;
                    color: white;
                    font-weight: 600;
                    padding: 12px 32px;
                    border-radius: 6px;
                    font-size: 1.1rem;
                }
                .btn-secondary:hover {
                    background: #7f8c8d;
                }
                .form-control, .form-select {
                    background: #ffffff;
                    border: 1px solid #ced4da;
                    color: #2c3e50;
                    border-radius: 6px;
                }
                .form-control:focus, .form-select:focus {
                    background: #ffffff;
                    border-color: #3498db;
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
                    color: #2c3e50;
                }
                .form-label {
                    color: #2c3e50;
                    font-weight: 600;
                }
                .param-box {
                    background: #f8f9fa;
                    border: 1px solid #e1e8ed;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 10px;
                }
                .metric-box {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 14px;
                    text-align: center;
                    border: 1px solid #e1e8ed;
                }
                .metric-value {
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #2c3e50;
                    text-align: center;
                }
                .metric-label {
                    color: #7f8c8d;
                    text-align: center;
                    font-size: 0.85rem;
                    margin-top: 4px;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                    margin-top: 12px;
                }
                /* Fix table width to prevent resizing during scroll */
                .scorecard-table-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: auto;
                    width: 100%;
                }
                /* Target Shiny DataGrid internal elements */
                .scorecard-table-container > div {
                    width: 100% !important;
                    min-width: 100% !important;
                }
                .scorecard-table-container table {
                    width: 100% !important;
                    min-width: 600px;
                    table-layout: fixed !important;
                }
                .scorecard-table-container th,
                .scorecard-table-container td {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    padding: 8px 12px;
                }
                /* Set fixed column widths - var, bin, woe, points */
                .scorecard-table-container th:nth-child(1),
                .scorecard-table-container td:nth-child(1) {
                    width: 150px !important;
                    min-width: 150px !important;
                    max-width: 150px !important;
                }
                .scorecard-table-container th:nth-child(2),
                .scorecard-table-container td:nth-child(2) {
                    width: 250px !important;
                    min-width: 250px !important;
                    max-width: 250px !important;
                }
                .scorecard-table-container th:nth-child(3),
                .scorecard-table-container td:nth-child(3) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
                .scorecard-table-container th:nth-child(4),
                .scorecard-table-container td:nth-child(4) {
                    width: 100px !important;
                    min-width: 100px !important;
                    max-width: 100px !important;
                }
            """)
        ),
        
        ui.h3("Scorecard Generator"),
        
        # Configuration Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard Parameters"),
            ui.row(
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("points", "Base Points", value=600, min=0, step=50),
                        ui.tags.small("Score at target odds", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("odds", "Odds Ratio (1:X)", value=20, min=2, step=1),
                        ui.tags.small("Target odds (e.g., 20 = 1:19)", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_numeric("pdo", "Points to Double Odds", value=50, min=10, step=10),
                        ui.tags.small("PDO scaling factor", style="color: #7f8c8d;")
                    )
                ),
                ui.column(3,
                    ui.div(
                        {"class": "param-box"},
                        ui.input_select("output_format", "Output Format", 
                                       choices=["Table", "List"],
                                       selected="Table"),
                        ui.tags.small("Scorecard output style", style="color: #7f8c8d;")
                    )
                )
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        {"style": "text-align: center; margin-top: 15px;"},
                        ui.input_action_button("analyze", "Generate Scorecard", class_="btn btn-primary btn-lg")
                    )
                )
            )
        ),
        
        # Summary Stats
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Model Summary"),
            ui.output_ui("summary_stats")
        ),
        
        # Scorecard Table Card
        ui.div(
            {"class": "card"},
            ui.div({"class": "card-header"}, "Scorecard"),
            ui.div(
                {"class": "scorecard-table-container"},
                ui.output_data_frame("scorecard_table")
            )
        ),
        
        # Action Buttons
        ui.div(
            {"class": "card", "style": "text-align: center;"},
            ui.input_action_button("run", "Run & Close", class_="btn btn-success btn-lg"),
            ui.input_action_button("close", "Close", class_="btn btn-secondary btn-lg")
        )
    )
    
    def server(input: Inputs, output: Outputs, session: Session):
        scorecard_rv = reactive.Value(None)
        
        @reactive.Effect
        @reactive.event(input.analyze)
        def generate_scorecard():
            points = input.points() or 600
            odds = input.odds() or 20
            pdo = input.pdo() or 50
            
            # Convert odds input (1:X format) to decimal
            odds_decimal = 1 / (odds - 1)
            
            try:
                card = create_scorecard(
                    bins=bins,
                    coefficients=coefficients,
                    points0=points,
                    odds0=odds_decimal,
                    pdo=pdo,
                    basepoints_eq0=False,
                    digits=0
                )
                
                # Use binValue for display instead of bin
                if 'binValue' in card.columns:
                    card['bin'] = card['binValue']
                    card = card.drop(columns=['binValue'])
                
                scorecard_rv.set(card)
                
            except Exception as e:
                print(f"Error generating scorecard: {e}")
                import traceback
                traceback.print_exc()
        
        @output
        @render.ui
        def summary_stats():
            card = scorecard_rv.get()
            if card is None:
                return ui.p("Click 'Generate Scorecard' to view summary", style="text-align: center; color: #7f8c8d;")
            
            # Calculate summary statistics
            num_vars = len([v for v in card['var'].unique() if v != 'basepoints'])
            total_bins = len(card) - 1  # Exclude basepoints row
            
            basepoints_row = card[card['var'] == 'basepoints']
            basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
            
            # Calculate min and max possible scores
            min_score = basepoints
            max_score = basepoints
            
            for var in card['var'].unique():
                if var == 'basepoints':
                    continue
                var_points = card[card['var'] == var]['points']
                if not var_points.empty:
                    min_score += var_points.min()
                    max_score += var_points.max()
            
            return ui.div(
                {"class": "metrics-grid"},
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{num_vars}"),
                    ui.div({"class": "metric-label"}, "Variables")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{total_bins}"),
                    ui.div({"class": "metric-label"}, "Total Bins")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(min_score)}"),
                    ui.div({"class": "metric-label"}, "Min Score")
                ),
                ui.div(
                    {"class": "metric-box"},
                    ui.div({"class": "metric-value"}, f"{int(max_score)}"),
                    ui.div({"class": "metric-label"}, "Max Score")
                )
            )
        
        @output
        @render.data_frame
        def scorecard_table():
            card = scorecard_rv.get()
            if card is None:
                return render.DataGrid(pd.DataFrame())
            
            # Format for display
            display_df = card.copy()
            
            # Round woe for display
            if 'woe' in display_df.columns:
                display_df['woe'] = display_df['woe'].round(4)
            
            return render.DataGrid(display_df, height="450px", width="100%")
        
        @reactive.Effect
        @reactive.event(input.run)
        async def run_and_close():
            card = scorecard_rv.get()
            if card is not None:
                app_results['scorecard'] = card
                app_results['completed'] = True
            await session.close()
        
        @reactive.Effect
        @reactive.event(input.close)
        async def close_app():
            await session.close()
    
    app = App(app_ui, server)
    app.results = app_results
    return app


def run_scorecard_ui(coefficients: pd.DataFrame, bins: pd.DataFrame, port: int = 8052):
    """Run the Scorecard Generator application and return results."""
    import threading
    import time
    import socket
    
    # Check if port is available
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except socket.error:
                return False
    
    if not is_port_available(port):
        print(f"WARNING: Port {port} is already in use!")
        print(f"Trying to use port {port+1} instead...")
        port = port + 1
        if not is_port_available(port):
            print(f"ERROR: Port {port} is also in use. Please close other applications.")
            print("Continuing anyway - the app may not work correctly...")
    
    app = create_scorecard_app(coefficients, bins)
    
    # Run app in a separate thread so we can monitor completion
    def run_server():
        try:
            print("=" * 70)
            print(f"Starting Shiny UI on http://127.0.0.1:{port}")
            print("=" * 70)
            print("IMPORTANT: A browser window should open automatically.")
            print("If it doesn't, manually open: http://127.0.0.1:{port}")
            print("")
            print("STEPS TO COMPLETE:")
            print("  1. Configure parameters in the browser UI")
            print("  2. Click 'Generate Scorecard' button")
            print("  3. Review the scorecard table")
            print("  4. Click 'Run & Close' button (green button at bottom)")
            print("")
            print("Waiting for you to complete the UI workflow...")
            print("=" * 70)
            app.run(port=port, launch_browser=True)
        except Exception as e:
            print(f"Server stopped: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give the server time to start
    time.sleep(2)
    
    # Wait for the app to complete (user clicks Run & Close button)
    wait_count = 0
    while not app.results.get('completed', False):
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            print(f"Still waiting... ({wait_count} seconds elapsed)")
            print(f"Make sure browser is open at: http://127.0.0.1:{port}")
    
    # Give a moment for cleanup
    time.sleep(0.5)
    print("=" * 70)
    print("Scorecard generation complete - returning results")
    print("=" * 70)
    
    return app.results


# =============================================================================
# Read Input Data
# =============================================================================
print("Scorecard Generator Node - Starting...")
print("=" * 70)

# Input 1: Coefficients from Logistic Regression
coefficients = knio.input_tables[0].to_pandas()
print(f"Input 1 (Coefficients): {len(coefficients)} terms")

# Input 2: Bins from WOE Editor
bins = knio.input_tables[1].to_pandas()
print(f"Input 2 (Bins): {len(bins)} rows")

# Bins table summary
if 'var' in bins.columns:
    bins_per_var = bins.groupby('var').size()
    max_bins = bins_per_var.max()
    avg_bins = bins_per_var.mean()
    
    print(f"\nBins per variable: min={bins_per_var.min()}, avg={avg_bins:.1f}, max={max_bins}")
    
    # Informational only - show variables with most bins
    if max_bins > 20:
        print(f"\nVariables with most bins:")
        for var, count in bins_per_var.nlargest(5).items():
            print(f"  - {var}: {count} bins")

# Debug: Show coefficient variable names
print("\nCoefficients:")
for var_name in coefficients.index:
    print(f"  - {var_name}")

# =============================================================================
# Check for Flow Variables (Headless Mode)
# =============================================================================
has_flow_vars = False
points = 600
odds = 20
pdo = 50
output_format = "Table"

try:
    points_fv = knio.flow_variables.get("Points", None)
    if points_fv is not None:
        points = int(points_fv)
        has_flow_vars = True
except:
    pass

try:
    odds_fv = knio.flow_variables.get("Odds", None)
    if odds_fv is not None:
        odds = int(odds_fv)
        has_flow_vars = True
except:
    pass

try:
    pdo_fv = knio.flow_variables.get("PDO", None)
    if pdo_fv is not None:
        pdo = int(pdo_fv)
        has_flow_vars = True
except:
    pass

try:
    output_format = knio.flow_variables.get("OutputFormat", "Table")
except:
    pass

print(f"\nParameters: Points={points}, Odds={odds}, PDO={pdo}")
print("=" * 70)

# =============================================================================
# Main Processing Logic
# =============================================================================

scorecard = pd.DataFrame()

if has_flow_vars:
    # =========================================================================
    # HEADLESS MODE
    # =========================================================================
    print("Running in HEADLESS mode")
    
    # Convert odds input (1:X format) to decimal
    odds_decimal = 1 / (odds - 1)
    
    try:
        scorecard = create_scorecard(
            bins=bins,
            coefficients=coefficients,
            points0=points,
            odds0=odds_decimal,
            pdo=pdo,
            basepoints_eq0=False,
            digits=0
        )
        
        # Use binValue for display instead of bin (matching R behavior)
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])
        
        print(f"\nScorecard created with {len(scorecard)} rows")
        
    except Exception as e:
        print(f"ERROR creating scorecard: {e}")
        import traceback
        traceback.print_exc()

else:
    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================
    if SHINY_AVAILABLE:
        print("Running in INTERACTIVE mode - launching Shiny UI...")
        
        results = run_scorecard_ui(coefficients, bins)
        
        if results['completed']:
            scorecard = results['scorecard']
            print("Interactive session completed successfully")
        else:
            print("Interactive session cancelled - returning empty results")
    else:
        print("=" * 70)
        print("ERROR: Interactive mode requires Shiny, but Shiny is not available.")
        print("Please provide flow variables for headless mode:")
        print("  - Points (int): Base score at target odds, default 600")
        print("  - Odds (int): Odds ratio (1:X), default 20")
        print("  - PDO (int): Points to Double the Odds, default 50")
        print("=" * 70)
        
        # Run with defaults anyway
        odds_decimal = 1 / (odds - 1)
        scorecard = create_scorecard(
            bins=bins,
            coefficients=coefficients,
            points0=points,
            odds0=odds_decimal,
            pdo=pdo,
            basepoints_eq0=False,
            digits=0
        )
        
        if 'binValue' in scorecard.columns:
            scorecard['bin'] = scorecard['binValue']
            scorecard = scorecard.drop(columns=['binValue'])

# =============================================================================
# Output Table
# =============================================================================

# Ensure scorecard is a valid DataFrame
if scorecard is None or scorecard.empty:
    scorecard = pd.DataFrame(columns=['var', 'bin', 'woe', 'points'])

# Output 1: Scorecard table
knio.output_tables[0] = knio.Table.from_pandas(scorecard)

# =============================================================================
# Print Summary
# =============================================================================
print("=" * 70)
print("Scorecard Generator completed successfully")
print("=" * 70)

if not scorecard.empty:
    # Calculate score range
    basepoints_row = scorecard[scorecard['var'] == 'basepoints']
    basepoints = basepoints_row['points'].iloc[0] if not basepoints_row.empty else 0
    
    min_score = basepoints
    max_score = basepoints
    
    for var in scorecard['var'].unique():
        if var == 'basepoints':
            continue
        var_points = scorecard[scorecard['var'] == var]['points']
        if not var_points.empty:
            min_score += var_points.min()
            max_score += var_points.max()
    
    num_vars = len([v for v in scorecard['var'].unique() if v != 'basepoints'])
    
    print(f"Variables in scorecard: {num_vars}")
    print(f"Base points: {int(basepoints)}")
    print(f"Score range: {int(min_score)} to {int(max_score)}")
    
print(f"\nOutput (Scorecard): {len(scorecard)} rows")
print("=" * 70)

# =============================================================================
# WOE Editor Advanced - Configuration Table Generator - DEBUG TOGGLE VERSION
# =============================================================================
# This script generates a table with all supported flow variables for the
# WOE Editor Advanced node. Use with a Table Creator or Python Script node
# to create a configuration row that can be converted to flow variables.
#
# DEBUG TOGGLE VERSION: Debug logging can be enabled/disabled via DEBUG_MODE
#
# USAGE:
# 1. Run this script in a Python Script node
# 2. Connect output to a "Table Row to Variables" node
# 3. Connect the variables output to the WOE Editor Advanced node
#
# PRESET OPTIONS:
# Change the PRESET variable below to generate different configurations:
# - "r_compatible": Matches R logiBin::getBins exactly (default)
# - "credit_scoring": Standard credit scoring settings
# - "fraud_model": Optimized for fraud detection (low event rates)
# - "fraud_enhanced": Fraud with all enhancements enabled (DecisionTree)
# - "iv_optimal": Maximize IV, allows non-monotonic patterns (sweet spots)
# - "custom": Define your own settings below
#
# Compatible with KNIME 5.9, Python 3.9
# Release Date: 2026-01-28
# Version: 1.0-DEBUG-TOGGLE
# =============================================================================

# =============================================================================
# DEBUG MODE TOGGLE - Set to True to enable debug logging, False to disable
# =============================================================================
DEBUG_MODE = True
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import logging
import sys
import traceback
from datetime import datetime

# =============================================================================
# DEBUG LOGGING SETUP (Toggleable)
# =============================================================================

def setup_debug_logging():
    """Configure debug logging with detailed formatting (respects DEBUG_MODE)."""
    logger = logging.getLogger('WOE_CONFIG_DEBUG')
    logger.handlers.clear()
    
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%H:%M:%S.%f'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
    
    return logger

# Initialize logger
logger = setup_debug_logging()

def log_function_entry(func_name, **kwargs):
    """Log function entry with parameters (only when DEBUG_MODE is True)."""
    if not DEBUG_MODE:
        return
    params_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
    logger.debug(f"ENTER {func_name}({params_str})")

def log_function_exit(func_name, result=None):
    """Log function exit with result (only when DEBUG_MODE is True)."""
    if not DEBUG_MODE:
        return
    result_str = repr(result) if result is not None else "None"
    if len(result_str) > 200:
        result_str = result_str[:200] + "..."
    logger.debug(f"EXIT {func_name} -> {result_str}")

def log_exception(func_name, e):
    """Log exception with full traceback (always logs, regardless of DEBUG_MODE)."""
    logger.error(f"EXCEPTION in {func_name}: {type(e).__name__}: {str(e)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")

# =============================================================================
# USER SETTINGS (CHANGE THESE)
# =============================================================================

# Your target column name (MUST match exactly a column in your input data)
DEPENDENT_VARIABLE = "isFPD_wRI"
if DEBUG_MODE:
    logger.debug(f"USER SETTING: DEPENDENT_VARIABLE = {repr(DEPENDENT_VARIABLE)}")

# Which value represents "bad" outcome (usually "1" for binary 0/1)
TARGET_CATEGORY = "1"
if DEBUG_MODE:
    logger.debug(f"USER SETTING: TARGET_CATEGORY = {repr(TARGET_CATEGORY)}")

# Select preset for algorithm/enhancement settings
PRESET = "fraud_enhanced"  # Options: "r_compatible", "credit_scoring", "fraud_model", "fraud_enhanced", "custom"
if DEBUG_MODE:
    logger.debug(f"USER SETTING: PRESET = {repr(PRESET)}")

# =============================================================================
# Preset Configurations
# =============================================================================

if DEBUG_MODE:
    logger.info("Defining preset configurations...")

PRESETS = {
    # R-Compatible: Matches R logiBin::getBins exactly
    "r_compatible": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 1,
        "GroupNA": 1,
        "Algorithm": "DecisionTree",
        "MinBinPct": 0.01,
        "MinBinCount": 20,
        "MaxBins": 10,
        "UseShrinkage": "false",
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "false",
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        "SingleBinProtection": "true",
    },
    
    # Credit Scoring: Standard industry settings (10-20% event rates)
    "credit_scoring": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 1,
        "GroupNA": 1,
        "Algorithm": "DecisionTree",
        "MinBinPct": 0.05,          # 5% per bin (industry standard)
        "MinBinCount": 50,
        "MaxBins": 10,
        "UseShrinkage": "false",
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "false",
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        "SingleBinProtection": "true",
    },
    
    # Fraud Model: Optimized for low event rates (1-5%)
    "fraud_model": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 1,
        "GroupNA": 1,
        "Algorithm": "DecisionTree",
        "MinBinPct": 0.01,          # 1% per bin (more granular)
        "MinBinCount": 20,
        "MaxBins": 10,
        "UseShrinkage": "true",     # Shrinkage for rare events
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "false",
        "AdaptiveMinProp": "true",  # Relax for sparse data
        "MinEventCount": "true",    # Ensure events per bin
        "AutoRetry": "true",        # Retry if no splits
        "ChiSquareValidation": "false",
        "SingleBinProtection": "true",
    },
    
    # Fraud Enhanced: All enhancements enabled (DecisionTree)
    "fraud_enhanced": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 1,
        "GroupNA": 1,
        "Algorithm": "DecisionTree",
        "MinBinPct": 0.01,
        "MinBinCount": 20,
        "MaxBins": 10,
        "UseShrinkage": "true",
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "true",  # Master switch - enables all
        "AdaptiveMinProp": "true",
        "MinEventCount": "true",
        "AutoRetry": "true",
        "ChiSquareValidation": "true",
        "SingleBinProtection": "true",
    },
    
    # IV-Optimal: Maximize IV, allows non-monotonic patterns (sweet spots)
    "iv_optimal": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 0,         # Do NOT force monotonicity - allow sweet spots
        "GroupNA": 1,
        "Algorithm": "IVOptimal", # IV-maximizing algorithm
        "MinBinPct": 0.01,
        "MinBinCount": 20,
        "MaxBins": 10,
        "UseShrinkage": "true",
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "false",  # IVOptimal doesn't need these
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        "SingleBinProtection": "true",
    },
    
    # Custom: Define your own settings
    "custom": {
        "DependentVariable": "",  # Set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Set by TARGET_CATEGORY above
        "OptimizeAll": 1,
        "GroupNA": 1,
        "Algorithm": "DecisionTree",  # "DecisionTree", "ChiMerge", or "IVOptimal"
        "MinBinPct": 0.02,
        "MinBinCount": 30,
        "MaxBins": 10,
        "UseShrinkage": "false",
        "ShrinkageStrength": 0.1,
        "UseEnhancements": "false",
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        "SingleBinProtection": "true",
    },
}

if DEBUG_MODE:
    logger.debug(f"Defined {len(PRESETS)} preset configurations: {list(PRESETS.keys())}")

# =============================================================================
# Generate Configuration Table
# =============================================================================

def generate_configuration():
    """Generate the configuration DataFrame based on user settings."""
    if DEBUG_MODE:
        log_function_entry('generate_configuration', 
                           PRESET=PRESET, 
                           DEPENDENT_VARIABLE=DEPENDENT_VARIABLE,
                           TARGET_CATEGORY=TARGET_CATEGORY)
    
    try:
        # Get selected preset
        if DEBUG_MODE:
            logger.debug(f"Checking if preset '{PRESET}' exists in PRESETS...")
        if PRESET not in PRESETS:
            if DEBUG_MODE:
                logger.warning(f"Unknown preset '{PRESET}', using 'r_compatible'")
            selected_preset = "r_compatible"
        else:
            selected_preset = PRESET
            if DEBUG_MODE:
                logger.debug(f"Using preset: {selected_preset}")
        
        config = PRESETS[selected_preset].copy()
        if DEBUG_MODE:
            logger.debug(f"Loaded preset config with {len(config)} settings")
        
        # Apply user settings (override preset values)
        if DEBUG_MODE:
            logger.debug(f"Overriding DependentVariable with: {DEPENDENT_VARIABLE}")
        config["DependentVariable"] = DEPENDENT_VARIABLE
        
        if DEBUG_MODE:
            logger.debug(f"Overriding TargetCategory with: {TARGET_CATEGORY}")
        config["TargetCategory"] = TARGET_CATEGORY
        
        # Log all config values
        if DEBUG_MODE:
            logger.debug("Final configuration values:")
            for key, value in config.items():
                logger.debug(f"  {key}: {repr(value)} (type: {type(value).__name__})")
        
        # Create DataFrame with proper column order
        column_order = [
            # Core settings
            "DependentVariable",
            "TargetCategory",
            "OptimizeAll",
            "GroupNA",
            # Algorithm settings
            "Algorithm",
            "MinBinPct",
            "MinBinCount",
            "MaxBins",
            # WOE options
            "UseShrinkage",
            "ShrinkageStrength",
            # Enhancements
            "UseEnhancements",
            "AdaptiveMinProp",
            "MinEventCount",
            "AutoRetry",
            "ChiSquareValidation",
            "SingleBinProtection",
        ]
        if DEBUG_MODE:
            logger.debug(f"Column order defined with {len(column_order)} columns")
        
        if DEBUG_MODE:
            logger.debug("Creating DataFrame...")
        df = pd.DataFrame([{col: config[col] for col in column_order}])
        
        if DEBUG_MODE:
            logger.debug(f"DataFrame created with shape: {df.shape}")
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame dtypes:\n{df.dtypes}")
        
        if DEBUG_MODE:
            log_function_exit('generate_configuration', result=f"DataFrame({df.shape})")
        return df, selected_preset
        
    except Exception as e:
        log_exception('generate_configuration', e)
        raise

# =============================================================================
# Main Execution
# =============================================================================

if DEBUG_MODE:
    logger.info("=" * 70)
    logger.info("WOE CONFIG GENERATOR - DEBUG TOGGLE VERSION")
    logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
    logger.info("=" * 70)
    logger.info(f"Script started at: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")

try:
    # Generate configuration
    if DEBUG_MODE:
        logger.info("Generating configuration table...")
    df, selected_preset = generate_configuration()
    
    # Output to KNIME
    if DEBUG_MODE:
        logger.info("Writing output to KNIME...")
        logger.debug(f"Calling knio.Table.from_pandas(df) with DataFrame shape {df.shape}")
    
    knio.output_tables[0] = knio.Table.from_pandas(df)
    if DEBUG_MODE:
        logger.debug("Output table successfully written to knio.output_tables[0]")
    
    # Print summary
    if DEBUG_MODE:
        logger.info("=" * 70)
        logger.info(f"WOE Editor Advanced Configuration")
        logger.info("=" * 70)
        logger.info(f"Preset: {selected_preset}")
        logger.info(f"Dependent Variable: {DEPENDENT_VARIABLE}")
        logger.info(f"Target Category: {TARGET_CATEGORY}")
        logger.info(f"Generated {len(df.columns)} flow variable columns:")
        
        # Group by category for display
        categories = {
            "Core Settings": ["DependentVariable", "TargetCategory", "OptimizeAll", "GroupNA"],
            "Algorithm": ["Algorithm", "MinBinPct", "MinBinCount", "MaxBins"],
            "WOE Options": ["UseShrinkage", "ShrinkageStrength"],
            "Enhancements": ["UseEnhancements", "AdaptiveMinProp", "MinEventCount", 
                             "AutoRetry", "ChiSquareValidation", "SingleBinProtection"],
        }
        
        for category, cols in categories.items():
            logger.info(f"  {category}:")
            for col in cols:
                val = df[col].iloc[0]
                logger.info(f"    {col}: {val}")
        
        logger.info("=" * 70)
        logger.info("NEXT STEP: Connect output to 'Table Row to Variables' node")
        logger.info("=" * 70)
        logger.info(f"Script completed successfully at: {datetime.now().isoformat()}")

except Exception as e:
    log_exception('main', e)
    if DEBUG_MODE:
        logger.critical(f"Script failed with error: {e}")
    raise

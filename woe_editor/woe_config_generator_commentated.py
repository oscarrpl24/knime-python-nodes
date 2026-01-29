# =============================================================================
# WOE Editor Advanced - Configuration Table Generator
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
# ----------------------
# This script creates a configuration table that contains all the settings
# (called "flow variables" in KNIME) needed to control the WOE Editor Advanced node.
# 
# WOE stands for "Weight of Evidence" - a statistical technique used in credit
# risk modeling to convert categorical and continuous variables into numeric
# values that represent their predictive power.
#
# WHY THIS SCRIPT EXISTS:
# -----------------------
# Instead of manually typing in all the settings every time you run the WOE
# Editor node, this script generates a table with predefined configurations
# (presets) that you can easily switch between. This is especially useful when
# you want to test different algorithm settings or share configurations with
# colleagues.
#
# HOW TO USE THIS SCRIPT IN KNIME:
# --------------------------------
# 1. Place this script in a Python Script node in KNIME
# 2. Run the node - it will output a table with one row containing all settings
# 3. Connect the output to a "Table Row to Variables" node
#    (This node converts table columns into flow variables)
# 4. Connect the flow variables output to the WOE Editor Advanced node
#    (The WOE Editor will read these variables and configure itself automatically)
#
# AVAILABLE PRESET OPTIONS:
# -------------------------
# The PRESET variable (defined below) controls which configuration is generated:
#
# - "r_compatible": 
#   Produces binning results identical to the R language package logiBin::getBins.
#   Use this if you need to match results from an existing R-based workflow.
#
# - "credit_scoring": 
#   Standard settings for traditional credit scoring models where the "bad" event
#   rate (defaults) is typically 10-20% of the population.
#
# - "fraud_model": 
#   Optimized for fraud detection where "bad" events are rare (1-5% of cases).
#   Uses shrinkage and adaptive settings to handle sparse data.
#
# - "fraud_enhanced": 
#   Same as fraud_model but with ALL enhancement features enabled.
#   Provides maximum protection against edge cases but may be slower.
#
# - "iv_optimal": 
#   Focuses on maximizing Information Value (IV) without enforcing monotonicity.
#   Allows "sweet spots" where middle bins can have different risk patterns.
#   Use this for exploratory analysis but not typically for production scorecards.
#
# - "custom": 
#   A template where you can define your own settings manually.
#
# COMPATIBILITY:
# --------------
# This script is designed for KNIME version 5.9 with Python 3.9
#
# =============================================================================

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

# Import the KNIME scripting interface module
# This module provides functions to read input tables from KNIME and write output
# tables back to KNIME. It's the bridge between Python code and the KNIME workflow.
# "knio" is a commonly used abbreviation (short for "knime io" - input/output)
import knime.scripting.io as knio

# Import the pandas library and give it the alias "pd"
# Pandas is Python's primary data manipulation library. It provides DataFrames,
# which are like Excel spreadsheets or database tables in Python.
# We use pandas to create the configuration table that will be output to KNIME.
import pandas as pd

# =============================================================================
# USER SETTINGS SECTION
# =============================================================================
# IMPORTANT: These are the settings you should modify for your specific use case.
# Everything else in this script can be left as-is.
# =============================================================================

# DEPENDENT_VARIABLE: The name of your target column
# --------------------------------------------------
# This is the column in your dataset that contains the outcome you're trying
# to predict. For credit risk, this is typically something like:
# - "is_default" (1 if customer defaulted, 0 otherwise)
# - "is_bad" (1 if the loan went bad, 0 if it was repaid)
# - "isFPD_wRI" (First Payment Default with Reject Inference)
#
# CRITICAL: This value MUST match the exact column name in your input data,
# including capitalization. "IsBad" is different from "is_bad" in Python/KNIME.
DEPENDENT_VARIABLE = "isFPD_wRI"

# TARGET_CATEGORY: Which value in the target column means "bad"?
# --------------------------------------------------------------
# Your target column is typically binary (two values: 0 and 1).
# This setting tells the algorithm which value represents the "bad" outcome
# that you're trying to predict.
#
# Common conventions:
# - "1" = bad (default, fraud, claim, etc.) - this is the most common
# - "0" = good (no default, no fraud, etc.)
#
# If you set this wrong, your model predictions will be inverted (high scores
# for low risk instead of high risk). Always verify this matches your data!
TARGET_CATEGORY = "1"

# PRESET: Choose which preset configuration to use
# -------------------------------------------------
# This single setting determines all the algorithm parameters below.
# See the detailed descriptions at the top of this file for each preset.
#
# Available options:
# - "r_compatible"    : Match R logiBin::getBins results exactly
# - "credit_scoring"  : Standard 5% per bin, suitable for 10-20% event rates
# - "fraud_model"     : Smaller bins, shrinkage, for 1-5% event rates
# - "fraud_enhanced"  : Fraud model with all enhancement flags enabled
# - "iv_optimal"      : Maximize IV, non-monotonic patterns allowed
# - "custom"          : Define your own settings in the custom section below
PRESET = "fraud_enhanced"  # <-- Change this to switch configurations

# =============================================================================
# PRESET CONFIGURATIONS DICTIONARY
# =============================================================================
# This is a Python dictionary that stores all the preset configurations.
# A dictionary is like a lookup table: you give it a key (preset name) and
# it returns the corresponding value (all the settings for that preset).
#
# Each preset is itself a dictionary containing key-value pairs where:
# - The key is the flow variable name (what KNIME will look for)
# - The value is the setting for that variable
#
# UNDERSTANDING THE SETTINGS:
# ---------------------------
# Each setting in the presets controls a specific aspect of the WOE binning:
#
# DependentVariable: Name of target column (set automatically from above)
# TargetCategory: Which value is "bad" (set automatically from above)
# OptimizeAll: 1 = force monotonic WOE trends, 0 = allow non-monotonic
# GroupNA: 1 = treat missing values as a separate bin, 0 = exclude them
# Algorithm: Which binning algorithm to use:
#   - "DecisionTree": Uses decision tree splits (most common, R-compatible)
#   - "ChiMerge": Uses chi-square test to merge bins
#   - "IVOptimal": Optimizes directly for Information Value
# MinBinPct: Minimum percentage of total observations per bin (0.01 = 1%)
# MinBinCount: Minimum number of observations per bin (absolute count)
# MaxBins: Maximum number of bins to create for each variable
# UseShrinkage: "true"/"false" - Apply Bayesian shrinkage to WOE values
#   (Shrinkage pulls extreme WOE values toward 0, useful for small bins)
# ShrinkageStrength: How much shrinkage to apply (0 = none, 1 = full)
# UseEnhancements: "true"/"false" - Master switch for all enhancements
# AdaptiveMinProp: "true"/"false" - Relax minimum bin size requirements
#   when data is sparse
# MinEventCount: "true"/"false" - Ensure each bin has minimum events
# AutoRetry: "true"/"false" - Automatically retry binning with relaxed
#   parameters if the first attempt fails
# ChiSquareValidation: "true"/"false" - Validate bins using chi-square test
# SingleBinProtection: "true"/"false" - Prevent variables from collapsing
#   into a single bin (which would have zero predictive power)
# =============================================================================

# Create the PRESETS dictionary containing all available configurations
PRESETS = {
    
    # =========================================================================
    # PRESET: R-COMPATIBLE
    # =========================================================================
    # This preset produces results that match the R package logiBin::getBins.
    # Use this when you need exact reproducibility with R-based workflows or
    # when validating that this Python implementation matches R output.
    #
    # Key characteristics:
    # - Uses DecisionTree algorithm (what R logiBin uses internally)
    # - MinBinPct of 1% allows granular bins
    # - All enhancements disabled to match R behavior exactly
    # =========================================================================
    "r_compatible": {
        # DependentVariable and TargetCategory are empty strings here because
        # they will be overwritten with the user's settings from above.
        # This is a pattern that allows presets to be complete templates while
        # still letting the user specify their own target variable.
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # OptimizeAll = 1 means enforce monotonic WOE patterns
        # (WOE should increase or decrease consistently across bins)
        # This is standard for credit scoring to ensure interpretable models
        "OptimizeAll": 1,
        
        # GroupNA = 1 means create a separate bin for missing/null values
        # This preserves information about missingness which can be predictive
        "GroupNA": 1,
        
        # DecisionTree algorithm uses information gain to find optimal splits
        # This is the algorithm used by R's logiBin package
        "Algorithm": "DecisionTree",
        
        # MinBinPct = 0.01 means each bin must contain at least 1% of observations
        # Lower values allow more granular binning but may create unstable bins
        "MinBinPct": 0.01,
        
        # MinBinCount = 20 is the absolute minimum observations per bin
        # This overrides MinBinPct if the percentage would result in fewer obs
        "MinBinCount": 20,
        
        # MaxBins = 10 limits each variable to maximum 10 bins
        # More bins = more granular but risk of overfitting; fewer = more stable
        "MaxBins": 10,
        
        # UseShrinkage = "false" means no Bayesian shrinkage applied
        # Note: This is a string "false", not a boolean False, because KNIME
        # flow variables treat boolean values as strings
        "UseShrinkage": "false",
        
        # ShrinkageStrength = 0.1 is the shrinkage intensity (not used when
        # UseShrinkage is false, but included for completeness)
        # Range is 0 (no shrinkage) to 1 (full shrinkage toward population WOE)
        "ShrinkageStrength": 0.1,
        
        # UseEnhancements = "false" disables all enhancement features
        # This ensures exact R compatibility
        "UseEnhancements": "false",
        
        # AdaptiveMinProp = "false" means strict adherence to MinBinPct
        # When true, would relax this requirement for sparse variables
        "AdaptiveMinProp": "false",
        
        # MinEventCount = "false" means no minimum event count per bin
        # When true, ensures each bin has enough "bad" events for stability
        "MinEventCount": "false",
        
        # AutoRetry = "false" means no automatic retry with relaxed parameters
        # When true, would retry binning if initial attempt fails
        "AutoRetry": "false",
        
        # ChiSquareValidation = "false" means no statistical validation of bins
        # When true, uses chi-square test to validate bin boundaries
        "ChiSquareValidation": "false",
        
        # SingleBinProtection = "true" prevents a variable from collapsing to 1 bin
        # A single bin has no predictive power (all WOE values would be the same)
        "SingleBinProtection": "true",
    },
    
    # =========================================================================
    # PRESET: CREDIT SCORING
    # =========================================================================
    # Standard settings for traditional credit scoring models.
    # Designed for datasets where the "bad" event rate is 10-20%.
    #
    # Key characteristics:
    # - 5% minimum bin size (industry standard for stability)
    # - 50 minimum observations per bin
    # - No enhancements (simple, interpretable binning)
    # =========================================================================
    "credit_scoring": {
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # Enforce monotonic WOE trends for regulatory compliance
        "OptimizeAll": 1,
        
        # Create separate bin for missing values
        "GroupNA": 1,
        
        # Use DecisionTree algorithm for stable, interpretable bins
        "Algorithm": "DecisionTree",
        
        # MinBinPct = 0.05 means each bin needs at least 5% of observations
        # This is more conservative than r_compatible (1%)
        # 5% is a common industry standard that balances granularity with stability
        "MinBinPct": 0.05,
        
        # MinBinCount = 50 requires at least 50 observations per bin
        # Higher than r_compatible (20) for more stable WOE estimates
        "MinBinCount": 50,
        
        # Maximum 10 bins per variable (standard practice)
        "MaxBins": 10,
        
        # No shrinkage needed with larger bins
        "UseShrinkage": "false",
        "ShrinkageStrength": 0.1,
        
        # All enhancements disabled for simple, standard binning
        "UseEnhancements": "false",
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        
        # Protect against single-bin collapse
        "SingleBinProtection": "true",
    },
    
    # =========================================================================
    # PRESET: FRAUD MODEL
    # =========================================================================
    # Optimized for fraud detection where event rates are very low (1-5%).
    # When "bad" events are rare, standard binning can create unstable bins
    # with few or no events. This preset addresses those challenges.
    #
    # Key characteristics:
    # - Smaller bins allowed (1%) for better granularity
    # - Shrinkage enabled to stabilize WOE estimates from small bins
    # - Adaptive settings to handle sparse data
    # - Auto-retry to recover from binning failures
    # =========================================================================
    "fraud_model": {
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # Enforce monotonic WOE trends
        "OptimizeAll": 1,
        
        # Create separate bin for missing values (missingness often predictive of fraud)
        "GroupNA": 1,
        
        # DecisionTree algorithm
        "Algorithm": "DecisionTree",
        
        # MinBinPct = 0.01 allows more granular bins (1% of data each)
        # Important for fraud where you need to identify small high-risk segments
        "MinBinPct": 0.01,
        
        # MinBinCount = 20 (lower count OK with shrinkage)
        "MinBinCount": 20,
        
        # Maximum 10 bins
        "MaxBins": 10,
        
        # UseShrinkage = "true" applies Bayesian shrinkage to WOE values
        # This is CRITICAL for fraud models! When bins have few events,
        # WOE values can be extreme (very large positive or negative).
        # Shrinkage pulls these extreme values toward 0 (the population average),
        # which produces more stable models that generalize better.
        "UseShrinkage": "true",
        
        # ShrinkageStrength = 0.1 is mild shrinkage
        # Higher values (up to 1.0) pull WOE more strongly toward 0
        "ShrinkageStrength": 0.1,
        
        # UseEnhancements = "false" but individual enhancements are enabled
        # This allows selective use of specific features
        "UseEnhancements": "false",
        
        # AdaptiveMinProp = "true" relaxes minimum bin size for sparse variables
        # If a variable has very few events, strict 1% bins might leave some
        # bins with zero events. This setting allows smaller bins when needed.
        "AdaptiveMinProp": "true",
        
        # MinEventCount = "true" ensures each bin has a minimum number of events
        # Prevents bins with 0 events which would have infinite WOE
        "MinEventCount": "true",
        
        # AutoRetry = "true" automatically retries binning with relaxed parameters
        # If initial binning fails (no valid splits found), this will try again
        # with larger bins or fewer bins
        "AutoRetry": "true",
        
        # No chi-square validation for this preset
        "ChiSquareValidation": "false",
        
        # Protect against single-bin collapse
        "SingleBinProtection": "true",
    },
    
    # =========================================================================
    # PRESET: FRAUD ENHANCED
    # =========================================================================
    # Maximum protection preset for fraud detection.
    # All enhancement features are enabled for the most robust binning.
    #
    # Key characteristics:
    # - Same base settings as fraud_model
    # - UseEnhancements = "true" enables ALL enhancement features
    # - Chi-square validation for statistical rigor
    # - May be slower but provides maximum stability
    # =========================================================================
    "fraud_enhanced": {
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # Enforce monotonic WOE trends
        "OptimizeAll": 1,
        
        # Create separate bin for missing values
        "GroupNA": 1,
        
        # DecisionTree algorithm
        "Algorithm": "DecisionTree",
        
        # Same bin size settings as fraud_model
        "MinBinPct": 0.01,
        "MinBinCount": 20,
        "MaxBins": 10,
        
        # Enable shrinkage for stable WOE estimates
        "UseShrinkage": "true",
        "ShrinkageStrength": 0.1,
        
        # UseEnhancements = "true" is the MASTER SWITCH that enables all enhancements
        # When this is "true", the WOE Editor will use all available enhancement
        # algorithms, regardless of the individual flags below
        "UseEnhancements": "true",
        
        # Individual enhancements (all enabled for maximum protection)
        "AdaptiveMinProp": "true",
        "MinEventCount": "true",
        "AutoRetry": "true",
        
        # ChiSquareValidation = "true" validates bin boundaries statistically
        # Uses chi-square test to ensure bins are significantly different
        # This can merge bins that aren't statistically distinct
        "ChiSquareValidation": "true",
        
        # Protect against single-bin collapse
        "SingleBinProtection": "true",
    },
    
    # =========================================================================
    # PRESET: IV OPTIMAL
    # =========================================================================
    # Maximize Information Value (IV) without forcing monotonicity.
    # This preset allows "sweet spots" - non-monotonic patterns where middle
    # bins might have different risk than the trend would suggest.
    #
    # USE CASE: Exploratory analysis to understand variable behavior.
    # NOT RECOMMENDED for production scorecards because:
    # - Non-monotonic patterns are harder to explain to regulators
    # - Sweet spots may be artifacts of the data rather than real patterns
    #
    # Key characteristics:
    # - OptimizeAll = 0 disables monotonicity enforcement
    # - IVOptimal algorithm directly optimizes for IV
    # - Shrinkage enabled for stability
    # =========================================================================
    "iv_optimal": {
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # OptimizeAll = 0 DISABLES monotonicity enforcement
        # This allows WOE patterns like: low, high, low (a "sweet spot" in the middle)
        # Such patterns CAN be real (e.g., moderate income might be safest)
        # but they're often statistical artifacts
        "OptimizeAll": 0,
        
        # Create separate bin for missing values
        "GroupNA": 1,
        
        # IVOptimal algorithm directly maximizes Information Value
        # Unlike DecisionTree which uses information gain for splits,
        # IVOptimal explicitly optimizes the IV formula:
        # IV = sum((dist_bad - dist_good) * WOE)
        "Algorithm": "IVOptimal",
        
        # Standard bin size settings
        "MinBinPct": 0.01,
        "MinBinCount": 20,
        "MaxBins": 10,
        
        # Shrinkage enabled for stability
        "UseShrinkage": "true",
        "ShrinkageStrength": 0.1,
        
        # Enhancements disabled because IVOptimal algorithm doesn't need them
        # The algorithm itself handles edge cases internally
        "UseEnhancements": "false",
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        
        # Protect against single-bin collapse
        "SingleBinProtection": "true",
    },
    
    # =========================================================================
    # PRESET: CUSTOM
    # =========================================================================
    # A template for defining your own settings.
    # Modify the values below to create a custom configuration.
    #
    # Instructions:
    # 1. Copy this preset or modify it directly
    # 2. Set PRESET = "custom" at the top of the file
    # 3. Adjust the values below to match your requirements
    # =========================================================================
    "custom": {
        "DependentVariable": "",  # Will be set by DEPENDENT_VARIABLE above
        "TargetCategory": "",     # Will be set by TARGET_CATEGORY above
        
        # OptimizeAll: 1 = monotonic WOE, 0 = non-monotonic allowed
        "OptimizeAll": 1,
        
        # GroupNA: 1 = separate NA bin, 0 = exclude NAs
        "GroupNA": 1,
        
        # Algorithm options:
        # - "DecisionTree": Information gain based splits (R-compatible)
        # - "ChiMerge": Chi-square based bin merging
        # - "IVOptimal": Direct IV optimization
        "Algorithm": "DecisionTree",
        
        # MinBinPct: Minimum percentage per bin (0.01 = 1%, 0.05 = 5%)
        "MinBinPct": 0.02,
        
        # MinBinCount: Minimum observations per bin (absolute number)
        "MinBinCount": 30,
        
        # MaxBins: Maximum number of bins per variable
        "MaxBins": 10,
        
        # UseShrinkage: "true" or "false" - apply Bayesian shrinkage
        "UseShrinkage": "false",
        
        # ShrinkageStrength: 0.0 to 1.0 - how much to shrink toward 0
        "ShrinkageStrength": 0.1,
        
        # UseEnhancements: "true" or "false" - master switch for enhancements
        "UseEnhancements": "false",
        
        # Individual enhancement flags (only matter if UseEnhancements is "true")
        "AdaptiveMinProp": "false",
        "MinEventCount": "false",
        "AutoRetry": "false",
        "ChiSquareValidation": "false",
        
        # SingleBinProtection: "true" or "false" - prevent single bin collapse
        "SingleBinProtection": "true",
    },
}

# =============================================================================
# CONFIGURATION TABLE GENERATION
# =============================================================================
# This section takes the user's selected preset and creates a pandas DataFrame
# that will be output to KNIME.
# =============================================================================

# Step 1: Validate the selected preset
# -------------------------------------
# Check if the preset name the user specified actually exists in our PRESETS
# dictionary. If not, print a warning and fall back to "r_compatible".
# This prevents the script from crashing if there's a typo in PRESET.
if PRESET not in PRESETS:
    # Print a warning message to the KNIME console
    # The f-string (f"...") allows us to embed the variable PRESET directly in the string
    print(f"Warning: Unknown preset '{PRESET}', using 'r_compatible'")
    
    # Reset PRESET to a known valid value
    PRESET = "r_compatible"

# Step 2: Create a copy of the selected preset
# ---------------------------------------------
# We use .copy() to create a new dictionary rather than referencing the original.
# This is important because we're about to modify the config (adding DependentVariable
# and TargetCategory), and we don't want to modify the original preset definition.
# Without .copy(), changes to config would also change PRESETS[PRESET].
config = PRESETS[PRESET].copy()

# Step 3: Apply user settings to override preset defaults
# -------------------------------------------------------
# The preset has empty strings for DependentVariable and TargetCategory.
# Here we replace those with the user's actual values from the top of the file.
# This pattern allows presets to be reusable templates while still letting
# users specify their specific target variable.
config["DependentVariable"] = DEPENDENT_VARIABLE
config["TargetCategory"] = TARGET_CATEGORY

# Step 4: Define the column order for the output table
# -----------------------------------------------------
# When we create the DataFrame, we want the columns to appear in a logical
# order (core settings first, then algorithm settings, then options).
# This makes the output easier to read and understand in KNIME.
# The order here matches the logical grouping of settings.
column_order = [
    # Core settings - the fundamental inputs that every run needs
    "DependentVariable",    # Which column is the target
    "TargetCategory",       # Which value is "bad"
    "OptimizeAll",          # Whether to enforce monotonicity
    "GroupNA",              # How to handle missing values
    
    # Algorithm settings - control how binning is performed
    "Algorithm",            # Which binning algorithm to use
    "MinBinPct",            # Minimum percentage per bin
    "MinBinCount",          # Minimum count per bin
    "MaxBins",              # Maximum number of bins
    
    # WOE options - modify how WOE values are calculated
    "UseShrinkage",         # Whether to apply shrinkage
    "ShrinkageStrength",    # How much shrinkage to apply
    
    # Enhancements - additional features for edge case handling
    "UseEnhancements",      # Master switch for enhancements
    "AdaptiveMinProp",      # Adapt bin sizes for sparse data
    "MinEventCount",        # Ensure minimum events per bin
    "AutoRetry",            # Retry with relaxed parameters
    "ChiSquareValidation",  # Statistical validation of bins
    "SingleBinProtection",  # Prevent single-bin collapse
]

# Step 5: Create the pandas DataFrame
# ------------------------------------
# A DataFrame is created from a list containing one dictionary (our config).
# The dictionary comprehension {col: config[col] for col in column_order}
# creates a new dictionary with only the columns we specified, in that order.
#
# Breaking down the syntax:
# - pd.DataFrame([...]) creates a DataFrame from a list of dictionaries
# - Each dictionary in the list becomes one row
# - We have only one dictionary, so we get a table with one row
# - {col: config[col] for col in column_order} is a "dictionary comprehension"
#   that creates a dictionary by iterating through column_order
df = pd.DataFrame([{col: config[col] for col in column_order}])

# =============================================================================
# OUTPUT SECTION
# =============================================================================
# This section sends the configuration table back to KNIME and prints a
# summary to the KNIME console.
# =============================================================================

# Step 6: Output the DataFrame to KNIME
# --------------------------------------
# knio.output_tables is a list of output ports for the Python Script node.
# output_tables[0] is the first (and in this case, only) output port.
# knio.Table.from_pandas(df) converts our pandas DataFrame into a KNIME table
# format that can be passed to downstream nodes.
# 
# After this line executes, the KNIME node's output port will contain our
# configuration table with one row and 16 columns (one for each setting).
knio.output_tables[0] = knio.Table.from_pandas(df)

# Step 7: Print a summary header to the KNIME console
# -----------------------------------------------------
# The print() function writes text to the KNIME console (visible in the
# Python Script node's Console tab). This helps users verify the configuration.

# Print a line of 70 equal signs as a visual separator
print("=" * 70)

# Print the title
print(f"WOE Editor Advanced Configuration")

# Print another separator
print("=" * 70)

# Print basic info about what was generated
# \n is a newline character that creates a blank line
print(f"\nPreset: {PRESET}")
print(f"Dependent Variable: {DEPENDENT_VARIABLE}")
print(f"Target Category: {TARGET_CATEGORY}")

# Print the count of columns generated
# len(df.columns) returns the number of columns in the DataFrame
# This should always be 16 for the current configuration
print(f"\nGenerated {len(df.columns)} flow variable columns:\n")

# Step 8: Define categories for organized display
# -------------------------------------------------
# To make the console output more readable, we group the settings into
# logical categories. This dictionary maps category names to lists of
# column names that belong in each category.
categories = {
    # Core Settings: fundamental inputs
    "Core Settings": ["DependentVariable", "TargetCategory", "OptimizeAll", "GroupNA"],
    
    # Algorithm: binning algorithm and parameters
    "Algorithm": ["Algorithm", "MinBinPct", "MinBinCount", "MaxBins"],
    
    # WOE Options: WOE calculation modifiers
    "WOE Options": ["UseShrinkage", "ShrinkageStrength"],
    
    # Enhancements: optional enhancement features
    "Enhancements": ["UseEnhancements", "AdaptiveMinProp", "MinEventCount", 
                     "AutoRetry", "ChiSquareValidation", "SingleBinProtection"],
}

# Step 9: Print each category and its settings
# ----------------------------------------------
# We iterate through the categories dictionary, printing each category name
# and then the settings within that category.

# .items() returns key-value pairs from the dictionary
# category is the category name (e.g., "Core Settings")
# cols is the list of column names in that category
for category, cols in categories.items():
    # Print the category header with indentation
    print(f"  {category}:")
    
    # For each column in this category, print the name and value
    for col in cols:
        # df[col].iloc[0] gets the first (and only) row's value for this column
        # iloc[0] means "get the row at integer location 0" (first row)
        val = df[col].iloc[0]
        
        # Print with double indentation for visual hierarchy
        print(f"    {col}: {val}")
    
    # Print a blank line between categories for readability
    print()

# Step 10: Print final instructions
# -----------------------------------
# Remind the user what to do next with this output.

# Print a separator
print("=" * 70)

# Print the next step instruction
# This tells users how to use the output table in their KNIME workflow
print("NEXT STEP: Connect output to 'Table Row to Variables' node")

# Print final separator
print("=" * 70)

# =============================================================================
# END OF SCRIPT
# =============================================================================
# After this script runs:
# 1. The KNIME node's output port contains a table with one row
# 2. Each column in that row is a configuration setting
# 3. Connect this output to "Table Row to Variables" to convert to flow variables
# 4. Connect those flow variables to the WOE Editor Advanced node
# 5. The WOE Editor will read the flow variables and configure itself
#
# Example workflow:
# [This Node] --> [Table Row to Variables] --> [WOE Editor Advanced]
#                         |
#                         v
#                  (Flow Variables)
# =============================================================================

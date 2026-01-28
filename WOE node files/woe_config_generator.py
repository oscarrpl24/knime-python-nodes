# =============================================================================
# WOE Editor Advanced - Configuration Table Generator
# =============================================================================
# This script generates a table with all supported flow variables for the
# WOE Editor Advanced node. Use with a Table Creator or Python Script node
# to create a configuration row that can be converted to flow variables.
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
# =============================================================================

import knime.scripting.io as knio
import pandas as pd

# =============================================================================
# USER SETTINGS (CHANGE THESE)
# =============================================================================

# Your target column name (MUST match exactly a column in your input data)
DEPENDENT_VARIABLE = "isFPD_wRI"

# Which value represents "bad" outcome (usually "1" for binary 0/1)
TARGET_CATEGORY = "1"

# Select preset for algorithm/enhancement settings
PRESET = "fraud_enhanced"  # Options: "r_compatible", "credit_scoring", "fraud_model", "fraud_enhanced", "custom"

# =============================================================================
# Preset Configurations
# =============================================================================

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

# =============================================================================
# Generate Configuration Table
# =============================================================================

# Get selected preset
if PRESET not in PRESETS:
    print(f"Warning: Unknown preset '{PRESET}', using 'r_compatible'")
    PRESET = "r_compatible"

config = PRESETS[PRESET].copy()

# Apply user settings (override preset values)
config["DependentVariable"] = DEPENDENT_VARIABLE
config["TargetCategory"] = TARGET_CATEGORY

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

df = pd.DataFrame([{col: config[col] for col in column_order}])

# =============================================================================
# Output
# =============================================================================

knio.output_tables[0] = knio.Table.from_pandas(df)

print("=" * 70)
print(f"WOE Editor Advanced Configuration")
print("=" * 70)
print(f"\nPreset: {PRESET}")
print(f"Dependent Variable: {DEPENDENT_VARIABLE}")
print(f"Target Category: {TARGET_CATEGORY}")
print(f"\nGenerated {len(df.columns)} flow variable columns:\n")

# Group by category for display
categories = {
    "Core Settings": ["DependentVariable", "TargetCategory", "OptimizeAll", "GroupNA"],
    "Algorithm": ["Algorithm", "MinBinPct", "MinBinCount", "MaxBins"],
    "WOE Options": ["UseShrinkage", "ShrinkageStrength"],
    "Enhancements": ["UseEnhancements", "AdaptiveMinProp", "MinEventCount", 
                     "AutoRetry", "ChiSquareValidation", "SingleBinProtection"],
}

for category, cols in categories.items():
    print(f"  {category}:")
    for col in cols:
        val = df[col].iloc[0]
        print(f"    {col}: {val}")
    print()

print("=" * 70)
print("NEXT STEP: Connect output to 'Table Row to Variables' node")
print("=" * 70)

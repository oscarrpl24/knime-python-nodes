# =============================================================================
# Model Performance Monitor - Configuration Preset Generator
# =============================================================================
#
# This script generates KNIME flow variable tables with preset configurations
# for different monitoring scenarios. Similar to woe_config_generator.py.
#
# Usage in KNIME:
# 1. Configure the PRESET at the top of this script
# 2. Run as Python Script node (no inputs required)
# 3. Connect output to Model Performance Monitor node
# 4. Monitor node will read configuration from input table
#
# Or use the generated configurations as a template for setting up
# your own flow variables manually.
#
# Release Date: 2026-01-28
# Version: 1.0
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Choose a preset configuration
# Options: "conservative", "standard", "aggressive", "early_warning", 
#          "stable_environment", "volatile_market", "custom"
PRESET = "standard"

# Required: Column names (update to match your data)
DEPENDENT_VARIABLE = "isBad"        # Name of your outcome column
SCORE_COLUMN = "score"              # Name of score column
APPROVAL_COLUMN = "isApproved"      # Name of approval column (1=approved, 0=declined)
FUNDED_COLUMN = "isFunded"          # Name of funded column (1=funded, 0=not funded)
ROI_COLUMN = "ROI"                  # Name of ROI column

# Scorecard parameters (MUST match your Scorecard Generator node)
SCORECARD_POINTS = 600              # Base score at target odds
SCORECARD_ODDS = 20                 # Target odds ratio (1:X, e.g., 20 = 1:19 odds)
SCORECARD_PDO = 50                  # Points to Double the Odds

# Custom threshold values (only used if PRESET = "custom")
CUSTOM_PSI_WARNING = 0.1
CUSTOM_PSI_CRITICAL = 0.25
CUSTOM_AUC_DECLINE_WARNING = 0.03
CUSTOM_AUC_DECLINE_CRITICAL = 0.05
CUSTOM_KS_DECLINE_WARNING = 0.05
CUSTOM_KS_DECLINE_CRITICAL = 0.10
CUSTOM_BADRATE_INCREASE_WARNING = 0.02
CUSTOM_CALIBRATION_ERROR_WARNING = 0.05
CUSTOM_MIN_SAMPLE_SIZE = 500

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

PRESETS = {
    # =========================================================================
    # CONSERVATIVE: Relaxed thresholds, fewer false alarms
    # Use when: Model is mature, environment is stable, retraining is costly
    # =========================================================================
    "conservative": {
        "description": "Relaxed thresholds for stable environments",
        "use_case": "Mature models in stable markets with high retraining cost",
        "PSI_Warning": 0.15,           # Higher threshold (allow more drift)
        "PSI_Critical": 0.35,          # Much higher (only critical drift)
        "AUC_Decline_Warning": 0.05,   # 5% decline before warning
        "AUC_Decline_Critical": 0.08,  # 8% decline before critical
        "KS_Decline_Warning": 0.08,    # More tolerant of K-S decline
        "KS_Decline_Critical": 0.15,   # Significant decline only
        "BadRate_Increase_Warning": 0.03,  # 3% increase allowed
        "CalibrationError_Warning": 0.08,  # More tolerance for miscalibration
        "MinSampleSize": 300,          # Lower minimum (more permissive)
        "rationale": "Avoids retraining unless truly necessary. Good for stable "
                    "industries like prime mortgages or established credit cards."
    },
    
    # =========================================================================
    # STANDARD: Balanced approach (DEFAULT)
    # Use when: General credit scoring, typical business environment
    # =========================================================================
    "standard": {
        "description": "Balanced thresholds for typical credit scoring",
        "use_case": "General purpose monitoring, most credit applications",
        "PSI_Warning": 0.1,            # Standard PSI thresholds
        "PSI_Critical": 0.25,          # Industry standard
        "AUC_Decline_Warning": 0.03,   # 3% decline triggers monitoring
        "AUC_Decline_Critical": 0.05,  # 5% decline triggers retraining
        "KS_Decline_Warning": 0.05,    # 5% K-S decline
        "KS_Decline_Critical": 0.10,   # 10% K-S decline
        "BadRate_Increase_Warning": 0.02,  # 2% bad rate increase
        "CalibrationError_Warning": 0.05,  # 5% calibration error
        "MinSampleSize": 500,          # Standard minimum
        "rationale": "Well-tested thresholds based on industry best practices. "
                    "Balances early detection with avoiding false alarms."
    },
    
    # =========================================================================
    # AGGRESSIVE: Strict thresholds, catch issues early
    # Use when: High risk tolerance, can retrain frequently, volatile market
    # =========================================================================
    "aggressive": {
        "description": "Strict thresholds for early issue detection",
        "use_case": "High-risk lending, volatile markets, frequent model updates",
        "PSI_Warning": 0.07,           # Lower threshold (catch drift early)
        "PSI_Critical": 0.15,          # Much stricter
        "AUC_Decline_Warning": 0.02,   # 2% decline triggers warning
        "AUC_Decline_Critical": 0.03,  # 3% decline triggers retraining
        "KS_Decline_Warning": 0.03,    # Very sensitive to K-S changes
        "KS_Decline_Critical": 0.06,   # Quick to recommend retraining
        "BadRate_Increase_Warning": 0.01,  # 1% bad rate increase
        "CalibrationError_Warning": 0.03,  # Tight calibration requirements
        "MinSampleSize": 1000,         # Higher minimum for reliability
        "rationale": "Catches problems early before they become severe. Good for "
                    "subprime lending, fintech, or when you can retrain frequently."
    },
    
    # =========================================================================
    # EARLY WARNING: Very sensitive, maximum protection
    # Use when: Cannot afford model failures, regulatory scrutiny, initial deployment
    # =========================================================================
    "early_warning": {
        "description": "Maximum sensitivity for critical applications",
        "use_case": "Regulatory scrutiny, high-stakes decisions, new models",
        "PSI_Warning": 0.05,           # Extremely low threshold
        "PSI_Critical": 0.10,          # Very strict
        "AUC_Decline_Warning": 0.01,   # 1% decline triggers warning
        "AUC_Decline_Critical": 0.02,  # 2% decline triggers retraining
        "KS_Decline_Warning": 0.02,    # Highly sensitive
        "KS_Decline_Critical": 0.04,   # Quick action required
        "BadRate_Increase_Warning": 0.005,  # 0.5% increase
        "CalibrationError_Warning": 0.02,   # Very tight calibration
        "MinSampleSize": 1500,         # Large sample for confidence
        "rationale": "Maximum protection against model failure. Use when stakes are "
                    "high and you can't afford to miss early warning signs."
    },
    
    # =========================================================================
    # STABLE ENVIRONMENT: Minimal false alarms for mature, stable models
    # Use when: Long-established model, very stable population, quarterly monitoring
    # =========================================================================
    "stable_environment": {
        "description": "Minimal false alarms for stable, mature models",
        "use_case": "Established models (5+ years), stable industries, quarterly reviews",
        "PSI_Warning": 0.18,           # Very high tolerance
        "PSI_Critical": 0.40,          # Extreme drift only
        "AUC_Decline_Warning": 0.06,   # 6% decline before warning
        "AUC_Decline_Critical": 0.10,  # 10% decline before critical
        "KS_Decline_Warning": 0.10,    # High tolerance
        "KS_Decline_Critical": 0.20,   # Severe degradation only
        "BadRate_Increase_Warning": 0.04,  # 4% increase
        "CalibrationError_Warning": 0.10,  # High tolerance for miscalibration
        "MinSampleSize": 200,          # Lower minimum
        "rationale": "For models that have proven stable over years. Reduces monitoring "
                    "overhead while catching truly critical issues. Good for mature portfolios."
    },
    
    # =========================================================================
    # VOLATILE MARKET: Adaptive to rapid changes
    # Use when: Economic uncertainty, new product launch, market disruption
    # =========================================================================
    "volatile_market": {
        "description": "Adaptive monitoring for changing environments",
        "use_case": "Economic uncertainty, product changes, market disruption, pandemic",
        "PSI_Warning": 0.12,           # Moderate - expect some change
        "PSI_Critical": 0.20,          # Still catch significant drift
        "AUC_Decline_Warning": 0.04,   # Allow some performance variation
        "AUC_Decline_Critical": 0.06,  # But not too much degradation
        "KS_Decline_Warning": 0.06,    # Moderate sensitivity
        "KS_Decline_Critical": 0.12,   # Catch major issues
        "BadRate_Increase_Warning": 0.025, # 2.5% increase
        "CalibrationError_Warning": 0.06,  # Some miscalibration expected
        "MinSampleSize": 400,          # Moderate minimum
        "rationale": "Balanced for changing environments. Allows adaptation to new "
                    "normal while catching sustained degradation. Good during COVID-19, "
                    "regulatory changes, or major product launches."
    },
    
    # =========================================================================
    # CUSTOM: User-defined thresholds
    # Use when: You have specific requirements or want fine-tuned control
    # =========================================================================
    "custom": {
        "description": "User-defined custom thresholds",
        "use_case": "Specific business requirements, fine-tuned monitoring",
        "PSI_Warning": CUSTOM_PSI_WARNING,
        "PSI_Critical": CUSTOM_PSI_CRITICAL,
        "AUC_Decline_Warning": CUSTOM_AUC_DECLINE_WARNING,
        "AUC_Decline_Critical": CUSTOM_AUC_DECLINE_CRITICAL,
        "KS_Decline_Warning": CUSTOM_KS_DECLINE_WARNING,
        "KS_Decline_Critical": CUSTOM_KS_DECLINE_CRITICAL,
        "BadRate_Increase_Warning": CUSTOM_BADRATE_INCREASE_WARNING,
        "CalibrationError_Warning": CUSTOM_CALIBRATION_ERROR_WARNING,
        "MinSampleSize": CUSTOM_MIN_SAMPLE_SIZE,
        "rationale": "Custom configuration based on your specific needs. Edit the "
                    "CUSTOM_* variables at the top of this script."
    }
}


# =============================================================================
# Configuration Generator Functions
# =============================================================================

def generate_config_table(preset_name: str) -> pd.DataFrame:
    """
    Generate a configuration table for the specified preset.
    
    Returns a DataFrame with two columns: VariableName and Value
    This can be read by the Model Performance Monitor node.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    # Create configuration table
    config_data = []
    
    # Add description and metadata
    config_data.append({"VariableName": "PresetName", "Value": preset_name})
    config_data.append({"VariableName": "PresetDescription", "Value": preset["description"]})
    config_data.append({"VariableName": "PresetUseCase", "Value": preset["use_case"]})
    config_data.append({"VariableName": "PresetRationale", "Value": preset["rationale"]})
    
    # Add required column names
    config_data.append({"VariableName": "DependentVariable", "Value": DEPENDENT_VARIABLE})
    config_data.append({"VariableName": "ScoreColumn", "Value": SCORE_COLUMN})
    config_data.append({"VariableName": "ApprovalColumn", "Value": APPROVAL_COLUMN})
    config_data.append({"VariableName": "FundedColumn", "Value": FUNDED_COLUMN})
    config_data.append({"VariableName": "ROIColumn", "Value": ROI_COLUMN})
    
    # Add scorecard parameters
    config_data.append({"VariableName": "Points", "Value": SCORECARD_POINTS})
    config_data.append({"VariableName": "Odds", "Value": SCORECARD_ODDS})
    config_data.append({"VariableName": "PDO", "Value": SCORECARD_PDO})
    
    # Add threshold parameters
    config_data.append({"VariableName": "PSI_Warning", "Value": preset["PSI_Warning"]})
    config_data.append({"VariableName": "PSI_Critical", "Value": preset["PSI_Critical"]})
    config_data.append({"VariableName": "AUC_Decline_Warning", "Value": preset["AUC_Decline_Warning"]})
    config_data.append({"VariableName": "AUC_Decline_Critical", "Value": preset["AUC_Decline_Critical"]})
    config_data.append({"VariableName": "KS_Decline_Warning", "Value": preset["KS_Decline_Warning"]})
    config_data.append({"VariableName": "KS_Decline_Critical", "Value": preset["KS_Decline_Critical"]})
    config_data.append({"VariableName": "BadRate_Increase_Warning", "Value": preset["BadRate_Increase_Warning"]})
    config_data.append({"VariableName": "CalibrationError_Warning", "Value": preset["CalibrationError_Warning"]})
    config_data.append({"VariableName": "MinSampleSize", "Value": preset["MinSampleSize"]})
    
    # Create DataFrame
    config_df = pd.DataFrame(config_data)
    
    return config_df


def print_preset_comparison():
    """Print a comparison table of all presets."""
    print("\n" + "="*100)
    print("PRESET CONFIGURATION COMPARISON")
    print("="*100)
    
    # Create comparison table
    comparison_data = []
    
    for preset_name, preset in PRESETS.items():
        if preset_name == "custom":
            continue  # Skip custom as values may not be set
        
        comparison_data.append({
            "Preset": preset_name,
            "PSI_Warn": preset["PSI_Warning"],
            "PSI_Crit": preset["PSI_Critical"],
            "AUC_Warn": preset["AUC_Decline_Warning"],
            "AUC_Crit": preset["AUC_Decline_Critical"],
            "BadRate": preset["BadRate_Increase_Warning"],
            "MinSize": preset["MinSampleSize"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("LEGEND:")
    print("  PSI_Warn/Crit: Population Stability Index thresholds")
    print("  AUC_Warn/Crit: AUC decline thresholds")
    print("  BadRate: Bad rate increase threshold")
    print("  MinSize: Minimum funded loans required")
    print("="*100)


def print_preset_details(preset_name: str):
    """Print detailed information about a specific preset."""
    if preset_name not in PRESETS:
        print(f"ERROR: Unknown preset '{preset_name}'")
        return
    
    preset = PRESETS[preset_name]
    
    print("\n" + "="*70)
    print(f"PRESET: {preset_name.upper()}")
    print("="*70)
    print(f"\nDescription: {preset['description']}")
    print(f"\nUse Case: {preset['use_case']}")
    print(f"\nRationale: {preset['rationale']}")
    
    print("\n" + "-"*70)
    print("THRESHOLD VALUES:")
    print("-"*70)
    
    thresholds = [
        ("PSI Warning", preset["PSI_Warning"], "Population drift monitoring threshold"),
        ("PSI Critical", preset["PSI_Critical"], "Population drift retraining threshold"),
        ("AUC Decline Warning", preset["AUC_Decline_Warning"], "Model discrimination warning"),
        ("AUC Decline Critical", preset["AUC_Decline_Critical"], "Model discrimination retraining"),
        ("K-S Decline Warning", preset["KS_Decline_Warning"], "K-S statistic warning"),
        ("K-S Decline Critical", preset["KS_Decline_Critical"], "K-S statistic retraining"),
        ("Bad Rate Increase", preset["BadRate_Increase_Warning"], "Bad rate increase threshold"),
        ("Calibration Error", preset["CalibrationError_Warning"], "Calibration error threshold"),
        ("Min Sample Size", preset["MinSampleSize"], "Minimum funded loans required")
    ]
    
    for name, value, description in thresholds:
        print(f"  {name:25s}: {value:6} - {description}")
    
    print("="*70)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate configuration table and display information."""
    
    print("="*70)
    print("MODEL PERFORMANCE MONITOR - CONFIGURATION GENERATOR")
    print("="*70)
    
    # Print comparison of all presets
    print_preset_comparison()
    
    # Print details of selected preset
    print_preset_details(PRESET)
    
    # Generate configuration table
    print(f"\nGenerating configuration for preset: {PRESET}")
    config_df = generate_config_table(PRESET)
    
    # Output to KNIME or CSV
    try:
        import knime.scripting.io as knio
        knio.output_tables[0] = knio.Table.from_pandas(config_df)
        print("\n✓ Configuration exported to KNIME output table")
        print(f"  {len(config_df)} flow variables configured")
    except:
        # Save to CSV if not in KNIME
        filename = f"monitor_config_{PRESET}.csv"
        config_df.to_csv(filename, index=False)
        print(f"\n✓ Configuration saved to: {filename}")
    
    print("\n" + "="*70)
    print("USAGE IN KNIME:")
    print("="*70)
    print("\nOption 1: Table Row to Variable Loop")
    print("  1. Connect this node's output to 'Table Row to Variable Loop Start'")
    print("  2. In the loop, connect to Model Performance Monitor")
    print("  3. Variables will be automatically set")
    print("\nOption 2: Manual Flow Variables")
    print("  1. Review the generated table")
    print("  2. Manually set flow variables in your workflow")
    print("  3. Use values from the 'Value' column")
    print("\nOption 3: Export and Reference")
    print("  1. Use generated CSV as reference")
    print("  2. Set up flow variables based on these values")
    print("  3. Adjust as needed for your use case")
    
    print("\n" + "="*70)
    print("Configuration generation complete!")
    print("="*70)
    
    return config_df


if __name__ == "__main__":
    # Execute main function
    config_df = main()
    
    # Display sample configuration
    print("\nSample configuration (first 15 rows):")
    print(config_df.head(15).to_string(index=False))


# =============================================================================
# PRESET SELECTION GUIDE
# =============================================================================
#
# Choose your preset based on your business context:
#
# 1. CONSERVATIVE - "stable_environment"
#    When: Mature model (5+ years), prime lending, stable economy
#    Risk: May miss early warning signs
#    Benefit: Avoids unnecessary retraining costs
#
# 2. STANDARD - "standard" (RECOMMENDED FOR MOST)
#    When: Typical credit scoring, general purpose
#    Risk: Balanced approach
#    Benefit: Industry best practices, well-tested thresholds
#
# 3. AGGRESSIVE - "aggressive" or "early_warning"
#    When: Subprime lending, new model, volatile market, high regulatory scrutiny
#    Risk: May trigger false alarms
#    Benefit: Catches issues early before they become severe
#
# 4. ADAPTIVE - "volatile_market"
#    When: Economic uncertainty, product changes, market disruption
#    Risk: May allow too much drift in stable periods
#    Benefit: Balanced for changing environments
#
# 5. CUSTOM - "custom"
#    When: Specific requirements, regulatory mandates, unique use case
#    Risk: Requires expertise to set appropriate thresholds
#    Benefit: Fine-tuned control for your exact needs
#
# =============================================================================
# 
# THRESHOLD EVOLUTION STRATEGY
# =============================================================================
#
# Month 1-3: Start with "standard" or "conservative"
#   - Learn your baseline variation
#   - Understand false alarm rate
#   - Build confidence in metrics
#
# Month 4-6: Adjust based on experience
#   - If too many false alarms: Move to "conservative" or "stable_environment"
#   - If missed issues: Move to "aggressive" or "early_warning"
#   - Fine-tune with "custom" preset
#
# Month 7+: Establish your optimal preset
#   - Document why you chose specific thresholds
#   - Review quarterly
#   - Adjust for changing business environment
#
# =============================================================================

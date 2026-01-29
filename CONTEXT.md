# KNIME Python Nodes - Project Context

## Quick Start for AI Agents

This document provides context for AI coding assistants working on this project.

---

## Project Purpose

Build Python scripts that run inside **KNIME 5.9 Python Script nodes** for credit risk modeling workflows. Scripts handle:
1. **Reject Inference** - Inferring outcomes for rejected loan applications
2. **WOE Binning** - Weight of Evidence transformation for logistic regression
3. **Attribute Editor** - Variable metadata configuration and type classification
4. **Variable Selection** - Feature selection with EBM interaction discovery
5. **Model Analyzer** - ROC curves, K-S charts, gains tables for model diagnostics
6. **Logistic Regression** - Logistic regression with stepwise variable selection (like R's stepAIC)
7. **Scorecard Generator** - Convert logistic regression + WOE bins into credit scorecard
8. **Model Performance Monitor** - Production model monitoring with drift detection and retraining recommendations

---

## Technical Stack

| Component | Version/Details |
|-----------|-----------------|
| KNIME | 5.9 |
| Python | 3.9.23 |
| Key Libraries | pandas, numpy, scikit-learn, shiny (for Python) |
| OS | Windows |

---

## File Inventory

### `reject_inference.py`
- **Status**: Working
- **Inputs**: 1 table
- **Outputs**: 1 table (original + new columns)
- **New Columns Created**:
  - `isFPD_wRI` - placed after `IsFPD`
  - `FRODI26_wRI` - placed after `FRODI26`
  - `GRODI26_wRI` - placed after `GRODI26`

### `woe_editor_knime.py`
- **Status**: Complete (base version)
- **Inputs**: 1 table + optional flow variables
- **Outputs**: 4 tables
- **Features**:
  - Shiny for Python interactive UI
  - Headless batch processing mode
  - Replaces R's logiBin package functionality

### `woe_editor_advanced.py` (RECOMMENDED)
- **Status**: Complete (v1.4)
- **Inputs**: 1 table + optional flow variables
- **Outputs**: 4 tables
- **Algorithm Options**:
  | Algorithm | Description |
  |-----------|-------------|
  | `DecisionTree` | R-compatible (matches logiBin::getBins) - DEFAULT |
  | `ChiMerge` | Chi-square based bin merging |
  | `IVOptimal` | **NEW** - Maximizes IV, allows non-monotonic patterns |
- **Key Flow Variables**:
  - `DependentVariable` (string): Target column name (triggers headless mode)
  - `Algorithm` (string): "DecisionTree", "ChiMerge", or "IVOptimal"
  - `MinBinPct` (float): Minimum bin proportion (default 0.02)
  - `MaxBins` (int): Maximum bins per variable (default 10)
  - `OptimizeAll` (int): Force monotonicity (0=no, 1=yes)
  - `UseEnhancements` (boolean): Enable all enhancements
  - Individual flags: `AdaptiveMinProp`, `MinEventCount`, `AutoRetry`, `ChiSquareValidation`, `SingleBinProtection`
- **IVOptimal Algorithm** (new in v1.4):
  - Directly maximizes Information Value
  - Does NOT enforce monotonicity - allows "sweet spots"
  - Dynamic starting granularity based on variable characteristics
  - Merges pure bins to closest adjacent by WOE value
  - Best for fraud detection with non-monotonic patterns
- **Documentation**: See `woe_editor/WOE_Editor_Enhancements.md`

### `woe_config_generator.py` (Utility)
- **Location**: `woe_editor/woe_config_generator.py`
- **Status**: Complete (v1.1)
- **Purpose**: Generates KNIME table with all flow variables for WOE Editor Advanced
- **Presets**:
  | Preset | Use Case |
  |--------|----------|
  | `r_compatible` | Match R logiBin exactly |
  | `credit_scoring` | Standard credit scoring |
  | `fraud_model` | Low event rate optimization |
  | `fraud_enhanced` | All DecisionTree enhancements |
  | `iv_optimal` | IV-maximizing, allows sweet spots |
  | `custom` | Define your own settings |
- **Usage**: Set `DEPENDENT_VARIABLE`, `TARGET_CATEGORY`, and `PRESET` at top of script

### `attribute_editor_knime.py`
- **Status**: Complete (needs testing)
- **Inputs**: 1 table + optional flow variables
- **Outputs**: 2 tables
  1. Variable metadata (VariableName, Include, Role, Usage, etc.)
  2. Converted data (type conversions applied, excluded columns removed)
- **Flow Variables**:
  - `DependentVariable` (string): Target column name
  - `VarOverride` (integer): If 1, force interactive mode
- **Features**:
  - Shiny for Python interactive UI for editing variable attributes
  - Headless mode when DependentVariable is set (and VarOverride ≠ 1)
  - Auto-detects variable types (continuous, discrete, nominal)
  - Applies type conversions: nominal→string, continuous→float, discrete→int
  - Removes columns where Include == False

### `logistic_regression_knime.py`
- **Status**: Complete (needs testing)
- **Inputs**: 1 table (typically WOE-transformed data from Variable Selection)
- **Outputs**: 2 tables
  1. Input data with predictions added:
     - `probabilities` (Float): Predicted probabilities, rounded to 6 decimals
     - `predicted` (String): "1" or "0" based on 0.5 threshold
  2. Model coefficients table:
     - `Variable`: Variable name (including "(Intercept)")
     - `coefficients`: Coefficient value
- **Flow Variables** (for headless mode):
  - `DependentVariable` (string): Binary target variable name
  - `TargetCategory` (string, optional): Which value represents "bad" outcome
  - `VarSelectionMethod` (string): "All", "Stepwise", "Forward", "Backward"
  - `Cutoff` (float, default 2): AIC penalty multiplier (k=2 for AIC, k=log(n) for BIC)
- **Features**:
  - Shiny for Python interactive UI with modern dark theme
  - Logistic regression using statsmodels
  - Stepwise variable selection (Forward, Backward, Both) equivalent to R's `MASS::stepAIC`
  - Automatic WOE variable detection (prefers `WOE_*` prefixed columns)
  - Coefficient table with odds ratios, p-values, confidence intervals
  - ROC curve and coefficient plots in UI
  - Model statistics: AUC, Gini, Pseudo-R², confusion matrix metrics
- **R Equivalent**:
  ```r
  # This Python script replaces this R code:
  full.model <- glm(DV ~ ., data = df, family = binomial)
  model <- stepAIC(full.model, direction = "both", k = 2)
  
  # Output 1: Data with predictions
  df$probabilities <- predict(model, df, type = "response") %>% round(6)
  df$predicted <- ifelse(df$probabilities > 0.5, "1", "0")
  knime.out <- df
  
  # Output 2: Coefficients
  knime.out <- as.data.frame(model$coefficients)
  ```
- **Usage Notes**:
  - In headless mode, automatically selects WOE_ prefixed variables if present
  - Excludes b_ prefixed variables (binned, not for regression)
  - Cutoff parameter: k=2 gives AIC selection, k=log(n) gives BIC selection

### `model_analyzer_knime.py`
- **Status**: Complete (needs testing)
- **Inputs**: 3 tables
  1. **Training data** (required): WOE features + DV + `predicted` (0/1) + `probabilities` (log-odds)
  2. **Coefficients table** (required): Row ID = variable names, column = coefficient values
  3. **Test data** (optional): Same WOE features + DV (predictions computed from coefficients)
- **Outputs**: 3 tables
  1. Combined data with predictions (training + test, with `probability` and `dataset` columns)
  2. Gains table (decile analysis)
  3. Model performance metrics (AUC, Gini, K-S, accuracy, sensitivity, specificity)
- **Required Input Columns (Training)**:
  - `probabilities` (Float): Log-odds / linear predictor from R model (auto-converted to probability)
  - `predicted` (Int): Predicted class (0/1)
  - Dependent variable column (actual outcomes)
- **Coefficients Table Format** (from R):
  - Row ID: Variable names (including "(Intercept)")
  - Column: `model$coefficients` or any numeric column with coefficient values
- **Flow Variables** (for headless mode):
  - `DependentVariable` (string): Target column name
  - `ProbabilitiesColumn` (string): Column with log-odds (default: "probabilities")
  - `AnalyzeDataset` (string): "Training", "Test", or "Both"
  - `ModelName` (string): Name for saved chart files
  - `FilePath` (string): Directory to save chart images
  - `saveROC`, `saveCaptureRate`, `saveK-S`, `saveLorenzCurve`, `saveDecileLift` (int): 1 to save
- **Features**:
  - **Automatic log-odds to probability conversion** using sigmoid function
  - **Test data predictions** computed from coefficients table (no R model needed)
  - Shiny for Python interactive UI with modern dark theme
  - ROC curve with AUC and Gini index
  - K-S (Kolmogorov-Smirnov) chart
  - Lorenz curve
  - Decile Lift chart
  - Capture Rate by Decile chart
  - Headless mode saves charts as JPEG files
- **R Integration**:
  Add these columns in R before passing training data to Python:
  ```r
  # Training data preparation
  df_train$predicted <- as.integer(predict(model, newdata = df_train, type = "response") > 0.5)
  df_train$probabilities <- predict(model, newdata = df_train, type = "link")  # log-odds
  
  # Coefficients table (Input 2)
  coef_df <- data.frame(row.names = names(model$coefficients), 
                        coefficient = model$coefficients)
  ```

### `model_performance_monitor.py`
- **Status**: Complete (v1.0 - newly created)
- **Inputs**: 2 tables
  1. **Production Data** (required): Current period loan performance
     - `score`: Scorecard points (integer)
     - `isApproved`: 1=approved, 0=declined
     - `isFunded`: 1=funded, 0=not funded
     - `<DependentVariable>`: Actual outcome 0/1 (only for funded loans)
     - `ROI`: Return on investment (>1=profit, <1=loss, only for funded loans)
  2. **Baseline Data** (optional): Training data or stable historical period
     - Same structure as Production Data
     - Used for PSI calculation and performance comparison
- **Outputs**: 4 tables
  1. Performance Summary - Key metrics with alerts (AUC, Gini, K-S, PSI, bad rate, ROI, Recommendation)
  2. Decile Analysis - Performance by score decile
  3. Calibration Table - Expected vs observed bad rates by score band
  4. Production Data with Diagnostics - Original data + probability, prediction_correct, score_decile
- **Flow Variables** (for headless mode):
  - `DependentVariable` (string, required): Name of actual outcome column
  - `ScoreColumn` (string, default "score"): Name of score column
  - `ApprovalColumn` (string, default "isApproved"): Name of approval column
  - `FundedColumn` (string, default "isFunded"): Name of funded column
  - `ROIColumn` (string, default "ROI"): Name of ROI column
  - Scorecard Parameters (for probability calculation):
    - `Points` (int, default 600): Base score at target odds
    - `Odds` (int, default 20): Target odds ratio (1:X)
    - `PDO` (int, default 50): Points to Double the Odds
  - Alert Thresholds:
    - `PSI_Warning` (float, default 0.1): PSI threshold for "MONITOR"
    - `PSI_Critical` (float, default 0.25): PSI threshold for "RETRAIN"
    - `AUC_Decline_Warning` (float, default 0.03): AUC decline for "MONITOR"
    - `AUC_Decline_Critical` (float, default 0.05): AUC decline for "RETRAIN"
    - `KS_Decline_Warning` (float, default 0.05): K-S decline for "MONITOR"
    - `KS_Decline_Critical` (float, default 0.10): K-S decline for "RETRAIN"
    - `BadRate_Increase_Warning` (float, default 0.02): Bad rate increase threshold
    - `CalibrationError_Warning` (float, default 0.05): Calibration error threshold
    - `MinSampleSize` (int, default 500): Minimum funded loans for reliable metrics
- **Key Metrics Calculated**:
  - **Discrimination**: AUC, Gini, K-S statistic
  - **Population Stability**: PSI (Population Stability Index)
  - **Calibration**: Expected vs observed rates by score band, calibration error
  - **Business**: Approval rate, bad rate (funded), average ROI, loss rate
  - **Recommendation**: OK, MONITOR, or RETRAIN based on threshold logic
- **Features**:
  - Shiny for Python interactive UI with threshold adjustment
  - Score-to-probability conversion using inverse scorecard formula
  - Automatic recommendation logic based on configurable thresholds
  - Decile-level performance analysis
  - Calibration analysis across probability bins
  - Sample size warnings for small datasets
  - Baseline comparison for drift detection
- **Score-to-Probability Formula**:
  ```python
  # Given scorecard parameters (Points, Odds, PDO):
  b = PDO / log(2)
  a = Points + b * log(odds0)
  log_odds = (a - score) / b
  probability = 1 / (1 + exp(-log_odds))
  ```
- **Recommendation Logic**:
  - **OK**: PSI < 0.1, AUC decline < 3%, K-S decline < 5%, calibration error < 5%
  - **MONITOR**: PSI 0.1-0.25, AUC decline 3-5%, K-S decline 5-10%, bad rate increase ≥ 2%
  - **RETRAIN**: PSI ≥ 0.25, AUC decline ≥ 5%, K-S decline ≥ 10%
- **Usage Notes**:
  - Provide baseline data for PSI and comparative metrics
  - Without baseline: calculates absolute metrics only (no drift detection)
  - Requires at least 500 funded loans for reliable metrics
  - Scorecard parameters must match those used in Scorecard Generator node
- **Documentation**: See `model_performance_monitor/README.md` and `BASELINE_STRATEGY.md`

### `scorecard_knime.py`
- **Status**: Complete (needs testing)
- **Inputs**: 2 tables
  1. **Coefficients table** (from Logistic Regression node Output 2):
     - Row ID = variable name (e.g., "(Intercept)", "WOE_Age")
     - Column = coefficient value
  2. **Bins table** (from WOE Editor node Output 4):
     - `var`, `bin`, `binValue`, `woe` columns required
- **Outputs**: 2 tables
  1. Scorecard as Table format (all bins with points in single table)
  2. Scorecard as List format (same data, for R compatibility)
- **Scorecard Output Columns**:
  - `var` (String): Variable name or "basepoints"
  - `bin` (String): Bin label/range
  - `woe` (Float): Weight of Evidence value
  - `points` (Int): Points assigned to bin
- **Flow Variables** (for headless mode):
  - `Points` (int, default 600): Base score at target odds
  - `Odds` (int, default 20): Odds ratio (1:X, e.g., 20 means 1:19 odds)
  - `PDO` (int, default 50): Points to Double the Odds
  - `OutputFormat` (string, default "Table"): "Table" or "List"
- **Scorecard Formula**:
  ```
  b = PDO / log(2)
  a = Points + b * log(1/(Odds-1))
  basepoints = a - b * intercept_coefficient
  bin_points = round(-b * coefficient * woe, 0)
  ```
- **Features**:
  - Shiny for Python interactive UI with gold/amber theme
  - Parameter configuration: Points, Odds, PDO
  - Summary statistics: number of variables, total bins, min/max score
  - Only includes variables that are in both bins table AND coefficients
  - Headless mode for batch processing
- **R Equivalent**:
  ```r
  # This Python script replaces this R scorecard function:
  ab <- function(points0=600, odds0=1/19, pdo=50) {
    b <- pdo/log(2)
    a <- points0 + b*log(odds0)
    return(list(a=a, b=b))
  }
  
  createScorecard <- function(bins, model, points0, odds0, pdo) {
    aabb <- ab(points0, odds0, pdo)
    basepoints <- aabb$a - aabb$b * coef(model)[1]
    
    for (var in model_vars) {
      card[[var]]$points <- round(-aabb$b * coef * woe, 0)
    }
  }
  ```
- **Usage Notes**:
  - Connect Logistic Regression Output 2 to Input 1
  - Connect WOE Editor Output 4 to Input 2
  - Variables in bins table must match coefficient names (minus WOE_ prefix)

---

## KNIME I/O Patterns

```python
import knime.scripting.io as knio

# Read input table
df = knio.input_tables[0].to_pandas()

# Write output table
knio.output_tables[0] = knio.Table.from_pandas(df)

# Read flow variable with default
value = knio.flow_variables.get("VarName", default)
```

---

## Important Conventions

1. **Self-contained scripts**: No external module imports from project files
2. **Auto-install dependencies**: Use try/except with pip install
3. **Case-sensitive columns**: Column names must match exactly
4. **Nullable types**: Use `Int32`, `Int64`, `Float64` (capital I/F) for nullable numerics
5. **Column ordering**: New columns should be placed after their source columns

---

## Common Tasks

### Adding a New Column
```python
# Create column with nullable integer type
df['new_col'] = pd.array([pd.NA] * len(df), dtype='Int32')

# Populate based on conditions
df.loc[condition, 'new_col'] = value

# Reorder to place after source column
cols = df.columns.tolist()
cols.remove('new_col')
source_idx = cols.index('source_col')
cols.insert(source_idx + 1, 'new_col')
df = df[cols]
```

### Checking for Missing Values
```python
def is_missing(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    return False
```

---

## Pending/Future Work

### Testing
- [ ] Test woe_editor_advanced.py in KNIME (especially IVOptimal algorithm)
- [ ] Test attribute_editor_knime.py in KNIME
- [ ] Test scorecard_knime.py in KNIME
- [ ] Test logistic_regression_knime.py in KNIME
- [ ] Test model_analyzer_knime.py in KNIME

### WOE Editor Enhancements (see `woe_editor/WOE_Editor_Enhancements.md`)
- [ ] Smart SingleBinProtection - merge bins for max IV instead of skipping
- [ ] Isotonic regression smoothing for WOE values
- [ ] Population Stability Index (PSI) calculation

### Other
- [ ] Add more reject inference columns as needed
- [ ] Potential: Add model scoring scripts
- [ ] Potential: Add validation/testing scripts

---

## Original R Script Reference

### WOE Editor
The `woe_editor_knime.py` is a Python port of an R Shiny application that used:
- `logiBin` package for binning (`getBins`, `forceIncrTrend`, `forceDecrTrend`, `naCombine`)
- R Shiny for interactive UI
- `data.table` and `tidyverse` for data manipulation

Key function mappings:
| R (logiBin) | Python (this project) |
|-------------|----------------------|
| `getBins()` | `get_bins()` |
| `forceIncrTrend()` | `force_incr_trend()` |
| `forceDecrTrend()` | `force_decr_trend()` |
| `naCombine()` | `na_combine()` |
| `createBins()` | `create_binned_columns()` |

**Advanced Version** (`woe_editor_advanced.py`):
- Adds `ChiMerge` algorithm (chi-square based merging)
- Adds `IVOptimal` algorithm (IV-maximizing, non-monotonic)
- Individual enhancement flags for fine-grained control
- WOE shrinkage estimators for rare events
- Use `woe_config_generator.py` to generate flow variable tables

### Logistic Regression
The `logistic_regression_knime.py` is a Python port of an R Shiny application that used:
- `MASS::stepAIC` for stepwise variable selection
- `glm()` with `family = binomial` for logistic regression
- R Shiny for interactive UI with variable checkbox selection
- `shinydashboard`, `DT`, `plotly` for UI components

Key function mappings:
| R (MASS/stats) | Python (this project) |
|----------------|----------------------|
| `glm(family=binomial)` | `sm.Logit().fit()` |
| `stepAIC(direction="forward")` | `stepwise_forward()` |
| `stepAIC(direction="backward")` | `stepwise_backward()` |
| `stepAIC(direction="both")` | `stepwise_both()` |
| `summary(model)` | `model.summary()` |
| `predict(type="response")` | `model.predict()` |

Flow variable mapping from R to Python:
| R Flow Variable | Python Flow Variable |
|-----------------|---------------------|
| `DependentVariable` | `DependentVariable` |
| `TargetCategory` | `TargetCategory` |
| `selMethod` (Stepwise/Forward/Backward) | `VarSelectionMethod` |
| `k` (cutoff) | `Cutoff` |

### Attribute Editor
The `attribute_editor_knime.py` is a Python port of an R Shiny application that:
- Analyzes DataFrame columns to determine variable types
- Provides interactive editing of variable metadata via Shiny UI
- Outputs metadata for downstream binning/modeling nodes

Output columns:
| Column | Description |
|--------|-------------|
| `VariableName` | Column name from input |
| `Include` | TRUE/FALSE - whether to include variable |
| `Role` | "dependent" or "independent" |
| `Usage` | Type: continuous, nominal, ordinal, discrete, no binning |
| `UsageOriginal` | Original detected type |
| `UsageProposed` | System-suggested type |
| `NullQty` | Count of null/missing values |
| `min` / `max` | Min/max for numeric columns |
| `Cardinality` | Number of unique values |
| `Samples` | Sample unique values |
| `DefaultBins` | Suggested number of bins |
| `IntervalsType` | "static" or "" |
| `BreakApart` | "yes" or "no" |
| `MissingValues` | "use", "ignore", or "float" |
| `OrderedDisplay` | "range" or "present" |
| `PValue` | P-value threshold (default 0.05) |


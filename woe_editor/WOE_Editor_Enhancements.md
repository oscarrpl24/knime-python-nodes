# WOE Editor Enhancements Documentation

## Overview

This document details the implemented features and planned future enhancements for the WOE (Weight of Evidence) Editor Python node for KNIME. The WOE Editor is a Python port of the R Shiny WOE Editor application, designed to create optimal binning for credit scoring and predictive modeling.

**Last Updated:** 2026-01-26

**Script Versions:**
| Script | Version | Description |
|--------|---------|-------------|
| `woe_editor_knime.py` | v1.0 | Base version |
| `woe_editor_knime_parallel.py` | v2.0 | Parallel processing |
| `woe_editor_advanced.py` | **v1.4** | IVOptimal + individual enhancements |
| `woe_config_generator.py` | v1.1 | Config table generator |

*Note: This document tracks enhancements across all WOE editor scripts.*

---

## CONTEXT FOR FUTURE SESSIONS

### Recently Completed (2026-01-26):
- **IVOptimal Algorithm** (v1.4): New algorithm that maximizes IV directly, allows non-monotonic patterns
- **Individual Enhancement Flags** (v1.3): `AdaptiveMinProp`, `MinEventCount`, `AutoRetry`, `ChiSquareValidation`, `SingleBinProtection`
- **Config Generator** (v1.1): `woe_config_generator.py` with presets including `iv_optimal`

### Key Files:
| File | Version | Purpose |
|------|---------|---------|
| `woe_editor_advanced.py` | v1.4 | Main WOE editor with all algorithms |
| `woe_config_generator.py` | v1.1 | Generate flow variable tables |
| `WOE_Editor_Enhancements.md` | v2.3 | This documentation |

### Technical Notes:
- Headless mode triggers when `DependentVariable` flow variable matches a column in input data
- Boolean flow variables use string "true"/"false" format in config generator
- `UseEnhancements=True` acts as master switch enabling all individual enhancements
- IVOptimal algorithm: Set `OptimizeAll=0` to preserve non-monotonic patterns (sweet spots)

### Pending Enhancements:
- **Smart SingleBinProtection**: Merge bins to create 2 bins with highest IV instead of skipping
- **Isotonic regression smoothing**: Apply isotonic regression to WOE values post-binning
- **PSI calculation**: Population Stability Index for monitoring

---

## Part 1: Implemented Features

### 1.1 Core Binning Engine

| Feature | Status | Description |
|---------|--------|-------------|
| Decision Tree Binning | ✅ Complete | Uses CART-based DecisionTreeClassifier for optimal splits (R logiBin compatible) |
| WOE Calculation | ✅ Complete | `WOE = ln((% Bads) / (% Goods))` with handling for zero distributions |
| Information Value (IV) | ✅ Complete | Calculates IV for each variable and bin |
| Entropy Calculation | ✅ Complete | Weighted entropy for bins and totals |
| Trend Detection | ✅ Complete | Detects Increasing (I) or Decreasing (D) bad rate trends |
| Monotonic Trend Flag | ✅ Complete | Identifies if variable has consistent monotonic trend |
| Pure Node Detection | ✅ Complete | Flags bins with 100% goods or 100% bads |
| Flip Ratio | ✅ Complete | Measures proportion of trend direction changes |

### 1.2 Bin Operations

| Feature | Status | Description |
|---------|--------|-------------|
| Group NA (naCombine) | ✅ Complete | Merges NA bin with closest bin by bad rate |
| Force Increasing Trend | ✅ Complete | Combines adjacent bins to force increasing bad rate |
| Force Decreasing Trend | ✅ Complete | Combines adjacent bins to force decreasing bad rate |
| Break Bin | ✅ Complete | Separates factor variable into individual value bins |
| Reset Bin | ✅ Complete | Restores variable to original optimal bins |
| Optimize (Single Variable) | ✅ Complete | Applies appropriate monotonic trend to one variable |
| Optimize All | ✅ Complete | Applies Group NA + monotonic trends to all variables |
| Merge Pure Bins | ✅ Complete | Combines 100% good/bad bins with nearest bin to prevent infinite WOE |
| Manual Split (Numeric) | ✅ Complete | Create custom numeric variable splits at specified points |

### 1.3 User Interface (Shiny for Python)

| Feature | Status | Description |
|---------|--------|-------------|
| Dependent Variable Selection | ✅ Complete | Dropdown for target variable |
| Target Category Selection | ✅ Complete | Select which value represents "bad" outcome |
| Independent Variable Selection | ✅ Complete | Navigate through variables |
| Previous/Next Navigation | ✅ Complete | Browse variables sequentially |
| Bin Details Table | ✅ Complete | DataGrid showing bin, count, goods, bads, propn, bad_rate, WOE, IV |
| WOE & Bad Rate Chart | ✅ Complete | Plotly line chart with dual y-axes |
| Count Distribution Bar Chart | ✅ Complete | Bar chart with count, proportion, and WOE labels |
| Good/Bad Proportion Chart | ✅ Complete | Stacked horizontal bar chart |
| Measurements Table | ✅ Complete | Initial vs Final IV and Entropy comparison |
| Run & Close Button | ✅ Complete | Finalizes bins and closes UI |

### 1.4 Processing Modes

| Feature | Status | Description |
|---------|--------|-------------|
| Interactive Mode | ✅ Complete | Opens Shiny UI when no flow variables provided |
| Headless Mode | ✅ Complete | Automatic processing with flow variable configuration |
| Multi-Instance Support | ✅ Complete | Random port selection and process isolation for parallel KNIME execution |
| Progress Logging | ✅ Complete | Timestamped progress messages with ETA |

### 1.5 Flow Variables (Headless Mode)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DependentVariable` | String | Required | Binary target variable name |
| `TargetCategory` | String | Optional | Which value represents "bad" |
| `OptimizeAll` | Boolean | False | Force monotonic trends on all variables |
| `GroupNA` | Boolean | False | Combine NA bins with closest bin |

### 1.6 Output Tables

| Output | Description |
|--------|-------------|
| Output 1 | Original input DataFrame (unchanged) |
| Output 2 | `df_with_woe` - Original + binned columns (`b_*`) + WOE columns (`WOE_*`) |
| Output 3 | `df_only_woe` - Only WOE columns + dependent variable |
| Output 4 | `bins` - Binning rules with WOE values |

### 1.7 Parallel Processing (v2.0)

| Feature | Status | File |
|---------|--------|------|
| CPU Core Detection | ✅ Complete | `woe_editor_knime_parallel.py` |
| Parallel Variable Binning | ✅ Complete | Uses `joblib.Parallel` for initial binning |
| Parallel NA Combine | ✅ Complete | Multi-threaded NA bin merging |
| Parallel Trend Forcing | ✅ Complete | Multi-threaded monotonic optimization |
| Parallel Binned Column Creation | ✅ Complete | Multi-threaded `b_*` column generation |
| Parallel WOE Column Creation | ✅ Complete | Multi-threaded `WOE_*` column generation |

### 1.8 Advanced Binning (v1.2 - Optional)

| Feature | Status | Description |
|---------|--------|-------------|
| ChiMerge Algorithm | ✅ Complete | Chi-square based bin merging (alternative to Decision Tree) |
| **IVOptimal Algorithm** | ✅ Complete | IV-maximizing binning, allows non-monotonic patterns (v1.4) |
| R-Compatible Mode | ✅ Complete | Default behavior matches R logiBin::getBins exactly |
| Adaptive min_prop | ✅ Complete | Relaxes constraints for sparse data (optional enhancement) |
| Chi-square Validation | ✅ Complete | Merges statistically similar adjacent bins (optional) |
| Minimum Event Count | ✅ Complete | Ensures stable bins for rare events (optional) |
| Auto-retry with Relaxed Constraints | ✅ Complete | Fallback when initial binning fails |
| Shrinkage Estimators | ✅ Complete | Regularized WOE for rare events (optional) |
| Diagnostic Logging | ✅ Complete | Identifies single-bin and zero-WOE variables |

**Advanced Flow Variables:**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Algorithm` | String | "DecisionTree" | "DecisionTree" or "ChiMerge" |
| `MinBinPct` | Float | 0.01 | Minimum percentage per bin (1%) |
| `MinBinCount` | Integer | 20 | Minimum absolute count per bin |
| `UseShrinkage` | Boolean | False | Apply shrinkage to WOE values |
| `UseEnhancements` | Boolean | False | Master switch - enables ALL enhancements |

**Individual Enhancement Flow Variables (v1.3+):**

These allow fine-grained control over which enhancements are applied:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AdaptiveMinProp` | Boolean | False | Relax min_prop for sparse data (< 500 samples) |
| `MinEventCount` | Boolean | False | Ensure at least 5 events per potential bin |
| `AutoRetry` | Boolean | False | Retry with relaxed constraints if no splits found |
| `ChiSquareValidation` | Boolean | False | Merge statistically similar bins post-binning |
| `SingleBinProtection` | Boolean | True | Prevent na_combine from creating single-bin variables |

**SingleBinProtection Behavior (Current Implementation):**
- When enabled: **Skips NA grouping** for variables that would become single-bin (WOE=0)
- Result: Variable keeps NA as separate bin (2 bins total: NA + non-NA)
- Does NOT: Intelligently merge to maximize IV (this would be a future enhancement)

**Flow Variable Hierarchy:**
- If `UseEnhancements=True`, ALL individual enhancements are enabled automatically
- Individual flags can override the master switch when explicitly set
- `SingleBinProtection` defaults to True (protective) even when `UseEnhancements=False`

---

## Part 1.9: Enhancements from "Nulls Assigned Zero Values" Session

This section documents the enhancements discussed in the agent conversation "WOE editor nulls assigned zero values" (2026-01-26).

### Original Issue

**Problem**: When running the WOE Editor Advanced for a fraud model (lower event rates than credit scoring), variables with many nulls were getting WOE=0 for all rows.

**Console Evidence**: Many variables had IV values of exactly `0.0090`, indicating near-zero WOE bins.

### Root Cause Analysis

1. **High `MIN_BIN_PCT` (5%)** - Too aggressive for fraud models with low event rates (~3-5%)
2. **Failsafe logic** in `calculate_woe()` returns zeros when no goods OR no bads:
   ```python
   if total_good == 0 or total_bad == 0:
       return np.zeros(len(freq_good))
   ```
3. **ChiMerge collapsing to 1 bin** - When non-null values are sparse and chi-square test shows no significant difference, all bins merge
4. **GroupNA creating single-bin variables** - Merging NA with only remaining bin → 1 total bin → WOE=0

### Implemented Enhancements

**v1.2**: Added as all-or-nothing `UseEnhancements` flag  
**v1.3**: Added individual enhancement flow variables for fine-grained control

| Enhancement | Flow Variable | Location | Description |
|-------------|---------------|----------|-------------|
| **Adaptive min_prop** | `AdaptiveMinProp` | `_get_decision_tree_splits()` | Auto-adjusts for sparse data (< 500 samples) |
| **Minimum event count** | `MinEventCount` | `_get_decision_tree_splits()` | Ensures at least 5 bads per potential bin |
| **Automatic retry** | `AutoRetry` | `_get_decision_tree_splits()` | Retry with relaxed constraints if no splits found |
| **Chi-square validation** | `ChiSquareValidation` | `validate_bins_chi_square()` | Merge bins where p-value > 0.10 |
| **Single-bin protection** | `SingleBinProtection` | `na_combine()` | Skip NA grouping if it would create WOE=0 |
| **Master switch** | `UseEnhancements` | Configuration | Enable ALL enhancements at once |

### NOT Implemented (Suggested but Skipped)

| Enhancement | Reason Skipped |
|-------------|----------------|
| **IV-aware splitting** | Would require modifying sklearn's DecisionTree or building custom splitter |
| **Pure node prevention during splitting** | Complex - easier to handle in post-processing with `merge_pure_bins()` |
| **Isotonic regression smoothing** | Would add scipy dependency and complexity; marginal benefit |
| **IV-based bin merging** | Chi-square validation already covers similar ground |

### Processing Pipeline Comparison

| Step | R Original | Python (UseEnhancements=False) | Python (UseEnhancements=True) |
|------|------------|-------------------------------|------------------------------|
| 1 | `getBins()` (CART) | `get_bins()` (CART) | `get_bins()` (CART) + adaptive min_prop |
| 2 | — | Merge pure bins | Merge pure bins |
| 3 | — | **Skipped** | Chi-square validation |
| 4 | `naCombine()` | `na_combine()` | `na_combine()` + single-bin protection |
| 5 | `forceIncrTrend`/`forceDecrTrend` | Same | Same |
| 6 | WOE calculation | Same | Same + diagnostics |

### Recommended Settings by Model Type

**Fraud Models (low event rates 1-5%):**
```
MinBinPct = 0.01 (1%)
MinBinCount = 20
UseShrinkage = True
UseEnhancements = True
```

**Credit Scoring (higher event rates 10-20%):**
```
MinBinPct = 0.05 (5%)
MinBinCount = 50
UseShrinkage = False
UseEnhancements = False (R-compatible)
```

---

## Part 2: Future Enhancements (Not Yet Implemented)

### 2.0 Pending Enhancements (High Priority)

| Enhancement | Priority | Technical Notes |
|-------------|----------|-----------------|
| ~~**IV-aware splitting**~~ | ✅ DONE | Implemented as `Algorithm="IVOptimal"` in v1.4 |
| ~~**IV-based bin merging**~~ | ✅ DONE | IVOptimal uses IV-loss based merging |
| **Smart SingleBinProtection** | High | Instead of skipping NA grouping, merge bins to create 2 bins with highest IV |
| **Isotonic regression smoothing** | Medium | Apply isotonic regression to WOE values post-binning to reduce overfitting; requires scipy |
| **Pure node prevention during splitting** | Low | IVOptimal handles pure bins by merging to closest WOE neighbor |

**Implementation Notes:**
- Isotonic regression is commonly used in industry for WOE smoothing
- Smart SingleBinProtection: When a variable would become single-bin after NA grouping, instead of skipping, find the optimal way to create 2 bins that maximizes IV

### 2.1 UI Enhancements

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Manual Splits Input Field | High | Text input for comma-separated manual split points (exists in R UI) |
| Group Selected Bins | High | Combine user-selected rows in the bin table (exists in R UI) |
| Update Button | Medium | Save modifications to individual variable without running all |
| Gini Index Calculations | Medium | Add Input Gini, Output Gini, and Gini improvement metrics (in R version) |
| Variable Search/Filter | Medium | Search box to find variables by name |
| Sort Variables by IV | Medium | Option to sort variable list by Information Value |
| Export Bins to CSV | Low | Button to export bin table directly from UI |
| Dark Theme Toggle | Low | Option to switch between light and dark UI themes |

### 2.2 Algorithm Improvements

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| MDLP Binning | High | Minimum Description Length Principle binning (more statistically rigorous) |
| Isotonic Regression | High | Alternative monotonic optimization method |
| Optimal Binning Package | Medium | Integration with `optbinning` library for state-of-the-art algorithms |
| Custom Split Points Import | Medium | Load predefined split points from external file or table |
| Bin Count Configuration per Variable | Medium | Allow different max_bins settings per variable |
| Special Values Handling | Medium | Separate bins for special codes (e.g., -1, 999, -999999) |
| Weighted Binning | Low | Support for sample weights in WOE calculation |
| Target Encoding Alternative | Low | Option to output target encoding instead of WOE |

### 2.3 Performance Optimizations

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| GPU Acceleration | Low | RAPIDS/cuDF for GPU-accelerated binning on large datasets |
| Incremental Binning | Medium | Apply existing bins to new data without recalculating |
| Caching Layer | Medium | Cache bin results for repeated runs on same data |
| Memory Optimization | Medium | Chunk-based processing for very large datasets (>1M rows) |

### 2.4 Validation & Quality

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Population Stability Index (PSI) | High | Detect bin distribution drift between train/test |
| Characteristic Analysis | High | Summary report of all variables with IV categories |
| Bin Stability Testing | Medium | Cross-validation to ensure bins are stable |
| Multicollinearity Detection | Medium | Flag highly correlated variables |
| IV Threshold Filtering | Medium | Auto-exclude variables below IV threshold |
| Missing Value Impact Report | Low | Detailed analysis of NA handling impact |

### 2.5 Integration Features

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Bin Export to R Format | High | Export bins compatible with R logiBin |
| Scorecard Direct Integration | High | Pass bins directly to Scorecard node |
| Attribute Editor Integration | Medium | Share variable metadata between nodes |
| PMML Export | Low | Export binning as PMML preprocessing step |
| JSON Bin Export | Low | Export bins in portable JSON format |

### 2.6 Reporting

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| HTML Report Generation | High | Comprehensive binning report with charts |
| Variable Importance Ranking | High | Sort and display variables by predictive power |
| Bin Comparison View | Medium | Side-by-side before/after optimization comparison |
| Audit Trail | Low | Log all user modifications for compliance |

---

## Part 3: Known Issues & Bugs

### 3.1 Resolved Issues

| Issue | Resolution |
|-------|------------|
| Infinite WOE for pure bins | Added `merge_pure_bins()` function to combine 100% good/bad bins |
| Port conflicts with parallel KNIME instances | Implemented random port selection with socket availability check |
| Threading conflicts (NumExpr, OpenMP, OpenBLAS) | Set environment variables to limit thread count |
| Unmapped bin values causing NaN WOE | Added fallback logic and error logging in `add_woe_columns()` |
| Factor variable NA handling | Fixed NA rule parsing in `create_binned_columns()` |
| **Nulls assigned zero WOE** (fraud models) | Added `UseEnhancements` flow variable with adaptive min_prop, min event count, auto-retry, chi-square validation, and single-bin protection |
| **High null variables collapsing to single bin** | Added single-bin protection in `na_combine()` that skips grouping if it would result in WOE=0 |
| **MinBinPct too aggressive for fraud** | Changed default from 5% to 1% and added configurable `MinBinPct` flow variable |

### 3.2 Potential Issues to Monitor

| Issue | Status | Notes |
|-------|--------|-------|
| Large cardinality factor variables | Monitor | May create too many bins for variables with >100 unique values |
| Very rare events (<1% bad rate) | Monitor | Consider using shrinkage estimators |
| Missing column handling edge cases | Monitor | Ensure all edge cases for NA/NULL are covered |

---

## Part 4: Testing Status

| Component | Tested in KNIME | Notes |
|-----------|-----------------|-------|
| `woe_editor_knime.py` | Pending | Needs comprehensive testing |
| `woe_editor_knime_parallel.py` | Pending | Needs performance benchmarking |
| `woe_editor_advanced.py` | Pending | Needs algorithm validation vs R output |
| Interactive Mode | Pending | Browser launch may vary by environment |
| Headless Mode | Pending | Flow variable handling needs validation |

---

## Part 5: Version History

### woe_editor_knime.py
- **v1.0** (2026-01-15): Initial release with full R logiBin parity

### woe_editor_knime_parallel.py  
- **v2.0** (2026-01-15): Added parallel processing with joblib

### woe_editor_advanced.py
- **v1.0** (2026-01-20): Initial advanced version with ChiMerge
- **v1.1** (2026-01-24): Added R-compatible mode as default
- **v1.2** (2026-01-26): Added fraud model support, shrinkage, diagnostics
  - Fixed: Nulls being assigned zero WOE values in fraud models
  - Added: `UseEnhancements` flow variable (makes R-compatible the default)
  - Added: Adaptive min_prop for sparse data (< 500 samples)
  - Added: Minimum event count (at least 5 bads per bin for low event rates)
  - Added: Auto-retry with relaxed constraints when binning fails
  - Added: Chi-square validation to merge statistically similar bins
  - Added: Single-bin protection in `na_combine()` to prevent WOE=0
  - Added: Diagnostic logging for single-bin and zero-WOE variables
- **v1.3** (2026-01-26): Individual enhancement flow variables
  - Added: `AdaptiveMinProp` - enable/disable adaptive min_prop independently
  - Added: `MinEventCount` - enable/disable minimum event count independently
  - Added: `AutoRetry` - enable/disable auto-retry independently
  - Added: `ChiSquareValidation` - enable/disable chi-square validation independently
  - Added: `SingleBinProtection` - enable/disable single-bin protection (default ON)
  - Changed: `UseEnhancements` now acts as master switch that enables all when True
  - Changed: Individual flags can override master switch when explicitly set
- **v1.4** (2026-01-26): IVOptimal algorithm
  - Added: `Algorithm="IVOptimal"` - new IV-maximizing binning algorithm
  - IVOptimal directly maximizes Information Value (not entropy/gini)
  - Does NOT enforce monotonicity - allows "sweet spots" and U-shaped patterns
  - Dynamic starting granularity based on variable characteristics
  - Merges pure bins to closest adjacent bin by WOE value
  - Best for fraud detection where non-monotonic patterns exist

---

## Part 6: Reference

### R Function Mappings

| R (logiBin) | Python (this project) | Status |
|-------------|----------------------|--------|
| `getBins()` | `get_bins()` | ✅ Complete |
| `forceIncrTrend()` | `force_incr_trend()` | ✅ Complete |
| `forceDecrTrend()` | `force_decr_trend()` | ✅ Complete |
| `naCombine()` | `na_combine()` | ✅ Complete |
| `createBins()` | `create_binned_columns()` | ✅ Complete |
| `manualSplit()` | `manual_split()` | ✅ Complete |
| `manualDiscreteSplit()` | `break_bin()` | ✅ Complete |
| `calculateWOE()` | `calculate_woe()` | ✅ Complete |
| `updateDiscreteBins()` | `update_bin_stats()` | ✅ Complete |

### Dependencies

```
pandas
numpy
scikit-learn
shiny
shinywidgets
plotly
scipy (for advanced version)
joblib (for parallel version)
```

---

## Appendix: Priority Legend

- **High**: Critical for production use or significant user value
- **Medium**: Nice to have, improves usability or performance
- **Low**: Future consideration, not blocking current use

---

*Document maintained as part of KNIME Python Nodes project*

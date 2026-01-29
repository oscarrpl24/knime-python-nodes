# WOE Editor High-Cardinality Categorical Variable Fix

## Problem

The WOE Editor was creating **thousands of bins** for high-cardinality categorical variables (variables with many unique values), causing:
- Millions of interaction term combinations in the Scorecard node
- Node hanging/freezing during execution
- Memory exhaustion

### Example
If `bvpattr166` is categorical with 2,000 unique values and `bvpattr168` has 2,000 unique values:
- WOE Editor creates: 2,000 bins + 2,000 bins = 4,000 bins
- Scorecard node creates: 2,000 × 2,000 = **4,000,000 interaction combinations** 💥

## Root Cause

**File:** `woe_editor_advanced.py`, function `_create_factor_bins()`

The function was creating **one bin for every unique categorical value**, regardless of how many unique values existed. This is correct for low-cardinality categoricals (e.g., Gender: M/F), but disastrous for high-cardinality ones (e.g., Customer ID with thousands of unique values).

## Why WOE-Based Grouping for Categoricals?

Unlike numeric variables (which have natural order: 20 < 30 < 40), categorical variables have **no inherent order**. This means:

✅ **Numeric bins must be adjacent:** Age 20-30, 30-40, 40-50 (contiguous ranges)
✅ **Categorical bins can group ANY similar categories:** 
- States by risk: Group ["CA", "FL", "NV"] (high risk) and ["ND", "WY", "VT"] (low risk)
- Products by default rate: Group products with similar WOE regardless of product code order

**Key Advantage:** We group by **risk similarity (WOE)**, not arbitrary alphabetical or sequential order. This preserves predictive power while respecting MAX_BINS.

```python
# OLD CODE - Created one bin per unique value (dangerous!)
unique_vals = x.dropna().unique()  # Could be thousands!
for val in unique_vals:
    # Create a bin for each value...
```

## Solution

Added **intelligent binning for categorical variables** with three strategies:

### Strategy 1: Low Cardinality (≤ MAX_BINS, default 10)
- Each category gets its own bin (traditional approach)
- Example: Gender (2 categories) → 2 bins

### Strategy 2: Medium Cardinality (> MAX_BINS but ≤ MAX_CATEGORIES, default 50)
- **Groups categories by WOE similarity** into MAX_BINS bins
- Categories with similar WOE (risk profiles) are combined, regardless of order
- No need for adjacency - any categories with similar WOE can be grouped
- Example: 35 state codes → 10 bins where similar-risk states are grouped together

### Strategy 3: Very High Cardinality (> MAX_CATEGORIES)
- Skips the variable with a warning
- Example: 2,000 unique dates → **skipped**

```python
# NEW CODE - Smart categorical binning
n_unique = len(unique_vals)

if n_unique <= max_bins:
    # Strategy 1: One bin per category
    for val in unique_vals:
        # Each category gets its own bin
        ...
elif n_unique <= max_categories:
    # Strategy 2: Group by WOE similarity
    # Calculate WOE for each category
    for val in unique_vals:
        woe = np.log((bads/total_bads) / (goods/total_goods))
        cat_stats.append({'value': val, 'woe': woe, ...})
    
    # Group categories with similar WOE (no adjacency required!)
    cat_df['bin_group'] = pd.qcut(cat_df['woe'], q=max_bins, ...)
    
    # Categories with similar risk are now in the same bin
    # Example: States with WOE ~0.5: ["CA", "NV", "FL"] 
    #          States with WOE ~-0.3: ["ND", "WY", "VT"]
    ...
else:
    # Strategy 3: Skip with warning
    print(f"WARNING: Variable '{var}' has {n_unique} unique categories!")
    return pd.DataFrame()
```

## Usage

### Default Behavior (No Changes Needed)
- Categorical variables with ≤ 10 unique values: **One bin per category** (traditional)
- Categorical variables with 11-50 unique values: **Grouped into 10 bins by WOE similarity** (smart grouping)
  - Categories are grouped by similar risk profiles, not by order
  - Any categories with similar WOE can be combined (no adjacency requirement)
- Categorical variables with > 50 unique values: **Skipped** with warning

### Adjusting the Limit (Flow Variable)

If you need to include high-cardinality categoricals, add a flow variable:

```
Flow Variable Name: MaxCategories
Type: Integer
Value: 100 (or your desired limit)
```

**Warning:** Setting this too high can cause:
- Long processing times
- Memory issues
- Millions of interaction combinations

### Recommended Approach for High-Cardinality Variables

Instead of increasing `MaxCategories`, **recode the variable** first:

1. **Group by target rate:** Cluster similar categories together
2. **Top N + Other:** Keep top 20 most frequent, group rest as "Other"
3. **Business logic:** Group by meaningful business categories
4. **Drop the variable:** If it's not predictive (e.g., random IDs)

## Changes Made

### 1. `_create_factor_bins()` (Line ~1150)
- Added `max_categories` parameter (default 50)
- Added `max_bins` parameter to respect MAX_BINS setting
- **Three-tier strategy:**
  - Low cardinality (≤ 10): One bin per category
  - Medium cardinality (11-50): Group into 10 bins by **WOE similarity**
    - Calculates WOE for each category individually
    - Groups categories with similar WOE values (no adjacency required)
    - Example: High-risk states grouped together, low-risk states grouped together
  - High cardinality (> 50): Skip with warning
- Categorical variables now respect MAX_BINS like numeric variables

### 2. `BinningConfig` class (Line ~227)
- Added `MAX_CATEGORIES = 50` setting

### 3. Flow Variable Support (Line ~2880)
- Added `MaxCategories` flow variable parsing

### 4. Configuration Logging (Line ~2919)
- Shows `MaxCategories` setting in console output

## Verification

After the fix, you should see in the KNIME console:

```
WOE EDITOR - HEADLESS MODE (DecisionTree (R-compatible))
MaxCategories: 50 (categorical variables with more unique values will be skipped)
...

# Low cardinality (≤ 10) - one bin per category
Processing variable: Gender (2 categories)

# Medium cardinality (11-50) - intelligent WOE-based grouping
INFO: Variable 'StateCode' has 35 categories - grouping into 10 bins by WOE similarity
  Grouped 4 categories (avg WOE: -0.523)
  Grouped 3 categories (avg WOE: -0.201)
  Grouped 5 categories (avg WOE: 0.089)
  ...
  Grouped 4 categories (avg WOE: 0.687)

# High cardinality (> 50) - skipped
WARNING: Variable 'bvpattr166' has 1974 unique categories!
  Exceeds max_categories=50. Skipping variable.
WARNING: Variable 'bvpattr168' has 2000 unique categories!
  Exceeds max_categories=50. Skipping variable.
```

## Impact on Downstream Nodes

### Scorecard Node
- **Processes ALL interactions without limits** (as intended)
- Will show progress for large interactions (> 1,000 combinations)
- High-cardinality variables controlled at WOE Editor level (before scorecard)
- Example output: "INFO: Interaction var1_x_var2 will create 10,000 combinations - This may take some time..."

### Logistic Regression Node
- Skipped variables won't be available as predictors
- Model will train faster
- Better for avoiding overfitting on high-cardinality features

## Design Philosophy

**WOE Editor:** Controls bin counts and cardinality (prevention)
- MAX_BINS = 10 (limits bins per variable)
- MAX_CATEGORIES = 50 (skips very high-cardinality variables)
- Smart grouping for categorical variables (11-50 categories)

**Scorecard Node:** Processes whatever you give it (execution)
- No artificial limits on combinations
- Informational logging for large interactions
- Trusts that WOE Editor provided clean inputs

## Identifying High-Cardinality Variables

Before running WOE Editor, check your data:

```python
# In a Python Script node before WOE Editor
import knime.scripting.io as knio
import pandas as pd

df = knio.input_tables[0].to_pandas()

# Find categorical columns with many unique values
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        n_unique = df[col].nunique()
        if n_unique > 50:
            print(f"High-cardinality: {col} has {n_unique} unique values")
```

## Version History

- **2026-01-26:** Added MAX_CATEGORIES limit to prevent high-cardinality issues
- **Default:** 50 categories (configurable via flow variable)

## Related Files

- `scorecard_knime.py` - Updated to process all interactions without artificial limits (informational logging only)
- `scorecard_node_inputs_guide.txt` - Guide for correct scorecard inputs and expected processing times

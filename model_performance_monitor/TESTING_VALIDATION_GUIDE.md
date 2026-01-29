# Testing & Validation Guide
## Model Performance Monitor

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Test with Synthetic Data](#quick-test-with-synthetic-data)
3. [Unit Testing Checklist](#unit-testing-checklist)
4. [Integration Testing](#integration-testing)
5. [Validation Against Known Results](#validation-against-known-results)
6. [Common Test Scenarios](#common-test-scenarios)
7. [Performance Testing](#performance-testing)
8. [Troubleshooting Test Failures](#troubleshooting-test-failures)

---

## Overview

This guide helps you validate that the Model Performance Monitor is working correctly before using it with production data.

### Testing Goals

- ✅ Verify core metrics calculate correctly
- ✅ Ensure recommendations match expectations
- ✅ Validate scorecard parameter conversions
- ✅ Test edge cases (small samples, extreme drift)
- ✅ Confirm KNIME integration works smoothly

---

## Quick Test with Synthetic Data

### Step 1: Generate Test Data

Use the included test data generator to create realistic synthetic data.

**In KNIME:**

```
1. Add Python Script node
2. Paste contents of test_data_generator.py
3. Configure for 2 output ports
4. Execute node

This generates:
  - Output Port 0: Baseline data (5,000 loans)
  - Output Port 1: Production data (2,000 loans)
```

**Configuration in test_data_generator.py:**
```python
# Default (moderate drift scenario):
POPULATION_DRIFT = "moderate"     # PSI ~0.18
MODEL_DEGRADATION = "slight"      # AUC decline ~3%

Expected Recommendation: MONITOR ⚠️
```

---

### Step 2: Run Monitor Node

Connect the test data to Model Performance Monitor:

```
┌─────────────────────┐
│ Test Data Generator │
└──────┬──────┬───────┘
       │      │
       │      └─────────────┐
       │                    │
       ▼                    ▼
   Output 1             Output 0
   (Production)         (Baseline)
       │                    │
       └─────────┬──────────┘
                 │
                 ▼
     ┌──────────────────────────┐
     │ Model Performance Monitor│
     │                          │
     │ Flow Variables:          │
     │   DependentVariable: isBad
     │   ScoreColumn: score     │
     │   Points: 600            │
     │   Odds: 20               │
     │   PDO: 50                │
     └──────────┬───────────────┘
                │
                ▼
         [4 Output Tables]
```

---

### Step 3: Verify Results

**Expected Output (with "moderate" drift):**

| Metric | Current | Baseline | Delta | Status | Alert |
|--------|---------|----------|-------|--------|-------|
| Sample Size | ~800 | ~2,000 | N/A | OK | ✓ |
| AUC | ~0.72 | ~0.75 | -0.03 | WARNING | ⚠️ |
| PSI | ~0.18 | N/A | N/A | WARNING | ⚠️ |
| Bad Rate | ~0.15 | ~0.12 | +0.03 | WARNING | ⚠️ |
| **Recommendation** | **MONITOR** | N/A | N/A | **MONITOR** | **⚠️** |

**✅ Test Passes If:**
- All 4 output tables generated
- Recommendation is "MONITOR"
- PSI between 0.15-0.25
- AUC shows slight decline
- Bad rate increased

---

### Step 4: Test Different Scenarios

Change `POPULATION_DRIFT` and `MODEL_DEGRADATION` in test_data_generator.py:

**Scenario A: Healthy Model**
```python
POPULATION_DRIFT = "none"
MODEL_DEGRADATION = "none"

Expected: Recommendation = "OK" ✓
```

**Scenario B: Significant Issues**
```python
POPULATION_DRIFT = "significant"
MODEL_DEGRADATION = "moderate"

Expected: Recommendation = "RETRAIN" 🔴
```

**Scenario C: Complete Failure**
```python
POPULATION_DRIFT = "significant"
MODEL_DEGRADATION = "severe"

Expected: 
  - PSI > 0.25
  - AUC decline > 5%
  - Recommendation = "RETRAIN" 🔴
```

---

## Unit Testing Checklist

### Test 1: Core Metrics Calculation

**Objective:** Verify discrimination metrics (AUC, K-S, Gini) calculate correctly.

**Test Data:**
```python
# Perfect separation (AUC should be 1.0)
scores = [900, 850, 800, 750, 700, 650, 600, 550, 500, 450]
outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

Expected:
  AUC = 1.0
  Gini = 1.0
  K-S = 1.0
```

**Test Data:**
```python
# Random model (AUC should be ~0.5)
scores = [random for all]
outcomes = [random for all]

Expected:
  AUC ≈ 0.5 (±0.05)
  Gini ≈ 0.0 (±0.1)
  K-S ≈ 0.0 (±0.1)
```

**✅ Pass Criteria:**
- AUC within expected range
- Gini = 2*AUC - 1 (verify formula)
- K-S between 0 and 1

---

### Test 2: PSI Calculation

**Objective:** Verify Population Stability Index calculates correctly.

**Test Data:**
```python
# No shift (PSI should be 0)
baseline_scores = [600, 650, 700, ...] * 100
production_scores = [600, 650, 700, ...] * 100

Expected: PSI ≈ 0.0

# Extreme shift (PSI should be high)
baseline_scores = [500-700] distribution
production_scores = [600-800] distribution (shifted +100)

Expected: PSI > 0.5
```

**Manual PSI Calculation:**
```
1. Create 10 deciles from baseline
2. Count % in each decile for baseline
3. Count % in each decile for production
4. PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```

**✅ Pass Criteria:**
- PSI = 0 when distributions identical
- PSI increases with distribution shift
- PSI values are non-negative

---

### Test 3: Score-to-Probability Conversion

**Objective:** Verify inverse scorecard formula works correctly.

**Test Cases:**

| Score | Points | Odds | PDO | Expected Probability |
|-------|--------|------|-----|---------------------|
| 600 | 600 | 20 | 50 | ~0.10 (10% bad rate) |
| 650 | 600 | 20 | 50 | ~0.05 (5% bad rate) |
| 550 | 600 | 20 | 50 | ~0.20 (20% bad rate) |

**Manual Calculation:**
```python
Points = 600, Odds = 20, PDO = 50

b = 50 / ln(2) = 72.13
odds0 = 1 / (20 - 1) = 1/19 = 0.0526
a = 600 + 72.13 * ln(0.0526) = 600 + 72.13 * (-2.944) = 387.6

For score = 650:
  log_odds = (387.6 - 650) / 72.13 = -3.64
  probability = 1 / (1 + exp(3.64)) = 0.026 (~2.6%)

For score = 600:
  log_odds = (387.6 - 600) / 72.13 = -2.944
  probability = 1 / (1 + exp(2.944)) = 0.050 (~5%)
```

**✅ Pass Criteria:**
- Calculated probabilities match manual calculation (±0.01)
- Higher scores → lower probabilities
- Probabilities between 0 and 1

---

### Test 4: Calibration Analysis

**Objective:** Verify calibration table identifies miscalibrated bins.

**Test Data:**
```python
# Perfect calibration
predicted = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, ...]
observed =  [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, ...]

Expected: 
  All bins marked as "calibrated"
  Calibration error ≈ 0

# Poor calibration
predicted = [0.1, 0.1, 0.1, 0.1, ...]
observed =  [0.3, 0.4, 0.3, 0.4, ...]

Expected:
  Bins marked as NOT calibrated
  Calibration error > 0.05
```

**✅ Pass Criteria:**
- Calibration error < 0.01 for perfect calibration
- Bins correctly marked as calibrated/not calibrated
- Calibration table has expected columns

---

### Test 5: Decile Analysis

**Objective:** Verify decile-level metrics calculate correctly.

**Test Data:**
```python
# 1,000 loans with clear score separation
Decile 1 (top): scores 800-900, all good outcomes (bad_rate = 0%)
Decile 10 (bottom): scores 300-400, all bad outcomes (bad_rate = 100%)
```

**Expected Decile Table:**
```
Decile  Score_Min  Score_Max  Bad_Rate  (should be monotonic)
1       800        900        0.00      ← Best
2       700        799        0.10
3       600        699        0.20
...
10      300        399        1.00      ← Worst
```

**✅ Pass Criteria:**
- 10 deciles created
- Bad rate increases from decile 1 to 10
- Cumulative bad rate increases monotonically
- All funded loans accounted for

---

### Test 6: Recommendation Logic

**Objective:** Verify recommendation triggers at correct thresholds.

**Test Cases:**

| PSI | AUC_Delta | KS_Delta | BadRate_Delta | Expected Rec |
|-----|-----------|----------|---------------|--------------|
| 0.05 | -0.01 | -0.02 | +0.01 | OK ✓ |
| 0.15 | -0.02 | -0.03 | +0.01 | MONITOR ⚠️ |
| 0.12 | -0.04 | -0.03 | +0.02 | MONITOR ⚠️ |
| 0.30 | -0.02 | -0.03 | +0.01 | RETRAIN 🔴 |
| 0.08 | -0.07 | -0.03 | +0.01 | RETRAIN 🔴 |
| 0.08 | -0.02 | -0.12 | +0.01 | RETRAIN 🔴 |

**✅ Pass Criteria:**
- Correct recommendation for each scenario
- Critical thresholds take precedence over warnings
- Multiple warnings escalate to MONITOR

---

## Integration Testing

### Test 7: KNIME I/O Integration

**Objective:** Verify data flows correctly through KNIME.

**Test Workflow:**
```
CSV Reader → Model Performance Monitor → CSV Writer
```

**Checklist:**
- [ ] Input tables load correctly
- [ ] All required columns present
- [ ] Flow variables read correctly
- [ ] All 4 output tables generated
- [ ] Output tables have expected columns
- [ ] Output tables can be written to CSV
- [ ] No KNIME errors in console

---

### Test 8: Flow Variables

**Objective:** Verify all flow variables work correctly.

**Test Method:**
1. Create workflow variables for all parameters
2. Set various values
3. Verify Monitor node reads them correctly

**Variables to Test:**
```
Required:
  ✓ DependentVariable (string)

Optional Column Names:
  ✓ ScoreColumn (string)
  ✓ ApprovalColumn (string)
  ✓ FundedColumn (string)
  ✓ ROIColumn (string)

Scorecard Parameters:
  ✓ Points (integer)
  ✓ Odds (integer)
  ✓ PDO (integer)

Thresholds:
  ✓ PSI_Warning (double)
  ✓ PSI_Critical (double)
  ✓ AUC_Decline_Warning (double)
  ... (all thresholds)
```

---

### Test 9: Missing Baseline Scenario

**Objective:** Verify node works without baseline data.

**Test Setup:**
```
Production Data → Model Performance Monitor (no baseline)
```

**Expected Behavior:**
- Node executes successfully
- PSI shows as "N/A"
- Delta columns show "N/A"
- Baseline columns show "N/A"
- Recommendation based on absolute metrics only
- No errors or crashes

**✅ Pass Criteria:**
- Node completes without errors
- Summary table generated (with N/A for baseline metrics)
- Decile and calibration tables still generated

---

## Validation Against Known Results

### Validation Test 1: Compare to R Model Analyzer

**If you have existing R-based monitoring:**

1. Run same data through R version
2. Run same data through Python Monitor
3. Compare key metrics:

| Metric | R Result | Python Result | Difference | Pass? |
|--------|----------|---------------|------------|-------|
| AUC | 0.7543 | 0.7541 | 0.0002 | ✓ (< 0.01) |
| K-S | 0.4201 | 0.4198 | 0.0003 | ✓ (< 0.01) |
| Gini | 0.5086 | 0.5082 | 0.0004 | ✓ (< 0.01) |
| PSI | 0.1823 | 0.1819 | 0.0004 | ✓ (< 0.01) |

**✅ Pass Criteria:**
- All metrics within ±0.01 of R results
- Recommendation matches R version

---

### Validation Test 2: Manual PSI Calculation

**Use spreadsheet to verify PSI:**

1. Export baseline and production scores
2. Create 10 deciles manually in Excel
3. Calculate % in each decile
4. Calculate PSI: `=SUM((actual% - expected%) * LN(actual% / expected%))`
5. Compare to node output

**Example:**

| Decile | Baseline % | Production % | (Act-Exp) | ln(Act/Exp) | PSI Component |
|--------|-----------|--------------|-----------|-------------|---------------|
| 1 | 10.0% | 8.5% | -1.5% | -0.163 | 0.0024 |
| 2 | 10.0% | 9.2% | -0.8% | -0.084 | 0.0007 |
| ... | ... | ... | ... | ... | ... |
| 10 | 10.0% | 12.3% | +2.3% | 0.207 | 0.0048 |
| **Total** | | | | | **0.1820** |

**✅ Pass Criteria:**
- Node PSI matches manual calculation (±0.005)

---

## Common Test Scenarios

### Scenario 1: Small Sample Size

**Test:** Only 200 funded loans (below 500 minimum)

**Expected Behavior:**
- Warning message in summary table
- Metrics still calculated but flagged
- Recommendation provided with caveat

**Verification:**
```
Summary Table should show:
  Sample Size: 200  ⚠️ WARNING
  Note: "Metrics may be unreliable with small sample"
```

---

### Scenario 2: No Bad Outcomes

**Test:** All funded loans are good (bad_rate = 0%)

**Expected Behavior:**
- AUC, K-S cannot be calculated (need both classes)
- Show as N/A
- Bad rate = 0.00
- Recommendation based on available metrics (PSI, approval rate)

---

### Scenario 3: Missing ROI Column

**Test:** Production data doesn't have ROI column

**Expected Behavior:**
- Node still works
- ROI metrics show as N/A
- Other metrics calculated normally
- No crash or error

---

### Scenario 4: Extreme Drift

**Test:** PSI > 1.0 (very rare)

**Expected Behavior:**
- PSI value displayed accurately
- Recommendation = "RETRAIN"
- Warning about extreme population shift

---

### Scenario 5: Identical Distributions

**Test:** Production and baseline have identical score distributions

**Expected Behavior:**
- PSI ≈ 0.0
- Recommendation = "OK" (if other metrics also stable)
- No false alarms

---

## Performance Testing

### Test 10: Large Dataset Performance

**Objective:** Verify node scales to production data sizes.

**Test Datasets:**

| Size | Funded Loans | Expected Runtime |
|------|--------------|------------------|
| Small | 1,000 | < 3 seconds |
| Medium | 10,000 | < 10 seconds |
| Large | 100,000 | < 60 seconds |
| Very Large | 1,000,000 | < 5 minutes |

**Monitoring:**
- KNIME console for memory warnings
- CPU usage during execution
- Output table generation time

**✅ Pass Criteria:**
- Completes within expected time
- No memory errors
- All outputs generated

---

### Test 11: Concurrent Execution

**Objective:** Verify multiple instances can run simultaneously.

**Test Setup:**
```
Run 3 parallel branches:
  Branch 1: Monitor with Dataset A
  Branch 2: Monitor with Dataset B
  Branch 3: Monitor with Dataset C
```

**✅ Pass Criteria:**
- All branches complete successfully
- No port conflicts (Shiny UI)
- Results are correct for each dataset

---

## Troubleshooting Test Failures

### Failure: "Cannot convert score to probability"

**Cause:** Invalid scorecard parameters

**Check:**
```python
Points, Odds, PDO values:
  ✓ All positive numbers
  ✓ Odds > 1 (e.g., 20, not 0.05)
  ✓ PDO reasonable (typically 20-100)

If Odds = 0.05:
  This is odds0 (1/19), not Odds!
  Should be: Odds = 20
```

---

### Failure: PSI calculation error

**Cause:** Insufficient data or extreme distributions

**Solutions:**
1. Check minimum 100 samples in each dataset
2. Verify score ranges overlap
3. Ensure no all-same-score scenarios

---

### Failure: Recommendation is always "OK"

**Cause:** Thresholds too lenient

**Check:**
```
Flow variables:
  PSI_Warning: Should be 0.1 (not 1.0)
  PSI_Critical: Should be 0.25 (not 2.5)
  AUC_Decline_Warning: Should be 0.03 (not 0.3)

Common mistake: Setting as percentages instead of decimals
  Wrong: PSI_Warning = 10 (meaning 10%)
  Right: PSI_Warning = 0.1 (meaning 0.1 or 10%)
```

---

### Failure: Output tables empty

**Cause:** Filtering too aggressive or data missing

**Debug Steps:**
1. Check input data:
   ```
   - isFunded column exists?
   - isFunded = 1 for some rows?
   - DV column has both 0 and 1 values?
   ```
2. Check column names match flow variables:
   ```
   DependentVariable = "isBad" matches actual column name "isBad"
   (Case sensitive!)
   ```
3. Add debug node to check funded count:
   ```
   Row Filter (isFunded = 1) → Row Counter
   If count = 0: No funded loans to analyze!
   ```

---

## Final Validation Checklist

Before deploying to production:

### Data Validation
- [ ] All required columns present
- [ ] Column names match flow variables
- [ ] Score values reasonable (300-850 range)
- [ ] Outcomes are binary (0/1)
- [ ] ROI values make sense (losses < 1.0)
- [ ] Sufficient sample size (> 500 funded)

### Configuration Validation
- [ ] Scorecard parameters match Scorecard Generator
- [ ] Thresholds appropriate for use case
- [ ] Column name flow variables set correctly
- [ ] DependentVariable points to actual outcome column

### Output Validation
- [ ] All 4 output tables generated
- [ ] Summary table has all metrics
- [ ] Decile table has 10 deciles
- [ ] Calibration table not empty
- [ ] Recommendation makes sense

### Integration Validation
- [ ] KNIME workflow executes without errors
- [ ] Can save outputs to CSV
- [ ] Can connect to downstream nodes
- [ ] Works in both interactive and headless modes

### Scenario Testing
- [ ] Tested with "OK" scenario (no issues)
- [ ] Tested with "MONITOR" scenario (moderate drift)
- [ ] Tested with "RETRAIN" scenario (significant issues)
- [ ] Tested without baseline (PSI shows N/A)
- [ ] Tested with small sample (warning appears)

---

## Test Report Template

Use this template to document your testing:

```
====================================================================
MODEL PERFORMANCE MONITOR - TEST REPORT
====================================================================

Test Date: _______________
Tester: _______________
Version: 1.0

QUICK TEST (Synthetic Data)
----------------------------
[ ] Generated test data successfully
[ ] Monitor node executed without errors
[ ] Recommendation matched expected for "moderate" drift scenario
[ ] All 4 output tables generated

UNIT TESTS
----------------------------
[ ] Test 1: Core Metrics (AUC, K-S, Gini) - PASS/FAIL
[ ] Test 2: PSI Calculation - PASS/FAIL
[ ] Test 3: Score-to-Probability - PASS/FAIL
[ ] Test 4: Calibration Analysis - PASS/FAIL
[ ] Test 5: Decile Analysis - PASS/FAIL
[ ] Test 6: Recommendation Logic - PASS/FAIL

INTEGRATION TESTS
----------------------------
[ ] Test 7: KNIME I/O - PASS/FAIL
[ ] Test 8: Flow Variables - PASS/FAIL
[ ] Test 9: Missing Baseline - PASS/FAIL

VALIDATION TESTS
----------------------------
[ ] Validation 1: Compare to R (if applicable) - PASS/FAIL/N/A
[ ] Validation 2: Manual PSI Calculation - PASS/FAIL

SCENARIO TESTS
----------------------------
[ ] Scenario 1: Small Sample Size - PASS/FAIL
[ ] Scenario 2: No Bad Outcomes - PASS/FAIL
[ ] Scenario 3: Missing ROI - PASS/FAIL
[ ] Scenario 4: Extreme Drift - PASS/FAIL
[ ] Scenario 5: Identical Distributions - PASS/FAIL

PERFORMANCE TESTS
----------------------------
[ ] Test 10: Large Dataset (_____ rows) - Runtime: _____ seconds
[ ] Test 11: Concurrent Execution - PASS/FAIL

FINAL VALIDATION
----------------------------
[ ] All items in Final Validation Checklist completed

ISSUES FOUND
----------------------------
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

OVERALL RESULT: PASS / FAIL

NOTES:
________________________________________________________
________________________________________________________
________________________________________________________

APPROVED FOR PRODUCTION: YES / NO

Signature: _______________  Date: _______________
====================================================================
```

---

**Testing complete! Your Model Performance Monitor is validated and ready for production use.**

---

**For more information, see:**
- README.md - Technical reference
- USAGE_EXAMPLE.md - Real-world scenarios
- KNIME_WORKFLOW_INTEGRATION_GUIDE.md - Integration patterns

---

**End of Testing & Validation Guide**

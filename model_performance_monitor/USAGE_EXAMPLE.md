# Model Performance Monitor - Usage Examples

## Quick Start

### Minimum Configuration (Headless Mode)

```python
# Flow Variables:
DependentVariable: "isBad"

# That's it! All other parameters have sensible defaults.
```

### With Baseline Comparison

```python
# Flow Variables:
DependentVariable: "isBad"
ScoreColumn: "score"
ApprovalColumn: "isApproved"
FundedColumn: "isFunded"
ROIColumn: "ROI"

# Scorecard parameters (must match your Scorecard Generator settings):
Points: 600
Odds: 20
PDO: 50

# Input Port 1: Current month production data (January 2026)
# Input Port 2: Training data or previous 6-month baseline (July-Dec 2025)
```

---

## Example Workflows

### Workflow 1: Initial Deployment (No Baseline)

```
[Production Data - Month 1] → [Model Performance Monitor] → [Summary Table]
                                                          → [Decile Analysis]
                                                          → [Calibration Table]
```

**Configuration:**
- Only Input Port 1 (production data)
- No baseline provided
- Set `DependentVariable: "isBad"`

**Expected Output:**
```
Summary Table:
  Sample Size: 1,250 funded loans  ✓ OK
  AUC: 0.73  ℹ️ INFO (no baseline to compare)
  Gini: 0.46  ℹ️ INFO
  K-S: 0.41  ℹ️ INFO
  PSI: N/A (no baseline)
  Approval Rate: 0.42  ℹ️ INFO
  Bad Rate (Funded): 0.12  ℹ️ INFO
  Avg ROI: 0.89  ℹ️ INFO
  Recommendation: OK
```

---

### Workflow 2: Monitoring with Training Baseline

```
[Production Data - Month 6] → [Model Performance Monitor] → [Summary with Alerts]
[Training Data]             →                            → [Decile Analysis]
                                                         → [Calibration]
```

**Configuration:**
- Input Port 1: Month 6 production data
- Input Port 2: Original training data
- Alert thresholds at defaults

**Scenario 1: Healthy Model**

Output Summary:
```
Metric                  Current    Baseline   Delta    Status      Alert
────────────────────────────────────────────────────────────────────────
Sample Size             1,500      3,000      N/A      OK          ✓
AUC                     0.74       0.76       -0.02    OK          ✓
Gini                    0.48       0.52       -0.04    OK          ✓
K-S Statistic           0.42       0.45       -0.03    OK          ✓
PSI (Score)             0.08       N/A        N/A      OK          ✓
Approval Rate           0.43       0.42       +0.01    INFO        ℹ️
Bad Rate (Funded)       0.13       0.12       +0.01    OK          ✓
Avg ROI (Funded)        0.87       0.92       -0.05    INFO        ℹ️
Calibration Error       0.03       N/A        N/A      OK          ✓
Recommendation          OK         N/A        N/A      OK          ✓
```

**Action:** Continue normal operations. Model performing as expected.

---

**Scenario 2: Population Drift Detected**

Output Summary:
```
Metric                  Current    Baseline   Delta    Status      Alert
────────────────────────────────────────────────────────────────────────
Sample Size             1,500      3,000      N/A      OK          ✓
AUC                     0.72       0.76       -0.04    WARNING     ⚠️
Gini                    0.44       0.52       -0.08    WARNING     ⚠️
K-S Statistic           0.38       0.45       -0.07    WARNING     ⚠️
PSI (Score)             0.18       N/A        N/A      WARNING     ⚠️
Approval Rate           0.48       0.42       +0.06    INFO        ℹ️
Bad Rate (Funded)       0.15       0.12       +0.03    WARNING     ⚠️
Avg ROI (Funded)        0.82       0.92       -0.10    INFO        ℹ️
Calibration Error       0.06       N/A        N/A      WARNING     ⚠️
Recommendation          MONITOR    N/A        N/A      MONITOR     ⚠️
```

**Interpretation:**
- PSI = 0.18: Moderate population shift (threshold: 0.1-0.25)
- AUC declined 4% (warning threshold: 3%)
- Bad rate increased 3% (threshold: 2%)
- Scores are miscalibrated (error > 5%)

**Action:** 
1. Investigate why population is shifting (marketing changes? economic conditions?)
2. Increase monitoring frequency to weekly
3. Review decile analysis for specific score ranges affected
4. Consider adjusting approval thresholds
5. Plan for retraining within 2-3 months if trend continues

---

**Scenario 3: Critical - Retraining Recommended**

Output Summary:
```
Metric                  Current    Baseline   Delta    Status      Alert
────────────────────────────────────────────────────────────────────────
Sample Size             1,500      3,000      N/A      OK          ✓
AUC                     0.68       0.76       -0.08    CRITICAL    🔴
Gini                    0.36       0.52       -0.16    CRITICAL    🔴
K-S Statistic           0.32       0.45       -0.13    CRITICAL    🔴
PSI (Score)             0.32       N/A        N/A      CRITICAL    🔴
Approval Rate           0.52       0.42       +0.10    INFO        ℹ️
Bad Rate (Funded)       0.19       0.12       +0.07    WARNING     ⚠️
Avg ROI (Funded)        0.74       0.92       -0.18    INFO        ℹ️
Calibration Error       0.11       N/A        N/A      WARNING     ⚠️
Recommendation          RETRAIN    N/A        N/A      RETRAIN     🔴
```

**Interpretation:**
- PSI = 0.32: Significant population shift (critical threshold: 0.25)
- AUC declined 8% (critical threshold: 5%)
- K-S declined 13% (critical threshold: 10%)
- Bad rate nearly doubled (+7%)
- Model no longer discriminates well

**Action:**
1. **IMMEDIATE:** Stop using model for auto-decisions (switch to manual review)
2. Investigate root cause of shift
3. Gather recent production data (last 6-12 months)
4. Retrain model with updated data
5. Validate new model thoroughly before deployment
6. Update scorecard with new bins and coefficients

---

### Workflow 3: Rolling Baseline Monitoring

```
[Current Month Data] → [Model Performance Monitor] → [Monthly Report]
[Last 6 Months]      →
```

**Use Case:** After 6+ months in production, compare current month to recent stable baseline.

**Configuration:**
- Input Port 1: October 2025 data (current month)
- Input Port 2: April-September 2025 data (6-month baseline)
- This detects recent changes, not long-term drift from training

**Why Rolling Baseline?**
- Training data from 2 years ago may no longer be relevant benchmark
- Economy, regulations, competition change over time
- Rolling baseline adapts to new "normal"
- Still catch sudden month-over-month changes

---

## Decile Analysis Interpretation

### Example Decile Table

```
Decile  Count  Score_Min  Score_Max  Score_Avg  Bad_Rate  Approval_Rate  Avg_ROI  Cumulative_Bad_Rate
──────────────────────────────────────────────────────────────────────────────────────────────────────
1       150    720        850        752        0.02      0.95           1.12     0.02
2       150    680        719        695        0.04      0.88           1.08     0.03
3       150    640        679        658        0.07      0.75           1.02     0.04
4       150    600        639        618        0.10      0.62           0.97     0.06
5       150    560        599        578        0.15      0.48           0.91     0.08
6       150    520        559        538        0.22      0.32           0.84     0.11
7       150    480        519        498        0.31      0.18           0.76     0.14
8       150    440        479        458        0.42      0.08           0.68     0.18
9       150    400        439        418        0.56      0.02           0.58     0.23
10      150    300        399        358        0.74      0.00           0.45     0.30
```

**Observations:**
- **Decile 1** (top scores 720-850): Only 2% bad rate, 95% approval rate, positive ROI (1.12)
- **Decile 5** (mid scores 560-599): 15% bad rate, 48% approval rate, slightly negative ROI (0.91)
- **Decile 10** (bottom scores 300-399): 74% bad rate, 0% approval rate (all declined)

**Good Sign:** 
- Clear separation between deciles (bad rate increases as score decreases)
- ROI decreases monotonically with score
- Highest deciles have positive ROI

**Warning Signs to Look For:**
- **Inversion:** Decile 2 has higher bad rate than Decile 3 (non-monotonic)
- **Compression:** All deciles have similar bad rates (model not discriminating)
- **Unexpected losses:** High-score deciles showing ROI < 1.0

---

## Calibration Analysis Interpretation

### Example Calibration Table

```
Bin             Count  Actual_Bads  Observed_Rate  Predicted_Rate  Difference  Calibrated
─────────────────────────────────────────────────────────────────────────────────────────
[0.01, 0.05)    120    3            0.025          0.030          -0.005       Yes
[0.05, 0.10)    180    14           0.078          0.075          +0.003       Yes
[0.10, 0.15)    220    31           0.141          0.125          +0.016       Yes
[0.15, 0.20)    195    41           0.210          0.175          +0.035       Yes
[0.20, 0.30)    180    52           0.289          0.250          +0.039       Yes
[0.30, 0.50)    105    51           0.486          0.400          +0.086       No
```

**Interpretation:**
- **Well-calibrated bins** (difference < 0.05): First 5 bins
- **Miscalibrated bin**: Bin 6 ([0.30, 0.50))
  - Predicted 40% bad rate
  - Observed 49% bad rate
  - Model is under-estimating risk in high-risk segment

**Calibration Error:** Average absolute difference = 0.030 (acceptable if < 0.05)

**Action if Poor Calibration:**
1. Check if training data was imbalanced
2. Consider probability calibration techniques (Platt scaling, isotonic regression)
3. May need to retrain with more recent data
4. Adjust cutoff thresholds based on observed rates

---

## Adjusting Thresholds in Interactive Mode

### Scenario: You're in a Volatile Market

Your industry is experiencing rapid changes. You want stricter thresholds to catch issues earlier.

**In Shiny UI:**

1. Launch the node without `ForceHeadless` flow variable
2. Browser opens with threshold inputs
3. Adjust:
   - PSI Warning: 0.10 → 0.07 (catch smaller shifts)
   - PSI Critical: 0.25 → 0.15 (retrain sooner)
   - AUC Decline Warning: 0.03 → 0.02 (more sensitive)
4. Click "Analyze Performance"
5. Recommendation updates based on new thresholds

**Result:**
- Same data, but now shows "MONITOR" instead of "OK"
- This helps you catch problems earlier in a changing environment

---

## Common Patterns and Actions

### Pattern 1: PSI High, But AUC Stable
```
PSI: 0.22 (WARNING)
AUC Delta: -0.01 (OK)
Bad Rate Delta: +0.01 (OK)
```

**Meaning:** Population has shifted, but model still works well.

**Action:** Monitor closely. Not urgent, but indicates change in applicant pool. May need retraining in 3-6 months.

---

### Pattern 2: PSI Low, But AUC Declining
```
PSI: 0.06 (OK)
AUC Delta: -0.07 (CRITICAL)
Bad Rate Delta: +0.05 (WARNING)
```

**Meaning:** Population looks similar, but model performance degraded. Possible causes:
- Data quality issues (missing features, changed definitions)
- External factor not captured in score (new competitor, regulation)
- Scoring errors (wrong WOE mapping, cutoffs)

**Action:** Urgent investigation. Check data pipeline, scoring logic, recent changes.

---

### Pattern 3: Everything Declining Together
```
PSI: 0.30 (CRITICAL)
AUC Delta: -0.08 (CRITICAL)
Bad Rate Delta: +0.06 (WARNING)
Calibration Error: 0.12 (WARNING)
```

**Meaning:** Complete model breakdown. Market has changed significantly.

**Action:** **IMMEDIATE RETRAINING REQUIRED**. Gather new data, rebuild model, validate thoroughly.

---

## Integration with Existing Workflow

### Full Credit Scoring Pipeline

```
┌─────────────────┐
│  Raw Data       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Attribute Editor│  ← Configure variable types
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  WOE Editor     │  ← Create WOE bins
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Variable Select │  ← Select best variables
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Logistic Regr.  │  ← Fit model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scorecard Gen.  │  ← Create scorecard
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scorecard Apply │  ← Score applications
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Analyzer  │  ← Initial validation (development)
└─────────────────┘

         ... Time Passes ...

         ▼
┌─────────────────────────┐
│ Production Data         │  ← 6 months later
│ (with outcomes observed)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Performance       │  ← THIS NODE
│ Monitor                 │  ← Decides: OK, MONITOR, or RETRAIN
└────────┬────────────────┘
         │
         ├─ If OK: Continue monitoring monthly
         ├─ If MONITOR: Investigate, monitor weekly
         └─ If RETRAIN: Loop back to top, rebuild model
```

---

## Tips and Best Practices

### 1. Start with Lenient Thresholds
When first deploying, use default thresholds. After 3-6 months, you'll understand normal variation and can tighten thresholds.

### 2. Store Multiple Baselines
Keep 3 baselines in your workflow:
- **Training baseline:** Original training data (for long-term drift)
- **6-month baseline:** Rolling 6 months (for recent trends)
- **12-month baseline:** Full year (for seasonal patterns)

Compare current month against all three to understand different drift horizons.

### 3. Monthly Monitoring Schedule
- **Week 1:** Wait for outcomes to mature (30-60 days after funding)
- **Week 2:** Run Model Performance Monitor
- **Week 3:** Review results, adjust if needed
- **Week 4:** Present to stakeholders, plan actions

### 4. Document Everything
Save output tables for each month. Build a time series to track:
- AUC over time
- PSI over time
- Bad rate trends
- ROI trends

This helps distinguish temporary blips from sustained degradation.

### 5. Combine with Business Context
A PSI of 0.20 might be acceptable if:
- You launched a new marketing campaign (expected population shift)
- Model still discriminates well (AUC stable)
- Business outcomes are positive (ROI improving)

Context matters! Don't retrain just because a metric crossed a threshold.

---

## Troubleshooting Common Issues

### Issue: "Only 250 funded loans (minimum recommended: 500)"

**Cause:** Not enough funded loans with observed outcomes yet.

**Solutions:**
1. Wait another month to accumulate more data
2. Lower `MinSampleSize` threshold (with caution)
3. Use quarterly instead of monthly monitoring initially

---

### Issue: Calibration table is empty

**Cause:** Insufficient data per probability bin.

**Solution:** Need at least 10-20 observations per bin. Wait for more data or reduce number of calibration bins (currently fixed at 10 in code).

---

### Issue: PSI shows as "N/A"

**Cause:** No baseline data provided.

**Solution:** Connect baseline data to Input Port 2. Cannot calculate population drift without something to compare to.

---

### Issue: Recommendation is "RETRAIN" on first month

**Cause:** Likely issue with data or configuration.

**Check:**
1. Are `Points`, `Odds`, `PDO` correct (matching scorecard)?
2. Is baseline data actually training data (not corrupted production data)?
3. Are outcomes correctly coded (0/1)?
4. Is ROI column present and properly calculated?

---

## Next Steps

After using this node for 3-6 months:
1. Analyze trends: Are metrics stable? Degrading? Improving?
2. Adjust thresholds: Tighten if too many false alarms, loosen if missing issues
3. Integrate with scheduling: Automate monthly runs
4. Build dashboard: Track metrics over time with visualizations
5. Plan retraining cycle: Every 12 months? When recommendation says "RETRAIN"?

---

**End of Usage Examples**

# Baseline Data Strategy for Model Performance Monitor

## The Challenge

You need to compare production model performance against a reference point to detect drift and degradation.

## Recommended Approach: **Dual-Baseline Strategy**

### Option A: Training/Validation Baseline (Initial Reference)
**Use Case:** First deployment, establishing expected behavior

**Data:** The original training/validation set used to build the model
- Contains known good/bad outcomes
- Model was optimized for this distribution
- Represents the "ideal" scenario

**Metrics Available:**
- Full discrimination metrics (AUC, K-S, Gini) 
- Complete calibration analysis
- ROI analysis for funded loans

**When to Use:**
- Right after model deployment
- When no production history exists yet
- For detecting initial population drift

---

### Option B: Rolling Production Baseline (Recent Stable Period)
**Use Case:** Ongoing monitoring after model has been in production

**Data:** Most recent 6-12 months of stable production data
- Large enough sample for statistical significance
- Recent enough to be relevant
- Excludes any known anomaly periods (e.g., COVID disruption, economic crisis)

**Why 6-12 months?**
- **6 months minimum:** Sufficient sample size (~10k+ funded loans), captures seasonal patterns
- **12 months ideal:** Full business cycle, more robust statistics
- **Not longer:** Avoids including outdated data that's no longer relevant

**Metrics Available:**
- Full discrimination metrics
- Calibration (if outcomes are mature)
- ROI analysis (if performance period has elapsed)

**When to Use:**
- After 6+ months in production
- When you want to detect recent changes
- For month-over-month drift monitoring

---

## Implementation in the Node

### Input Port Configuration

**Recommended:** Make baseline **optional** with intelligent defaults

```
Input Port 1: Production Data (Current Period) - REQUIRED
Input Port 2: Baseline Data - OPTIONAL
```

### Logic Flow

```python
if baseline_data_provided:
    # Calculate PSI and comparative metrics
    recommendation = compare_to_baseline(current, baseline)
else:
    # First run or standalone mode
    # Still calculate absolute metrics (AUC, K-S, calibration)
    # But skip PSI and delta calculations
    recommendation = evaluate_absolute_metrics(current)
```

### Flow Variable for Baseline Type

```python
BaselineType (str, default: "auto")
  - "training": Baseline is original training data
  - "production": Baseline is recent stable production period  
  - "auto": Node auto-detects based on data characteristics
```

---

## Practical Workflow Recommendations

### Phase 1: Initial Deployment (Months 1-6)
```
Production Data → Monitor Node (no baseline)
↓
Outputs: Absolute metrics only (AUC, bad rate, ROI)
No PSI calculation (nothing to compare to yet)
```

### Phase 2: Establishing Baseline (Month 6)
```
Training Data → Input 2 (Baseline)
Current Production → Input 1
↓
Monitor Node calculates:
- PSI (score distribution shift)
- AUC delta
- Calibration drift
- Recommendation: OK/MONITOR/RETRAIN
```

### Phase 3: Ongoing Monitoring (Month 7+)
```
Months 1-6 Production (stable baseline) → Input 2
Current Month Production → Input 1
↓
Monitor Node detects month-over-month changes
```

### Phase 4: Model Refresh Cycle
After retraining, reset baseline to new training data, restart cycle.

---

## Best Practice: Store Multiple Baselines

**In KNIME workflow:**
1. Store **training baseline** permanently (for long-term drift detection)
2. Store **rolling 12-month baseline** (updated quarterly)
3. Use **current month** as production data

**Monitoring Strategy:**
- **Monthly:** Compare current month vs. rolling baseline → Detect recent drift
- **Quarterly:** Compare rolling baseline vs. training → Detect long-term drift
- **Annual:** Full model review with training baseline comparison

---

## Sample Sizes for Reliability

### Minimum Sample Sizes for Metrics

| Metric | Minimum Funded Loans | Reason |
|--------|---------------------|--------|
| AUC, K-S, Gini | 500 | Statistical stability |
| PSI (score) | 1,000 | Decile-level comparison |
| Calibration | 1,000 | Score band analysis |
| ROI Analysis | 500 | Meaningful averages |
| Bad Rate (by decile) | 100 per decile | 10% bad rate → 10 bads minimum |

**Warning Thresholds:**
- **< 500 funded:** "INSUFFICIENT DATA - Metrics may be unreliable"
- **500-1,000:** "LIMITED DATA - Use caution interpreting results"
- **> 1,000:** Reliable metrics

---

## My Recommendation for Your Node

**Implement a flexible baseline approach:**

1. **Accept baseline as optional Input Port 2**
   - If provided: Full comparative analysis
   - If not provided: Absolute metrics only

2. **Add flow variable: `MinSampleSize` (default: 500)**
   - Warn user if sample too small

3. **Add flow variable: `BaselinePeriodMonths` (default: 6)**
   - Document expected baseline period for context

4. **Auto-detect baseline type:**
   - If baseline approval rate differs significantly from production: likely training data
   - If similar distributions: likely production data

5. **Store metadata:**
   - Add columns to output indicating baseline period used
   - Store sample sizes for transparency

---

## Decision Tree for Users

```
START
│
├─ First 6 months of production?
│  └─ Yes → No baseline, monitor absolute metrics
│  └─ No → Continue
│
├─ Do you have stable production history?
│  └─ No → Use training data as baseline
│  └─ Yes → Continue
│
├─ Want to detect recent drift?
│  └─ Yes → Use rolling 6-12 month baseline
│  └─ No → Use training data baseline
│
└─ DONE
```

---

## Summary

**For your use case, I recommend:**
- **Accept optional baseline** (Input Port 2)
- **Default to 6-month rolling baseline** after initial deployment
- **Store both training and production baselines** in your KNIME workflow
- **Add sample size warnings** to prevent unreliable conclusions
- **Document baseline period** in output metadata

This gives you maximum flexibility while ensuring robust monitoring.

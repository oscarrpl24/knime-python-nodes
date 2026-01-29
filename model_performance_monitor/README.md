# Model Performance Monitor Node

## Overview

The **Model Performance Monitor** is a comprehensive KNIME Python node designed to track production model performance and detect when retraining is needed. It monitors discrimination metrics, calibration, population stability, and business outcomes.

## Purpose

- **Detect Model Drift**: Identify when the production population differs significantly from training data
- **Track Performance Degradation**: Monitor AUC, K-S, and Gini coefficient over time
- **Assess Calibration**: Verify predicted probabilities match observed outcomes
- **Monitor Business Metrics**: Track approval rates, bad rates, and ROI
- **Automate Retraining Decisions**: Get clear recommendations: OK, MONITOR, or RETRAIN

---

## Inputs

### Input Port 1: Production Data (REQUIRED)
Current period loan data with the following columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `score` | Integer | Scorecard points from scoring model | Yes |
| `isApproved` | Integer | 1 = approved, 0 = declined | Yes |
| `isFunded` | Integer | 1 = funded, 0 = not funded | Yes |
| `<DependentVariable>` | Integer | Actual outcome (0/1) for funded loans only | Yes |
| `ROI` | Float | Return on investment for funded loans (>1 = profit, <1 = loss) | Yes |

### Input Port 2: Baseline Data (OPTIONAL)
Training data or stable historical period with the same structure as Input Port 1.

**When to provide baseline:**
- For PSI (Population Stability Index) calculation
- To compare current vs. historical performance
- To detect drift from training distribution

**When to skip baseline:**
- First deployment (no historical data yet)
- Want absolute metrics only (no comparisons)

---

## Outputs

### Output Port 1: Performance Summary Table
Key metrics with alert status:

| Column | Description |
|--------|-------------|
| `Metric` | Name of the metric |
| `Current` | Current period value |
| `Baseline` | Baseline period value (if provided) |
| `Delta` | Change from baseline |
| `Status` | OK, WARNING, CRITICAL, INFO, or N/A |
| `Alert` | Visual indicator: ✓ (OK), ⚠️ (WARNING), 🔴 (CRITICAL), ℹ️ (INFO) |

**Metrics included:**
- Sample Size (Funded)
- AUC (Area Under ROC Curve)
- Gini Coefficient
- K-S Statistic (Kolmogorov-Smirnov)
- PSI (Population Stability Index) - if baseline provided
- Approval Rate
- Bad Rate (Funded)
- Avg ROI (Funded)
- Calibration Error
- **Recommendation** (OK / MONITOR / RETRAIN)

### Output Port 2: Decile Analysis Table
Performance metrics by score decile:

| Column | Description |
|--------|-------------|
| `decile` | Score decile (1 = highest scores) |
| `count` | Number of applications |
| `score_min` | Minimum score in decile |
| `score_max` | Maximum score in decile |
| `score_avg` | Average score in decile |
| `bad_rate` | Bad rate in funded population |
| `bad_count` | Number of bads |
| `approval_rate` | Approval rate |
| `avg_roi` | Average ROI for funded loans |
| `cumulative_bad_rate` | Cumulative bad rate from top decile |

### Output Port 3: Calibration Table
Expected vs. observed bad rates by probability bin:

| Column | Description |
|--------|-------------|
| `bin` | Probability range |
| `count` | Number of funded loans |
| `actual_bads` | Number of actual bad outcomes |
| `observed_rate` | Observed bad rate |
| `predicted_rate` | Predicted bad rate (from model) |
| `difference` | observed - predicted |
| `calibrated` | True if abs(difference) < 0.05 |

### Output Port 4: Production Data with Diagnostics
Original production data plus:

| Column | Description |
|--------|-------------|
| `probability` | Predicted probability (derived from score) |
| `prediction_correct` | 1 if prediction matches actual (funded loans only) |
| `score_decile` | Score decile (1-10, where 1 = highest scores) |

---

## Flow Variables

### Required

| Variable | Type | Description |
|----------|------|-------------|
| `DependentVariable` | String | Name of actual outcome column (e.g., "isBad") |

### Column Names (Optional)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ScoreColumn` | String | `"score"` | Name of scorecard score column |
| `ApprovalColumn` | String | `"isApproved"` | Name of approval decision column |
| `FundedColumn` | String | `"isFunded"` | Name of funded indicator column |
| `ROIColumn` | String | `"ROI"` | Name of ROI column |

### Scorecard Parameters (for Probability Calculation)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `Points` | Integer | `600` | Base score at target odds |
| `Odds` | Integer | `20` | Target odds ratio (1:X, e.g., 20 = 1:19 odds) |
| `PDO` | Integer | `50` | Points to Double the Odds |

**Note:** These should match the parameters used in your Scorecard node.

### Alert Thresholds

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PSI_Warning` | Float | `0.1` | PSI threshold for "MONITOR" status |
| `PSI_Critical` | Float | `0.25` | PSI threshold for "RETRAIN" status |
| `AUC_Decline_Warning` | Float | `0.03` | AUC decline for "MONITOR" (e.g., 0.03 = 3% decline) |
| `AUC_Decline_Critical` | Float | `0.05` | AUC decline for "RETRAIN" |
| `KS_Decline_Warning` | Float | `0.05` | K-S decline for "MONITOR" |
| `KS_Decline_Critical` | Float | `0.10` | K-S decline for "RETRAIN" |
| `BadRate_Increase_Warning` | Float | `0.02` | Bad rate increase threshold (e.g., 0.02 = 2% increase) |
| `CalibrationError_Warning` | Float | `0.05` | Calibration error threshold for "MONITOR" |
| `MinSampleSize` | Integer | `500` | Minimum funded loans for reliable metrics |

### Other

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ForceHeadless` | Boolean | `False` | If True, skip interactive UI even when available |

---

## Modes of Operation

### Interactive Mode (Default)
When Shiny is available and `ForceHeadless=False`:
- Opens browser-based UI
- Adjust alert thresholds in real-time
- See how recommendations change with different thresholds
- View summary, decile, and calibration tables
- Click "Save Results & Close" to finalize

### Headless Mode
When Shiny is not available or `ForceHeadless=True`:
- Runs automatically with configured thresholds
- No user interaction required
- Suitable for scheduled workflows

---

## Key Metrics Explained

### PSI (Population Stability Index)
Measures how much the score distribution has shifted from baseline.

**Formula:** `PSI = Σ (actual% - expected%) × ln(actual% / expected%)`

**Interpretation:**
- **< 0.1**: Stable, no significant change
- **0.1 - 0.25**: Moderate shift, monitor closely
- **≥ 0.25**: Significant shift, investigate and consider retraining

**Example:**
If your training data had average score of 650, but production data averages 580, PSI will be high, indicating the incoming population is riskier than expected.

### AUC (Area Under ROC Curve)
Measures the model's ability to discriminate between goods and bads.

**Range:** 0.5 (random) to 1.0 (perfect)

**Typical for credit scoring:** 0.65-0.85

**Decline warning:** If AUC drops by 3-5%, the model may be losing predictive power.

### K-S Statistic (Kolmogorov-Smirnov)
Maximum separation between cumulative good and bad distributions.

**Range:** 0.0 (no separation) to 1.0 (perfect separation)

**Typical for credit scoring:** 0.30-0.60

### Gini Coefficient
Related to AUC: `Gini = 2 × AUC - 1`

**Range:** 0.0 (random) to 1.0 (perfect)

**Typical for credit scoring:** 0.30-0.70

### Calibration Error
Mean absolute difference between predicted and observed bad rates across probability bins.

**Example:** 
- Bin 1: Predicted 10% bads, Observed 12% → Difference = 0.02
- Bin 2: Predicted 20% bads, Observed 18% → Difference = 0.02
- Calibration Error = 0.02

**Threshold:** If > 0.05, model probabilities may need recalibration.

---

## Recommendation Logic

The node produces one of three recommendations:

### ✅ OK
Model is performing well, no action needed.

**Criteria:**
- PSI < 0.1 (or no baseline)
- AUC decline < 3%
- K-S decline < 5%
- Bad rate increase < 2%
- Calibration error < 5%

### ⚠️ MONITOR
Model showing signs of degradation, watch closely.

**Triggered by:**
- PSI between 0.1 and 0.25
- AUC decline between 3% and 5%
- K-S decline between 5% and 10%
- Bad rate increase ≥ 2%
- Calibration error ≥ 5%

**Action:** Increase monitoring frequency, investigate causes.

### 🔴 RETRAIN
Model degradation is significant, retraining recommended.

**Triggered by:**
- PSI ≥ 0.25 (significant population shift)
- AUC decline ≥ 5%
- K-S decline ≥ 10%

**Action:** Gather fresh data, retrain model, validate before deployment.

---

## Usage Example

### KNIME Workflow

```
[Production Data] ──────┐
                        │
                        ├──> [Model Performance Monitor] ──┐
                        │                                   │
[Baseline/Training] ────┘                                   │
                                                            │
                                            ┌───────────────┼───────────────┐
                                            │               │               │
                                            ▼               ▼               ▼
                                    [Summary Table]  [Decile Table]  [Calibration]
```

### Flow Variables Configuration

```
DependentVariable: "isBad"
ScoreColumn: "score"
ApprovalColumn: "isApproved"
FundedColumn: "isFunded"
ROIColumn: "ROI"

Points: 600
Odds: 20
PDO: 50

PSI_Warning: 0.1
PSI_Critical: 0.25
AUC_Decline_Warning: 0.03
```

### Interpretation

**Scenario 1: Healthy Model**
```
AUC: 0.75 (baseline: 0.76, delta: -0.01)  ✓ OK
PSI: 0.08  ✓ OK
Calibration Error: 0.03  ✓ OK
Recommendation: OK
```

**Scenario 2: Population Shift**
```
AUC: 0.74 (baseline: 0.76, delta: -0.02)  ✓ OK
PSI: 0.18  ⚠️ WARNING
Calibration Error: 0.04  ✓ OK
Recommendation: MONITOR
```
→ Incoming applications are different from training data, but model still performing okay.

**Scenario 3: Model Degradation**
```
AUC: 0.68 (baseline: 0.76, delta: -0.08)  🔴 CRITICAL
PSI: 0.32  🔴 CRITICAL
Calibration Error: 0.09  ⚠️ WARNING
Recommendation: RETRAIN
```
→ Model is no longer discriminating well, population has shifted significantly. Time to retrain.

---

## Baseline Data Strategy

See [BASELINE_STRATEGY.md](BASELINE_STRATEGY.md) for comprehensive guidance.

### Quick Recommendations

**First 6 months of production:**
- Don't provide baseline (none available yet)
- Monitor absolute metrics (AUC, bad rate, ROI)
- Build confidence in your thresholds

**After 6 months:**
- Use training data as baseline initially
- Transition to rolling 6-12 month production baseline
- Compare current month vs. baseline monthly

**Best Practice:**
Store both training baseline (long-term drift) and rolling production baseline (recent drift) in your workflow.

---

## Sample Size Warnings

The node checks if you have sufficient data for reliable metrics:

| Funded Loans | Status | Reliability |
|--------------|--------|-------------|
| < 500 | ⚠️ WARNING | Metrics may be unreliable |
| 500 - 1,000 | ℹ️ LIMITED | Use caution interpreting results |
| > 1,000 | ✓ OK | Reliable metrics |

**For decile analysis:** Need at least 100 loans per decile (1,000 total) for stable bad rates.

---

## Troubleshooting

### "DependentVariable flow variable is required"
**Solution:** Set the `DependentVariable` flow variable to the name of your actual outcome column (e.g., "isBad", "isDefault").

### "Only X funded loans (minimum recommended: 500)"
**Solution:** 
- Accumulate more data before running analysis
- Lower `MinSampleSize` threshold (with caution)
- Use warnings as guidance, not hard stops

### PSI shows as "N/A"
**Solution:** Provide baseline data via Input Port 2. PSI cannot be calculated without a comparison distribution.

### Calibration table is empty
**Solution:** Need at least 10 funded loans with known outcomes. Calibration requires sufficient data per bin.

### Shiny UI doesn't open
**Solution:**
- Check if port 8054 is available
- Set `ForceHeadless=True` to run without UI
- Check console for error messages

---

## Technical Details

### Score to Probability Conversion

The node converts scorecard points back to probabilities using the inverse formula:

```
Given:
  b = PDO / log(2)
  a = Points + b * log(odds0)
  odds0 = 1 / (odds - 1)

Then:
  log_odds = (a - score) / b
  probability = 1 / (1 + exp(-log_odds))
```

**Example with defaults (Points=600, Odds=20, PDO=50):**
- Score 650 → Probability ≈ 0.05 (5% bad rate)
- Score 600 → Probability ≈ 0.10 (10% bad rate)
- Score 550 → Probability ≈ 0.20 (20% bad rate)

This allows the node to calculate AUC, K-S, and calibration without needing the original logistic regression model.

---

## Limitations

### Current Version (1.0)
- **Slack notifications:** Not implemented yet (Phase 4)
- **Reject inference integration:** Not included (use separate node)
- **Custom threshold rules:** Limited to predefined logic
- **Historical tracking:** Single period comparison (no time series)

### Data Requirements
- Requires funded loans with known outcomes for discrimination metrics
- Cannot calculate PSI without baseline data
- Small samples (< 500) produce unreliable metrics

### Assumptions
- Score follows the standard scorecard formula
- Outcomes are binary (0/1)
- ROI is available for funded loans
- All funded loans eventually have outcomes

---

## Version History

### Version 1.0 (2026-01-28)
- Initial release
- Core metrics: AUC, Gini, K-S, PSI, calibration
- Business metrics: approval rate, bad rate, ROI
- Decile analysis
- Interactive Shiny UI with threshold adjustment
- Automatic recommendation logic
- Baseline comparison support

---

## Future Enhancements (Roadmap)

### Phase 2: Reject Inference Integration
- Expected ROI for declined applications
- Threshold sensitivity analysis
- "What-if" approval rate scenarios

### Phase 4: Slack Notifications
- Automatic alerts when thresholds breached
- Configurable alert levels (WARNING, CRITICAL, ALL)
- Webhook integration

### Phase 5: Time Series Tracking
- Store historical performance in database
- Trend charts (AUC over time, PSI over time)
- Automated reports

### Phase 6: Advanced Analytics
- Characteristic Stability Index (CSI) for individual variables
- Variable importance drift
- Segment-level performance (e.g., by geography, product)

---

## Support

For issues, questions, or feature requests:
1. Check COMPREHENSIVE_DEVELOPMENT_LOG.md for known issues
2. Review BASELINE_STRATEGY.md for baseline guidance
3. Consult CONTEXT.md for integration with other nodes

---

## License

Part of the KNIME Credit Risk Modeling Toolkit.
Developed using Claude Sonnet 4.5 via Cursor IDE.

---

**End of Documentation**

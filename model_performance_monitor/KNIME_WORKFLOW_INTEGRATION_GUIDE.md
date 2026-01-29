# KNIME Workflow Integration Guide
## Model Performance Monitor

---

## Table of Contents

1. [Overview](#overview)
2. [Development Phase Integration](#development-phase-integration)
3. [Production Phase Integration](#production-phase-integration)
4. [Complete End-to-End Workflow](#complete-end-to-end-workflow)
5. [Configuration Patterns](#configuration-patterns)
6. [Automated Monitoring Workflows](#automated-monitoring-workflows)
7. [Troubleshooting Integration Issues](#troubleshooting-integration-issues)
8. [Best Practices](#best-practices)

---

## Overview

The **Model Performance Monitor** integrates into your KNIME workflow **after** model deployment, during the **production monitoring phase**. It complements the existing development nodes but serves a different purpose.

### Workflow Phases

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: MODEL DEVELOPMENT (One-time)                      │
├─────────────────────────────────────────────────────────────┤
│ Attribute Editor → WOE Editor → Variable Selection →       │
│ Logistic Regression → Scorecard Generator →                │
│ Model Analyzer (validates initial performance)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    MODEL DEPLOYED
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: PRODUCTION MONITORING (Recurring)                 │
├─────────────────────────────────────────────────────────────┤
│ Production Data (6 months later) →                         │
│ Model Performance Monitor (checks if model still good)     │
│   → If OK: Continue                                         │
│   → If MONITOR: Investigate                                 │
│   → If RETRAIN: Loop back to Phase 1                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Phase Integration

### Existing Development Workflow

Your current workflow for model development:

```
┌────────────────────┐
│  Raw Data          │
│  (historical loans)│
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Attribute Editor   │ ← Configure variable types
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ WOE Editor         │ ← Create WOE bins (Output 4: bins table)
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Variable Selection │ ← Select best variables
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Logistic Regression│ ← Fit model (Output 2: coefficients)
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Scorecard Generator│ ← Create scorecard (Output 1: scorecard table)
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Model Analyzer     │ ← Validate initial performance
└────────────────────┘   (Use ONCE during development)
```

### Key Outputs to Save

Before deployment, **save these outputs** for later use:

1. **Training Data** (with scores) - Becomes your baseline
2. **WOE Bins Table** - For scoring new applications
3. **Scorecard Table** - For scoring new applications
4. **Coefficients Table** - For future reference

**KNIME Pattern:**
```
┌────────────────────┐
│ Scorecard Apply    │
└──────┬─────────────┘
       │
       ├─────────────────┐
       │                 ▼
       │         ┌─────────────────┐
       │         │ CSV Writer      │ ← Save training data with scores
       │         │ "baseline.csv"  │
       │         └─────────────────┘
       │
       └─────────────────┐
                         ▼
                 ┌─────────────────┐
                 │ CSV Writer      │ ← Save for production scoring
                 │ "scorecard.csv" │
                 └─────────────────┘
```

---

## Production Phase Integration

### Scenario 1: Simple Monthly Monitoring

**Use Case:** Check model health once per month

```
┌────────────────────┐
│ Production Data    │ ← Last month's loans with observed outcomes
│ (current month)    │
└──────┬─────────────┘
       │
       ├──────────────────┐
       │                  ▼
       │          ┌─────────────────┐
       │          │ CSV Reader      │ ← Load training baseline
       │          │ "baseline.csv"  │
       │          └────────┬────────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────┐
│ Model Performance Monitor            │
│                                      │
│ Flow Variables:                      │
│   DependentVariable: "isBad"         │
│   ScoreColumn: "score"               │
│   Points: 600, Odds: 20, PDO: 50    │
└──────┬───────────────────────────────┘
       │
       ├───────┬────────┬────────┐
       │       │        │        │
       ▼       ▼        ▼        ▼
   Summary  Decile  Calib.  Diagnostics
    Table   Table   Table    Table
       │       │        │        │
       └───────┴────────┴────────┘
                   │
                   ▼
         ┌──────────────────┐
         │ CSV Writer       │ ← Save monthly report
         │ "monitor_202601" │
         └──────────────────┘
```

**KNIME Setup:**
1. Add **CSV Reader** for production data
2. Add **CSV Reader** for baseline data
3. Add **Model Performance Monitor** node
4. Configure flow variables (see Configuration Patterns section)
5. Add **CSV Writers** to save outputs

---

### Scenario 2: Integrated Scoring + Monitoring

**Use Case:** Score new applications AND monitor model health

```
┌────────────────────┐
│ New Applications   │ ← Raw application data
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Data Preprocessing │ ← Apply same transformations as training
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ WOE Apply          │ ← Apply WOE transformations (using saved bins)
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Scorecard Apply    │ ← Calculate scores
└──────┬─────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌────────────────┐    ┌────────────────┐
│ Decision Engine│    │ Wait for       │ ← Wait 30-60 days for outcomes
│ (approve/      │    │ Outcomes       │
│  decline)      │    └────────┬───────┘
└────────────────┘             │
                               ▼
                        ┌──────────────────┐
                        │ Join Outcomes    │ ← Merge observed outcomes
                        └────────┬─────────┘
                                 │
                                 ├──────────────────┐
                                 │                  ▼
                                 │          ┌─────────────────┐
                                 │          │ CSV Reader      │
                                 │          │ "baseline.csv"  │
                                 │          └────────┬────────┘
                                 │                   │
                                 ▼                   ▼
                          ┌──────────────────────────────────┐
                          │ Model Performance Monitor        │
                          └──────┬───────────────────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ CSV Writer   │ ← Monthly monitoring report
                          └──────────────┘
```

**Key Points:**
- Score applications immediately
- Store outcomes separately
- Join outcomes when mature (30-60 days later)
- Run monitoring monthly or quarterly

---

### Scenario 3: Rolling Baseline Strategy

**Use Case:** Compare against recent stable period, not original training

```
┌────────────────────┐
│ Current Month      │ ← January 2026
│ Production Data    │
└──────┬─────────────┘
       │
       ├────────────────────────┐
       │                        ▼
       │                ┌─────────────────┐
       │                │ CSV Reader      │ ← July-Dec 2025 (6 months)
       │                │ "rolling_       │   (becomes new baseline)
       │                │  baseline.csv"  │
       │                └────────┬────────┘
       │                         │
       ▼                         ▼
┌──────────────────────────────────────┐
│ Model Performance Monitor            │
└──────┬───────────────────────────────┘
       │
       ▼
  [Outputs...]
       │
       ▼
┌────────────────────────────────────────┐
│ Concatenate (Row Append)               │ ← Add current month to history
│ Rolling Baseline + Current Month       │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────┐
│ Row Filter         │ ← Keep only last 6 months
│ (by date)          │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ CSV Writer         │ ← Update rolling baseline
│ "rolling_baseline" │
└────────────────────┘
```

**Benefits:**
- Baseline adapts to "new normal"
- Detects recent changes (month-over-month)
- More relevant than 2-year-old training data

---

## Complete End-to-End Workflow

### Full Lifecycle: Development → Deployment → Monitoring → Retraining

```
┌═══════════════════════════════════════════════════════════════┐
║ PHASE 1: MODEL DEVELOPMENT (January 2024)                    ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────┐
│ Historical  │ (Jan 2023 - Dec 2023 data)
│ Loan Data   │
└──────┬──────┘
       │
[Attribute Editor → WOE Editor → Variable Selection →
 Logistic Regression → Scorecard Generator → Model Analyzer]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ SAVE ARTIFACTS:                                             │
│  1. training_data_scored.csv (baseline for monitoring)     │
│  2. woe_bins.csv (for production scoring)                  │
│  3. scorecard.csv (for production scoring)                 │
│  4. coefficients.csv (for reference)                       │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌═══════════════════════════════════════════════════════════════┐
║ DEPLOYMENT: Model goes live (February 2024)                  ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────┐
│ New Loan    │
│ Applications│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Scorecard   │ ← Using saved bins and scorecard
│ Apply       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Approval    │ ← Based on score thresholds
│ Decision    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Store with  │ ← Wait for outcomes to mature
│ Loan ID     │   (30-60 days)
└─────────────┘

       ↓ (Time passes: 6 months)

┌═══════════════════════════════════════════════════════════════┐
║ PHASE 2: FIRST MONITORING (August 2024)                      ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────┐
│ Production Data     │ ← Feb-July 2024 (with outcomes)
│ (6 months)          │
└──────┬──────────────┘
       │
       ├──────────────────────┐
       │                      ▼
       │              ┌─────────────────────┐
       │              │ Training Baseline   │ ← Jan 2023 - Dec 2023
       │              │ (baseline.csv)      │
       │              └──────────┬──────────┘
       │                         │
       ▼                         ▼
┌──────────────────────────────────────────┐
│ Model Performance Monitor                │
│                                          │
│ Result: PSI = 0.08, AUC = 0.74          │
│ Recommendation: OK ✓                     │
└──────────────────────────────────────────┘
       │
       ▼
Continue using model monthly monitoring

       ↓ (Time passes: 6 more months)

┌═══════════════════════════════════════════════════════════════┐
║ PHASE 3: ONGOING MONITORING (February 2025)                  ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────┐
│ Production Data     │ ← Jan 2025 (current month)
│ (current month)     │
└──────┬──────────────┘
       │
       ├──────────────────────┐
       │                      ▼
       │              ┌─────────────────────┐
       │              │ Rolling Baseline    │ ← July 2024 - Dec 2024
       │              │ (last 6 months)     │   (more recent)
       │              └──────────┬──────────┘
       │                         │
       ▼                         ▼
┌──────────────────────────────────────────┐
│ Model Performance Monitor                │
│                                          │
│ Result: PSI = 0.22, AUC = 0.70          │
│ Recommendation: MONITOR ⚠️                │
└──────────────────────────────────────────┘
       │
       ▼
Investigate, increase monitoring frequency

       ↓ (Time passes: 3 more months)

┌═══════════════════════════════════════════════════════════════┐
║ PHASE 4: RETRAINING TRIGGER (May 2025)                       ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────┐
│ Production Data     │ ← April 2025
│ (current month)     │
└──────┬──────────────┘
       │
       ├──────────────────────┐
       │                      ▼
       │              ┌─────────────────────┐
       │              │ Rolling Baseline    │ ← Oct 2024 - Mar 2025
       │              └──────────┬──────────┘
       │                         │
       ▼                         ▼
┌──────────────────────────────────────────┐
│ Model Performance Monitor                │
│                                          │
│ Result: PSI = 0.28, AUC = 0.67          │
│ Recommendation: RETRAIN 🔴               │
└──────────────────────────────────────────┘
       │
       ▼
┌═══════════════════════════════════════════════════════════════┐
║ LOOP BACK TO PHASE 1: Rebuild Model                          ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────┐
│ New Training│ ← Use last 12 months of production data
│ Data        │   (May 2024 - April 2025)
└──────┬──────┘
       │
       ▼
[Repeat Phase 1: Attribute Editor → WOE Editor → ... → Scorecard]
       │
       ▼
Deploy new model, reset monitoring cycle
```

---

## Configuration Patterns

### Pattern 1: Using Preset Configurations

**Recommended for most users**

```
┌─────────────────────────┐
│ Monitor Config Generator│ ← Python Script node
│ (preset: "standard")    │
└──────────┬──────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Table Row to Variable Loop Start      │ ← Converts table to flow vars
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ [Your workflow with Monitor node]     │
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Loop End                               │
└────────────────────────────────────────┘
```

**Steps:**
1. Add `monitor_config_generator.py` as Python Script node
2. Set `PRESET = "standard"` (or other preset) at top of script
3. Connect to "Table Row to Variable Loop Start"
4. Variables automatically available in loop
5. Connect your Monitor node inside loop

**Presets Available:**
- `"conservative"` - Relaxed thresholds, stable environments
- `"standard"` - Balanced, recommended for most (DEFAULT)
- `"aggressive"` - Strict thresholds, catch issues early
- `"early_warning"` - Maximum sensitivity for critical apps
- `"stable_environment"` - Minimal false alarms
- `"volatile_market"` - Adaptive to rapid changes
- `"custom"` - Your own thresholds

---

### Pattern 2: Manual Flow Variables

**For fine-grained control**

```
┌──────────────────────────────┐
│ Create Flow Variables        │ ← Configuration node
│                              │
│ Variables:                   │
│   DependentVariable = "isBad"│
│   ScoreColumn = "score"      │
│   Points = 600               │
│   Odds = 20                  │
│   PDO = 50                   │
│   PSI_Warning = 0.1          │
│   PSI_Critical = 0.25        │
│   ... (all thresholds)       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Model Performance Monitor    │
└──────────────────────────────┘
```

**KNIME Steps:**
1. Right-click workspace → "Workflow Variables"
2. Add each variable with correct type:
   - Strings: DependentVariable, ScoreColumn, etc.
   - Integers: Points, Odds, PDO, MinSampleSize
   - Doubles: All threshold values (PSI_Warning, etc.)
3. Model Performance Monitor reads variables automatically

---

### Pattern 3: Environment-Specific Configurations

**For dev/staging/production environments**

```
┌──────────────────────────────┐
│ Load Config File             │ ← CSV Reader or JSON Reader
│ (config_production.csv)      │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Table Row to Variable        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Model Performance Monitor    │
└──────────────────────────────┘
```

**Config Files:**
- `config_dev.csv` - Lenient thresholds for development
- `config_staging.csv` - Standard thresholds for testing
- `config_production.csv` - Production thresholds

**Benefits:**
- Version control your configurations
- Easy to switch environments
- Auditability

---

## Automated Monitoring Workflows

### Scenario 1: Monthly Scheduled Monitoring

**KNIME Server Setup**

```
┌──────────────────────────────────────────────────────────┐
│ KNIME Server Scheduler: First Monday of Each Month      │
└──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Load Production Data (Previous Month)                │
│    - From database or file share                         │
│    - Filter to funded loans with mature outcomes        │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Load Baseline Data                                   │
│    - Rolling 6-month baseline OR training data          │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 3. Model Performance Monitor                             │
│    - With "standard" preset                              │
└──────────┬───────────────────────────────────────────────┘
           │
           ├────────┬─────────┬────────┐
           │        │         │        │
           ▼        ▼         ▼        ▼
       Summary  Decile   Calib.   Diagnostics
           │        │         │        │
           └────────┴─────────┴────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│ 4. Email Report Generator                                │
│    - If recommendation = "OK": Info email                │
│    - If recommendation = "MONITOR": Warning email        │
│    - If recommendation = "RETRAIN": Alert email          │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 5. Save Reports                                          │
│    - CSV files to shared folder                          │
│    - Archive in database                                 │
│    - Dashboard update                                    │
└────────────────────────────────────────────────────────────┘
```

**KNIME Nodes Used:**
- Database Connector
- Database Reader (for production data)
- Model Performance Monitor
- Email Sender (conditional)
- CSV Writer
- Database Writer

---

### Scenario 2: Real-Time Monitoring with Alerts

**For critical applications**

```
┌──────────────────────────────────────────────────────────┐
│ Stream Processing Loop (Every 24 hours)                 │
└──────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Incremental Load                                      │
│    - New funded loans with outcomes from yesterday      │
│    - Append to running production dataset               │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Check Sample Size                                     │
│    - If >= 500 funded: Run monitor                       │
│    - If < 500: Wait for more data                        │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼ (if sufficient data)
┌──────────────────────────────────────────────────────────┐
│ 3. Model Performance Monitor                             │
│    - With "aggressive" or "early_warning" preset         │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ 4. Decision Logic                                        │
│    - IF Recommendation = "RETRAIN":                      │
│        → Trigger retraining workflow                     │
│        → Disable auto-decisions (switch to manual)       │
│        → Send urgent alert                               │
│    - IF Recommendation = "MONITOR":                      │
│        → Increase monitoring frequency                   │
│        → Send warning alert                              │
│    - IF Recommendation = "OK":                           │
│        → Continue normal operations                      │
└────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting Integration Issues

### Issue 1: "DependentVariable flow variable is required"

**Cause:** Flow variable not set

**Solution:**
```
Add flow variable configuration:

Method A: Using Config Generator
  ┌─────────────────┐
  │ Config Generator│
  └────────┬────────┘
           │
           ▼
  ┌──────────────────────┐
  │ Table Row to Variable│
  └────────┬─────────────┘
           │
           ▼
  ┌────────────────┐
  │ Monitor Node   │
  └────────────────┘

Method B: Manual
  Right-click workspace → Workflow Variables
  Add: DependentVariable (String) = "isBad"
```

---

### Issue 2: Baseline data has different columns than production

**Cause:** Schema mismatch between baseline and production data

**Solution:**
```
┌─────────────┐     ┌─────────────┐
│ Production  │     │ Baseline    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌────────────────────────────────┐
│ Column Filter                  │ ← Keep only common columns
│ Include: score, isApproved,    │
│          isFunded, isBad, ROI  │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────┐
│ Monitor Node       │
└────────────────────┘
```

---

### Issue 3: Scorecard parameters don't match

**Cause:** Points/Odds/PDO different from Scorecard Generator

**Solution:**
1. Check Scorecard Generator flow variables used
2. Update Monitor flow variables to match:
   ```
   Points: 600  (must match Scorecard Generator)
   Odds: 20     (must match Scorecard Generator)
   PDO: 50      (must match Scorecard Generator)
   ```
3. Or read from saved configuration:
   ```
   ┌─────────────────┐
   │ Read Scorecard  │ ← CSV with scorecard metadata
   │ Metadata        │
   └────────┬────────┘
            │
            ▼
   ┌────────────────────┐
   │ Extract Parameters │ ← Parse Points, Odds, PDO
   └────────┬───────────┘
            │
            ▼
   ┌────────────────┐
   │ Monitor Node   │
   └────────────────┘
   ```

---

### Issue 4: Node runs but gives "INSUFFICIENT DATA" warning

**Cause:** < 500 funded loans in production data

**Solution:**
```
Option A: Wait for more data
  - Accumulate 2-3 months of production data
  - Run quarterly instead of monthly initially

Option B: Lower threshold (with caution)
  - Set MinSampleSize = 200 (flow variable)
  - Be aware metrics may be less reliable

Option C: Use only when sufficient
  ┌─────────────────┐
  │ Production Data │
  └────────┬────────┘
           │
           ▼
  ┌────────────────────────┐
  │ Row Filter             │ ← Filter to isFunded = 1
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │ Row Counter            │
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │ Rule Engine            │ ← IF count >= 500:
  │ (conditional execution)│     run Monitor
  └────────┬───────────────┘   ELSE: skip
           │
           ▼
  ┌────────────────┐
  │ Monitor Node   │
  └────────────────┘
```

---

## Best Practices

### 1. Version Control Your Configurations

**Store in Git:**
```
project/
├── configs/
│   ├── monitor_config_standard.csv
│   ├── monitor_config_conservative.csv
│   └── scorecard_params.csv
├── baselines/
│   ├── training_baseline_2024.csv
│   └── rolling_baseline_latest.csv
└── workflows/
    └── monthly_monitoring.knwf
```

### 2. Document Your Decisions

**Create a monitoring log:**
```
monitoring_log.csv:
  Date,       Recommendation, Action_Taken,           Reason
  2024-02-01, OK,             Continue,               All metrics healthy
  2024-03-01, MONITOR,        Increase freq,          PSI = 0.18
  2024-04-01, MONITOR,        Investigate,            Bad rate up 3%
  2024-05-01, RETRAIN,        Initiated retrain,      PSI = 0.28, AUC down 7%
```

### 3. Test with Historical Data First

**Before production:**
```
1. Load last 12 months of historical data
2. Split into months
3. Run Monitor on each month (using month 1 as baseline)
4. Verify recommendations make sense
5. Adjust thresholds if needed
6. Document final configuration
```

### 4. Start Conservative, Tighten Over Time

**Threshold evolution:**
```
Months 1-3:   "conservative" preset
              → Learn baseline variation
              → Understand false alarm rate

Months 4-6:   "standard" preset
              → More sensitive monitoring
              → Build confidence

Months 7+:    Custom tuned thresholds
              → Optimized for your use case
              → Documented rationale
```

### 5. Automate Report Distribution

**KNIME Email Node Configuration:**
```
To: model_team@company.com
Subject: [Model Monitor] Monthly Report - {{recommendation}}

Body:
  Model Performance Report - {{date}}
  
  Recommendation: {{recommendation}}
  
  Key Metrics:
  - PSI: {{psi}}
  - AUC: {{auc}} (baseline: {{auc_baseline}})
  - Bad Rate: {{bad_rate}}% (baseline: {{bad_rate_baseline}}%)
  
  {{if recommendation == "RETRAIN"}}
    ⚠️ ACTION REQUIRED: Model retraining recommended
    Please review attached diagnostic reports.
  {{endif}}
  
  Attached:
  - summary_table.csv
  - decile_analysis.csv
  - calibration_table.csv
```

---

## Conclusion

The Model Performance Monitor integrates seamlessly into your KNIME workflow, providing ongoing production model validation. Key integration points:

1. **Save artifacts** from development phase (baseline data, scorecard params)
2. **Schedule monthly** monitoring workflows
3. **Use presets** initially, tune over time
4. **Automate alerts** based on recommendations
5. **Document decisions** for auditability
6. **Loop back** to development when "RETRAIN" recommended

By following these patterns, you'll have a robust model monitoring system that catches degradation early and guides retraining decisions with data-driven recommendations.

---

**For more details, see:**
- README.md - Complete technical reference
- BASELINE_STRATEGY.md - Baseline selection guidance
- USAGE_EXAMPLE.md - Practical scenarios
- monitor_config_generator.py - Preset configurations

---

**End of KNIME Workflow Integration Guide**

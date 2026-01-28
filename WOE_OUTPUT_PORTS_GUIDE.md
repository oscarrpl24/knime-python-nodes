# WOE Editor Output Ports Guide

## Summary
The WOE Editor now has **5 output ports** for maximum flexibility:

```
┌─────────────────────────────────────────────────────────────┐
│ WOE Editor Node (Base & Advanced versions)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Port 1: Original Data (unchanged)                           │
│         - Your input data exactly as provided               │
│         - No transformations applied                        │
│                                                              │
│ Port 2: Full Output (df_with_woe)                          │
│         - Original columns + b_* + WOE_*                    │
│         - Everything in one table                           │
│         - Use when you need all data together               │
│                                                              │
│ Port 3: WOE Only (df_only_woe)                             │
│         - Only WOE_* columns + dependent variable           │
│         - Use for: Logistic Regression, ML models           │
│         - Lean and focused on modeling                      │
│                                                              │
│ Port 4: Bins Only (df_only_bins) ⭐ NEW!                   │
│         - Only b_* columns (binned categories)              │
│         - Use for: Scorecard Apply node                     │
│         - LEAN output for scoring (no WOE, no originals)    │
│         - This is what you need for scoring!                │
│                                                              │
│ Port 5: Bin Rules (bins - metadata)                        │
│         - Binning rules with var, bin, count, WOE, etc.     │
│         - Use for: Documentation, validation, reporting     │
│         - Not actual data - just metadata about bins        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Port 4: The Lean Scorecard Input

**Port 4** contains ONLY the binned columns (b_*) with no extra baggage:

### Example Output Structure:
```
b_Age               | b_Income          | b_State | ...
--------------------|-------------------|---------|
 <= '30'           | > '50000'         | (CA)    |
 > '30' & <= '45'  | > '30000' & ...   | (TX)    |
 ...
```

### Why Port 4 is Perfect for Scoring:
- ✅ **Lean**: Only what you need (b_* columns)
- ✅ **Fast**: No extra WOE or original columns to transfer
- ✅ **Clean**: Matches scorecard expectations exactly
- ✅ **Efficient**: Smaller data size, faster processing

## Typical Workflow

### For Model Building:
```
Raw Data 
  → Attribute Editor 
  → WOE Editor 
     └─ Port 3 (WOE only) → Logistic Regression → Scorecard Generator
```

### For Scoring:
```
New Data 
  → Attribute Editor 
  → WOE Editor 
     └─ Port 4 (Bins only) → Scorecard Apply → Scores!
```

### For Analysis:
```
Raw Data 
  → Attribute Editor 
  → WOE Editor 
     ├─ Port 2 (Full data) → Excel/CSV export
     └─ Port 5 (Bin rules) → Documentation
```

## Quick Reference Table

| Port | Name          | Columns                      | Use Case                    |
|------|---------------|------------------------------|-----------------------------|
| 1    | Original      | Original only                | Passthrough, reference      |
| 2    | Full          | Original + b_* + WOE_*       | All-in-one analysis         |
| 3    | WOE Only      | WOE_* + DV                   | Logistic regression         |
| 4    | Bins Only ⭐  | b_* only                     | **Scorecard Apply**         |
| 5    | Rules         | Metadata (var, bin, woe...)  | Documentation, validation   |

## Fix Your Scorecard Apply Connection

### Before (WRONG):
```
WOE Editor [Port 5 - Bin Rules] ❌ → Scorecard Apply [Port 2]
```
This gives you metadata, not data!

### After (CORRECT):
```
WOE Editor [Port 4 - Bins Only] ✅ → Scorecard Apply [Port 2]
```
This gives you the lean binned data for scoring!

## Column Naming Convention

- **Original columns**: `Age`, `Income`, `State`, etc.
- **Binned columns (b_*)**: `b_Age`, `b_Income`, `b_State`, etc.
- **WOE columns (WOE_*)**: `WOE_Age`, `WOE_Income`, `WOE_State`, etc.

The Scorecard Apply node needs the **b_*** columns to match bin labels with score points.

## Example: What Each Port Contains

Given input variables: `Age`, `Income`, `State` with DV `IsBad`

| Port | Columns Example                                          |
|------|----------------------------------------------------------|
| 1    | `Age`, `Income`, `State`, `IsBad`                        |
| 2    | `Age`, `Income`, `State`, `IsBad`, `b_Age`, `b_Income`, `b_State`, `WOE_Age`, `WOE_Income`, `WOE_State` |
| 3    | `WOE_Age`, `WOE_Income`, `WOE_State`, `IsBad`            |
| 4    | `b_Age`, `b_Income`, `b_State` ⭐                        |
| 5    | `var`, `bin`, `count`, `bads`, `goods`, `woe`, etc.     |

## Benefits of the New Structure

1. **Flexibility**: Choose exactly what you need
2. **Efficiency**: Port 4 is 3-5x smaller than Port 2 for large datasets
3. **Clarity**: Each port has a clear, single purpose
4. **Compatibility**: Port 2 still works for backward compatibility

---

**Updated**: 2026-01-27  
**Version**: 2.0 (5-port structure)

# KNIME Python Script - Reject Inference Column Generation
# Compatible with KNIME 5.9, Python 3.9.23

import knime.scripting.io as knio
import pandas as pd
import numpy as np

# Read input table as pandas DataFrame
df = knio.input_tables[0].to_pandas()

# Helper function to check if a value is null/empty/missing
def is_missing(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    return False

# Create masks for LoanID presence
loan_id_present = df['LoanID'].apply(lambda x: not is_missing(x))
loan_id_missing = ~loan_id_present

# Initialize the new column with pandas nullable integer type (allows missing values)
df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')

# Step 1: When LoanID is NOT null/empty/missing, take value from IsFPD
df.loc[loan_id_present, 'isFPD_wRI'] = df.loc[loan_id_present, 'IsFPD']

# Step 2: When LoanID IS null/empty/missing, take value from FPD
df.loc[loan_id_missing, 'isFPD_wRI'] = df.loc[loan_id_missing, 'FPD']

# Step 3: Handle remaining missing values
# Only process rows where:
#   - LoanID IS missing (rejected applications - need inference)
#   - isFPD_wRI is still missing (FPD didn't have a value)
#   - expected_DefaultRate2 is NOT missing (we have a probability to use)
still_missing_mask = loan_id_missing & df['isFPD_wRI'].isna()
has_expected_rate = df['expected_DefaultRate2'].notna()
rows_to_process = still_missing_mask & has_expected_rate

if rows_to_process.any():
    num_rows = rows_to_process.sum()
    
    # Generate random values between 0 and 1 for each row
    random_values = np.random.random(num_rows)
    
    # Get the expected default rates for these rows
    expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
    
    # If random <= expected_DefaultRate2, then isFPD_wRI = 1, else 0
    assigned_values = (random_values <= expected_rates).astype(int)
    
    df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values

# Ensure the column is nullable integer type
df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')

# =============================================================================
# FRODI26_wRI and GRODI26_wRI Column Generation
# =============================================================================

# Calculate average FRODI26 and GRODI26 grouped by IsFPD (using original IsFPD column)
avg_frodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'FRODI26'].mean()
avg_frodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'FRODI26'].mean()
avg_grodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'GRODI26'].mean()
avg_grodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'GRODI26'].mean()

# Create mask for missing FRODI26 values
frodi26_missing = df['FRODI26'].apply(is_missing)
frodi26_present = ~frodi26_missing

# Initialize FRODI26_wRI with existing FRODI26 values where present
df['FRODI26_wRI'] = pd.NA
df.loc[frodi26_present, 'FRODI26_wRI'] = df.loc[frodi26_present, 'FRODI26']

# For missing FRODI26, use average based on isFPD_wRI value
df.loc[frodi26_missing & (df['isFPD_wRI'] == 1), 'FRODI26_wRI'] = avg_frodi26_fpd1
df.loc[frodi26_missing & (df['isFPD_wRI'] == 0), 'FRODI26_wRI'] = avg_frodi26_fpd0

# Convert to float type
df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')

# Create mask for missing GRODI26 values
grodi26_missing = df['GRODI26'].apply(is_missing)
grodi26_present = ~grodi26_missing

# Initialize GRODI26_wRI with existing GRODI26 values where present
df['GRODI26_wRI'] = pd.NA
df.loc[grodi26_present, 'GRODI26_wRI'] = df.loc[grodi26_present, 'GRODI26']

# For missing GRODI26, use average based on isFPD_wRI value
df.loc[grodi26_missing & (df['isFPD_wRI'] == 1), 'GRODI26_wRI'] = avg_grodi26_fpd1
df.loc[grodi26_missing & (df['isFPD_wRI'] == 0), 'GRODI26_wRI'] = avg_grodi26_fpd0

# Convert to float type
df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')

# =============================================================================
# Reorder Columns
# =============================================================================

# Reorder columns to place new columns after their source columns
cols = df.columns.tolist()

# Remove new columns from their current positions
cols.remove('isFPD_wRI')
cols.remove('FRODI26_wRI')
cols.remove('GRODI26_wRI')

# Insert isFPD_wRI after IsFPD
isfpd_index = cols.index('IsFPD')
cols.insert(isfpd_index + 1, 'isFPD_wRI')

# Insert FRODI26_wRI after FRODI26
frodi26_index = cols.index('FRODI26')
cols.insert(frodi26_index + 1, 'FRODI26_wRI')

# Insert GRODI26_wRI after GRODI26 (account for FRODI26_wRI already inserted)
grodi26_index = cols.index('GRODI26')
cols.insert(grodi26_index + 1, 'GRODI26_wRI')

df = df[cols]

# Output the result (all original columns + new columns)
knio.output_tables[0] = knio.Table.from_pandas(df)


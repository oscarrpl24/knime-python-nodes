# =============================================================================
# KNIME Python Script - Reject Inference Column Generation
# =============================================================================
# 
# PURPOSE:
# This script implements "reject inference" for credit risk modeling.
# In credit scoring, we only observe outcomes (default/no default) for
# applications that were APPROVED. Rejected applications never get loans,
# so we never see their actual outcomes. This creates a biased sample.
#
# Reject inference addresses this by:
# 1. Using actual outcomes for approved applications (those with a LoanID)
# 2. Probabilistically assigning outcomes to rejected applications based
#    on their predicted default probability (expected_DefaultRate2)
#
# This script also imputes missing values for performance metrics (FRODI26,
# GRODI26) using group averages based on the inferred default status.
#
# COMPATIBLE WITH:
# - KNIME Version: 5.9
# - Python Version: 3.9.23
#
# INPUT:
# - Table with columns: LoanID, IsFPD, FPD, expected_DefaultRate2, FRODI26, GRODI26
#
# OUTPUT:
# - Same table with three new columns: isFPD_wRI, FRODI26_wRI, GRODI26_wRI
#
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------

# Import the KNIME scripting interface
# This module provides access to KNIME's input/output tables and flow variables
# It is the bridge between the Python script and the KNIME workflow
import knime.scripting.io as knio

# Import pandas for data manipulation
# pandas is the primary library for working with tabular data in Python
# It provides DataFrame objects (similar to Excel spreadsheets or SQL tables)
import pandas as pd

# Import numpy for numerical operations
# numpy provides efficient array operations and random number generation
# We use it here specifically for generating random values for reject inference
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 2: LOAD INPUT DATA
# -----------------------------------------------------------------------------

# Read the first input table from KNIME and convert it to a pandas DataFrame
# knio.input_tables is a list of all input ports connected to this Python node
# [0] accesses the first (and in this case, only) input port
# .to_pandas() converts the KNIME table format into a pandas DataFrame
# The DataFrame 'df' now contains all rows and columns from the input table
df = knio.input_tables[0].to_pandas()

# -----------------------------------------------------------------------------
# SECTION 3: HELPER FUNCTION FOR MISSING VALUE DETECTION
# -----------------------------------------------------------------------------

# Define a helper function to check if a value is null, empty, or missing
# This function handles multiple ways data can be "missing":
#   - Standard Python/pandas null values (None, NaN, pd.NA)
#   - Empty strings or strings containing only whitespace
#
# Parameters:
#   value: Any value to check for missingness
#
# Returns:
#   True if the value is considered missing, False otherwise
def is_missing(value):
    
    # First check: Use pandas' isna() function to detect standard null values
    # This catches None, NaN, pd.NA, and numpy's nan values
    # pd.isna() is the recommended way to check for null values in pandas
    if pd.isna(value):
        # If pandas recognizes this as a null value, return True (it IS missing)
        return True
    
    # Second check: Handle empty strings
    # Sometimes data comes with empty strings "" instead of true null values
    # We need to check if the value is a string first (using isinstance)
    # Then we strip whitespace and check if anything remains
    if isinstance(value, str) and value.strip() == '':
        # The value is a string that contains only whitespace (or nothing)
        # We consider this as missing, so return True
        return True
    
    # If neither check triggered, the value is NOT missing
    # Return False to indicate this is a valid, non-missing value
    return False

# -----------------------------------------------------------------------------
# SECTION 4: CREATE LOAN ID PRESENCE MASKS
# -----------------------------------------------------------------------------

# Create a boolean mask (Series of True/False values) indicating where LoanID is present
# 
# We iterate over each value in the 'LoanID' column using .apply()
# For each value, we call is_missing() to check if it's null/empty
# The lambda function returns True if the value is NOT missing (note the 'not')
# 
# Result: A pandas Series where:
#   - True means this row HAS a valid LoanID (approved application with a loan)
#   - False means this row does NOT have a LoanID (rejected application)
loan_id_present = df['LoanID'].apply(lambda x: not is_missing(x))

# Create the inverse mask for convenience
# The tilde (~) operator flips all True values to False and vice versa
# 
# Result: A pandas Series where:
#   - True means this row does NOT have a LoanID (rejected application)
#   - False means this row HAS a LoanID (approved application)
loan_id_missing = ~loan_id_present

# -----------------------------------------------------------------------------
# SECTION 5: INITIALIZE THE NEW isFPD_wRI COLUMN
# -----------------------------------------------------------------------------

# Create a new column called 'isFPD_wRI' (is First Payment Default with Reject Inference)
# 
# We initialize it with all missing values (pd.NA) for now
# We'll fill in the values in the following steps
#
# IMPORTANT: We use 'Int32' (capital I) instead of 'int32' (lowercase i)
# The capital I version is pandas' nullable integer type
# Regular Python integers cannot represent missing values (they must have a number)
# The nullable Int32 type CAN hold missing values (pd.NA), which we need here
# because some rows might not have a value after all our logic runs
#
# We create an array of pd.NA values with the same length as our DataFrame
# pd.array() creates a pandas array with the specified dtype
df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')

# -----------------------------------------------------------------------------
# SECTION 6: STEP 1 - ASSIGN VALUES FOR APPROVED APPLICATIONS
# -----------------------------------------------------------------------------

# Step 1: For approved applications (those with a LoanID), use the actual IsFPD value
#
# When a loan was actually issued (LoanID is present), we have the real outcome
# IsFPD = 1 means the customer defaulted on their first payment (bad)
# IsFPD = 0 means the customer made their first payment successfully (good)
#
# .loc[] is pandas' label-based indexer for selecting rows and columns
# The first part (loan_id_present) selects which ROWS to modify
# The second part ('isFPD_wRI') specifies which COLUMN to modify
# We copy the actual outcome from the 'IsFPD' column
df.loc[loan_id_present, 'isFPD_wRI'] = df.loc[loan_id_present, 'IsFPD']

# -----------------------------------------------------------------------------
# SECTION 7: STEP 2 - ASSIGN VALUES FROM FPD FOR REJECTED APPLICATIONS
# -----------------------------------------------------------------------------

# Step 2: For rejected applications (no LoanID), try to use the FPD column
#
# The FPD column might contain pre-assigned values for some rejected applications
# This could come from a previous reject inference step or external data
# If FPD has a value, we use it; if not, we'll handle it in Step 3
#
# We only modify rows where LoanID is missing (rejected applications)
# We copy whatever value exists in the 'FPD' column (could be a number or null)
df.loc[loan_id_missing, 'isFPD_wRI'] = df.loc[loan_id_missing, 'FPD']

# -----------------------------------------------------------------------------
# SECTION 8: STEP 3 - PROBABILISTIC INFERENCE FOR REMAINING MISSING VALUES
# -----------------------------------------------------------------------------

# Step 3: Handle remaining missing values using probabilistic assignment
#
# After Steps 1 and 2, some rows might still have missing isFPD_wRI values
# These are rejected applications where FPD was also null
# For these, we use the predicted default probability to randomly assign outcomes

# Create a mask for rows that STILL have missing isFPD_wRI values
# AND are rejected applications (LoanID is missing)
# We combine two conditions:
#   1. loan_id_missing: The application was rejected (no loan was issued)
#   2. df['isFPD_wRI'].isna(): The isFPD_wRI value is still null after Steps 1 & 2
# The & operator combines these with AND logic
still_missing_mask = loan_id_missing & df['isFPD_wRI'].isna()

# Check which rows have a valid expected_DefaultRate2 value
# expected_DefaultRate2 is the model's predicted probability of default
# We can only do probabilistic inference if we have this probability
# .notna() returns True for non-null values, False for null values
has_expected_rate = df['expected_DefaultRate2'].notna()

# Combine the masks to find rows that need probabilistic inference
# These rows must satisfy ALL of these conditions:
#   1. Rejected application (no LoanID)
#   2. isFPD_wRI is still missing (FPD didn't provide a value)
#   3. We have a predicted default rate to use for inference
rows_to_process = still_missing_mask & has_expected_rate

# Only proceed if there are actually rows to process
# .any() returns True if at least one value in the Series is True
# This avoids unnecessary computation if all values are already filled
if rows_to_process.any():
    
    # Count how many rows need probabilistic assignment
    # .sum() counts True values (since True=1 and False=0 in Python)
    num_rows = rows_to_process.sum()
    
    # Generate random numbers between 0 and 1 for each row that needs assignment
    # np.random.random(n) generates n random floats uniformly distributed in [0, 1)
    # Each row gets its own random number to determine its outcome
    random_values = np.random.random(num_rows)
    
    # Get the expected default rates for the rows we're processing
    # .loc[rows_to_process, 'expected_DefaultRate2'] selects only the relevant rows
    # .values converts the pandas Series to a numpy array for faster comparison
    # These probabilities typically range from 0.0 (very unlikely to default) 
    # to 1.0 (very likely to default)
    expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
    
    # Perform the probabilistic assignment using Monte Carlo simulation
    # 
    # For each row, we compare its random value to its expected default rate:
    #   - If random <= expected_rate: assign 1 (default)
    #   - If random > expected_rate: assign 0 (no default)
    #
    # Example: If expected_DefaultRate2 = 0.3 (30% chance of default)
    #   - Random values 0.0 to 0.3 will result in default (1)
    #   - Random values 0.3 to 1.0 will result in no default (0)
    #   - This gives approximately 30% of such cases a default outcome
    #
    # The comparison (random_values <= expected_rates) returns boolean array
    # .astype(int) converts True to 1 and False to 0
    assigned_values = (random_values <= expected_rates).astype(int)
    
    # Assign the probabilistically determined values to the isFPD_wRI column
    # Only the rows identified by rows_to_process are modified
    df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values

# -----------------------------------------------------------------------------
# SECTION 9: ENSURE CORRECT DATA TYPE FOR isFPD_wRI
# -----------------------------------------------------------------------------

# Ensure the isFPD_wRI column is stored as nullable Int32
# 
# This step is a safety measure because the assignments above might have
# changed the column's dtype to something else (like float64)
# 
# Int32 (capital I) is pandas' nullable integer type, which:
#   - Stores integers (0, 1, 2, etc.)
#   - Can also store missing values (pd.NA)
#   - Is compatible with KNIME's integer column type
df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')

# =============================================================================
# SECTION 10: FRODI26_wRI AND GRODI26_wRI COLUMN GENERATION
# =============================================================================
#
# This section creates imputed versions of the FRODI26 and GRODI26 columns
# 
# FRODI26 and GRODI26 are performance metrics (likely related to income or payments)
# For approved applications, we have actual values
# For rejected applications, these values are missing (we never issued a loan)
#
# We impute (fill in) missing values using the AVERAGE of similar applications
# "Similar" is defined by their default status (IsFPD = 0 or 1)
# This is called "hot deck imputation" or "mean imputation by group"
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 11: CALCULATE GROUP AVERAGES FOR IMPUTATION
# -----------------------------------------------------------------------------

# Calculate the average FRODI26 value for applications that DID default (IsFPD = 1)
# 
# df['IsFPD'] == 1 creates a boolean mask selecting only defaulted applications
# df.loc[mask, 'FRODI26'] selects the FRODI26 values for those applications
# .mean() calculates the arithmetic mean of those values
# 
# This average will be used to impute FRODI26 for rejected applications
# that we inferred as likely defaulters (isFPD_wRI = 1)
avg_frodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'FRODI26'].mean()

# Calculate the average FRODI26 value for applications that did NOT default (IsFPD = 0)
# This will be used for rejected applications inferred as non-defaulters (isFPD_wRI = 0)
avg_frodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'FRODI26'].mean()

# Calculate the average GRODI26 value for defaulted applications (IsFPD = 1)
avg_grodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'GRODI26'].mean()

# Calculate the average GRODI26 value for non-defaulted applications (IsFPD = 0)
avg_grodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'GRODI26'].mean()

# -----------------------------------------------------------------------------
# SECTION 12: CREATE FRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

# Create a boolean mask identifying rows where FRODI26 is missing
# We use our is_missing() helper function to check each value
# .apply() runs the function on every value in the column
frodi26_missing = df['FRODI26'].apply(is_missing)

# Create the inverse mask for rows where FRODI26 is present (has a valid value)
# The tilde (~) operator flips True to False and vice versa
frodi26_present = ~frodi26_missing

# Initialize the new FRODI26_wRI column with all missing values
# We'll fill it in step by step
df['FRODI26_wRI'] = pd.NA

# For rows where FRODI26 already has a value, copy that value to FRODI26_wRI
# These are typically approved applications where we have actual performance data
df.loc[frodi26_present, 'FRODI26_wRI'] = df.loc[frodi26_present, 'FRODI26']

# For rows where FRODI26 is missing AND isFPD_wRI is 1 (defaulter):
# Use the average FRODI26 of actual defaulters as the imputed value
# This assumes rejected applications that are inferred as defaulters
# would have similar FRODI26 values to actual defaulters
df.loc[frodi26_missing & (df['isFPD_wRI'] == 1), 'FRODI26_wRI'] = avg_frodi26_fpd1

# For rows where FRODI26 is missing AND isFPD_wRI is 0 (non-defaulter):
# Use the average FRODI26 of actual non-defaulters as the imputed value
df.loc[frodi26_missing & (df['isFPD_wRI'] == 0), 'FRODI26_wRI'] = avg_frodi26_fpd0

# Convert FRODI26_wRI to nullable Float64 type
# 
# Float64 (capital F) is pandas' nullable floating-point type
# We use this because FRODI26 values are likely decimals/fractions
# The capital F version can store missing values (pd.NA), unlike regular float64
df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')

# -----------------------------------------------------------------------------
# SECTION 13: CREATE GRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

# Create a boolean mask identifying rows where GRODI26 is missing
# Same logic as FRODI26 above
grodi26_missing = df['GRODI26'].apply(is_missing)

# Create the inverse mask for rows where GRODI26 is present
grodi26_present = ~grodi26_missing

# Initialize the new GRODI26_wRI column with all missing values
df['GRODI26_wRI'] = pd.NA

# Copy existing GRODI26 values to GRODI26_wRI where available
# These are approved applications with actual performance data
df.loc[grodi26_present, 'GRODI26_wRI'] = df.loc[grodi26_present, 'GRODI26']

# Impute missing values for inferred defaulters (isFPD_wRI = 1)
# Use the average GRODI26 of actual defaulters
df.loc[grodi26_missing & (df['isFPD_wRI'] == 1), 'GRODI26_wRI'] = avg_grodi26_fpd1

# Impute missing values for inferred non-defaulters (isFPD_wRI = 0)
# Use the average GRODI26 of actual non-defaulters
df.loc[grodi26_missing & (df['isFPD_wRI'] == 0), 'GRODI26_wRI'] = avg_grodi26_fpd0

# Convert GRODI26_wRI to nullable Float64 type for KNIME compatibility
df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')

# =============================================================================
# SECTION 14: REORDER COLUMNS FOR BETTER ORGANIZATION
# =============================================================================
#
# This section reorganizes the DataFrame columns to place the new columns
# immediately after their source columns for better readability
#
# Current state: New columns are at the end of the DataFrame
# Desired state: Each _wRI column appears right after its source column
#   - IsFPD followed by isFPD_wRI
#   - FRODI26 followed by FRODI26_wRI
#   - GRODI26 followed by GRODI26_wRI
# =============================================================================

# Get the current list of column names as a Python list
# .columns returns a pandas Index object; .tolist() converts it to a regular list
# We need a list because we'll be modifying the order
cols = df.columns.tolist()

# -----------------------------------------------------------------------------
# SECTION 15: REMOVE NEW COLUMNS FROM CURRENT POSITIONS
# -----------------------------------------------------------------------------

# Remove the new columns from wherever they currently are in the list
# They were added at the end when we created them
# We need to remove them first so we can insert them in the correct positions

# Remove 'isFPD_wRI' from the column list
# .remove() finds and removes the first occurrence of the specified value
cols.remove('isFPD_wRI')

# Remove 'FRODI26_wRI' from the column list
cols.remove('FRODI26_wRI')

# Remove 'GRODI26_wRI' from the column list
cols.remove('GRODI26_wRI')

# -----------------------------------------------------------------------------
# SECTION 16: INSERT isFPD_wRI AFTER IsFPD
# -----------------------------------------------------------------------------

# Find the current position (index) of the 'IsFPD' column in the list
# .index() returns the position of the first occurrence of the specified value
# Python list indices are 0-based, so the first column is at index 0
isfpd_index = cols.index('IsFPD')

# Insert 'isFPD_wRI' immediately after 'IsFPD'
# .insert(position, value) inserts the value at the specified position
# We use isfpd_index + 1 to place it right after IsFPD
# All columns after this position are shifted right by one
cols.insert(isfpd_index + 1, 'isFPD_wRI')

# -----------------------------------------------------------------------------
# SECTION 17: INSERT FRODI26_wRI AFTER FRODI26
# -----------------------------------------------------------------------------

# Find the current position of the 'FRODI26' column
# Note: The position might have shifted after we inserted isFPD_wRI above
# But since FRODI26 is likely AFTER IsFPD, the shift doesn't affect its relative position
frodi26_index = cols.index('FRODI26')

# Insert 'FRODI26_wRI' immediately after 'FRODI26'
cols.insert(frodi26_index + 1, 'FRODI26_wRI')

# -----------------------------------------------------------------------------
# SECTION 18: INSERT GRODI26_wRI AFTER GRODI26
# -----------------------------------------------------------------------------

# Find the current position of the 'GRODI26' column
# The position accounts for the previous insertions of isFPD_wRI and FRODI26_wRI
grodi26_index = cols.index('GRODI26')

# Insert 'GRODI26_wRI' immediately after 'GRODI26'
cols.insert(grodi26_index + 1, 'GRODI26_wRI')

# -----------------------------------------------------------------------------
# SECTION 19: APPLY THE NEW COLUMN ORDER
# -----------------------------------------------------------------------------

# Reorder the DataFrame columns according to our new list
# df[cols] creates a new DataFrame with columns in the order specified by 'cols'
# We assign it back to 'df' to apply the reordering
df = df[cols]

# =============================================================================
# SECTION 20: OUTPUT THE RESULTS TO KNIME
# =============================================================================

# Write the processed DataFrame to the first (and only) output port
# 
# knio.output_tables is a list representing the output ports of this Python node
# [0] accesses the first output port
# 
# knio.Table.from_pandas(df) converts our pandas DataFrame back to KNIME's
# internal table format, which is required for passing data to downstream nodes
# 
# The output table contains:
#   - All original columns from the input
#   - Three new columns: isFPD_wRI, FRODI26_wRI, GRODI26_wRI
#   - Columns reordered so _wRI columns appear next to their source columns
knio.output_tables[0] = knio.Table.from_pandas(df)

# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# SUMMARY OF WHAT THIS SCRIPT DOES:
#
# 1. READS input data containing loan applications (approved and rejected)
#
# 2. CREATES isFPD_wRI (First Payment Default with Reject Inference):
#    - For APPROVED applications (have LoanID): Uses actual IsFPD value
#    - For REJECTED applications (no LoanID): 
#      a. First tries to use the FPD column if available
#      b. Otherwise, uses Monte Carlo simulation based on expected_DefaultRate2
#         to probabilistically assign 0 or 1
#
# 3. CREATES FRODI26_wRI and GRODI26_wRI:
#    - For applications WITH existing values: Copies the original value
#    - For applications WITHOUT values: Imputes using group averages
#      based on the inferred default status (isFPD_wRI)
#
# 4. REORDERS columns so new _wRI columns appear next to their source columns
#
# 5. OUTPUTS the enriched DataFrame to KNIME for downstream processing
#
# This reject inference approach helps address sample bias in credit scoring
# models by including rejected applications in the analysis with statistically
# inferred outcomes based on their predicted risk profiles.
# =============================================================================

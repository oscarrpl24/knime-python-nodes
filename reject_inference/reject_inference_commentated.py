# =============================================================================
# KNIME Python Script - Reject Inference Column Generation
# =============================================================================
# 
# PURPOSE:
# This script performs "Reject Inference" - a credit risk modeling technique used
# to estimate the likely outcomes (default or non-default) for loan applications
# that were REJECTED and therefore have no actual performance data.
#
# In credit risk modeling, we have two populations:
#   1. APPROVED applications - these have actual outcomes (did they default or not?)
#   2. REJECTED applications - these have no outcomes (we never gave them a loan)
#
# Reject Inference attempts to infer what would have happened if we HAD approved
# the rejected applications, using probability scores from a model.
#
# COMPATIBILITY:
# - KNIME Version: 5.9
# - Python Version: 3.9.23
# - Platform: Windows
#
# REQUIRED INPUT COLUMNS:
# - LoanID: Identifier for the loan (missing/empty means rejected application)
# - IsFPD: "Is First Payment Default" - actual default flag for approved loans
# - FPD: Alternative default flag column
# - expected_DefaultRate2: Model-predicted probability of default
# - FRODI26: Some fraud/risk indicator metric
# - GRODI26: Another fraud/risk indicator metric
#
# OUTPUT COLUMNS CREATED:
# - isFPD_wRI: Default flag "with Reject Inference" applied
# - FRODI26_wRI: FRODI26 "with Reject Inference" applied  
# - GRODI26_wRI: GRODI26 "with Reject Inference" applied
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------

# Import the KNIME scripting interface - this is the bridge between Python and KNIME.
# The 'knio' module provides access to input/output tables and flow variables.
# This is a KNIME-specific module that only works inside KNIME Python Script nodes.
import knime.scripting.io as knio

# Import pandas - the primary data manipulation library in Python.
# Pandas provides DataFrame objects (similar to Excel spreadsheets or SQL tables)
# that make it easy to work with tabular data.
import pandas as pd

# Import numpy - the fundamental package for numerical computing in Python.
# NumPy provides fast array operations and mathematical functions.
# Here we use it specifically for generating random numbers.
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 2: READ INPUT DATA FROM KNIME
# -----------------------------------------------------------------------------

# Read the first input table (index 0) from KNIME and convert it to a pandas DataFrame.
# In KNIME, input_tables is a list of all tables connected to the node's input ports.
# The [0] gets the first (and in this case, only) input table.
# The .to_pandas() method converts it from KNIME's internal format to a pandas DataFrame.
# After this line, 'df' contains all the data we'll be working with.
df = knio.input_tables[0].to_pandas()

# -----------------------------------------------------------------------------
# SECTION 3: DEFINE HELPER FUNCTION FOR MISSING VALUE DETECTION
# -----------------------------------------------------------------------------

# Define a helper function to check if a value should be considered "missing".
# This is important because missing values can come in many forms:
#   - pandas NA (pd.NA)
#   - numpy NaN (np.nan)
#   - Python None
#   - Empty strings ("")
#   - Strings with only whitespace ("   ")
#
# This function returns True if the value is any kind of missing, False otherwise.
def is_missing(value):
    
    # First, check if the value is a pandas/numpy null value.
    # pd.isna() returns True for: None, pd.NA, np.nan, and pd.NaT (datetime null).
    # This handles the standard "null" cases that pandas recognizes.
    if pd.isna(value):
        return True
    
    # Second, check if the value is a string that is effectively empty.
    # isinstance(value, str) checks if the value is a text string.
    # value.strip() removes all leading and trailing whitespace from the string.
    # If after stripping whitespace the string is empty (''), it's considered missing.
    # This catches cases like "", "   ", "\t\n", etc.
    if isinstance(value, str) and value.strip() == '':
        return True
    
    # If neither condition was met, the value is NOT missing.
    # Return False to indicate this value contains actual data.
    return False

# -----------------------------------------------------------------------------
# SECTION 4: CREATE MASKS FOR LOANID PRESENCE
# -----------------------------------------------------------------------------

# A "mask" in pandas is a boolean Series (column) where each row is True or False.
# Masks are used to filter or select specific rows of a DataFrame.

# Create a mask indicating which rows have a valid (non-missing) LoanID.
# The .apply() method runs our is_missing() function on every value in the 'LoanID' column.
# The 'lambda x' creates an anonymous function that takes each value (x) and returns
# 'not is_missing(x)' - so True if the LoanID is present, False if it's missing.
# Result: loan_id_present[i] is True if row i has a valid LoanID.
loan_id_present = df['LoanID'].apply(lambda x: not is_missing(x))

# Create the inverse mask - True where LoanID is missing.
# The ~ operator in pandas/numpy is the logical NOT operator.
# This flips all True values to False and all False values to True.
# Result: loan_id_missing[i] is True if row i has a missing LoanID (rejected application).
loan_id_missing = ~loan_id_present

# -----------------------------------------------------------------------------
# SECTION 5: INITIALIZE THE NEW DEFAULT FLAG COLUMN (isFPD_wRI)
# -----------------------------------------------------------------------------

# Create a new column called 'isFPD_wRI' (IsFPD with Reject Inference).
# This column will eventually contain:
#   - For approved loans (LoanID present): the actual default status
#   - For rejected loans (LoanID missing): an inferred default status

# Initialize the column with all missing values (pd.NA).
# pd.array() creates a pandas array with specific values.
# [pd.NA] * len(df) creates a list of pd.NA values, one for each row in the DataFrame.
# dtype='Int32' specifies this is a nullable integer column.
# IMPORTANT: We use 'Int32' (capital I) not 'int32' (lowercase i) because:
#   - 'Int32' is a nullable integer type that can hold missing values (pd.NA)
#   - 'int32' is a regular integer that CANNOT hold missing values (would fail)
# KNIME requires nullable types when columns might contain missing values.
df['isFPD_wRI'] = pd.array([pd.NA] * len(df), dtype='Int32')

# -----------------------------------------------------------------------------
# SECTION 6: STEP 1 - APPROVED LOANS: COPY ACTUAL DEFAULT STATUS
# -----------------------------------------------------------------------------

# For approved loans (where LoanID is present), we use the actual default status.
# These loans have real outcomes - we know if they actually defaulted or not.

# df.loc[mask, column] is pandas' way of selecting specific rows and columns.
# - loan_id_present is our True/False mask saying which rows to select
# - 'isFPD_wRI' is the column we want to modify (left side of =)
# - df.loc[loan_id_present, 'IsFPD'] gets the IsFPD values for those same rows

# This line says: "For all rows where LoanID is present, set isFPD_wRI equal to IsFPD"
# Essentially copying the actual default flag for approved loans.
df.loc[loan_id_present, 'isFPD_wRI'] = df.loc[loan_id_present, 'IsFPD']

# -----------------------------------------------------------------------------
# SECTION 7: STEP 2 - REJECTED LOANS: TRY USING FPD COLUMN FIRST
# -----------------------------------------------------------------------------

# For rejected loans (where LoanID is missing), first try to use the FPD column.
# The FPD column might have pre-populated values for some rejected applications.
# This could be from previous reject inference runs or external data sources.

# This line says: "For all rows where LoanID is missing, set isFPD_wRI equal to FPD"
# If FPD is also missing for a row, isFPD_wRI will remain missing (we handle that next).
df.loc[loan_id_missing, 'isFPD_wRI'] = df.loc[loan_id_missing, 'FPD']

# -----------------------------------------------------------------------------
# SECTION 8: STEP 3 - REJECTED LOANS: PROBABILISTIC INFERENCE FOR REMAINING
# -----------------------------------------------------------------------------

# Some rejected loans might still have missing isFPD_wRI values after Step 2.
# This happens when both LoanID is missing AND FPD is missing.
# For these cases, we'll use probabilistic reject inference based on model scores.

# Create a mask for rows that STILL need values after Steps 1 and 2.
# loan_id_missing: True for rejected applications (LoanID is missing)
# df['isFPD_wRI'].isna(): True for rows where isFPD_wRI is still missing/null
# The & operator combines both conditions - both must be True.
# Result: still_missing_mask is True only for rejected loans that still need inference.
still_missing_mask = loan_id_missing & df['isFPD_wRI'].isna()

# Create a mask for rows that have an expected default rate (probability score).
# .notna() returns True if the value is NOT null/missing.
# We can only do probabilistic inference if we have a probability to use.
has_expected_rate = df['expected_DefaultRate2'].notna()

# Combine both masks: we only process rows that:
#   1. Still need a value (still_missing_mask = True)
#   2. Have an expected default rate to base our inference on (has_expected_rate = True)
# The & operator requires BOTH conditions to be True.
rows_to_process = still_missing_mask & has_expected_rate

# Check if there are ANY rows that need probabilistic inference.
# .any() returns True if at least one value in the mask is True.
# If no rows need processing, we skip this entire block to save computation.
if rows_to_process.any():
    
    # Count how many rows need probabilistic inference.
    # .sum() on a boolean mask counts the True values (True=1, False=0).
    # We need this number to generate the right amount of random values.
    num_rows = rows_to_process.sum()
    
    # Generate random numbers for each row that needs inference.
    # np.random.random(n) generates n random floats uniformly distributed between 0 and 1.
    # Each value is independent and equally likely to be anywhere in [0, 1).
    # Example: if num_rows=5, might get [0.23, 0.87, 0.12, 0.56, 0.91]
    random_values = np.random.random(num_rows)
    
    # Get the model-predicted default probabilities for the rows we're processing.
    # df.loc[mask, column] selects specific rows from a column.
    # .values converts the pandas Series to a numpy array (faster for comparison).
    # These are the probabilities that each rejected applicant would have defaulted.
    # Example: [0.15, 0.45, 0.08, 0.72, 0.33] means 15%, 45%, 8%, 72%, 33% default prob.
    expected_rates = df.loc[rows_to_process, 'expected_DefaultRate2'].values
    
    # CORE REJECT INFERENCE LOGIC:
    # We assign a simulated default (1) or non-default (0) based on probability.
    # 
    # The logic: If random_value <= expected_default_rate, assign 1 (default), else 0.
    # 
    # Why this works:
    # - If expected_rate = 0.30 (30% default probability)
    # - random_value is uniformly distributed from 0 to 1
    # - There's a 30% chance random_value will be <= 0.30
    # - So 30% of such cases will be assigned "default" (1)
    # - This matches the expected default rate!
    #
    # (random_values <= expected_rates) produces a boolean array: [True, False, True, ...]
    # .astype(int) converts booleans to integers: True->1, False->0
    # Result: array of 0s and 1s representing inferred default status.
    assigned_values = (random_values <= expected_rates).astype(int)
    
    # Assign the inferred default values back to the DataFrame.
    # Only the rows matching rows_to_process mask will be updated.
    # Other rows remain unchanged (either have actual values or stay missing).
    df.loc[rows_to_process, 'isFPD_wRI'] = assigned_values

# -----------------------------------------------------------------------------
# SECTION 9: ENSURE PROPER DATA TYPE FOR isFPD_wRI
# -----------------------------------------------------------------------------

# Ensure the column has the correct nullable integer type for KNIME.
# Even though we initialized it as Int32, operations might have changed it.
# This explicit cast ensures KNIME will receive the expected data type.
# 'Int32' (capital I) is pandas' nullable integer type that:
#   - Can store integer values: 0, 1, 2, etc.
#   - Can store missing values: pd.NA
#   - Maps correctly to KNIME's integer column type
df['isFPD_wRI'] = df['isFPD_wRI'].astype('Int32')

# =============================================================================
# SECTION 10: FRODI26_wRI AND GRODI26_wRI COLUMN GENERATION
# =============================================================================
# 
# Next, we create "with Reject Inference" versions of two fraud/risk metrics.
# The logic:
#   - If the original value exists, keep it
#   - If the original value is missing, impute it with the average value
#     from approved loans with the same inferred default status
#
# Why? Rejected applications don't have these metrics (they were never approved).
# We estimate them by using averages from similar approved applications.
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 11: CALCULATE AVERAGE FRODI26 AND GRODI26 BY DEFAULT STATUS
# -----------------------------------------------------------------------------

# Calculate the average FRODI26 value for loans that DID default (IsFPD = 1).
# We use the ORIGINAL IsFPD column (not isFPD_wRI) because we want averages
# based on ACTUAL outcomes, not inferred ones.
# df.loc[condition, column] selects rows matching condition from that column.
# .mean() calculates the arithmetic average of all selected values (ignores NaN).
avg_frodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'FRODI26'].mean()

# Calculate the average FRODI26 value for loans that did NOT default (IsFPD = 0).
# This gives us the typical FRODI26 for non-defaulters.
avg_frodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'FRODI26'].mean()

# Calculate the average GRODI26 value for loans that DID default (IsFPD = 1).
avg_grodi26_fpd1 = df.loc[df['IsFPD'] == 1, 'GRODI26'].mean()

# Calculate the average GRODI26 value for loans that did NOT default (IsFPD = 0).
avg_grodi26_fpd0 = df.loc[df['IsFPD'] == 0, 'GRODI26'].mean()

# -----------------------------------------------------------------------------
# SECTION 12: CREATE FRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

# Create a mask indicating which rows have a missing FRODI26 value.
# We reuse our is_missing() function to catch all forms of missing values.
# .apply() runs the function on every value in the FRODI26 column.
frodi26_missing = df['FRODI26'].apply(is_missing)

# Create the inverse mask - True where FRODI26 is present (has a valid value).
frodi26_present = ~frodi26_missing

# Initialize the new column with all missing values.
# We'll fill in values in the next steps.
df['FRODI26_wRI'] = pd.NA

# For rows where FRODI26 already has a value, copy it directly to FRODI26_wRI.
# These are typically approved loans that have actual data.
# We preserve existing data rather than overwriting it with averages.
df.loc[frodi26_present, 'FRODI26_wRI'] = df.loc[frodi26_present, 'FRODI26']

# For rows where FRODI26 is missing AND the inferred default status is 1 (defaulter),
# fill in the average FRODI26 from actual defaulters.
# This uses both conditions combined with & (AND operator).
# frodi26_missing: True where FRODI26 needs imputation
# df['isFPD_wRI'] == 1: True where inferred default status is "did default"
df.loc[frodi26_missing & (df['isFPD_wRI'] == 1), 'FRODI26_wRI'] = avg_frodi26_fpd1

# For rows where FRODI26 is missing AND the inferred default status is 0 (non-defaulter),
# fill in the average FRODI26 from actual non-defaulters.
df.loc[frodi26_missing & (df['isFPD_wRI'] == 0), 'FRODI26_wRI'] = avg_frodi26_fpd0

# Convert the column to nullable float type for KNIME compatibility.
# 'Float64' (capital F) is pandas' nullable float type that:
#   - Can store decimal numbers: 0.5, 1.23, etc.
#   - Can store missing values: pd.NA
#   - Maps correctly to KNIME's double column type
df['FRODI26_wRI'] = df['FRODI26_wRI'].astype('Float64')

# -----------------------------------------------------------------------------
# SECTION 13: CREATE GRODI26_wRI COLUMN
# -----------------------------------------------------------------------------

# Create a mask indicating which rows have a missing GRODI26 value.
# Same logic as FRODI26 - we check for all forms of missing.
grodi26_missing = df['GRODI26'].apply(is_missing)

# Create the inverse mask - True where GRODI26 has a valid value.
grodi26_present = ~grodi26_missing

# Initialize the new column with all missing values.
df['GRODI26_wRI'] = pd.NA

# For rows where GRODI26 already has a value, copy it directly to GRODI26_wRI.
# Preserves actual data from approved loans.
df.loc[grodi26_present, 'GRODI26_wRI'] = df.loc[grodi26_present, 'GRODI26']

# For rows where GRODI26 is missing AND inferred as defaulter (isFPD_wRI = 1),
# fill with average GRODI26 from actual defaulters.
df.loc[grodi26_missing & (df['isFPD_wRI'] == 1), 'GRODI26_wRI'] = avg_grodi26_fpd1

# For rows where GRODI26 is missing AND inferred as non-defaulter (isFPD_wRI = 0),
# fill with average GRODI26 from actual non-defaulters.
df.loc[grodi26_missing & (df['isFPD_wRI'] == 0), 'GRODI26_wRI'] = avg_grodi26_fpd0

# Convert the column to nullable float type for KNIME compatibility.
df['GRODI26_wRI'] = df['GRODI26_wRI'].astype('Float64')

# =============================================================================
# SECTION 14: REORDER COLUMNS FOR BETTER ORGANIZATION
# =============================================================================
#
# We want to place each new "_wRI" column right after its source column.
# This makes the output easier to read and understand:
#   - isFPD_wRI appears right after IsFPD
#   - FRODI26_wRI appears right after FRODI26
#   - GRODI26_wRI appears right after GRODI26
# =============================================================================

# Get the current list of all column names in the DataFrame.
# .columns returns an Index object with column names.
# .tolist() converts it to a regular Python list that we can manipulate.
cols = df.columns.tolist()

# Remove the new columns from their current positions.
# When we created them, they were added at the end of the DataFrame.
# We'll re-insert them at the desired positions.
# .remove() finds and removes the first occurrence of the item from the list.
cols.remove('isFPD_wRI')      # Remove from current position (at the end)
cols.remove('FRODI26_wRI')    # Remove from current position
cols.remove('GRODI26_wRI')    # Remove from current position

# Find the position of the 'IsFPD' column in our list.
# .index() returns the position (0-based) of the first occurrence.
# Example: if IsFPD is the 5th column, isfpd_index = 4 (0-based indexing)
isfpd_index = cols.index('IsFPD')

# Insert 'isFPD_wRI' right after 'IsFPD'.
# .insert(position, item) adds the item at the specified position.
# isfpd_index + 1 means "one position after IsFPD".
# All subsequent items shift right to make room.
cols.insert(isfpd_index + 1, 'isFPD_wRI')

# Find the position of 'FRODI26' column.
# Note: positions may have shifted after the previous insert, but we're finding
# the index fresh from the current state of the list.
frodi26_index = cols.index('FRODI26')

# Insert 'FRODI26_wRI' right after 'FRODI26'.
cols.insert(frodi26_index + 1, 'FRODI26_wRI')

# Find the position of 'GRODI26' column.
# Note: The list has changed since we inserted FRODI26_wRI, so positions
# of columns after FRODI26 have shifted. This is accounted for automatically
# by finding the index fresh.
grodi26_index = cols.index('GRODI26')

# Insert 'GRODI26_wRI' right after 'GRODI26'.
cols.insert(grodi26_index + 1, 'GRODI26_wRI')

# Reorder the DataFrame columns according to our new column order.
# df[cols] creates a new DataFrame with columns in the order specified by 'cols'.
# We assign it back to df to replace the original column order.
df = df[cols]

# =============================================================================
# SECTION 15: OUTPUT RESULTS TO KNIME
# =============================================================================

# Send the final DataFrame back to KNIME through output port 0.
# knio.output_tables[0] is the first (and only) output port of this node.
# knio.Table.from_pandas(df) converts our pandas DataFrame back to KNIME's
# internal table format that can be passed to downstream nodes.
# 
# The output table contains:
#   - All original columns from the input
#   - Three new columns: isFPD_wRI, FRODI26_wRI, GRODI26_wRI
#   - Columns reordered so new columns appear next to their source columns
knio.output_tables[0] = knio.Table.from_pandas(df)

# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# SUMMARY OF WHAT THIS SCRIPT DOES:
#
# 1. Reads a table containing loan application data with both approved and
#    rejected applications (rejected identified by missing LoanID).
#
# 2. Creates isFPD_wRI (Is First Payment Default with Reject Inference):
#    - For approved loans: uses actual default status (IsFPD)
#    - For rejected loans: first tries FPD column, then uses probabilistic
#      inference based on expected_DefaultRate2 (assigns 1 or 0 randomly
#      with probability matching the expected default rate)
#
# 3. Creates FRODI26_wRI and GRODI26_wRI:
#    - Preserves original values where they exist
#    - For missing values: imputes with the average from approved loans
#      that have the same default status as the inferred status
#
# 4. Reorders columns so new columns appear next to their source columns.
#
# 5. Outputs the enhanced table to KNIME.
#
# KEY CONCEPTS USED:
# - Boolean masks for row selection
# - Probabilistic assignment using random numbers
# - Mean imputation stratified by outcome group
# - Pandas nullable types (Int32, Float64) for KNIME compatibility
# =============================================================================

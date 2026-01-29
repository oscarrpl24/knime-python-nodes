# =============================================================================
# IS BAD FLAG NODE FOR KNIME - FULLY COMMENTATED VERSION
# =============================================================================
#
# OVERVIEW:
# ---------
# This Python script is designed to run inside a KNIME Python Script node.
# Its purpose is to create a binary "isBad" target variable for credit risk 
# modeling. In credit risk, we need to identify which customers are "bad" 
# (defaulted, delinquent, or risky) versus "good" (paid on time, low risk).
#
# WHAT IS GRODI26_wRI?
# --------------------
# GRODI26_wRI is a credit performance metric. The "26" typically refers to
# 26 months of observation. "GRODI" stands for "Gross Roll-Down Indicator"
# or similar performance metric. Values below 1 indicate poor performance
# (the customer is considered "bad"), while values >= 1 indicate acceptable
# performance (the customer is considered "good").
#
# BUSINESS LOGIC:
# ---------------
#   - If GRODI26_wRI < 1: Customer is "bad" → isBad = 1
#   - If GRODI26_wRI >= 1: Customer is "good" → isBad = 0
#
# INPUT:
# ------
# A single table containing at least the GRODI26_wRI column
#
# OUTPUT:
# -------
# The same table with a new "isBad" column added as the FIRST column
# (This makes it easy to find the target variable in subsequent nodes)
#
# =============================================================================


# =============================================================================
# SECTION 1: IMPORT STATEMENTS
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: import knime.scripting.io as knio
# -----------------------------------------------------------------------------
# This imports the KNIME Python scripting I/O module and gives it the 
# shorter alias "knio" for convenience.
#
# WHAT IS knio?
# - It is the bridge between KNIME and Python
# - It provides access to input tables (data flowing into this node)
# - It provides access to output tables (data flowing out of this node)
# - It also provides access to flow variables (configuration parameters)
#
# WHY "as knio"?
# - This is a Python aliasing convention
# - Instead of typing "knime.scripting.io" every time, we just type "knio"
# - It makes the code shorter and easier to read
# -----------------------------------------------------------------------------
import knime.scripting.io as knio

# -----------------------------------------------------------------------------
# LINE: import pandas as pd
# -----------------------------------------------------------------------------
# This imports the pandas library and gives it the alias "pd".
#
# WHAT IS pandas?
# - Pandas is Python's most popular data manipulation library
# - It provides DataFrame objects (similar to Excel tables or SQL tables)
# - It allows for filtering, transforming, aggregating, and analyzing data
#
# WHY "as pd"?
# - This is the universally accepted convention for importing pandas
# - Almost every Python data script uses "pd" as the alias for pandas
# - It makes code more readable for other data scientists
#
# WHAT WILL WE USE IT FOR?
# - Converting KNIME table data to a pandas DataFrame
# - Adding new columns to the data
# - Reordering columns
# - Performing logical comparisons on data
# -----------------------------------------------------------------------------
import pandas as pd


# =============================================================================
# SECTION 2: READ INPUT DATA FROM KNIME
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: df = knio.input_tables[0].to_pandas()
# -----------------------------------------------------------------------------
# This line reads the first input table from KNIME and converts it to a 
# pandas DataFrame.
#
# BREAKING IT DOWN:
#
# knio.input_tables
#   - This is a list (array) of all input tables connected to this node
#   - In KNIME, a Python Script node can have multiple input ports
#   - Each input port provides a separate table
#
# knio.input_tables[0]
#   - The [0] means we are accessing the FIRST input table
#   - Python uses zero-based indexing (first item is 0, not 1)
#   - If there were a second input table, it would be input_tables[1]
#   - This returns a KNIME Table object, not a pandas DataFrame yet
#
# .to_pandas()
#   - This method converts the KNIME Table to a pandas DataFrame
#   - A DataFrame is a 2D table structure (rows and columns)
#   - After this conversion, we can use all pandas functions on the data
#
# df = ...
#   - We store the result in a variable named "df"
#   - "df" is a common abbreviation for "DataFrame"
#   - This variable now contains all the data from the input table
#
# EXAMPLE:
# If the input table has 1000 rows and 50 columns, df will be a DataFrame
# with 1000 rows and 50 columns, containing all the same data.
# -----------------------------------------------------------------------------
df = knio.input_tables[0].to_pandas()


# =============================================================================
# SECTION 3: VALIDATE THAT REQUIRED COLUMN EXISTS
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: if "GRODI26_wRI" not in df.columns:
# -----------------------------------------------------------------------------
# This is a validation check to ensure the required column exists before
# we try to use it.
#
# BREAKING IT DOWN:
#
# df.columns
#   - This returns an Index object containing all column names in the DataFrame
#   - It's like a list of all the header names in the table
#   - Example: Index(['CustomerID', 'Age', 'Income', 'GRODI26_wRI', ...])
#
# "GRODI26_wRI" not in df.columns
#   - This checks if the string "GRODI26_wRI" is NOT present in the columns
#   - The "in" operator checks membership in a collection
#   - The "not in" operator is the opposite - checks for NON-membership
#   - Returns True if the column is missing, False if it exists
#
# if ...
#   - This is a conditional statement
#   - If the condition is True (column is missing), execute the indented block
#   - If the condition is False (column exists), skip the indented block
#
# WHY DO WE CHECK THIS?
# - Defensive programming: Don't assume the data is correct
# - If the column is missing and we try to use it, we'd get a cryptic error
# - By checking first, we can provide a clear, helpful error message
# - This helps users understand what went wrong and how to fix it
# -----------------------------------------------------------------------------
if "GRODI26_wRI" not in df.columns:
    
    # -------------------------------------------------------------------------
    # LINE: raise ValueError("Required column 'GRODI26_wRI' not found...")
    # -------------------------------------------------------------------------
    # This line raises (throws) an error with a descriptive message.
    #
    # BREAKING IT DOWN:
    #
    # raise
    #   - This Python keyword causes an exception to be thrown
    #   - When an exception is raised, normal execution stops
    #   - The error message is displayed to the user in KNIME
    #
    # ValueError
    #   - This is a type of exception in Python
    #   - It indicates that a function received an argument with the right
    #     type but an inappropriate value
    #   - In this case, the "value" is the input table, which is missing
    #     a required column
    #
    # "Required column 'GRODI26_wRI' not found in input table"
    #   - This is the error message that will be displayed
    #   - It clearly tells the user WHAT is missing
    #   - Good error messages help users fix problems quickly
    #
    # WHY ValueError AND NOT OTHER EXCEPTIONS?
    #   - KeyError would also be appropriate for missing columns
    #   - ValueError is more general and commonly used for validation
    #   - Either would work; the important thing is the clear message
    # -------------------------------------------------------------------------
    raise ValueError("Required column 'GRODI26_wRI' not found in input table")


# =============================================================================
# SECTION 4: CREATE THE isBad BINARY TARGET COLUMN
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: df["isBad"] = (df["GRODI26_wRI"] < 1).astype("Int32")
# -----------------------------------------------------------------------------
# This is the core logic of the script. It creates a new column called 
# "isBad" that contains 1 for bad customers and 0 for good customers.
#
# BREAKING IT DOWN (from inside out):
#
# df["GRODI26_wRI"]
#   - This accesses the GRODI26_wRI column from the DataFrame
#   - It returns a pandas Series (a single column of data)
#   - Example values: [0.5, 1.2, 0.8, 1.0, 0.99, ...]
#
# df["GRODI26_wRI"] < 1
#   - This performs an element-wise comparison
#   - Each value in the column is compared to 1
#   - Returns a Series of True/False values
#   - Example: [True, False, True, False, True, ...]
#       0.5 < 1  → True
#       1.2 < 1  → False
#       0.8 < 1  → True
#       1.0 < 1  → False (1 is NOT less than 1)
#       0.99 < 1 → True
#
# (df["GRODI26_wRI"] < 1)
#   - The parentheses group the comparison operation
#   - This is necessary because we want to call .astype() on the result
#
# .astype("Int32")
#   - This converts the True/False values to integers
#   - True becomes 1, False becomes 0
#   - Result: [1, 0, 1, 0, 1, ...]
#   
#   WHY "Int32" AND NOT "int"?
#   - "Int32" (capital I) is a NULLABLE integer type in pandas
#   - "int" (lowercase) is a standard Python integer that cannot be null
#   - KNIME tables may contain null/missing values
#   - If there are nulls in GRODI26_wRI, the comparison would produce NaN
#   - Using "Int32" allows the isBad column to contain null values if needed
#   - This prevents errors when transferring data back to KNIME
#
# df["isBad"] = ...
#   - This creates a new column called "isBad" in the DataFrame
#   - If a column named "isBad" already existed, it would be overwritten
#   - The new column contains the integer values we just computed
#
# BUSINESS INTERPRETATION:
#   isBad = 1 means "this customer is bad" (GRODI26_wRI < 1)
#   isBad = 0 means "this customer is good" (GRODI26_wRI >= 1)
# -----------------------------------------------------------------------------
df["isBad"] = (df["GRODI26_wRI"] < 1).astype("Int32")


# =============================================================================
# SECTION 5: REORDER COLUMNS TO PUT isBad FIRST
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: cols = df.columns.tolist()
# -----------------------------------------------------------------------------
# This line gets all column names as a Python list.
#
# BREAKING IT DOWN:
#
# df.columns
#   - Returns an Index object with all column names
#   - Example: Index(['CustomerID', 'Age', 'GRODI26_wRI', 'isBad', ...])
#
# .tolist()
#   - Converts the Index object to a regular Python list
#   - Lists are easier to manipulate (add, remove, reorder items)
#   - Example: ['CustomerID', 'Age', 'GRODI26_wRI', 'isBad', ...]
#
# cols = ...
#   - We store the list in a variable called "cols" (short for columns)
#   - We'll modify this list to change the column order
#
# WHY DO WE NEED A LIST?
#   - Index objects are immutable (cannot be changed)
#   - Lists are mutable (can be changed)
#   - We need to remove and add items, which requires a mutable type
# -----------------------------------------------------------------------------
cols = df.columns.tolist()

# -----------------------------------------------------------------------------
# LINE: cols.remove("isBad")
# -----------------------------------------------------------------------------
# This removes "isBad" from the column list.
#
# BREAKING IT DOWN:
#
# cols.remove("isBad")
#   - The .remove() method finds and removes the first occurrence of an item
#   - It modifies the list in-place (doesn't create a new list)
#   - After this, "isBad" is no longer in the cols list
#
# WHY REMOVE IT?
#   - We want to put "isBad" at the beginning of the list
#   - First we remove it from wherever it currently is
#   - Then we'll add it back at the beginning
#   - If we didn't remove it first, it would appear twice
#
# BEFORE: ['CustomerID', 'Age', 'GRODI26_wRI', 'isBad', ...]
# AFTER:  ['CustomerID', 'Age', 'GRODI26_wRI', ...]
# -----------------------------------------------------------------------------
cols.remove("isBad")

# -----------------------------------------------------------------------------
# LINE: cols = ["isBad"] + cols
# -----------------------------------------------------------------------------
# This prepends "isBad" to the beginning of the column list.
#
# BREAKING IT DOWN:
#
# ["isBad"]
#   - This creates a new list with just one element: "isBad"
#
# ["isBad"] + cols
#   - The + operator concatenates (joins) two lists together
#   - The left list comes first, then the right list
#   - Result: ["isBad"] + ['CustomerID', 'Age', ...]
#           = ['isBad', 'CustomerID', 'Age', ...]
#
# cols = ...
#   - We reassign the result back to cols
#   - Now cols has "isBad" at the beginning
#
# WHY PUT isBad FIRST?
#   - The target variable (dependent variable) is typically placed first
#   - This makes it easy to find when viewing the data
#   - Many modeling tools expect the target in the first column
#   - It's a standard convention in data science workflows
#
# BEFORE: ['CustomerID', 'Age', 'GRODI26_wRI', ...]
# AFTER:  ['isBad', 'CustomerID', 'Age', 'GRODI26_wRI', ...]
# -----------------------------------------------------------------------------
cols = ["isBad"] + cols

# -----------------------------------------------------------------------------
# LINE: df = df[cols]
# -----------------------------------------------------------------------------
# This reorders the DataFrame columns to match our new column order.
#
# BREAKING IT DOWN:
#
# df[cols]
#   - When you pass a list of column names to a DataFrame with square brackets,
#     it returns a new DataFrame with only those columns, in that order
#   - This is called column selection/reordering
#   - cols = ['isBad', 'CustomerID', 'Age', 'GRODI26_wRI', ...]
#   - df[cols] returns the DataFrame with columns in that exact order
#
# df = ...
#   - We reassign the reordered DataFrame back to the variable df
#   - This overwrites the old column order with the new one
#
# EXAMPLE:
#   Before: df has columns [CustomerID, Age, GRODI26_wRI, isBad]
#   After:  df has columns [isBad, CustomerID, Age, GRODI26_wRI]
#
# NOTE: The data itself hasn't changed, only the column arrangement.
# All values remain exactly the same.
# -----------------------------------------------------------------------------
df = df[cols]


# =============================================================================
# SECTION 6: LOG A SUMMARY OF THE RESULTS
# =============================================================================
# These print statements output helpful information to the KNIME console.
# This allows users to see a quick summary of the binary target distribution.
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: total_rows = len(df)
# -----------------------------------------------------------------------------
# This counts the total number of rows in the DataFrame.
#
# BREAKING IT DOWN:
#
# len(df)
#   - The len() function returns the length (number of items) of an object
#   - For a DataFrame, len() returns the number of rows
#   - Example: If df has 10,000 customers, len(df) returns 10000
#
# total_rows = ...
#   - We store this count in a variable for later use
#   - We'll use it to calculate percentages
# -----------------------------------------------------------------------------
total_rows = len(df)

# -----------------------------------------------------------------------------
# LINE: bad_count = (df["isBad"] == 1).sum()
# -----------------------------------------------------------------------------
# This counts how many rows have isBad = 1 (i.e., "bad" customers).
#
# BREAKING IT DOWN:
#
# df["isBad"]
#   - Accesses the isBad column
#   - Returns a Series of values: [1, 0, 1, 0, 1, 0, 0, ...]
#
# df["isBad"] == 1
#   - Performs element-wise comparison to 1
#   - Returns a Series of True/False: [True, False, True, False, ...]
#
# (df["isBad"] == 1).sum()
#   - The .sum() method adds up all values in the Series
#   - In Python/pandas, True = 1 and False = 0 for arithmetic
#   - So sum() effectively counts how many True values there are
#   - Example: [True, False, True, False, True] → 3
#
# bad_count = ...
#   - We store this count for use in the print statement
#
# WHY CHECK == 1 INSTEAD OF JUST sum()?
#   - We could do df["isBad"].sum() directly since isBad contains 1s and 0s
#   - Using == 1 is more explicit and self-documenting
#   - It makes the intent clearer: we're counting "bad" customers
# -----------------------------------------------------------------------------
bad_count = (df["isBad"] == 1).sum()

# -----------------------------------------------------------------------------
# LINE: good_count = (df["isBad"] == 0).sum()
# -----------------------------------------------------------------------------
# This counts how many rows have isBad = 0 (i.e., "good" customers).
#
# This works exactly the same as bad_count, but checks for 0 instead of 1.
#
# df["isBad"] == 0
#   - Returns True for every row where isBad is 0
#
# .sum()
#   - Counts the number of True values
#
# ALTERNATIVE CALCULATION:
#   We could calculate: good_count = total_rows - bad_count
#   But using the same pattern for both makes the code more consistent
#   and easier to understand. It also handles edge cases like null values.
# -----------------------------------------------------------------------------
good_count = (df["isBad"] == 0).sum()

# -----------------------------------------------------------------------------
# LINE: print(f"Total rows: {total_rows}")
# -----------------------------------------------------------------------------
# This prints the total number of rows to the KNIME console.
#
# BREAKING IT DOWN:
#
# print()
#   - Python's built-in function to output text
#   - In KNIME, this text appears in the node's console output
#   - Useful for debugging and verification
#
# f"..."
#   - This is an f-string (formatted string literal)
#   - Introduced in Python 3.6
#   - Allows embedding expressions inside string literals using { }
#
# f"Total rows: {total_rows}"
#   - The {total_rows} part is replaced with the value of the variable
#   - If total_rows = 10000, the output is: "Total rows: 10000"
#
# EXAMPLE OUTPUT: "Total rows: 10000"
# -----------------------------------------------------------------------------
print(f"Total rows: {total_rows}")

# -----------------------------------------------------------------------------
# LINE: print(f"isBad = 1 (GRODI26_wRI < 1): {bad_count} ({100*bad_count/total_rows:.2f}%)")
# -----------------------------------------------------------------------------
# This prints the count and percentage of "bad" customers.
#
# BREAKING IT DOWN:
#
# f"isBad = 1 (GRODI26_wRI < 1): {bad_count}"
#   - Shows the raw count of bad customers
#
# 100*bad_count/total_rows
#   - Calculates the percentage
#   - Example: 100 * 1500 / 10000 = 15.0 (meaning 15%)
#
# {100*bad_count/total_rows:.2f}
#   - :.2f is a format specifier
#   - .2 means show 2 decimal places
#   - f means format as a floating-point number
#   - Example: 15.0 becomes "15.00"
#
# FULL EXAMPLE OUTPUT:
#   "isBad = 1 (GRODI26_wRI < 1): 1500 (15.00%)"
#
# WHY IS THIS USEFUL?
#   - In credit risk, the "bad rate" is a crucial metric
#   - Typical bad rates range from 1% to 15% depending on the product
#   - This quick summary helps verify the data makes sense
#   - If you see 90% bad rate, something is probably wrong!
# -----------------------------------------------------------------------------
print(f"isBad = 1 (GRODI26_wRI < 1): {bad_count} ({100*bad_count/total_rows:.2f}%)")

# -----------------------------------------------------------------------------
# LINE: print(f"isBad = 0 (GRODI26_wRI >= 1): {good_count} ({100*good_count/total_rows:.2f}%)")
# -----------------------------------------------------------------------------
# This prints the count and percentage of "good" customers.
#
# This follows the exact same pattern as the previous print statement,
# but for good customers (isBad = 0).
#
# EXAMPLE OUTPUT:
#   "isBad = 0 (GRODI26_wRI >= 1): 8500 (85.00%)"
#
# VERIFICATION:
#   bad_count + good_count should equal total_rows
#   bad_percentage + good_percentage should equal 100%
#   If not, there might be null values in the data
# -----------------------------------------------------------------------------
print(f"isBad = 0 (GRODI26_wRI >= 1): {good_count} ({100*good_count/total_rows:.2f}%)")


# =============================================================================
# SECTION 7: WRITE OUTPUT DATA BACK TO KNIME
# =============================================================================

# -----------------------------------------------------------------------------
# LINE: knio.output_tables[0] = knio.Table.from_pandas(df)
# -----------------------------------------------------------------------------
# This is the final and critical step: sending the data back to KNIME.
#
# BREAKING IT DOWN (from inside out):
#
# df
#   - Our pandas DataFrame with the new "isBad" column
#   - Contains all original columns plus isBad as the first column
#
# knio.Table.from_pandas(df)
#   - Converts the pandas DataFrame back to a KNIME Table object
#   - This is the reverse of .to_pandas() we used at the beginning
#   - KNIME Table objects are what KNIME nodes use to pass data
#
# knio.output_tables[0]
#   - This is the first output port of the Python Script node
#   - Like input_tables, this is a list (0-indexed)
#   - output_tables[0] is the first output port
#   - output_tables[1] would be a second output port (if configured)
#
# knio.output_tables[0] = ...
#   - We assign our KNIME Table to the first output port
#   - This makes the data available to downstream nodes in KNIME
#   - Whatever node is connected to this output will receive this table
#
# WHAT HAPPENS NEXT?
#   - When this script finishes, KNIME takes the output_tables
#   - KNIME makes them available at the node's output ports
#   - Any node connected to this output will receive the table
#   - The workflow can then continue processing with the new isBad column
#
# ERROR SCENARIOS:
#   - If you try to write to output_tables[1] but the node only has 1 output
#     port configured, you'll get an IndexError
#   - If the DataFrame has invalid data types, KNIME may fail to convert
#   - If the DataFrame is empty, KNIME will pass through an empty table
# -----------------------------------------------------------------------------
knio.output_tables[0] = knio.Table.from_pandas(df)


# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# SUMMARY OF WHAT THIS SCRIPT DOES:
# 1. Reads input data from KNIME (first input port)
# 2. Validates that the required GRODI26_wRI column exists
# 3. Creates a binary isBad column (1 = bad, 0 = good)
# 4. Reorders columns to put isBad first
# 5. Prints a summary of the bad rate to the console
# 6. Writes the result back to KNIME (first output port)
#
# TYPICAL USE IN A WORKFLOW:
# This node would typically come early in a credit risk modeling pipeline,
# right after data loading. The isBad column created here becomes the
# target variable (dependent variable) for:
# - WOE binning (to calculate Weight of Evidence)
# - Variable selection (to find predictive features)
# - Logistic regression (to build the scorecard model)
# - Model evaluation (to measure AUC, K-S, etc.)
#
# =============================================================================

# =============================================================================
# CCR Score Filter Node for KNIME - Fully Commentated Version
# =============================================================================
# 
# This comment block provides a high-level overview of what this script does.
# 
# Purpose: This script is designed to run inside a KNIME 5.9 Python Script node.
#          It takes credit score data (specifically "CCR.score" which stands for
#          Consumer Credit Risk score) and performs two main operations:
#          1. Converts the score from a text/string format to a numeric format
#          2. Filters out any records that have scores below a specified cutoff value
#
# Why this is needed:
#   - CCR scores often come from external data sources as text strings
#   - For credit risk modeling, we need numeric values we can compare and calculate with
#   - We filter out low scores because they represent lower credit quality applicants
#     that may not meet minimum underwriting criteria
#
# Input: A single KNIME data table containing a column called "CCR.score" (as text)
# Output: A filtered KNIME data table with:
#         - A new column "CCR.score.num" containing the numeric score
#         - Only rows where the score meets or exceeds the cutoff threshold
#
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORT SECTION
# -----------------------------------------------------------------------------
# We import the libraries (pre-written code packages) that we need to use.

import knime.scripting.io as knio
# This line imports the KNIME Scripting Input/Output library and gives it a short name "knio".
# This library is REQUIRED for any Python script running inside KNIME.
# It provides the connection between KNIME's workflow (the visual node interface) and
# this Python script. Without it, we cannot read input data from KNIME or send output
# data back to KNIME.
# The "as knio" part creates an alias (nickname) so we can type "knio" instead of
# the full name "knime.scripting.io" every time we need to use it.

import pandas as pd
# This line imports the pandas library and gives it the short name "pd".
# Pandas is the most important Python library for working with tabular data (rows and columns).
# A pandas DataFrame is like an Excel spreadsheet or database table held in memory.
# We use pandas to manipulate, filter, and transform our data.
# The "as pd" is a standard convention - nearly all Python data scientists use this alias.

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# This section contains settings that control how the script behaves.
# By putting configuration values at the top of the script, they are easy to find
# and modify without having to search through the entire code.

CCR_SCORE_CUTOFF = 480
# This line creates a variable (a named storage container) called CCR_SCORE_CUTOFF.
# We set its value to 480, which is the minimum acceptable credit score threshold.
# 
# What this means for the business:
#   - Any applicant with a CCR score of 480 or higher will be kept in the data
#   - Any applicant with a CCR score below 480 will be filtered out (removed)
#   - Any applicant with a missing/invalid score will also be filtered out
#
# Why 480? This is a business decision based on credit risk policy.
# Scores below this level may represent unacceptable default risk.
#
# Variable naming convention:
#   - We use ALL_CAPS_WITH_UNDERSCORES for configuration constants
#   - This makes it visually obvious that this is a setting that should not
#     change during script execution, only between runs
#
# =============================================================================

# -----------------------------------------------------------------------------
# READ INPUT DATA FROM KNIME
# -----------------------------------------------------------------------------
# This section retrieves the data that was passed into this Python node from
# the upstream KNIME workflow.

df = knio.input_tables[0].to_pandas()
# This line does several things in one statement. Let's break it down:
#
# knio.input_tables
#   - This is a list (ordered collection) of all the input tables connected to this node
#   - KNIME Python Script nodes can have multiple input ports (connection points)
#   - Each input port feeds data into this list
#
# [0]
#   - This accesses the FIRST table in the list (Python counts from 0, not 1)
#   - So [0] means "the first input table" which is the top input port in KNIME
#   - If we had a second input, we would use [1], third would be [2], etc.
#
# .to_pandas()
#   - This is a method (function attached to an object) that converts the KNIME table
#     format into a pandas DataFrame format
#   - KNIME has its own internal table format that is not directly usable by pandas
#   - This conversion allows us to use all of pandas' powerful data manipulation features
#
# df =
#   - We assign the resulting pandas DataFrame to a variable named "df"
#   - "df" is a very common abbreviation for "DataFrame" in Python data science
#   - This variable now holds all our input data in memory, ready to be processed
#
# At this point, df contains all rows and columns from the KNIME input table,
# exactly as they appeared in the previous KNIME node's output.

# -----------------------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# -----------------------------------------------------------------------------
# Before processing, we check that the data contains the columns we expect.
# This is called "defensive programming" - we anticipate problems and handle them
# gracefully with clear error messages, rather than letting the script crash
# with a confusing error later.

if "CCR.score" not in df.columns:
    raise ValueError("Required column 'CCR.score' not found in input table")
# This is a conditional check (if statement) that validates our input data.
# Let's break it down:
#
# df.columns
#   - This property of the DataFrame returns a list of all column names
#   - Example: if df has columns A, B, C, then df.columns returns ["A", "B", "C"]
#
# "CCR.score" not in df.columns
#   - This is a boolean (True/False) expression
#   - It checks whether the text "CCR.score" is NOT present in the list of column names
#   - If the column exists, this evaluates to False and we skip the indented code
#   - If the column is missing, this evaluates to True and we execute the indented code
#
# raise ValueError("...")
#   - "raise" is a Python keyword that triggers an error (called an "exception")
#   - ValueError is a type of error indicating that a value is invalid or missing
#   - The message in quotes explains what went wrong
#   - When this line runs, the script stops immediately and KNIME shows this error message
#   - This is much better than letting the script continue and fail with a confusing
#     "KeyError" or similar when we try to use a column that doesn't exist
#
# Why we check for this specific column:
#   - The entire purpose of this script is to process the CCR.score column
#   - If it doesn't exist, there's nothing useful we can do
#   - By failing early with a clear message, we help the user understand what's wrong

# -----------------------------------------------------------------------------
# CONVERT CCR.SCORE FROM STRING TO NUMERIC
# -----------------------------------------------------------------------------
# The CCR.score column comes in as text (string type) but we need it as a number
# so we can compare it to our cutoff threshold.

df["CCR.score.num"] = pd.to_numeric(df["CCR.score"], errors="coerce")
# This line creates a new column and fills it with the numeric version of CCR.score.
# Let's break it down:
#
# df["CCR.score.num"]
#   - This is how we create a new column in a pandas DataFrame
#   - We use square brackets with the desired column name as a string
#   - If the column doesn't exist, pandas creates it
#   - If it did exist, this would overwrite it (which is fine in our case)
#   - The name "CCR.score.num" indicates this is the numeric version of CCR.score
#
# pd.to_numeric(...)
#   - This is a pandas function that converts values to numeric type
#   - It can convert strings like "480", "750.5", "-100" to actual numbers
#   - It works on entire columns (series) at once, which is very fast
#
# df["CCR.score"]
#   - This is the first argument to pd.to_numeric - the data we want to convert
#   - We're passing the entire CCR.score column
#
# errors="coerce"
#   - This is a named argument that tells pd.to_numeric how to handle conversion errors
#   - "coerce" means: if a value cannot be converted to a number, replace it with NaN
#   - NaN stands for "Not a Number" and represents missing/invalid data in pandas
#   - Other options are:
#     - errors="raise" - throw an error if any value can't be converted (we don't want this)
#     - errors="ignore" - return the original column unchanged (not useful for us)
#
# Example of what this does:
#   - "480" becomes 480.0 (a number)
#   - "575" becomes 575.0 (a number)
#   - "N/A" becomes NaN (not a number, will be filtered out later)
#   - "" (empty) becomes NaN
#   - "INVALID" becomes NaN
#   - null/None becomes NaN
#
# After this line, df has a new column called "CCR.score.num" with numeric values.

# -----------------------------------------------------------------------------
# LOG PRE-FILTER SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# Before we filter the data, we calculate and display summary statistics.
# This helps users understand their data and verify the script is working correctly.
# These print statements output to KNIME's Python console view.

total_rows = len(df)
# This line counts the total number of rows in the DataFrame.
#
# len() is a built-in Python function that returns the "length" of an object.
# For a DataFrame, this means the number of rows (not columns).
#
# We store this in a variable called "total_rows" for two reasons:
# 1. We'll use it later to calculate how many rows were removed
# 2. It makes our print statement more readable

null_count = df["CCR.score.num"].isna().sum()
# This line counts how many values in the new numeric column are null/NaN.
# Let's break it down:
#
# df["CCR.score.num"]
#   - Access the column we just created
#
# .isna()
#   - This is a pandas method that returns a series of True/False values
#   - True means the value is NaN/null, False means it's a valid number
#   - Example: [480, NaN, 575, NaN] becomes [False, True, False, True]
#
# .sum()
#   - When you sum True/False values, Python treats True as 1 and False as 0
#   - So summing [False, True, False, True] gives us 2
#   - This effectively counts how many null values exist
#
# Why we care about this:
#   - Null values will be filtered out (they fail the >= cutoff comparison)
#   - Knowing the null count helps users understand their data quality
#   - If the null count is unexpectedly high, it might indicate upstream data issues

below_cutoff = (df["CCR.score.num"] < CCR_SCORE_CUTOFF).sum()
# This line counts how many scores are BELOW the cutoff threshold.
# Let's break it down:
#
# df["CCR.score.num"] < CCR_SCORE_CUTOFF
#   - This compares every value in the column to our cutoff (480)
#   - It returns a series of True/False values
#   - True means the score is below 480, False means it's 480 or higher
#   - NaN values result in False for this comparison (NaN < 480 is False)
#
# The parentheses around the comparison ensure it's evaluated before .sum()
#
# .sum()
#   - Same as before, counts the True values
#   - Gives us the count of scores that are below the cutoff
#
# Note: This count does NOT include null values (they're counted separately above).
# The total rows that will be removed = null_count + below_cutoff (roughly).

print(f"Pre-filter summary:")
# This line prints a header text to the console.
#
# print() is a built-in Python function that outputs text to the console/terminal.
# In KNIME, this output appears in the Python Script node's console view.
#
# f"..." is called an f-string (formatted string literal).
# The 'f' prefix allows us to embed variables and expressions inside {curly braces}.
# In this case, we're not using any variables, just printing plain text.
# We could have written print("Pre-filter summary:") but f-strings are a good habit.

print(f"  Total rows: {total_rows}")
# This line prints the total row count with some formatting.
#
# The two spaces at the start ("  Total...") create visual indentation,
# making it clear this is a sub-item under the "Pre-filter summary" header.
#
# {total_rows} is replaced with the actual value of the total_rows variable.
# If total_rows is 10000, this prints: "  Total rows: 10000"

print(f"  Null/NA values: {null_count}")
# Prints the count of null/missing values, indented to align with the line above.
# This helps users quickly see how many records have invalid or missing CCR scores.

print(f"  Values below cutoff ({CCR_SCORE_CUTOFF}): {below_cutoff}")
# Prints the count of scores below the cutoff.
# We also include the cutoff value itself ({CCR_SCORE_CUTOFF}) so users can see
# exactly what threshold was used without having to look at the configuration section.
# Example output: "  Values below cutoff (480): 1234"

# -----------------------------------------------------------------------------
# FILTER ROWS BASED ON CUTOFF THRESHOLD
# -----------------------------------------------------------------------------
# This is the core filtering operation - we remove rows that don't meet our criteria.

df_filtered = df[df["CCR.score.num"] >= CCR_SCORE_CUTOFF].copy()
# This line creates a new DataFrame containing only the rows that pass our filter.
# This is the most important line in the script. Let's break it down:
#
# df["CCR.score.num"] >= CCR_SCORE_CUTOFF
#   - This creates a boolean (True/False) series
#   - For each row, it checks: is this score greater than or equal to 480?
#   - Example: [480, 475, 550, NaN, 400] becomes [True, False, True, False, False]
#   - IMPORTANT: NaN >= 480 evaluates to False, so null values are automatically excluded
#
# df[...]
#   - When you put a boolean series inside square brackets after a DataFrame,
#     pandas returns only the rows where the value is True
#   - This is called "boolean indexing" or "boolean filtering"
#   - It's the standard way to filter data in pandas
#
# .copy()
#   - This creates an independent copy of the filtered data
#   - Without .copy(), df_filtered would be a "view" of df, not a separate object
#   - Views can cause confusing warnings and bugs when you try to modify them
#   - Using .copy() is a best practice for creating filtered subsets
#
# df_filtered =
#   - We store the filtered result in a new variable called "df_filtered"
#   - We use a new variable name (not overwriting 'df') for two reasons:
#     1. We can still access the original data in 'df' if needed
#     2. It's clearer to readers that 'df_filtered' is the processed version
#
# What gets filtered OUT (removed):
#   - Rows where CCR.score.num is less than 480
#   - Rows where CCR.score.num is null/NaN (because NaN >= 480 is False)
#
# What stays IN:
#   - Rows where CCR.score.num is 480 or higher

# -----------------------------------------------------------------------------
# LOG POST-FILTER SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# After filtering, we display summary statistics to show what happened.
# This helps users verify the filter worked as expected.

filtered_rows = len(df_filtered)
# Count the number of rows that passed the filter (i.e., rows remaining).
# We use the same len() function as before, but now on the filtered DataFrame.

removed_rows = total_rows - filtered_rows
# Calculate how many rows were removed by the filter.
# This is simply: (original count) minus (remaining count).
# This includes both rows with low scores AND rows with null scores.

print(f"\nPost-filter summary:")
# Print a header for the post-filter statistics.
#
# The \n at the beginning is a "newline character" - it creates a blank line
# before this text. This visually separates the post-filter stats from the
# pre-filter stats in the console output, making it easier to read.

print(f"  Rows kept: {filtered_rows}")
# Print how many rows remain after filtering.
# This is the number of applicants whose scores met the minimum threshold.

print(f"  Rows removed: {removed_rows}")
# Print how many rows were removed by the filter.
# This includes both low-score rows and null/invalid score rows.
# If this number seems too high or too low, users should investigate their data.

print(f"  Cutoff used: CCR.score.num >= {CCR_SCORE_CUTOFF}")
# Print the exact filter condition that was applied.
# This documents what threshold was used, which is helpful for:
# - Audit trails and compliance documentation
# - Debugging if results seem unexpected
# - Confirming the correct cutoff was applied

# -----------------------------------------------------------------------------
# WRITE OUTPUT DATA BACK TO KNIME
# -----------------------------------------------------------------------------
# This final section sends our processed data back to KNIME so it can be
# used by downstream nodes in the workflow.

knio.output_tables[0] = knio.Table.from_pandas(df_filtered)
# This line sends our filtered DataFrame back to KNIME as the output.
# Let's break it down:
#
# knio.output_tables
#   - This is a list (like input_tables) that holds the output tables for KNIME
#   - KNIME Python Script nodes can have multiple output ports
#   - Each index in this list corresponds to one output port
#
# [0]
#   - We're writing to the FIRST (and only) output port
#   - If this node had multiple output ports, we could write to [1], [2], etc.
#
# knio.Table.from_pandas(df_filtered)
#   - This converts our pandas DataFrame back into KNIME's table format
#   - It's the reverse of the .to_pandas() we used at the beginning
#   - KNIME cannot directly use pandas DataFrames, so this conversion is required
#
# The = assignment puts our converted table into the output port.
# When the script finishes, KNIME reads this output and passes it to the next node.
#
# The output table will contain:
#   - All original columns from the input (including CCR.score as a string)
#   - The new CCR.score.num column (as a numeric type)
#   - Only rows where CCR.score.num >= 480

# =============================================================================
# END OF SCRIPT
# =============================================================================
# 
# Summary of what this script accomplished:
# 1. Read a data table from KNIME's input port
# 2. Validated that the required CCR.score column exists
# 3. Converted the CCR.score string column to a numeric column (CCR.score.num)
# 4. Logged statistics about the data before filtering
# 5. Filtered out rows with scores below 480 or with null/invalid scores
# 6. Logged statistics about the filtered results
# 7. Sent the filtered data to KNIME's output port
#
# Common troubleshooting:
# - If you see "Required column 'CCR.score' not found", check that the upstream
#   node provides a column with exactly that name (case-sensitive!)
# - If too many rows are filtered out, check for data quality issues in CCR.score
# - If no rows are filtered, the cutoff might be set too low
# - To change the cutoff, modify the CCR_SCORE_CUTOFF value at the top
#
# =============================================================================

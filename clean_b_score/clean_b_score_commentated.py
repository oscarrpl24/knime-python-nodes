# ==============================================================================
# KNIME Python Script: Clean b_Score Column (Fully Commentated Version)
# ==============================================================================
#
# PURPOSE:
# This script is designed to run inside a KNIME 5.9 Python Script node.
# Its sole function is to clean the 'b_Score' column by removing any single
# quote characters (') that may have been inadvertently introduced during
# data import, transformation, or transfer between systems.
#
# WHY THIS IS NEEDED:
# In credit risk modeling workflows, score values are often passed between
# systems as text. Sometimes single quotes get wrapped around numeric values
# (e.g., "'750'" instead of "750"), which causes issues when:
#   - Converting the score to a numeric type
#   - Performing mathematical operations on the score
#   - Comparing or sorting scores
#   - Exporting data to downstream systems that expect clean numeric strings
#
# INPUT:
# - Port 0: A single KNIME table containing at least a column named 'b_Score'
#   The b_Score column typically contains credit score values that may have
#   unwanted single quote characters embedded in them.
#
# OUTPUT:
# - Port 0: The same table with the 'b_Score' column cleaned (quotes removed)
#   All other columns remain unchanged.
#
# ==============================================================================

# ------------------------------------------------------------------------------
# IMPORT SECTION
# ------------------------------------------------------------------------------

# Import the KNIME Python scripting I/O module.
# This module is REQUIRED for all KNIME Python Script nodes - it provides
# the interface between your Python code and the KNIME workflow.
#
# The 'knio' alias is a conventional shorthand that makes the code more readable.
# This module provides access to:
#   - knio.input_tables[]  : Read data from KNIME input ports
#   - knio.output_tables[] : Write data to KNIME output ports
#   - knio.flow_variables  : Read/write KNIME flow variables
#   - knio.Table           : Convert between KNIME tables and pandas DataFrames
import knime.scripting.io as knio

# ------------------------------------------------------------------------------
# DATA INPUT SECTION
# ------------------------------------------------------------------------------

# Read the first (and in this case, only) input table from KNIME.
#
# BREAKDOWN OF THIS LINE:
# - knio.input_tables    : This is a list-like object containing all input tables
#                          connected to the Python Script node in KNIME
# - [0]                  : Access the first input table (Python uses 0-based indexing,
#                          so [0] means "the first element")
# - .to_pandas()         : Convert the KNIME table to a pandas DataFrame
#                          This allows us to use all of pandas' powerful data
#                          manipulation functions
#
# The result is stored in a variable called 'df' (short for DataFrame),
# which is a conventional name used in data science for DataFrame variables.
#
# IMPORTANT: This creates a COPY of the data in memory. Any changes we make
# to 'df' will not affect the original KNIME table until we explicitly write
# the modified DataFrame back to an output port.
df = knio.input_tables[0].to_pandas()

# ------------------------------------------------------------------------------
# VALIDATION SECTION
# ------------------------------------------------------------------------------

# Check if the required 'b_Score' column exists in the input DataFrame.
#
# BREAKDOWN OF THIS LINE:
# - 'b_Score'           : The exact name of the column we're looking for
#                         (column names in KNIME/pandas are CASE-SENSITIVE!)
# - not in              : Python's membership test operator (checks if something
#                         is NOT contained in a collection)
# - df.columns          : A list-like object (Index) containing all column names
#                         in the DataFrame
#
# This is a DEFENSIVE PROGRAMMING practice - we verify our assumptions about
# the input data before proceeding. Without this check, the script would fail
# with a cryptic "KeyError" message if the column didn't exist.
if 'b_Score' not in df.columns:
    
    # If the column is missing, raise a ValueError exception with a clear,
    # human-readable error message.
    #
    # BREAKDOWN:
    # - raise             : Python keyword that triggers an exception
    # - ValueError        : A built-in exception type indicating that a function
    #                       received an argument of the right type but an
    #                       inappropriate value
    # - "Column..."       : The error message that will be displayed to the user
    #                       in KNIME's error log
    #
    # This is MUCH better than letting the script fail with a generic error,
    # because the user will immediately know what went wrong and how to fix it
    # (i.e., ensure the input table has a column named 'b_Score').
    raise ValueError("Column 'b_Score' not found in input table")

# ------------------------------------------------------------------------------
# DATA CLEANING SECTION
# ------------------------------------------------------------------------------

# Remove all single quote characters (') from the b_Score column.
#
# This is a CHAINED operation, meaning multiple methods are called in sequence.
# Let's break down each part of this line:
#
# STEP 1: df['b_Score']
#   - Access the 'b_Score' column from the DataFrame
#   - This returns a pandas Series (a single column of data)
#
# STEP 2: .astype(str)
#   - Convert all values in the column to string type
#   - This is a SAFETY measure because:
#     * If some values are already numeric (int/float), str.replace() would fail
#     * If there are None/NaN values, they become the string "nan"
#     * This ensures we can safely perform string operations on every value
#
# STEP 3: .str.replace("'", "", regex=False)
#   - .str           : Access the string methods of the Series (pandas string accessor)
#   - .replace()     : The string replacement method
#   - "'"            : The pattern to search for (a single quote character)
#   - ""             : The replacement string (empty string = delete the match)
#   - regex=False    : IMPORTANT! This tells pandas to treat the search pattern
#                      as a literal string, not a regular expression. This is:
#                      a) Faster (no regex parsing overhead)
#                      b) Safer (single quote doesn't have special regex meaning,
#                         but being explicit is good practice)
#
# The result of this entire chain is assigned back to df['b_Score'], effectively
# replacing the original column with the cleaned version.
#
# EXAMPLE:
#   Before: "'750'"  ->  After: "750"
#   Before: "700'"   ->  After: "700"
#   Before: "'650"   ->  After: "650"
#   Before: "600"    ->  After: "600" (no change if no quotes present)
df['b_Score'] = df['b_Score'].astype(str).str.replace("'", "", regex=False)

# ------------------------------------------------------------------------------
# DATA OUTPUT SECTION
# ------------------------------------------------------------------------------

# Write the cleaned DataFrame back to KNIME's first output port.
#
# BREAKDOWN OF THIS LINE:
# - knio.output_tables[0]  : Access the first output port of the KNIME node
#                            (again, 0-based indexing)
# - =                      : Assignment operator - we're setting the output
# - knio.Table.from_pandas(df) : Convert our pandas DataFrame back into a KNIME
#                                table format that KNIME can understand and pass
#                                to downstream nodes
#
# IMPORTANT NOTES:
# 1. This is a REQUIRED step! If you don't assign to output_tables, the KNIME
#    node will have no output and downstream nodes won't receive any data.
#
# 2. The number of output assignments must match the number of output ports
#    configured in the KNIME Python Script node settings. This script only
#    uses one output port [0].
#
# 3. The conversion from pandas to KNIME table handles data type mapping
#    automatically, but be aware that:
#    - pandas 'object' dtype becomes KNIME 'String'
#    - pandas 'int64' becomes KNIME 'Long Integer'
#    - pandas 'float64' becomes KNIME 'Double'
#    - For nullable integers, use pandas 'Int64' (capital I) to preserve nulls
#
# After this line executes, the KNIME workflow can continue with the next node,
# which will receive the cleaned data.
knio.output_tables[0] = knio.Table.from_pandas(df)

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
#
# SUMMARY OF WHAT THIS SCRIPT DOES:
# 1. Imports the KNIME Python interface module
# 2. Reads the input data table from KNIME into a pandas DataFrame
# 3. Validates that the required 'b_Score' column exists
# 4. Removes all single quote characters from the 'b_Score' column
# 5. Outputs the cleaned DataFrame back to KNIME
#
# POTENTIAL IMPROVEMENTS (not implemented to keep script simple):
# - Handle NaN values explicitly (currently they become the string "nan")
# - Remove double quotes as well, if needed
# - Convert the cleaned b_Score to numeric type (int or float)
# - Add logging for debugging purposes
# - Handle multiple score columns with a loop
#
# TROUBLESHOOTING:
# - If you get "Column 'b_Score' not found": Check the input table has this
#   exact column name (case-sensitive)
# - If quotes aren't being removed: Check if they're actually single quotes (')
#   and not backticks (`) or other similar characters
# - If you see "nan" values in output: Some input values were NaN/None
#
# ==============================================================================

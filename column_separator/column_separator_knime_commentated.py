# =============================================================================
# COLUMN SEPARATOR NODE FOR KNIME - FULLY COMMENTATED VERSION
# =============================================================================
#
# FILE NAME: column_separator_knime_commentated.py
#
# WHAT THIS SCRIPT DOES:
# This script is designed to run inside a KNIME Python Script node. Its job is
# to take a single input table and split its columns into two separate output
# tables based on specific naming patterns. Think of it as a column filter that
# routes columns to different destinations based on what their names look like.
#
# WHY THIS IS USEFUL:
# In credit risk modeling workflows, you often have datasets with many different
# types of columns mixed together:
#   - Raw feature columns (like "age", "income", "loan_amount")
#   - Derived WOE (Weight of Evidence) columns (like "WOE_age", "WOE_income")
#   - Scorecard point columns (like "age_points", "income_points")
#   - Metadata columns (like "application_date", "customer_code", "node_id")
#
# This script helps you separate the "main" analytical columns from the
# metadata/derived columns so you can process them differently downstream.
#
# OUTPUTS:
#   Output 1 (Port 0): Columns that DON'T match any filtering criteria
#                      (typically your main analytical data)
#   Output 2 (Port 1): Columns that DO match filtering criteria
#                      (metadata, WOE columns, score columns, etc.)
#
# FILTERING CRITERIA (columns sent to Output 2):
#   - Starts with "b_"           → Binary indicator columns
#   - Contains "date"            → Date-related columns
#   - Contains "code"            → Code/ID columns
#   - Starts with "WOE_"         → Weight of Evidence transformed columns
#   - Ends with "_points"        → Scorecard point columns
#   - Contains "basescore"       → Scorecard base score columns
#   - Contains "unq_id"          → Unique identifier columns
#   - Contains "nodeid"          → KNIME node tracking columns
#   - Contains "avg_"            → Averaged/aggregated columns
#
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

# This line imports the KNIME scripting input/output module and gives it the
# short alias "knio". This module is the bridge between Python and KNIME.
# It provides access to:
#   - knio.input_tables[]  → Read data from KNIME input ports
#   - knio.output_tables[] → Write data to KNIME output ports
#   - knio.flow_variables  → Read/write KNIME flow variables
# The "knime.scripting.io" module is automatically available in KNIME Python
# Script nodes - you don't need to install it separately.
import knime.scripting.io as knio


# =============================================================================
# SECTION 2: READ INPUT DATA
# =============================================================================
# This section retrieves the input data from KNIME and converts it to a format
# that Python can work with easily (a pandas DataFrame).
# =============================================================================

# DETAILED BREAKDOWN OF THIS LINE:
#
# knio.input_tables[0]
#   ├── knio            → The KNIME I/O module we imported above
#   ├── input_tables    → A list-like object containing all input tables
#   │                     connected to this KNIME node's input ports
#   └── [0]             → Access the FIRST input port (ports are 0-indexed)
#                         If this node had multiple input ports, [1] would be
#                         the second port, [2] the third, etc.
#
# .to_pandas()
#   └── This method converts the KNIME table format into a pandas DataFrame.
#       Pandas is a powerful Python library for data manipulation. DataFrames
#       are essentially in-memory tables with rows and columns, similar to
#       Excel spreadsheets or SQL tables.
#
# df = ...
#   └── We store the result in a variable named "df" (short for DataFrame).
#       This is a common naming convention in Python data science code.
#
# After this line executes, "df" contains all the data from the first input
# port, with column names preserved and data types automatically converted.
df = knio.input_tables[0].to_pandas()


# =============================================================================
# SECTION 3: DEFINE THE COLUMN CLASSIFICATION FUNCTION
# =============================================================================
# This section defines a function that decides whether a given column should
# be routed to Output 1 (main data) or Output 2 (filtered/metadata).
# =============================================================================

# FUNCTION DEFINITION:
# The "def" keyword creates a new function in Python.
# "should_go_to_output2" is the name we're giving this function.
# "(col_name)" means this function takes one input parameter called "col_name",
# which will be the name of a column (as a text string).
def should_go_to_output2(col_name):
    """
    DOCSTRING (documentation inside triple quotes):
    This function examines a column name and determines if it matches any of
    the criteria that would route it to Output 2.
    
    PARAMETERS:
        col_name (str): The name of a column to evaluate
        
    RETURNS:
        bool: True if this column should go to Output 2
              False if this column should go to Output 1
    
    The function uses pattern matching on the column name to make its decision.
    Some patterns are case-sensitive (like "WOE_") while others are
    case-insensitive (like "date").
    """
    
    # CREATE A LOWERCASE VERSION OF THE COLUMN NAME:
    # The .lower() method converts all uppercase letters to lowercase.
    # For example: "ApplicationDate" becomes "applicationdate"
    #              "WOE_Age" becomes "woe_age"
    #              "customer_code" stays "customer_code" (already lowercase)
    #
    # We store this in a separate variable (col_lower) because:
    #   1. We need to keep the original name (col_name) for case-sensitive checks
    #   2. Calling .lower() once and reusing is more efficient than calling it
    #      multiple times
    col_lower = col_name.lower()
    
    # -------------------------------------------------------------------------
    # CHECK 1: Does the column name start with "b_"?
    # -------------------------------------------------------------------------
    # The .startswith() method returns True if the string begins with the
    # specified prefix, False otherwise.
    #
    # WHY THIS PATTERN:
    # In credit risk modeling, columns prefixed with "b_" often represent
    # binary (0/1) indicator variables or binned versions of continuous
    # variables. For example:
    #   - "b_is_employed" → Binary flag for employment status
    #   - "b_age_group"   → Binned age category
    #
    # NOTE: This check is CASE-SENSITIVE (uses col_name, not col_lower)
    # because the "b_" prefix is typically lowercase by convention.
    if col_name.startswith("b_"):
        # If the condition is True, immediately return True and exit the function.
        # No need to check the remaining conditions.
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 2: Does the column name contain "date" anywhere?
    # -------------------------------------------------------------------------
    # The "in" operator checks if one string exists inside another string.
    # For example:
    #   "date" in "application_date"     → True
    #   "date" in "ApplicationDate"      → False (case mismatch)
    #   "date" in "applicationdate"      → True (after lowercasing)
    #   "date" in "created_at_timestamp" → False (no "date" substring)
    #
    # WHY THIS PATTERN:
    # Date columns (like "application_date", "DateOfBirth", "created_date")
    # are typically metadata rather than analytical features. They're often
    # used for filtering or joining but not for model building.
    #
    # NOTE: This check is CASE-INSENSITIVE (uses col_lower) because date
    # columns can be named in various styles (camelCase, snake_case, etc.)
    if "date" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 3: Does the column name contain "code" anywhere?
    # -------------------------------------------------------------------------
    # Similar to the date check, but looking for "code" substring.
    #
    # WHY THIS PATTERN:
    # Code columns (like "branch_code", "ProductCode", "occupation_code")
    # are typically categorical identifiers. While they might be useful for
    # some analyses, they're often considered metadata rather than features.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    if "code" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 4: Does the column name start with "WOE_"?
    # -------------------------------------------------------------------------
    # Checks if the column name begins with the prefix "WOE_".
    #
    # WHY THIS PATTERN:
    # WOE (Weight of Evidence) columns are transformed versions of original
    # features. In a credit risk workflow, you often want to keep the WOE
    # columns separate from the original raw features because:
    #   - They're used for logistic regression modeling
    #   - They shouldn't be confused with the original values
    #
    # TYPICAL NAMING:
    #   Original column: "age"
    #   WOE column: "WOE_age"
    #
    # NOTE: This check is CASE-SENSITIVE (uses col_name, not col_lower)
    # because "WOE_" is a standardized uppercase prefix in this workflow.
    if col_name.startswith("WOE_"):
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 5: Does the column name end with "_points"?
    # -------------------------------------------------------------------------
    # The .endswith() method returns True if the string ends with the
    # specified suffix, False otherwise.
    #
    # WHY THIS PATTERN:
    # Scorecard point columns (like "age_points", "income_points") are
    # derived columns that represent the points assigned for each binned
    # value in a credit scorecard. These are outputs of the scorecard
    # transformation and should be kept separate from raw data.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower) because different
    # systems might use "Age_Points" or "age_points".
    if col_lower.endswith("_points"):
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 6: Does the column name contain "basescore" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "basescore" substring.
    #
    # WHY THIS PATTERN:
    # The base score is a constant offset in a credit scorecard calculation.
    # It's derived from the model intercept and is a derived value, not a
    # raw feature. Columns like "BaseScore" or "basescore_adjustment" would
    # match this pattern.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    if "basescore" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 7: Does the column name contain "unq_id" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "unq_id" substring.
    #
    # WHY THIS PATTERN:
    # "unq_id" typically stands for "unique identifier". These are key
    # columns used to uniquely identify each row (like customer ID,
    # application ID, etc.). They're essential for joining tables but
    # are not analytical features.
    #
    # EXAMPLES:
    #   - "customer_unq_id" → Unique customer identifier
    #   - "unq_id_application" → Application identifier
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    if "unq_id" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 8: Does the column name contain "nodeid" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "nodeid" substring.
    #
    # WHY THIS PATTERN:
    # "nodeid" columns are typically internal KNIME tracking columns that
    # record which node processed the data. These are metadata for workflow
    # debugging and not analytical features.
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    if "nodeid" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # CHECK 9: Does the column name contain "avg_" anywhere?
    # -------------------------------------------------------------------------
    # Checks for the "avg_" substring.
    #
    # WHY THIS PATTERN:
    # Columns prefixed or containing "avg_" are typically aggregated values
    # (averages). These are often summary statistics or derived features
    # that might need to be handled separately from raw features.
    #
    # EXAMPLES:
    #   - "avg_balance" → Average account balance
    #   - "monthly_avg_spend" → Monthly average spending
    #
    # NOTE: CASE-INSENSITIVE check (uses col_lower)
    if "avg_" in col_lower:
        return True
    
    # -------------------------------------------------------------------------
    # DEFAULT CASE: No patterns matched
    # -------------------------------------------------------------------------
    # If we've reached this point, the column name didn't match ANY of the
    # criteria above. This means it should go to Output 1 (main data).
    # We return False to indicate "No, this column should NOT go to Output 2".
    return False


# =============================================================================
# SECTION 4: SEPARATE COLUMNS INTO TWO GROUPS
# =============================================================================
# This section loops through all columns in the input DataFrame and decides
# which output each column should go to.
# =============================================================================

# CREATE TWO EMPTY LISTS TO STORE COLUMN NAMES:
# In Python, [] creates an empty list. Lists are ordered collections that
# can grow as you add items to them.
#
# output1_cols will store names of columns destined for Output 1 (main data)
output1_cols = []

# output2_cols will store names of columns destined for Output 2 (filtered)
output2_cols = []

# LOOP THROUGH ALL COLUMNS IN THE DATAFRAME:
# df.columns gives us an Index object containing all column names in the
# DataFrame. The "for col in ..." syntax iterates through each column name
# one at a time, assigning the current name to the variable "col".
#
# For example, if df has columns ["name", "age", "WOE_income"], this loop
# will run 3 times with col being "name", then "age", then "WOE_income".
for col in df.columns:
    
    # CALL OUR CLASSIFICATION FUNCTION:
    # For each column, we call the should_go_to_output2() function we defined
    # earlier, passing in the current column name. The function returns True
    # or False based on whether the column matches any filtering criteria.
    if should_go_to_output2(col):
        # IF THE FUNCTION RETURNED TRUE:
        # This column matches one of the filtering criteria, so it should
        # go to Output 2. We add (append) this column name to the output2_cols
        # list using the .append() method.
        output2_cols.append(col)
    else:
        # IF THE FUNCTION RETURNED FALSE:
        # This column doesn't match any criteria, so it stays in the main
        # data (Output 1). We add this column name to output1_cols.
        output1_cols.append(col)


# =============================================================================
# SECTION 5: CREATE OUTPUT DATAFRAMES
# =============================================================================
# Now we create the actual DataFrames for each output by selecting only the
# relevant columns from the original DataFrame.
# =============================================================================

# CREATE OUTPUT 1 DATAFRAME (Main Data):
# This line has a conditional expression (Python's ternary operator):
#   result = value_if_true if condition else value_if_false
#
# CONDITION: output1_cols (evaluates to True if the list is not empty)
#
# IF TRUE (there are columns for output 1):
#   df[output1_cols] creates a new DataFrame containing only the columns
#   whose names are in the output1_cols list. This is called column selection
#   or column slicing in pandas.
#
# IF FALSE (output1_cols is empty, no columns for output 1):
#   df.iloc[:, 0:0] creates an empty DataFrame with:
#     - All the same rows as the original df (the ":" before the comma)
#     - Zero columns (the "0:0" after the comma selects columns 0 up to 0,
#       which is an empty range)
#   This preserves the DataFrame structure (index, etc.) while having no data.
#
# WHY THE EMPTY CHECK IS IMPORTANT:
# If we tried to do df[[]] (select with an empty list), pandas would give us
# an error or unexpected behavior. The iloc approach safely creates an empty
# DataFrame with the correct structure.
df_output1 = df[output1_cols] if output1_cols else df.iloc[:, 0:0]

# CREATE OUTPUT 2 DATAFRAME (Filtered Columns):
# Same logic as above, but for the filtered columns going to Output 2.
df_output2 = df[output2_cols] if output2_cols else df.iloc[:, 0:0]


# =============================================================================
# SECTION 6: LOGGING / SUMMARY OUTPUT
# =============================================================================
# This section prints a summary of what the script did. These print statements
# appear in KNIME's console output, which is useful for debugging and
# verifying that the column separation worked as expected.
# =============================================================================

# PRINT THE TOTAL NUMBER OF INPUT COLUMNS:
# The f"..." syntax creates a "formatted string" (f-string) in Python.
# Expressions inside curly braces {} are evaluated and inserted into the string.
#
# len(df.columns) returns the number of columns in the original DataFrame.
# For example, if df has 50 columns, this prints: "Input columns: 50"
print(f"Input columns: {len(df.columns)}")

# PRINT THE NUMBER OF COLUMNS GOING TO OUTPUT 1:
# len(output1_cols) counts how many items are in the output1_cols list.
print(f"Output 1 columns (main data): {len(output1_cols)}")

# PRINT THE NUMBER OF COLUMNS GOING TO OUTPUT 2:
# len(output2_cols) counts how many items are in the output2_cols list.
print(f"Output 2 columns (filtered): {len(output2_cols)}")

# OPTIONALLY PRINT THE NAMES OF FILTERED COLUMNS:
# This "if" statement checks if output2_cols is not empty (any non-empty
# list evaluates to True in Python).
if output2_cols:
    # Print a blank line and header for readability
    # The \n creates a newline character, adding vertical spacing
    print(f"\nColumns sent to Output 2:")
    
    # Loop through each column name in output2_cols and print it
    # The "  - " prefix creates a bulleted list appearance in the console
    for col in output2_cols:
        print(f"  - {col}")


# =============================================================================
# SECTION 7: WRITE OUTPUTS TO KNIME
# =============================================================================
# This final section sends our two DataFrames back to KNIME so they can be
# used by downstream nodes in the workflow.
# =============================================================================

# WRITE OUTPUT 1 (Main Data) TO KNIME:
# DETAILED BREAKDOWN:
#
# knio.output_tables[0]
#   ├── knio           → The KNIME I/O module
#   ├── output_tables  → A list-like object for all output ports
#   └── [0]            → The FIRST output port (index 0)
#
# = knio.Table.from_pandas(df_output1)
#   ├── knio.Table     → The KNIME Table class
#   ├── .from_pandas() → A class method that converts a pandas DataFrame
#   │                    into a KNIME-compatible table format
#   └── (df_output1)   → The DataFrame we want to convert and send
#
# IMPORTANT: The KNIME node must be configured with the correct number of
# output ports. If this script writes to output_tables[0] and [1], the node
# needs at least 2 output ports configured.
#
# After this line, the main data (columns that didn't match any filter
# criteria) will be available at Output Port 1 in KNIME.
knio.output_tables[0] = knio.Table.from_pandas(df_output1)

# WRITE OUTPUT 2 (Filtered Columns) TO KNIME:
# Same pattern as above, but writing to the second output port (index 1).
# After this line, the filtered columns (WOE columns, date columns, etc.)
# will be available at Output Port 2 in KNIME.
knio.output_tables[1] = knio.Table.from_pandas(df_output2)


# =============================================================================
# END OF SCRIPT
# =============================================================================
#
# EXECUTION FLOW SUMMARY:
# 1. Import the KNIME I/O module
# 2. Read the input table from KNIME's first input port
# 3. Define a function to classify columns based on naming patterns
# 4. Loop through all columns and categorize them into two groups
# 5. Create two separate DataFrames from those groups
# 6. Print a summary of the separation for logging purposes
# 7. Send both DataFrames to KNIME's output ports
#
# COMMON MODIFICATIONS:
# - To add a new filtering pattern, add another "if" block in the
#   should_go_to_output2() function following the same pattern
# - To change a pattern from case-insensitive to case-sensitive (or vice
#   versa), change whether you use col_name or col_lower in the check
# - To invert the logic (so matched columns go to Output 1), swap the
#   output1_cols and output2_cols assignments in the for loop
#
# DEBUGGING TIPS:
# - If columns are going to the wrong output, add print statements inside
#   the should_go_to_output2() function to see which patterns are matching
# - Check that your column names match the expected case (especially for
#   case-sensitive patterns like "WOE_")
# - Verify that your KNIME node has two output ports configured
#
# =============================================================================

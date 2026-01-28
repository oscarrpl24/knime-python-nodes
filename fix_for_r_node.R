# R Script for KNIME Table to R Node
# This version handles problematic column names

# Read the input - wrap in tryCatch to get better error messages
df <- tryCatch({
  knime.in
}, error = function(e) {
  message("Error reading knime.in: ", e$message)
  # Try to get more diagnostic info
  message("Class of knime.in: ", class(knime.in))
  stop(e)
})

# If we got here, data was read successfully
message("Successfully read data: ", nrow(df), " rows, ", ncol(df), " columns")

# Fix column names that have R-incompatible characters
# This is important if KNIME passed problematic names
original_names <- colnames(df)
fixed_names <- make.names(original_names, unique = TRUE)

# Log any changes
changed <- which(original_names != fixed_names)
if (length(changed) > 0) {
  message("Fixed ", length(changed), " column names:")
  for (i in head(changed, 10)) {
    message("  '", original_names[i], "' -> '", fixed_names[i], "'")
  }
  if (length(changed) > 10) {
    message("  ... and ", length(changed) - 10, " more")
  }
  colnames(df) <- fixed_names
}

# Handle any NA issues - convert empty strings to NA if needed
# (Uncomment if you suspect empty strings are causing issues)
# df[df == ""] <- NA

# Output
knime.out <- df

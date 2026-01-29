card <- knime.in

library(data.table)
library(dplyr)


binValue_num <- function(cardx){
  cardx$bins_value <- cardx$binValue
  pattern <- "[<>=() ]"
  cardx$bins_value <- gsub(pattern, "", gsub("is.na", "1", gsub(" & ", "|", cardx$bins_value)))
  cardx <- cardx %>% separate(bins_value, c("min", "max", "is_na"), "([|])", remove = FALSE)
  cardx$max[1] <- cardx$min[1]
  cardx$min[1] <- -99999
  cardx$max[nrow(cardx)] <- 99999
  
  return(cardx)
}


# Helper function to check if a variable is an interaction
is_interaction <- function(var_name) {
  grepl("_x_", as.character(var_name))
}


# Helper function to parse interaction variable name into component variable names
parse_interaction <- function(var_name) {
  # Split by "_x_" to get component variable names
  parts <- strsplit(as.character(var_name), "_x_")[[1]]
  return(parts)
}


# Helper function to extract the separator character from an existing interaction bin in the scorecard
# This ensures we use the exact same character (handles × vs x vs other lookalikes)
extract_interaction_separator <- function(card) {
  # Find an interaction variable in the scorecard
  interaction_vars <- card[grepl("_x_", as.character(var)), unique(var)]
  
  if (length(interaction_vars) > 0) {
    # Get first bin value for this interaction
    sample_bin <- as.character(card[var == interaction_vars[1], bin][1])
    
    # The separator is between the two variable conditions
    # Pattern: "VarName1: condition1 <SEP> VarName2: condition2"
    # Look for " × " or " x " pattern (space, character, space)
    
    # Try to find the separator by looking for pattern after first colon-space and before second var name
    # Extract what's between the conditions
    if (grepl(" × ", sample_bin)) {
      return(" × ")  # Unicode multiplication sign
    } else if (grepl(" x ", sample_bin)) {
      return(" x ")  # Letter x
    } else {
      # Try to extract it dynamically - find the character between ": ... " and the next var name
      # This is a fallback
      return(" × ")
    }
  }
  return(" × ")  # Default
}


# Helper function to create interaction bin value from component bin values
create_interaction_bin <- function(var1_name, var1_bin, var2_name, var2_bin, separator = " × ") {
  # Convert to character
  bin1 <- as.character(var1_bin)
  bin2 <- as.character(var2_bin)
  

  # Handle actual R NA values
  if (is.na(bin1) || bin1 == "NA") {
    bin1 <- "is.na()"
  } else {
    # Remove only LEADING whitespace (keep trailing for format consistency)
    bin1 <- sub("^\\s+", "", bin1)
  }
  
  if (is.na(bin2) || bin2 == "NA") {
    bin2 <- "is.na()"
  } else {
    # Remove only LEADING whitespace
    bin2 <- sub("^\\s+", "", bin2)
  }
  
  # Create the interaction bin string: "VarName1: condition1 × VarName2: condition2"
  interaction_bin <- paste0(as.character(var1_name), ": ", bin1, separator, as.character(var2_name), ": ", bin2)
  
  return(interaction_bin)
}


scorepoint_ply <- function(card, df, debug = FALSE){

  df_columns <- data.frame(colnames = colnames(df))
  df <-  df[,grepl("b_", df_columns$colnames)]
  dt <- setDT(df)
  
  # Convert all b_ columns to character to avoid factor issues
  b_cols <- names(dt)
  for (col in b_cols) {
    if (is.factor(dt[[col]])) {
      dt[[col]] <- as.character(dt[[col]])
    }
  }
  
  data <- data.table()
  
  card <- setDT(card)
  
  # Ensure card columns are character type
  if (is.factor(card$var)) card$var <- as.character(card$var)
  if (is.factor(card$bin)) card$bin <- as.character(card$bin)
  
  # Extract the separator used in the scorecard for interaction variables
  interaction_separator <- extract_interaction_separator(card)
  if (debug) {
    cat("Detected interaction separator:", interaction_separator, "\n")
    cat("Separator bytes:", paste(charToRaw(interaction_separator), collapse=" "), "\n")
  }
  
  xs = card[var != "basepoints", unique(var)]
  xs_len = length(xs)
  
  for (i in 1:xs_len){
    x_i = xs[i]
    point_col_name <- paste0(x_i, "_points")
    
    # Check if this is an interaction variable
    if (is_interaction(x_i)) {
      # Parse the interaction variable to get component variable names
      component_vars <- parse_interaction(x_i)
      
      if (length(component_vars) == 2) {
        var1_name <- component_vars[1]
        var2_name <- component_vars[2]
        
        b_var1 <- paste0("b_", var1_name)
        b_var2 <- paste0("b_", var2_name)
        
        # Check if both component columns exist
        if (b_var1 %in% colnames(dt) && b_var2 %in% colnames(dt)) {
          # Get the scorecard rows for this interaction variable
          cardx = card[var == x_i]
          
          # Create interaction bin values by combining component bin values
          interaction_bins <- mapply(
            create_interaction_bin,
            var1_name, dt[[b_var1]],
            var2_name, dt[[b_var2]],
            MoreArgs = list(separator = interaction_separator),
            SIMPLIFY = TRUE
          )
          
          # Debug: show first few generated bins vs scorecard bins
          if (debug && i == which(sapply(xs, is_interaction))[1]) {
            cat("\n=== DEBUG: First interaction variable:", x_i, "===\n")
            cat("Generated bins (first 5):\n")
            print(head(unique(interaction_bins), 5))
            cat("\nScorecard bins for this variable:\n")
            print(cardx$bin)
            cat("\nSample b_", var1_name, " values: ", paste(head(unique(dt[[b_var1]]), 3), collapse=", "), "\n", sep="")
            cat("Sample b_", var2_name, " values: ", paste(head(unique(dt[[b_var2]]), 3), collapse=", "), "\n", sep="")
          }
          
          # Create a temporary data.table with the interaction bin values
          dtx <- data.table(interaction_bin = interaction_bins)
          
          # Join with the scorecard to get points
          joined <- left_join(x = dtx, y = cardx, by = c("interaction_bin" = "bin"))
          
          # Debug: Check for unmatched bins
          if (debug) {
            unmatched <- sum(is.na(joined$points))
            if (unmatched > 0) {
              cat("WARNING:", x_i, "has", unmatched, "unmatched rows out of", nrow(joined), "\n")
              unmatched_bins <- unique(dtx$interaction_bin[is.na(joined$points)])
              cat("  Unmatched bin examples:", paste(head(unmatched_bins, 3), collapse="; "), "\n")
            }
          }
          
          data <- cbind(data, joined %>% select(one_of("points")))
          names(data)[names(data) == "points"] <- point_col_name
          
        } else {
          # If component columns don't exist, try using pre-computed interaction column
          b_x_i <- paste0("b_", x_i)
          if (b_x_i %in% colnames(dt)) {
            bin <- "bin"
            cardx = card[var == x_i]
            dtx = dt[, b_x_i, with=FALSE]
            data <- cbind(data, left_join(x = dtx, y = cardx, by = setNames(bin, b_x_i)) %>% 
                            select(one_of("points")))
            names(data)[names(data) == "points"] <- point_col_name
          } else {
            # Neither component columns nor pre-computed interaction column exists
            warning(paste("Cannot find columns for interaction variable:", x_i))
            data <- cbind(data, data.table(temp = rep(NA, nrow(dt))))
            names(data)[names(data) == "temp"] <- point_col_name
          }
        }
      }
      
    } else {
      # Non-interaction variable - original logic
      b_x_i <- paste0("b_", x_i)
      
      # Check if column exists
      if (!b_x_i %in% colnames(dt)) {
        if (debug) cat("WARNING: Column", b_x_i, "not found in data\n")
        data <- cbind(data, data.table(temp = rep(NA, nrow(dt))))
        names(data)[names(data) == "temp"] <- point_col_name
        next
      }
      
      cardx = card[var==x_i]
      dtx = dt[, b_x_i, with=FALSE]
      
      # Ensure the column is character type and normalize whitespace
      # The scorecard bins have leading space (e.g., " <= 0") but data may not
      dtx[[b_x_i]] <- as.character(dtx[[b_x_i]])
      
      # Create normalized versions for joining (trim whitespace from both sides)
      dtx$bin_normalized <- trimws(dtx[[b_x_i]])
      cardx$bin_normalized <- trimws(cardx$bin)
      
      joined <- left_join(x = dtx, y = cardx, by = "bin_normalized")
      
      # Debug: Check for unmatched bins
      if (debug) {
        unmatched <- sum(is.na(joined$points))
        if (unmatched > 0) {
          cat("WARNING:", x_i, "has", unmatched, "unmatched rows out of", nrow(joined), "\n")
          unmatched_bins <- unique(dtx$bin_normalized[is.na(joined$points)])
          cat("  Unmatched bin examples:", paste(head(unmatched_bins, 3), collapse="; "), "\n")
          cat("  Scorecard bins (normalized):", paste(unique(cardx$bin_normalized), collapse="; "), "\n")
        }
      }
      
      data <- cbind(data, joined %>% select(one_of("points")))
      names(data)[names(data) == "points"] <- point_col_name
    }
  }
  
  basescore <- card[var == "basepoints", "points"][[1]]
  
  # total score
  data$basescore <- basescore
  data <-  data %>%
    dplyr::mutate(Score = rowSums(dplyr::across(), na.rm = T))
  
  return(data)
}

# scorecard with row data
scorepoint_ply1 <- function(card, df, withOrgData = TRUE, debug = FALSE){
  
  # apply score points
  data <- scorepoint_ply(card, df, debug = debug)
  
  # append orginal dataset
  if(withOrgData){
    data <- cbind(df, data)
  }
  
  return(data)
}

# Set debug = TRUE to see diagnostic output about mismatched bins
DEBUG_MODE <- FALSE

# score points only
df_points <- scorepoint_ply(card, df, debug = DEBUG_MODE)

# score points with original data
df_points_dat <- scorepoint_ply1(card, df, debug = DEBUG_MODE)

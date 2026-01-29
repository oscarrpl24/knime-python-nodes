################################################################################
# R Variable Selection Reference Code
# This is the original R implementation for reference
################################################################################

df <- data.frame(knime.in, check.names = TRUE)
#install.packages("Rserve",,"http://rforge.net")
#install.packages("shiny")
#install.packages("future")
#install.packages("shinydashboard")
#install.packages("plotly")
#install.packages("logiBin")
#install.packages("DT")
#install.packages("shinyjs")
#install.packages("data.table")
#install.packages("tidyr")
#install.packages("shinyalert")
#install.packages("shinycssloaders")

library(shiny)
library(future)
library(shinydashboard)
library(plotly)
library(logiBin)
library(DT)
library(shinyjs)
library(data.table)
library(tidyr)
library(shinyalert)
library(shinycssloaders)

df_variable <- NULL
for (i in 1:length(df)) {
  if (length(unique(df[[i]])) == 2) {
    df_variable[i] <- colnames(df[i])
  } else {
    next
  }
}

df_variable <- na.omit(df_variable)

################################# Predictive Measures ##############################
# Predictive Measures Calculation on Bins generated
####################################################################################

#******************************** 1. Entropy Calculation ***************************
# Entropy Core calculation
entropy <- function(probs) {
  if (any(probs == 0)) {
    probs[probs == 0] <- NA#0.0000000001
  }
  output <- NA
  for (i in 1:length(probs)) {
    output[i] <- probs[i]*log2(probs[i])
  }
  output <- sum(output) * -1
  return(output)
}

# Input Entropy Function
inputEntropy <- function(df) {
  total0 <- df$goods[length(df$goods)]
  total1 <- df$bads[length(df$bads)]
  total <- total0 + total1
  probs <- c(total0 / total, total1 / total)
  output <- entropy(probs = probs)
  return(round(output, 5))
}

# Output Entropy Function
outputEntropy <- function(df) {
  probs0 <- (df$goods / df$count)[1:length(df$goods)-1]
  probs1 <- (df$bads / df$count)[1:length(df$bads)-1]
  probs <- data.frame(probs0, probs1)
  ent <- NA
  for (i in 1:length(probs0)) {
    p <- unlist(probs[i,])
    ent[i] <- entropy(p)
  }
  total <- df$count[length(df$count)]
  prop <- df$count[1:length(df$count)-1] / total
  output <- sum(prop * ent)
  return(round(output,5))
}

#******************************** 2. Gini Index ****************************
# Gini Core Function
gini <- function(totals, overalltotal) {
  output <- 1 - (sum(totals ^ 2) / (overalltotal ^ 2))
  return(output)
}

# Input Gini Function
inputGini <- function(df) {
  totals <- c(df$goods[length(df$goods)], df$bads[length(df$bads)])
  overtotal <- df$count[length(df$count)]
  output <- gini(totals, overtotal)
  return(round(output,5))
}

# Output Gini Function
outputGini <- function(df) {
  totals0 <- df$goods[1:length(df$goods)-1]
  totals1 <- df$bads[1:length(df$bads)-1]
  totals <- df$count[1:length(df$count)-1]
  alltotals <- data.frame(totals0, totals1)
  gi <- NA
  for (i in 1:length(totals0)) {
    gi[i] <- gini(unlist(alltotals[i,]), totals[i])
  }
  output <- sum(gi * (totals / sum(totals)))
  return(round(output,5))
}

#******************************** 3. Pearson Chi-Square ****************************
# Chi-square function
chisquare <- function(observed, expected) {
  if (any(expected == 0)) {
    expected[expected == 0] <- NA#0.0000000001
  }
  output <- sum(((observed - expected)^2)/expected)
  return(output)
}

#******************************** 4. Likelihood Ratio ****************************
# Likelihood Function
likelihoodRatio <- function(observed, expected) {
  if (any(observed == 0)) {
    observed[observed == 0] <- NA#0.0000000001
  }
  if (any(expected == 0)) {
    expected[expected == 0] <- NA#0.0000000001
  }
  mls <- observed * log(observed / expected)
  output <- sum(mls) * 2
  return(output)
}

chi_mls_calc <- function(df, method = 'chisquare') {
  total0 <- df$goods[length(df$goods)]
  total1 <- df$bads[length(df$bads)]
  totalbins <- df$count[1:length(df$count)-1]
  total <- df$count[length(df$count)]
  prop0 <- total0 / total
  prop1 <- total1 / total
  propbins <- totalbins / total
  exp0 <- prop0 * propbins * total
  exp1 <- prop1 * propbins * total
  expected <- c(exp0, exp1)
  obs0 <- df$goods[1:length(df$goods)-1]
  obs1 <- df$bads[1:length(df$bads)-1]
  observed <- c(obs0, obs1)
  if (method == 'chisquare') {
    output <- chisquare(observed = observed, expected = expected)
  } else if (method == 'mls') {
    output <- likelihoodRatio(observed = observed, expected = expected)
  } else {
    output <- 'Please insert valid method: (chisquare, mls)'
  }
  return(round(output,5))
}

#******************************** 5. Odds Ratio ****************************
odds_ratio <- function(df) {
  if (df$goods[2] == 0) {
    df$goods[2] <- NA#0.0000000001
  }
  if (df$bads[2] == 0) {
    df$bads[2] <- NA#0.0000000001
  }
  prop_good <- df$goods[1] / df$goods[2]
  prop_bad <- df$bads[1] / df$bads[2]
  output <- prop_good / prop_bad
  return(round(output,5))
}

#******************************** 6. Filtering Variables ****************************
# KEY SORTING LOGIC:
#   ENTROPY: decreasing = FALSE (LOWER is better)  -> selects bottom N
#   IV: decreasing = TRUE (HIGHER is better) -> selects top N
#   OR: decreasing = TRUE (HIGHER is better) -> selects top N (raw OR, not |log(OR)|)
#   LR: decreasing = TRUE (HIGHER is better) -> selects top N
#   CHI: decreasing = TRUE (HIGHER is better) -> selects top N
#   GINI: decreasing = TRUE (HIGHER is better) -> selects top N

filterVar <- function(df, criteria, numOfVariables, degree) {
  if (criteria == 'Union') {
    if (length(df$"Variable") < numOfVariables) {
      NUMOFVALUES <- length(df$"Variable")
    } else {
      NUMOFVALUES <- numOfVariables
    }
    CUTOFFENTROPY <- sort(na.omit(df$"Entropy"), decreasing = FALSE)[NUMOFVALUES]
    CUTOFFIV <- sort(na.omit(df$"Information Value"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFOR <- sort(na.omit(df$"Odds Ratio"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFLIKELIHOOD <- sort(na.omit(df$"Likelihood Ratio"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFCHI <- sort(na.omit(df$"Chi-Square"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFGINI <- sort(na.omit(df$"Gini"), decreasing = TRUE)[NUMOFVALUES]
    includeVar <- NA
    for (i in 1:length(df$"Variable")) {
      truthVector <- c(df$"Entropy"[i] <= CUTOFFENTROPY, 
                       df$"Information Value"[i] >= CUTOFFIV, 
                       df$"Odds Ratio"[i] >= CUTOFFOR, 
                       df$"Likelihood Ratio"[i] >= CUTOFFLIKELIHOOD, 
                       df$"Chi-Square"[i] >= CUTOFFCHI, 
                       df$"Gini"[i] >= CUTOFFGINI)
      if (any(na.omit(truthVector)) == TRUE) {
        includeVar[i] <- TRUE
      } else {
        includeVar[i] <- FALSE
      }
    }
    df$includeVar <- includeVar
    return(df)
  } else if (criteria == 'Intersection') {
    if (length(df$"Variable") < numOfVariables) {
      NUMOFVALUES <- length(df$"Variable")
    } else {
      NUMOFVALUES <- numOfVariables
    }
    CUTOFFENTROPY <- sort(na.omit(df$"Entropy"), decreasing = FALSE)[NUMOFVALUES]
    CUTOFFIV <- sort(na.omit(df$"Information Value"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFOR <- sort(na.omit(df$"Odds Ratio"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFLIKELIHOOD <- sort(na.omit(df$"Likelihood Ratio"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFCHI <- sort(na.omit(df$"Chi-Square"), decreasing = TRUE)[NUMOFVALUES]
    CUTOFFGINI <- sort(na.omit(df$"Gini"), decreasing = TRUE)[NUMOFVALUES]
    includeVar <- NA
    for (i in 1:length(df$"Variable")) {
      truthVector <- c(df$"Entropy"[i] <= CUTOFFENTROPY, 
                       df$"Information Value"[i] >= CUTOFFIV, 
                       df$"Odds Ratio"[i] >= CUTOFFOR, 
                       df$"Likelihood Ratio"[i] >= CUTOFFLIKELIHOOD, 
                       df$"Chi-Square"[i] >= CUTOFFCHI, 
                       df$"Gini"[i] >= CUTOFFGINI)
      if (sum(na.omit(truthVector)) >= degree) {
        includeVar[i] <- TRUE
      } else {
        includeVar[i] <- FALSE
      }
    }
    df$includeVar <- includeVar
    return(df)
  } else {
    stop("Incorrect criteria. Criteria needs to be either \"Union\" or \"Intersection\".")
  }
}


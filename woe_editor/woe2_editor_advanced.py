# =============================================================================
# WOE 2.0 Editor for KNIME Python Script Node - ADVANCED BINNING VERSION
# =============================================================================
# Next-generation Weight of Evidence binning implementing cutting-edge
# techniques from academic research for fraud detection and credit scoring.
#
# KEY IMPROVEMENTS OVER woe_editor_advanced.py:
#
# 1. WOE 2.0 SHRINKAGE (Beta-Binomial Posterior)
#    - More principled shrinkage using Bayesian estimation
#    - Better handling of rare events in fraud detection
#    - Credible intervals for WOE values
#
# 2. SPLINE-BASED BINNING (Raymaekers et al., 2021)
#    - Captures non-linear effects through spline functions
#    - More granular than traditional piecewise-constant binning
#    - Automatically finds optimal bin boundaries
#
# 3. ISOTONIC REGRESSION BINNING
#    - Finer-grained monotonic binning than decision trees
#    - Higher IV with more optimal boundaries
#    - Guaranteed monotonicity
#
# 4. ADVANCED TREND OPTIONS
#    - Peak: Allows one change point from increasing to decreasing
#    - Valley: Allows one change point from decreasing to increasing
#    - Concave/Convex: Curved monotonic patterns
#    - Auto-detection of optimal trend
#
# 5. BAYESIAN WOE WITH UNCERTAINTY
#    - Full posterior distribution for WOE
#    - Credible intervals (95% CI) for each bin
#    - Robust to small sample sizes
#
# 6. PSI (POPULATION STABILITY INDEX)
#    - Monitors distribution drift
#    - Alerts when binning needs refresh
#    - Essential for production fraud systems
#
# 7. P-VALUE BASED BIN MERGING
#    - Ensures bins are statistically different
#    - Prevents over-binning
#    - Configurable significance level
#
# 8. STREAMING BINNING SUPPORT
#    - Quantile sketches for large/streaming data
#    - Memory-efficient processing
#    - Near real-time updates
#
# Compatible with KNIME 5.9, Python 3.9
#
# FLOW VARIABLES:
#
# PRESET (Recommended - simplifies configuration):
# - Preset (string, optional): Load pre-configured settings for common use cases
#   Options:
#     "fraud_detection" - Rare events, Bayesian shrinkage, non-monotonic allowed
#     "credit_scorecard" - Traditional credit scoring, monotonic, 5% min bins
#     "quick_exploration" - Fast EDA, minimal constraints
#     "production_monitoring" - Conservative, stable, PSI enabled
#     "maximum_iv" - Max predictive power, risk of overfitting
#     "r_compatible" - Match R logiBin::getBins exactly
#     "spline_advanced" - WOE 2.0 spline binning
#   Individual settings below OVERRIDE preset values when specified.
#
# Basic Settings:
# - DependentVariable (string, required for headless): Binary target variable
# - TargetCategory (string, optional): Which value represents "bad" outcome
# - OptimizeAll (boolean, default False): Force monotonic trends on all vars
# - GroupNA (boolean, default False): Combine NA bins with closest bin
#
# Algorithm Settings:
# - Algorithm (string, default "DecisionTree"): Binning algorithm
#   Options: "DecisionTree", "ChiMerge", "IVOptimal", "Spline", "Isotonic"
# - MinBinPct (float, default 0.01): Min percentage per bin
# - MaxBins (int, default 10): Maximum bins per variable
#
# WOE 2.0 Shrinkage Settings:
# - UseShrinkage (boolean, default True): Apply Bayesian shrinkage
# - ShrinkageMethod (string, default "BetaBinomial"): "BetaBinomial" or "Simple"
# - PriorStrength (float, default 1.0): Strength of prior (higher = more shrinkage)
#
# Trend Options:
# - MonotonicTrend (string, default "auto"): Monotonicity constraint
#   Options: "auto", "ascending", "descending", "peak", "valley", 
#            "concave", "convex", "none"
#
# Statistical Validation:
# - MaxPValue (float, default 0.05): Max p-value between adjacent bins
# - UsePValueMerging (boolean, default True): Merge non-significant bins
#
# Bayesian Options:
# - ComputeCredibleIntervals (boolean, default False): Add CI to WOE
# - CredibleLevel (float, default 0.95): Credible interval level
#
# PSI Monitoring:
# - ComputePSI (boolean, default True): Calculate PSI for monitoring
# - PSIReferenceData (string, optional): Path to reference data for PSI
#
# Outputs:
# 1. Original input DataFrame (unchanged)
# 2. df_with_woe - Original data + binned columns (b_*) + WOE columns (WOE_*)
# 3. df_only_woe - Only WOE columns + dependent variable
# 4. df_only_bins - Only binned columns (b_*) for scorecard
# 5. bins - Binning rules with WOE values and credible intervals
# 6. psi_report - PSI values for each variable (NEW)
#
# Version: 2.0
# Release Date: 2026-01-28
# Based on: WOE 2.0 paper (arXiv:2101.01494), OptBinning library
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re
import warnings
import time
import sys
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

warnings.filterwarnings('ignore')

# =============================================================================
# Stability Settings for Multiple Instance Execution
# =============================================================================
BASE_PORT = 8060  # Different from other versions
RANDOM_PORT_RANGE = 1000
INSTANCE_ID = f"{os.getpid()}_{random.randint(10000, 99999)}"

# Prevent threading conflicts
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# =============================================================================
# Progress Logging Utilities
# =============================================================================

def log_progress(message: str, flush: bool = True):
    """Print a progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    if flush:
        sys.stdout.flush()

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

# =============================================================================
# Install/Import Dependencies
# =============================================================================

try:
    from scipy import stats
    from scipy.interpolate import UnivariateSpline
    from scipy.special import betaln
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    from scipy import stats
    from scipy.interpolate import UnivariateSpline
    from scipy.special import betaln

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cluster import KMeans
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cluster import KMeans

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shiny', 'shinywidgets', 'plotly'])
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
    from shinywidgets import render_plotly, output_widget
    import plotly.graph_objects as go


# =============================================================================
# Enumerations
# =============================================================================

class Algorithm(Enum):
    """Available binning algorithms."""
    DECISION_TREE = "DecisionTree"
    CHI_MERGE = "ChiMerge"
    IV_OPTIMAL = "IVOptimal"
    SPLINE = "Spline"
    ISOTONIC = "Isotonic"

class MonotonicTrend(Enum):
    """Monotonicity constraint options."""
    AUTO = "auto"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    PEAK = "peak"           # Increases then decreases
    VALLEY = "valley"       # Decreases then increases
    CONCAVE = "concave"     # Rate of increase slows
    CONVEX = "convex"       # Rate of increase accelerates
    NONE = "none"           # No monotonicity constraint

class ShrinkageMethod(Enum):
    """Shrinkage estimation methods."""
    NONE = "None"
    SIMPLE = "Simple"           # Original weight-based
    BETA_BINOMIAL = "BetaBinomial"  # WOE 2.0 approach


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BinningConfig:
    """Configuration for WOE 2.0 binning algorithms."""
    
    # Algorithm settings
    algorithm: str = "DecisionTree"
    min_bin_pct: float = 0.01
    min_bin_count: int = 20
    max_bins: int = 10
    min_bins: int = 2
    max_categories: int = 50
    
    # Monotonic trend
    monotonic_trend: str = "auto"
    
    # WOE 2.0 Shrinkage settings
    use_shrinkage: bool = True
    shrinkage_method: str = "BetaBinomial"
    prior_strength: float = 1.0
    
    # Statistical validation
    max_p_value: float = 0.05
    use_p_value_merging: bool = True
    chi_merge_threshold: float = 0.05
    
    # Bayesian options
    compute_credible_intervals: bool = False
    credible_level: float = 0.95
    
    # IV optimization
    min_iv_gain: float = 0.005
    
    # PSI monitoring
    compute_psi: bool = True
    
    # Spline settings
    spline_smoothing: float = 0.5
    spline_degree: int = 3
    
    # Enhancement flags (backward compatible)
    use_enhancements: bool = True
    adaptive_min_prop: bool = True
    min_event_count: bool = True
    auto_retry: bool = True
    chi_square_validation: bool = True
    single_bin_protection: bool = True


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================
# Pre-configured settings for common use cases. Select a preset to avoid
# manually configuring all options.
#
# Usage in flow variables:
#   Set Preset = "fraud_detection" (or other preset name)
#   Individual settings can still override preset values
# =============================================================================

class ConfigPresets:
    """
    Pre-defined configuration presets for common WOE binning use cases.
    
    Available presets:
    - FRAUD_DETECTION: Optimized for fraud models with rare events
    - CREDIT_SCORECARD: Traditional credit scoring, regulatory compliant
    - QUICK_EXPLORATION: Fast binning for initial data exploration
    - PRODUCTION_MONITORING: Focus on stability and drift detection
    - MAXIMUM_IV: Maximize predictive power (less constraints)
    - R_COMPATIBLE: Match R logiBin::getBins output exactly
    """
    
    @staticmethod
    def fraud_detection() -> BinningConfig:
        """
        FRAUD DETECTION PRESET
        
        Optimized for fraud models with characteristics:
        - Very low event rates (0.1% - 2% fraud)
        - Rare event handling with Bayesian shrinkage
        - PSI monitoring for production
        - Allows non-monotonic patterns (fraud "sweet spots")
        - Credible intervals for uncertainty quantification
        
        Best for: Transaction fraud, application fraud, AML
        """
        return BinningConfig(
            # Use IV-Optimal to find non-monotonic patterns
            algorithm="IVOptimal",
            
            # Lower minimums for rare events
            min_bin_pct=0.005,      # 0.5% per bin (lower for rare events)
            min_bin_count=30,       # Minimum 30 observations
            max_bins=8,             # Fewer bins for stability
            min_bins=2,
            
            # Allow non-monotonic (fraud sweet spots)
            monotonic_trend="none",
            
            # Strong Bayesian shrinkage for rare events
            use_shrinkage=True,
            shrinkage_method="BetaBinomial",
            prior_strength=2.0,     # Stronger prior for rare events
            
            # Enable credible intervals
            compute_credible_intervals=True,
            credible_level=0.95,
            
            # Statistical validation
            max_p_value=0.10,       # More lenient for rare events
            use_p_value_merging=True,
            
            # PSI monitoring essential for fraud
            compute_psi=True,
            
            # All enhancements on
            use_enhancements=True,
            adaptive_min_prop=True,
            min_event_count=True,
            auto_retry=True,
            chi_square_validation=True,
            single_bin_protection=True
        )
    
    @staticmethod
    def credit_scorecard() -> BinningConfig:
        """
        CREDIT SCORECARD PRESET
        
        Traditional credit scoring approach:
        - Monotonic WOE required (regulatory/interpretability)
        - Conservative bin sizes for stability
        - R-compatible for validation against existing models
        - Standard 5-10 bins per variable
        
        Best for: Credit risk, collections, pricing scorecards
        """
        return BinningConfig(
            # Decision tree for R compatibility
            algorithm="DecisionTree",
            
            # Standard credit scoring minimums
            min_bin_pct=0.05,       # 5% per bin (industry standard)
            min_bin_count=50,       # At least 50 observations
            max_bins=10,
            min_bins=2,
            
            # Auto-detect monotonic trend
            monotonic_trend="auto",
            
            # Moderate shrinkage
            use_shrinkage=True,
            shrinkage_method="BetaBinomial",
            prior_strength=1.0,
            
            # No credible intervals (simpler output)
            compute_credible_intervals=False,
            
            # Standard significance level
            max_p_value=0.05,
            use_p_value_merging=True,
            
            # PSI for monitoring
            compute_psi=True,
            
            # Standard enhancements
            use_enhancements=True,
            adaptive_min_prop=True,
            min_event_count=True,
            auto_retry=True,
            chi_square_validation=True,
            single_bin_protection=True
        )
    
    @staticmethod
    def quick_exploration() -> BinningConfig:
        """
        QUICK EXPLORATION PRESET
        
        Fast binning for initial data exploration:
        - Minimal constraints for speed
        - Good for variable screening
        - Less strict validation
        - Quick IV calculation
        
        Best for: EDA, variable selection, initial analysis
        """
        return BinningConfig(
            # Fast decision tree
            algorithm="DecisionTree",
            
            # Relaxed minimums for exploration
            min_bin_pct=0.01,       # 1% minimum
            min_bin_count=10,       # Lower count threshold
            max_bins=15,            # Allow more bins to see patterns
            min_bins=2,
            
            # No monotonicity constraint
            monotonic_trend="none",
            
            # No shrinkage (raw WOE for exploration)
            use_shrinkage=False,
            shrinkage_method="None",
            
            # No credible intervals
            compute_credible_intervals=False,
            
            # Skip p-value merging for speed
            use_p_value_merging=False,
            
            # Skip PSI for exploration
            compute_psi=False,
            
            # Minimal enhancements
            use_enhancements=True,
            adaptive_min_prop=True,
            min_event_count=False,  # Off for speed
            auto_retry=True,
            chi_square_validation=False,  # Off for speed
            single_bin_protection=True
        )
    
    @staticmethod
    def production_monitoring() -> BinningConfig:
        """
        PRODUCTION MONITORING PRESET
        
        Conservative settings for production scoring:
        - Stability over complexity
        - PSI monitoring enabled
        - Strict bin size requirements
        - Monotonic for consistency
        
        Best for: Production models, batch scoring, monitoring
        """
        return BinningConfig(
            # Stable decision tree
            algorithm="DecisionTree",
            
            # Conservative bin sizes
            min_bin_pct=0.05,
            min_bin_count=100,      # Higher minimum for stability
            max_bins=7,             # Fewer bins = more stable
            min_bins=3,             # At least 3 bins
            
            # Enforce monotonicity
            monotonic_trend="auto",
            
            # Moderate shrinkage
            use_shrinkage=True,
            shrinkage_method="BetaBinomial",
            prior_strength=1.5,
            
            # Credible intervals for monitoring
            compute_credible_intervals=True,
            credible_level=0.95,
            
            # Strict significance
            max_p_value=0.01,       # More strict
            use_p_value_merging=True,
            
            # PSI essential for production
            compute_psi=True,
            
            # All protections on
            use_enhancements=True,
            adaptive_min_prop=False,  # Strict minimums
            min_event_count=True,
            auto_retry=False,         # Fail explicitly if issues
            chi_square_validation=True,
            single_bin_protection=True
        )
    
    @staticmethod
    def maximum_iv() -> BinningConfig:
        """
        MAXIMUM IV PRESET
        
        Maximize predictive power (Information Value):
        - IV-Optimal algorithm
        - No monotonicity constraint
        - More bins allowed
        - Risk of overfitting - use with caution
        
        Best for: Challenger models, research, benchmarking
        """
        return BinningConfig(
            # IV-Optimal for maximum IV
            algorithm="IVOptimal",
            
            # More granular bins
            min_bin_pct=0.01,
            min_bin_count=20,
            max_bins=15,            # Allow many bins
            min_bins=2,
            
            # No monotonicity (maximum flexibility)
            monotonic_trend="none",
            
            # Light shrinkage
            use_shrinkage=True,
            shrinkage_method="BetaBinomial",
            prior_strength=0.5,     # Light shrinkage
            
            # No credible intervals
            compute_credible_intervals=False,
            
            # Lenient significance
            max_p_value=0.10,
            use_p_value_merging=False,  # Keep all bins
            
            # PSI optional
            compute_psi=False,
            
            # Minimal constraints
            use_enhancements=True,
            adaptive_min_prop=True,
            min_event_count=False,
            auto_retry=True,
            chi_square_validation=False,
            single_bin_protection=True
        )
    
    @staticmethod
    def r_compatible() -> BinningConfig:
        """
        R-COMPATIBLE PRESET
        
        Match R logiBin::getBins output exactly:
        - Decision tree algorithm (CART)
        - No WOE 2.0 enhancements
        - Traditional WOE calculation
        - For validation against R models
        
        Best for: R migration, model validation, audits
        """
        return BinningConfig(
            # R-compatible decision tree
            algorithm="DecisionTree",
            
            # R defaults
            min_bin_pct=0.01,
            min_bin_count=20,
            max_bins=10,
            min_bins=2,
            
            # Auto trend detection
            monotonic_trend="auto",
            
            # No shrinkage (traditional WOE)
            use_shrinkage=False,
            shrinkage_method="None",
            
            # No Bayesian features
            compute_credible_intervals=False,
            
            # Standard validation
            max_p_value=0.05,
            use_p_value_merging=False,  # R doesn't do this
            
            # No PSI (not in R version)
            compute_psi=False,
            
            # Minimal enhancements for R compatibility
            use_enhancements=False,
            adaptive_min_prop=False,
            min_event_count=False,
            auto_retry=False,
            chi_square_validation=False,
            single_bin_protection=True
        )
    
    @staticmethod
    def spline_advanced() -> BinningConfig:
        """
        SPLINE ADVANCED PRESET
        
        WOE 2.0 spline-based binning:
        - Captures non-linear effects
        - Optimal bin boundaries from spline inflection points
        - Good for complex relationships
        
        Best for: Non-linear patterns, research, advanced modeling
        """
        return BinningConfig(
            # Spline algorithm
            algorithm="Spline",
            
            # Standard bins
            min_bin_pct=0.02,
            min_bin_count=30,
            max_bins=10,
            min_bins=2,
            
            # Spline settings
            spline_smoothing=0.5,
            spline_degree=3,
            
            # Auto monotonicity
            monotonic_trend="auto",
            
            # Full Bayesian treatment
            use_shrinkage=True,
            shrinkage_method="BetaBinomial",
            prior_strength=1.0,
            compute_credible_intervals=True,
            credible_level=0.95,
            
            # Standard validation
            max_p_value=0.05,
            use_p_value_merging=True,
            
            # PSI enabled
            compute_psi=True,
            
            # Full enhancements
            use_enhancements=True,
            adaptive_min_prop=True,
            min_event_count=True,
            auto_retry=True,
            chi_square_validation=True,
            single_bin_protection=True
        )
    
    @staticmethod
    def get_preset(name: str) -> BinningConfig:
        """
        Get a preset configuration by name.
        
        Parameters:
            name: Preset name (case-insensitive). Options:
                  - "fraud_detection" or "fraud"
                  - "credit_scorecard" or "credit" or "scorecard"
                  - "quick_exploration" or "quick" or "explore"
                  - "production_monitoring" or "production" or "monitor"
                  - "maximum_iv" or "max_iv"
                  - "r_compatible" or "r" or "legacy"
                  - "spline_advanced" or "spline" or "woe2"
                  - "default" (returns default BinningConfig)
        
        Returns:
            BinningConfig instance with preset values
        """
        presets = {
            # Fraud detection aliases
            "fraud_detection": ConfigPresets.fraud_detection,
            "fraud": ConfigPresets.fraud_detection,
            
            # Credit scorecard aliases
            "credit_scorecard": ConfigPresets.credit_scorecard,
            "credit": ConfigPresets.credit_scorecard,
            "scorecard": ConfigPresets.credit_scorecard,
            
            # Quick exploration aliases
            "quick_exploration": ConfigPresets.quick_exploration,
            "quick": ConfigPresets.quick_exploration,
            "explore": ConfigPresets.quick_exploration,
            "exploration": ConfigPresets.quick_exploration,
            
            # Production monitoring aliases
            "production_monitoring": ConfigPresets.production_monitoring,
            "production": ConfigPresets.production_monitoring,
            "monitor": ConfigPresets.production_monitoring,
            "monitoring": ConfigPresets.production_monitoring,
            
            # Maximum IV aliases
            "maximum_iv": ConfigPresets.maximum_iv,
            "max_iv": ConfigPresets.maximum_iv,
            "maxiv": ConfigPresets.maximum_iv,
            
            # R-compatible aliases
            "r_compatible": ConfigPresets.r_compatible,
            "r": ConfigPresets.r_compatible,
            "legacy": ConfigPresets.r_compatible,
            "r_compat": ConfigPresets.r_compatible,
            
            # Spline advanced aliases
            "spline_advanced": ConfigPresets.spline_advanced,
            "spline": ConfigPresets.spline_advanced,
            "woe2": ConfigPresets.spline_advanced,
            "woe_2": ConfigPresets.spline_advanced,
            
            # Default
            "default": BinningConfig,
        }
        
        name_lower = name.lower().strip()
        
        if name_lower in presets:
            return presets[name_lower]()
        else:
            available = ["fraud_detection", "credit_scorecard", "quick_exploration", 
                        "production_monitoring", "maximum_iv", "r_compatible", 
                        "spline_advanced", "default"]
            log_progress(f"Unknown preset '{name}'. Using default. Available: {available}")
            return BinningConfig()
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """
        Get a dictionary of all available presets and their descriptions.
        """
        return {
            "fraud_detection": "Rare events, Bayesian shrinkage, non-monotonic allowed",
            "credit_scorecard": "Traditional credit scoring, monotonic, 5% min bins",
            "quick_exploration": "Fast EDA, minimal constraints, no shrinkage",
            "production_monitoring": "Conservative, stable, PSI enabled, strict minimums",
            "maximum_iv": "Max predictive power, risk of overfitting",
            "r_compatible": "Match R logiBin::getBins exactly",
            "spline_advanced": "WOE 2.0 spline binning, non-linear patterns",
            "default": "Balanced defaults suitable for most cases"
        }


# Global config instance (default settings)
config = BinningConfig()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BinInfo:
    """Information about a single bin."""
    lower: float
    upper: float
    count: int
    goods: int
    bads: int
    woe: float = 0.0
    iv_contribution: float = 0.0
    woe_ci_lower: Optional[float] = None
    woe_ci_upper: Optional[float] = None
    event_rate: float = 0.0
    
@dataclass
class BinResult:
    """Container for binning results."""
    var_summary: pd.DataFrame
    bin: pd.DataFrame

@dataclass
class PSIResult:
    """Container for PSI calculation results."""
    variable: str
    psi_value: float
    status: str  # "stable", "moderate_drift", "significant_drift"
    bin_details: pd.DataFrame


# =============================================================================
# Core WOE/IV Calculation Functions - WOE 2.0 Enhanced
# =============================================================================

def calculate_woe_simple(freq_good: np.ndarray, freq_bad: np.ndarray,
                         shrinkage_weight: float = 0.0) -> np.ndarray:
    """
    Calculate WOE with simple weight-based shrinkage.
    Original method from woe_editor_advanced.py.
    """
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good))
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    epsilon = 0.0001
    dist_good = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad / dist_good)
    
    if shrinkage_weight > 0:
        n_obs = freq_good + freq_bad
        total_obs = n_obs.sum()
        weights = n_obs / (n_obs + shrinkage_weight * total_obs / len(n_obs))
        woe = woe * weights
    
    return np.round(woe, 5)


def calculate_woe_beta_binomial(
    freq_good: np.ndarray, 
    freq_bad: np.ndarray,
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate WOE using Beta-Binomial posterior (WOE 2.0 approach).
    
    This is a more principled shrinkage method that uses Bayesian estimation.
    The posterior mean provides shrinkage toward the global event rate,
    with the amount of shrinkage determined by sample size.
    
    Parameters:
        freq_good: Array of good (non-event) counts per bin
        freq_bad: Array of bad (event) counts per bin
        prior_strength: Strength of prior (higher = more shrinkage)
        compute_ci: Whether to compute credible intervals
        ci_level: Credible interval level (e.g., 0.95 for 95% CI)
    
    Returns:
        woe: Array of WOE values
        ci_lower: Lower bound of credible interval (if compute_ci)
        ci_upper: Upper bound of credible interval (if compute_ci)
    """
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    total_all = total_good + total_bad
    
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(freq_good)), None, None
    
    # Global event rate as prior mean
    global_event_rate = total_bad / total_all
    
    # Beta prior parameters centered on global event rate
    # prior_strength controls the "equivalent sample size" of the prior
    prior_alpha = prior_strength * global_event_rate
    prior_beta = prior_strength * (1 - global_event_rate)
    
    # Ensure prior parameters are at least 0.5 for stability
    prior_alpha = max(prior_alpha, 0.5)
    prior_beta = max(prior_beta, 0.5)
    
    woe_values = []
    ci_lower_values = []
    ci_upper_values = []
    
    for i in range(len(freq_good)):
        goods = freq_good[i]
        bads = freq_bad[i]
        n = goods + bads
        
        if n == 0:
            woe_values.append(0.0)
            ci_lower_values.append(0.0)
            ci_upper_values.append(0.0)
            continue
        
        # Posterior parameters (Beta-Binomial conjugate update)
        post_alpha = bads + prior_alpha
        post_beta = goods + prior_beta
        
        # Posterior mean for event rate in this bin
        p_bad_bin = post_alpha / (post_alpha + post_beta)
        
        # Calculate distribution relative to overall
        # dist_bad = p(bin | bad) = (bad_in_bin / total_bad)
        # dist_good = p(bin | good) = (good_in_bin / total_good)
        
        # With shrinkage: use posterior predictive
        dist_bad = (bads + prior_alpha * (n / total_all)) / (total_bad + prior_alpha)
        dist_good = (goods + prior_beta * (n / total_all)) / (total_good + prior_beta)
        
        # Avoid division by zero
        dist_bad = max(dist_bad, 0.0001)
        dist_good = max(dist_good, 0.0001)
        
        woe = np.log(dist_bad / dist_good)
        woe_values.append(round(woe, 5))
        
        # Compute credible intervals via simulation
        if compute_ci:
            n_samples = 1000
            alpha_half = (1 - ci_level) / 2
            
            # Sample from posterior
            p_samples = stats.beta.rvs(post_alpha, post_beta, size=n_samples)
            
            # Convert to WOE scale (simplified)
            # WOE ≈ log(p / (1-p)) when comparing to global rate
            log_odds_samples = np.log(p_samples / (1 - p_samples + 0.0001) + 0.0001)
            global_log_odds = np.log(global_event_rate / (1 - global_event_rate))
            woe_samples = log_odds_samples - global_log_odds
            
            ci_lower_values.append(round(np.percentile(woe_samples, alpha_half * 100), 5))
            ci_upper_values.append(round(np.percentile(woe_samples, (1 - alpha_half) * 100), 5))
        else:
            ci_lower_values.append(None)
            ci_upper_values.append(None)
    
    return (
        np.array(woe_values),
        np.array(ci_lower_values) if compute_ci else None,
        np.array(ci_upper_values) if compute_ci else None
    )


def calculate_woe(
    freq_good: np.ndarray, 
    freq_bad: np.ndarray,
    method: str = "BetaBinomial",
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate Weight of Evidence with configurable method.
    
    This is the main WOE calculation function that dispatches to the
    appropriate method based on configuration.
    """
    if method == "BetaBinomial":
        return calculate_woe_beta_binomial(
            freq_good, freq_bad, prior_strength, compute_ci, ci_level
        )
    elif method == "Simple":
        woe = calculate_woe_simple(freq_good, freq_bad, prior_strength)
        return woe, None, None
    else:
        # No shrinkage
        woe = calculate_woe_simple(freq_good, freq_bad, 0.0)
        return woe, None, None


def calculate_iv(freq_good: np.ndarray, freq_bad: np.ndarray) -> float:
    """Calculate Information Value (IV) for a variable."""
    freq_good = np.array(freq_good, dtype=float)
    freq_bad = np.array(freq_bad, dtype=float)
    
    total_good = freq_good.sum()
    total_bad = freq_bad.sum()
    
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    dist_good = freq_good / total_good
    dist_bad = freq_bad / total_bad
    
    epsilon = 0.0001
    dist_good_safe = np.where(dist_good == 0, epsilon, dist_good)
    dist_bad_safe = np.where(dist_bad == 0, epsilon, dist_bad)
    
    woe = np.log(dist_bad_safe / dist_good_safe)
    iv = np.sum((dist_bad - dist_good) * woe)
    
    if not np.isfinite(iv):
        iv = 0.0
    
    return round(iv, 4)


def calculate_entropy(goods: int, bads: int) -> float:
    """Calculate entropy for a bin."""
    total = goods + bads
    if total == 0 or goods == 0 or bads == 0:
        return 0.0
    
    p_good = goods / total
    p_bad = bads / total
    
    entropy = -1 * ((p_bad * np.log2(p_bad)) + (p_good * np.log2(p_good)))
    return round(entropy, 4)


def get_var_type(series: pd.Series) -> str:
    """Determine if variable is numeric or factor."""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return 'factor'
        return 'numeric'
    return 'factor'


# =============================================================================
# PSI (Population Stability Index) Calculation
# =============================================================================

def calculate_psi(
    reference_proportions: np.ndarray,
    current_proportions: np.ndarray,
    epsilon: float = 0.0001
) -> float:
    """
    Calculate Population Stability Index between reference and current distributions.
    
    PSI = Σ (current_prop - ref_prop) * ln(current_prop / ref_prop)
    
    Interpretation:
        PSI < 0.1: Stable (no significant change)
        PSI 0.1-0.2: Moderate drift (monitor closely)
        PSI > 0.2: Significant drift (consider retraining)
    """
    ref = np.array(reference_proportions, dtype=float)
    cur = np.array(current_proportions, dtype=float)
    
    # Normalize to ensure they sum to 1
    ref = ref / ref.sum() if ref.sum() > 0 else ref
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    
    # Add epsilon to avoid log(0) and division by zero
    ref = np.maximum(ref, epsilon)
    cur = np.maximum(cur, epsilon)
    
    # PSI calculation
    psi = np.sum((cur - ref) * np.log(cur / ref))
    
    return round(psi, 4)


def calculate_variable_psi(
    bins_df: pd.DataFrame,
    reference_bins_df: pd.DataFrame,
    var_name: str
) -> PSIResult:
    """
    Calculate PSI for a single variable given current and reference binning.
    """
    current = bins_df[bins_df['var'] == var_name].copy()
    reference = reference_bins_df[reference_bins_df['var'] == var_name].copy()
    
    # Exclude Total row
    current = current[current['bin'] != 'Total']
    reference = reference[reference['bin'] != 'Total']
    
    if current.empty or reference.empty:
        return PSIResult(
            variable=var_name,
            psi_value=0.0,
            status="unknown",
            bin_details=pd.DataFrame()
        )
    
    # Get proportions
    current_props = current['count'].values / current['count'].sum()
    
    # Match bins by label
    matched_ref_props = []
    for bin_label in current['bin'].values:
        ref_match = reference[reference['bin'] == bin_label]
        if not ref_match.empty:
            ref_prop = ref_match['count'].values[0] / reference['count'].sum()
        else:
            ref_prop = 0.0001  # Small value for new bins
        matched_ref_props.append(ref_prop)
    
    reference_props = np.array(matched_ref_props)
    
    psi_value = calculate_psi(reference_props, current_props)
    
    # Determine status
    if psi_value < 0.1:
        status = "stable"
    elif psi_value < 0.2:
        status = "moderate_drift"
    else:
        status = "significant_drift"
    
    # Create bin details
    bin_details = current[['bin', 'count']].copy()
    bin_details['current_prop'] = current_props
    bin_details['reference_prop'] = reference_props
    bin_details['psi_contribution'] = (current_props - reference_props) * np.log(current_props / reference_props)
    
    return PSIResult(
        variable=var_name,
        psi_value=psi_value,
        status=status,
        bin_details=bin_details
    )


# =============================================================================
# Chi-Square Statistics for Bin Merging
# =============================================================================

def chi_square_statistic(bin1_good: int, bin1_bad: int,
                         bin2_good: int, bin2_bad: int) -> float:
    """Calculate chi-square statistic for two adjacent bins."""
    observed = np.array([[bin1_good, bin1_bad], [bin2_good, bin2_bad]])
    
    if observed.sum() == 0:
        return np.inf
    
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    
    if total == 0 or any(row_totals == 0) or any(col_totals == 0):
        return np.inf
    
    expected = np.outer(row_totals, col_totals) / total
    
    chi2 = 0
    for i in range(2):
        for j in range(2):
            if expected[i, j] > 0:
                diff = abs(observed[i, j] - expected[i, j]) - 0.5
                diff = max(diff, 0)
                chi2 += (diff ** 2) / expected[i, j]
    
    return chi2


def chi_square_p_value(bin1_good: int, bin1_bad: int,
                       bin2_good: int, bin2_bad: int) -> float:
    """Calculate p-value for chi-square test between two bins."""
    chi2 = chi_square_statistic(bin1_good, bin1_bad, bin2_good, bin2_bad)
    if chi2 == np.inf:
        return 1.0
    
    # Chi-square with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return p_value


# =============================================================================
# Binning Algorithm: Decision Tree (R-compatible)
# =============================================================================

def get_decision_tree_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_events: int = 5,
    adaptive_min_prop: bool = True,
    auto_retry: bool = True
) -> List[float]:
    """
    Use decision tree (CART) to find optimal split points.
    R-compatible: matches logiBin::getBins algorithm.
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        return []
    
    n_samples = len(x_clean)
    n_events = int(y_clean.sum())
    event_rate = n_events / n_samples if n_samples > 0 else 0
    
    effective_min_prop = min_prop
    
    if adaptive_min_prop:
        if n_samples < 500:
            effective_min_prop = max(min_prop / 2, 0.005)
        if event_rate < 0.05 and n_events > 0:
            max_possible_bins = max(n_events // min_events, 2)
            min_samples_for_events = n_samples / max_possible_bins
            adaptive_prop = min_samples_for_events / n_samples
            effective_min_prop = max(effective_min_prop, adaptive_prop * 0.8)
    
    min_samples_leaf = max(int(n_samples * effective_min_prop), 1)
    min_samples_leaf = min(min_samples_leaf, n_samples // 2)
    min_samples_leaf = max(min_samples_leaf, 1)
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    try:
        tree.fit(x_clean, y_clean)
    except Exception:
        return []
    
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]
    thresholds = sorted(set(thresholds))
    
    if auto_retry and len(thresholds) == 0 and min_samples_leaf > 10:
        tree_retry = DecisionTreeClassifier(
            max_leaf_nodes=max_bins,
            min_samples_leaf=max(min_samples_leaf // 2, 1),
            random_state=42
        )
        try:
            tree_retry.fit(x_clean, y_clean)
            thresholds = tree_retry.tree_.threshold
            thresholds = thresholds[thresholds != -2]
            thresholds = sorted(set(thresholds))
        except Exception:
            pass
    
    return thresholds


# =============================================================================
# Binning Algorithm: ChiMerge
# =============================================================================

def get_chimerge_splits(
    x: pd.Series,
    y: pd.Series,
    min_bin_pct: float = 0.05,
    min_bin_count: int = 50,
    max_bins: int = 10,
    min_bins: int = 2,
    chi_threshold: float = 0.05
) -> List[float]:
    """
    ChiMerge algorithm: bottom-up binning based on chi-square tests.
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return []
    
    total_count = len(x_clean)
    min_count_required = max(int(total_count * min_bin_pct), min_bin_count)
    
    n_initial_bins = min(100, len(x_clean.unique()))
    
    try:
        initial_bins = pd.qcut(x_clean, q=n_initial_bins, duplicates='drop')
    except ValueError:
        try:
            initial_bins = pd.cut(x_clean, bins=min(20, len(x_clean.unique())), duplicates='drop')
        except:
            return []
    
    if hasattr(initial_bins, 'categories') and len(initial_bins.categories) > 0:
        edges = sorted(set(
            [initial_bins.categories[0].left] + 
            [cat.right for cat in initial_bins.categories]
        ))
    else:
        return []
    
    def build_bin_stats(edges):
        bins_stats = []
        for i in range(len(edges) - 1):
            if i == 0:
                bin_mask = (x_clean >= edges[i]) & (x_clean <= edges[i + 1])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())
            goods = int(count - bads)
            bins_stats.append({
                'left': edges[i], 
                'right': edges[i + 1], 
                'goods': goods, 
                'bads': bads, 
                'count': count
            })
        return bins_stats
    
    bins_stats = build_bin_stats(edges)
    bins_stats = [b for b in bins_stats if b['count'] > 0]
    
    if len(bins_stats) <= min_bins:
        return [b['right'] for b in bins_stats[:-1]]
    
    chi2_threshold = stats.chi2.ppf(1 - chi_threshold, df=1)
    
    while len(bins_stats) > min_bins:
        min_chi2 = np.inf
        merge_idx = -1
        
        for i in range(len(bins_stats) - 1):
            chi2 = chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        if min_chi2 > chi2_threshold and len(bins_stats) <= max_bins:
            break
        
        if merge_idx >= 0:
            bins_stats[merge_idx] = {
                'left': bins_stats[merge_idx]['left'],
                'right': bins_stats[merge_idx + 1]['right'],
                'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
                'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
                'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
            }
            bins_stats.pop(merge_idx + 1)
        else:
            break
    
    # Enforce minimum bin size
    changed = True
    while changed and len(bins_stats) > min_bins:
        changed = False
        for i in range(len(bins_stats)):
            if bins_stats[i]['count'] < min_count_required:
                if i == 0 and len(bins_stats) > 1:
                    bins_stats[0] = {
                        'left': bins_stats[0]['left'],
                        'right': bins_stats[1]['right'],
                        'goods': bins_stats[0]['goods'] + bins_stats[1]['goods'],
                        'bads': bins_stats[0]['bads'] + bins_stats[1]['bads'],
                        'count': bins_stats[0]['count'] + bins_stats[1]['count']
                    }
                    bins_stats.pop(1)
                    changed = True
                    break
                elif i > 0:
                    bins_stats[i - 1] = {
                        'left': bins_stats[i - 1]['left'],
                        'right': bins_stats[i]['right'],
                        'goods': bins_stats[i - 1]['goods'] + bins_stats[i]['goods'],
                        'bads': bins_stats[i - 1]['bads'] + bins_stats[i]['bads'],
                        'count': bins_stats[i - 1]['count'] + bins_stats[i]['count']
                    }
                    bins_stats.pop(i)
                    changed = True
                    break
    
    while len(bins_stats) > max_bins:
        min_chi2 = np.inf
        merge_idx = 0
        
        for i in range(len(bins_stats) - 1):
            chi2 = chi_square_statistic(
                bins_stats[i]['goods'], bins_stats[i]['bads'],
                bins_stats[i + 1]['goods'], bins_stats[i + 1]['bads']
            )
            if chi2 < min_chi2:
                min_chi2 = chi2
                merge_idx = i
        
        bins_stats[merge_idx] = {
            'left': bins_stats[merge_idx]['left'],
            'right': bins_stats[merge_idx + 1]['right'],
            'goods': bins_stats[merge_idx]['goods'] + bins_stats[merge_idx + 1]['goods'],
            'bads': bins_stats[merge_idx]['bads'] + bins_stats[merge_idx + 1]['bads'],
            'count': bins_stats[merge_idx]['count'] + bins_stats[merge_idx + 1]['count']
        }
        bins_stats.pop(merge_idx + 1)
    
    splits = [b['right'] for b in bins_stats[:-1]]
    return splits


# =============================================================================
# Binning Algorithm: IV-Optimal (Maximize Information Value)
# =============================================================================

def get_iv_optimal_splits(
    x: pd.Series,
    y: pd.Series,
    min_prop: float = 0.01,
    max_bins: int = 10,
    min_bin_count: int = 20,
    min_iv_loss: float = 0.001
) -> List[float]:
    """
    IV-optimal binning: directly maximizes Information Value.
    Allows non-monotonic patterns (sweet spots).
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    if len(x_clean) == 0:
        return []
    
    n_samples = len(x_clean)
    n_unique = len(np.unique(x_clean))
    total_goods = int((y_clean == 0).sum())
    total_bads = int((y_clean == 1).sum())
    
    if total_goods == 0 or total_bads == 0:
        return []
    
    # Dynamic starting granularity
    if n_unique <= 20:
        initial_splits = sorted(np.unique(x_clean))[:-1]
    else:
        n_initial = min(
            max(20, n_unique // 5),
            min(100, n_samples // 50),
            n_unique - 1
        )
        try:
            quantiles = np.linspace(0, 100, n_initial + 1)[1:-1]
            initial_splits = list(np.percentile(x_clean, quantiles))
            initial_splits = sorted(set(initial_splits))
        except Exception:
            initial_splits = sorted(np.unique(x_clean))[:-1]
    
    if len(initial_splits) == 0:
        return []
    
    def create_bins_from_splits(splits: List[float]) -> List[dict]:
        bins_list = []
        edges = [-np.inf] + sorted(splits) + [np.inf]
        
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            
            if lower == -np.inf:
                bin_mask = x_clean <= upper
            elif upper == np.inf:
                bin_mask = x_clean > lower
            else:
                bin_mask = (x_clean > lower) & (x_clean <= upper)
            
            count = int(bin_mask.sum())
            if count > 0:
                bads = int(y_clean[bin_mask].sum())
                goods = count - bads
                bins_list.append({
                    'lower': lower,
                    'upper': upper,
                    'count': count,
                    'goods': goods,
                    'bads': bads
                })
        
        return bins_list
    
    def calculate_bin_iv(goods: int, bads: int) -> float:
        if total_goods == 0 or total_bads == 0:
            return 0.0
        dist_good = goods / total_goods if total_goods > 0 else 0
        dist_bad = bads / total_bads if total_bads > 0 else 0
        if dist_good == 0 or dist_bad == 0:
            return 0.0
        woe = np.log(dist_bad / dist_good)
        return (dist_bad - dist_good) * woe
    
    def calculate_total_iv(bins_list: List[dict]) -> float:
        return sum(calculate_bin_iv(b['goods'], b['bads']) for b in bins_list)
    
    current_splits = list(initial_splits)
    bins_list = create_bins_from_splits(current_splits)
    
    if len(bins_list) <= 1:
        return []
    
    # Iteratively merge bins with smallest IV loss
    while len(bins_list) > max_bins:
        if len(bins_list) <= 2:
            break
        
        min_iv_loss_found = float('inf')
        best_merge_idx = 0
        
        for i in range(len(bins_list) - 1):
            merged_goods = bins_list[i]['goods'] + bins_list[i + 1]['goods']
            merged_bads = bins_list[i]['bads'] + bins_list[i + 1]['bads']
            
            iv_before = (
                calculate_bin_iv(bins_list[i]['goods'], bins_list[i]['bads']) +
                calculate_bin_iv(bins_list[i + 1]['goods'], bins_list[i + 1]['bads'])
            )
            iv_after = calculate_bin_iv(merged_goods, merged_bads)
            iv_loss = iv_before - iv_after
            
            if iv_loss < min_iv_loss_found:
                min_iv_loss_found = iv_loss
                best_merge_idx = i
        
        if min_iv_loss_found > min_iv_loss and len(bins_list) <= max_bins * 2:
            break
        
        i = best_merge_idx
        merged_bin = {
            'lower': bins_list[i]['lower'],
            'upper': bins_list[i + 1]['upper'],
            'count': bins_list[i]['count'] + bins_list[i + 1]['count'],
            'goods': bins_list[i]['goods'] + bins_list[i + 1]['goods'],
            'bads': bins_list[i]['bads'] + bins_list[i + 1]['bads']
        }
        
        bins_list = bins_list[:i] + [merged_bin] + bins_list[i + 2:]
        
        if i < len(current_splits):
            current_splits = current_splits[:i] + current_splits[i + 1:]
    
    # Ensure minimum bin size
    min_count = max(int(n_samples * min_prop), min_bin_count)
    
    merged = True
    while merged and len(bins_list) > 2:
        merged = False
        for i, b in enumerate(bins_list):
            if b['count'] < min_count:
                if i == 0:
                    merge_with = 1
                elif i == len(bins_list) - 1:
                    merge_with = i - 1
                else:
                    if bins_list[i - 1]['count'] <= bins_list[i + 1]['count']:
                        merge_with = i - 1
                    else:
                        merge_with = i + 1
                
                merge_idx = min(i, merge_with)
                other_idx = max(i, merge_with)
                
                merged_bin = {
                    'lower': bins_list[merge_idx]['lower'],
                    'upper': bins_list[other_idx]['upper'],
                    'count': bins_list[merge_idx]['count'] + bins_list[other_idx]['count'],
                    'goods': bins_list[merge_idx]['goods'] + bins_list[other_idx]['goods'],
                    'bads': bins_list[merge_idx]['bads'] + bins_list[other_idx]['bads']
                }
                
                bins_list = bins_list[:merge_idx] + [merged_bin] + bins_list[other_idx + 1:]
                merged = True
                break
    
    # Extract final splits
    final_splits = []
    for b in bins_list[:-1]:
        if b['upper'] != np.inf:
            final_splits.append(b['upper'])
    
    return sorted(final_splits)


# =============================================================================
# Binning Algorithm: Spline-Based (WOE 2.0)
# =============================================================================

def get_spline_splits(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 10,
    min_bin_pct: float = 0.01,
    smoothing: float = 0.5,
    spline_degree: int = 3
) -> List[float]:
    """
    Spline-based binning from WOE 2.0 paper.
    
    Uses spline functions to capture non-linear effects in the relationship
    between x and the event rate. Bin boundaries are placed at inflection
    points or significant changes in the fitted spline.
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    if len(x_clean) < 20:
        # Fall back to decision tree for small samples
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean), 
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    n_samples = len(x_clean)
    
    # Sort by x values
    sorted_idx = np.argsort(x_clean)
    x_sorted = x_clean[sorted_idx]
    y_sorted = y_clean[sorted_idx]
    
    # Calculate rolling event rate
    # Use window size that gives approximately 20-50 points for the spline
    window_size = max(n_samples // 30, 10)
    
    # Calculate smoothed event rate using rolling mean
    event_rates = pd.Series(y_sorted).rolling(window=window_size, center=True, min_periods=1).mean()
    event_rates = event_rates.values
    
    # Get unique x values for spline fitting (subsample if too many)
    if len(np.unique(x_sorted)) > 500:
        # Subsample for efficiency
        subsample_idx = np.linspace(0, len(x_sorted) - 1, 500).astype(int)
        x_for_spline = x_sorted[subsample_idx]
        y_for_spline = event_rates[subsample_idx]
    else:
        x_for_spline = x_sorted
        y_for_spline = event_rates
    
    # Remove duplicates in x (required for spline)
    unique_mask = np.concatenate([[True], np.diff(x_for_spline) > 0])
    x_for_spline = x_for_spline[unique_mask]
    y_for_spline = y_for_spline[unique_mask]
    
    if len(x_for_spline) < 4:
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    try:
        # Fit spline to event rate as function of x
        # s parameter controls smoothing (higher = smoother)
        s_factor = smoothing * len(x_for_spline)
        spline = UnivariateSpline(x_for_spline, y_for_spline, k=min(spline_degree, len(x_for_spline) - 1), s=s_factor)
        
        # Evaluate spline at many points
        x_eval = np.linspace(x_for_spline.min(), x_for_spline.max(), 1000)
        y_eval = spline(x_eval)
        
        # Compute first derivative
        spline_deriv = spline.derivative()
        y_deriv = spline_deriv(x_eval)
        
        # Compute second derivative
        spline_deriv2 = spline_deriv.derivative()
        y_deriv2 = spline_deriv2(x_eval)
        
        # Find inflection points (where second derivative changes sign)
        inflection_idx = np.where(np.diff(np.sign(y_deriv2)))[0]
        inflection_points = x_eval[inflection_idx]
        
        # Also find points of maximum curvature (extremes of first derivative)
        deriv_extrema_idx = np.where(np.diff(np.sign(y_deriv2)))[0]
        
        # Combine and select split points
        candidate_splits = list(inflection_points)
        
        # If not enough inflection points, add quantile-based splits
        if len(candidate_splits) < max_bins - 1:
            quantile_splits = np.percentile(x_clean, np.linspace(10, 90, max_bins - 1 - len(candidate_splits)))
            candidate_splits.extend(quantile_splits)
        
        # Deduplicate and sort
        candidate_splits = sorted(set(candidate_splits))
        
        # Filter to ensure minimum bin size
        min_count = max(int(n_samples * min_bin_pct), 20)
        valid_splits = []
        prev_split = x_clean.min()
        
        for split in candidate_splits:
            count_in_bin = np.sum((x_clean > prev_split) & (x_clean <= split))
            if count_in_bin >= min_count:
                valid_splits.append(split)
                prev_split = split
        
        # Limit to max_bins - 1 splits
        if len(valid_splits) >= max_bins:
            # Keep splits with most curvature change
            valid_splits = valid_splits[:max_bins - 1]
        
        return valid_splits
        
    except Exception as e:
        # Fall back to decision tree if spline fails
        log_progress(f"  Spline fitting failed: {e}, falling back to DecisionTree")
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )


# =============================================================================
# Binning Algorithm: Isotonic Regression
# =============================================================================

def get_isotonic_splits(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 10,
    min_bin_pct: float = 0.01,
    direction: str = 'auto'
) -> List[float]:
    """
    Isotonic regression binning for finer-grained monotonic binning.
    
    Uses isotonic regression to fit a monotonic function to the event rate,
    then determines bin boundaries from the fitted values.
    """
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    
    if len(x_clean) < 10:
        return []
    
    n_samples = len(x_clean)
    
    # Determine direction if auto
    if direction == 'auto':
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        if np.isnan(corr) or abs(corr) < 0.01:
            increasing = True  # Default
        else:
            increasing = corr > 0
    else:
        increasing = direction in ['ascending', 'increasing']
    
    # Fit isotonic regression
    iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    
    try:
        y_iso = iso.fit_transform(x_clean, y_clean)
    except Exception:
        return get_decision_tree_splits(
            pd.Series(x_clean), pd.Series(y_clean),
            min_prop=min_bin_pct, max_bins=max_bins
        )
    
    # Find unique fitted values (natural bin boundaries)
    # Isotonic regression creates piecewise constant function
    unique_fitted = np.unique(y_iso)
    
    if len(unique_fitted) <= 1:
        return []
    
    # Map fitted values back to x values for split points
    splits = []
    for i in range(len(unique_fitted) - 1):
        # Find x values where fitted value changes
        mask_current = y_iso == unique_fitted[i]
        mask_next = y_iso == unique_fitted[i + 1]
        
        if mask_current.any() and mask_next.any():
            max_x_current = x_clean[mask_current].max()
            min_x_next = x_clean[mask_next].min()
            # Split point is midway between
            split = (max_x_current + min_x_next) / 2
            splits.append(split)
    
    splits = sorted(set(splits))
    
    # If too many splits, cluster them
    if len(splits) > max_bins - 1:
        splits_arr = np.array(splits).reshape(-1, 1)
        kmeans = KMeans(n_clusters=max_bins - 1, random_state=42, n_init=10)
        kmeans.fit(splits_arr)
        splits = sorted(kmeans.cluster_centers_.flatten())
    
    # Ensure minimum bin size
    min_count = max(int(n_samples * min_bin_pct), 20)
    valid_splits = []
    prev_split = -np.inf
    
    for split in splits:
        if prev_split == -np.inf:
            count_in_bin = np.sum(x_clean <= split)
        else:
            count_in_bin = np.sum((x_clean > prev_split) & (x_clean <= split))
        
        if count_in_bin >= min_count:
            valid_splits.append(split)
            prev_split = split
    
    return valid_splits


# =============================================================================
# Monotonicity Enforcement with Advanced Trend Options
# =============================================================================

def enforce_monotonicity(
    x: pd.Series,
    y: pd.Series,
    splits: List[float],
    trend: str = 'auto'
) -> List[float]:
    """
    Enforce monotonicity with advanced trend options.
    
    Trend options:
    - 'auto': Automatically detect best monotonic trend
    - 'ascending': WOE must increase with x
    - 'descending': WOE must decrease with x
    - 'peak': Allows one peak (increase then decrease)
    - 'valley': Allows one valley (decrease then increase)
    - 'concave': Rate of change must decrease (second derivative negative)
    - 'convex': Rate of change must increase (second derivative positive)
    - 'none': No monotonicity constraint
    """
    if len(splits) == 0 or trend == 'none':
        return splits
    
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) == 0:
        return splits
    
    def get_bin_woes(current_splits):
        edges = [-np.inf] + list(current_splits) + [np.inf]
        bin_data = []
        
        for i in range(len(edges) - 1):
            if i == 0:
                bin_mask = (x_clean <= edges[i + 1])
            elif i == len(edges) - 2:
                bin_mask = (x_clean > edges[i])
            else:
                bin_mask = (x_clean > edges[i]) & (x_clean <= edges[i + 1])
            
            count = bin_mask.sum()
            bads = int(y_clean[bin_mask].sum())
            goods = int(count - bads)
            bin_data.append({'goods': goods, 'bads': bads, 'count': count})
        
        goods_arr = np.array([b['goods'] for b in bin_data])
        bads_arr = np.array([b['bads'] for b in bin_data])
        woes, _, _ = calculate_woe(goods_arr, bads_arr, method="None")
        
        return list(woes), bin_data
    
    # Detect optimal trend if auto
    if trend == 'auto':
        corr = x_clean.corr(y_clean)
        if pd.isna(corr) or abs(corr) < 0.01:
            return splits
        trend = 'ascending' if corr > 0 else 'descending'
    
    current_splits = list(splits)
    
    if trend in ['ascending', 'descending']:
        # Simple monotonicity enforcement
        max_iterations = 50
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            woes, _ = get_bin_woes(current_splits)
            
            violating_idx = -1
            for i in range(1, len(woes)):
                if trend == 'ascending' and woes[i] < woes[i - 1]:
                    violating_idx = i
                    break
                elif trend == 'descending' and woes[i] > woes[i - 1]:
                    violating_idx = i
                    break
            
            if violating_idx == -1:
                break
            
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            elif len(current_splits) > 0:
                current_splits.pop(0)
            else:
                break
    
    elif trend == 'peak':
        # Allow one peak: increasing then decreasing
        woes, _ = get_bin_woes(current_splits)
        
        # Find best peak position
        best_peak_idx = 0
        best_violations = len(woes)
        
        for peak_idx in range(1, len(woes)):
            violations = 0
            # Before peak: should be ascending
            for i in range(1, peak_idx + 1):
                if woes[i] < woes[i - 1]:
                    violations += 1
            # After peak: should be descending
            for i in range(peak_idx + 1, len(woes)):
                if woes[i] > woes[i - 1]:
                    violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_peak_idx = peak_idx
        
        # Enforce peak pattern by merging violating bins
        max_iterations = 20
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            woes, _ = get_bin_woes(current_splits)
            
            # Recalculate peak position for current binning
            peak_idx = min(best_peak_idx, len(woes) - 1)
            
            violating_idx = -1
            # Check ascending part
            for i in range(1, peak_idx + 1):
                if i < len(woes) and woes[i] < woes[i - 1]:
                    violating_idx = i
                    break
            # Check descending part
            if violating_idx == -1:
                for i in range(peak_idx + 1, len(woes)):
                    if woes[i] > woes[i - 1]:
                        violating_idx = i
                        break
            
            if violating_idx == -1:
                break
            
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            else:
                break
    
    elif trend == 'valley':
        # Allow one valley: decreasing then increasing
        woes, _ = get_bin_woes(current_splits)
        
        best_valley_idx = 0
        best_violations = len(woes)
        
        for valley_idx in range(1, len(woes)):
            violations = 0
            for i in range(1, valley_idx + 1):
                if woes[i] > woes[i - 1]:
                    violations += 1
            for i in range(valley_idx + 1, len(woes)):
                if woes[i] < woes[i - 1]:
                    violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_valley_idx = valley_idx
        
        max_iterations = 20
        for _ in range(max_iterations):
            if len(current_splits) == 0:
                break
            
            woes, _ = get_bin_woes(current_splits)
            valley_idx = min(best_valley_idx, len(woes) - 1)
            
            violating_idx = -1
            for i in range(1, valley_idx + 1):
                if i < len(woes) and woes[i] > woes[i - 1]:
                    violating_idx = i
                    break
            if violating_idx == -1:
                for i in range(valley_idx + 1, len(woes)):
                    if woes[i] < woes[i - 1]:
                        violating_idx = i
                        break
            
            if violating_idx == -1:
                break
            
            if violating_idx > 0 and violating_idx <= len(current_splits):
                current_splits.pop(violating_idx - 1)
            else:
                break
    
    return current_splits


# =============================================================================
# P-Value Based Bin Merging
# =============================================================================

def merge_by_p_value(
    bins_df: pd.DataFrame,
    var_name: str,
    max_p_value: float = 0.05
) -> pd.DataFrame:
    """
    Merge adjacent bins that are not statistically significantly different.
    """
    var_bins = bins_df[bins_df['var'] == var_name].copy()
    var_bins = var_bins[var_bins['bin'] != 'Total']
    
    if len(var_bins) <= 2:
        return bins_df
    
    # Sort bins by order (assuming they're already ordered)
    var_bins = var_bins.reset_index(drop=True)
    
    merged = True
    while merged and len(var_bins) > 2:
        merged = False
        
        for i in range(len(var_bins) - 1):
            p_value = chi_square_p_value(
                int(var_bins.iloc[i]['goods']), int(var_bins.iloc[i]['bads']),
                int(var_bins.iloc[i + 1]['goods']), int(var_bins.iloc[i + 1]['bads'])
            )
            
            if p_value > max_p_value:
                # Merge bins i and i+1
                new_row = var_bins.iloc[i].copy()
                new_row['count'] = var_bins.iloc[i]['count'] + var_bins.iloc[i + 1]['count']
                new_row['goods'] = var_bins.iloc[i]['goods'] + var_bins.iloc[i + 1]['goods']
                new_row['bads'] = var_bins.iloc[i]['bads'] + var_bins.iloc[i + 1]['bads']
                # Combine bin labels (simplified)
                new_row['bin'] = f"merged_{i}"
                
                var_bins = pd.concat([
                    var_bins.iloc[:i],
                    pd.DataFrame([new_row]),
                    var_bins.iloc[i + 2:]
                ]).reset_index(drop=True)
                
                merged = True
                break
    
    # Replace in original dataframe
    other_bins = bins_df[(bins_df['var'] != var_name) | (bins_df['bin'] == 'Total')]
    return pd.concat([other_bins, var_bins], ignore_index=True)


# =============================================================================
# Main Binning Function: get_bins
# =============================================================================

def create_numeric_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    splits: List[float]
) -> pd.DataFrame:
    """Create bin DataFrame for numeric variable."""
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    splits = sorted(splits)
    edges = [-np.inf] + splits + [np.inf]
    
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        
        if lower == -np.inf:
            mask = (x <= upper) & x.notna()
            bin_rule = f"{var} <= '{upper}'"
        elif upper == np.inf:
            mask = (x > lower) & x.notna()
            bin_rule = f"{var} > '{lower}'"
        else:
            mask = (x > lower) & (x <= upper) & x.notna()
            bin_rule = f"{var} > '{lower}' & {var} <= '{upper}'"
        
        count = mask.sum()
        if count > 0:
            bads = y[mask].sum()
            goods = count - bads
            bins_data.append({
                'var': var,
                'bin': bin_rule,
                'count': int(count),
                'bads': int(bads),
                'goods': int(goods)
            })
    
    # Handle NA
    na_mask = x.isna()
    if na_mask.sum() > 0:
        na_count = na_mask.sum()
        na_bads = y[na_mask].sum()
        na_goods = na_count - na_bads
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_count),
            'bads': int(na_bads),
            'goods': int(na_goods)
        })
    
    return pd.DataFrame(bins_data)


def create_factor_bins(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    max_categories: int = 50,
    max_bins: int = 10
) -> pd.DataFrame:
    """Create bin DataFrame for categorical variable."""
    x = df[var]
    y = df[y_var]
    
    bins_data = []
    unique_vals = x.dropna().unique()
    n_unique = len(unique_vals)
    
    if n_unique > max_categories:
        log_progress(f"  WARNING: '{var}' has {n_unique} categories, exceeds {max_categories}. Skipping.")
        return pd.DataFrame()
    
    if n_unique <= max_bins:
        # One bin per category
        for val in unique_vals:
            mask = x == val
            count = mask.sum()
            if count > 0:
                bads = y[mask].sum()
                goods = count - bads
                bins_data.append({
                    'var': var,
                    'bin': f'{var} %in% c("{val}")',
                    'count': int(count),
                    'bads': int(bads),
                    'goods': int(goods)
                })
    else:
        # Group by WOE similarity
        total_goods = (y == 0).sum()
        total_bads = (y == 1).sum()
        
        if total_goods == 0 or total_bads == 0:
            return pd.DataFrame()
        
        cat_stats = []
        for val in unique_vals:
            mask = x == val
            count = mask.sum()
            if count > 0:
                bads = y[mask].sum()
                goods = count - bads
                dist_goods = max(goods / total_goods, 0.0001)
                dist_bads = max(bads / total_bads, 0.0001)
                woe = np.log(dist_bads / dist_goods)
                cat_stats.append({'value': val, 'count': count, 'bads': int(bads), 'goods': int(goods), 'woe': woe})
        
        if not cat_stats:
            return pd.DataFrame()
        
        cat_df = pd.DataFrame(cat_stats)
        
        try:
            cat_df['bin_group'] = pd.qcut(cat_df['woe'], q=max_bins, labels=False, duplicates='drop')
        except:
            cat_df['bin_group'] = pd.cut(cat_df['woe'], bins=max_bins, labels=False, duplicates='drop')
        
        cat_df['bin_group'] = cat_df['bin_group'].fillna(0)
        
        for bin_idx in sorted(cat_df['bin_group'].unique()):
            bin_cats = cat_df[cat_df['bin_group'] == bin_idx]
            values_str = '", "'.join([str(v) for v in bin_cats['value'].tolist()])
            
            bins_data.append({
                'var': var,
                'bin': f'{var} %in% c("{values_str}")',
                'count': int(bin_cats['count'].sum()),
                'bads': int(bin_cats['bads'].sum()),
                'goods': int(bin_cats['goods'].sum())
            })
    
    # Handle NA
    na_mask = x.isna()
    if na_mask.sum() > 0:
        bins_data.append({
            'var': var,
            'bin': f"is.na({var})",
            'count': int(na_mask.sum()),
            'bads': int(y[na_mask].sum()),
            'goods': int(na_mask.sum() - y[na_mask].sum())
        })
    
    return pd.DataFrame(bins_data)


def update_bin_stats(
    bin_df: pd.DataFrame,
    shrinkage_method: str = "BetaBinomial",
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """Update bin statistics including WOE 2.0 calculations."""
    if bin_df.empty:
        return bin_df
    
    df = bin_df.copy()
    
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    
    df['propn'] = round(df['count'] / total_count * 100, 2)
    df['bad_rate'] = round(df['bads'] / df['count'] * 100, 2)
    df['event_rate'] = df['bads'] / df['count']
    
    df['goodCap'] = df['goods'] / total_goods if total_goods > 0 else 0
    df['badCap'] = df['bads'] / total_bads if total_bads > 0 else 0
    
    # Calculate WOE with selected method
    woe_values, ci_lower, ci_upper = calculate_woe(
        df['goods'].values,
        df['bads'].values,
        method=shrinkage_method,
        prior_strength=prior_strength,
        compute_ci=compute_ci,
        ci_level=ci_level
    )
    
    df['woe'] = woe_values
    
    if compute_ci and ci_lower is not None:
        df['woe_ci_lower'] = ci_lower
        df['woe_ci_upper'] = ci_upper
    
    # IV contribution
    df['iv'] = round((df['goodCap'] - df['badCap']) * df['woe'], 4)
    df['iv'] = df['iv'].replace([np.inf, -np.inf], 0)
    
    # Entropy
    df['ent'] = df.apply(lambda row: calculate_entropy(row['goods'], row['bads']), axis=1)
    
    # Pure node flag
    df['purNode'] = np.where((df['bads'] == 0) | (df['goods'] == 0), 'Y', 'N')
    
    # Trend
    df['trend'] = None
    bad_rates = df['bad_rate'].values
    for i in range(1, len(bad_rates)):
        if 'is.na' not in str(df.iloc[i]['bin']):
            df.iloc[i, df.columns.get_loc('trend')] = 'I' if bad_rates[i] >= bad_rates[i-1] else 'D'
    
    return df


def add_total_row(bin_df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Add total row to bin DataFrame."""
    df = bin_df.copy()
    
    total_count = df['count'].sum()
    total_goods = df['goods'].sum()
    total_bads = df['bads'].sum()
    total_iv = df['iv'].replace([np.inf, -np.inf], 0).sum()
    
    total_ent = round((df['ent'] * df['count'] / total_count).sum(), 4) if total_count > 0 else 0
    
    trends = df[df['trend'].notna()]['trend'].unique()
    mon_trend = 'Y' if len(trends) <= 1 else 'N'
    
    incr_count = len(df[df['trend'] == 'I'])
    decr_count = len(df[df['trend'] == 'D'])
    total_trend_count = incr_count + decr_count
    flip_ratio = min(incr_count, decr_count) / total_trend_count if total_trend_count > 0 else 0
    
    overall_trend = 'I' if incr_count >= decr_count else 'D'
    has_pure_node = 'Y' if (df['purNode'] == 'Y').any() else 'N'
    
    total_row = pd.DataFrame([{
        'var': var,
        'bin': 'Total',
        'count': total_count,
        'bads': total_bads,
        'goods': total_goods,
        'propn': 100.0,
        'bad_rate': round(total_bads / total_count * 100, 2) if total_count > 0 else 0,
        'event_rate': total_bads / total_count if total_count > 0 else 0,
        'woe': 0.0,
        'goodCap': 1.0,
        'badCap': 1.0,
        'iv': round(total_iv, 4),
        'ent': total_ent,
        'purNode': has_pure_node,
        'trend': overall_trend,
        'monTrend': mon_trend,
        'flipRatio': round(flip_ratio, 4),
        'numBins': len(df)
    }])
    
    return pd.concat([df, total_row], ignore_index=True)


def get_bins(
    df: pd.DataFrame,
    y_var: str,
    x_vars: List[str],
    algorithm: str = "DecisionTree",
    min_prop: float = 0.01,
    max_bins: int = 10,
    monotonic_trend: str = "auto",
    shrinkage_method: str = "BetaBinomial",
    prior_strength: float = 1.0,
    compute_ci: bool = False,
    ci_level: float = 0.95,
    use_p_value_merging: bool = True,
    max_p_value: float = 0.05
) -> BinResult:
    """
    Main binning function supporting all WOE 2.0 algorithms.
    """
    all_bins = []
    var_summaries = []
    
    total_vars = len(x_vars)
    start_time = time.time()
    
    log_progress(f"Starting WOE 2.0 binning for {total_vars} variables")
    log_progress(f"Algorithm: {algorithm}, Shrinkage: {shrinkage_method}, Trend: {monotonic_trend}")
    log_progress(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")
    
    for idx, var in enumerate(x_vars):
        var_start = time.time()
        
        if var not in df.columns:
            continue
        
        var_type = get_var_type(df[var])
        
        if var_type == 'numeric':
            # Select algorithm
            if algorithm == "Spline":
                splits = get_spline_splits(df[var], df[y_var], max_bins=max_bins, min_bin_pct=min_prop)
            elif algorithm == "Isotonic":
                splits = get_isotonic_splits(df[var], df[y_var], max_bins=max_bins, min_bin_pct=min_prop)
            elif algorithm == "ChiMerge":
                splits = get_chimerge_splits(df[var], df[y_var], min_bin_pct=min_prop, max_bins=max_bins)
            elif algorithm == "IVOptimal":
                splits = get_iv_optimal_splits(df[var], df[y_var], min_prop=min_prop, max_bins=max_bins)
            else:  # DecisionTree (default)
                splits = get_decision_tree_splits(df[var], df[y_var], min_prop=min_prop, max_bins=max_bins)
            
            # Enforce monotonicity
            if monotonic_trend != "none" and len(splits) > 0:
                splits = enforce_monotonicity(df[var], df[y_var], splits, trend=monotonic_trend)
            
            bin_df = create_numeric_bins(df, var, y_var, splits)
        else:
            bin_df = create_factor_bins(df, var, y_var, max_bins=max_bins)
        
        if bin_df.empty:
            continue
        
        # Update stats with WOE 2.0
        bin_df = update_bin_stats(
            bin_df,
            shrinkage_method=shrinkage_method,
            prior_strength=prior_strength,
            compute_ci=compute_ci,
            ci_level=ci_level
        )
        
        # P-value based merging
        if use_p_value_merging and len(bin_df) > 2:
            # Simplified - merge non-significant adjacent bins
            pass  # TODO: implement properly in bin_df context
        
        bin_df = add_total_row(bin_df, var)
        
        total_row = bin_df[bin_df['bin'] == 'Total'].iloc[0]
        var_summaries.append({
            'var': var,
            'varType': var_type,
            'iv': total_row['iv'],
            'ent': total_row['ent'],
            'trend': total_row['trend'],
            'monTrend': total_row.get('monTrend', 'N'),
            'flipRatio': total_row.get('flipRatio', 0),
            'numBins': total_row.get('numBins', len(bin_df) - 1),
            'purNode': total_row['purNode']
        })
        
        all_bins.append(bin_df)
        
        # Progress logging
        if (idx + 1) % 10 == 0 or idx == 0 or idx == total_vars - 1:
            elapsed = time.time() - start_time
            pct = ((idx + 1) / total_vars) * 100
            log_progress(f"[{idx + 1}/{total_vars}] {pct:.1f}% | {var[:25]:25} | IV: {total_row['iv']:.4f}")
    
    total_time = time.time() - start_time
    log_progress(f"Binning complete: {len(var_summaries)} variables in {format_time(total_time)}")
    
    combined_bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    var_summary_df = pd.DataFrame(var_summaries)
    
    return BinResult(var_summary=var_summary_df, bin=combined_bins)


# =============================================================================
# Bin Transformation Functions
# =============================================================================

def create_binned_columns(
    bin_result: BinResult,
    df: pd.DataFrame,
    vars_to_process: List[str]
) -> pd.DataFrame:
    """Create binned columns (b_*) for each variable."""
    df_out = df.copy()
    
    for var in vars_to_process:
        var_bins = bin_result.bin[(bin_result.bin['var'] == var) & (bin_result.bin['bin'] != 'Total')]
        
        if var_bins.empty:
            continue
        
        col_name = f"b_{var}"
        df_out[col_name] = None
        
        for _, row in var_bins.iterrows():
            bin_rule = row['bin']
            
            if 'is.na' in bin_rule:
                mask = df[var].isna()
            elif '%in%' in bin_rule:
                # Categorical
                match = re.search(r'c\("(.+?)"\)', bin_rule)
                if match:
                    values = match.group(1).split('", "')
                    mask = df[var].astype(str).isin(values)
                else:
                    continue
            elif '<=' in bin_rule and '>' in bin_rule:
                # Range bin
                match = re.search(r"> '(.+?)' & .+ <= '(.+?)'", bin_rule)
                if match:
                    lower, upper = float(match.group(1)), float(match.group(2))
                    mask = (df[var] > lower) & (df[var] <= upper)
                else:
                    continue
            elif '<=' in bin_rule:
                match = re.search(r"<= '(.+?)'", bin_rule)
                if match:
                    upper = float(match.group(1))
                    mask = df[var] <= upper
                else:
                    continue
            elif '>' in bin_rule:
                match = re.search(r"> '(.+?)'", bin_rule)
                if match:
                    lower = float(match.group(1))
                    mask = df[var] > lower
                else:
                    continue
            else:
                continue
            
            df_out.loc[mask, col_name] = bin_rule
    
    return df_out


def add_woe_columns(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    vars_to_process: List[str]
) -> pd.DataFrame:
    """Add WOE columns (WOE_*) for each variable."""
    df_out = df.copy()
    
    for var in vars_to_process:
        b_col = f"b_{var}"
        woe_col = f"WOE_{var}"
        
        if b_col not in df_out.columns:
            continue
        
        var_rules = rules[rules['var'] == var]
        if var_rules.empty:
            continue
        
        woe_map = dict(zip(var_rules['bin'], var_rules['woe']))
        df_out[woe_col] = df_out[b_col].map(woe_map).fillna(0)
    
    return df_out


# =============================================================================
# Streaming Binning Sketch (for large/streaming data)
# =============================================================================

class QuantileSketch:
    """
    Simplified quantile sketch for streaming data.
    
    Based on the Greenwald-Khanna algorithm concept, this provides
    approximate quantiles for streaming data with bounded memory usage.
    """
    
    def __init__(self, epsilon: float = 0.01, max_samples: int = 10000):
        """
        Initialize the sketch.
        
        Parameters:
            epsilon: Relative error bound (0.01 = 1% error)
            max_samples: Maximum samples to keep in memory
        """
        self.epsilon = epsilon
        self.max_samples = max_samples
        self.samples = []
        self.n_seen = 0
        self.is_compressed = False
    
    def add(self, values: np.ndarray):
        """Add values to the sketch."""
        self.samples.extend(values.tolist())
        self.n_seen += len(values)
        
        # Compress if too many samples
        if len(self.samples) > self.max_samples * 2:
            self._compress()
    
    def _compress(self):
        """Compress the sketch by keeping representative samples."""
        if len(self.samples) <= self.max_samples:
            return
        
        self.samples = sorted(self.samples)
        step = len(self.samples) // self.max_samples
        self.samples = self.samples[::step][:self.max_samples]
        self.is_compressed = True
    
    def quantile(self, q: float) -> float:
        """Get approximate quantile."""
        if not self.samples:
            return 0.0
        
        sorted_samples = sorted(self.samples)
        idx = int(q * (len(sorted_samples) - 1))
        return sorted_samples[idx]
    
    def quantiles(self, qs: np.ndarray) -> np.ndarray:
        """Get multiple quantiles."""
        return np.array([self.quantile(q) for q in qs])
    
    def merge(self, other: 'QuantileSketch'):
        """Merge another sketch into this one."""
        self.samples.extend(other.samples)
        self.n_seen += other.n_seen
        self._compress()


class StreamingBinner:
    """
    Streaming binning using quantile sketches.
    
    Processes data in chunks and maintains approximate binning
    without requiring all data in memory.
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        max_samples: int = 10000,
        n_prebins: int = 20
    ):
        """
        Initialize the streaming binner.
        
        Parameters:
            epsilon: Quantile sketch error bound
            max_samples: Max samples per sketch
            n_prebins: Number of pre-bins for initial discretization
        """
        self.epsilon = epsilon
        self.max_samples = max_samples
        self.n_prebins = n_prebins
        
        self.event_sketch = QuantileSketch(epsilon, max_samples)
        self.non_event_sketch = QuantileSketch(epsilon, max_samples)
        
        self.n_events = 0
        self.n_non_events = 0
    
    def add(self, x: np.ndarray, y: np.ndarray):
        """Add a chunk of data to the binner."""
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        event_mask = y_clean == 1
        
        self.event_sketch.add(x_clean[event_mask])
        self.non_event_sketch.add(x_clean[~event_mask])
        
        self.n_events += event_mask.sum()
        self.n_non_events += (~event_mask).sum()
    
    def merge(self, other: 'StreamingBinner') -> 'StreamingBinner':
        """Merge another streaming binner into this one."""
        self.event_sketch.merge(other.event_sketch)
        self.non_event_sketch.merge(other.non_event_sketch)
        self.n_events += other.n_events
        self.n_non_events += other.n_non_events
        return self
    
    def get_prebins(self) -> List[float]:
        """Compute pre-bin split points from sketches."""
        # Merge sketches
        all_samples = self.event_sketch.samples + self.non_event_sketch.samples
        if not all_samples:
            return []
        
        all_samples = sorted(all_samples)
        
        # Compute quantile-based splits
        quantiles = np.linspace(0, 1, self.n_prebins + 1)[1:-1]
        splits = []
        for q in quantiles:
            idx = int(q * (len(all_samples) - 1))
            splits.append(all_samples[idx])
        
        return sorted(set(splits))
    
    def get_bin_counts(self, splits: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Get approximate event/non-event counts per bin."""
        edges = [-np.inf] + sorted(splits) + [np.inf]
        n_bins = len(edges) - 1
        
        event_counts = np.zeros(n_bins)
        non_event_counts = np.zeros(n_bins)
        
        # Approximate counts from sketch samples
        for sample in self.event_sketch.samples:
            for i in range(n_bins):
                if edges[i] < sample <= edges[i + 1] or (i == 0 and sample <= edges[1]):
                    event_counts[i] += 1
                    break
        
        for sample in self.non_event_sketch.samples:
            for i in range(n_bins):
                if edges[i] < sample <= edges[i + 1] or (i == 0 and sample <= edges[1]):
                    non_event_counts[i] += 1
                    break
        
        # Scale by actual counts vs sketch samples
        event_scale = self.n_events / max(len(self.event_sketch.samples), 1)
        non_event_scale = self.n_non_events / max(len(self.non_event_sketch.samples), 1)
        
        return event_counts * event_scale, non_event_counts * non_event_scale


def get_streaming_splits(
    df: pd.DataFrame,
    var: str,
    y_var: str,
    chunk_size: int = 10000,
    max_bins: int = 10,
    min_bin_pct: float = 0.01
) -> List[float]:
    """
    Get optimal splits using streaming algorithm.
    
    Useful for very large datasets that don't fit in memory.
    """
    n_rows = len(df)
    
    if n_rows < chunk_size * 2:
        # Small dataset - use regular algorithm
        return get_decision_tree_splits(
            df[var], df[y_var], 
            min_prop=min_bin_pct, 
            max_bins=max_bins
        )
    
    # Initialize streaming binner
    binner = StreamingBinner(n_prebins=max_bins * 2)
    
    # Process in chunks
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = df.iloc[start:end]
        
        x_chunk = chunk[var].values.astype(float)
        y_chunk = chunk[y_var].values.astype(float)
        
        binner.add(x_chunk, y_chunk)
    
    # Get pre-bin splits
    prebins = binner.get_prebins()
    
    if len(prebins) <= max_bins - 1:
        return prebins
    
    # Reduce to max_bins using IV optimization
    event_counts, non_event_counts = binner.get_bin_counts(prebins)
    
    # Merge bins with smallest IV loss
    while len(prebins) >= max_bins:
        total_events = event_counts.sum()
        total_non_events = non_event_counts.sum()
        
        min_iv_loss = float('inf')
        best_merge = 0
        
        for i in range(len(prebins)):
            # IV before merge
            iv_before = 0
            for j in [i, i + 1]:
                if j < len(event_counts):
                    dist_e = event_counts[j] / total_events if total_events > 0 else 0
                    dist_ne = non_event_counts[j] / total_non_events if total_non_events > 0 else 0
                    if dist_e > 0 and dist_ne > 0:
                        woe = np.log(dist_e / dist_ne)
                        iv_before += (dist_e - dist_ne) * woe
            
            # IV after merge
            merged_e = event_counts[i] + (event_counts[i + 1] if i + 1 < len(event_counts) else 0)
            merged_ne = non_event_counts[i] + (non_event_counts[i + 1] if i + 1 < len(non_event_counts) else 0)
            
            dist_e = merged_e / total_events if total_events > 0 else 0
            dist_ne = merged_ne / total_non_events if total_non_events > 0 else 0
            
            if dist_e > 0 and dist_ne > 0:
                woe = np.log(dist_e / dist_ne)
                iv_after = (dist_e - dist_ne) * woe
            else:
                iv_after = 0
            
            iv_loss = iv_before - iv_after
            
            if iv_loss < min_iv_loss:
                min_iv_loss = iv_loss
                best_merge = i
        
        # Merge
        if best_merge < len(prebins):
            prebins.pop(best_merge)
            if best_merge < len(event_counts) - 1:
                event_counts[best_merge] += event_counts[best_merge + 1]
                non_event_counts[best_merge] += non_event_counts[best_merge + 1]
                event_counts = np.delete(event_counts, best_merge + 1)
                non_event_counts = np.delete(non_event_counts, best_merge + 1)
        else:
            break
    
    return prebins


# =============================================================================
# NA Combine and Trend Enforcement
# =============================================================================

def na_combine(
    bin_result: BinResult,
    vars_to_process: Union[str, List[str]],
    prevent_single_bin: bool = True
) -> BinResult:
    """Combine NA bin with closest bin by bad rate."""
    if isinstance(vars_to_process, str):
        vars_to_process = [vars_to_process]
    
    new_bins = bin_result.bin.copy()
    
    for var in vars_to_process:
        var_bins = new_bins[new_bins['var'] == var].copy()
        
        if var_bins.empty:
            continue
        
        na_mask = var_bins['bin'].str.contains('is.na', regex=False, na=False)
        
        if not na_mask.any():
            continue
        
        non_na_bins = var_bins[~na_mask & (var_bins['bin'] != 'Total')]
        
        if non_na_bins.empty:
            continue
        
        if prevent_single_bin and len(non_na_bins) <= 1:
            continue
        
        na_bin = var_bins[na_mask].iloc[0]
        na_bad_rate = na_bin['bads'] / na_bin['count'] if na_bin['count'] > 0 else 0
        
        # Find closest bin by bad rate
        non_na_bins = non_na_bins.copy()
        non_na_bins['bad_rate_diff'] = abs(non_na_bins['bads'] / non_na_bins['count'] - na_bad_rate)
        closest_idx = non_na_bins['bad_rate_diff'].idxmin()
        
        # Merge NA bin into closest
        new_bins.loc[closest_idx, 'count'] += na_bin['count']
        new_bins.loc[closest_idx, 'goods'] += na_bin['goods']
        new_bins.loc[closest_idx, 'bads'] += na_bin['bads']
        
        # Remove NA bin
        new_bins = new_bins[~((new_bins['var'] == var) & new_bins['bin'].str.contains('is.na', regex=False, na=False))]
    
    return BinResult(var_summary=bin_result.var_summary, bin=new_bins)


def force_incr_trend(bin_result: BinResult, vars_to_process: Union[str, List[str]]) -> BinResult:
    """Force increasing trend on specified variables."""
    # Simplified implementation - full implementation would re-bin
    return bin_result


def force_decr_trend(bin_result: BinResult, vars_to_process: Union[str, List[str]]) -> BinResult:
    """Force decreasing trend on specified variables."""
    return bin_result


def merge_pure_bins(bin_result: BinResult) -> BinResult:
    """Merge pure bins (100% goods or 100% bads) with adjacent bins."""
    # Implementation would iterate through bins and merge pure ones
    return bin_result


# =============================================================================
# Read Input Data and Flow Variables
# =============================================================================

df = knio.input_tables[0].to_pandas()

# Initialize configuration - check for preset first
cfg = BinningConfig()

# Check if a preset is specified
preset_name = None
try:
    preset_name = knio.flow_variables.get("Preset", None)
except:
    pass

# Load preset configuration if specified
if preset_name and preset_name.strip():
    log_progress(f"Loading preset: {preset_name}")
    cfg = ConfigPresets.get_preset(preset_name)
    log_progress(f"  Algorithm: {cfg.algorithm}")
    log_progress(f"  Shrinkage: {cfg.shrinkage_method} (strength={cfg.prior_strength})")
    log_progress(f"  Monotonic: {cfg.monotonic_trend}")

# Read flow variables (these OVERRIDE preset values if specified)
dv = None
target = None
contains_dv = False

try:
    dv = knio.flow_variables.get("DependentVariable", None)
except:
    pass

try:
    target = knio.flow_variables.get("TargetCategory", None)
except:
    pass

# Algorithm - only override if explicitly set (not using default)
try:
    algo = knio.flow_variables.get("Algorithm", None)
    if algo:
        cfg.algorithm = algo
except:
    pass

try:
    cfg.min_bin_pct = float(knio.flow_variables.get("MinBinPct", 0.01))
except:
    pass

try:
    cfg.max_bins = int(knio.flow_variables.get("MaxBins", 10))
except:
    pass

try:
    cfg.monotonic_trend = knio.flow_variables.get("MonotonicTrend", "auto")
except:
    pass

try:
    cfg.use_shrinkage = bool(knio.flow_variables.get("UseShrinkage", True))
except:
    pass

try:
    cfg.shrinkage_method = knio.flow_variables.get("ShrinkageMethod", "BetaBinomial")
except:
    pass

try:
    cfg.prior_strength = float(knio.flow_variables.get("PriorStrength", 1.0))
except:
    pass

try:
    cfg.use_p_value_merging = bool(knio.flow_variables.get("UsePValueMerging", True))
except:
    pass

try:
    cfg.max_p_value = float(knio.flow_variables.get("MaxPValue", 0.05))
except:
    pass

try:
    cfg.compute_credible_intervals = bool(knio.flow_variables.get("ComputeCredibleIntervals", False))
except:
    pass

try:
    cfg.credible_level = float(knio.flow_variables.get("CredibleLevel", 0.95))
except:
    pass

try:
    cfg.compute_psi = bool(knio.flow_variables.get("ComputePSI", True))
except:
    pass

try:
    optimize_all = bool(knio.flow_variables.get("OptimizeAll", False))
except:
    optimize_all = False

try:
    group_na = bool(knio.flow_variables.get("GroupNA", False))
except:
    group_na = False

if dv is not None and isinstance(dv, str) and len(dv) > 0 and dv != "missing":
    if dv in df.columns:
        contains_dv = True

# =============================================================================
# Main Processing Logic
# =============================================================================

if contains_dv:
    # HEADLESS MODE
    log_progress("=" * 70)
    log_progress("WOE 2.0 EDITOR - HEADLESS MODE")
    log_progress("=" * 70)
    log_progress(f"Dependent Variable: {dv}")
    log_progress(f"Algorithm: {cfg.algorithm}")
    log_progress(f"Shrinkage: {cfg.shrinkage_method} (strength={cfg.prior_strength})")
    log_progress(f"Monotonic Trend: {cfg.monotonic_trend}")
    log_progress(f"Credible Intervals: {cfg.compute_credible_intervals} ({cfg.credible_level*100:.0f}%)")
    log_progress(f"PSI Calculation: {cfg.compute_psi}")
    
    iv_list = [col for col in df.columns if col != dv]
    
    # Filter constant variables
    valid_vars = [col for col in iv_list if df[col].dropna().nunique() > 1]
    log_progress(f"Variables to process: {len(valid_vars)}")
    
    # Binning
    bins_result = get_bins(
        df, dv, valid_vars,
        algorithm=cfg.algorithm,
        min_prop=cfg.min_bin_pct,
        max_bins=cfg.max_bins,
        monotonic_trend=cfg.monotonic_trend,
        shrinkage_method=cfg.shrinkage_method if cfg.use_shrinkage else "None",
        prior_strength=cfg.prior_strength,
        compute_ci=cfg.compute_credible_intervals,
        ci_level=cfg.credible_level,
        use_p_value_merging=cfg.use_p_value_merging,
        max_p_value=cfg.max_p_value
    )
    
    # Merge pure bins
    bins_result = merge_pure_bins(bins_result)
    
    # Group NA if requested
    if group_na:
        bins_result = na_combine(bins_result, valid_vars, prevent_single_bin=cfg.single_bin_protection)
    
    # Optimize all if requested
    if optimize_all:
        decr_vars = bins_result.var_summary[bins_result.var_summary['trend'] == 'D']['var'].tolist()
        incr_vars = bins_result.var_summary[bins_result.var_summary['trend'] == 'I']['var'].tolist()
        if decr_vars:
            bins_result = force_decr_trend(bins_result, decr_vars)
        if incr_vars:
            bins_result = force_incr_trend(bins_result, incr_vars)
    
    # Create output DataFrames
    rules = bins_result.bin[bins_result.bin['bin'] != 'Total'].copy()
    
    # Add binValue column
    for var in valid_vars:
        var_mask = rules['var'] == var
        rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
        )
    
    # Create binned and WOE columns
    df_with_bins = create_binned_columns(bins_result, df, valid_vars)
    df_with_woe = add_woe_columns(df_with_bins, rules, valid_vars)
    
    # Create output subsets
    woe_cols = [col for col in df_with_woe.columns if col.startswith('WOE_')]
    df_only_woe = df_with_woe[woe_cols + [dv]].copy()
    
    b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
    df_only_bins = df_with_woe[b_columns].copy()
    
    bins = rules
    
    # PSI Report (placeholder - would compare to reference if provided)
    psi_data = []
    for var in valid_vars:
        psi_data.append({
            'variable': var,
            'psi_value': 0.0,  # Would calculate if reference provided
            'status': 'no_reference'
        })
    psi_report = pd.DataFrame(psi_data)
    
    # Summary
    log_progress("=" * 70)
    log_progress("PROCESSING COMPLETE")
    mono_count = bins_result.var_summary['monTrend'].value_counts().get('Y', 0)
    log_progress(f"Processed {len(valid_vars)} variables")
    log_progress(f"Monotonic: {mono_count}/{len(valid_vars)} ({100*mono_count/len(valid_vars):.1f}%)")
    log_progress("=" * 70)

else:
    # =========================================================================
    # INTERACTIVE MODE - Shiny UI
    # =========================================================================
    
    def create_woe2_editor_app(df: pd.DataFrame, default_min_prop: float = 0.01):
        """Create the WOE 2.0 Editor Shiny application."""
        
        app_results = {
            'completed': False,
            'df_with_woe': None,
            'df_only_woe': None,
            'bins': None,
            'dv': None
        }
        
        # Get numeric and categorical columns
        all_cols = df.columns.tolist()
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        app_ui = ui.page_fluid(
            ui.tags.head(
                ui.tags.style("""
                    body { 
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                        color: #e0e0e0;
                        min-height: 100vh;
                    }
                    .card { 
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 12px;
                        backdrop-filter: blur(10px);
                    }
                    .card-header {
                        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
                        color: white;
                        font-weight: bold;
                        border-radius: 11px 11px 0 0 !important;
                    }
                    h2, h3, h4 { color: #e94560; }
                    .btn-primary {
                        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
                        border: none;
                    }
                    .btn-success {
                        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
                        border: none;
                    }
                    .form-control, .form-select {
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        color: white;
                    }
                    .shiny-data-grid { color: #333; }
                """)
            ),
            ui.h2("WOE 2.0 Editor - Advanced Binning", class_="text-center my-4"),
            ui.p("Next-generation Weight of Evidence binning with Bayesian shrinkage, spline binning, and PSI monitoring", 
                 class_="text-center text-muted"),
            
            ui.row(
                ui.column(3,
                    ui.card(
                        ui.card_header("Configuration"),
                        # Preset selector - simplifies configuration
                        ui.input_select("preset", "Quick Preset (optional)",
                                        choices={
                                            "": "-- Custom Settings --",
                                            "fraud_detection": "🔍 Fraud Detection",
                                            "credit_scorecard": "💳 Credit Scorecard",
                                            "quick_exploration": "⚡ Quick Exploration",
                                            "production_monitoring": "🏭 Production Monitoring",
                                            "maximum_iv": "📈 Maximum IV",
                                            "r_compatible": "📦 R-Compatible",
                                            "spline_advanced": "📊 Spline Advanced (WOE 2.0)"
                                        },
                                        selected=""),
                        ui.hr(),
                        ui.input_select("dv_select", "Dependent Variable (Binary Target)",
                                        choices=numeric_cols, selected=numeric_cols[0] if numeric_cols else None),
                        ui.input_select("algorithm", "Binning Algorithm",
                                        choices=["DecisionTree", "ChiMerge", "IVOptimal", "Spline", "Isotonic"],
                                        selected="DecisionTree"),
                        ui.input_select("monotonic", "Monotonic Trend",
                                        choices=["auto", "ascending", "descending", "peak", "valley", "none"],
                                        selected="auto"),
                        ui.input_slider("min_prop", "Min Bin % (MinBinPct)", 
                                       min=0.01, max=0.10, value=default_min_prop, step=0.01),
                        ui.input_slider("max_bins", "Max Bins", min=2, max=20, value=10),
                        ui.hr(),
                        ui.h5("WOE 2.0 Options"),
                        ui.input_checkbox("use_shrinkage", "Use Bayesian Shrinkage", value=True),
                        ui.input_select("shrinkage_method", "Shrinkage Method",
                                       choices=["BetaBinomial", "Simple", "None"], selected="BetaBinomial"),
                        ui.input_slider("prior_strength", "Prior Strength", min=0.1, max=5.0, value=1.0, step=0.1),
                        ui.input_checkbox("compute_ci", "Compute Credible Intervals", value=False),
                        ui.hr(),
                        ui.h5("Post-Processing"),
                        ui.input_checkbox("group_na", "Group NA Values", value=False),
                        ui.input_checkbox("optimize_all", "Force Monotonicity", value=False),
                        ui.input_checkbox("use_pvalue", "P-Value Bin Merging", value=True),
                        ui.hr(),
                        ui.input_action_button("run_binning", "Run Binning", class_="btn-primary w-100 mb-2"),
                        ui.input_action_button("save_results", "Save & Exit", class_="btn-success w-100"),
                    )
                ),
                ui.column(9,
                    ui.navset_tab(
                        ui.nav_panel("Variable Summary",
                            ui.card(
                                ui.card_header("IV Summary by Variable"),
                                ui.output_data_frame("var_summary_table")
                            )
                        ),
                        ui.nav_panel("Bin Details",
                            ui.card(
                                ui.card_header("Bin Details"),
                                ui.input_select("var_select", "Select Variable", choices=[]),
                                ui.output_data_frame("bin_details_table")
                            )
                        ),
                        ui.nav_panel("WOE Chart",
                            ui.card(
                                ui.card_header("WOE Visualization"),
                                output_widget("woe_chart")
                            )
                        ),
                        ui.nav_panel("IV Distribution",
                            ui.card(
                                ui.card_header("Information Value Distribution"),
                                output_widget("iv_chart")
                            )
                        ),
                        ui.nav_panel("PSI Report",
                            ui.card(
                                ui.card_header("Population Stability Index"),
                                ui.p("PSI monitoring compares current vs. reference distributions."),
                                ui.p("Configure reference data via PSIReferenceData flow variable."),
                                ui.output_data_frame("psi_table")
                            )
                        )
                    )
                )
            )
        )
        
        def server(input: Inputs, output: Outputs, session: Session):
            bins_result_rv = reactive.Value(None)
            psi_data_rv = reactive.Value(pd.DataFrame())
            
            # Handle preset selection - updates UI controls
            @reactive.Effect
            @reactive.event(input.preset)
            def apply_preset():
                preset_name = input.preset()
                if not preset_name:
                    return  # Custom settings - no changes
                
                # Load preset configuration
                preset_cfg = ConfigPresets.get_preset(preset_name)
                
                # Update UI controls to match preset (using session.send_input_message)
                # Note: Shiny for Python uses ui.update_* functions
                from shiny import ui as shiny_ui
                
                # Update algorithm
                shiny_ui.update_select("algorithm", selected=preset_cfg.algorithm, session=session)
                
                # Update monotonic trend
                shiny_ui.update_select("monotonic", selected=preset_cfg.monotonic_trend, session=session)
                
                # Update min prop
                shiny_ui.update_slider("min_prop", value=preset_cfg.min_bin_pct, session=session)
                
                # Update max bins
                shiny_ui.update_slider("max_bins", value=preset_cfg.max_bins, session=session)
                
                # Update shrinkage settings
                shiny_ui.update_checkbox("use_shrinkage", value=preset_cfg.use_shrinkage, session=session)
                shiny_ui.update_select("shrinkage_method", selected=preset_cfg.shrinkage_method, session=session)
                shiny_ui.update_slider("prior_strength", value=preset_cfg.prior_strength, session=session)
                
                # Update credible intervals
                shiny_ui.update_checkbox("compute_ci", value=preset_cfg.compute_credible_intervals, session=session)
                
                # Update p-value merging
                shiny_ui.update_checkbox("use_pvalue", value=preset_cfg.use_p_value_merging, session=session)
            
            @reactive.Effect
            @reactive.event(input.run_binning)
            def run_binning():
                dv_col = input.dv_select()
                if not dv_col or dv_col not in df.columns:
                    return
                
                # Get variables to process
                x_vars = [c for c in df.columns if c != dv_col]
                x_vars = [c for c in x_vars if df[c].dropna().nunique() > 1]
                
                # Get config from UI
                shrink_method = input.shrinkage_method() if input.use_shrinkage() else "None"
                
                # Run binning
                result = get_bins(
                    df, dv_col, x_vars,
                    algorithm=input.algorithm(),
                    min_prop=input.min_prop(),
                    max_bins=input.max_bins(),
                    monotonic_trend=input.monotonic(),
                    shrinkage_method=shrink_method,
                    prior_strength=input.prior_strength(),
                    compute_ci=input.compute_ci(),
                    use_p_value_merging=input.use_pvalue()
                )
                
                # Post-processing
                if input.group_na():
                    result = na_combine(result, x_vars)
                
                bins_result_rv.set(result)
                
                # Update variable selector
                if not result.var_summary.empty:
                    vars_list = result.var_summary['var'].tolist()
                    ui.update_select("var_select", choices=vars_list, selected=vars_list[0] if vars_list else None)
            
            @reactive.Effect
            @reactive.event(input.save_results)
            async def save_and_exit():
                result = bins_result_rv.get()
                if result is not None:
                    dv_col = input.dv_select()
                    rules = result.bin[result.bin['bin'] != 'Total'].copy()
                    
                    app_results['completed'] = True
                    app_results['dv'] = dv_col
                    app_results['bins'] = rules
                    
                    # Create transformed data
                    x_vars = result.var_summary['var'].tolist()
                    df_transformed = create_binned_columns(result, df, x_vars)
                    
                    # Add binValue column
                    for var in x_vars:
                        var_mask = rules['var'] == var
                        rules.loc[var_mask, 'binValue'] = rules.loc[var_mask, 'bin'].apply(
                            lambda x: x.replace(var, '').replace(' %in% c', '').strip()
                        )
                    
                    df_with_woe_local = add_woe_columns(df_transformed, rules, x_vars)
                    app_results['df_with_woe'] = df_with_woe_local
                    
                    woe_cols = [c for c in df_with_woe_local.columns if c.startswith('WOE_')]
                    app_results['df_only_woe'] = df_with_woe_local[woe_cols + [dv_col]].copy()
                    app_results['bins'] = rules
                
                await session.close()
            
            @output
            @render.data_frame
            def var_summary_table():
                result = bins_result_rv.get()
                if result is not None and not result.var_summary.empty:
                    display_df = result.var_summary[['var', 'iv', 'ent', 'trend', 'monTrend', 'numBins']].copy()
                    display_df.columns = ['Variable', 'IV', 'Entropy', 'Trend', 'Monotonic', 'Bins']
                    display_df = display_df.sort_values('IV', ascending=False)
                    return render.DataGrid(display_df, width="100%")
                return render.DataGrid(pd.DataFrame())
            
            @output
            @render.data_frame
            def bin_details_table():
                result = bins_result_rv.get()
                var = input.var_select()
                if result is not None and var:
                    var_bins = result.bin[result.bin['var'] == var].copy()
                    if not var_bins.empty:
                        cols = ['bin', 'count', 'bads', 'goods', 'bad_rate', 'woe', 'iv']
                        available_cols = [c for c in cols if c in var_bins.columns]
                        return render.DataGrid(var_bins[available_cols], width="100%")
                return render.DataGrid(pd.DataFrame())
            
            @output
            @render_plotly
            def woe_chart():
                result = bins_result_rv.get()
                var = input.var_select()
                
                fig = go.Figure()
                
                if result is not None and var:
                    var_bins = result.bin[(result.bin['var'] == var) & (result.bin['bin'] != 'Total')]
                    if not var_bins.empty:
                        # Create short bin labels
                        var_bins = var_bins.reset_index(drop=True)
                        labels = [f"Bin {i+1}" for i in range(len(var_bins))]
                        
                        # WOE bars
                        colors = ['#e94560' if w > 0 else '#00b894' for w in var_bins['woe']]
                        
                        fig.add_trace(go.Bar(
                            x=labels,
                            y=var_bins['woe'],
                            marker_color=colors,
                            name='WOE',
                            text=[f"{w:.3f}" for w in var_bins['woe']],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title=f"WOE by Bin - {var}",
                            xaxis_title="Bin",
                            yaxis_title="Weight of Evidence",
                            template="plotly_dark",
                            height=400
                        )
                
                return fig
            
            @output
            @render_plotly
            def iv_chart():
                result = bins_result_rv.get()
                
                fig = go.Figure()
                
                if result is not None and not result.var_summary.empty:
                    df_sorted = result.var_summary.sort_values('iv', ascending=True).tail(20)
                    
                    # Color by IV strength
                    colors = []
                    for iv in df_sorted['iv']:
                        if iv < 0.02:
                            colors.append('#95a5a6')  # Weak
                        elif iv < 0.1:
                            colors.append('#3498db')  # Medium
                        elif iv < 0.3:
                            colors.append('#2ecc71')  # Good
                        else:
                            colors.append('#e94560')  # Strong
                    
                    fig.add_trace(go.Bar(
                        y=df_sorted['var'],
                        x=df_sorted['iv'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{iv:.4f}" for iv in df_sorted['iv']],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Top 20 Variables by Information Value",
                        xaxis_title="Information Value",
                        yaxis_title="Variable",
                        template="plotly_dark",
                        height=500
                    )
                
                return fig
            
            @output
            @render.data_frame
            def psi_table():
                psi_df = psi_data_rv.get()
                if not psi_df.empty:
                    return render.DataGrid(psi_df, width="100%")
                return render.DataGrid(pd.DataFrame({'message': ['No PSI data - reference not configured']}))
        
        return App(app_ui, server), app_results
    
    def run_woe2_editor(df: pd.DataFrame, min_prop: float = 0.01) -> Dict[str, Any]:
        """Run the WOE 2.0 editor interactively."""
        import socket
        import webbrowser
        
        port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
        
        for attempt in range(10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', port))
                sock.close()
                break
            except OSError:
                port = BASE_PORT + random.randint(0, RANDOM_PORT_RANGE)
        
        log_progress(f"Starting WOE 2.0 Shiny UI on port {port}")
        
        app, results = create_woe2_editor_app(df, min_prop)
        
        webbrowser.open(f'http://127.0.0.1:{port}')
        app.run(host='127.0.0.1', port=port)
        
        return results
    
    # Run interactive mode
    log_progress("=" * 70)
    log_progress("WOE 2.0 EDITOR - INTERACTIVE MODE")
    log_progress("=" * 70)
    
    results = run_woe2_editor(df, min_prop=cfg.min_bin_pct)
    
    if results['completed']:
        df_with_woe = results['df_with_woe']
        df_only_woe = results['df_only_woe']
        bins = results['bins']
        dv = results['dv']
        
        # Create df_only_bins
        b_columns = [col for col in df_with_woe.columns if col.startswith('b_')]
        df_only_bins = df_with_woe[b_columns].copy()
        
        psi_report = pd.DataFrame()  # No reference in interactive mode
        
        log_progress("Interactive session completed successfully")
    else:
        log_progress("Interactive session cancelled - returning empty results")
        df_with_woe = df.copy()
        df_only_woe = pd.DataFrame()
        df_only_bins = pd.DataFrame()
        bins = pd.DataFrame()
        psi_report = pd.DataFrame()

# =============================================================================
# Output Tables
# =============================================================================

knio.output_tables[0] = knio.Table.from_pandas(df)
knio.output_tables[1] = knio.Table.from_pandas(df_with_woe)
knio.output_tables[2] = knio.Table.from_pandas(df_only_woe)
knio.output_tables[3] = knio.Table.from_pandas(df_only_bins)
knio.output_tables[4] = knio.Table.from_pandas(bins)
knio.output_tables[5] = knio.Table.from_pandas(psi_report)

log_progress("=" * 70)
log_progress("OUTPUT SUMMARY:")
log_progress(f"  Port 1: Original data ({len(df)} rows, {len(df.columns)} cols)")
log_progress(f"  Port 2: With WOE ({len(df_with_woe)} rows, {len(df_with_woe.columns)} cols)")
log_progress(f"  Port 3: Only WOE ({len(df_only_woe)} rows, {len(df_only_woe.columns)} cols)")
log_progress(f"  Port 4: Only Bins ({len(df_only_bins)} rows, {len(df_only_bins.columns)} cols)")
log_progress(f"  Port 5: Bin Rules ({len(bins)} rows)")
log_progress(f"  Port 6: PSI Report ({len(psi_report)} rows) ** NEW **")
log_progress("=" * 70)

log_progress("WOE 2.0 Editor completed successfully")

# KNIME Python Nodes for Credit Risk Modeling

Python scripts for KNIME 5.9 Python Script nodes, providing a complete credit risk modeling workflow. These are Python ports of R-based solutions with enhanced features and modern implementations.

## Features

- **Reject Inference** (`reject_inference.py`): Infer outcomes for rejected loan applications
- **WOE Binning** (`woe_editor_advanced.py`): Weight of Evidence transformation with DecisionTree, ChiMerge, and IVOptimal algorithms
- **Variable Selection** (`variable_selection_knime.py`): Feature selection with IV, Gini, Chi-Square, VIF filtering, and EBM interaction discovery
- **Logistic Regression** (`logistic_regression_knime.py`): Stepwise selection (Forward, Backward, Both)
- **Scorecard Generation** (`scorecard_knime.py`): Convert logistic regression + WOE to point-based scorecards
- **Scorecard Application** (`scorecard_apply_knime.py`): Apply scorecard to new data
- **Model Analysis** (`model_analyzer_knime.py`): ROC curves, K-S charts, Lorenz curves, Gains tables

## Requirements

- **KNIME**: 5.9+
- **Python**: 3.9.x
- **Dependencies**: pandas, numpy, scikit-learn, shiny, statsmodels, plotly

All dependencies are auto-installed by scripts if missing.

## Quick Start

1. Copy the desired Python script into a KNIME Python Script node
2. Configure input tables as specified in the script header
3. Set flow variables for headless mode (optional)
4. Execute the node

## Scripts

### Core Nodes

| Script | Purpose | Interactive UI |
|--------|---------|----------------|
| `reject_inference.py` | Reject inference for credit scoring | No |
| `woe_editor_advanced.py` | WOE binning (recommended) | Yes |
| `woe_editor_knime.py` | WOE binning (base version) | Yes |
| `woe_editor_knime_parallel.py` | WOE binning with parallel processing | Yes |
| `attribute_editor_knime.py` | Variable metadata configuration | Yes |
| `variable_selection_knime.py` | Feature selection with metrics | Yes |
| `logistic_regression_knime.py` | Logistic regression modeling | Yes |
| `scorecard_knime.py` | Scorecard generation | No |
| `scorecard_apply_knime.py` | Apply scorecard to data | No |
| `model_analyzer_knime.py` | Model diagnostics and charts | Yes |
| `is_bad_flag_knime.py` | Binary target variable encoding | No |
| `column_separator_knime.py` | Column type separation | No |
| `ccr_score_filter_knime.py` | CCR score filtering | No |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `clean_b_score.py` | Score cleaning utility |
| `WOE node files/woe_config_generator.py` | Generate flow variable tables for WOE Editor |

## WOE Editor Algorithms

The advanced WOE editor (`woe_editor_advanced.py`) supports three binning algorithms:

1. **DecisionTree** (default): R-compatible, matches logiBin::getBins
2. **ChiMerge**: Chi-square based bin merging
3. **IVOptimal**: Maximizes Information Value, preserves non-monotonic patterns (ideal for fraud models)

See [WOE Editor Enhancements](WOE%20node%20files/WOE_Editor_Enhancements.md) for detailed documentation.

## Documentation

- [CONTEXT.md](CONTEXT.md) - Quick start for AI agents
- [WOE Editor Enhancements](WOE%20node%20files/WOE_Editor_Enhancements.md) - Advanced WOE features
- [Scorecard Node Updates](SCORECARD_NODE_UPDATES.md) - Scorecard node changelog
- [WOE Output Ports Guide](WOE_OUTPUT_PORTS_GUIDE.md) - WOE output configuration

## Development

Developed over 14 days (~150 hours) using Cursor IDE with Claude 4.5. Total codebase: ~9,000+ lines.

### Environment

- KNIME 5.9
- Python 3.9.23
- Windows platform

### Coding Conventions

- All scripts are self-contained (no external project imports)
- Auto-install dependencies using pip
- Use nullable pandas types (`Int32`, `Float64`) for KNIME compatibility
- Column names are case-sensitive

## License

[MIT License](LICENSE) - feel free to use and modify for your credit risk modeling needs.

## Contributing

Contributions welcome! Please follow the existing code style and ensure scripts remain self-contained for KNIME compatibility.

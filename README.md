# KNIME Python Nodes for Credit Risk Modeling

Python scripts for KNIME 5.9 Python Script nodes, providing a complete credit risk modeling workflow. These are Python ports of R-based solutions with enhanced features and modern implementations.

## Features

- **Reject Inference**: Infer outcomes for rejected loan applications
- **WOE Binning**: Weight of Evidence transformation with DecisionTree, ChiMerge, and IVOptimal algorithms
- **Variable Selection**: Feature selection with IV, Gini, Chi-Square, VIF filtering, and EBM interaction discovery
- **Logistic Regression**: Stepwise selection (Forward, Backward, Both)
- **Scorecard Generation**: Convert logistic regression + WOE to point-based scorecards
- **Scorecard Application**: Apply scorecard to new data
- **Model Analysis**: ROC curves, K-S charts, Lorenz curves, Gains tables

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

## Project Structure

Each KNIME node has its own folder containing the Python script and related files (R references, documentation, sample data):

```
├── woe_editor/              # WOE binning (3 versions + docs + R reference)
├── variable_selection/      # Feature selection + R reference
├── logistic_regression/     # Logistic regression modeling
├── scorecard/               # Scorecard generation + guides
├── scorecard_apply/         # Apply scorecard + R reference
├── model_analyzer/          # Model diagnostics
├── attribute_editor/        # Variable metadata configuration
├── reject_inference/        # Reject inference
├── is_bad_flag/             # Binary target encoding
├── column_separator/        # Column type separation
├── ccr_score_filter/        # CCR score filtering
├── clean_b_score/           # Score cleaning utility
├── archive/                 # Obsolete/legacy files
├── .cursorrules             # AI assistant configuration
├── CONTEXT.md               # Quick start for AI agents
├── COMPREHENSIVE_DEVELOPMENT_LOG.md
└── README.md
```

## Core Nodes

| Folder | Script | Purpose | Interactive UI |
|--------|--------|---------|----------------|
| `woe_editor/` | `woe_editor_advanced.py` | WOE binning (recommended) | Yes |
| `woe_editor/` | `woe_editor_knime.py` | WOE binning (base version) | Yes |
| `woe_editor/` | `woe_editor_knime_parallel.py` | WOE binning with parallel processing | Yes |
| `variable_selection/` | `variable_selection_knime.py` | Feature selection with metrics | Yes |
| `logistic_regression/` | `logistic_regression_knime.py` | Logistic regression modeling | Yes |
| `scorecard/` | `scorecard_knime.py` | Scorecard generation | No |
| `scorecard_apply/` | `scorecard_apply_knime.py` | Apply scorecard to data | No |
| `model_analyzer/` | `model_analyzer_knime.py` | Model diagnostics and charts | Yes |
| `attribute_editor/` | `attribute_editor_knime.py` | Variable metadata configuration | Yes |
| `reject_inference/` | `reject_inference.py` | Reject inference for credit scoring | No |
| `is_bad_flag/` | `is_bad_flag_knime.py` | Binary target variable encoding | No |
| `column_separator/` | `column_separator_knime.py` | Column type separation | No |
| `ccr_score_filter/` | `ccr_score_filter_knime.py` | CCR score filtering | No |
| `clean_b_score/` | `clean_b_score.py` | Score cleaning utility | No |

## WOE Editor Algorithms

The advanced WOE editor (`woe_editor/woe_editor_advanced.py`) supports three binning algorithms:

1. **DecisionTree** (default): R-compatible, matches logiBin::getBins
2. **ChiMerge**: Chi-square based bin merging
3. **IVOptimal**: Maximizes Information Value, preserves non-monotonic patterns (ideal for fraud models)

See [WOE Editor Enhancements](woe_editor/WOE_Editor_Enhancements.md) for detailed documentation.

## Documentation

- [CONTEXT.md](CONTEXT.md) - Quick start for AI agents
- [WOE Editor Enhancements](woe_editor/WOE_Editor_Enhancements.md) - Advanced WOE features
- [Scorecard Node Updates](scorecard/SCORECARD_NODE_UPDATES.md) - Scorecard node changelog
- [WOE Output Ports Guide](woe_editor/WOE_OUTPUT_PORTS_GUIDE.md) - WOE output configuration

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

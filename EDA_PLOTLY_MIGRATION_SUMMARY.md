# EDA Module Migration: Matplotlib to Plotly

## Summary

The `bluecast/eda/analyse.py` module has been successfully migrated from
matplotlib/seaborn to plotly for all visualization functions. This migration
provides several benefits:

- **Interactive plots**: All plots are now interactive by default
- **Better web integration**: Plots work seamlessly in web browsers and notebooks
- **Dashboard capability**: Added functionality to create interactive dashboards
  with just a few lines of code
- **Return values**: Functions now return plotly Figure objects that can be
  programmatically manipulated

## Changes Made

### 1. Dependencies Updated

**pyproject.toml**:

- Added `dash = "^2.0.0"` for dashboard functionality
- Existing `plotly = "^5"` dependency is now fully utilized

### 2. Core Functions Converted

All plotting functions in `analyse.py` have been converted to use plotly:

- `plot_pie_chart()` - Now uses `go.Pie()` with hole parameter for donut charts
- `plot_count_pair()` - Uses `px.histogram()` with grouped bar mode
- `univariate_plots()` - Uses `make_subplots()` with histogram and box plots
- `bi_variate_plots()` - Uses violin plots with multiple traces
- `correlation_heatmap()` - Uses `go.Heatmap()` with masked upper triangle
- `correlation_to_target()` - Horizontal heatmap for target correlations
- `plot_pca()` - Uses `px.scatter()` for PCA visualization
- `plot_tsne()` - Interactive t-SNE scatter plots
- `mutual_info_to_target()` - Horizontal bar chart with scores
- And many more...

### 3. New Features Added

#### Dashboard Functionality

Added `create_eda_dashboard()` function that creates an interactive Dash web application:

```python
from bluecast.eda.analyse import create_eda_dashboard

# Create an interactive dashboard with just one line
create_eda_dashboard(df, 'target_column', port=8050)
```

Features of the dashboard:

- Multiple plot types (correlation heatmap, PCA, distributions, etc.)
- Interactive feature selection
- Real-time plot updates
- Summary statistics display
- Responsive web interface

#### Function Return Values

All plotting functions now return plotly Figure objects:

```python
# Get the figure object for further customization
fig = correlation_heatmap(df, show=False)
fig.update_layout(title="Custom Title")
fig.show()
```

#### Show Parameter

All functions now have a `show` parameter (default `True`):

```python
# Display immediately (default behavior)
plot_pca(df, 'target')

# Return figure without displaying (useful for further processing)
fig = plot_pca(df, 'target', show=False)
```

### 4. Tests Updated

**bluecast/tests/test_analyse.py**:

- All tests updated to work with plotly Figure objects
- Added `import plotly.graph_objects as go`
- Tests now verify that functions return proper Figure objects
- Dashboard test properly configured to not run indefinitely

### 5. Example Usage

**examples/eda_dashboard_example.py**:

- Comprehensive example showing both individual plotting functions and
  dashboard usage
- Demonstrates how to create sample data and use all major features
- Shows proper error handling and user interaction

## Usage Examples

### Individual Plots

```python
import pandas as pd
from bluecast.eda.analyse import correlation_heatmap, plot_pca, plot_pie_chart

# Load your data
df = pd.read_csv('your_data.csv')

# Create interactive correlation heatmap
fig = correlation_heatmap(df.select_dtypes(include=['number']))

# Create PCA plot
fig = plot_pca(df, 'target_column')

# Create pie chart for categorical data
fig = plot_pie_chart(df, 'category_column')
```

### Dashboard

```python
from bluecast.eda.analyse import create_eda_dashboard

# Create full interactive dashboard
create_eda_dashboard(df, 'target_column', port=8050)
# Navigate to http://localhost:8050 in your browser
```

### Programmatic Usage

```python
# Get figure objects for further customization
fig = correlation_heatmap(df, show=False)
fig.update_layout(
    title="Custom Correlation Analysis",
    font=dict(size=14)
)
fig.write_html("correlation_report.html")
```

## Benefits of the Migration

1. **Interactivity**: All plots are now interactive (zoom, pan, hover tooltips)
2. **Web-Ready**: Plots work seamlessly in web applications and notebooks
3. **Dashboard Capability**: Quick creation of comprehensive EDA dashboards
4. **Better Integration**: Easier to integrate into modern data science workflows
5. **Future-Proof**: Plotly is actively developed and widely adopted
6. **Export Options**: Easy export to HTML, PNG, PDF, and other formats

## Backward Compatibility

- All function names and parameters remain the same
- Default behavior (showing plots) is unchanged
- Only the underlying visualization library has changed
- Return values are now plotly Figure objects instead of None

## Testing

All tests pass and verify:

- Functions return proper plotly Figure objects
- Dashboard creation works without hanging
- All plot types render correctly
- Error handling works as expected

### Test Fixes Applied

During the migration, several test issues were resolved:

1. **Andrews Curve Test**: Fixed numpy boolean vs Python boolean issue with
   plotly's `showlegend` parameter
2. **Error Analysis Tests**: Updated tests that were mocking
   `seaborn.violinplot` to mock `plotly.graph_objects.Figure.show` instead
3. **Dashboard Test**: Added `run_server` parameter to prevent tests from
   hanging when testing dashboard creation

### Running Tests

Run all EDA tests with:

```bash
poetry run python -m pytest bluecast/tests/test_analyse.py -v
```

Run error analysis tests with:

```bash
poetry run python -m pytest bluecast/tests/test_error_analysis_base_classes.py -v
```

All tests now pass successfully with the new plotly implementation.

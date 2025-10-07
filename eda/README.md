# Exploratory Data Analysis (EDA) Project

This project is designed to facilitate Exploratory Data Analysis (EDA) through a structured set of modules. Each module focuses on different aspects of data analysis, providing functions and tools to help users understand their datasets better.

## Project Structure

The project is organized into several directories, each containing specific types of analyses:

- **data_quality**: Functions for assessing and improving the quality of data.
  - `missing_values.py`: Identify and handle missing values.
  - `duplicates.py`: Detect and remove duplicate entries.
  - `outliers.py`: Identify and manage outliers.
  - `data_types.py`: Check and convert data types.

- **univariate**: Analysis of single variables.
  - `distributions.py`: Visualize and analyze distributions.
  - `summary_stats.py`: Calculate and summarize descriptive statistics.
  - `categorical_analysis.py`: Analyze categorical variables.

- **bivariate**: Analysis of relationships between two variables.
  - `correlations.py`: Compute and visualize correlations.
  - `scatter_plots.py`: Create scatter plots.
  - `cross_tabulations.py`: Analyze relationships between categorical variables.

- **multivariate**: Analysis involving multiple variables.
  - `pca.py`: Perform Principal Component Analysis (PCA).
  - `clustering.py`: Apply clustering algorithms.
  - `feature_interactions.py`: Analyze interactions between features.

- **temporal**: Analysis of time-related data.
  - `time_series.py`: Analyze time series data.
  - `seasonality.py`: Identify and analyze seasonal patterns.
  - `trends.py`: Detect and visualize trends.

- **visualization**: Tools for visualizing data.
  - `plots.py`: Create various types of plots.
  - `dashboards.py`: Create interactive dashboards.
  - `reports.py`: Generate summary reports.

- **utils**: Utility functions and configurations.
  - `helpers.py`: General utility functions.
  - `config.py`: Configuration settings for the project.
  - `constants.py`: Define constants used throughout the project.

## Usage

To use the EDA modules, import the relevant functions from the respective files. Each module is designed to be self-contained, allowing for easy integration into your data analysis workflow.

## Contribution

Contributions to enhance the functionality of this project are welcome. Please follow the standard procedures for contributing to open-source projects.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
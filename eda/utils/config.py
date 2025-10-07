# Configuration settings for the EDA project

# File paths
DATA_PATH = 'path/to/data'
OUTPUT_PATH = 'path/to/output'
FIGURES_PATH = 'path/to/figures'

# Parameters
MISSING_VALUE_THRESHOLD = 0.2  # Threshold for handling missing values
OUTLIER_THRESHOLD = 1.5  # IQR multiplier for outlier detection
DEFAULT_PCA_COMPONENTS = 2  # Default number of components for PCA

# Visualization settings
DEFAULT_FIGURE_SIZE = (10, 6)  # Default figure size for plots
DEFAULT_COLOR_MAP = 'viridis'  # Default color map for visualizations

# Logging settings
LOGGING_LEVEL = 'INFO'  # Default logging level
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'  # Default logging format
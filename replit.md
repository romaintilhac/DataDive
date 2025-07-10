# Geochemical Data Analysis Platform

## Overview

This is a Streamlit-based web application for comprehensive geochemical data analysis. The application provides tools for data upload, validation, processing, visualization, and statistical analysis of geochemical datasets. It's designed for geologists and geochemists to analyze rock samples and their chemical compositions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Structure**: Multi-page application with navigation sidebar
- **Layout**: Wide layout with expandable sidebar for optimal data visualization
- **State Management**: Streamlit session state for data persistence across pages

### Backend Architecture
- **Language**: Python
- **Architecture Pattern**: Utility-based modular design
- **Core Components**:
  - Data processing utilities
  - Geochemical calculation engine
  - Visualization utilities
  - Constants and reference values

### Data Storage
- **Primary Storage**: In-memory using Pandas DataFrames
- **Session Storage**: Streamlit session state for temporary data persistence
- **File Support**: CSV and Excel file formats for data import

## Key Components

### 1. Main Application (`app.py`)
- Entry point with application configuration
- Session state initialization
- Data status indicators
- Navigation overview

### 2. Data Upload Module (`pages/1_Data_Upload.py`)
- File upload interface supporting CSV and Excel
- Data validation and quality checks
- Multi-file processing capabilities
- Interactive data preview

### 3. Data Processing Module (`pages/2_Data_Processing.py`)
- Geochemical calculations and derived parameters
- Data cleaning and transformation
- Isotope ratio calculations
- Normalization procedures

### 4. Visualization Module (`pages/3_Visualizations.py`)
- Interactive plotting with Plotly
- Geochemical classification diagrams
- Statistical visualizations
- Custom plot generation

### 5. Statistical Analysis Module (`pages/4_Statistical_Analysis.py`)
- Correlation analysis
- Principal Component Analysis (PCA)
- Cluster analysis
- Statistical summaries

### 6. Utility Classes
- **DataProcessor**: File loading and validation
- **GeochemicalCalculator**: Chemical calculations and ratios
- **GeochemicalPlotter**: Visualization generation
- **Constants**: Reference values for normalization

## Data Flow

1. **Data Input**: Users upload CSV/Excel files through the web interface
2. **Validation**: Data structure and quality checks are performed
3. **Processing**: Optional calculations and transformations applied
4. **Analysis**: Statistical analysis and visualization generation
5. **Output**: Interactive plots and statistical results displayed

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scipy**: Statistical analysis
- **Scikit-learn**: Machine learning algorithms

### Visualization Libraries
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

### File Processing
- **openpyxl**: Excel file handling (via pandas)

## Deployment Strategy

### Local Development
- Run using `streamlit run app.py`
- No database setup required
- All data processing in-memory

### Production Deployment
- Streamlit Cloud or similar platform
- No persistent storage configured
- Session-based data handling

### Configuration
- Page configuration set for wide layout
- Expandable sidebar for navigation
- Responsive design for data visualization

## Technical Considerations

### Performance
- In-memory data processing suitable for typical geochemical datasets
- Efficient pandas operations for data manipulation
- Plotly for responsive interactive visualizations

### Scalability
- Current architecture supports datasets up to memory limits
- Session state management for multi-user scenarios
- Modular design allows for easy feature expansion

### Data Security
- No persistent data storage
- Session-based data handling
- File upload validation and error handling

### Error Handling
- Comprehensive validation for data uploads
- Graceful error handling for file processing
- User-friendly error messages and warnings
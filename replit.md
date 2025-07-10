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
- Enhanced geochemical calculations and derived parameters
- Data cleaning and transformation
- Advanced isotope ratio calculations with age corrections
- Comprehensive error propagation
- Normalization procedures
- Enhanced column reordering with intelligent suffix handling

### 4. Visualization Module (`pages/3_Visualizations.py`)
- Interactive plotting with Plotly
- Geochemical classification diagrams
- Enhanced REE spider diagrams with proper formatting
- Statistical visualizations
- Custom plot generation

### 5. Statistical Analysis Module (`pages/4_Statistical_Analysis.py`)
- Correlation analysis
- Principal Component Analysis (PCA)
- Cluster analysis
- Statistical summaries

### 6. Enhanced Utility Classes
- **DataProcessor**: Enhanced file loading, validation, and intelligent merging
- **GeochemicalCalculator**: Advanced chemical calculations and ratios
- **GeochemicalPlotter**: Enhanced visualization generation
- **Constants**: Comprehensive reference values with accurate isotope parameters
- **GeochemicalFunctions**: Advanced calculation functions with safe assignment
- **PlotFormatting**: Enhanced label formatting and styling utilities

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

### Specialized Geochemical Libraries
- **pyrolite**: Advanced geochemical analysis and normalization capabilities

## Recent Enhancements (December 2024)

### Enhanced Constants and References
- Updated isotope decay constants with more accurate values (Begemann et al. 2001, Scherer et al. 2001)
- Added CHUR values from Bouvier et al. 2008
- Included depleted mantle values from Griffin et al. 2000
- Added Hf-Nd mantle array parameters for ΔεHf calculations
- Comprehensive molecular weight database for major elements

### Advanced Calculation Functions
- Safe assignment functionality to prevent column overwrites
- Comprehensive isotope ratio calculations with age corrections
- Enhanced error propagation for initial isotope ratios
- Automatic relative error handling for missing uncertainties
- Delta epsilon Hf calculations using proper mantle array corrections

### Enhanced Data Processing
- Intelligent column reordering with suffix handling (_err, _meta, _conflict, _calc)
- Enhanced merge capabilities with conflict detection
- Improved data validation and cleaning procedures
- Better handling of metadata and duplicate resolution

### Improved Visualization Features
- Enhanced plot label formatting with LaTeX-style subscripts
- Consistent color mapping for elements and lithologies
- Better REE spider diagram formatting
- Improved axis labels and scientific notation handling

### Multi-File Loading and Global Database Integration (January 2025)

#### Multi-File Loading System
- MultiFileLoader class for handling multiple geochemical data files
- Sample-based catalogue logic for combining datasets
- Automatic catalogue file detection based on metadata columns
- Intelligent conflict resolution with multiple strategies (keep_first, keep_last, average)
- Sample consistency validation across files
- Detailed merge logging and overlap analysis

#### Global Database Comparison
- GlobalDatabase class with reference datasets (MORB, OIB, Arc Basalts, Continental Crust)
- Comprehensive comparison tools with similarity metrics
- Range overlap analysis and statistical comparisons
- Database search and filtering capabilities
- Custom database integration support

#### Enhanced Data Upload Interface
- Three-mode upload system: Single File, Multi-File Combination, Global Database Comparison
- Interactive file combination settings with catalogue file selection
- Real-time validation and sample overlap analysis
- Comprehensive merge summary and logging
- Database preview and comparison interface

#### Technical Features
- Robust error handling and user feedback
- Automatic column reordering and conflict detection
- Priority-based file merging with customizable resolution
- Statistical comparison metrics and visualization
- Memory-efficient data processing for large datasets

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
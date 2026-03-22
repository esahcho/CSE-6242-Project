# Forecasting Residential Solar Production and ROI Across U.S. Regions

## Project Overview
A machine learning tool that forecasts solar production across U.S. regions using NREL data and estimates return on investment (ROI) for residential solar installations. The project combines time series forecasting models with financial analysis to help homeowners make informed decisions about solar panel adoption.

**Team 67**: Benjamin Martinez, Coleman Treadwell, Elizabeth Haenel, Erin Sinah Cho, Shu Hang Leung, Tan Nguyen

## Project Structure
```
CSE-6242-Project/
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── Project Proposal                        # Original project proposal
├── time_series_model.py                    # Main time series forecasting module
├── notebooks/
│   ├── load_data.ipynb                    # Data loading from NREL API
│   ├── clean_data_eda.ipynb               # Data cleaning and EDA
│   ├── irradiance_2021_2024.parquet       # Processed irradiance data
│   └── nsrdb_eda.png                      # EDA visualization
└── .git/                                   # Git repository
```

## Setup & Installation

### Prerequisites
- Python 3.12+
- Git

### Installation Steps
1. Clone the repository and navigate to the project directory
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name solar-env --display-name "Solar Project"
   ```

## Key Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **tensorflow**: Deep learning models (LSTM, GRU)
- **matplotlib & seaborn**: Data visualization
- **scipy**: Statistical functions
- **requests**: API calls to NREL
- **jupyter**: Interactive notebooks

## Data Sources
- **NREL National Solar Radiation Database (NSRDB)**: Hourly solar irradiance data (GHI, DNI, DHI)
- Years: 2021-2024
- Format: Parquet file (`irradiance_2021_2024.parquet`)

## Project Workflow

### Phase 1: Data Collection & Processing
- `load_data.ipynb`: Retrieves hourly solar irradiance data from NREL API
- `clean_data_eda.ipynb`: Data cleaning, feature engineering, and exploratory analysis

### Phase 2: Time Series Forecasting
- `time_series_model.py`: Core forecasting module (in development)
- Models to implement: LSTM, GRU, ARIMA, Prophet
- Evaluation metrics: MSE, MAE, MAPE

### Phase 3: ROI Analysis & Visualization
- Financial modeling with cost benchmarks from NREL
- Interactive visualization (Leaflet/OpenStreetMap)
- Payback period estimation

## Key Metrics & Success Criteria
- **Prediction Accuracy**: MSE, MAE, MAPE on holdout test set
- **Regional Coverage**: Multiple U.S. regions with varying climate conditions
- **ROI Validation**: Compare estimates with published benchmarks

## Branch Information
- **Current Branch**: `coleman_time_series` (time series model development)
- **Main Branch**: `main` (production-ready code)

## Next Steps
1. Develop baseline time series models
2. Compare model performance across regions
3. Implement ROI calculation framework
4. Build interactive web visualization
5. Validate against published benchmarks

---
*Last Updated: March 22, 2026*
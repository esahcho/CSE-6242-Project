# Forecasting Residential Solar Production and ROI Across U.S. Regions

## Project Overview
This project is an interactive solar energy decision-support tool that forecasts solar irradiance and estimates the financial return of residential rooftop solar panel installations across U.S. regions. It uses machine learning–based time series models trained on historical solar irradiance data from the National Renewable Energy Laboratory (NREL) to predict future Global Horizontal Irradiance (GHI). 

These predictions are converted into estimated photovoltaic energy production using standard assumptions such as system size, performance ratio, and degradation rate. The resulting energy estimates are used to compute long-term financial outcomes including cost savings, return on investment (ROI), and payback period. The system is deployed as an interactive Streamlit dashboard.

**Team 67**: Benjamin Martinez, Coleman Treadwell, Elizabeth Haenel, Erin Sinah Cho, Shu Hang Leung, Tan Nguyen

## EXECUTION

### Run the Streamlit application
 https://cse-6242-project-team-67.streamlit.app/
### How to use the tool:
1. Open the Streamlit URL above.
2. On the rightside control panel, select:
   - Installation Size
   - Region
   - Monthly Electric Bill ($)
   - System Cost ($)
4. View outputs:
   - ROI & Financial Analysis
   - Panel Count
   - Selected Region
   - 25-Year Utility Cost
   - 25-Year Solar Cost
   - Estimated Net Savings
5. Explore visualizations including:
   - Regional map
   - 25-year ROI Analysis
   - Annual forecasted irradiance

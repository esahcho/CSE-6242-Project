# Forecasting Residential Solar Production and ROI Across U.S. Regions

## Project Overview
This project is an interactive solar energy decision-support tool that forecasts solar irradiance and estimates the financial return of residential rooftop solar panel installations across regions in the United States. It integrates machine learning–based time series forecasting with financial modeling to help users evaluate solar feasibility, expected energy production, cost savings, and payback period.

The system uses historical solar irradiance data from the National Renewable Energy Laboratory (NREL) to train forecasting models that predict future Global Horizontal Irradiance (GHI). These forecasts are then converted into estimated photovoltaic (PV) energy production using industry-standard assumptions such as panel wattage, performance ratio, and degradation rate. Financial analysis combines projected energy output with regional electricity prices and installation cost benchmarks to compute return on investment (ROI) and breakeven time.

The final output is delivered through an interactive Streamlit dashboard that allows users to select a U.S. region, input system size and electricity usage parameters, and visualize energy production and financial outcomes through choropleth maps and burndown plots.

**Team 67**: Benjamin Martinez, Coleman Treadwell, Elizabeth Haenel, Erin Sinah Cho, Shu Hang Leung, Tan Nguyen

## INSTALLATION


## EXECUTION

### Run the Streamlit application
 ```bash
   streamlit run app.py
   ```
### How to use the tool:
1. Open the local Streamlit URL displayed in the terminal.
2. Select a U.S. region from the interactive choropleth map.
3. Enter:
   - City
   - Size of Solar Panels
   - 
4. View outputs:
   - Forecasted solar energy production
   - Estimated savings over time
   - Payback period (break-even point)
5. Explore visualizations including:
   - Regional solar potential map
   - Financial burndown chart (cost vs savings over time)

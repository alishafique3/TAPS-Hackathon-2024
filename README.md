# TAPS-Hackathon-2024

This project provides an interactive dashboard for agricultural data analysis using Streamlit, featuring advanced visualization and AI-driven insights. It incorporates various datasets and analytical tools for informed decision-making in precision agriculture.

## Datasets
1. **Ceres Imager**
2. **Climate and Weather**
3. **Electrical Conductivity**
4. **TAPS Management**
5. **Neutron Probe Dataset**
6. **Plot Boundaries**

## Streamlit Parameters
1. **Nitrogen Fertilizer:** Datewise
2. **Irrigation Applied:** Datewise
3. **Electrical Conductivity and pH:** Depth-wise
4. **Volumetric Water Content:** CWR, ER, Planting Date, ET₀
5. **Weather and Climate:** Climatic Parameters
6. **NDVI_Ceres:** Datewise
7. **Crop Water Requirement Analysis:** Datewise, Climate Dates, Planting and Growth Stage-wise

## Data Preprocessing
1. **Data Filtering:** Dropping rows, Computing statistics while ignoring NA
2. **Data Merging:** By TRT_ID, Plot_ID, Block_ID
3. **Feature Engineering:** Creating new variables for data analysis
4. **Spatial Averaging:** Taking means for plotting and data analysis
5. **Spatial Data Enrichment:** Adding spatial information to non-spatial data

## Dashboard Results
1. **Nitrogen Fertilizer:** Treatments, Hybrids, Spatial and Box Plots + AI
2. **Irrigation Applied:** Treatments, Hybrids, Spatial and Box Plots + AI
3. **Electrical Conductivity and pH:** Depthwise, Spatial and Box plots + AI
4. **Volumetric Water Content:** Depthwise, Spatial and Box Plots + AI
5. **Weather and Climate:** Values of Variables like Temperature, Precipitation + AI
6. **NDVI_Ceres:** Spatial and Box Plots in time series + AI
7. **Crop Water Requirement:** Customized Relationship between Variables + AI

## A.I WildCat
1. **Statistical Analysis:** Mean, Maximum, Minimum values
2. **Trend Analysis:** Data Analysis using timeseries
3. **Expert Opinion:** Provides expert insights based on input parameters

---

This project aims to enhance agricultural productivity through data-driven insights, making it easier to monitor and manage key agricultural variables. 

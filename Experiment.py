#######################
# Import libraries

import geopandas as gpd
import pandas as pd
import os
import glob
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from rasterio.mask import mask
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from openai import OpenAI
import openai
import os
import time
import plotly.graph_objects as go

#######################

# %% Page Configurations

# Page configuration
st.set_page_config(
    page_title="SustainAqua Solutions Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded")

# %% Loading the data variables

# Path of my work directory
wd = "C:/Users/Apollo/Desktop/TAPS/datasets"

# Loading the plot boundaries shapefile from the work directory and naming it as plots_boundary
plots_boundary = gpd.read_file(os.path.join(wd, "2024_Colby_TAPS_Harvest_Area.shp"))

# Loading the 2024_TAPS_summary_stats dataset

TAPS_ss = pd.read_excel(os.path.join(wd, "2024_TAPS_summary_stats.xlsx"))

# Loading the KSU TAPS Neutron Tube Readings dataset

TAPS_np = pd.read_excel(os.path.join(wd, "24 KSU TAPS Neutron Tube Readings_VWC.xlsx"), header= 2)

# Loading the 2024_TAPS_Management dataset sheet 1

TAPS_mgt_Hybrids = pd.read_excel(os.path.join(wd, "2024_TAPS_management.xlsx"), sheet_name=0, header=1)

# Loading the 2024_TAPS_Management dataset sheet 2

TAPS_mgt_nitrogen = pd.read_excel(os.path.join(wd, "2024_TAPS_management.xlsx"), sheet_name=1, header=2)

# Loading the 2024_TAPS_Management dataset sheet 3
TAPS_mgt_Irr = pd.read_excel(os.path.join(wd, "2024_TAPS_management.xlsx"), sheet_name=2, header=1)

# %% Renaming the column names of TAPS summary stats dataset and Nitrogen probe

#Renaming the Plot ID as the Treatment ID to avoid the confusions for TAPS summary stats dataset
TAPS_ss.rename(columns={'Plot ID': 'TRT_ID'}, inplace= True)

# Lets do it for TAPS management dataset sheet 1
# First adding the 35 Treatment
# Copying the row with the index number 29 to the new row added as it has the same hybrid information 
#with some variation in the plants planted and also modifying the value based on the TAPS_mgt infromation doc file
TAPS_mgt_Hybrids.loc[34] = TAPS_mgt_Hybrids.loc[29]
TAPS_mgt_Hybrids.loc[34, TAPS_mgt_Hybrids.columns[[0, 3]]] = [35, 28000]
#Renaming the IDs as the TRT_ID and plants/ac as the Seeding rate
TAPS_mgt_Hybrids = TAPS_mgt_Hybrids.rename(columns={TAPS_mgt_Hybrids.columns[0]: 'TRT_ID', TAPS_mgt_Hybrids.columns[3]: 'Seeding_rate'})

# Lets do it for TAPS management dataset sheet 2
TAPS_mgt_nitrogen = TAPS_mgt_nitrogen.rename(columns={TAPS_mgt_nitrogen.columns[0]: 'TRT_ID', TAPS_mgt_nitrogen.columns[2]: 'At_plant_Variable', TAPS_mgt_nitrogen.columns[8]: 'Total'})

# Lets do it for TAPS management dataset sheet 3
TAPS_mgt_Irr = TAPS_mgt_Irr.rename(columns={TAPS_mgt_Irr.columns[0]: 'TRT_ID'})

# Lets do it for TAPS_np
#Renaming the Plot # as the TRT_ID and Block # to Block_ID to avoid the confusions
TAPS_np.rename(columns={'Plot #': 'Plot_ID', 'Block #': 'Block_ID'}, inplace= True)


# %% Cleaning the datasets

# Making the columns of the TAPS_np dataframe as shallow VWC and deep VWC in such a way by taking the mean of 6 to 30 inches 
#and writing the value in shallow VWC column and taking the mean of 42 to 114 inches and writing the values in deep VWC column
TAPS_np1 = TAPS_np
# Summarising the shallow Volumetric content depth values
TAPS_np1['shallow_VWC_avg'] = TAPS_np1.iloc[:, 3:5].mean(axis = 1, skipna = True)
# Summarising the deep Volumetric content depth values
TAPS_np1['deep_VWC_avg'] = TAPS_np1.iloc[:, 6:12].mean(axis = 1, skipna = True)
#Selecting the relevant columns which contain the above average values
TAPS_np1 = TAPS_np1.iloc[:, :3].join(TAPS_np1[['shallow_VWC_avg', 'deep_VWC_avg']])
# Joining the plots boundary with the TAPS_ss
TAPS_pb_ss = plots_boundary.merge(TAPS_ss, on='TRT_ID', how='left', suffixes=('_pb', '_ss'))

# Selecting relevant columns
TAPS_pb_ss1 = TAPS_pb_ss.iloc[:, :5].join(TAPS_pb_ss[['EC shallow avg', 'EC shallow sd', 'EC deep avg', 'EC deep sd', 'pH avg', 'pH sd']])

#Joining the TAPS_pb_ss1 with the TAPS_mgt_Hybrids

TAPS_pb_ss2 = TAPS_pb_ss1.merge(TAPS_mgt_Hybrids, on='TRT_ID', how='left')

# %% Variables created for plotting the data for Nitrogen Fertilizer

gdf_pb_ss2 = TAPS_pb_ss2  

# Joining the plots boundary with the TAPS_mgt_Hybrids
TAPS_pb_mgt_hb = plots_boundary.merge(TAPS_mgt_Hybrids, on='TRT_ID', how='left')

# Joining it with the Nitrogen data 
TAPS_pb_mgt_hb_N = TAPS_pb_mgt_hb.merge(TAPS_mgt_nitrogen, on = 'TRT_ID', how='left')

#Creating the variable gdf1 which contains the plots boundary, Hybrid, and the Nitrogen Fertilizer Information    
gdf1 = TAPS_pb_mgt_hb_N
gdf1.iloc[:, 11] = gdf1.iloc[:, 11].astype("float64") # Toconver object data type to float data type

nitrogen_date_columns = gdf1.columns[8:16]

# %% Variables created for  plotting the Irrigation applied
# Joining the plots boundary with the TAPS_mgt_Hybrids
TAPS_pb_mgt_hb = plots_boundary.merge(TAPS_mgt_Hybrids, on='TRT_ID', how='left')

# Joining it with the Irrigation data 
TAPS_pb_mgt_hb_I = TAPS_pb_mgt_hb.merge(TAPS_mgt_Irr, on = 'TRT_ID', how='left')

#gdf2 variable used for plotting the Irrigation data
gdf2 = TAPS_pb_mgt_hb_I

#Getting the Date column names
irrigation_date_columns = gdf2.columns[8:29]


# %%Data of VWC

#Joining plots boundary with neutron probe dataset 
TAPS_pb_np = plots_boundary.merge(TAPS_np, on='Plot_ID', how='left')

# Joining TAPS_pb_np with the TAPS_mgt_Hybrids
TAPS_pb_np_hb = TAPS_pb_np.merge(TAPS_mgt_Hybrids, on='TRT_ID', how='left')

# gdf3 variable created for copying the VWC data
gdf3 = TAPS_pb_np_hb
# Assuming gdf3 is your DataFrame and 'Date_x' is in datetime format
# Convert 'Date_x' to datetime format if not already
gdf3['Date_x'] = pd.to_datetime(gdf3['Date_x'])

# Define a function to group dates with a one-day difference
def group_dates(date_series):
    date_groups = {
        "2024-06-10": "10-11 June",
        "2024-06-11": "10-11 June",
        "2024-07-11": "11-12 July",
        "2024-07-12": "11-12 July",
        "2024-08-01": "1-2 Aug",
        "2024-08-02": "1-2 Aug",
        "2024-08-19": "19-20 Aug",
        "2024-08-20": "19-20 Aug",
        "2024-09-09": "9-10 Sept",
        "2024-09-10": "9-10 Sept"
    }
    # Extract the date part and convert it to string in the format yyyy-mm-dd for mapping
    return date_series.dt.strftime("%Y-%m-%d").map(date_groups)

# Apply the grouping function to create the Group_date column
gdf3['Group_date'] = group_dates(gdf3['Date_x'])

# Display the DataFrame to check the new Group_date column
print(gdf3[['Date_x', 'Group_date']])

#gdf4 variable used for the plots
gdf4 = gdf3

# Define depth columns for volumetric water content up to 90 inches
depth_columns = [6, 18, 30, 42, 54, 66, 78, 90]

# Define a color map for volumetric water content levels
cmap = cm.Blues  # Customize the color map for water content visualization

# %%Weather and Climate inputs
wd_weather_climate = f"{wd}/Climate and weather data"

climate = f"{wd_weather_climate}/colby_climate_1990_2019.csv"
climate90_19 = pd.read_csv(climate)
climate90_19_copy = climate90_19.copy()

# Convert 'time' column to datetime format
climate90_19_copy['time'] = pd.to_datetime(climate90_19_copy['time'])

# Extract year from 'time' column and calculate average precipitation per year
climate90_19_copy['year'] = climate90_19_copy['time'].dt.year
annual_precipitation = climate90_19_copy.groupby('year')['pr'].mean()

#calculate average evapotranspiration per year
annual_eto = climate90_19_copy.groupby('year')['eto'].mean()
#calculate average mean minimum temperature per year
annual_tmmn = climate90_19_copy.groupby('year')['tmmn'].mean()
#calculate average mean maximum temperature per year
annual_tmmx = climate90_19_copy.groupby('year')['tmmx'].mean()
#Calculate average relative humidity per year
annual_rmin = climate90_19_copy.groupby('year')['rmin'].mean()
#average mean maximum relative humidity per year
annual_rmax = climate90_19_copy.groupby('year')['rmax'].mean()
#average wind speed per year
annual_vs = climate90_19_copy.groupby('year')['vs'].mean()
#average solar radiation per year
annual_srad = climate90_19_copy.groupby('year')['srad'].mean()

# %% Data of NDVI

plots_boundary1 = plots_boundary.copy()

#Merging the plots_boundary1 with the TAPS_mgt_Hybrids

plots_boundary1 = plots_boundary1.merge(TAPS_mgt_Hybrids, on='TRT_ID', how='left', suffixes=('_pb', '_ss'))


# Loading the NDVI dataset from CERES folder
wd_Ceres_NDVI = f"{wd}/Ceres_NDVI/*corn NDVI.tif"


# Initialize a list of NDVI dates to store NDVI data
ndvi_dates = []

# Get all files ending with "corn NDVI.tif"
file_list = sorted(glob.glob(wd_Ceres_NDVI))

# Loop through each NDVI file
for file in file_list:
    # Extract the date part from the filename (assuming it's in the format yyyy-mm-dd)
    base_name = os.path.basename(file)
    date_str = base_name.split()[0]  # Extract the date part
    ndvi_dates.append(date_str)       # Store the date for creating the column later

    # Open the .tif file and mask with the plot boundaries
    with rasterio.open(file) as dataset:
        # Loop over each polygon in plots_boundary1
        ndvi_means = []  # List to store NDVI means for each plot
        for geom in plots_boundary1.geometry:
            try:
                # Mask the dataset with the current geometry
                out_image, out_transform = mask(dataset, [geom], crop=True)
                out_image = out_image[0]  # Select the first band
                
                # Calculate mean NDVI, ignoring NaNs
                mean_ndvi = np.nanmean(out_image[out_image > 0])  # Filter out no-data values (assumed 0 or negative)
                ndvi_means.append(mean_ndvi)
            except Exception as e:
                print(f"Error processing geometry: {e}")
                ndvi_means.append(np.nan)  # Append NaN if error occurs

        # Add the NDVI means as a new column in plots_boundary1 with the date as the column name
        plots_boundary1[f"NDVI_{date_str}"] = ndvi_means

# Display a sample of the GeoDataFrame with NDVI data added
print(plots_boundary1.head())

# Save the updated plots_boundary1 with NDVI data to a new shapefile or GeoJSON
output_fp = r"C:\Users\Apollo\Desktop\TAPS\datasets\plots_boundary1_with_NDVI.shp"
plots_boundary1.to_file(output_fp)

print(f"Extracted NDVI values and saved to {output_fp}.")


# Get a list of NDVI columns in plots_boundary1 (assuming it has been previously loaded)
ndvi_columns = plots_boundary1.columns[8:14]
# Get NDVI columns and create a dictionary with cleaned date strings
ndvi_date_map = {col.replace("NDVI_", ""): col for col in ndvi_columns}

#%% Crop water requirement data

#Uploading the Mesonet data
colby_mesonet = pd.read_csv(f"{wd_weather_climate}/colby_station_kansas_mesonet.csv")
# Calculating ET0 using Hargreaves Method
# For that Ra is used in mm/day
# SRAVG is converted to Ra by First dividing SRAVG by 0.7 to adjust transmissivity and after that converting W/m2
#to mm/day by multiplying the value 0.03527 (correction factor)
colby_mesonet['ET0'] = 0.0023 * (colby_mesonet['TEMP2MAVG'] + 17.8) * (np.sqrt(colby_mesonet['TEMP2MMAX'] - colby_mesonet['TEMP2MMIN'])) * colby_mesonet['SRAVG'] * 0.03527 * 1.4286

# Creating another variable named gdf5 is equal to TAPS_mgt_hybrids
gdf5 = TAPS_mgt_Hybrids.copy()
# Add new columns with default None or NaN values
gdf5['Initiation_Stage'] = None
gdf5['Mid_Stage'] = None
gdf5['End_Stage'] = None

# Fill Initiation_Stage with Date + 30 days
gdf5['Initiation_Stage'] = gdf5['Date'] + pd.Timedelta(days=30)

# Fill Mid_Stage with Date + 120 days
gdf5['Mid_Stage'] = gdf5['Date'] + pd.Timedelta(days=120)

# Set End_Stage to a fixed date of 2024-10-15
gdf5['End_Stage'] = pd.to_datetime('2024-10-15')

# Convert each column to datetime format
gdf5['Date'] = pd.to_datetime(gdf5['Date']).dt.strftime('%Y-%m-%d')
gdf5['Initiation_Stage'] = pd.to_datetime(gdf5['Initiation_Stage']).dt.strftime('%Y-%m-%d')
gdf5['Mid_Stage'] = pd.to_datetime(gdf5['Mid_Stage']).dt.strftime('%Y-%m-%d')
gdf5['End_Stage'] = pd.to_datetime(gdf5['End_Stage']).dt.strftime('%Y-%m-%d')

# Rename the 'ID' column to 'TRT_ID'
gdf5 = gdf5.rename(columns={'ID': 'TRT_ID'})

# Assuming gdf5 is already defined and the Date column is in datetime format
# Filtering rows and creating the dataframes based on different planting dates
Early_Hybrid = gdf5[gdf5['Date'] == '2024-05-08']
Mid_Hybrid = gdf5[gdf5['Date'] == '2024-05-15']
Late_Hybrid = gdf5[gdf5['Date'] == '2024-05-31']

for df in [Early_Hybrid, Mid_Hybrid, Late_Hybrid]:
    df['Kc_Ini'] = 0.4
    df['Kc_Mid'] = 1.2
    df['Kc_End'] = 0.5

colby_mesonet_Early = colby_mesonet.copy() #Making dataframe based on the Early Hybrid
colby_mesonet_Mid = colby_mesonet.copy() #Making dataframe based on the Mid Hybrid
colby_mesonet_Late = colby_mesonet.copy() #Making dataframe based on the Late Hybrid

# Select rows from index 37 to 197
colby_mesonet_Early = colby_mesonet_Early.loc[37:197]
# Select rows from index 44 to 197
colby_mesonet_Mid = colby_mesonet_Mid.loc[44:197]
# Select rows from index 60 to 197
colby_mesonet_Late = colby_mesonet_Late.loc[60:197]

for df in [colby_mesonet_Early, colby_mesonet_Mid, colby_mesonet_Late]:
    df['Kc'] = None
    df['CWR'] = None
    df['ER'] = None
    
# Fill Kc values based on index ranges
colby_mesonet_Early.loc[37:67, 'Kc'] = 0.4
colby_mesonet_Early.loc[68:157, 'Kc'] = 1.2
colby_mesonet_Early.loc[158:, 'Kc'] = 0.5

# Fill Kc values based on index ranges
colby_mesonet_Mid.loc[44:74, 'Kc'] = 0.4
colby_mesonet_Mid.loc[75:164, 'Kc'] = 1.2
colby_mesonet_Mid.loc[165:, 'Kc'] = 0.5

# Fill Kc values based on index ranges
colby_mesonet_Late.loc[60:90, 'Kc'] = 0.4
colby_mesonet_Late.loc[91:180, 'Kc'] = 1.2
colby_mesonet_Late.loc[180:, 'Kc'] = 0.5

# Calculate ER in each dataframe as PRECIP * 0.8 as the irrigation is sprinkler
for df in [colby_mesonet_Early, colby_mesonet_Mid, colby_mesonet_Late]:
    df['ER'] = df['PRECIP'] * 0.8

# Calculate CWR in each dataframe as ET0 * Kc

for df in [colby_mesonet_Early, colby_mesonet_Mid, colby_mesonet_Late]:
    df['CWR'] = df['ET0'] * df['Kc']

    
# Add Planting column with respective types
colby_mesonet_Early['Planting'] = 'Early'
colby_mesonet_Mid['Planting'] = 'Mid'
colby_mesonet_Late['Planting'] = 'Late'

# Join the dataframes along the rows into a new dataframe named CWR
CWR = pd.concat([colby_mesonet_Early, colby_mesonet_Mid, colby_mesonet_Late], ignore_index=True)

# CWR1 variable used for plotting
CWR1 = CWR.copy()

# Ensure TIMESTAMP is in datetime format
CWR1['TIMESTAMP'] = pd.to_datetime(CWR1['TIMESTAMP'])

# Filter for each Planting type and create individual plots
for planting_type in CWR1['Planting'].unique():
    planting_data = CWR1[CWR1['Planting'] == planting_type]
    
    plt.figure(figsize=(10, 6))
    plt.plot(planting_data['TIMESTAMP'], planting_data['ER'], label='ER (Effective Rainfall)', color='green')
    plt.plot(planting_data['TIMESTAMP'], planting_data['CWR'], label='CWR (Crop Water Requirement)', color='purple')
    plt.plot(planting_data['TIMESTAMP'], planting_data['ET0'], label='ET0 (Reference Evapotranspiration)', color='red', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Readings in mm')
    plt.title(f'{planting_type} Planting - Crop Water Requirement Analysis')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Show plot for each planting type
    plt.tight_layout()
    plt.show()



#############################
# Define the HTML code for a Google search bar
google_search_html = """
<div style="text-align: center;">
    <form action="https://www.google.com/search" method="get" target="_blank">
        <input type="text" name="q" placeholder="Search on Google" style="width: 80%; padding: 10px; font-size: 16px;">
        <input type="submit" value="Search" style="padding: 10px; font-size: 16px;">
    </form>
</div>
"""

# Display the Google search bar using st.markdown with allow HTML enabled
st.markdown(google_search_html, unsafe_allow_html=True)

#############################

########### Generating the Conditional statements
# Sidebar configuration with additional dropdowns for gdf4
with st.sidebar:
    # Display logo image
    st.image("wildcat_logo.png", use_column_width=True)
    
    st.title("🌾 TAPS Dashboard")

    # Dataset selection
    dataset_option = st.selectbox("Select Dataset", ["Nitrogen Fertilizer Application", "Irrigation Applied", "Electrical Conductivity and pH", "Volumetric Water Content", "Weather and Climate", "NDVI_Ceres", "Crop Water Requirement Analysis"])

    # Parameter selection based on dataset choice
    if dataset_option == "Electrical Conductivity and pH":
        parameter_option = st.selectbox("Select Parameter", ["EC shallow", "EC deep", "pH"])
        gdf = gdf_pb_ss2

    elif dataset_option == "Nitrogen Fertilizer Application":
        parameter_option = st.selectbox("Select Date ", nitrogen_date_columns)
        gdf = gdf1

    elif dataset_option == "Irrigation Applied":
        parameter_option = st.selectbox("Select Date ", irrigation_date_columns)
        gdf = gdf2

    elif dataset_option == "Volumetric Water Content":
        date_option = st.selectbox("Select Date Group", sorted(gdf4['Group_date'].dropna().unique()))
        depth_option = st.selectbox("Select Depth (inches)", depth_columns)
        
    elif dataset_option == "Weather and Climate":
        climate_parameter_option = st.selectbox("Select Weather or Climate Parameter", [
           "Annual Precipitation", "Annual Evapotranspiration",
           "Annual Minimum and Maximum Temperature",
           "Annual Minimum and Maximum Relative Humidity",
           "Annual Wind Speed", "Annual Solar Radiation"])
    # NDVI selection
    elif dataset_option == "NDVI_Ceres":
        # Show cleaned date options in the dropdown
        selected_date = st.selectbox("Select Date", list(ndvi_date_map.keys()))
        # Map back to the full column name using selected_date
        ndvi_date_option = ndvi_date_map[selected_date]
    
    #Crop Water Requirement
    elif dataset_option == "Crop Water Requirement Analysis":
        # Show planting date options in the dropdown
        planting_date = st.selectbox("Planting Time", ["Early", "Mid", "Late"])
        
    
    
    # Project information at the bottom of the sidebar
    st.markdown("---")  # Adds a horizontal line separator
    st.markdown("### Project Information")
    st.markdown("""
        This dashboard is part of the **Testing Ag Performance Solutions (TAPS) Program**.
        The project aims to sustainably use water in agriculture.
        It contains data of irrigation, nitrogen fertilzer application, soil physical and chemical properties, and volumetric water content 
        to aid decision-making for crop consultants, farmers, and water managers.
    """)

# Function to plot spatial data for EC/pH, Nitrogen, Irrigation, or VWC with latitude and longitude in degrees
def plot_data(gdf, avg_col, title="", cbar_label="", cmap_choice=cm.viridis):
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
    norm = mcolors.Normalize(vmin=gdf[avg_col].min(), vmax=gdf[avg_col].max())

    # Plot polygons
    gdf.plot(
        column=avg_col,
        cmap=cmap_choice,
        linewidth=1,
        edgecolor="black",
        legend=False,
        ax=ax,
        norm=norm
    )

    # Title and labels
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}°"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}°"))

    # Color bar
    sm = plt.cm.ScalarMappable(cmap=cmap_choice, norm=norm)
    sm._A = []
    fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02).set_label(cbar_label, fontsize=12)

    # Add TRT_ID labels
    for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf['TRT_ID']):
        ax.text(x, y, str(label), fontsize=8, ha='center', va='center', color='black', fontweight='bold')

    st.pyplot(fig)

# Function for box plot by hybrid
def boxplot_by_hybrid(gdf, avg_col, hybrid_col="Hybrid"):
    plt.figure(figsize=(8, 6))  # Adjust size as needed
    if 'Group_date' in gdf.columns:
        sns.boxplot(data=gdf, x=hybrid_col, y=avg_col, hue='Group_date')
    else:
        sns.boxplot(data=gdf, x=hybrid_col, y=avg_col)
    plt.title(f"Box Plot: {str(avg_col)[0:10]} by {hybrid_col}")
    plt.xlabel("Hybrid", fontsize=12)
    plt.ylabel(avg_col)
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())


# Function to create box plots of NDVI values grouped by Hybrid
def boxplot_ndvi_by_hybrid(gdf, ndvi_col, hybrid_col='Hybrid'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=gdf.dropna(subset=[ndvi_col]), x=hybrid_col, y=ndvi_col)
    
    # Extract the date from the NDVI column name
    date_str = ndvi_col.split('_')[1]
    
    # Set plot title and labels
    plt.title(f"Box Plot of NDVI on {date_str}")
    plt.xlabel("Hybrid")
    plt.ylabel("NDVI")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

# Display spatial and box plots in one row based on selection
def display_spatial_and_box_plots(gdf, avg_col, title="", cbar_label="", cmap_choice=cm.viridis, hybrid_col="Hybrid"):
    # Create two columns for spatial and box plots side-by-side
    col1,col2,col3 = st.columns([0.8,0.1,0.8])  # Adjust proportions to control space between columns

    with col1:
        plot_data(gdf, avg_col, title=title, cbar_label=cbar_label, cmap_choice=cmap_choice)
    
    with col3:
        boxplot_by_hybrid(gdf, avg_col, hybrid_col=hybrid_col)
        
def ndvi_display_spatial_and_box_plots(gdf, ndvi_col, title="", hybrid_col="Hybrid", cmap_choice=cm.viridis):
    # Create two columns for spatial and box plots side-by-side
    col1, col2, col3 = st.columns([0.8, 0.1, 0.8])  # Adjust proportions to control space between columns

    with col1:
        # Plot NDVI spatial map
        plot_ndvi(gdf, ndvi_col, title=title, cmap_choice=cmap_choice)
    
    with col3:
        # Plot NDVI box plot grouped by hybrid
        boxplot_ndvi_by_hybrid(gdf, ndvi_col, hybrid_col=hybrid_col)
        
    

def display_description_box(title, description):
    st.markdown(
        f"""
        <div style="
            background-color: #f0f8ff;
            padding: 10px 20px;
            border-radius: 5px;
            border-left: 5px solid #5D3FD3;
            margin-top: 20px;
        ">
            <h4 style="margin-bottom: 5px; color: #5D3FD3;">{title}</h4>
            <p style="font-size: 18px; color: #333333; margin-top: 0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def get_chatgpt_response(prompt):
    openai.api_key = 'sk-pKSjjLwVyeFcFtp5cduZUDqBfOvh2LdAlA-YFGjd9gT3BlbkFJuX-FrRxdZJubZFc8CUz_o31536kUq0dlTtKFri6KUA'
    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                    }
                ]
            )
    except Exception as e:
        return f"Error {e}"
    
    return completion.choices[0].message.content

# Function to plot NDVI data for a specific date
def plot_ndvi(gdf, ndvi_col, title="", cmap_choice='viridis'):
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = gdf[ndvi_col].min(), gdf[ndvi_col].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot NDVI with color mapping
    gdf.plot(column=ndvi_col, cmap=cmap_choice, edgecolor='black', legend=False, ax=ax, norm=norm)
    ax.set_title(f"NDVI on {title}", fontsize=16)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    
    # Format axes to display degree symbol
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}°"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}°"))
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap_choice, norm=norm)
    sm._A = []
    fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02).set_label("NDVI", fontsize=12)
    
    # Label TRT_IDs on each polygon
    for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf['TRT_ID']):
        ax.text(x, y, str(label), fontsize=8, ha='center', va='center', color='black', fontweight='bold')
    st.pyplot(fig)

# Example usage with conditional statements
if dataset_option == "Electrical Conductivity and pH":
    if parameter_option == "EC shallow":
        st.markdown(f"<h3 style='text-align: center;'>EC Shallow Data</h3>",unsafe_allow_html=True)
        display_spatial_and_box_plots(gdf, avg_col="EC shallow avg", title="Map of EC Shallow Average", cbar_label="EC shallow avg", cmap_choice=cm.viridis)
        # Calculate max, min, and mean values for EC shallow average
        ec_shallow_max = gdf["EC shallow avg"].max()
        ec_shallow_min = gdf["EC shallow avg"].min()
        ec_shallow_mean = gdf["EC shallow avg"].mean()
    
        # Display EC shallow statistics in metrics format
        st.markdown("<h3 style='text-align: center;'>EC Shallow Statistics</h3>", unsafe_allow_html=True)
    
        cols = st.columns(3)
        cols[0].metric(label="Max EC Shallow", value=f"{ec_shallow_max:.2f}")
        cols[1].metric(label="Min EC Shallow", value=f"{ec_shallow_min:.2f}")
        cols[2].metric(label="Mean EC Shallow", value=f"{ec_shallow_mean:.2f}")
       # Description with A.I. WildCat analysis
        response = get_chatgpt_response(
            f"The selected shallow Electrical Conductivity (EC) analysis shows a maximum value of {ec_shallow_max:.2f}, "
            f"a minimum value of {ec_shallow_min:.2f}, and an average of {ec_shallow_mean:.2f}. "
            "Please interpret these values in the context of sustainable irrigation, providing a concise analysis."
        )
        display_description_box(" 🤖 Analysis of EC Shallow Data by A.I. WildCat", response)
    


    elif parameter_option == "EC deep":
        st.markdown(f"<h3 style='text-align: center;'>EC Deep Data</h3>", unsafe_allow_html=True)
    
    # Display spatial and box plots for EC deep data
        display_spatial_and_box_plots(
        gdf, 
        avg_col="EC deep avg", 
        title="Map of EC Deep Average", 
        cbar_label="EC deep avg", 
        cmap_choice=cm.plasma
    )
    
    # Calculate max, min, and mean values for EC deep average
        ec_deep_max = gdf["EC deep avg"].max()
        ec_deep_min = gdf["EC deep avg"].min()
        ec_deep_mean = gdf["EC deep avg"].mean()
    
    # Display EC deep statistics in metrics format
        st.markdown("<h3 style='text-align: center;'>EC Deep Statistics</h3>", unsafe_allow_html=True)
    
        cols = st.columns(3)
        cols[0].metric(label="Max EC Deep", value=f"{ec_deep_max:.2f}")
        cols[1].metric(label="Min EC Deep", value=f"{ec_deep_min:.2f}")
        cols[2].metric(label="Mean EC Deep", value=f"{ec_deep_mean:.2f}")
    
    # Generate the A.I. WildCat prompt with max and min EC deep values
        response = get_chatgpt_response(
        f"The selected deep Electrical Conductivity (EC) analysis shows an average value distribution. "
        f"The maximum EC deep value is {ec_deep_max}, and the minimum value is {ec_deep_min}. "
        "Please interpret these EC deep values in the context of sustainable irrigation, providing a concise analysis."
    )

    # Display A.I. WildCat's response in the description box
        display_description_box(
        " 🤖 Analysis of Deep Electrical Conductivity by A.I. WildCat",
        response
    )

    elif parameter_option == "pH":
        st.markdown(f"<h3 style='text-align: center;'>pH Data</h3>", unsafe_allow_html=True)
    
    # Display spatial and box plots for pH data
        display_spatial_and_box_plots(
        gdf, 
        avg_col="pH avg", 
        title="Map of pH Average", 
        cbar_label="pH avg", 
        cmap_choice=cm.inferno
    )
    
    # Calculate max and min pH values
        ph_max = gdf["pH avg"].max()
        ph_min = gdf["pH avg"].min()
        ph_mean = gdf["pH avg"].mean()
    
    # Display pH statistics in metrics format
        st.markdown("<h3 style='text-align: center;'>pH Statistics</h3>", unsafe_allow_html=True)
    
        cols = st.columns(3)
        cols[0].metric(label="Max pH", value=f"{ph_max:.2f}")
        cols[1].metric(label="Min pH", value=f"{ph_min:.2f}")
        cols[2].metric(label="Mean pH", value=f"{ph_mean:.2f}")
    
    # Generate the A.I. WildCat prompt with max and min pH values
        response = get_chatgpt_response(
        f"The selected pH analysis for soil is based on the average pH values. "
        f"The maximum pH value observed is {ph_max}, and the minimum pH value is {ph_min}. "
        "Please interpret these pH levels in the context of sustainable irrigation, and provide a concise analysis."
    )

    # Display A.I. WildCat's response in the description box
        display_description_box(
        " 🤖 Analysis of Soil pH by A.I. WildCat",
        response
    )


elif dataset_option == "Nitrogen Fertilizer Application":
    # Display metrics for each state
    # Display metrics in a single row
    
    
    st.markdown(
        f"<h3 style='text-align: center;'>Nitrogen Fertilizer Application on {str(parameter_option)[0:10]}</h3>",
        unsafe_allow_html=True
    )
    
    # Display spatial and box plots for the selected nitrogen parameter
    display_spatial_and_box_plots(
        gdf, 
        avg_col=parameter_option, 
        title=f"Nitrogen Fertilizer Application on {str(parameter_option)[0:10]}", 
        cbar_label="Nitrogen Level (lbs/ac)", 
        cmap_choice=cm.YlGn
    )
    
    # Calculating max, min, and mean values for the selected parameter
    nitrogen_max = gdf[parameter_option].max()
    nitrogen_min = gdf[parameter_option].min()
    nitrogen_mean = gdf[parameter_option].mean()
    
    st.markdown(
        f"<h3 style='text-align: center;'>Nitrogen Fertilizer Application Statistics</h3>",
        unsafe_allow_html=True
    )
    
    cols = st.columns(3)
    cols[0].metric(label="Max Nitrogen", value=f"{nitrogen_max} lbs/ac")
    cols[1].metric(label="Min Nitrogen", value=f"{nitrogen_min} lbs/ac")
    cols[2].metric(label="Mean Nitrogen", value=f"{nitrogen_mean:.2f} lbs/ac")
    
    # Generate the A.I. WildCat prompt with selected parameter, max, and min values
    response = get_chatgpt_response(
        f"The selected nitrogen fertilizer application analysis is for the date {str(parameter_option)[0:10]}. "
        f"The maximum nitrogen application for this date is {nitrogen_max} lbs/ac, and the minimum application is {nitrogen_min} lbs/ac. "
        "Please interpret these values in the context of sustainable irrigation and provide a concise analysis."
    )

    # Display A.I. WildCat's response in the description box
    display_description_box(
        " 🤖 Analysis of Nitrogen Fertilizer Application by A.I. WildCat",
        response
    )

    
elif dataset_option == "Irrigation Applied":
    st.markdown(f"<h3 style='text-align: center;'>Irrigation Applied on {str(parameter_option)[0:10]}</h3>", unsafe_allow_html=True)
    
    # Display spatial and box plots for the selected irrigation parameter
    display_spatial_and_box_plots(gdf, avg_col=parameter_option, title=f"Irrigation Application on {str(parameter_option)[0:10]}", cbar_label="Irrigation Amount (inches)", cmap_choice=cm.Blues)
    
    # Calculate max, min, and mean irrigation values for the selected parameter
    irrigation_max = gdf[parameter_option].max()
    irrigation_min = gdf[parameter_option].min()
    irrigation_mean = gdf[parameter_option].mean()
    
    # Display irrigation statistics in metrics format
    st.markdown("<h3 style='text-align: center;'>Irrigation Statistics</h3>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    cols[0].metric(label="Max Irrigation", value=f"{irrigation_max} inches")
    cols[1].metric(label="Min Irrigation", value=f"{irrigation_min} inches")
    cols[2].metric(label="Mean Irrigation", value=f"{irrigation_mean:.2f} inches")
    
    
    # Generate the A.I. WildCat prompt with selected parameter, max, and min values
    response = get_chatgpt_response(
        f"The selected irrigation analysis is for the date {str(parameter_option)[0:10]}. "
        f"The maximum irrigation amount applied for this date is {irrigation_max} inches, and the minimum irrigation amount is {irrigation_min} inches. "
        "Please interpret these values in the context of sustainable irrigation and provide a concise analysis."
    )

    # Display A.I. WildCat's response in the description box
    display_description_box(
        " 🤖 Analysis of Irrigation by A.I. WildCat",
        response
    )

elif dataset_option == "Volumetric Water Content":
    st.markdown(f"<h3 style='text-align: center;'>Volumetric Water Content at {depth_option} inches on {date_option}</h3>", unsafe_allow_html=True)
    
    # Filter data for the selected date and depth option
    filtered_data = gdf4[(gdf4['Group_date'] == date_option) & (~gdf4[depth_option].isna())]

    # Display spatial and box plots for VWC
    display_spatial_and_box_plots(filtered_data, avg_col=depth_option, title=f"VWC at {depth_option} inches on {date_option}", cbar_label="Volumetric Water Content (%)", cmap_choice=cm.Blues)
    
    # Calculate max, min, and mean VWC values for the selected depth and date
    vwc_max = filtered_data[depth_option].max()
    vwc_min = filtered_data[depth_option].min()
    vwc_mean = filtered_data[depth_option].mean()
    
    # Display VWC statistics in metrics format
    st.markdown("<h3 style='text-align: center;'>VWC Statistics</h3>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    cols[0].metric(label="Max VWC", value=f"{vwc_max:.2f} %")
    cols[1].metric(label="Min VWC", value=f"{vwc_min:.2f} %")
    cols[2].metric(label="Mean VWC", value=f"{vwc_mean:.2f} %")
    
    # Generate the A.I. WildCat prompt with selected depth, date, max, and min values
    response = get_chatgpt_response(
        f"The selected Volumetric Water Content (VWC) Analysis is for {depth_option} inches depth on {date_option}. "
        f"The maximum VWC value for this depth and date is {vwc_max}, and the minimum VWC value is {vwc_min}. "
        "Please interpret these values in the context of sustainable irrigation and provide a concise analysis."
    )

    # Display A.I. WildCat's response in the description box
    display_description_box(
        " 🤖 Analysis of Volumetric Water Content by A.I. WildCat",
        response
    )


# Annual Precipitation
def plot_annual_precipitation_interactive(annual_precipitation):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_precipitation.index,
        y=annual_precipitation.values,
        mode='lines+markers',
        name='Annual Precipitation',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title="Average Annual Precipitation (1990-2019)",
        xaxis_title="Year",
        yaxis_title="Average Precipitation (mm)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Annual Evapotranspiration
def plot_annual_evapotranspiration_interactive(annual_eto):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_eto.index,
        y=annual_eto.values,
        mode='lines+markers',
        name='Annual Evapotranspiration',
        line=dict(color='green')
    ))
    fig.update_layout(
        title="Average Annual Evapotranspiration (ETO)",
        xaxis_title="Year",
        yaxis_title="Average ETO (mm)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Minimum and Maximum Temperatures
def plot_annual_temperature_interactive(annual_tmmn, annual_tmmx):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_tmmn.index,
        y=annual_tmmn.values,
        mode='lines+markers',
        name='Minimum Temperature',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=annual_tmmx.index,
        y=annual_tmmx.values,
        mode='lines+markers',
        name='Maximum Temperature',
        line=dict(color='red')
    ))
    fig.update_layout(
        title="Average Annual Minimum and Maximum Temperatures",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Minimum and Maximum Relative Humidity
def plot_annual_relative_humidity_interactive(annual_rmin, annual_rmax):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_rmin.index,
        y=annual_rmin.values,
        mode='lines+markers',
        name='Minimum Relative Humidity',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=annual_rmax.index,
        y=annual_rmax.values,
        mode='lines+markers',
        name='Maximum Relative Humidity',
        line=dict(color='green')
    ))
    fig.update_layout(
        title="Average Annual Minimum and Maximum Relative Humidity",
        xaxis_title="Year",
        yaxis_title="Relative Humidity (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Annual Wind Speed
def plot_annual_wind_speed_interactive(annual_vs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_vs.index,
        y=annual_vs.values,
        mode='lines+markers',
        name='Average Wind Speed',
        line=dict(color='purple')
    ))
    fig.update_layout(
        title="Average Annual Wind Speed",
        xaxis_title="Year",
        yaxis_title="Wind Speed (m/s)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Annual Solar Radiation
def plot_annual_solar_radiation_interactive(annual_srad):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual_srad.index,
        y=annual_srad.values,
        mode='lines+markers',
        name='Average Solar Radiation',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title="Average Annual Solar Radiation",
        xaxis_title="Year",
        yaxis_title="Solar Radiation (W/m²)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to display the selected Climate and Weather plot
if dataset_option == "Weather and Climate":
    if climate_parameter_option == "Annual Precipitation":
        plot_annual_precipitation_interactive(annual_precipitation)
    elif climate_parameter_option == "Annual Evapotranspiration":
        plot_annual_evapotranspiration_interactive(annual_eto)
    elif climate_parameter_option == "Annual Minimum and Maximum Temperature":
        plot_annual_temperature_interactive(annual_tmmn, annual_tmmx)
    elif climate_parameter_option == "Annual Minimum and Maximum Relative Humidity":
        plot_annual_relative_humidity_interactive(annual_rmin, annual_rmax)
    elif climate_parameter_option == "Annual Wind Speed":
        plot_annual_wind_speed_interactive(annual_vs)
    elif climate_parameter_option == "Annual Solar Radiation":
        plot_annual_solar_radiation_interactive(annual_srad)

if dataset_option == "NDVI_Ceres" and selected_date:
    st.markdown(f"<h3 style='text-align: center;'>NDVI Data on {selected_date}</h3>", unsafe_allow_html=True)
    ndvi_display_spatial_and_box_plots(plots_boundary1, ndvi_date_option, title=selected_date)
   # Calculate max, min, and mean NDVI values for the selected date column
    ndvi_max = plots_boundary1[ndvi_date_option].max()
    ndvi_min = plots_boundary1[ndvi_date_option].min()
    ndvi_mean = plots_boundary1[ndvi_date_option].mean()
    
    # Display NDVI statistics in metrics format
    st.markdown("<h3 style='text-align: center;'>NDVI Statistics</h3>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    cols[0].metric(label="Max NDVI", value=f"{ndvi_max:.2f}")
    cols[1].metric(label="Min NDVI", value=f"{ndvi_min:.2f}")
    cols[2].metric(label="Mean NDVI", value=f"{ndvi_mean:.2f}")
    
    # Generate the A.I. WildCat prompt with selected date, max, min, and mean values
    response = get_chatgpt_response(
        f"The selected NDVI Analysis is for the date {selected_date}. "
        f"The maximum NDVI value is {ndvi_max:.2f}, the minimum NDVI value is {ndvi_min:.2f}, and the average NDVI is {ndvi_mean:.2f}. "
        "Please interpret these values in the context of sustainable irrigation and provide a concise analysis."
    )
    
    # Display ChatGPT's response in the description box
    display_description_box(
        " 🤖 Analysis of NDVI by A.I. WildCat",
        response
    )

# Plotting CWR with Plotly
def plot_cwr_interactive(planting_data, planting_date):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=planting_data['TIMESTAMP'], y=planting_data['ER'],
                             mode='lines', name='ER (Effective Rainfall)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=planting_data['TIMESTAMP'], y=planting_data['CWR'],
                             mode='lines', name='CWR (Crop Evapotranspiration)', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=planting_data['TIMESTAMP'], y=planting_data['ET0'],
                             mode='lines', name='ET0 (Reference Evapotranspiration)', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title=f'{planting_date} Planting - Crop Water Requirement Analysis',
        xaxis_title='Date',
        yaxis_title='Readings in mm',
        hovermode='x unified'
    )
    fig.update_xaxes(tickangle=45)
    
    # Display Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


if dataset_option == "Crop Water Requirement Analysis":
    st.markdown(f"<h3 style='text-align: center;'>Crop Water Requirement Analysis</h3>", unsafe_allow_html=True)
    if planting_date == "Early":
        planting_data = CWR1[CWR1['Planting'] == "Early"]
        plot_cwr_interactive(planting_data, "Early")
        
        # Generate the A.I. WildCat prompt with selected date, max, min, and mean values
        response = get_chatgpt_response(
            "How crop water requirement is related with Effective Precipitation, and crop Evapotranspiration. Give no formula. Also tell me the water requirement by maize according to the Late planting"
        )
        
        # Display A.I. WildCat's response in the description box
        display_description_box(
            " 🤖 Analysis of NDVI by A.I. WildCat",
            response
        )
        
    elif planting_date == "Mid":
        planting_data = CWR1[CWR1['Planting'] == "Mid"]
        plot_cwr_interactive(planting_data, "Mid")
        
        # Generate the A.I. WildCat prompt with selected date, max, min, and mean values
        response = get_chatgpt_response(
            "How crop water requirement is related with Effective Precipitation, and crop Evapotranspiration. Give no formula. Also tell me the water requirement by maize according to the Late planting"
        )
        
        # Display A.I. WildCat's response in the description box
        display_description_box(
            " 🤖 Analysis of NDVI by A.I. WildCat",
            response
        )
        
    elif planting_date == "Late":
        planting_data = CWR1[CWR1['Planting'] == "Late"]
        plot_cwr_interactive(planting_data, "Late")
        
        # Generate the A.I. WildCat prompt with selected date, max, min, and mean values
        response = get_chatgpt_response(
            "How crop water requirement is related with Effective Precipitation, and crop Evapotranspiration. Give no formula. Also tell me the water requirement by maize according to the Late planting"
        )
        
        # Display A.I. WildCat's response in the description box
        display_description_box(
            " 🤖 Analysis of NDVI by A.I. WildCat",
            response
        )
"""This module contains auxiliary functions for generating graphs which are used in the main notebook."""

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import statsmodels.formula.api as smf
import seaborn as sns

#For spatial analysis
import geopandas as gpd
import shapely.geometry as geom
import libpysal as lp #For spatial weights

from pysal.viz import splot #exploratory analysis
from pysal.explore import esda #exploratory analysis
from pysal.model import spreg #For spatial regression

pd.options.display.float_format = "{:,.2f}".format

from auxiliary.data_import import *
from auxiliary.plots import *
from auxiliary.simulations import *
from auxiliary.tables import *

def map_countries():
    """
    Generates a map of the countries with country names
    and saves it in material as a *.png file
 
    """
    #import the shapefiles
    districts, coast = get_shapefile()
    

    #Plotting the map
    f, ax = plt.subplots(1, figsize=(15, 15))
    #coast.plot(ax=ax, color="antiquewhite")
    coast.plot(ax=ax, color="grey")
    #display country names
    coast.apply(lambda x: ax.annotate(s=x.ADM0_NAME, xy=x.geometry.centroid.coords[0], ha="center", fontsize=14), axis=1)
    districts.plot(ax=ax, column="iso3v10_y", legend=False, scheme='Quantiles', cmap="Blues")
    coast.boundary.plot(ax=ax, color="gray", linewidth=0.2)

    ax.set_axis_off()
    ax.set_title("Countries in the sample", fontsize=14)
    plt.axis('equal')
    plt.savefig("material/map_countries.png", bbox_inches='tight')
    plt.close(f) #avoids the plot being printed

def map_data_section(districts, coast, citydata):
    """
    Generates a graph with two maps, side by side.
        Map 1: district level change of moisture
        Map 2: city level change of rainfall
 
    """
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
    axs = axs.flatten()# Make the axes accessible with single indexing

    # Districts
    districts.plot(column="ADsm0_2moistu", ax=axs[0], scheme='quantiles', legend=True, linewidth=0, cmap='RdPu')
    coast.boundary.plot(ax=axs[0], color='grey')
    axs[0].set_axis_off()
    axs[0].set_title("Moisture change at first census", fontweight="bold")

    # City-level
    citydata.plot(column="dlnrain30", ax=axs[1], scheme='quantiles', legend=True,markersize=2, cmap='RdPu')
    coast.boundary.plot(ax=axs[1], color='grey')
    axs[1].set_axis_off()
    axs[1].set_title("City rainfall change (1992)", fontweight="bold")

    # Display the figure
    plt.show()
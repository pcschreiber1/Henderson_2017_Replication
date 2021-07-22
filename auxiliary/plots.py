"""This module contains auxiliary functions for generating graphs which are used in the main notebook."""

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd
import shapely.geometry as geom

from auxiliary.data_import import *

pd.options.display.float_format = "{:,.2f}".format

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
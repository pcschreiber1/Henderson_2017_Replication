"""This module contains auxiliary functions for importing data which are used in the main notebook."""

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd
import shapely.geometry as geom

pd.options.display.float_format = "{:,.2f}".format

#get simulation results
def get_simulation_results():
    SLX_sim = pd.read_csv("data/SLX_sim.csv", index_col=0)
    SDM_sim = pd.read_csv("data/SLX_sim.csv", index_col=0)
    backdoor_sim = pd.read_csv("data/backdoor_sim.csv", index_col=0)

    table = pd.concat([SLX_sim, SDM_sim, backdoor_sim], axis=1)
    table.columns = pd.MultiIndex.from_product(
        [['SLX Simulation', 'SDM Simulation', 'backdoor Simulation'],['Simple', 'Small', 'Large']]
        )
        
    return Sim_results

# Importing data
def importing_regiondata():
    """
    Loads the regiondata
        Should convert the year column to proper year
        
        Should immediately create geopandas dataframe
    Returns: a dataframe
    """
    regiondata = pd.read_stata("data/regiondata.dta")
    return regiondata

# Get spatial data
def get_spatialdata():
    """
    Converts regiondata and citydata into GeoPandas DF and projects it

    Returns: two GeoPandas dataframes (regiondata, citydata)
    """
    #district level
    ##creating pandas dataframe
    regiondata = pd.read_stata("data/regiondata.dta") #--> need to also do for other 
    #regiondata = regiondata.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")

    ##creating geopandas dataframe
    regiondata["geometry"] = regiondata[["lon", "lat"]].apply(geom.Point, axis=1) #take each row
    regiondata = gpd.GeoDataFrame(regiondata)
    regiondata.crs = "EPSG:4326"
    
    #city level
    ##creating pandas dataframe
    citydata = pd.read_stata("data/citydata.dta")
    #regiondata = regiondata.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")

    ##creating geopandas dataframe
    citydata["geometry"] = citydata[["lon", "lat"]].apply(geom.Point, axis=1) #take each row
    citydata = gpd.GeoDataFrame(citydata)
    citydata.crs = "EPSG:4326"
    
    return regiondata, citydata


# Get shape file
def get_shapefile():
    #creating relevant shapefile
    #--------------------
    regiondata = pd.read_stata("data/regiondata.dta") #--> need to also do for other 
    #regiondata = regiondata.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")

    ###creating geopandas dataframe
    regiondata["geometry"] = regiondata[["lon", "lat"]].apply(geom.Point, axis=1) #take each row
    regiondata = gpd.GeoDataFrame(regiondata)
    regiondata.crs = "EPSG:4326"

    ### districts shapefile
    areg = gpd.read_file("data/Henderson_shapefile/afrregnew.gdb")
    areg.crs = "EPSG:4326"

    ### coastlien shapefile
    coast = gpd.read_file("data/afr_g2014_2013_0.shp")
    coast.crs = "EPSG:4326"

    ### Joining region data and districts
    gdb_join = gpd.sjoin(regiondata, areg, how="right", op="within")

    return gdb_join, coast

# Get shape file
def get_shapefile_2():
    #Shapefile African Countries
    ##Source https://africaopendata.org/dataset?tags=shapefiles
    A_countries = gpd.read_file("C:/Projects/ose-data-science-course-project-pcschreiber1/data/afr_g2014_2013_0.shp")
    A_countries.crs = "EPSG:4326"

    #Global shapefile with provinces
    ##Source https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    prov = gpd.read_file("C:/Projects/ose-data-science-course-project-pcschreiber1/data/shapefile_provinces/ne_10m_admin_1_states_provinces.shp")
    prov.crs = "EPSG:4326"

    #Create Shapefile of Africa with provinces
    A_countries = A_countries["ISO2"].dropna() #identify country codes
    prov["Africa"] = None #container column

    for i in A_countries: #loop over country codes
        cond = prov["iso_a2"] == i
        prov.loc[cond, "Africa"] = "True" #identify african countries

    africa = prov[prov["Africa"] == "True"] #create new shapefile

    return africa

# Creating table 1 a (regiondata)
def table_1_a(data):
    """
    Input regiondata from "importing_regiondata()"

    Returns: ?
    """
    df = data.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")
    
    var_list = ["ADsm0_2moistu", "mean_moistu1950_69", "ADurbfrac", "firsturbfrac", "lndiscst",
            "areasqkm", "extent_agE", "extent_agH", "D_moist_GT1"]

    df = df[var_list].sort_values(by = var_list)
    
    df = df.rename(columns={"ADsm0_2moistu" : "Annualized moisture growth"})

    return df

# Creating Table 2 (regression table: Effect of moisture on urbanization: heterogeneity by industrialization)
def table_2(data):
    """
    check out: https://www.vincentgregoire.com/standard-errors-in-python/ AND http://aeturrell.com/2018/02/20/econometrics-in-python-partII-fixed-effects/ 
    """
    df = data.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")
    #c(var) for fixed effect - "-1" for dropping intercept
    formula = "ADurbfrac ~ ADsm0_2moistu + firsturbfrac + lndiscst + C(countryyear) -1"
    #for clustered errors
    stat = smf.ols(formula = formula, data=df).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df['afruid']},
                                                        use_t=True)
    return stat

# Creating Figure 4 (Variability of climate change in Africa)
def figure_4(data):
    
    #Moisture, three-year moving average  normalized by country 1950-69 mean
    data["sm0_2normarid"] = data["sm0_2moistu"]/ data["mean_moistu1950_69"]
    
    fig, ax = plt.subplots(figsize=(10,4))
    for key, grp in data.groupby(['iso3v10']):
        ax.plot(grp['year'], grp['sm0_2normarid'], label=key)

    ax.legend(bbox_to_anchor=(0, 0, 1, -0.1), ncol=5, mode="expand", borderaxespad=0.)
    #ax.legend(bbox_to_anchor=(0, -0,5))#, loc="lower center")
    figure = plt.show()
    return figure

# Creating Table 6 (Change in city output and rainfall: heterogeneity by industrialization)
def table_6(data):

    #i.year in stata means time fixed effects
    formula = "dlnl1 ~ dlnrain30 + C(year) -1"
    #for clustered errors
    stat = smf.ols(formula = formula, data=data).fit(cov_type='cluster',
                                                        cov_kwds={'groups': data['agidison']},
                                                        use_t=True)
    return stat
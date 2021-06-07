"""This module contains auxiliary functions for importing data which are used in the main notebook."""

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

pd.options.display.float_format = "{:,.2f}".format


# Importing data
def importing_regiondata():
    """
    Loads the regiondata

    Returns: a dataframe
    """
    regiondata = pd.read_stata("data/regiondata.dta")
    return regiondata

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
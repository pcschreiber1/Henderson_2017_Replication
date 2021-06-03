#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.2f}".format


# Importing data

def importing_regiondata():
    regiondata = pd.read_stata("data/regiondata.dta")
    return regiondata

# Creating table 1 a (regiondata)
def table_1_a(data):
    """
    Input regiondata from "importing_regiondata()"
    """
    df = data.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")
    
    data = regiondata.query("abspctileADsm0_2moistu > 6 & abspctileADurbfrac > 6")

    var_list = ["ADsm0_2moistu", "mean_moistu1950_69", "ADurbfrac", "firsturbfrac", "lndiscst",
            "areasqkm", "extent_agE", "extent_agH", "D_moist_GT1"]

    data.sort_values(by = var_list)

    data = data[var_list]
    data.rename(columns={"ADsm0_2moistu" : "Annualized moisture growth"})
    


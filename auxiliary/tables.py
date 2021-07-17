"""This module contains auxiliary functions for generating graphs which are used in the main notebook."""

#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd
import shapely.geometry as geom
import libpysal as lp #For spatial weights

from auxiliary.data_import import *

pd.options.display.float_format = "{:,.2f}".format

# get reg table
def get_table_2(regressors, specification, data):
    """
    Generates the regression table 2
    Inputs:
        - regressors: array of column names
        - specification: dictionary with column names
        - data: data frame (regiondata)
        
    Returns: container (pandas data frame)
    """
    ## get list of regressors
    #regressors = []
    #for i in specification.keys():
    #    regressors.extend(specification[i])
    #regressors = list(set(regressors))


    container = pd.DataFrame()

    container['regressors'] = regressors
    container = container.set_index('regressors')

    for key in specification.keys():
        table = pd.DataFrame({'Urbanization rate': [], 'Std.err': [], 'P-Value': [],})

        table['regressors'] = regressors
        table = table.set_index('regressors')
        
        formula = "ADurbfrac ~"
        for regressors in specification[key]:
            formula = formula + f" {regressors} +"
            
        
        formula = formula + " C(countryyear) -1"
        #c(var) for fixed effect - "-1" for dropping intercept
        result = smf.ols(
            formula = formula, data=data
            ).fit(
            cov_type='cluster',cov_kwds={'groups': data['afruid']},use_t=True
            )
        for coef in specification[key]:
            outputs = [result.params[coef], result.bse[coef], result.pvalues[coef]]
            table.loc[coef] = outputs
        
        container = pd.concat([container, table], axis=1)

    container.columns = pd.MultiIndex.from_product(
            [specification.keys(),
            ['Urbanization rate', 'Std.err', 'P-Value',]]
            )
    
    return container
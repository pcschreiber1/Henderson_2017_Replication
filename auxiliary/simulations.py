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

#get simulation results
def get_simulation_results():
    SLX_sim = pd.read_csv("data/SLX_sim.csv", index_col=0)
    SDM_sim = pd.read_csv("data/SDM_sim.csv", index_col=0)
    backdoor_sim = pd.read_csv("data/backdoor_sim.csv", index_col=0)

    table = pd.concat([SLX_sim, SDM_sim, backdoor_sim], axis=1)
    table.columns = pd.MultiIndex.from_product(
        [['SLX Simulation', 'SDM Simulation', 'backdoor Simulation'],['Simple', 'Small', 'Large']]
        )
        
    return table

# SLX sample
def simulate_SLX_sample(num_obs,
                        knn = 10,
                        beta = 0.9,
                        gamma = 0.25):
    """Simulate spatial sample with only spillover from treatment variable
        "Y = WD + X"

    Args:
        num_obs: An integer that specifies the number of individuals
            to sample. Needs to be either 100 or 10.000 for the weight matrix.

    Returns:
        Returns a dataframe with the observables (Y, X, D) as well as
        the unobservables (Y_1, Y_0) and the weight matrix w.
    """

    # Initialize empty data frame
    columns = ["Y", "Y_1", "Y_0", "D", "X", "WD"]
    df = pd.DataFrame(columns=columns, index=range(num_obs))

    df["D"] = np.random.randint(2, size=num_obs) #binary treatment
    df["X"] = np.random.normal(size=num_obs)
    
    if num_obs == 100:
            # generate grid for weight matrix
            x,y=np.indices((10, 10))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    elif num_obs == 10000:
            # generate grid for weight matrix
            x,y=np.indices((100, 100))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    else:
            raise AssertionError # needs to be 100 or 10.000 for constructing weight matrix

    # weight matrix
    w = lp.weights.KNN(data, k = knn)
    
    # calculate spillovers
    df["WD"] = np.dot(w.full()[0], df["D"].to_numpy())#not standardized
    
    # outcomes
    df["Y"] = beta*df["X"] + gamma*df["WD"] + gamma*df["D"] #add D since W is sparse
    df["Y_1"] = beta*df["X"] + gamma*df["WD"] + gamma
    df["Y_0"] = beta*df["X"] + gamma*df["WD"]
    
    return df, w

#SDM sample
def simulate_SDM_sample(num_obs,
                            knn = 10,
                            beta = 0.9,
                            gamma = 0.25,
                            rho = 0.05):
    """Simulate spatial sample with spillover from the treatment and the outcome variable
        "Y = WY+ WD + X"

    Args:
        num_obs: An integer that specifies the number of individuals
            to sample.

    Returns:
        Returns a dataframe with the observables (Y, X, D) as well as
        the unobservables (Y_1, Y_0)and the weight matrix w.
    """

    # Initialize empty data frame
    columns = ["Y", "Y_1", "Y_0", "D", "X", "WD"]
    df = pd.DataFrame(columns=columns, index=range(num_obs))

    df["D"] = np.random.randint(2, size=num_obs) #binary treatment
    df["X"] = np.random.normal(size=num_obs)
    
    if num_obs == 100:
            # generate grid for weight matrix
            x,y=np.indices((10, 10))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    elif num_obs == 2500:
            # generate grid for weight matrix
            x,y=np.indices((50, 50))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    else:
            raise AssertionError # needs to be 100 or 10.000 for constructing weight matrix

    # weight matrix
    w = lp.weights.KNN(data, k = knn)
    
    # calculate spillovers
    df["WD"] = np.dot(w.full()[0], df["D"].to_numpy())#not standardized
    
    # outcomes
    df["Y"] = beta*df["X"] + gamma*df["WD"] + gamma*df["D"] #add D since W is sparse
    df["Y_1"] = beta*df["X"] + gamma*df["WD"] + gamma
    df["Y_0"] = beta*df["X"] + gamma*df["WD"]
    
    df["Y_no_spill"] = df["Y"]
    df["Y_1_no_spill"] = df["Y_1"]
    df["Y_0_no_spill"] = df["Y_0"]
    
    # iterate to generate general equilibrium effect (already settles after 5 times for small rho)
    for i in range(0,10):
        df["WY"] = np.dot(w.full()[0], df["Y"].to_numpy())#not standardized
        df["Y"] = beta*df["X"] + gamma*df["WD"] + gamma*df["D"] + rho* df["WY"]   
        df["Y_1"] = beta*df["X"] + gamma*df["WD"] + gamma + rho* df["WY"]   
        df["Y_0"] = beta*df["X"] + gamma*df["WD"] + rho* df["WY"]   
    
    return df, w

#backdoor
def simulate_backdoor_sample(num_obs,
                            knn = 10,
                            beta = 0.9,
                            gamma = 0.25,
                            rho = 0.05):
    """Simulate spatial sample with spillover from treatment and outcome variable
        "Y = WY+ WD + X" and "D = WD"

    Args:
        num_obs: An integer that specifies the number of individuals
            to sample.

    Returns:
        Returns a dataframe with the observables (Y, X, D) as well as
        the unobservables (Y_1, Y_0) and the weight matrix w.
    """

    # Initialize empty data frame
    columns = ["Y", "Y_1", "Y_0", "D", "X", "WD"]
    df = pd.DataFrame(columns=columns, index=range(num_obs))

    df["D"] = np.random.randint(2, size=num_obs) #binary treatment
    df["X"] = np.random.normal(size=num_obs)
    
    if num_obs == 100:
            # generate grid for weight matrix
            x,y=np.indices((10, 10))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    elif num_obs == 2500:
            # generate grid for weight matrix
            x,y=np.indices((50, 50))
            x.shape=(num_obs,1)
            y.shape=(num_obs,1)
            data=np.hstack([x,y])
    else:
            raise AssertionError # needs to be 100 or 10.000 for constructing weight matrix

    # weight matrix
    w = lp.weights.KNN(data, k = knn)
    
    # calculate spillovers
    df["WD"] = np.dot(w.full()[0], df["D"].to_numpy())/knn #standardized

    # whether you are treated is function of other's treatment
    df.loc[(df.WD > 0.5), 'D']=1
    df.loc[(df.WD <= 0.5), 'D']=0

    # outcomes
    df["Y"] = beta*df["X"] + gamma*df["WD"] + gamma*df["D"] #add D since W is sparse
    df["Y_1"] = beta*df["X"] + gamma*df["WD"] + gamma
    df["Y_0"] = beta*df["X"] + gamma*df["WD"]
    
    df["Y_no_spill"] = df["Y"]
    df["Y_1_no_spill"] = df["Y_1"]
    df["Y_0_no_spill"] = df["Y_0"]
    
    # iterate to generate general equilibrium effect (already settles after 5 times for small rho)
    for i in range(0,10):
        df["WY"] = np.dot(w.full()[0], df["Y"].to_numpy())#not standardized
        df["Y"] = beta*df["X"] + gamma*df["WD"] + gamma*df["D"] + rho* df["WY"]   
        df["Y_1"] = beta*df["X"] + gamma*df["WD"] + gamma + rho* df["WY"]   
        df["Y_0"] = beta*df["X"] + gamma*df["WD"] + rho* df["WY"]   
    
    return df, w


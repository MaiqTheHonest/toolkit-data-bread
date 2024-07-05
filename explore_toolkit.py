
import re
import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
#matplotlib.use('Qt5Agg')     #  use this to make plots NOT in-line
#df = pd.read_csv('titanic.csv')


def unpack(model):
    
    model = model.split('~')
    y_variable = model[0].strip()
    x_variables = model[1]
    x_variables = [x.strip() for x in x_variables.split('+')]
    
    return y_variable, x_variables   



def lm(dataframe, model, plot = False):

    lin_model = linear_model.LinearRegression()
    y_variable, x_variables = unpack(model) 

    X = dataframe[x_variables]
    Y = dataframe[y_variable]

    lin_model.fit(X, Y)
    lin_slope = lin_model.coef_
    lin_intercept = lin_model.intercept_

    Z = lin_model.predict(X)
    
    clean_eq = f"{y_variable} = {lin_model.intercept_.round(decimals=3)}"

    for idx, x in enumerate(x_variables):  
        clean_eq = str(clean_eq) + f" + {lin_model.coef_[idx].round(decimals=2)}*{x_variables[idx]}"

    if len(x_variables) > 2:
        print('4D+ not plottable')

    elif len(x_variables) == 2:
     
        fig1 = plt.figure(figsize=(5,5))
        ax = fig1.add_subplot(111, projection='3d')
        ax.scatter3D(X.iloc[:,0], X.iloc[:,1], Y)
        ax.set_xlabel(x_variables[0], fontweight ='bold') 
        ax.set_ylabel(x_variables[1], fontweight ='bold') 
        ax.set_zlabel(y_variable, fontweight ='bold', rotation=90)
        ax.zaxis.labelpad=-0.5
    
        x_plane, y_plane = np.meshgrid([X.iloc[:,0].min(), X.iloc[:,0].max()], [X.iloc[:,1].min(), X.iloc[:,1].max()])
        z_plane = lin_slope[0] * x_plane + lin_slope[1] * y_plane + lin_intercept
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.7, color='grey')
        ax.set_title(clean_eq)
        
    else:

        fig1 = plt.figure()
        plt.scatter(X.iloc[:,0], Y, color="black")
        plt.plot(X, lin_intercept+lin_slope*X, c='red')  # plot equation of line
        plt.xlabel(x_variables[0])
        plt.ylabel(y_variable)
        plt.title(clean_eq)

    print(clean_eq)

    
    if plot==False:
        plt.close()






def logit(dataframe, model, plot = False):

    y_variable, x_variables = unpack(model) 

    X = dataframe[x_variables]
    P = np.array(dataframe[y_variable])

    if ((P==0) | (P==1)).all() == False:
        binary_indicator = False
    # if any 0 or 1 values of explained var are present, change them to the closest possible to 0 or 1
        P[P == 0]=np.min(P[P != 0])  
        P[P == 1]=np.max(P[P != 1])

        Y = np.log(P / (1 - P))  # format explained var to linear form

        lin_model = linear_model.LinearRegression()
        logit_model = lin_model.fit(X, Y)
        lin_slope = logit_model.coef_
        lin_intercept = logit_model.intercept_
    else:
        binary_indicator = True
        lin_model = linear_model.LogisticRegression()
        Y = P
        logit_model = lin_model.fit(X, Y)
        lin_slope = logit_model.coef_[0]
        lin_intercept = logit_model.intercept_[0]

    clean_eq = f"log(p/(1 - p) = {lin_intercept.round(decimals=5)}"
    TeXclean_eq = r"$log(\frac{p}{1 - p})$ =" + f"{lin_intercept.round(decimals=5)}"

    for idx, x in enumerate(x_variables):  
        clean_eq = str(clean_eq) + f" + {lin_slope[idx].round(decimals=5)}*{x_variables[idx]}"
        TeXclean_eq = str(TeXclean_eq)+ f" + {lin_slope[idx].round(decimals=5)}*{x_variables[idx]}"

    if len(x_variables) > 2:
        print('4D+ not plottable')

    elif len(x_variables) == 2:
     
        fig1 = plt.figure(figsize=(5,5))
        ax = fig1.add_subplot(111, projection='3d')
        ax.scatter3D(X.iloc[:,0], X.iloc[:,1], P)
        ax.set_xlabel(x_variables[0], fontweight ='bold') 
        ax.set_ylabel(x_variables[1], fontweight ='bold') 
        ax.set_zlabel(f"{y_variable} (p)", fontweight ='bold', rotation=90)
        ax.zaxis.labelpad=-0.5
    
        x_plane = np.array(X.iloc[:,0])
        y_plane = np.array(X.iloc[:,1])
        
        xi = np.linspace(min(x_plane), max(x_plane))  # necessary transformation for correct gradient plotting
        yi = np.linspace(min(y_plane), max(y_plane))

        x_plane, y_plane = np.meshgrid(xi, yi)
        z_plane = 1/(1+np.exp(-x_plane * np.ravel(lin_slope[0])-y_plane * np.ravel(lin_slope[1])-(lin_intercept)))
        
        ax.plot_surface(x_plane, y_plane, z_plane, rstride=1, cstride=1, linewidth=1, vmin=0, vmax=1.1, cmap=cm.magma)
        
        ax.set_title(TeXclean_eq)
        
    else:

        fig1 = plt.figure()
        plt.scatter(X, P, color="black")
        S = 1/(1+np.exp(-X * np.ravel(logit_model.coef_)-(logit_model.intercept_)))

        # convert to arrays and sort so matplotlib plots correctly
        X = np.array(X)
        S = np.array(S)
        X, S = zip(*sorted(zip(X, S)))

        plt.plot(X, S)
        plt.xlabel(f"{x_variables[0]} (p)")
        plt.ylabel(y_variable)
        plt.title(TeXclean_eq)

    print(clean_eq)

    #plt.show()
    if plot==False:
        plt.close()



#logit(df, 'Survived ~ SibSp + Fare', plot=True)

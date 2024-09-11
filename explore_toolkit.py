
import re
import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        ax.scatter3D(X.iloc[:,0], X.iloc[:,1], Y, s=12)
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
        plt.scatter(X.iloc[:,0], Y, s=10)
        plt.plot(X, lin_intercept+lin_slope*X, c='black')  # plot equation of line
        plt.xlabel(x_variables[0])
        plt.ylabel(y_variable)
        plt.title(clean_eq)

    print(clean_eq)

    
    if plot==False:
        plt.close()






def logit(dataframe, model, plot = False):
    from matplotlib import cm
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
        
        ax.plot_surface(x_plane, y_plane, z_plane, rstride=1, cstride=1, linewidth=1, vmin=0, vmax=1.1, cmap=matplotlib.cm.magma)
        
        ax.set_title(TeXclean_eq)
        
    else:

        fig1 = plt.figure()
        plt.scatter(X, P, s=12)
        S = 1/(1+np.exp(-X * np.ravel(logit_model.coef_)-(logit_model.intercept_)))

        # convert to arrays and sort so matplotlib plots correctly
        X = np.array(X)
        S = np.array(S)
        X, S = zip(*sorted(zip(X, S)))

        plt.plot(X, S, c="black")
        plt.xlabel(f"{x_variables[0]} (p)")
        plt.ylabel(y_variable)
        plt.title(TeXclean_eq)

    print(clean_eq)

    #plt.show()
    if plot==False:
        plt.close()

#logit(df, 'Survived ~ SibSp + Fare', plot=True)





def elbow(dataframe, variables):

    X = dataframe[variables]

    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(1)
    
    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

# elbow(df, ['sniperKillsPC', 'rifleKillsPC', 'winrate'])





def kmeansclusters(dataframe, variables, n_clusters, plot=True, append=False, spit=False):

    
    X = dataframe[variables]
    kmeans = KMeans(n_clusters=n_clusters)  # set n_clusters based on elbow from above
    cluster_model_2k = kmeans.fit(X)
    latest_cluster = cluster_model_2k.labels_
    dataframe.insert(1, 'latest_cluster', latest_cluster)

    color = iter(matplotlib.colormaps['Dark2'](np.linspace(0, n_clusters / 8, n_clusters)))
    to_hex = lambda a : 'rgba({:.0f},{:.0f},{:.0f},{:.2f})'.format(a[0]*255, a[1]*255, a[2]*255, a[3])


    PLOT = go.Figure()

    plot_params = {'mode' : 'markers', 
                   'marker_size' : 9, 
                   'marker_line_width' : 0.4
                   }
    
    if len(variables) == 1:

        dataframe['dummy'] = 0

        for i in range(0, n_clusters):
            
            cn_hex = to_hex(next(color))
            
            PLOT.add_trace(go.Scatter(x = dataframe.loc[dataframe['latest_cluster'] == i, variables[0]],
                                      y = dataframe['dummy'],
                                      marker = dict(color = cn_hex),
                                      name = f'Cluster {i}',
                                      mode = 'markers',
                                      marker_size = 7,
                                      marker_line_width = 0.25
                                      ))
            
            PLOT.update_layout(width = 850, height = 800, autosize = True, showlegend = True,
                                xaxis=dict(title = variables[0], titlefont_color = 'black'))
        
        dataframe.drop(["dummy"], axis=1)

    if len(variables) == 2:
        # a one-liner from seaborne: facet = sns.lmplot(dataframe, x=variables[0], y=variables[1], hue='latest_cluster', fit_reg=False, legend=True)
        for i in range(0, n_clusters):
            
            cn_hex = to_hex(next(color))

            PLOT.add_trace(go.Scatter(x = dataframe.loc[dataframe['latest_cluster'] == i, variables[0]],
                                      y = dataframe.loc[dataframe['latest_cluster'] == i, variables[1]],
                                      marker = dict(color = cn_hex),
                                      name = f'Cluster {i}',
                                      **plot_params
                                      ))
            
        PLOT.update_layout(width = 850, height = 800, autosize = True, showlegend = True,
                                xaxis=dict(title = variables[0], titlefont_color = 'black'),
                                yaxis=dict(title = variables[1], titlefont_color = 'black'))


    if len(variables) == 3:
        
        for i in range(0, n_clusters):

            cn_hex = to_hex(next(color))

            PLOT.add_trace(go.Scatter3d(x = dataframe.loc[dataframe['latest_cluster'] == i, variables[0]],
                                        y = dataframe.loc[dataframe['latest_cluster'] == i, variables[1]],
                                        z = dataframe.loc[dataframe['latest_cluster'] == i, variables[2]],
                                        name = f'Cluster {i}',
                                        marker = dict(color = cn_hex),
                                        mode = 'markers',
                                        marker_size = 5,
                                        marker_line_width = 1))
            
        PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                            scene = dict(xaxis=dict(title = variables[0], titlefont_color = 'black'),
                                        yaxis=dict(title = variables[1], titlefont_color = 'black'),
                                        zaxis=dict(title = variables[2], titlefont_color = 'black')),
                                        scene_aspectmode='cube',
                                        font = dict(family = "Gilroy", color  = 'black', size = 12),
                                        scene_camera = dict(eye=dict(x=1.5, y=1.5, z=1)))
        

    if plot == True:
        PLOT.show()
    
    if spit == True:
        return dataframe['latest_cluster']

    if append == False: 
        dataframe.drop(["latest_cluster"], axis=1, inplace=True) # mind the inplace
    

# a = kmeansclusters(df, ['sniperKillsPC', 'winrate'], n_clusters=8, append=False, spit=True, plot=True)
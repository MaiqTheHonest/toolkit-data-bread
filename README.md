A toolkit for quickly getting insights from `pandas` Dataframes via common regression methods and their visualizations. Effectively a combination of `R`'s `glm` and `ggplot` functionalities as a single Python model call with the addition of one-liner clustering methods.

Currently supports:
1. (multiple) linear regression with 2D, 3D plots
2. (multiple) logistic regression; robust for both binary and proportional (i.e. 0 < y < 1) regressands + 2D, 3D plots
3. k-means clustering with inertia plots to determine optimal cluster number; plots in 1D, 2D, 3D but clustering for any number of variables

<br />

Example usage: <br />

## Linear regression:

```python
from explore_toolkit import lm 
df = pd.read_csv("titanic.csv") 
df = df.fillna({'Age': df['Age'].median()})
``` 
<br />

```python
lm(df, 'Fare ~ Age', plot=True)
lm(df, 'Fare ~ Age + SibSp', plot=True)
```

2D | 3D
:-------------------------:|:-------------------------:
![image](https://github.com/user-attachments/assets/bd603e03-14a7-4a49-a9f6-a02493cb0ca3)  |  ![image](https://github.com/user-attachments/assets/7f195e3d-19a6-42dd-a11c-b0c446471570)






## Logistic regression (for binary and proportional predictors):
 
```python
from explore-toolkit import logit 
td = pd.read_csv('ReedfrogPred.csv') #  propsurv is between 0 and 1, but also works if binary
```
<br />



```python
logit(td, 'propsurv ~ surv', plot=True)
logit(td, 'propsurv ~ density + surv', plot=True)
```

2D | 3D
:-------------------------:|:-------------------------:
![image](https://github.com/user-attachments/assets/52a2e7e5-24a4-4d9b-ac81-167623d82448)  |  ![image](https://github.com/MaiqTheHonest/toolkit-data-bread/assets/60844551/8b728f55-416a-41ad-9ac0-fe7d0a86c4f7)


## k-means clustering and the "elbow" rule:
 
```python
from explore_toolkit kmeansclusters, elbow
df = pd.read_csv('Iris.csv')
```
<br />

Here I am using the very popular `iris` dataset

```python
elbow(df, ['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm'])
```

<img src="https://github.com/user-attachments/assets/331425aa-ba75-4899-9033-ef09ee997406" width="400" >

<br />

<br />

and now clustering with the optimal `n_clusters = 3` :


```python
kmeansclusters(df, ['SepalLengthCm', 'SepalWidthCm', ], n_clusters=3, plot=True, append=True, spit=False)
kmeansclusters(df, ['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm'], n_clusters=3, plot=True, append=True)
```

2D | 3D
:-------------------------:|:-------------------------:
![image](https://github.com/user-attachments/assets/7d92e4aa-90cb-45c8-b5f7-f12836e65f4c)  |  ![image](https://github.com/user-attachments/assets/c33b47f4-febe-47e8-b62b-a7bf087ce9f3)


Use `spit = True` to return the standalone column of cluster numbers and use `append = True` to insert the column of cluster numbers into the analysed dataframe (position 1)





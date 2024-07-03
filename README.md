A toolkit for quickly getting insights from `pandas` Dataframes via common regression methods and their visualizations. Effectively a combination of `R`'s `glm` and `ggplot` functionalities, as a single Python model call.
Currently supports:
1. (multiple) linear regression with 2D, 3D plots
2. (multiple) logistic regression; robust for both binary and proportional (i.e. 0 < y < 1) regressands + 2D, 3D plots

<br />

Example usage: <br />

## Linear regression:

```python
from explore-toolkit import lm 
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
![image](https://github.com/MaiqTheHonest/toolkit-data-bread/assets/60844551/68580914-a759-4841-b745-b8a9d58e5f5f) |  ![image](https://github.com/MaiqTheHonest/toolkit-data-bread/assets/60844551/fcde0a0e-89c3-4a2d-a334-369a547b2729)


## Logistic regression (for binary and proportional predictors):
 
```python
from explore-toolkit import logit 
td = pd.read_csv('ReedfrogPred.csv') #  propsurv is between 0 and 1, but would work if it was binary as well
```
<br />



```python
logit(td, 'propsurv ~ surv', plot=True)
logit(td, 'propsurv ~ density + surv', plot=True)
```

2D | 3D
:-------------------------:|:-------------------------:
![image](https://github.com/MaiqTheHonest/toolkit-data-bread/assets/60844551/40f73790-2a78-4d7c-bf9d-872c15321a1c)  |  ![image](https://github.com/MaiqTheHonest/toolkit-data-bread/assets/60844551/8b728f55-416a-41ad-9ac0-fe7d0a86c4f7)



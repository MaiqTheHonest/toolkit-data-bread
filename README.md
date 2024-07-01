A toolkit for quickly getting insights from `pd.Dataframes`, combining features of `sklearn` and `matplotlib` into a single model call, similar to how it is done in `R`.

<br />

Example usage: <br />
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

 

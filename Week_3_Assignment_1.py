# %% [markdown]
# 1a. Make a histogram of the variable Sepal.Width.

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True) 
df = iris.frame
print(df['sepal width (cm)'])
sns.histplot(df['sepal width (cm)'],color='blue',edgecolor='black')
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width(cm)')
plt.ylabel('Frequency')
plt.show()



# %% [markdown]
# 1b. Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
# 
# The sepal width values mean is expected to be slightly higher than the median. The histogram shows slightly-right skewed or positively skewed because most values are clusters around 3.0 cm and few flowers have larger sepal width around 4 cm that create a right tail.

# %% [markdown]
# 1c. Confirm your answer to #1b by actually finding these values.

# %%
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True) 
df = iris.frame
print(df['sepal width (cm)'])
mean=np.mean(df['sepal width (cm)'])
median=np.median(df['sepal width (cm)'])
print(f"Mean:{mean}")
print(f"Median:{median}")
sns.histplot(df['sepal width (cm)'],color='blue',kde=True)
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width(cm)')
plt.ylabel('Frequency')
plt.show()



# %% [markdown]
# 1d. Only 27% of the flowers have a Sepal.Width higher than ________ cm.

# %%
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True) 
df = iris.frame
sw=df['sepal width (cm)']
cutoff=sw.quantile(0.73)
pct_above=(sw>cutoff).mean()*100
print(f"cutoff(cm={cutoff:.3f})")
print(f"only 27% of the flowers have a sepal width higher than{cutoff:.3f}")

# %% [markdown]
# 1e. Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).

# %%
pairs = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal length (cm)', 'petal width (cm)'),
    ('sepal width (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)'),
    ('petal length (cm)', 'petal width (cm)')
]
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(pairs, 1):
    plt.subplot(2, 3, i)
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")
plt.tight_layout()
plt.show()


# %% [markdown]
# 1f. Based on #1e, which two variables appear to have the strongest relationship? And which two appear to have the weakest relationship?
# 
# The scatterplots in #1e indicate that petal length and petal width share the closest relationship, with a clear, almost linear pattern, reflecting a strong positive correlation. In contrast, sepal width and petal length are more dispersed, showing a weaker connection.
# 
# These impressions are supported by correlation coefficients. The correlation between petal length and petal width is 0.963, the strongest among all pairs. Conversely, sepal width and petal length have the weakest association with a correlation of −0.428. 
# 
# 

# %%
pairs = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal length (cm)', 'petal width (cm)'),
    ('sepal width (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)'),
    ('petal length (cm)', 'petal width (cm)')
]
for x, y in pairs:
    corr = df[x].corr(df[y])
    print(f"{x} vs {y}: {corr:.3f}")

# %% [markdown]
# 2a. Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], 
        "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
bins=np.arange(3.3,max(data['weight'])+0.3,0.3)
sns.histplot(data['weight'],bins=bins,color='skyblue')
plt.title('Histogram of plant growth')
plt.xlabel('plant weight')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# 2b. Make boxplots of weight separated by group in a single graph.

# %%
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], 
        "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
sns.boxplot(x='group',y='weight',hue='group',data=data,palette='coolwarm')
plt.title('Boxplots of plant weight by treatment group')
plt.show()

# %% [markdown]
# 2c. Based on the boxplots in #2b, approximately what percentage of the "trt1" weights are below the minimum "trt2" weight?
# 
# From the boxplot, the minimum weights of trt2 is near 5.0 and most of the trt1 weights lies below that value. Approximately 75-80% of the trt1 distribution is below the minimum weights of trt2.

# %% [markdown]
# 2d. Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.

# %%

min_trt2 = PlantGrowth[PlantGrowth["group"] == "trt2"]["weight"].min()

# Get all trt1 weights
trt1_weights = PlantGrowth[PlantGrowth["group"] == "trt1"]["weight"]

# Calculate exact percentage
percentage = (trt1_weights < min_trt2).sum() / len(trt1_weights) * 100

print("Minimum trt2 weight:", min_trt2)
print("Percentage of trt1 below min(trt2):", percentage, "%")

# %% [markdown]
# 2e. Only including plants with a weight above 5.5, make a barplot of the variable group. Make the barplot colorful using some color palette 

# %%
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], 
        "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
heavy_plants = PlantGrowth [PlantGrowth ['weight'] > 5.5]
sns.countplot(x='group',hue='group',legend=False,data=heavy_plants, palette='tab20')
plt.title('Number of heavy Plants by Treatment Group')
plt.xlabel('Treatment')
plt.ylabel('Number of Plants')
plt.show()

# %%




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
iris = pd.read_csv('iris.csv')
print(iris.head())

iris['class'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',figsize=(10,8))
plt.title("Pie Chart IRIS Classes")
plt.ylabel("")
plt.savefig("plots/Pie_Chart_class.png")

iris.boxplot(by="class", figsize=(12, 8))
plt.savefig("plots/box_Chart_class.png")

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.violinplot(x='class',y='sepal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='class',y='sepal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='class',y='petal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='class',y='petal_width',data=iris)
plt.savefig("plots/class_violin_Chart_X.png")


sns.pairplot(iris, hue='class')
plt.savefig("plots/class_pair_Chart.png")

iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.savefig("plots/class_hist_Chart_X.png")

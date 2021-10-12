from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class Perceptron(object):
    """
    Perceptron收斂的條件有下面兩個，否則會無法停止。
    1. 分開的兩類可用線性分離。
    2. 學習速率要小。
    Input:
    X
    y
    eta
    epoch_iter:
    Output:
    object
    """

    def __init__(self, eta=0.01, epoch_iter=10):
        """
        初始化感知器的學習速率與迭代次數
        Parameters:
        1.eta: float
          learning rate (0-1)學習速率
        2.epoch_iter: integer
          Passes over the training dataset, iteration 迭代次數
        """
        self.eta = eta
        self.epoch_iter = epoch_iter

    def fit_model(self, X, y):
        """
        利用訓練集(Traiining DataSet)進行建模
        Parameters:
        1.X: array, shape = [n_samples, n_featrues]
          Training data.輸入的訓練資料格式。
          n_samples代表有n個樣本，n_features代表有n個特徵。
        2.y: array, shape = [n_sampels]
          Traget data.輸出的訓練結果。
        3.weight_fit: 1d-array
          Weights after fitting.更新後的權重
        4.errors_result: list
          Number of misclassifications in every epoch.
          每個迴圈中錯誤分類的數量
        """
        self.weight_fit = np.zeros(1+X.shape[1])
        self.errors_result = []

        for _ in range(self.epoch_iter):
            errors = 0
            for x_i, y_i in zip(X, y):
                update_weight = self.eta*(y_i-self.predict(x_i))
                self.weight_fit[1:] += update_weight * x_i
                self.weight_fit[0] += update_weight
                errors += int(update_weight != 0.0)
            self.errors_result.append(errors)
        return self

    def net_input(self, X):
        """
        Calcuate net input
        z = w0x0+w1x1+...+wmxm
        x0 = 1
        """
        return np.dot(X, self.weight_fit[1:]) + self.weight_fit[0]

    def predict(self, X):
        """
        Return class label after sign step.
        fi(z) =  1   z >= 0
                -1   z <  0
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plotScatter(dataframe, x_string, y_string, colors):
    """
    Input:
    1.dataframe: pd.Datframe
    input data
    2.x_string: list
      scatter plot in x-axis and by user defined
    3.y_string: list
      scatter plot in y-axis and by user defined
    4.colors: dictionary
      check the each color in category
    Parameter:
    1.fig_size default(10,10)
    2.lenged_size default(8)
    Output:
    1.figure
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(dataframe[x_string], dataframe[y_string],
                c=dataframe['species'].map(colors))
    plt.xlabel(x_string)
    plt.ylabel(y_string)
    # colr mpatches list append
    patchlist = []
    for key in colors:
        data_key = mpatches.Patch(color=colors[key], label=key)
        patchlist.append(data_key)
        plt.legend(handles=patchlist, fontsize=8)
    plt.show()


def plot_binary_classification(X, y, classifier, plot_step=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # x軸 選擇的兩個參數先x在y
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, plot_step),
                           np.arange(x2_min, x2_max, plot_step))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# 直接透過Webj網路的UCI Machine Learning 拿取DataSet
column_name = ["sepal length", "sepal width",
               "petal length", "petal width", "species"]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                 "iris/iris.data", header=None)
df.columns = column_name
df.head()
# 確認類別分類
iris_set = set(df['species'])
iris_set = list(iris_set)
# 由於Percepton.py建置是分兩類且要盡量能夠用線性方式去分隔兩群
# 先看數據決定挑選的兩群
# Dataframe欲分析的X軸
x_string = ["sepal length", "petal length", "sepal length",
            "sepal width"]
# Dataframe欲分析的Y軸
y_string = ["sepal width", "petal width", "petal length",
            "petal width"]
# 圖例X軸
x_label = ["sepal length(cm)", "petal length(cm)",
           "sepal length(cm)", "sepal width(cm)"]
# 圖例Y軸
y_label = ["sepal width(cm)", "petal width(cm)",
           "petal length(cm)", "petal width(cm)"]
colors = {iris_set[0]: "red", iris_set[1]: "green", iris_set[2]: "blue"}

for i in range(len(x_string)):
    plotScatter(df, x_string[i], y_string[i], colors)
    plt.show()

# 依據圖例的結果選擇setosa和versicolor，前100筆進行建模
# 選擇的參數為petal length和sepal length作為訓練
# y為非類setosa為-1，versicolor為1
# 要加.value才會變成ndarray的形式
X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

# 建模
model = Perceptron(eta=0.01, epoch_iter=10)
model.fit_model(X, y)
plt.plot(range(1, len(model.errors_result)+1), model.errors_result)
plt.xticks(range(1, 11, 1))
plt.xlabel("Epochs")
plt.ylabel("No. Misclassifications")
plt.title("Each Epoch of No. Misclassifications")
plt.show()

# 結果圖
plot_binary_classification(X, y, classifier=model)
plt.xlabel("sepal length[cm]")
plt.ylabel("petal length[cm]")
plt.legend(loc='upper left')
plt.show()

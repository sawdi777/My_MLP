import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MLP_Tools:
    def __init__(self, data_path, label_index=-1, seed=42, method='zscore', test_size=0.2):
        self.data_path = data_path
        self.label_index = label_index
        self.seed = seed
        self.method = method
        self.test_size = test_size

        self.df = self.load_data()
        self.x, self.y = self.shuffle_and_extract()
        self.preprocessed_x = self.preprocess_features()
        self.x_train, self.y_train, self.x_test, self.y_test = self.train_test_split()

    def load_data(self):
        df = pd.read_excel(self.data_path, header=None)
        df = pd.DataFrame(df)
        return df

    def shuffle_and_extract(self):
        np.random.seed(self.seed)
        df = self.df.sample(frac=1).reset_index(drop=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        x = df.iloc[:, :self.label_index]
        y = df.iloc[:, self.label_index]
        return x, y

    def visualize_data(self):
        df = pd.concat([self.x, self.y], axis=1)

        sns.pairplot(df, hue=self.y.name)
        plt.suptitle('Features PairPlot', size=20)
        plt.show()

        plt.title("Correlation Matrix HeatMap")
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()

        for col in self.x.columns:
            sns.distplot(self.x[col], kde=False)
            plt.title(f'Distribution of {col}')
            plt.show()

    def preprocess_features(self):
        x = self.x.copy()

        if self.method == 'zscore':
            means = x.mean()
            stds = x.std()
            x = (x - means) / stds

        elif self.method == 'minmax':
            mins = x.min()
            maxs = x.max()
            x = (x - mins) / (maxs - mins)

        elif self.method == 'log':
            x = np.log1p(x)

        else:
            raise ValueError(f"Invalid scaling method: {self.method}")

        return x

    def train_test_split(self):
        split_index = int(len(self.preprocessed_x) * (1 - self.test_size))

        x_train = self.preprocessed_x[:split_index]
        x_test = self.preprocessed_x[split_index:]
        y_train = self.y[:split_index]
        y_test = self.y[split_index:]

        return x_train, y_train, x_test, y_test



mlp_tools = MLP_Tools('boston.csv')
print(mlp_tools.x_train)
print(mlp_tools.y_train)
print(mlp_tools.x_test)
print(mlp_tools.y_test)
mlp_tools.visualize_data()
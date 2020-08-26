import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from logistic_regression import BinaryLogisticRegression

CSV_PATH = "./titanic_my.csv"


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def preprocess_dataset(dataset):
    # Encode gender using one-hot encoding
    one_hot = pd.get_dummies(dataset['Gender'])
    dataset = dataset.join(one_hot).drop('Gender', axis=1)

    # Fill NaN values
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(dataset)
    dataset = pd.DataFrame(imputer.transform(dataset), columns=dataset.columns)

    # Split to features and labels
    x = dataset.drop(['Survived'], axis=1)
    y = dataset['Survived']

    # Scale features
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)

    # Split to train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


def main():
    dataset = load_dataset(CSV_PATH)

    x_train, x_test, y_train, y_test = preprocess_dataset(dataset)

    regressor = BinaryLogisticRegression()

    regressor.fit(x_train, y_train)

    regressor.predict()

    regressor.probas(x_test)


if __name__ == '__main__':
    main()

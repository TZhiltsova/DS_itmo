import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def prepare_data():
    def room_counter(x):
        if '-' in x:
            n = x[0]
        else:
            n = 0
        return n

    pd.options.display.max_columns = None

    price = pd.read_csv('realty_data.csv')
    price['rooms'] = price['product_name'].apply(lambda x: room_counter(x))
    price_filt = price[['price', 'lat', 'lon', 'rooms']]

    return price_filt

def train_model(train):
    y = train[['price']]
    X = train[['lat', 'lon', 'rooms']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2025, test_size=0.25)

    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model
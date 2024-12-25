import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import Pool


def prepare_data(data: pd.DataFrame, test_size: float, random_state: int = 42) -> pd.DataFrame:
    
    #drop unnecessary columns
    unnecessary_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    data.drop(unnecessary_cols, axis=1, inplace=True)
    
    # encode categorical variable
    label_encoder = LabelEncoder()
    data['type'] = label_encoder.fit_transform(data['type'])

    # split data into features (X) and target (y)
    X = data.drop(['isFraud'], axis = 1)
    y = data['isFraud']

    X_scaled = scale_feautres(X)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    print(f'Train size is: {X_train.shape} \nTest size is: {X_test.shape}')

    return X_train, X_test, y_train, y_test


def scale_feautres(data: pd.DataFrame) -> pd.DataFrame:
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return pd.DataFrame(scaled_data, columns=data.columns)


def get_pool(X, y):
    
    #make pool
    data_pool = Pool(X, y)
    
    return data_pool

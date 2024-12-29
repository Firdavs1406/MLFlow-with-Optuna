from catboost import CatBoostClassifier

def train_model(X_train, y_train, params, train_data):

    model = CatBoostClassifier(**params, verbose=False)
    model.fit(train_data)

    return model

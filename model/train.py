from catboost import CatBoostClassifier

def train_model(train_data):

    model = CatBoostClassifier(
        iterations = 300,
        learning_rate = 0.01,
        depth = 3,
        verbose=False
        )

    model.fit(train_data)

    return model

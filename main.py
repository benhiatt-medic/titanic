import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def format_name(data):
    data['Surname'] = data.Name.apply(lambda x: x.split(' ')[0])
    data['NamePrefix'] = data.Name.apply(lambda x: x.split(' ')[1])
    return data


def simplify_ages(data):
    data.Age = data.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(data.Age, bins, labels=group_names)
    data.Age = categories
    return data


def simplify_fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(data.Fare, bins, labels=group_names)
    data.Fare = categories
    return data


def simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data


def fill_embarked(data):
    data.Embarked = data.Embarked.fillna('N')
    return data


def drop_features(data):
    return data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


def transform_features(data):
    data = format_name(data)
    data = simplify_ages(data)
    data = simplify_fares(data)
    data = simplify_cabins(data)
    data = fill_embarked(data)
    data = drop_features(data)
    return data


def encode_features(data1, data2, features):
    data_combined = pd.concat([data1[features], data2[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data_combined[feature])
        data1[feature] = le.transform(data1[feature])
        data2[feature] = le.transform(data2[feature])
    return data1, data2


def prepare_data():
    training_data = transform_features(pd.read_csv('data/train.csv'))
    test_data = transform_features(pd.read_csv('data/test.csv'))

    training_data, test_data = encode_features(
        training_data,
        test_data,
        ['NamePrefix', 'Surname', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked']
    )

    X_train, y_train = training_data.drop('Survived', axis=1), training_data.Survived
    X_test, y_test = test_data, pd.read_csv('data/survived.csv').Survived

    return X_train, y_train, X_test, y_test


def random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier()

    parameters = {
        'n_estimators': [4, 6, 9],
        'max_features': ['log2', 'sqrt','auto'],
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 3, 5, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }

    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)

    clf = grid_obj.best_estimator_

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    print(accuracy_score(y_test, predictions))


def neural_net(X_train, y_train, X_test, y_test):
    y_train, y_test = y_train.values, y_test.values

    X_train = preprocessing.scale(X_train)
    y_train = y_train.reshape(y_train.size, 1)
    y_test = y_test.reshape(y_test.size, 1)

    print(X_train.shape, y_train.shape)

    model = Sequential([
        Dense(400, activation='relu', input_dim=X_train.shape[1]),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    epochs = 10
    batch_size = 10

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

    print(acc)


if __name__ == '__main__':
    random_forest(*prepare_data())

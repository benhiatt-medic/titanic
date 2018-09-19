import pytest

import main
import pandas as pd
import math


def setup_module():
    """Initializes the training and prediction data to be tested."""
    global training_data, test_data, X_train, y_train, X_test, y_test
    training_data = main.transform_features(pd.read_csv('data/train.csv')).drop('Survived', axis=1)
    test_data = main.transform_features(pd.read_csv('data/test.csv'))
    X_train, y_train, X_test, y_test = main.prepare_data()


def test_sizes_equivalent():
    """Input and output for a set of data should have the same number of rows. Input data should have 12 features."""
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    for X in X_train, X_test:
        assert X.shape[1] == 12


def test_columns_equivalent():
    """The fields of the input data should be identical."""
    for col in X_train:
        assert col in X_test

    for col in X_test:
        assert col in X_train


def test_none():
    """There should be no empty values in any processed data."""
    for col in X_train:
        for e in X_train[col]:
            assert e is not None
        for e in X_test[col]:
            assert e is not None

    for e in y_train:
        assert e is not None

    for e in y_test:
        assert e is not None


@pytest.mark.xfail
def test_surnames():
    """Every row should have a surname that is not blank.
    Raw and encoded input data should have the same number of distinct surnames."""
    for surname in training_data.Surname:
        assert len(surname) != 0

    for surname in test_data.Surname:
        assert len(surname) != 0

    assert len(set(X_train.Surname)) == len(set(training_data.Surname))
    assert len(set(X_test.Surname)) == len(set(test_data.Surname))


@pytest.mark.xfail
def test_prefixes():
    """Every row should have a name prefix that is not blank.
    Raw and encoded input data should have the same number of distinct name prefix values."""
    for prefix in training_data.NamePrefix:
        assert len(prefix) != 0

    for prefix in test_data.NamePrefix:
        assert len(prefix) != 0

    assert len(set(X_train.NamePrefix)) == len(set(training_data.NamePrefix))
    assert len(set(X_test.NamePrefix)) == len(set(test_data.NamePrefix))


def test_ages():
    """Raw and encoded input data should have the same number of distinct Age values as the number of age categories."""
    assert len(set(X_train.Age)) == len(set(training_data.Age)) == len(['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'])
    assert len(set(X_test.Age)) == len(set(test_data.Age)) == len(['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'])


def test_fares():
    """Raw and encoded input data should have the same number of distinct Fare values as the number of fare categories."""
    assert len(set(X_train.Fare)) == len(set(training_data.Fare)) == 5
    assert len(set(X_test.Fare)) == len(set(test_data.Fare)) == 5


def test_cabins():
    """Raw Cabin values should be characters.
    Raw and encoded input data should have the same number of distinct Cabin values."""
    for cabin in training_data.Cabin:
        assert cabin.isalpha()

    for cabin in test_data.Cabin:
        assert cabin.isalpha()

    assert len(set(X_train.Cabin)) == len(set(training_data.Cabin)) <= 26
    assert len(set(X_test.Cabin)) == len(set(test_data.Cabin)) <= 26


def test_embarked():
    """Raw port values should be certain characters.
    Raw and encoded input data should have the same number of distinct port values."""
    for port in training_data.Embarked:
        assert port in ('C', 'N', 'Q', 'S')

    for port in test_data.Embarked:
        assert port in ('C', 'N', 'Q', 'S')

    assert len(set(X_train.Embarked)) == len(set(training_data.Embarked)) <= 4
    assert len(set(X_test.Embarked)) == len(set(test_data.Embarked)) <= 4


def test_dropped():
    """Dropped fields should not be in the processed data."""
    for col in ('PassengerId', 'Name', 'Ticket'):
        assert col not in X_train
        assert col not in X_test


def test_nan():
    """There should be no Nan values in any processed data."""
    for col in X_train:
        for e in X_train[col]:
            assert not math.isnan(e)
        for e in X_test[col]:
            assert not math.isnan(e)

    for e in y_train:
        assert not math.isnan(e)

    for e in y_test:
        assert not math.isnan(e)


def teardown_module():
    """Delete the data variables."""
    global training_data, test_data, X_train, y_train, X_test, y_test
    del training_data, test_data, X_train, y_train, X_test, y_test

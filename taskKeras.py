#!/usr/bin/env python

'''
Usage- python -W ignore taskKeras.py [path to test data file]
if the test data file is in the same directory, just the name of the file will work
label 0 for abnormal, 1 for normal
'''
from __future__ import print_function

import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import sys


def neuralNetKeras(training_data, training_labels, test_data, test_labels, n_dim):
    print("Initiating Neural Network")
    seed = 7
    numpy.random.seed(seed)
    model = Sequential()
    model.add(Dense(1024, input_dim=n_dim,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, epochs=1000, batch_size=50)
    scores = model.evaluate(test_data, test_labels)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model.save('trained_neural_net.h5')
    return None


def factorize(array):
    uniques, labels = pandas.factorize(array)
    return uniques


def convert(array):
    for x in range(array.shape[0]):
        if array[x][0] == "$":
            array[x] = array[x][1:]
        elif array[x] == "#VALUE!":
            array[x] = 0
    return array


def prepare(data):
    data = data.drop('Loan ID', 1)
    data = data.drop('Customer ID', 1)
    data = data.fillna(0)
    data['Loan Status'] = factorize(data['Loan Status'])
    data['Term'] = factorize(data['Term'])
    data['Years in current job'] = factorize(data['Years in current job'])
    data['Home Ownership'] = factorize(data['Home Ownership'])
    data['Purpose'] = factorize(data['Purpose'])
    data['Monthly Debt'] = convert(data['Monthly Debt'])
    data['Maximum Open Credit'] = convert(data['Maximum Open Credit'])
    return data


def main():
    try:
        src = sys.argv[1]
    except Exception as ex:
        src = 0
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    training_data = prepare(pandas.read_csv("loan.csv", low_memory=False))
    if (src == 0):
        print("Test Data file not found, setting training data as test data")
        test_data = training_data
    else:
        print("Loading Test Data")
        test_data = prepare(pandas.read_csv(src, low_memory=False))

    training_labels = training_data['Loan Status'].values.astype(int)
    training_data = training_data.drop('Loan Status', 1)
    training_data = training_data.values.astype(float)

    test_labels = test_data['Loan Status'].values.astype(int)
    test_data = test_data.drop('Loan Status', 1)
    test_data = test_data.values.astype(float)

    encoder = LabelEncoder()
    encoder.fit(training_labels)
    training_labels = encoder.transform(training_labels)
    encoder.fit(test_labels)
    test_labels = encoder.transform(test_labels)
    neuralNetKeras(training_data, training_labels, test_data, test_labels, 16)


if __name__ == '__main__':
    main()

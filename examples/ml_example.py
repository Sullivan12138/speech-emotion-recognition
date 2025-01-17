"""
This example demonstrates how to use `NN` model ( any ML model in general) from
`speechemotionrecognition` package
"""
import sys
from common import extract_data


from mlmodel import NN, SVM, RF
from utilities import get_feature_vector_from_mfcc


def ml_example():
    to_flatten = True
    x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
    model = SVM()
    print('Starting', model.name)
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    filename = '../dataset/Sad/09b03Ta.wav'
    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual 3')


if __name__ == "__main__":
    ml_example()

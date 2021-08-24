from scr.model import BreastCancerModel
from scr.utils import plot_model

import pandas as pd


# load data
X_train = pd.read_csv("data/xtrain.csv", header=None)
Y_train = pd.read_csv("data/ytrain.csv", header=None)
X_test = pd.read_csv("data/xtest.csv", header=None)
Y_test = pd.read_csv("data/ytest.csv", header=None)


# load model
classifier = BreastCancerModel().load_model()
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

# validate (predict on test set)
T = 0.5  # classification threshold
Y_pred = classifier.predict(X_test)
Y_pred = [1 if y >= 0.5 else 0 for y in Y_pred]

total = 0
correct = 0
wrong = 0
for i in Y_pred:
  total = total+1
  if (Y_test.at[i, 0] == Y_pred[i]):
    correct = correct+1

  else:
    wrong = wrong+1


print("Total " + str(total))
print("Correct " + str(correct))
print("Wrong " + str(wrong))

# get value in validation (test)
value_test = classifier.evaluate(X_test, Y_test)
print("Loss in validation = %.4f" % value_test[0])
print("Accucary in validation = %.4f" % value_test[1])
from scr.utils import plot_performance
from scr.model import BreastcancerModel
import pandas as pd
#load data
X_train = pd.read_csv("data/xtrain.csv", header=None)
Y_train = pd.read_csv("data/ytrain.csv", header=None)
X_test = pd.read_csv("data/xtest.csv", header=None)
Y_test = pd.read_csv("data/ytest.csv", header=None)
#load model
classifier = BreastcancerModel().load_model()
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ["accuracy"])
#train
history=classifier.fit(X_train, Y_train, batch_size = 1, epochs = 20 )
#validate (predict on test set)
#T=0.5
#Y_pred = classifier.predict(X_test)
#Y_pred = [ 1 if y=T   else 0 for y in Y_pred ]
plot_performance(history)
"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

"""
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

n_samples = len(digits.images)
n_training_data = n_samples // 2
print("Number of samples: ", n_samples)
print("Number of training data: ", n_training_data)

# definimos las features.-
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
features = digits.images.reshape((n_samples, -1))
# establecemos los labels para cada conjunto de features.-
labels = digits.target

# con este indice falla
indexToPredict = 5
# con este indice funciona
# indexToPredict = 10

# creamos el classifier -> support vector machine.-
clf = svm.SVC(gamma=0.001)

# entrenamos al classifier.-
# We learn the digits on the first half of the digits
clf.fit(features[:n_training_data], labels[:n_training_data])

# Now predict the value of the digit:
expected = labels[indexToPredict]
print("Expected value: ", expected)

# prediccion
predicted = clf.predict([features[indexToPredict]])
print("Predicted value: ", predicted[0])

plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(digits.images[indexToPredict], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Expected: %i - Prediction: %i' % (expected, predicted))

plt.show()

"""
========================
Metrics - Train and Test
========================

Toma la mitad de los datos para entrenar el classifier
La otra mitad la usa para predecir valores

Luego compara los labels de test con las predicciones para determinar el grado de exactitud de nuestro modelo.-

https://es.wikipedia.org/wiki/Iris_flor_conjunto_de_datos

"""
print(__doc__)

from sklearn import datasets

iris = datasets.load_iris()
# X tiene las features
X = iris.data
# y tiene los labels para cada conjunto de features
y = iris.target

# separamos la mitad de los datos para training y la mitad para testing.-
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = .5)

# decision tree classifier.-
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

# kneighbors classifier.-
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

# entrenamos el classifier con los datos de training.-
clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

# validamos que tan acertado es nuestro classifier.-
from sklearn.metrics import accuracy_score
print("accuracy score: ", accuracy_score(y_test, predictions))

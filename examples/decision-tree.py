"""
========================
Decision Tree Classifier
========================

An example showing how fit a decision tree classifier, predict a case and show the decision tree in a pdf file.
"""
print(__doc__)

from sklearn import tree
import graphviz

feature_names = ["peso", "textura"]
label_names = ["manzana", "naranja"]

# tipos de labels
labels = {
  "manzana": "manzana",
  "naranja": "naranja"
}

# features
liso = 0
rugoso = 1

peso = 100
textura = rugoso

# definimos las features.-
features = [[150, rugoso], [170, rugoso], [140, liso], [130, liso]]
# establecemos los labels para cada conjunto de features.-
labels = [labels["naranja"], labels["naranja"], labels["manzana"], labels["manzana"]]

# mostramos training data.-
print("---Training Data---")
for index, feature in enumerate(features):
  print(feature, " -> ", labels[index])

# creamos el classifier -> decision tree.-
clf = tree.DecisionTreeClassifier()

# entrenamos al classifier.-
# fit busca patrones en los datos de entrenamiento.-
clf = clf.fit(features, labels)

# prediccion
predicted = clf.predict([[peso, textura]])

print("\n---Prediction---")
print("Si pesa %sg y la textura es %s -> es una %s" % (peso, ("rugoso" if textura == rugoso else "liso"), predicted[0]))

# viz classifier decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=feature_names,
                         class_names=label_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data, filename="decision-tree", format="pdf")
graph.render()

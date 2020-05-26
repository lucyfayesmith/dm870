import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from IPython.display import Image
import pydotplus


def printTree(classifier):
    feature_names = ['top-left-square', 'top-middle-square', 'top-right-square',
                    'middle-left-square', 'middle-middle-square','middle-right-square',
                    'bottom-left-square','bottom-middle-square','bottom-right-square']
    target_names = ['positive', 'negative']
    # Build the daya
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names)
    # Build the graph
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Show the image
    Image(graph.create_png())
    graph.write_png("tree.png")



if __name__=="__main__":


    file = "testGames.csv"
    data = pd.read_csv(file)

    # Get the data
    topLeftSquare = data["topLeftSquare"].values
    topMiddleSquare = data["topMiddleSquare"].values
    topRightSquare = data["topRightSquare"].values
    middleLeftSquare = data["middleLeftSquare"].values
    middleMiddleSquare = data["middleMiddleSquare"].values
    middleRightSquare = data["middleRightSquare"].values
    bottomLeftSquare = data["bottomLeftSquare"].values
    bottomMiddleSquare = data["bottomMiddleSquare"].values
    bottomRightSquare = data["bottomRightSquare"].values

    classes = data["class"].values

    labelEncoder = preprocessing.LabelEncoder()

    # Encode the features and the labels
    encodedTopLeftSquare = labelEncoder.fit_transform(topLeftSquare)
    encodedTopMiddleSquare = labelEncoder.fit_transform(topMiddleSquare)
    encodedTopRightSquare = labelEncoder.fit_transform(topRightSquare)
    encodedMiddleLeftSquare = labelEncoder.fit_transform(middleLeftSquare)
    encodedMiddleMiddleSquare = labelEncoder.fit_transform(middleMiddleSquare)
    encodedMiddleRightSquare = labelEncoder.fit_transform(middleRightSquare)
    encodedBottomLeftSquare = labelEncoder.fit_transform(bottomLeftSquare)
    encodedBottomMiddleSquare = labelEncoder.fit_transform(bottomMiddleSquare)
    encodedBottomRightSquare = labelEncoder.fit_transform(bottomRightSquare)
    encodedOutcome = labelEncoder.fit_transform(classes)

    # Build the features
    features = []
    for i in range(len(encodedTopRightSquare)):
        features.append([encodedTopLeftSquare[i], encodedTopMiddleSquare[i], encodedTopRightSquare[i],
                        encodedMiddleLeftSquare[i], encodedMiddleMiddleSquare[i], encodedMiddleRightSquare[i],
                        encodedBottomLeftSquare[i], encodedBottomMiddleSquare[i], encodedBottomRightSquare[i]])

    print(features)

    # classifier = tree.DecisionTreeClassifier()
    # classifier = classifier.fit(features, encodedOutcome)

    # #o,x,x,b,x,o,o,x,b,positive
    # # 1 = o 2 = x 0 = b
    # print(classifier.predict([[1,2,2,0,2,1,1,2,0]]))
    # # o,o,o,x,x,b,x,o,x,negative
    # print(classifier.predict([[1,1,1,2,2,0,2,1,2]]))

    # printTree(classifier)


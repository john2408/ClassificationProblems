# Sentiment Analysis with Convolutional Neural Networks
------------------------------

Make sure conda install scikit-learn==0.21.2 is installed otherwise it will throw the error:

Cannot clone object <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x7fe06095e210>, as the constructor either does not set or modifies parameter embedding_matrix. on 01.02.2021 it is still an open issue https://github.com/scikit-learn/scikit-learn/issues/15722.


## Build Docker Container
--------------------------

docker build text_classifier:1.0 .

## Run Docker Container
-------------------------

docker run -v ~/results_docker:/opt/program/Sentiment_Analysis/results --name sa text_classifier:1.0

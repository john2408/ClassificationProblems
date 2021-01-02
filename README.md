# Sentiment Analysis with Convolutional Neural Networks
------------------------------


The model uses a data set containing two columns: 

- 'sentences': plain text in English language. 
- 'label': category classification (0,1,2,3,4)

and it is trained to be able to classify every sentence (text) correspondingly. 

# Packages Dependency Issues
--------------------------

Make sure to conda install scikit-learn==0.21.2  otherwise it will throw the following error:

Cannot clone object <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x7fe06095e210>, as the constructor either does not set or modifies parameter embedding_matrix. on 01.02.2021 it is still an open issue https://github.com/scikit-learn/scikit-learn/issues/15722. 


# Build Docker Container
--------------------------

docker build text_classifier:1.0 .

# Run Docker Container
-------------------------

docker run -v ~/results_docker:/opt/program/Sentiment_Analysis/results --name sa text_classifier:1.0
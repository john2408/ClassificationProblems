# Text Classification Model
------------------------------

The model uses a data set (file: data/ML_data_2.0.xlsx) containing two columns:

- 'sentences': 'str' plain text in English language.
- 'label': 'int' category classification (0,1,2,3,4)


and it is trained to be able to classify every sentence (text) correspondingly.

The model is available in the file main.py in the folder Sentiment_Analysis. You can also use the file main.ipynb to go line by line through the code and functions.

The model can classify n type of categories using the following approaches:

1. Convolutional Neural Networks
2. Support Vector Machines

In this approach the category '0' stands for 'other', meaning the sentence
does not fit in any other category. Therefore to avoid imbalanced classes
use the class_weights parameter in the config_param.yml.

Any other data base with the same structure can be use for modelling porpuses.

## Packages Dependency Issues
--------------------------

Make sure to conda install scikit-learn==0.21.2  otherwise it will throw the following error:

Cannot clone object <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x7fe06095e210>, as the constructor either does not set or modifies parameter embedding_matrix. on 01.02.2021 it is still an open issuein the scikit-learn GitHub https://github.com/scikit-learn/scikit-learn/issues/15722.

## Build Conda Environment
--------------------------

``` bash
cd Sentiment_Analysis
conda env create --name envname --file=environment.yaml
python main.py
```
or also

``` bash
cd Sentiment_Analysis
conda create --name class --file requirements.txt
python main.py
```

## Build Docker Container
--------------------------
``` bash
cd Sentiment_Analysis
docker build -t text_classifier:1.0 .
```

## Run Docker Container
-------------------------

Create Volume to store results

``` bash
docker volume create results
docker run -v results:/opt/program/Sentiment_Analysis/results --name sa text_classifier:1.0
```

**From Windows**

Mapping to a local folder in Windows Host

``` bash
docker run -v /mnt/d/Data_Science/Classification_Problems/Sentiment_Analysis/results_docker:/opt/program/Sentiment_Analysis/results --name sa text_classifier:1.0
```

**From Linux**
Mapping to a local folder in Linux Host

``` bash
cd Sentiment_Analysis
docker run -v ~/results_docker:/opt/program/Sentiment_Analysis/results --name sa text_classifier:1.0
```

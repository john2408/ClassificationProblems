from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

from tensorflow.python.client import device_lib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV

from postprocessing import cal_label_accuracy, store_to_pickle

import numpy as np
from os.path import join

import pickle


def create_model(num_filters_cv, kernel_size_cv, vocab_size, embedding_dim, embedding_matrix,
                 seq_input_len, output_label, nodes_hidden_dense_layer, use_pretrained_embeddings ):
    """Function to create CNN model. 

    Parameters
    ----------
    num_filters_cv : obj: tuple: `int`
        number of filter for the first and second convolutional layers
    
    kernel_size_cv : obj: tuple: `int`
        number of kernel sizes for the first and second convolutional layers

    vocab_size: `int`
        Vocab size if keras embedding space training is wanted

    embedding_dim: `int`
        Length of the word vector ( dimension in the embedding space)
    
    embedding_matrix: obj: numpy.array
        2d numpy array containing the embedding space
        rows = vocab_size
        columns = embedding_dim

    seq_input_len: `int`
        Length of the vector sentence ( no. of words per sentence)
    
    output_label: `int`
        Number of unique labeled categories
    
    nodes_hidden_dense_layer: `int`
        No. of nodes for hidden Dense layer
    
    use_pretrained_embeddings: `bool`
        If set to FALSE then keras embedding space training is used instead
        Embedding Space possibilites are GloVe or TFIDF
        see function: data_vectorization()

    Returns
    -------
    model obj: keras.Sequential()
    """

    model = Sequential()
    

    if use_pretrained_embeddings:  

        # Use Glove Embedding Matrix
        
        model.add(layers.Embedding(embedding_matrix.shape[0], # Vocabulary Size
                                   embedding_matrix.shape[1], # Word Vector Dimension
                                   weights=[embedding_matrix], 
                                   input_length=seq_input_len, 
                                   trainable=False))
    else: 
        
        # Keras Embedding Matrix Generation
        
        model.add(layers.Embedding(vocab_size, # Vocabulary Size
                                   embedding_dim, # Word Vector Dimension
                                   input_length = seq_input_len))

    # Filters: No. of output filter in the convolution
    # kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    model.add(layers.Conv1D(filters = num_filters_cv[0], kernel_size = kernel_size_cv[0], activation='relu'))
  
    # Filters: No. of output filter in the convolution
    # kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    model.add(layers.Conv1D(filters = num_filters_cv[1], kernel_size = kernel_size_cv[1], activation='relu'))

    # Global max pooling operation for 1D temporal data.
    # Downsamples the input representation by taking the maximum value over the time dimension
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(nodes_hidden_dense_layer, activation='relu'))

    model.add(layers.Dense(output_label, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    
    return model


def hyperparameter_optimization(X_train, Y_train, X_test, Y_test, 
                                epochs, batch_size, param_grid, cv, n_iter,
                                verbose = False ):
    """Function to clean raw corpus text.

    Parameters
    ----------
    X_train : numpy.array: `float`
        senteces text to train the model
    
    Y_train : numpy.array: `int`
        labeled categories to trained the model

    X_test : numpy.array: `float`
        senteces text to test the model
    
    Y_test : numpy.array: `int`
        labeled categories to test the model
    
    epochs: `int`
        No. of optimizatoin runs

    batch_size: `int`
        No. of sentences batch to train

    param_grid: `dict`
        Dict cointaining the parameters for the model to run 
        RandomGridSearch on.

    cv: `int`
        No. of Cross validations

    n_iter: `int`
        No. of Iterations for the Random Search

    verbose: ´int´
        Controls the level of messaging. If > 1, it
        prints out the label accuracy.

    Returns
    -------
    output: `dict`
        'best_train_acc': `float` 
        'best_train_param': `dict` with the best parameters
        'test_acc': `float`
        'conf_matrix': numpy.array 
        'Y_pred': numpy.array
        'grid_result': numpy.array 
    """  

    output = {}  

    # Create NN Model 
    print("Creating Model...")
    model = KerasClassifier(build_fn = create_model,
                            epochs = epochs, 
                            batch_size = batch_size,
                            verbose = verbose)
    
    print("Selecting Parameters...")
    # Make Random Search Cross Validation
    grid = RandomizedSearchCV(estimator = model, 
                              param_distributions = param_grid,
                              cv = cv, 
                              n_iter = n_iter,
                              verbose = verbose)
    
    print("Evaluating Model...")
    # Fit Selected Model with Random Parameters
    grid_result = grid.fit(X_train, Y_train, verbose = verbose)
    
    # Predict Y values
    Y_pred = grid.predict(X_test)
    
    # Create empty conf_matrix
    conf_matrix = np.array([])

    try:
        # Generate Confusion Matrix
        conf_matrix = confusion_matrix(Y_pred, Y_test.argmax(axis=1)) / len(Y_pred)

        # Calculate Label Accuracy
        label_acc = cal_label_accuracy(conf_matrix, verbose  = 1)
    except:
        print(" label accuracy could not be calculated")
        conf_matrix = "Could not be calculated"

    # Evaluate testing set
    test_accuracy = grid.score(X_test, Y_test)
    
    # store output variable into dict
    output['best_train_acc'] = grid_result.best_score_
    output['best_train_param'] = grid_result.best_params_
    output['test_acc'] = test_accuracy
    output['conf_matrix'] = conf_matrix
    output['Y_pred'] = Y_pred
    output['grid_result'] = grid_result

    print("Best Test Accuracy was: " , test_accuracy)

    return output
 

def keras_tokenizer(sentences_train, sentences_test, num_words, seq_input_len, 
                    store_keras_tokenizer, remove_class_0, make_all_other_classes_1, 
                    output_path_vectorizer, 
                    timestamp):
    """Function to tokenize the word with the keras word tokenizer engine. 

    Parameters
    ----------
    sentences_train : obj: numpy.array: `str`
        Corpus text, Sentences to train the model

    sentences_test: obj: numpy.array: `str`
        Corpus text, Sentences to test the model

    num_words: `int` 
        No. of words to use in the embedding space of GloVe or TFIDF

    seq_input_len: `int`
        Length of the vector sentence ( no. of words per sentence)

    store_keras_tokenizer : `bool`
        whether to store the Keras tokenizer model 
    
    remove_class_0: `bool`
        model for 0 and 1. 1 representing the classes 1,2,3,4. 

    make_all_other_classes_1: `bool`
        model for 1,2,3 and 4. The output will be mapped as 1:0, 2:1, 3:2, 4:3.

    timestamp : `str`
        Timestamp in str as "%Y-%m-d_%H-%M-%S"
    
    output_path_vectorizer : `str`
        output path for vectorization model

    Returns
    -------
    X_train: numpy.array: `float`
        Array of the tokenized train sentences

    X_test: numpy.array: `float`
        Array of the tokenized test sentences

    vocab_size: `int`
        Vocab size if keras embedding space training is wanted
    
    vocab: `dict`
        Containing key pair 'int':'word', each 'int' represents
        the encoding which the keras tokenizer generates

    """

    # Start Tokenizer Object
    kears_tokenizer = Tokenizer(num_words = num_words)

    # Train vocabulary
    kears_tokenizer.fit_on_texts(sentences_train)


    if store_keras_tokenizer: 

        file_name = 'KERAS_vectorizer'

        if remove_class_0:
            file_name = f'KERAS_vectorizer_1234'

        if make_all_other_classes_1:
            file_name = f'KERAS_vectorizer_01'

        store_to_pickle(data = kears_tokenizer, 
                        output_path= output_path_vectorizer, 
                        timestamp = timestamp,
                        file_name = file_name)
            


    X_train = kears_tokenizer.texts_to_sequences(sentences_train) 
    X_test = kears_tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(kears_tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    vocab = kears_tokenizer.word_index

    X_train = pad_sequences(X_train, padding = 'post', maxlen = seq_input_len)
    X_test = pad_sequences(X_test, padding = 'post', maxlen = seq_input_len)

    return X_train, X_test, vocab_size, vocab


def tfidf_tokenizer(num_words, corpus, sentences_train, sentences_test, timestamp, output_path_vectorizer,
                    remove_class_0, make_all_other_classes_1, store_tfidf_tokenizer = False):
    """Function to tokenize the words with the TFIDF tokenizer 

    Parameters
    ----------
    sentences_train : numpy.array: `str`
        Corpus text, Sentences to train the model

    sentences_test: numpy.array: `str`
        Corpus text, Sentences to test the model

    num_words: `int` 
        No. of words to use in the embedding space of GloVe or TFIDF

    corpus : obj: pandas.dataframe
        df containing two columns 'text' and 'label'

    timestamp : `str`
        Timestamp in str as "%Y-%m-d_%H-%M-%S"
    
    output_path_vectorizer : `str`
        output path for vectorization model
    
    store_tfidf_tokenizer : `bool`
        whether to store the TFIDF model 
    
    remove_class_0: `bool`
        model for 0 and 1. 1 representing the classes 1,2,3,4. 

    make_all_other_classes_1: `bool`
        model for 1,2,3 and 4. The output will be mapped as 1:0, 2:1, 3:2, 4:3.

    Returns
    -------
    X_train: obj: numpy.array: `float`
        Array of the tokenized train sentences

    X_test: obj: numpy.array: `float`
        Array of the tokenized test sentences

    vocab_size: `int`
        Vocab size if keras embedding space training is wanted
    
    vocab: `dict`
        Containing key pair 'int':'word', each 'int' represents
        the encoding which the keras tokenizer generates

    """
    

    # Create new Class TfidfVectorizer with max 5000 features
    Tfidf_vect = TfidfVectorizer(max_features=num_words)

    # Learn vocabulary and idf from training set
    Tfidf_vect.fit(corpus['text'])

    # Store TFIDF Vectorizer to given location


    if store_tfidf_tokenizer: 

        file_name = 'TFIDF_vectorizer'

        if remove_class_0:
            file_name = f'TFIDF_vectorizer_1234'

        if make_all_other_classes_1:
            file_name = f'TFIDF_vectorizer_01'

        store_to_pickle(data = Tfidf_vect, 
                        output_path= output_path_vectorizer, 
                        timestamp = timestamp,
                        file_name = file_name)
        


    # Transfor both the train and the test to document-term matrix
    X_train = Tfidf_vect.transform(sentences_train)
    X_test = Tfidf_vect.transform(sentences_test)
    
    vocab = Tfidf_vect.vocabulary_
    
    vocab_size = len(vocab) + 1
    
    return X_train, X_test, vocab_size, vocab

def tfidf_as_embedding_weights(num_words, corpus, sentences_train):
    """Function to tokenize the words with the TFIDF tokenizer 

    Parameters
    ----------
    sentences_train : numpy.array: `str`
        Corpus text, Sentences to train the model

    num_words: `int` 
        No. of words to use in the embedding space of GloVe or TFIDF

    corpus : obj: pandas.dataframe
        df containing two columns 'text' and 'label'

    Returns
    -------

    embedding_matrix: obj: numpy.array
        2d numpy array containing the embedding space
        rows = vocab_size
        columns = embedding_dim
    
    embedding_dim: `int`
        Length of the word vector ( dimension in the embedding space)

    """



    # Create new Class TfidfVectorizer with max 5000 features
    Tfidf_vect = TfidfVectorizer(max_features=num_words)

    # Learn vocabulary and idf from training set
    Tfidf_vect.fit(corpus['text'])

    # Transfor both the train and the test to document-term matrix
    embedding_matrix = Tfidf_vect.transform(sentences_train).toarray().transpose()
    
    # Calculate embedding dimension - sequence length
    embedding_dim = len(embedding_matrix[0])
    
    return embedding_matrix, embedding_dim


def create_embedding_matrix(filepath, word_index, embedding_dim):
    """Function to create embedding matrix using GloVe embedding space.

    Parameters
    ----------
    filepath : `str`
        Corpus text, each element being a sentence
    word_index : `str`
        Corpus text, each element being a sentence
    embedding_dim : `str`
        Corpus text, each element being a sentence        
    
    Returns
    -------
    embedding_matrix: obj: numpy.array
        2d numpy array containing the embedding space
        rows = vocab_size
        columns = embedding_dim
    """    


    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def fit_pretrained_embedding_space_glove(embedding_dim, filepath, vocab):
    """Fit pretrained embedding using embedding space GloVe

    Parameters
    ----------
    embedding_dim: `int`
        Length of the word vector ( dimension in the embedding space)
    
    filepath: `str`
        File path to GLoVe pretrained embedding words
    
    vocab: `dict`
        Containing key pair 'int':'word', each 'int' represents
        the encoding which the keras tokenizer generates

    Returns
    -------
    embedding_matrix: obj: numpy.array
        2d numpy array containing the embedding space
        rows = vocab_size
        columns = embedding_dim
    
    embedding_dim: `int`
        Length of the word vector ( dimension in the embedding space)
    """
       
    embedding_matrix = create_embedding_matrix(
                        filepath = filepath,
                        word_index = vocab, 
                        embedding_dim = embedding_dim)
    
    return embedding_matrix, embedding_dim
    

# def data_vectorization(sentences_train_CNN, 
#                        sentences_test_CNN, 
#                        sentences_train_SVM, 
#                        sentences_test_SVM,
#                        num_words, 
#                        seq_input_len, 
#                        filepath,
#                        corpus,
#                        vocab,
#                        embedding_dim,
#                        timestamp , 
#                        output_path_vectorizer ,
#                        store_tfidf_tokenizer ,
#                        running_CNN = True, 
#                        running_SVM = True, 
#                        use_tfidf_as_embedding_weights = True,
#                        use_glove_pretrained_embeddings_weights = False, 
#                        ):

#     """Apply data vectorization on the train and test data.  

#     Parameters
#     ----------

#     sentences_train_CNN : numpy.array: `str`
#         Corpus text, Sentences to train the CNN model

#     sentences_test_CNN: numpy.array: `str`
#         Corpus text, Sentences to test the CNN model
    
#     sentences_train_SVM : numpy.array: `str`
#         Corpus text, Sentences to train the SVM model

#     sentences_test_SVM: numpy.array: `str`
#         Corpus text, Sentences to test the SVM model

#     num_words: `int` 
#         No. of words to use in the embedding space of GloVe or TFIDF

#     seq_input_len: `int`
#         Length of the vector sentence ( no. of words per sentence)

#     filepath: `str`
#         File path to GLoVe pretrained embedding words
    
#     corpus : obj: pandas.dataframe
#         df containing two columns 'text' and 'label'
    
#     vocab: `dict`
#         Containing key pair 'int':'word', each 'int' represents
#         the encoding which the keras tokenizer generates
    
#     embedding_dim: `int`
#         Length of the word vector ( dimension in the embedding space)
    
#     running_CNN: `bool`
#         For CNN use keras word tokenizer

#     running_SVM: `bool`
#         For SVM use TFIDF tokenizer 
    
#     use_tfidf_as_embedding_weights: `bool`
#         If using TFIDF tokenizer as embedding weights
    
#     use_glove_pretrained_embeddings_weights: `bool` 
#         If using GloVe Petrained embedding weights

#     Returns
#     -------
#     output: `dict` 
#         X_train: obj: numpy.array: `float`
#             Array of the tokenized train sentences
#         X_test: obj: numpy.array: `float`
#             Array of the tokenized test sentences
#         vocab_size: `int`
#             Vocab size if keras embedding space training is wanted
#         vocab: `dict`
#             Containing key pair 'int':'word', each 'int' represents
#             the encoding which the keras tokenizer generates.
#         embedding_matrix: obj: numpy.array
#             2d numpy array containing the embedding space
#             rows = vocab_size
#             columns = embedding_dim
#         embedding_dim: `int`
#             Length of the word vector ( dimension in the embedding space)
#     """
#     output = {}
    
#     if running_CNN:
        
<<<<<<< HEAD
#         output['X_train_CNN'], output['X_test_CNN'], output['vocab_size'], output['vocab'] = keras_tokenizer(sentences_train_CNN, sentences_test_CNN, num_words, seq_input_len)
               
#     if running_SVM:
        
#         output['X_train_SVM'], output['X_test_SVM'], output['vocab_size_SVM'], output['vocab'] = tfidf_tokenizer(num_words, corpus, sentences_train_SVM, sentences_test_SVM, 
#                                                                                                                 timestamp = timestamp, 
#                                                                                                                 output_path_vectorizer = output_path_vectorizer,
#                                                                                                                 store_tfidf_tokenizer = store_tfidf_tokenizer)
    
#     if use_tfidf_as_embedding_weights: 
=======
        output['X_train_CNN'], output['X_test_CNN'], output['vocab_size'], output['vocab'] = keras_tokenizer(sentences_train_CNN, sentences_test_CNN, num_words, seq_input_len)
 
    if use_tfidf_as_embedding_weights: 
>>>>>>> debug
        
#         output['embedding_matrix'], output['embedding_dim'] = tfidf_as_embedding_weights(num_words, corpus, sentences_train_CNN)

    
#     if use_glove_pretrained_embeddings_weights:  
        
#         output['embedding_matrix'] = create_embedding_matrix(
#                              filepath = filepath,
#                              word_index = vocab, 
#                              embedding_dim = embedding_dim)
        
#     return output

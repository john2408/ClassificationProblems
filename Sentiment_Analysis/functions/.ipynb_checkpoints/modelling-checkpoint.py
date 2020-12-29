from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

from tensorflow.python.client import device_lib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV

from postprocessing import cal_label_accuracy

import numpy as np



def create_model(num_filters_cv, kernel_size_cv, vocab_size, embedding_dim, embedding_matrix,
                 seq_input_len, output_label, nodes_hidden_dense_layer, use_pretrained_embeddings ):
    """
    
    """      

    model = Sequential()

    if use_pretrained_embeddings:  

        model.add(layers.Embedding(embedding_matrix.shape[0], # Vocabulary Size
                                   embedding_matrix.shape[1], # Word Vector Dimension
                                   weights=[embedding_matrix], 
                                   input_length=seq_input_len, 
                                   trainable=False))
    else: 
        
        # embedding_dim = 100 # Output Dimension - seq output length
        
        model.add(layers.Embedding(vocab_size, # Vocabulary Size
                                   embedding_dim, # Word Vector Dimension
                                   input_length = seq_input_len))

    # Filters: No. of output filter in the convolution
    # kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    model.add(layers.Conv1D(filters = num_filters_cv[0], kernel_size = kernel_size_cv[0], activation='relu'))

    # Global max pooling operation for 1D temporal data.
    # Downsamples the input representation by taking the maximum value over the time dimension
    #model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Conv1D(filters = num_filters_cv[1], kernel_size = kernel_size_cv[1], activation='relu'))

    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(nodes_hidden_dense_layer, activation='relu'))

    model.add(layers.Dense(output_label, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    
    return model


def hyperparameter_optimization(X_train, Y_train, X_test, Y_test, 
                                epochs, batch_size, param_grid, cv, n_iter, output_file,
                                verbose = False ):
    """
    
    """  
        

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
    
    try:
        # Generate Confusion Matrix
        conf_matrix = confusion_matrix(Y_pred.argmax(axis=1), Y_test.argmax(axis=1)) / len(Y_pred)

        # Calculate Label Accuracy
        label_acc = cal_label_accuracy(conf_matrix, verbose  = 1)
    except:
        print(" label accuracy could not be calculated")
        conf_matrix = "Could not be calculated"

    # Evaluate testing set
    test_accuracy = grid.score(X_test, Y_test)

    print("Writting results...")
    with open(output_file, 'a') as f:
        s = ('Running {} data set\nBest Accuracy : ''{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        
        output_string = s.format(
            "CNN Modeling",
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy,
            conf_matrix)
        
        print(output_string)
        f.write(output_string)
    
    return Y_pred
        


def keras_tokenizer(sentences_train, sentences_test, num_words, seq_input_len):

    # Start Tokenizer Object
    tokenizer = Tokenizer(num_words = num_words)

    # Train vocabulary
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train) 
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    vocab = tokenizer.word_index

    X_train = pad_sequences(X_train, padding = 'post', maxlen = seq_input_len)
    X_test = pad_sequences(X_test, padding = 'post', maxlen = seq_input_len)

    return X_train, X_test, vocab_size, vocab


def tfidf_tokenizer(num_words, corpus, sentences_train, sentences_test):
   
    """
    """
    
    # Create new Class TfidfVectorizer with max 5000 features
    Tfidf_vect = TfidfVectorizer(max_features=num_words)

    # Learn vocabulary and idf from training set
    Tfidf_vect.fit(corpus['text'])

    # Transfor both the train and the test to document-term matrix
    X_train = Tfidf_vect.transform(sentences_train)
    X_test = Tfidf_vect.transform(sentences_test)
    
    vocab = Tfidf_vect.vocabulary_
    
    vocab_size = len(vocab) + 1
    
    return X_train, X_test, vocab_size, vocab

def tfidf_as_embedding_weights(num_words, corpus, sentences_train):

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

       
    embedding_matrix = create_embedding_matrix(
                        filepath = filepath,
                        word_index = vocab, 
                        embedding_dim = embedding_dim)
    
    return embedding_matrix, embedding_dim
    

def data_vectorization(sentences_train, 
                       sentences_test, 
                       num_words, 
                       seq_input_len, 
                       filepath,
                       corpus,
                       vocab,
                       embedding_dim,
                       use_keras_tokenizer = True, 
                       use_tfidf_tokenizer = False, 
                       use_tfidf_as_embedding_weights = True,
                       use_glove_pretrained_embeddings_weights = False):

    output = {}
    
    if use_keras_tokenizer:
        
        output['X_train'], output['X_test'], output['vocab_size'], output['vocab'] = keras_tokenizer(sentences_train, sentences_test, num_words, seq_input_len)
               
    elif use_tfidf_tokenizer:
        
        output['X_train'], output['X_test'], output['vocab_size'], output['vocab'] = tfidf_tokenizer(sentences_train, sentences_test, num_words)
    
    if use_tfidf_as_embedding_weights: 
        
        output['embedding_matrix'], output['embedding_dim'] = tfidf_as_embedding_weights(num_words, corpus, sentences_train)

    
    if use_glove_pretrained_embeddings_weights:  
        
        output['embedding_matrix'] = create_embedding_matrix(
                             filepath = filepath,
                             word_index = vocab, 
                             embedding_dim = embedding_dim)
        
    return output

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from datetime import datetime
from os import getcwd
from os.path import join

from sklearn.metrics import accuracy_score, confusion_matrix

from sys import path
path.append( join( join( getcwd() , 'functions/' ) ) )

from functions import preprocessing, modelling, postprocessing


def preprocess(data_dir, data, param_grid):

    #----------------------------------------------------------------#
    # 1. Ingest Data
    #----------------------------------------------------------------#   

    corpus = pd.read_csv(data_dir , encoding='latin-1', sep = ';')

    # Perform data cleaning
    corpus = preprocessing.data_cleaning(corpus = corpus,
                        sent_tokenizer = False, 
                        text_cleaning = True, 
                        use_nltk_cleaning = False)


    #----------------------------------------------------------------#
    # 2. Preprocess Data
    #----------------------------------------------------------------#

    model_data = preprocessing.prepare_training_data(corpus)

    # Concat two dictionaries
    data = {**data, **model_data}

    #----------------------------------------------------------------#
    # 3. Vectorization
    #----------------------------------------------------------------#


    if use_keras_tokenizer:
        data['X_train'], data['X_test'], data['vocab_size'], data['vocab'] = modelling.keras_tokenizer(num_words = data['num_words'], 
                                                                                            sentences_train = data['sentences_train'] , 
                                                                                            sentences_test = data['sentences_test'],
                                                                                            seq_input_len = data['seq_input_len'])
    elif use_tfidf_tokenizer: # Not implemented yet
        data['X_train'], data['X_test'], data['vocab_size'], data['vocab'] = modelling.tfidf_tokenizer(num_words = data['num_words'],
                                                                                            corpus = corpus,
                                                                                            sentences_train = data['sentences_train'],
                                                                                            sentences_test = data['sentences_test'])
        
    if use_tfidf_as_embedding_weights:
        
        data['embedding_matrix'], data['embedding_dim']  = modelling.tfidf_as_embedding_weights(num_words = data['num_words'], 
                                                                    corpus = corpus, 
                                                                    sentences_train = data['sentences_train'])
        
    elif use_glove_pretrained_embeddings_weights:
        
        data['embedding_matrix'], data['embedding_dim'] = modelling.fit_pretrained_embedding_space_glove(embedding_dim = data['embedding_dim'], 
                                                                            filepath = data['filepath'] , 
                                                                            vocab = data['vocab'])

    data_pre = modelling.data_vectorization(sentences_train = data['sentences_train'], 
                        sentences_test = data['sentences_test'], 
                        num_words = data['num_words'], 
                        seq_input_len = data['seq_input_len'], 
                        filepath = data['filepath'],
                        corpus = corpus,
                        vocab = data['vocab'],
                        embedding_dim = data['embedding_dim'],
                        use_keras_tokenizer = use_keras_tokenizer, 
                        use_tfidf_tokenizer = use_tfidf_tokenizer, 
                        use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
                        use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights)

    # Concat two dictionaries
    data = {**data, **data_pre}

    # Add final parameters
    param_grid['embedding_matrix'] = ([data['embedding_matrix']])
    param_grid['output_label'] = [data['output_label']]

    return data, param_grid

def train(data, param_grid):

    #----------------------------------------------------------------#
    # 4. Hyperparameter Optimization
    #----------------------------------------------------------------#


    model_output = modelling.hyperparameter_optimization( 
                                X_train = data['X_train'], 
                                Y_train = data['Y_train'], 
                                X_test = data['X_test'], 
                                Y_test = data['Y_test'] , 
                                epochs = data['epochs'] , 
                                batch_size = data['batch_size'],
                                param_grid = param_grid,
                                cv = data['cv'], 
                                n_iter = data['n_iter'],
                                verbose = False)

    return model_output


def results(model_output, 
            data,
            use_nltk_cleaning,
            text_cleaning, 
            use_tfidf_tokenizer, 
            use_keras_tokenizer, 
            use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights):

    #----------------------------------------------------------------#
    # 5. Score Analysis
    #----------------------------------------------------------------#

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(model_output['Y_pred'], data['Y_test'].argmax(axis=1)) / len(model_output['Y_pred'])

    # Calculate Label Accuracy
    model_output['label_acc'] = postprocessing.cal_label_accuracy(conf_matrix, verbose  = 1)


    #----------------------------------------------------------------#
    # 6. Write Results to text file
    #----------------------------------------------------------------#

    postprocessing.write_results_txt(output_file = data['output_file'], 
                    best_train_acc = model_output['best_train_acc'], 
                    best_train_param = model_output['best_train_param'],
                    test_acc = model_output['test_acc'], 
                    label_acc = model_output['label_acc'] , 
                    sent_tokenizer = sent_tokenizer, 
                    use_nltk_cleaning = use_nltk_cleaning, 
                    text_cleaning = text_cleaning , 
                    use_tfidf_tokenizer = use_tfidf_tokenizer, 
                    use_keras_tokenizer = use_keras_tokenizer, 
                    use_pretrained_embeddings = use_pretrained_embeddings,
                    use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights,
                    use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
                    epochs = data['epochs'],
                    batch_size = data['batch_size'],
                    num_words = data['num_words'], 
                    cv = data['cv'] ,
                    n_iter = data['n_iter']
    )


if __name__ == "__main__":


    # ----------------------------------------------------------------#
    # 0. Parameters
    # ----------------------------------------------------------------#

    current_time = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
    data_dir = 'D:/Data_Science/ClassificationProblems/Sentiment_Analysis/src/data/SA_4_Categories.csv'

    sent_tokenizer = False # TODO: Adjust for input to CNN

    # Text Cleaning Options
    use_nltk_cleaning = False
    text_cleaning = True

    # Word Tokenizer Options
    use_tfidf_tokenizer = False # TODO: Adjust for input to CNN
    use_keras_tokenizer = True

    # If set to FALSE then keras embedding space training is used instead
    # Embedding Space possibilites are GloVe or TFIDF
    use_pretrained_embeddings = True

    # Only if use_pretrained_embeddings == True then select embedding vector space type
    use_glove_pretrained_embeddings_weights = True
    use_tfidf_as_embedding_weights = False

    # Dictionary which will cotain all the model's variables
    data = {}

    # Initialize Model
    data['epochs'] = 30 # NO. of optimizatoin runs
    data['batch_size'] = 16 # No. of sentences batch to train
    data['num_words'] = 5000 # No. of words to use in the embedding space of GloVe or TFIDF
    data['cv'] = 4 # No. of Cross Validations
    data['n_iter'] = 5 # No. of Iterations
    data['seq_input_len'] = 40 # Length of the vector sentence ( no. of words per sentence)
    data['embedding_dim'] = 40 # Length of the word vector ( dimension in the embedding space)
    data['nodes_hidden_dense_layer'] = 5 # No. of nodes for hidden Dense layer


    data['filepath'] = 'D:/Semillero Data Science/Deep Learning/pre-trained Word Embeddings/GloVe/glove.6B.50d.txt' # File path to GLoVe pretrained embedding words
    data['output_file'] = f"results/{current_time}_Result.txt" # Name of output result file


    param_grid = dict(num_filters_cv = [(64,16), (64,32), (128,16), (128,32), (256,64), (256,32), (256,64), (512,128), (512, 32)], # No of filter to use in convolution
                    kernel_size_cv = [(2,3), (2,4), (3,4), (3,5)], # No of words to check per Convolution 
                    vocab_size = [3000, 4000, 5000, 6000], # Vocab size if keras embedding space training is wanted
                    embedding_dim = [20, 30, 40, 50], 
                    seq_input_len = [50, 40, 30, 20, 10], 
                    nodes_hidden_dense_layer = [5, 10, 15, 20, 40],
                    use_pretrained_embeddings = [True, False])


    # Small Test
    param_grid = dict(num_filters_cv = [(64,16)],
                        kernel_size_cv = [(2,3)],
                        vocab_size = [5000], 
                        embedding_dim = [50],
                        seq_input_len = [50], 
                        nodes_hidden_dense_layer = [5],
                        use_pretrained_embeddings = [True])


    # Preprocess data
    data, param_grid = preprocess(data_dir = data_dir, 
                                data = data, 
                                param_grid = param_grid)

    # Train CNN Model
    model_output = train(data = data, 
                        param_grid = param_grid)

    # Generate Results
    results(model_output = model_output, 
            data = data,
            use_nltk_cleaning = use_nltk_cleaning, 
            text_cleaning = text_cleaning , 
            use_tfidf_tokenizer = use_tfidf_tokenizer, 
            use_keras_tokenizer = use_keras_tokenizer, 
            use_pretrained_embeddings = use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights
            )
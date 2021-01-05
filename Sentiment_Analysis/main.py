import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from datetime import datetime
from os import getcwd
from os.path import join

from sklearn.metrics import accuracy_score, confusion_matrix

from sys import path
path.append(  join( getcwd() , 'functions/' ) )

from functions import preprocessing, modelling, postprocessing
from config import ConfigDict


# Run only for the first time#
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


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


    config = ConfigDict.read('config/config_param.yml')
    current_time = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

    sent_tokenizer = config['params']['tokenization_options']['sent_tokenizer'] # TODO: Adjust for input to CNN

    # Text Cleaning Options
    use_nltk_cleaning = config['params']['tokenization_options']['use_nltk_cleaning']
    text_cleaning = config['params']['tokenization_options']['text_cleaning']

    # Word Tokenizer Options
    use_tfidf_tokenizer = config['params']['tokenization_options']['use_tfidf_tokenizer'] # TODO: Adjust for input to CNN
    use_keras_tokenizer = config['params']['tokenization_options']['use_keras_tokenizer']

    # If set to FALSE then keras embedding space training is used instead
    # Embedding Space possibilites are GloVe or TFIDF
    use_pretrained_embeddings = config['params']['tokenization_options']['use_pretrained_embeddings']

    # Only if use_pretrained_embeddings == True then select embedding vector space type
    use_glove_pretrained_embeddings_weights = config['params']['tokenization_options']['use_glove_pretrained_embeddings_weights']
    use_tfidf_as_embedding_weights = config['params']['tokenization_options']['use_tfidf_as_embedding_weights']

    # Dictionary which will cotain all the model's variables
    data = {}

    # Initialize Model
    data['epochs'] = config['params']['data']['epochs'] # NO. of optimizatoin runs
    data['batch_size'] = config['params']['data']['batch_size'] # No. of sentences batch to train
    data['num_words'] = config['params']['data']['num_words'] # No. of words to use in the embedding space of GloVe or TFIDF
    data['cv'] = config['params']['data']['cv'] # No. of Cross Validations
    data['n_iter'] = config['params']['data']['n_iter'] # No. of Iterations
    data['seq_input_len'] = config['params']['data']['seq_input_len'] # Length of the vector sentence ( no. of words per sentence)
    data['embedding_dim'] = config['params']['data']['embedding_dim'] # Length of the word vector ( dimension in the embedding space)
    data['nodes_hidden_dense_layer'] = config['params']['data']['nodes_hidden_dense_layer'] # No. of nodes for hidden Dense layer


    data['filepath'] = config['params']['data']['filepath'] # File path to GLoVe pretrained embedding words
    data['output_file'] = f"results/{current_time}_Result.txt" # Name of output result file

    # Small Test
    param_grid = dict(num_filters_cv = [(64,16)],
                        kernel_size_cv = [(2,3)],
                        vocab_size = [5000], 
                        embedding_dim = [50],
                        seq_input_len = [50], 
                        nodes_hidden_dense_layer = [5],
                        use_pretrained_embeddings = [True])


    # Preprocess data
    data, param_grid = preprocess(data_dir = config['params']['input_data']['5_labels'], 
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
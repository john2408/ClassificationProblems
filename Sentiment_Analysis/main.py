import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from datetime import datetime
from os import getcwd
from os.path import join

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import model_selection, naive_bayes, svm

from sys import path
path.append( join( join( getcwd() , 'functions/' ) ) )

from functions import preprocessing, modelling, postprocessing
from config import ConfigDict

import openpyxl


# Run only for the first time#
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def preprocess(data_dir, data, param_grid, read_type, sep, remove_class_0, 
                make_all_other_classes_1, running_CNN, running_SVM ):

    #----------------------------------------------------------------#
    # 1. Ingest Data
    #----------------------------------------------------------------#   

    if read_type == 'excel':
    
        corpus = pd.read_excel(input_data, engine='openpyxl')

    elif read_type == 'csv':

        corpus = pd.read_csv( input_data, sep = sep)

    # Filter all NAs values
    corpus.dropna(inplace= True)

    # Make Sure labels are integers
    corpus['label'] = corpus['label'].astype(int)

    # Perform data cleaning
    corpus = preprocessing.data_cleaning(corpus = corpus,
                        sent_tokenizer = False, 
                        text_cleaning = True, 
                        use_nltk_cleaning = False)


    #----------------------------------------------------------------#
    # 2. Preprocess Data
    #----------------------------------------------------------------#

    # Create Filter on the data to avoid the imbalance classes problem
    if make_all_other_classes_1:

        corpus['label_orignal'] = corpus.loc[:,'label']
        corpus['label'] = np.where( corpus['label'] > 0 , 1, corpus['label'])

    elif remove_class_0:

        corpus['label_orignal'] = corpus.loc[:,'label']
        corpus = corpus[~corpus['label'].isin([0])]

        # Reindex classes
        corpus['label'] = corpus['label'].map({1:0,2:1,3:2,4:3})

    

    model_data = preprocessing.prepare_training_data(corpus)

    # Concat two dictionaries
    data = {**data, **model_data}

    #----------------------------------------------------------------#
    # 3. Vectorization
    #----------------------------------------------------------------#


    if running_CNN:
    
        data['X_train_CNN'], data['X_test_CNN'], data['vocab_size'], data['vocab'] = modelling.keras_tokenizer(num_words = data['num_words'], 
                                                                                            sentences_train = data['sentences_train_CNN'] , 
                                                                                            sentences_test = data['sentences_test_CNN'],
                                                                                            seq_input_len = data['seq_input_len'])
        if use_tfidf_as_embedding_weights:
        
            data['embedding_matrix'], data['embedding_dim']  = modelling.tfidf_as_embedding_weights(num_words = data['num_words'], 
                                                                        corpus = corpus, 
                                                                        sentences_train = data['sentences_train_CNN'])
        
        elif use_glove_pretrained_embeddings_weights:
            
            data['embedding_matrix'], data['embedding_dim'] = modelling.fit_pretrained_embedding_space_glove(embedding_dim = data['embedding_dim'], 
                                                                                filepath = data['filepath'] , 
                                                                                vocab = data['vocab'])

    if running_SVM: 

        data['X_train_SVM'], data['X_test_SVM'], data['vocab_size'], data['vocab'] = modelling.tfidf_tokenizer(num_words = data['num_words'],
                                                                                            corpus = corpus,
                                                                                            sentences_train = data['sentences_train_SVM'],
                                                                                            sentences_test = data['sentences_test_SVM'])
    


    data_pre = modelling.data_vectorization(sentences_train_CNN = data['sentences_train_CNN'], 
                        sentences_test_CNN = data['sentences_test_CNN'], 
                        sentences_train_SVM = data['sentences_train_SVM'], 
                        sentences_test_SVM = data['sentences_test_SVM'], 
                        num_words = data['num_words'], 
                        seq_input_len = data['seq_input_len'], 
                        filepath = data['filepath'],
                        corpus = corpus,
                        vocab = data['vocab'],
                        embedding_dim = data['embedding_dim'],
                        running_CNN = running_CNN, 
                        running_SVM = running_SVM, 
                        use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
                        use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights)

    # Concat two dictionaries
    data = {**data, **data_pre}

    # Add final parameters
    param_grid['embedding_matrix'] = ([data['embedding_matrix']])
    param_grid['output_label'] = [data['output_label']]

    return data, param_grid


def train_SVM(data,C, kernel, degree, gamma, class_weight,
            sent_tokenizer, 
            use_nltk_cleaning, 
            text_cleaning, 
            use_tfidf_tokenizer, 
            use_keras_tokenizer, 
            use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights,
            imbalanced_classes,
            make_all_other_classes_1,
            remove_class_0):

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier

    if imbalanced_classes: 
        
        if make_all_other_classes_1: 

            SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                class_weight = class_weight)


        elif remove_class_0:

            SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                class_weight = class_weight)

            

        else:

            SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                class_weight = class_weight)    

    else: 

        SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                )

    # Fit SVM Model
    SVM.fit(data['X_train_SVM'], data['Y_train_SVM'])

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(data['X_test_SVM'])

    # Use accuracy_score function to get the accuracy
    data['test_acc'] = np.round( accuracy_score(predictions_SVM, data['Y_test_SVM'])*100 , 4)

    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, data['Y_test_SVM'])*100)

    # Print Confusion Matrix
    Pred_Y = SVM.predict(data['X_train_SVM'])
    data['conf_matrix'] = confusion_matrix(data['Y_test_SVM'], predictions_SVM)/len(predictions_SVM)

    # Calculate Label Accuracy
    data['label_acc'] = postprocessing.cal_label_accuracy(data['conf_matrix'], verbose  = 1)

    postprocessing.write_results_txt_SVM( output_file = data['output_file'],  
                      test_acc = data['test_acc'] , 
                      label_acc = data['label_acc'], 
                      sent_tokenizer = sent_tokenizer, 
                      use_nltk_cleaning = use_nltk_cleaning, 
                      text_cleaning = text_cleaning , 
                      use_tfidf_tokenizer = use_tfidf_tokenizer, 
                      use_keras_tokenizer = use_keras_tokenizer, 
                      use_pretrained_embeddings = use_pretrained_embeddings,
                      use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights,
                      use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
                      imbalanced_classes = imbalanced_classes,
                      make_all_other_classes_1 = make_all_other_classes_1,
                      remove_class_0 = remove_class_0,
                      C = C ,
                      kernel = kernel,
                      degree = degree, 
                      gamma = gamma,
                      class_weight = class_weight )



def train_CNN(data, param_grid,
            sent_tokenizer, 
            use_nltk_cleaning, 
            text_cleaning, 
            use_tfidf_tokenizer, 
            use_keras_tokenizer, 
            use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights):

    #----------------------------------------------------------------#
    # Run CNN with Hyperparameter Optimization
    #----------------------------------------------------------------#

    model_output = modelling.hyperparameter_optimization( 
                                    X_train = data['X_train_CNN'], 
                                    Y_train = data['Y_train_CNN'], 
                                    X_test = data['X_test_CNN'], 
                                    Y_test = data['Y_test_CNN'] , 
                                    epochs = data['epochs'] , 
                                    batch_size = data['batch_size'],
                                    param_grid = param_grid,
                                    cv = data['cv'], 
                                    n_iter = data['n_iter'],
                                    verbose = False)

    # 5. Score Analysis

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(model_output['Y_pred'], data['Y_test_CNN'].argmax(axis=1)) / len(model_output['Y_pred'])

    # Calculate Label Accuracy
    model_output['label_acc'] = postprocessing.cal_label_accuracy(conf_matrix, verbose  = 1)

    # 6. Write Results to text file
    postprocessing.write_results_txt_CNN(output_file = data['output_file'], 
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

    # Set seed
    np.random.seed(config['params']['model']['seed'])

    # Current Date
    current_time = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

    # Input data
    input_data = config['params']['input_data']['data']
    read_type  = config['params']['input_data']['read_type']

    # Models to run
    running_CNN = config['params']['model']['running_CNN']
    running_SVM = config['params']['model']['running_SVM']

    # To test for only clases 0 and 1
    make_all_other_classes_1 =  config['params']['tokenization_options']['make_all_other_classes_1']
    remove_class_0 = config['params']['tokenization_options']['remove_class_0']


    # Sentence Tokenizer
    sent_tokenizer = config['params']['tokenization_options']['sent_tokenizer'] # TODO: Adjust for input to CNN

    # Text Cleaning Options
    use_nltk_cleaning = config['params']['tokenization_options']['use_nltk_cleaning']
    text_cleaning = config['params']['tokenization_options']['text_cleaning']

    # Word Tokenizer Options
    use_tfidf_tokenizer = config['params']['tokenization_options']['use_tfidf_tokenizer'] # For SVM
    use_keras_tokenizer = config['params']['tokenization_options']['use_keras_tokenizer'] # For CNN

    # If set to FALSE then keras embedding space training is used instead
    # Embedding Space possibilites are GloVe or TFIDF
    use_pretrained_embeddings = config['params']['tokenization_options']['use_pretrained_embeddings']

    # Only if use_pretrained_embeddings == True then select embedding vector space type
    use_glove_pretrained_embeddings_weights = config['params']['tokenization_options']['use_glove_pretrained_embeddings_weights']
    use_tfidf_as_embedding_weights = config['params']['tokenization_options']['use_tfidf_as_embedding_weights']

    # Options for SVM
    imbalanced_classes = config['params']['tokenization_options']['imbalanced_classes']
    C = config['params']['data']['SVM']['C']
    kernel = config['params']['data']['SVM']['kernel']
    degree = config['params']['data']['SVM']['degree']
    gamma = config['params']['data']['SVM']['gamma']

   
    class_weight = {0: config['params']['data']['SVM']['class_weights']['0'],
                    1: config['params']['data']['SVM']['class_weights']['1'],
                    2: config['params']['data']['SVM']['class_weights']['2'],
                    3: config['params']['data']['SVM']['class_weights']['3'],
                    4: config['params']['data']['SVM']['class_weights']['4']}


    if make_all_other_classes_1: 

        class_weight = {0: config['params']['data']['SVM']['class_weights_2']['0'],
                    1: config['params']['data']['SVM']['class_weights_2']['1']}
    
    if remove_class_0: 
        # Remember that without the 0 , the other labels are reindexed
        class_weight = {0: config['params']['data']['SVM']['class_weights_1_2_3_4']['0'],
                        1: config['params']['data']['SVM']['class_weights_1_2_3_4']['1'],
                        2: config['params']['data']['SVM']['class_weights_1_2_3_4']['2'],
                        3: config['params']['data']['SVM']['class_weights_1_2_3_4']['3']}

    # Dictionary which will cotain all the model's variables
    data = {}

    # Initialize Model
    data['epochs'] = config['params']['data']['epochs'] # NO. of optimization runs
    data['batch_size'] = config['params']['data']['batch_size'] # No. of sentences batch to train
    data['num_words'] = config['params']['data']['num_words'] # No. of words to use in the embedding space of GloVe or TFIDF
    data['cv'] = config['params']['data']['cv'] # No. of Cross Validations
    data['n_iter'] = config['params']['data']['n_iter'] # No. of Iterations
    data['seq_input_len'] = config['params']['data']['seq_input_len'] # Length of the vector sentence ( no. of words per sentence)
    data['embedding_dim'] = config['params']['data']['embedding_dim'] # Length of the word vector ( dimension in the embedding space)
    data['nodes_hidden_dense_layer'] = config['params']['data']['nodes_hidden_dense_layer'] # No. of nodes for hidden Dense layer


    data['filepath'] = config['params']['data']['filepath'] # File path to GLoVe pretrained embedding words
    data['output_file'] = f"results/{current_time}_Result" # Name of output result file

    # Hyperparameters for CNN
    param_grid = dict(num_filters_cv = config['params']['hyperparam']['num_filters_cv'], # No of filter to use in convolution
                  kernel_size_cv = config['params']['hyperparam']['kernel_size_cv'], # No of words to check per Convolution 
                  vocab_size = config['params']['hyperparam']['vocab_size'], # Vocab size if keras embedding space training is wanted
                  embedding_dim = config['params']['hyperparam']['embedding_dim'], 
                  seq_input_len = config['params']['hyperparam']['seq_input_len'], 
                  nodes_hidden_dense_layer = config['params']['hyperparam']['nodes_hidden_dense_layer'],
                  use_pretrained_embeddings = config['params']['hyperparam']['use_pretrained_embeddings']
                   )
    #Small Test
    # param_grid = dict(num_filters_cv = [(64,16)],
    #                     kernel_size_cv = [(2,3)],
    #                     vocab_size = [5000], 
    #                     embedding_dim = [50],
    #                     seq_input_len = [50], 
    #                     nodes_hidden_dense_layer = [5],
    #                     use_pretrained_embeddings = [True])


    # Preprocess data
    data, param_grid = preprocess(data_dir = config['params']['input_data']['data'], 
                                data = data, 
                                param_grid = param_grid, 
                                read_type = config['params']['input_data']['read_type'], 
                                sep = config['params']['input_data']['read_type'],
                                remove_class_0 = remove_class_0,
                                make_all_other_classes_1 = make_all_other_classes_1, 
                                running_CNN = running_CNN, 
                                running_SVM = running_SVM)

    
    # Train and Calculate Accuracy for SVM
    if running_SVM:

        train_SVM(data = data, 
            C = C, 
            kernel = kernel, 
            degree = degree, 
            gamma = gamma, 
            class_weight = class_weight,
            sent_tokenizer = sent_tokenizer, 
            use_nltk_cleaning = use_nltk_cleaning, 
            text_cleaning = text_cleaning, 
            use_tfidf_tokenizer = use_tfidf_tokenizer, 
            use_keras_tokenizer = use_keras_tokenizer, 
            use_pretrained_embeddings = use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
            imbalanced_classes = imbalanced_classes,
            make_all_other_classes_1 = make_all_other_classes_1,
            remove_class_0 = remove_class_0)
    
    
    # Train and Calculate Accuracy for CNN
    if running_CNN:
        
        train_CNN(data = data, 
            param_grid = param_grid,
            sent_tokenizer = sent_tokenizer, 
            use_nltk_cleaning = use_nltk_cleaning, 
            text_cleaning = text_cleaning, 
            use_tfidf_tokenizer = use_tfidf_tokenizer, 
            use_keras_tokenizer = use_keras_tokenizer, 
            use_pretrained_embeddings = use_pretrained_embeddings,
            use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights,
            use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights)
    
    

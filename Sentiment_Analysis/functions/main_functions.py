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
            remove_class_0, 
            store_SVM_model, 
            timestamp,
            output_path_models, 
            output_path_parameters,
            seed):

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier

    if imbalanced_classes: 
        
        if make_all_other_classes_1: 

            SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                class_weight = class_weight)


        if remove_class_0:

            SVM = svm.SVC(C = C, 
                kernel = kernel,
                degree = degree, 
                gamma = gamma,
                class_weight = class_weight)

            

        if not(make_all_other_classes_1 and remove_class_0) and class_weight :

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

    if store_SVM_model:

        file_name = f'SVM'

        if remove_class_0:
            file_name = f'SVM_1234'

        if make_all_other_classes_1:
            file_name = f'SVM_01'

        print("make_all_other_classes_1: ",  make_all_other_classes_1)
        print("remove_class_0 :" , remove_class_0)

        postprocessing.store_to_pickle(data = SVM, 
                        output_path = output_path_models, 
                        timestamp = timestamp , 
                        file_name = file_name  )


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

    postprocessing.write_results_txt_SVM( output_file = output_path_parameters,  
                      timestamp = timestamp, 
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
                      class_weight = class_weight, 
                      seed = seed)


def preprocess(data_dir, data, param_grid, read_type, sep, remove_class_0, 
                make_all_other_classes_1, running_CNN, running_SVM,
                timestamp, output_path_vectorizer, store_tfidf_tokenizer, 
                store_keras_tokenizer, file_path_glove,  debbug = False ):

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

    if remove_class_0:

        corpus['label_orignal'] = corpus.loc[:,'label']
        corpus = corpus[~corpus['label'].isin([0])]

        # Reindex classes
        corpus['label'] = corpus['label'].map({1:0,2:1,3:2,4:3})

    print(" The unique labels are ", corpus['label'].unique())

    model_data = preprocessing.prepare_training_data(corpus)

    # Concat two dictionaries
    data = {**data, **model_data}

    if debbug:
        data['corpus'] = corpus
        return data, param_grid

    #----------------------------------------------------------------#
    # 3. Vectorization
    #----------------------------------------------------------------#


    if running_CNN:
    
        data['X_train_CNN'], data['X_test_CNN'], data['vocab_size'], data['vocab'] = modelling.keras_tokenizer(num_words = data['num_words'], 
                                                                                            sentences_train = data['sentences_train_CNN'] , 
                                                                                            sentences_test = data['sentences_test_CNN'],
                                                                                            seq_input_len = data['seq_input_len'],
                                                                                            store_keras_tokenizer= store_keras_tokenizer, 
                                                                                            remove_class_0 = remove_class_0, 
                                                                                            make_all_other_classes_1 =make_all_other_classes_1, 
                                                                                            output_path_vectorizer = output_path_vectorizer, 
                                                                                            timestamp = timestamp)
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
                                                                                            sentences_test = data['sentences_test_SVM'], 
                                                                                            timestamp = timestamp, 
                                                                                            output_path_vectorizer = output_path_vectorizer,
                                                                                            store_tfidf_tokenizer = store_tfidf_tokenizer, 
                                                                                            remove_class_0 = remove_class_0, 
                                                                                            make_all_other_classes_1 = make_all_other_classes_1
                                                                                            )

    if use_tfidf_as_embedding_weights: 
        
        data['embedding_matrix'], data['embedding_dim'] = modelling.tfidf_as_embedding_weights(num_words = data['num_words'], 
                                                                                                    corpus = data['corpus'], 
                                                                                                    sentences_train_CNN = data['X_train_CNN'])

    
    if use_glove_pretrained_embeddings_weights:  
        
        data['embedding_matrix'] = modelling.create_embedding_matrix(
                             filepath = file_path_glove,
                             word_index = data['vocab'], 
                             embedding_dim = data['embedding_dim'])

    # data_pre = modelling.data_vectorization(sentences_train_CNN = data['sentences_train_CNN'], 
    #                     sentences_test_CNN = data['sentences_test_CNN'], 
    #                     sentences_train_SVM = data['sentences_train_SVM'], 
    #                     sentences_test_SVM = data['sentences_test_SVM'], 
    #                     num_words = data['num_words'], 
    #                     seq_input_len = data['seq_input_len'], 
    #                     filepath = data['filepath'],
    #                     corpus = corpus,
    #                     vocab = data['vocab'],
    #                     embedding_dim = data['embedding_dim'],
    #                     running_CNN = running_CNN, 
    #                     running_SVM = running_SVM, 
    #                     use_tfidf_as_embedding_weights = use_tfidf_as_embedding_weights,
    #                     use_glove_pretrained_embeddings_weights = use_glove_pretrained_embeddings_weights, 
    #                     timestamp = timestamp, 
    #                     output_path_vectorizer = output_path_vectorizer, 
    #                     store_tfidf_tokenizer = store_tfidf_tokenizer)

    # Concat two dictionaries
    #data = {**data, **data_pre}

    # Add final parameters
    param_grid['embedding_matrix'] = ([data['embedding_matrix']]) 
    param_grid['output_label'] = [data['output_label']]
    param_grid['corpus'] = corpus

    return data, param_grid
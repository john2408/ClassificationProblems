import numpy as np
import pickle
from os.path import join



def load_pickle(output_path, timestamp, file_name):

    with open(join(output_path,f'{timestamp}_{file_name}.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)
    
    return loaded_data


def classify_new_sentences(new_sentence, model, vectorizer ):

    sentence_train = vectorizer.transform(new_sentence['text'])

    Pred_Y = model.predict(sentence_train)

    return Pred_Y

def store_to_pickle(data, output_path, timestamp, file_name ):
    """Function to calculate the accuracy on
    ever categorical expected output variable

    Parameters
    ----------
    conf_matrix : 
    
    Returns
    -------
    label_acc: ´dict´
        dictionary containing the label accuracy
        'label_n': ´str´ - accuracy of label n 
    """


    serialized = pickle.dumps(data)
    _file = join(output_path,f'{timestamp}_{file_name}.pkl')

    with open(_file,'wb') as file_object:
        file_object.write(serialized)



def cal_label_accuracy(conf_matrix, verbose = 0):
    """Function to calculate the accuracy on
    ever categorical expected output variable

    Parameters
    ----------
    conf_matrix : ´numpy.array´: ´str´
        Corpus text, each element being a sentence
    
    verbose: ´int´
        Controls the level of messaging. If > 1, it
        prints out the label accuracy.
    
    Returns
    -------
    label_acc: ´dict´
        dictionary containing the label accuracy
        'label_n': ´str´ - accuracy of label n 
    """
    
    
    label_acc = {}
    
    for index, x in enumerate(conf_matrix): 
        
        label_acc[index] = np.round( conf_matrix[index][index]/ sum(conf_matrix[index]) *100 , 2)
        
        if verbose > 0:
        
            print("Accuracy for label", index, ": ", label_acc[index] , " %" )
    
    return label_acc


       

def write_results_txt_CNN(output_file, best_train_acc, best_train_param, 
                      test_acc, label_acc, sent_tokenizer, use_nltk_cleaning, 
                      text_cleaning , use_tfidf_tokenizer, 
                      use_keras_tokenizer, use_pretrained_embeddings,
                      use_glove_pretrained_embeddings_weights,
                      use_tfidf_as_embedding_weights,
                      epochs,
                      batch_size,
                      num_words, 
                      cv,
                      n_iter):


    """Function to generate results file. 

    Parameters
    ----------
    output_file : 

    best_train_acc:

    best_train_param:

    test_acc:

    label_acc: 

    sent_tokenizer:

    use_nltk_cleaning: 

    text_cleaning: 

    use_tfidf_tokenizer:

    use_keras_tokenizer:

    use_pretrained_embeddings:

    use_glove_pretrained_embeddings_weights:

    use_tfidf_as_embedding_weights:
    
    """


    print("Writting results...")

    output_file  = f"{output_file}_CNN.txt"

    with open(output_file, 'w+') as f:

        output_string = f"""Running CNN Modeling \n 
            Best Accuracy : {best_train_acc}\n  
            Test Accuracy : {test_acc}\n
            epochs : {epochs} \n
            batch size : {batch_size} \n
            cross validations : {cv} \n
            No. Iterations : {n_iter} \n
            sent_tokenizer : {sent_tokenizer} \n   
            use_nltk_cleaning: {use_nltk_cleaning}\n 
            text_cleaning: {text_cleaning}\n  
            use_tfidf_tokenizer: {use_tfidf_tokenizer}\n 
            use_keras_tokenizer: {use_keras_tokenizer}\n 
            use_pretrained_embeddings: {use_pretrained_embeddings}\n 
            use_glove_pretrained_embeddings_weights: {use_glove_pretrained_embeddings_weights}\n 
            use_tfidf_as_embedding_weights: {use_tfidf_as_embedding_weights}\n 
            best param: {best_train_param}\n 
            label accuracy: {label_acc}"""  
    
        print(output_string)

        f.write(output_string)




def write_results_txt_SVM(output_file, timestamp, 
                      test_acc, label_acc, sent_tokenizer, use_nltk_cleaning, 
                      text_cleaning , use_tfidf_tokenizer, 
                      use_keras_tokenizer, use_pretrained_embeddings,
                      use_glove_pretrained_embeddings_weights,
                      use_tfidf_as_embedding_weights,
                      imbalanced_classes,
                      make_all_other_classes_1,
                      remove_class_0,
                      C,
                      kernel,
                      degree, 
                      gamma,
                      class_weight, 
                      seed):


    """Function to generate results file. 

    Parameters
    ----------
    output_file : 

    best_train_acc:

    best_train_param:

    test_acc:

    label_acc: 

    sent_tokenizer:

    use_nltk_cleaning: 

    text_cleaning: 

    use_tfidf_tokenizer:

    use_keras_tokenizer:

    use_pretrained_embeddings:

    use_glove_pretrained_embeddings_weights:

    use_tfidf_as_embedding_weights:

    seed: 
    
    """


    print("Writting results...")


    txt_name = f'{timestamp}_SVM.txt'

    if make_all_other_classes_1:
        txt_name = f'{timestamp}_SVM_01.txt'

    if remove_class_0:
        txt_name = f'{timestamp}_SVM_1234.txt'

    output_file  = join( output_file, txt_name )

    

    with open(output_file, 'w+') as f:

        output_string = f"""Running SVM Modeling \n  
            Seed : {seed}\n
            Test Accuracy : {test_acc}\n
            C : {C}\n
            kernel : {kernel}\n
            degree : {degree}\n 
            gamma : {gamma}\n
            class_weight : {class_weight}\n
            sent_tokenizer : {sent_tokenizer} \n   
            use_nltk_cleaning: {use_nltk_cleaning}\n 
            text_cleaning: {text_cleaning}\n  
            make_all_other_classes_1: {make_all_other_classes_1}\n  
            remove_class_0: {remove_class_0} \n
            use_tfidf_tokenizer: {use_tfidf_tokenizer}\n 
            use_keras_tokenizer: {use_keras_tokenizer}\n 
            use_pretrained_embeddings: {use_pretrained_embeddings}\n 
            use_glove_pretrained_embeddings_weights: {use_glove_pretrained_embeddings_weights}\n 
            use_tfidf_as_embedding_weights: {use_tfidf_as_embedding_weights}\n 
            imbalanced_classes: {imbalanced_classes}\n 
            label accuracy: {label_acc}"""  
    
        print(output_string)

        f.write(output_string)
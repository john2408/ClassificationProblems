import numpy as np

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
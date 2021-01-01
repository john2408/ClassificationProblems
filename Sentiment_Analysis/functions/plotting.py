def plot_history(history):
    """Function to create two plots. 
    1. Training and Test accuracy
    2. Cost function on train and test data set

    Parameters
    ----------
    history : ´dict´
        dict containing the data for
        'accuracy': training accuracy
        'val_accuracy': testing accuracy
        'loss': cost function on training data set
        'val_loss': cost function on test data set
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # create x values - number of training runs
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
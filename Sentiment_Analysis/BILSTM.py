model = Sequential()

model.add(layers.Embedding(embedding_matrix.shape[0], embedding_dim, 
                            weights = [embedding_matrix], 
                            input_length = seq_input_len, 
                            trainable = False))

model.add(layers.Bidirectional(layers.LSTM(64,  return_sequences=True)))

model.add(layers.Bidirectional(layers.LSTM(32)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(output_label, activation="sigmoid"))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.summary()

model.fit(X_train_keras, Y_train,
                    epochs = epochs,
                    verbose = False,
                    validation_data = (X_test_keras, Y_test),
                    class_weight = create_class_weight(labels_dict),
                    batch_size = batch_size, 
                    callbacks = callback)


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

history = model.fit(X_train_keras, Y_train,
                    epochs = 15,
                    verbose = False,
                    validation_data = (X_test_keras, Y_test),
                    batch_size = batch_size, 
                    class_weight = create_class_weight(labels_dict),
                    )

loss, accuracy = model.evaluate(X_train_keras, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_keras, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
 

Y_pred = model.predict(X_test_keras)

for pred in Y_pred:
    for index, value in enumerate(pred):
        if value == max(pred):
            pred[index] = int(1)
        else: 
            pred[index] = int(0)


conf_matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1)) / len(Y_pred)
cal_label_accuracy(conf_matrix, verbose = 1)
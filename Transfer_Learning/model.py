class Model:
    
    def __init__(self):
        """
        Initialize the neural network model architecture.
        """
        self.cnn_model = Sequential()
        self.cnn_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
        self.cnn_model.add(GlobalMaxPooling1D())
        self.cnn_model.add(Dense(16, activation='relu'))
        self.cnn_model.add(Dense(1, activation='linear'))
        self.cnn_model.compile(optimizer='adam', loss='mse')

    def fit(self,sequences,labels,weights,epochs):
        
        # Shuffle sequences and labels
        shuffled_indices = np.arange(len(sequences))
        np.random.shuffle(shuffled_indices)
        sequences = sequences[shuffled_indices]
        labels = labels[shuffled_indices]

        if weights is not None: 
            weights=weights[shuffled_indices]
            # fit model on data
            self.cnn_model.fit(sequences, labels, epochs=epochs, batch_size=32, verbose=1,sample_weight= weights)
        else:
            #fit model on data
            self.cnn_model.fit(sequences, labels, epochs=epochs, batch_size=32, verbose=1)
            
    def predict(self,test):
        
        return self.cnn_model.predict(test)
    
    def save(self):
        
        self.cnn_model.save('pretrained_cnn_model.h5')

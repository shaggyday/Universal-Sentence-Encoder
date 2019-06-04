# Universal-Sentence-Encoder
## REU project

### June 3rd
    Goal: apply USE to deep neural network that uses keras (instead of tf.estimator)
    Model:  USE lambda layer
            dense (512) - relu
            dropout (0.4)
            dense (128) - relu
            dropout (0.4)
            dense (1) - sigmoid
    Results:
    ![Accuracy](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/acc.png)
        highest validation accuracy is 86.82%
        suffers from overfitting

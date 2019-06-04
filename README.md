# Universal-Sentence-Encoder
## REU project

June 3rd
* Goal: apply USE to deep neural network that uses keras (instead of tf.estimator). Run on IMBD raw data. Treats an entire review as **an individual sentence**.
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.0.py):
  * USE lambda layer
  * dense (512) - relu
  * dropout (0.4)
  * dense (128) - relu
  * dropout (0.4)
  * dense (1) - sigmoid
  * epochs = 20, batch_size = 64
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras%20v_1.0%20results.pdf):
  * highest validation accuracy is 86.82%
  * suffers from overfitting\
  ![Accuracy](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/acc.png)
\
June 4th
* Goal:= June 3rd.goal
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.1.py):
  * USE lambda layer
  * **dense (256) - relu**
  * dropout (0.4)
  * **dense (64) - relu**
  * dropout (0.4)
  * dense (1) - sigmoid
  * epochs = 20, batch_size = 64
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras%20v_1.0%20results.pdf):
  * highest validation accuracy is 86.82%
  * suffers from overfitting\
  ![Accuracy](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/acc.png)

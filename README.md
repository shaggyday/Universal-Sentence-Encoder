# Universal-Sentence-Encoder
Goal: apply USE to DNN using keras. Run on IMBD raw data. **Treats an entire review as an individual sentence**.
## v1.1 
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.1.py):
  * USE lambda layer
  * ~~dense (512) - relu~~
  * **dense (256) - relu** 
  * dropout (0.4)
  * ~~dense (128) - relu~~
  * **dense (64) - relu** 
  * dropout (0.4)
  * dense (1) - sigmoid
  * epochs = 20, batch_size = 64
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.1_results.pdf):
  * ~~highest validation accuracy is 86.82%~~
  * highest validation accuracy is **86.94%**
  * **still** suffers from overfitting\
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.1_plot.png)

## v1.0
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
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v_1.0_plot.png)

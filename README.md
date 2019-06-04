# Universal-Sentence-Encoder
Goal: apply USE to DNN using keras. Run on IMBD raw data. **Treats an entire review as an individual sentence**.
## v1.4.1
* [Pre embedded data](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/picklem.py) into [pickle files]
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4.1.py)
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4_results.pdf):
  * ~~highest validation accuracy is 87.16%~~
  * **highest validation accuracy is 87.12%**
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4.1_plot.png)
* Approximately the same results and accuracy, but time decreased from 80s/epoch to **2s/epoch**
  
## v1.4
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4.py):
  * USE lambda layer
  * dense (128) - relu
  * dropout (0.4)
  * dense (1) - sigmoid
  * ~~epochs = 20, batch_size = 64~~
  * **epochs = 20, batch_size = 32**
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4_results.pdf):
  * ~~highest validation accuracy is 87.11%~~
  * **highest validation accuracy is 87.16%**
  * **less** overfitting and **higher** validation accuracy
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.4.plot.png)

## v1.3
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.3.py):
  * USE lambda layer
  * ~~dense (256) - relu~~
  * **dense (128) - relu**
  * dropout (0.4)
  * dense (1) - sigmoid
  * epochs = 20, batch_size = 64
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.3_results.pdf):
  * ~~highest validation accuracy is 87.18%~~
  * **highest validation accuracy is 87.11%**
  * **less** overfitting but also **lowers** validation accuracy
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.3.plot.png)

## v1.2
* [Model](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.2.py):
  * USE lambda layer
  * dense (256) - relu 
  * dropout (0.4)
  * ~~dense (64) - relu~~
  * ~~dropout (0.4)~~
  * dense (1) - sigmoid
  * epochs = 20, batch_size = 64
* [Results](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.2_results.pdf):
  * ~~highest validation accuracy is 86.94%~~
  * **highest validation accuracy is 87.18%**
  * **less** overfitting than before because of weaker model ...\
  ![Plot](https://github.com/shaggyday/Universal-Sentence-Encoder/blob/master/USE/USE%2BIMBD%2Bkeras_v1.2.plot.png)
  
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

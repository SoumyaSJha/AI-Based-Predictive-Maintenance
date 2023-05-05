# AI-Based-Predictive-Maintenance

## Introduction
This project aims to build deep learning models based on Long Short-Term Memory (LSTM) network and Autoencoders to predict failures in hard drives and detect anomalies, respectively. The models are trained on the Backblaze data center hard drive dataset which includes information about the hard drive's health, performance, and various other features.

## Why AI-for-Predictive-Maintenance ?
According to the International Society of Automation, $647 billion is lost globally each year due to downtime from machine failure. Organizations across manufacturing, aerospace, energy, and other industrial sectors are overhauling maintenance processes to minimize costs and improve efficiency. With artificial intelligence (AI) and machine learning, organizations can apply predictive maintenance to their operation, processing huge amounts of sensor data to detect equipment failure before it happens. Compared to routine-based or time-based preventative maintenance, predictive maintenance gets ahead of the problem and can save a business from costly downtime.

## Dataset Used
Backblaze. "Hard Drive Failure Rates." Backblaze, https://www.backblaze.com/b2/hard-drive-test-data.html#downloading-the-raw-hard-drive-test-data (accessed November 23, 2022)

## LSTM Model
The LSTM model is designed to predict hard drive failures. The model takes in sequences of data for a particular hard drive model and outputs a binary classification (i.e. failed or not failed). The model architecture consists of multiple LSTM layers followed by a dense layers for classification. 

## Autoencoder Model
The Autoencoder model is designed to detect anomalies in the data. The model is trained on non-failure sequences and is expected to reconstruct it with low error. Any data that results in high reconstruction error (failure sequences) is considered anomalous. The model architecture consists of multiple lstm layers for encoding and decoding.

## Results
The LSTM model classified the test sequences as failure and non failure with 89.09% recall and 80.96% accuracy. For anomaly detection, the autoencoders model was trained with only non-failure sequences, as it involves unsupervised learning. The model predicted 156 sequences as failure sequences of the 158 actual failure sequences in test data.

## Conclusion
The LSTM and Autoencoder models have shown promising results in predicting hard drive failures and detecting anomalies, respectively. These models can be further improved with more sophisticated architectures and by incorporating more features from the dataset.

## References
- Cahyadi and M. Forshaw, ["Hard Disk Failure Prediction on Highly Imbalanced Data using LSTM Network,"](https://ieeexplore.ieee.org/document/9671555) 2021 IEEE International Conference on Big Data (Big Data), Orlando, FL, USA, 2021, pp. 3985-3991.

- Yufei Wang, Xiaoshe Dong, Longxiang Wang, Weiduo Chen, and Xingjun Zhang. 2022. ["Optimizing Small-Sample Disk Fault Detection Based on LSTM-GAN Model."](https://dl.acm.org/doi/abs/10.1145/3500917) ACM Trans. Archit. Code Optim. 19, 1, Article 13 (March 2022), 24 pages.




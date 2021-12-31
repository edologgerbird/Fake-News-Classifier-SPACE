# Fake News Classifier SPACE
We explore combinations of Machine Learning models and vectorisation methods in classifying Covid-19 Fake News, evaluating models using a novel metric which combines F1-Score and training time.

## Authors:
- Loh Hong Tak Edmund
- Darren Lee Sheng Hao
- Tan Yi Bing
- Teo Zhi Feng
- Yap Kai Herng

## About the Project

### Objective

This project aims to create a classification model that effectively predicts Covid-19 Fake News given a piece of text. 

### Research Question

Which text vectorisation method and machine learning model can best predict Covid-19 Fake News across classification performance and time?

### Methodology

This project explores different text verctorisation methods and machine learning models to determine which vectorisation method-machine learning model pair produces the best classification performance.

Vectorisation methods:
- Count Vectorisation
- TF-IDF Vectorisation
- Word2Vec

Machine Learning Models:
- TextCNN
- LSTM (Unidirectional)
- LSTM (Bidirectional)
- LSTM + CNN
- BiLSTM + CNN
- XGBoost Classifier
- RandomForest Classifier
- Gaussian Naive Bayes Classifier
- Multinomial Naive Bayes Classifier
- PassiveAggressive Classifier

### Metrics for Evaluation - SPACE

There are two dimensions that we will be considering in our evaluation of each model: Classification Performance and Training Time. Training time is an important aspect is due to the nature of the problem at hand. Fake news classification over social media entails a large and constant stream of data, and fake news trends change all the time. In addition, predictions must be made promptly and in real time. This requires constant updates of models to account for drift, which means constantly retaining. Models that take a long time to train are thus unsuited for the fast-moving nature of the problem at hand. 

Hence, we propose a Speed and Performance Adjusted Combined Evaluation, SPACE, as a way to score models. SPACE(n) is evaluated as: 

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;tanh(\frac{F1^{n}}{tanh(time)}))

where n is an indicated preference of classification performance over training speed. 


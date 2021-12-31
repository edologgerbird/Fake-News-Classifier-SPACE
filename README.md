# Fake News Classifier SPACE
We explore combinations of Machine Learning models and vectorisation methods in classifying Covid-19 Fake News, evaluating models using a novel metric which combines F1-Score and training time.

## Authors:
- Loh Hong Tak Edmund
- Darren Lee Sheng Hao
- Tan Yi Bing
- Teo Zhi Feng
- Yap Kai Herng

## Codes and Resources Used

**Python Version:** 3.9.5
**Built with:** Jupyter Notebook
**Packages:** pandas, numpy, matplotlib, collections, seaborn, imblearn, sklearn, tensorflow, keras, wordcloud, scipy, statistics, xgboost, yellowbricks

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

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;tanh(\frac{F1^{n}}{tanh(time)}))

where n is an indicated preference of classification performance over training speed. 

### Dataset

For this project, we will be using the [COVID19 Fake News Dataset NLP](https://www.kaggle.com/elvinagammed/covid19-fake-news-dataset-nlp?select=Constraint_Train.csv) from Kaggle. It contains a column comprising the news content and a column comprising the label (real or fake) for each news content.

### Results
Using our SPACE(24) metric, our PAC model with TF-IDF gave us the best blended performance and has an exceptional blend of performance and speed, with a SPACE(24) score of 1.00000. Hence, it would be our pick to achieve our project objective of  predicting Covid-19 Fake News given our selected dataset. Notably, it has been designed as an online learning algorithm that accounts for the memory and speed requirements that learning from an large, unending, online stream of data entails. 

Analysing model performance using the other metrics, the Bidirectional LSTM-CNN with Word2Vec is our best performer in terms of classification performance, achieving an F1 score of 0.9374 and has the potential to be increased with further hyperparameter tuning. Our fastest model is Multinomial Naive Bayes with Word2Vec, training in 0.0110s. However, it must be noted that this model performed terribly in classifying fake news, achieving 0.6056 for F1 score and 0.000539 for our SPACE metric. 

For full Project Methodology, Analysis, Results and Insights, please refer to our Project Report [here](https://github.com/edologgerbird/Fake-News-Classifier-SPACE/blob/main/project-report.pdf) 

## References
BBC News. (2020, April 24). Coronavirus: Trump’s disinfectant and sunlight claims fact-checked. https://www.bbc.com/news/world-us-canada-52399464

Chen, T., & Guestrin, C. (2016, August). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).

Chew, H. M. (2020, May 21). 6 in 10 people in Singapore have received fake COVID-19 news, likely on social media: Survey. CNA. https://www.channelnewsasia.com/singapore/fake-covid-19-news-study-ncid-messaging-platforms-whatsapp-673131

Depoux, A., Martin, S., Karafillakis, E., Preet, R., Wilder-Smith, A., & Larson, H. (2020). The pandemic of social media panic travels faster than the COVID-19 outbreak. Journal of Travel Medicine, 27(3). https://doi.org/10.1093/jtm/taaa031

Freeman, D., Waite, F., Rosebrock, L., Petit, A., Causier, C., East, A., Jenner, L., Teale, A. L., Carr, L., Mulhall, S., Bold, E., & Lambe, S. (2020). Coronavirus conspiracy beliefs, mistrust, and compliance with government guidelines in England. Psychological Medicine, 1–13. https://doi.org/10.1017/s0033291720001890

Goldberg, Y. (2017). Neural network methods for natural language processing. Synthesis lectures on human language technologies, 10(1), 1-309.

KPMG. (2020, July). The Impact of Covid-19 - A Global to Local Overview. https://assets.kpmg/content/dam/kpmg/sg/pdf/2020/08/The-Impact-of-COVID-19.pdf

Lazer, D. M. J., Baum, M. A., Benkler, Y., Berinsky, A. J., Greenhill, K. M., Menczer, F., Metzger, M. J., Nyhan, B., Pennycook, G., Rothschild, D., Schudson, M., Sloman, S. A., Sunstein, C. R., Thorson, E. A., Watts, D. J., & Zittrain, J. L. (2018). The science of fake news. Science, 359(6380), 1094–1096. https://doi.org/10.1126/science.aao2998

Lim, Y.L. (2021, October 25). Pofma correction direction issued to Truth Warriors website over Covid-19 false claims. The Straits Times. https://www.straitstimes.com/singapore/correction-direction-issued-to-truth-warriors-website

Nasir, J. A., Khan, O. S., & Varlamis, I. (2021). Fake news detection: A hybrid CNN-RNN based deep learning approach. International Journal of Information Management Data Insights, 1(1), 100007.

Nyilasy, G. N. (2021, November 1). Fake news in the age of COVID-19. Pursuit. https://pursuit.unimelb.edu.au/articles/fake-news-in-the-age-of-covid-19

Preston, S., Anderson, A., Robertson, D. J., & Huhe, N. (2021, November). Detecting fake news on Facebook: The role of emotional intelligence. PLOS ONE. https://doi.org/10.1371/journal.pone.0246757

Revez, J., & Corujo, L. (2020, December). Librarians against fake news: A systematic literature review of library practices (Jan. 2018–Sept. 2020). The Journal of Academic Librarianship. https://doi.org/10.1016/j.acalib.2020.102304 

Stewart, E. (2021). Detecting Fake News: Two Problems for Content Moderation. Philosophy & Technology. Published. https://doi.org/10.1007/s13347-021-00442-x

Surowiecki, J. (2021, November 12). Covid misinformation spreads because so many Americans are awful at math. The Washington Post. https://www.washingtonpost.com/outlook/math-covid-vaccinations-jeremy-mcanulty/2021/11/12/bfe89018-417f-11ec-a3aa-0255edc02eb7_story.html

TensorFlow. (n.d.). Recurrent neural networks (RNN) with Keras. https://www.tensorflow.org/guide/keras/rnn

Uscinski, J. E., Enders, A. M., Klofstad, C., Seelig, M., Funchion, J., Everett, C., Wuchty, S., Premaratne, K., & Murthi, M. (2020). Why do people believe COVID-19 conspiracy theories? Harvard Kennedy School Misinformation Review. Published. https://doi.org/10.37016/mr-2020-015

Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science, 359(6380), 1146-1151.

 Yoon, K. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.

Zhou, C., Sun, C., Liu, Z., & Lau, F. (2015). A C-LSTM neural network for text classification. arXiv preprint arXiv:1511.08630.


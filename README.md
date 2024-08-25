Last updated: August 2024

New in 2024:
- "9. Machine learning with **PySpark**: Blackblaze hard drive failure prediction"
  - Keyword: **PySpark**
  - Linked from a separate [repo](https://github.com/cedricyu803/blackblaze_hdd_pyspark)
- "8. End-to-end example with the Disater Tweets dataset:
  - Keywords: **FastAPI**, **Streamlit**, **GitHub workflow**
  - Linked from a separate [repo](https://github.com/cedricyu803/e2e_disaster_tweets).

# Data Science Work Samples
This folder contains select works I have done in data science and machine learning. <br>
From this collection I have omitted a lot of works on different other topics that I did following some Coursera courses. They include: machine translation and trigger-word detection with LSTM; object detection with YOLOv3 using transfer learning, image segmentation using U-Net, face recognition using the Siamese network, image classification using InceptionV3 with data augmentations, and then some.

Due to copyrights, I have removed the datasets and pre-trained models.

Each folder contains a PDF report of what I have done, as well as the relevant Python scripts and some plots.

**1. Detroit Blight Ticket Compliance**

**Keywords**: tabular, classification, cross validation. Time series forecast, autoregression

A binary classification problem on predicting whether a given set of blight tickets will be paid
on time. (Blight violations are issued by the city of Detroit, Michigan, to individuals who allow their
properties to remain in a deteriorated condition.) 

Informed by an exploratory data analysis, we clean up the data (missing and mis-labeled values), and engineer features. The latter includes extracting mailing address zip codes and ticket datetime features, as well as encoding categorical features.

We then train a few models, namely KNeighborsClassifier, XGBClassifier, RandomForestClassifier, LGBMClassifier and deep neural networks, and tune the hyperparameters. The model performances are evaluated by the validation set's AUC of the ROC curve. The best performance is attained by RandomForestClassifier and LGBMClassifier. Our models predict higher compliance rates than the training dataset. 

We explore the monthly aggregated compliance rate as a time series and find that it can be described by an AR(1) model. We also explore the impact of dimensionality reduction on model performance.

**2. Competitive Data Science: Predict Future Sales**

**Keywords**: time series, regression, RegEx, feature encoding and engineering, lag features, feature importance

Given the sales records of the items sold in each shop of 1C Company during the time period January 2013 – October 2015, we are tasked with predicting the total sales (by number) for every product and store in the coming month, i.e. November 2015.

We are asked to predict the sales, i.e. time series, of O(10^5) items in different shops. Given the large number of the time series and possible interactions between them, it would be computationally costly to fit time-series models on them. Therefore, we employ tree-based algorithms and deep neural networks, but perform feature engineering in a time-conscious manner. This includes the engineering and use of lag features, as well as performing train-validation split in a time-ordered way.

In our exploratory data analysis, we identify seasonal trends on the sales, as well as annual decline in sales. This helps us in deciding the lag features of the sales and item prices to use. We also perform other feature engineering tasks, such as extracting city and type of the shops, as well as the category and platform of the items.

We then train three tree-based models: XGBRegressor, RandomForestRegressor and LGBMRegressor, chosen due to their superior performance and faster training time. We also explore the feature importances. Our best RMSE on the test set is 1.50672.

**3. New York City Taxi Fare Prediction**

**Keywords**: large dataset, time series, regression, reverse geocoding, KDTree

We are given a large training dataset of ~55M instances, with pickup and dropoff datetime and locations, and passenger count as features.

The large size of the dataset requires an efficient memory management and optimised workflow. To this end, we process the datetime features to extract year, month, etc. and save into new files. We also downcast datatypes where appropriate. 

From our exploratory data analysis, we identify outliers in the features and training labels. We also identify potentially important features such as the pickup and dropoff boroughs and zip codes, whether they involve one of the three airports. In addition to these new features, as part of the feature engineering process, we compute the Euclidean geodesic distance of each trip--- a compromise over the driving distance considering the large dataset size and the limited capability of the local machine (my laptop). 

We train XGBRegressor and LGBMRegressor on 20M of the 55M training samples (to fit them into my 16GB RAM). After tuning the hyperparameters, we attained our best test RMSE of 2.96150 with LGBMRegressor. The test prediction follows closely the distribution of the training set fares (with outliers discarded), with expected small peaks that can be attributed to the flat rate between Manhattan and the airports.

We also performed a baseline model evaluation with minimal pre-processing on Google Cloud.

**4. IMDB Movie Review Sentiment Classification**

**Keywords**: NLP, text preprocessing, NLTK, TfidfVectorizer, GloVe Embedding, LSTM, Huggingface BERT transformer

Classifying the sentiment of IMDB movie reviews into good or bad. 

We first perform text cleanup and pre-processing. We make use of various techniques in natural language processing: CountVectorizer and TfidfVectorizer from nltk, LSTM with GloVe embedding vectors, and a transformer network from Hugging Face, and compare the results and performance of different approaches. 

All these approaches result in similar AUC validation and test socres, with the transformer network giving the best score (0.92304). The similarity in scores suggests that the order of the words is only marginally important. Given the significantly higher computational costs and slight gain in performance of the neural netowrk models, TfidfVectorizer turns out to be the most cost-effective option.

**5. Disaster Tweets Classification**

**Keywords**: NLP, text preprocessing, NLTK, TfidfVectorizer, GloVe Embedding, LSTM, Huggingface BERT transformer

Classifying tweets into whether they are referring to a disaster. 

Informed by an exploratory data analysis, we perform a text pre-processing, and extract and engineer features from the texts. Once again, we make use of various techniques in natural language processing: CountVectorizer and TfidfVectorizer from nltk, LSTM with GloVe embedding vectors, and a transformer network from Hugging Face, and compare the results and performance of different approaches. In particular, an optimal LSTM-based neural network is found using Keras Tuner. Except for the transformer network, in all our approaches, we use both the text vectors and other extracted features in training the models and making predictions.

All these approaches result in similar F1 scores, with the transformer network giving the best score (0.80784). Due to the small dataset size, no model has a significantly longer training time. This however contributes to a lower-than-desired performance and overfitting in the LSTM-based networks. 

**6. Named Entity Recognition in Resumes**

**Keywords**: NLP, named-entity recognition, text preprocessing, sequence classification, Huggingface, BERT tokenization, BERT transformer

We perform a named-entity recognition task on a dataset of resumes. Our approach is based on an assignment in the Sequence Models course on Coursera, offered by DeepLearning.AI. We expand it by studying the raw data, performing a more accurate and streamlined tokenisation, before re-training a Huggingface transformer model. Evaluating the model performance on the F1 score, we find a macro-averaged F1 score of 0.71 on the validation set.

**7. Time Series: Global Temperatures**

**Keywords**: time series forecasting, rolling forecast, statistical models, stationarity, autocorrelation, autoregression, LSTM, sequence models

We study the global temperatures dataset available on Kaggle. In the Jupyter notebook EDA_AR(3).ipynb, we perform an exploratory data analysis, and decide to take the yearly averages and only use the data from 1850-2015. We find a clear increasing trend (hence global 'warming'), which is rendered stationary by taking the first difference. The auto-correlation function (ACF) and partial auto-correlation function (PACF) suggest that the first difference can be described by an AR(3) model. We fit AR(3) models. In LSTM.ipynb, we make use of lag features and fit LSTM models on the series. For both models, we use fixed partitioning and rolling forecast.

In the end, on the validation set (yearly average temperature in 1986-2015), the mean absolute error (MAE) from LSTM fitted on first difference using rolling forecast is the lowest: 0.14682. This is only slight lower than that from using fixed partition. It is to be compared to the MAE of 0.18931 from naive forecast (lag 1), and 0.15548 from the AR(3) model using rolling forecast.

**8. End-to-end example with the Disater Tweets dataset**

**Keywords**: FastAPI, Streamlit, deployment, GitHub workflow, end-to-end (e2e), NLP, text preprocessing, NLTK, TfidfVectorizer

See https://github.com/cedricyu803/e2e_disaster_tweets.

We provide an end-to-end example based on the 'Disaster Tweet Classification' example, which classifies a tweet (text string) into whether it is a disaster or not. Previously, we performed an exploratory data analysis and ran model architecture and hyperparameter searches, done in `5. Disaster Tweets Classification`.

In a separate [repo](https://github.com/cedricyu803/e2e_disaster_tweets), we implement a) a training pipeline, b) an inference **FastAPI** backend, and c) an example **Streamlit** app to query the FastAPI backend.

**9. Machine learning with PySpark: Blackblaze hard drive failure prediction**

**Keywords**: PySpark, time series

See https://github.com/cedricyu803/blackblaze_hdd_pyspark.

We provide an end-to-end PySpark example with the 'Blackblaze hard drive failure prediction' example, in which we predict from the S.M.A.R.T. telemetry from a hard drive whether it has failed. The goal of this exercise is to showcase the use of **PySpark** to perform data science tasks on a real dataset: from EDA, date cleaning and preprocessing, to feature engineering and scaling, ML model training, to inference.
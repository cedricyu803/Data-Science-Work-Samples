# Data Science Work Samples
This folder contains select works I have done in data science and machine learning. 

Due to copyrights, I have removed the datasets.

Each folder contains a PDF report of what I have done, as well as the relevant Python scripts and some plots.

**1. Detroit Blight Ticket Compliance**

A binary classification problem on predicting whether a given set of blight tickets will be paid
on time. (Blight violations are issued by the city of Detroit, Michigan, to individuals who allow their
properties to remain in a deteriorated condition.) 

Informed by an exploratory data analysis, we clean up the data (missing and mis-labeled values), and engineer features. The latter includes extracting mailing address zip codes and ticket datetime features, as well as encoding categorical features.

We then train a few models, namely KNeighborsClassifier, XGBClassifier, RandomForestClassifier, LGBMClassifier and deep neural networks, and tune the hyperparameters. The model performances are evaluated by the validation set's AUC of the ROC curve. The best performance is attained by RandomForestClassifier and LGBMClassifier. Our models predict higher compliance rates than the training dataset. 

We explore the monthly aggregated compliance rate as a time series and find that it can be described by an AR(1) model. We also explore the impact of dimensionality reduction on model performance.

**2. Competitive Data Science: Predict Future Sales**

A Kaggle competition: given the sales records of the items sold in each shop of 1C Company during the time period January 2013 – October 2015, we are tasked with predicting the total sales (by number) for every product and store in the coming month, i.e. November 2015.

We are asked to predict the sales, i.e. time series, of O(10^5) items in different shops. Given the large number of the time series and possible interactions between them, it would be computationally costly to fit time-series models on them. Therefore, we employ tree-based algorithms and deep neural networks, but perform feature engineering in a time-conscious manner. This includes the engineering and use of lag features, as well as performing train-validation split in a time-ordered way.

In our exploratory data analysis, we identify seasonal trends on the sales, as well as annual decline in sales. This helps us in deciding the lag features of the sales and item prices to use. We also perform other feature engineering tasks, such as extracting city and type of the shops, as well as the category and platform of the items.

We then train three tree-based models: XGBClassifier, RandomForestClassifier and LGBMClassifier, chosen due to their superior performance and faster training time. We also explore the feature importances. Our best RMSE on the test set is 1.50672.

**3. New York City Taxi Fare Prediction**

A Kaggle competition about predicting taxi fares in New York City. We are given a large training dataset of ~55M instances, with pickup and dropoff datetime and locations, and passenger count as features.

The large size of the dataset requires an efficient memory management and optimised workflow. To this end, we process the datetime features to extract year, month, etc. and save into new files. We also downcast datatypes where appropriate. 

From our exploratory data analysis, we identify outliers in the features and training labels. We also identify potentially important features such as the pickup and dropoff boroughs and zip codes, whether they involve one of the three airports. In addition to these new features, as part of the feature engineering process, we compute the Euclidean geodesic distance of each trip--- a compromise over the driving distance considering the large dataset size and the limited capability of the local machine (my laptop). 

We train XGBRegressor and LGBMRegressor on 20M of the 55M training samples (to fit them into my 16GB RAM). After tuning the hyperparameters, we attained our best test RMSE of 2.96150 with LGBMRegressor. The test prediction follows closely the distribution of the training set fares (with outliers discarded), with expected small peaks that can be attributed to the flat rate between Manhattan and the airports.

**4. IMDB Movie Review Sentiment Classification**

A Kaggle competition about classifying the sentiment of IMDB movie reviews into good or bad. 

We first perform text cleanup and pre-processing. We make use of various techniques in natural language processing: CountVectorizer and TfidfVectorizer from nltk, LSTM with GloVe embedding vectors, and a transformer network from Hugging Face, and compare the results and performance of different approaches. 

All these approaches result in similar AUC validation and test socres, with the transformer network giving the best score (0.92304). The similarity in scores suggests that the order of the words is only marginally important. Given the significantly higher computational costs and slight gain in performance of the neural netowrk models, TfidfVectorizer turns out to be the most cost-effective option.

**5. Disaster Tweets Classification**

A Kaggle competition about classifying tweets to whether they are referring to a disaster. 

Informed by an exploratory data analysis, we perform a text pre-processing, and extract and engineer features from the texts. Once again, we make use of various techniques in natural language processing: CountVectorizer and TfidfVectorizer from nltk, LSTM with GloVe embedding vectors, and a transformer network from Hugging Face, and compare the results and performance of different approaches. In particular, an optimal LSTM-based neural network is found using Keras Tuner. Except for the transformer netowrk, in all our approaches, we use both the text vectors and other extracted features in training the models and making predictions.

All these approaches result in similar F1 socres, with the transformer network giving the best score (0.80784). Due to the small dataset size, no model has a significantly longer training time. This however contributes to a lower-than-desired performance and overfitting in the LSTM-based networks. 





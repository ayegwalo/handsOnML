## How would you define Machine Learning?
Machine Learning is part of AI that involves training algorithms to recognize patterns in data. 

## Can you name four types of applications where Machine Learning shines?
1. Image recognition (e.g., facial recognition)
2. Natural Language Processing (NLP) (e.g., language translation, sentiment analysis)
3. Predictive analytics (e.g., predicting customer behavior or stock prices)
4. Recommendation systems (e.g., Netflix or Amazon product recommendations)

## What is a labeled training set?
A labeled training set is a dataset where each input data point is paired with the correct output. In supervised learning, this helps the algorithm learn the relationship between input features and output labels to make predictions on new, unseen data.

## What are the two most common supervised tasks?
1. Classification (e.g., spam email detection, image classification)
2. Regression (e.g., predicting house prices, stock market predictions)

## Can you name four common unsupervised tasks?
1. Clustering (e.g., grouping similar articles)
2. Anomaly detection (e.g., fraud detection in transactions)
3. Dimensionality reduction (e.g., PCA to reduce feature space)
4. Association rule learning (e.g., market basket analysis to identify item associations)

## What type of algorithm would you use to allow a robot to walk in various unknown terrains?
Reinforcement learning algorithm will be used for this type of problem, as it involves an agent (the robot) learning to make decisions based on feedback from the environment (rewards or penalties) over time.

## What type of algorithm would you use to segment your customers into multiple groups?
A clustering algorithm, such as K-means or DBSCAN, is commonly used to segment customers into groups based on shared characteristics or behaviors.

## Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
Supervised learning, this is because we can train a model using a labeled dataset of emails, where each email is labeled as "spam" or "not spam."

## What is an online learning system?
An online learning system is a type of machine learning where a model learns incrementally, processing one data point or a small batch at a time.

## What is out-of-core learning?
Out-of-core learning refers to algorithms that can handle datasets too large to fit into memory, by processing the data in chunks or batches, and using techniques like disk-based storage to load data incrementally. 

## What type of algorithm relies on a similarity measure to make predictions?
It is the Instance-based learning algorithms, **such as K-nearest neighbors (KNN)**, rely on a similarity measure to classify or predict the output based on the most similar instances from the training data. (Example was done as test).

## What is the difference between a model parameter and a model hyperparameter?
1. The model parameters are learned by the algorithm from the training data, such as the weights in a linear regression model.
2. Hyperparameters are set before training the model, and they control aspects of the model’s training process, such as the learning rate or the number of clusters in a K-means algorithm.

## What do model-based algorithms search for? 
## What is the most common strategy they use to succeed? 
## How do they make predictions?
Model-based algorithms search for patterns that allow them to make predictions about new, unseen data. The most common strategy is optimization, where the algorithm adjusts model parameters to minimize a loss function. Predictions are made by applying the learned model to new inputs.

## Can you name four of the main challenges in Machine Learning?
1. Overfitting (when the model learns the noise in the training data instead of general patterns)
2. Underfitting (when the model is too simple to capture the underlying patterns)
3. Bias in data (if the data is not representative of the real-world scenario)
4. Insufficient data (not having enough examples to train the model effectively)

## If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
Overfitting. The model has learned the specifics of the training data too well, but it doesn’t generalize to new data.

TRY:
1. Regularization???
2. Cross-validation to evaluate the model's performance on unseen data
3. Increasing the amount of training data so that the model learn more on the patterns   

## What is a test set, and why would you want to use it?
A test set part of the dataset that is not used during training and is only used to test the model’s performance. It helps you see how well the model generalizes to new, unseen data.

## What is the purpose of a validation set?
A validation set is used when training the model to tune the model’s hyperparameters and to evaluate performance while adjusting model settings, without using the test set. It acts as a proxy for new, unseen data.

## What is the train-dev set, when do you need it, and how do you use it?
A train-dev set (development set) is a part of the training data used to fine-tune the model, typically when you have limited data. It allows you to test hyperparameters or intermediate versions of the model before final testing on the test set.

## What can go wrong if you tune hyperparameters using the test set?
If you use the test set to tune hyperparameters, the model can overfit to the test set, and its performance may be artificially inflated. **The test set should only be used for final evaluation after training and hyperparameter tuning.**


**Gradient Descent**
Gradient Descent helps when training a model to minimize the loss function, which measures how well the model's predictions match the actual values in the training data. It’s an important method for model-based algorithms, especially in supervised learning.


# My Machine Learning Answers
1. What’s the trade-off between bias and variance?
- Bias is error due to wrong or overly simplistic assumptions in the learning algorithm you’re using. This can lead to __the model underfitting your data__, __making it hard for it to have high predictive accuracy__ and __for you to generalize your knowledge from the training set to the test set.__
- Variance is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being __highly sensitive to high degrees of variation in your training data__, __which can lead your model to overfit the data.__ You’ll be carrying too much noise from your training data for your model to be very useful for your test data.
- Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that may fail to capture important regularities (i.e. underfit) in the data.
- The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. 
_Noisy data are data that is corrupted, distorted, or has a low signal-to-noise ratio._
2. What is the difference between supervised and unsupervised machine learning?     
As the name indicates, supervised learning involves machine learning algorithms that learn under the presence of a supervisor. 
When training a machine, supervised learning refers to a category of methods in which we teach or train a machine learning algorithm using data, while guiding the algorithm model with labels associated with the data. If the model gave a correct answer, then there is nothing for us to do. Our job is to correct the model when the output of the model is wrong. If this is the case, we need to make sure that the model makes necessary updates so that the next time a cat image is shown to the model, it can correctly identify the image. 
Supervised learning requires training labeled data. For example, in order to do classification (a supervised learning task), you’ll need to first label the data you’ll use to train the model to classify data into your labeled groups. 
Here, the machine learning model learns to fit mapping between examples of input features with their associated labels. When models are trained with these examples, we can use them to make new predictions on unseen data.
The predicted labels can be both numbers or categories. For instance, if we are predicting house prices, then the output is a number. In this case, the model is a regression model. If we are predicting if an email is spam or not, the output is a category and the model is a classification model. 
Unsupervised learning, in contrast, does not require labeling data explicitly.We use the data points as references to find meaningful structure and patterns in the observations. Unsupervised learning is commonly used for finding meaningful patterns and groupings inherent in data, extracting generative features, and exploratory purposes.
3. How is KNN different from k-means clustering?
K-Nearest Neighbors (K-NN) is a supervised algorithm used for classification.
k-Means is an unsupervised algorithm used for clustering. By unsupervised we mean that we don’t have any labeled data upfront to train the model. Hence the algorithm just relies on the dynamics of the independent features to make inferences on unseen data.
n order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.
The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t—and is thus unsupervised learning.
4. Explain how a ROC curve works.
The ROC curve (Receiver Operating Characteristic) is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds.
’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

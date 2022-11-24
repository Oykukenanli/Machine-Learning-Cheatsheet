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
5. Define precision and recall.
Recall is also known as the true positive rate: the amount of positives your model claims compared to the actual number of positives there are throughout the data.
Precision is also known as the positive predictive value, and it is a measure of the amount of accurate positives your model claims compared to the number of positives it actually claims. 
In pattern recognition, information retrieval, object detection and classification (machine learning), precision and recall are performance metrics that apply to data retrieved from a collection, corpus or sample space.
Consider a computer program for recognizing dogs (the relevant element) in a digital photograph. Upon processing a picture which contains ten cats and twelve dogs, the program identifies eight dogs. Of the eight elements identified as dogs, only five actually are dogs (true positives), while the other three are cats (false positives). Seven dogs were missed (false negatives), and seven cats were correctly excluded (true negatives). The program's precision is then 5/8 (true positives / selected elements) while its recall is 5/12 (true positives / relevant elements).
6. What is Bayes’ Theorem? How is it useful in a machine learning context?
Basically, Bayes' theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately (by conditioning it on their age) than simply assuming that the individual is typical of the population as a whole.
Mathematically, it’s expressed as the true positive rate of a condition sample divided by the sum of the false positive rate of the population and the true positive rate of a condition. Say you had a 60% chance of actually having the flu after a flu test, but out of people who had the flu, the test will be false 50% of the time, and the overall population only has a 5% chance of having the flu. Would you actually have a 60% chance of having the flu after having a positive test?
Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition Sample) / (.6*0.05)(True Positive Rate of a Condition Sample) + (.5*0.95) (False Positive Rate of a Population)  = 0.0594 or 5.94% chance of getting a flu.
Bayes’ Theorem is the basis behind a branch of machine learning that most notably includes the Naive Bayes classifier. 
*More reading: https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/*
7. Why is “Naive” Bayes naive?
Naïve Bayes machine learning algorithm is considered Naïve because the assumptions the algorithm makes are virtually impossible to find in real-life data. Conditional probability is calculated as a pure product of individual probabilities of components. This means that the algorithm assumes the presence or absence of a specific feature of a class is not related to the presence or absence of any other feature (absolute independence of features), given the class variable. For instance, a fruit may be considered to be a banana if it is yellow, long and about 5 inches in length.Lets take Naive Bayes on a intuition level to figure out what food tastes good together!
Naive Bayes will make the following assumption:
If you like Pickles, and you like Ice Cream, naive bayes will assume independence and give you a Pickle Ice Cream and think that you'll like it.
Which is may not be true at all.
However, if these features depend on each other or are based on the existence of other features, a naïve Bayes classifier will assume all these properties to contribute independently to the probability that this fruit is a banana. Assuming that all features in a given dataset are equally important and independent rarely exists in the real-world scenario.
8. Explain the difference between L1 and L2 regularization.
In supervised machine learning, models are trained on a subset of data aka training data.
Overfitting happens when model learns signal as well as noise in the training data and wouldn’t perform well on new data on which model wasn’t trained on.
Now, there are few ways you can avoid overfitting your model on training data like cross-validation sampling, reducing number of features, pruning, regularization etc.
Regularization basically adds the penalty as model complexity increases.
In order to create less complex model when you have a large number of features in your dataset, some of the Regularization techniques used to address over-fitting and feature selection are L1 and L2 regularizations.
L2 regularization tends to spread error among all the terms, while L1 is more binary/sparse, with many variables either being assigned a 1 or 0 in weighting. L1 corresponds to setting a Laplacean prior on the terms, while L2 corresponds to a Gaussian prior.
*More reading: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c*
9. What’s your favorite algorithm, and can you explain it to me in less than a minute?
**No answer for now**
10. What’s the difference between Type I and Type II error?
Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is.
A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.
11. What’s a Fourier transform?

 




# Spam-Email-Classification
   Analyzing the content of an Email dataset which contains above 5000 email sample with labled spam or not.We have built a model to classify given email Spam((junk email) or ham (good email) using Naive Bayes Classification algorithm with accuracy score of ~99 .
 #Naive Bayes Classifier Introduction
  Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features
 #Checking the distribution of data.
 
![with lier](https://user-images.githubusercontent.com/40944675/44002023-a89200e2-9e59-11e8-923b-a2c3b074c2ec.png)

we can see some extreme outliers, we'll set a threshold for length of text (here threshold is 10000, I have not applied this threshold in algotithm implementaion) and plot the histogram again

![with outlier](https://user-images.githubusercontent.com/40944675/44002036-dabc7430-9e59-11e8-8a0d-e950527fdbe6.png)

 Below are metrics about the results:
 
#Confusion Matrix
 
  ![image](https://user-images.githubusercontent.com/40944675/44002064-8bfe309e-9e5a-11e8-8eed-ec9cf8f33ef5.png)
  
  We achieved 98.836899942163114% accuracy(Mean) and 0.4% standard variance. We are in low bias and low variance region, below plot of the Learning curve.

![learning curve](https://user-images.githubusercontent.com/40944675/44002117-969be22a-9e5b-11e8-8a35-91b230fcc829.png)

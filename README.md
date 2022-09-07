	INDEX

Sr. No.
Topic
Page No.
1
 Introduction

	1.1. Problem Definition
 	1.2  Scope of Project
 	1.3 Users and their requirements
 	1.4  Technologies to be used


2
 Literature Review

    2.1 Explain Min 5 -10 papers relevant to mini project topic 
    2.2  Research Gap


3
Conceptual System Design
    
      3.1Conceptual system diagram
      3.2 Design/flowchart of each module/submodule  
 	and each module' s description


4
Implementation and evaluation

    4.1 Implementation Code ( well documented)
    4.2  Results and Evaluation ( Explain the  results)


5
Conclusion and Future scope


6
References














Introduction

1.1. Problem Definition:

The purpose of this analysis is to build a prediction model to predict whether a review on the restaurant is positive or negative and assist the restaurant owners in improving the negative reviews. People nowadays rely a lot on reviews from other people regarding the restaurants. Thus, it is essential to analyse various restaurant reviews and predict whether they are positive or negative. To do so, we will work on the Restaurant Review dataset, we will load it into predictive algorithms. To build a model to predict if review is positive or negative, following steps are performed.
Importing Dataset
Exploratory Analysis
Sentiment Analysis 
Topic Modelling 
Recommendation System

1.2  Scope of Project:

The objective of this project is to build a restaurant reviews engine based on the content. For this project, the data that we decided to use was taken from the Yelp dataset which is a subset of their businesses, reviews and user data which was made publicly available to be used in personal, educational and academic purposes. Since the data was too large, we decided to narrow down to restaurants in the city of Toronto because it had the most number of reviews as compared to the other 9 cities in the data set. After filtering out the data, we ended up with a total of 5,471 Businesses, 44,485 Users and 23,050 reviews. 
WIth the use of the text reviews which were given by users for every restaurant they visited and get a score that will be able to indicate whether the text was positive or negative using Textblob’s Polarity Score. Thus, the project will help in sentimental analysis of various restaurant reviews.Also we plan to make a recommendation model which gives recommendations to customers based on location that will aid them in decision making.


1.3 Users and their requirements

Users of the system :
Restaurant owners 
Customers looking for hotel recommendations
Functional Requirements : 
Predict reviews of restaurants - positive or negative
Assist restaurant owners in improving negative reviews 
Helping restaurant owners in advertisement using positive reviews

Non Functional Requirements : 
The reviews should be accurately predicted 
Data security should be ensured by keeping the data given by users safe 
System must be reliable for all types of users
Platform should be easy to use


1.4  Technologies to be used

Python 
Pandas ,Scikit-learn
Google Collab
NLTK library





















2. Literature Review

2.1 Literature review 

K. Kaviya et al. [1] , The rating systems are important to find the quality of the product or service. These rating systems serve as a guide in finding the perfect one based on different user criteria.Therefore, a sentiment analysis system has been designed for automatic restaurant rating which will be useful to the people in picking their favorite restaurant. The sentiment analysis for restaurant rating system rates the restaurant depending upon the reviews given by the users. The system breaks user comments to check for sentiment keywords . Sentiment analysis can also be extended further to improve the business process. With the customers’ reviews, one can understand the changes in the market and improve their product/service. It is also scalable to any type of environment. The rating system should be by the customers and for the customers. So the rating system is based on the customer’s reviews for a restaurant. Each review will be split and searched for sentiment keywords. The reviews will be classified into positive and negative and finally an overall rating will be provided to the restaurant. The sentiment words in the reviews are checked and the highlighting given to each emotion of the customer is taken into account. The sentiment score of the review is emphasized with extra points for every highlighted emotion with emoticons, adverbs, special case idioms etc.

M. GOVINDARAJAN, et al.[2] , The area of sentiment mining (also called sentiment extraction, opinion mining, opinion  extraction, sentiment analysis, etc.) has seen a large increase in academic interest in the last few  years. Researchers in the areas of natural language processing, data mining, machine learning,  and others have tested a variety of methods of automating the sentiment analysis process. In this  research work, a new hybrid classification method is proposed based on coupling classification  methods using arcing classifiers and their performances are analyzed in terms of accuracy. A  Classifier ensemble was designed using Naïve Bayes (NB), Support Vector Machine (SVM) and  Genetic Algorithm (GA). In the proposed work, a comparative study of the effectiveness of  ensemble technique is made for sentiment classification. The feasibility and the benefits of the  proposed approaches are demonstrated by means of restaurant review that is widely used in the  field of sentiment classification. A wide range of comparative experiments are conducted and  finally, some in-depth discussion is presented and conclusions are drawn about the effectiveness  of ensemble technique for sentiment classification.  

Boya Yu, et al.[3] , Many people use Yelp to find a good restaurant. Nonetheless, with only an overall  rating for each restaurant, Yelp offers not enough information for independently judging its  various aspects such as environment, service or flavor. In this paper, we introduced a machine  learning based method to characterize such aspects for particular types of restaurants. The  main approach used in this paper is to use a support vector machine (SVM) model to decipher  the sentiment tendency of each review from word frequency. Word scores generated from the  SVM models are further processed into a polarity index indicating the significance of each  word for special types of restaurant. Customers overall tend to express more sentiment  regarding service. The method was based on a high-accuracy SVM model, calculating word  scores and measuring the polarity. The essential features we discovered might not only help customers to choose their favorite cuisine, but also provide restaurants with their advantages  and shortages

Tri Doan et al.[4] , Sentiment analysis of customer reviews has a crucial impact on a business development  strategy. Despite the fact that a repository of reviews evolves over time, sentiment analysis often  relies on offline solutions where training data is collected before the model is built. If we want to  avoid retraining the entire model from time to time, incremental learning becomes the best  alternative solution for this task. In this work, we present a variant of online random forests to  perform sentiment analysis on customers’ reviews. Their model is able to achieve accuracy  similar to offline methods and comparable to other online models. They have used the Random  Forests approach, a popular ensemble method, in building our proposed incremental learning  model. They have proposed an incremental learning approach to train with high accuracy  compared to other state-of-the-art incremental methods and demonstrated that their solution  generates comparable results with offline models at each incremental size.


Devina Ekawati, et al.[5] , Aspect-based sentiment analysis summarizes what people like and dislike from reviews  of products or services. In this paper, they have adapted the first rank research at SemEval 2016  to improve the performance of aspect-based sentiment analysis for Indonesian restaurant reviews  and used six steps for aspect-based sentiment analysis i.e.: preprocess the reviews, aspect  extraction, aspect categorization, sentiment classification, opinion structure generation, and  rating calculation. They collected 992 sentences for experiment and 383 sentences for evaluation  and conducted experiments to find the best feature combination for aspect extraction, aspect  categorization, and sentiment classification. For improving the performance, additional  restaurant reviews can be added to build a distributional semantic model. Adding training data for aspect extraction, aspect categorization, and sentiment classification step can also improve  the performance of the models.  

Veenendaal, Anne, et al.[6] , This research focuses on two aspects of sentiment analysis on a restaurant review website. First aim is to estimate the polarity of the comments (positive review vs negative review). The second aim is to estimate the polarity of emotions of the reviewer between negative emotions (angry, frustrated, disappointed, dissatisfied and Positive emotions (happy and content). For carrying out this analysis they have collected 788 comments from various restaurant review sites. A panel of 2 annotators was created to manually annotate each comment as positive and negative review. Additionally, each comment was also annotated with positive and negative emotion. In case of conflicting annotation, a third annotator was used to resolve the conflict.The automatic classification of positive vs negative restaurant review comments showed more than 80% accuracy.

Zhang, Ziqiong, et al.[7] , This paper focuses on the restaurant reviews written in Cantonese as it is an important dialect spoken in and around the cities of southern China and typical areas with rapid development in China. In this paper,they have used standard machine learning techniques and have incorporated into the domain of online Cantonese-written restaurant reviews to automatically classify user reviews as thumbs-up or thumbs-down. Two popular text classification algorithms – naive Bayes and SVM, and six feature presentations concerning n-gram presence/frequency are chosen to examine the effects of the classifiers and the feature options on Cantonese sentiment classification. For analysis data is collected from consumer reviews on a Cantonese site OpenRice (URL: http://www.openrice.com).The highest accuracy achieved through naive bayes using Bigram is 95.67% and through Support Vector Machine by using Bigram_frequency is  94.83%.

Mullen, Tony et al.[8] , This paper introduces an approach to sentiment analysis which uses support vector machines (SVMs) to bring together diverse sources of potentially pertinent information, including several favorability measures for phrases and adjectives and, where available, knowledge of the topic of the text. This paper introduces an approach to classifying texts as positive or negative using Support Vector Machines (SVMs). The accuracy value represents the percentage of test texts which were classified correctly by the mode. They have used different methods such as Semantic orientation with PMI ,  Osgood semantic differentiation with WordNet , Topic proximity and syntactic-relation features and Support Vector Machines out of which Osgood values does not seem to yield improvement in any of the models. The Turney values appear to be more helpful.The average score over the four n-fold cross validation experiments for the hybrid SVM is 86.5%, whereas the average score for the second-best performing model, incoporating all semantic value features and lemmas, is 85%. The simple lemmas model obtains an average score of 84% and the simple unigrams model obtains 79.75%.

Krishna, Akshay, et al.[9] , This paper mainly focuses on the implementation of various classification algorithms and their performance analysis. They have used  dataset in Tab Spaced Values (TSV) format.They have generated confusion matrix that contains the amount of True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) values generated by the given dataset.The proposed algorithm was implemented using Python 3.6 . The simulation results showed that SVM classifier resulted in the highest accuracy of 94.56% for the given dataset.

Nakayama, Makoto, et al [10] ,In this paper they have examined Japanese restaurant reviews in English at Yelp.com and those in Japanese at Yelp.co.jp from a cross-cultural perspective. Using bilingual text mining software, they have demonstrated that Japanese customers have significantly different sentiment distribution patterns on four basic attributes of dining experience (food quality, service, ambiance, and price fairness) than Western customers.It is observed that Japanese diners place more negative sentiment emphasis on ambiance (3.4% vs. 0.3%), whereas Western diners place more negative sentiment emphasis on service (17.5% vs. 2.3%) and price fairness (9.5% vs. 1.7%) than Japanese diners.The results also show that Western customers have slightly more negative sentiment emphasis (33.1% vs. 27.7%) on food quality than Japanese customers

Ref No
Paper Title
Research Gap
1

[1] K. Kaviya, C. Roshini, V. Vaidhehi and J. Dhalia Sweetlin, “Sentiment Analysis for Restaurant
Rating”, IEEE International Conference on Smart Technologies and Management for Computing, Communication, Controls, Energy and Materials (ICSTM), Veltech Dr.RR & Dr.SR University, Chennai, T.N., India. 2 - 4 August 2017.


This paper does not provide support for reviews based on  multiple languages and also they could provide better results if the factors such as restaurant  locality , time of the year would have been considered  
2
M. GOVINDARAJAN, “SENTIMENT ANALYSIS OF RESTAURANT REVIEWS USING  HYBRID CLASSIFICATION METHOD”, International Journal of Soft Computing and Artificial  Intelligence, ISSN: 2321-404X, Volume-2, Issue-1, May-2014 
In this paper they have used 4 machine learning and deep learning techniques , naive bayes , Support vector machine , genetic algorithms and Proposed Hybrid Method for  determining whether the dataset in consideration provides better results in which technique . They could have used deep learning techniques to provide more accurate results  
3
Boya Yu, Jiaxu Zhou, Yi Zhang, Yunong Cao, “Identifying Restaurant Features via Sentiment  Analysis on Yelp Reviews" 
In This paper they have used SVM for review analysis , they could have used other Machine Learning techniques as well so as to see which technique provides better results , also they do not have multi language support. 
5
Devina Ekawati, Masayu Leylia Khodra, “Aspect-based Sentiment Analysis for Indonesian  Restaurant Reviews” 
In this paper there are some informal words which should be normalized to provide accurate results 
 
7
Zhang, Ziqiong, et al. "Sentiment classification of Internet restaurant reviews written in Cantonese." Expert Systems with Applications 38.6 (2011): 7674-7682.


5. Conclusion and Future Scope 

CONCLUSION

Implemented the following on Yelp dataset, restricted to Toronto city. 
Exploratory data analysis
Sentiment analysis 
Topic Modelling
Location Based recommendation system

FUTURE ENHANCEMENTS
Dashboard for restaurant and customers 
Explore collaborative and content based recommendation









6. References

[1] K. Kaviya, C. Roshini, V. Vaidhehi and J. Dhalia Sweetlin, “Sentiment Analysis for Restaurant
Rating”, IEEE International Conference on Smart Technologies and Management for Computing, Communication, Controls, Energy and Materials (ICSTM), Veltech Dr.RR & Dr.SR University, Chennai, T.N., India. 2 - 4 August 2017.

[2] M. GOVINDARAJAN, “SENTIMENT ANALYSIS OF RESTAURANT REVIEWS USING  HYBRID CLASSIFICATION METHOD”, International Journal of Soft Computing and Artificial  Intelligence, ISSN: 2321-404X, Volume-2, Issue-1, May-2014  

[3] Boya Yu, Jiaxu Zhou, Yi Zhang, Yunong Cao, “Identifying Restaurant Features via Sentiment  Analysis on Yelp Reviews" 

[4] Tri Doan and Jugal Kalita, “Sentiment Analysis of Restaurant Reviews on Yelp with  Incremental Learning”, 15th IEEE International Conference on Machine Learning and  Applications, 2016  

[5] Devina Ekawati, Masayu Leylia Khodra, “Aspect-based Sentiment Analysis for Indonesian  Restaurant Reviews”  

[6]Veenendaal, Anne, et al. "Polarity Analysis of Restaurant Review Comment Board." Computer Science and Emerging Research Journal 2 (2014).

[7]Zhang, Ziqiong, et al. "Sentiment classification of Internet restaurant reviews written in Cantonese." Expert Systems with Applications 38.6 (2011): 7674-7682.

[8].Mullen, Tony, and Nigel Collier. "Sentiment analysis using support vector machines with diverse information sources." Proceedings of the 2004 conference on empirical methods in natural language processing. 2004.

[9]Krishna, Akshay, et al. "Sentiment analysis of restaurant reviews using machine learning techniques." Emerging Research in Electronics, Computer Science and Technology. Springer, Singapore, 2019. 687-696.

[10] Nakayama, Makoto, and Yun Wan. "The cultural impact on social commerce: A sentiment analysis on Yelp ethnic restaurant reviews." Information & Management 56.2 (2019): 271-27

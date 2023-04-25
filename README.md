# Identifying Clickbait Headlines using Features Optimized through Camel Algorithm

The following repository contains code of the model developed to identify clickbait headlines. The experiment is done on the Webis-Clickbait Corpus 17. The dataset contains 19536 samples of Twitter posts, the linked content associated with posts and the labels. We have provided the 100 samples of data from corpus for quick result analysis.

 The features are extracted from (1) postText, targetTitle, targetParagraph (individually) ; (2) pairwise with mentioned field to depict the degree of relatedness of post and target content ; (3) fetaures combining mentioned three fields.
 
The ratio of class in corpus is imbalanced in nature. We have applied BorderLine-SMOTE to oversample the minority class.
 
 We have used Camel Algorithm (CA), metaheuristic and nature inspired algorithm to reduce the feature set. The total of 184 features are extracted. The CA algorithm will reduce the feature dimension and improve model performance.
 
 We have used six ML classification algorithm namely; Support Vector Machine(SVM), Random Forest(RF), Decision Tree(DT), Gradient Boosting(GB), Naive Bayes(NB), and Logistic Regression(LR) to check model performance and the best performing algorithm is taken.
 
 Our result, as compared to previous other works on same dataset has shown good results.
 We have done the experiment on both original and balanced dataset and achieved the following result.
 
 1) Original dataset - Accuracy : 83% (Gradient Boosting) ; Precision : 82% ; Recall : 83% ; F1 Score : 82%
 
 2) Balanced dataset - Accuracy : 88% (Random Forest) ; Precision : 88% ; Recall : 88% ; F1 Score : 88%
 
 3) Balnced dataset with optimized features - Accuracy : 89% (Random Forest) ; Precision : 89% ; Recall : 89% ; F1 Score : 89%
 
 
 Our proposed model achieved effective result on the competitive state-of-the-arts and we believed that our method will help in assisting further studies on clickbait identification.
 

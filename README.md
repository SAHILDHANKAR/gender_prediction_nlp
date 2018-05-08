# Gender_prediction_nlp
This Repository contains Program code for gender prediction using names and comparing the accuracy of Naive Bayes classifier and Linear SV classifier.
The data set consists of total 38,179 names which contains 14,000 Indian names for each gender and names from NLTK names corpus.
For NaiveBayes I have used nltk.NaiveBayesClassifier and for LinearSV classifier nltk.classify.SklearnClassifier(LinearSVC()).
For most of the cases LinearSVC has higher accuracy.

--------------------------------------------------------------------------------------------------------------------------------------
References :
* Males.txt : Dataset of ~14,000 Indian male names for NLP training and analysis. The names have been retrieved from public records.
              (https://gist.github.com/mbejda/7f86ca901fe41bc14a63)
* Femles.txt : Dataset of ~14,000 Indian female names for NLP training and analysis. The names have been retrieved from public records.
              (https://gist.github.com/mbejda/9b93c7545c9dd93060bd)

# -*- coding: utf-8 -*-
import nltk, random
from sklearn.svm import LinearSVC

class PredictGender(object):
    def __init__(self):
        names_sample = nltk.corpus.names

        with open('Males.txt','r',encoding='utf8') as f:
            lines = f.readlines()
            self.names = [(x.strip(),'male') for x in lines]

        with open('Females.txt','r',encoding='utf8') as f1:
            lines1 = f1.readlines()
            self.names = self.names + [(x.strip(),'female') for x in lines1]

        self.names = self.names + [(name.lower(), 'male') for name in names_sample.words('male.txt')] + [(name.lower(), 'female')
                                                                   for name in names_sample.words('female.txt')]

        for i in range(0,10):
            random.shuffle(self.names)

        namelength=len(self.names)
        self.feature_sets = [(PredictGender.gender_Prediction(name), gender) for name, gender in self.names]
        self.train_set = self.feature_sets[:round(0.75*namelength)]
        self.test_set = self.feature_sets[round(0.75*namelength):]
        self.classifiersv = nltk.classify.SklearnClassifier(LinearSVC())
        self.classifiersv = self.classifiersv.train(self.train_set)
        self.classifierNB = nltk.NaiveBayesClassifier.train(self.train_set)

    def gender_Prediction(word):
        name = word.lower()
        features = dict()
        features['first_letter'] = name[0]
        features['last_letter'] = name[-1]
        features['second_letter'] = name[1]
        features['second_last_letter'] = name[-2]
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count' + letter] = name.count(letter)
            features['has' + letter] = letter in name
        return features

    def check_gender(self, name):
        name = name.lower()
        print('Gender for ' + name + ' : ' + self.classifiersv.classify(PredictGender.gender_Prediction(name)) +  ' as predicted by LinearSV Classifier')
        print('Gender for ' + name + ' : ' + self.classifierNB.classify(PredictGender.gender_Prediction(name)) + ' as predicted by NaiveBayesClassifier')

    def check_accuracy_of_the_classifier(self):
        print('Accuracy of LinearSV classifier is : ', nltk.classify.accuracy(self.classifiersv, self.test_set) * 100)
        print('Accuracy of NaiveBayes Classifier is : ', nltk.classify.accuracy(self.classifierNB, self.test_set) * 100)


if __name__ == '__main__':
    app = PredictGender()
    app.check_gender(input("Give the name you want to predict gender:"))
    app.check_accuracy_of_the_classifier()
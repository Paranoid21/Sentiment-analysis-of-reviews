#
# запуск совешается через demo.py, там же в шапке есть небольшая инструкция
#
import joblib
import pymorphy2
import pandas as pd

class SentimentClassifier(object):
    def __init__(self):
        # здесь нужно указать путь к обученному классификатору (reviews_clf),
        # если он не лежит в одной папке с кодом визуализации
        self.pipe = joblib.load(open("reviews_clf", "rb"))
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            return self.pipe.predict(pd.Series([text]))[0], \
                   self.pipe.predict_proba(pd.Series([text]))[0].max()
        except:
            print("prediction error")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            return self.pipe.predict(pd.Series(list_of_texts)), \
                   self.pipe.predict_proba(pd.Series(list_of_texts))
        except:
            print('prediction error')
            return None

    def lemmatization(self, text):
        text = pd.DataFrame({'text': [text]}).iloc[0].replace('[^а-яА-ЯёЁa-zA-Z0-9 ]', '', regex=True)[0]
        morph = pymorphy2.MorphAnalyzer()
        tmp = ''
        for nov in text.split():
            tmp += morph.parse(nov)[0].normal_form + ' '
        return tmp[:-1]

    def get_prediction_message(self, text):
        text = self.lemmatization(text)
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]

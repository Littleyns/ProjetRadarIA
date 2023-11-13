import random

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from Data.RadarDataSet import RadarDataSet
from Evaluation.plots import PredictedStepPlot

from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur, R2Score


class PolynomialRegressor:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2)
        self.model = LinearRegression()
    def train(self, X_train, y_train):
        self.X_poly = self.poly.fit_transform(X_train)
        self.model.fit(self.X_poly, y_train)

    def evaluate(self, X_test, y_test):
        X_test_poly = self.poly.fit_transform(X_test)

        y_predicted = self.model.predict(X_test_poly)

        y_predicted[y_predicted <= 0.5] = 0 # Negliger les valeurs en dessoude 0.1

        MSEEvaluateur().evaluate(y_test, y_predicted)
        RMSEEvaluateur().evaluate(y_test, y_predicted)
        R2Score().evaluate(y_test, y_predicted)

        randomTestIndex = random.randint(0,len(y_predicted))
        PredictedStepPlot().evaluate(y_test[randomTestIndex], y_predicted[randomTestIndex])


    def serialize(self):
        pass
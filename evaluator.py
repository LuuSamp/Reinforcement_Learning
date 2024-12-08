from formats import CLASSIFIERS
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Evaluator:
    def __init__(self, X, y):
        self.current_classifier = None
        self.current_params = None
        self.classifier_name = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def set_classifier(self, classifier_name, params):
        self.current_classifier = CLASSIFIERS[classifier_name](**params)
        self.classifier_name = classifier_name
        self.current_params = params

    def fit_classifier(self):
        self.current_classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.current_classifier.predict(self.X_test)
        y_true = self.y_test
        
        precision = precision_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {"precision": precision, "accuracy": accuracy, "recall": recall, "f1": f1}
    
    def respond(self):
        self.fit_classifier()
        response = self.evaluate()
        response["classifier"] = self.classifier_name
        response["params"] = self.current_params
        return response
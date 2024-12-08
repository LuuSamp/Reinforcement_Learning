from agent import Coder, Reviewer
from evaluator import Evaluator
from sklearn.datasets import load_wine
# Import the InvalidParameterError from the sklearn module
from sklearn.utils._param_validation import InvalidParameterError

class Environment:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.coder = Coder(X, y)
        self.evaluator = Evaluator(X, y)
        self.reviewer = Reviewer()
        self.history = {
            "model": [],
            "f1": [],
            "recall": [],
            "precision": [],
            "accuracy": []
        }
        self.model_history = []
        
    def run(self):
        self.coder.set_classifier()
        self.coder.set_params()
        
        try:
            self.evaluator.set_classifier(self.coder.classifier_name, self.coder.params)
            response = self.evaluator.respond()
            response["reviewer_feedback"] = self.reviewer.review(response).grade
            # response["reviewer_suggestion"] = self.reviewer.review(response).suggestion
            # Sugestão não implementada bem;
            self.coder.get_feedback(response)
        except:
            response = {"f1": "NaN", "recall": "NaN", "precision": "NaN", "accuracy": "NaN", "reviewer_feedback": 0, "reviewer_suggestion": "Parâmetros inválidos"} 
            self.coder.get_feedback(response)
            return response

        self.history["f1"].append(response["f1"])
        self.history["recall"].append(response["recall"])
        self.history["precision"].append(response["precision"])
        self.history["accuracy"].append(response["accuracy"])
        self.history["model"].append(self.coder.classifier_name)

        return response
    
    
if __name__ == "__main__":
    wine_df = load_wine()
    X = wine_df.data
    y = wine_df.target
    
    env = Environment(X, y)
    for i in range(10):
        response = env.run()
        print(response)
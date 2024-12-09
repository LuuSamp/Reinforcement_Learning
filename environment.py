from agent import Coder, Reviewer
from evaluator import Evaluator
from sklearn.datasets import load_wine
# Import the InvalidParameterError from the sklearn module
from sklearn.utils._param_validation import InvalidParameterError
from time import time
from tqdm import tqdm

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
            "accuracy": [],
            "reward": []
        }
        self.model_history = []
    
    def step(self):        
        self.coder.act()
        try:
            self.evaluator.set_classifier(self.coder.classifier_name, self.coder.params)
            start_time = time()
            response = self.evaluator.respond()
            response["time"] = time() - start_time
            response["reviewer_feedback"] = self.reviewer.review(response).grade
            response["reviewer_suggestion"] = self.reviewer.review(response).suggestion
            response["reviewer_elaborate"] = self.reviewer.review(response).elaborate
            self.coder.get_feedback(response)
            self.coder.get_reward(response["reviewer_feedback"] - 0.01*response["time"])
        except:
            response = {"f1": "NaN", "recall": "NaN", "precision": "NaN", "accuracy": "NaN", "reviewer_feedback": 0, "reviewer_suggestion": "Parâmetros inválidos"} 
            self.coder.get_feedback(response)
            self.coder.get_reward(-1)
            return response

        self.history["f1"].append(response["f1"])
        self.history["recall"].append(response["recall"])
        self.history["precision"].append(response["precision"])
        self.history["accuracy"].append(response["accuracy"])
        self.history["model"].append(self.coder.classifier_name)
        self.history["reward"].append(response["reviewer_feedback"])
        
        return response
    
    def run(self, epochs = 10):
        for i in tqdm(range(epochs)):
            response = self.step()
            reward = response["reviewer_feedback"]
            self.coder.get_reward(reward)
        return self.history
    
if __name__ == "__main__":
    wine_df = load_wine()
    X = wine_df.data
    y = wine_df.target
    
    env = Environment(X, y)
    for i in range(10):
        response = env.run()
        print(response)
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
            "reward": [],
            "params": []
        }
        self.model_history = []
    
    def step(self):        
        self.coder.act()
        try:
            self.evaluator.set_classifier(self.coder.classifier_name, self.coder.params)
            start_time = time()
            response = self.evaluator.respond()
            training_time = time() - start_time
            response["training_time"] = training_time
            reviewer_response = self.reviewer.review(response)
            response["reviewer_feedback"] = reviewer_response.grade
            response["reviewer_suggestion"] = reviewer_response.suggestion
            response["reviewer_elaborate"] = reviewer_response.elaborate
            self.coder.get_feedback(response)
            reward = response["reviewer_feedback"] - 0.01*training_time
            self.coder.get_reward(reward)
        except:
            response = {"f1": "NaN", "recall": "NaN", "precision": "NaN", "accuracy": "NaN", 
                        "reviewer_feedback": 0, 
                        "reviewer_suggestion": "Parâmetros inválidos", 
                        "reviewer_elaborate": "Parâmetros inválidos! OCORREU UM ERRO NA EXECUÇÃO!"} 
            self.coder.get_feedback(response)
            reward = -1
            self.coder.get_reward(reward)
            return reward

        self.history["f1"].append(response["f1"])
        self.history["recall"].append(response["recall"])
        self.history["precision"].append(response["precision"])
        self.history["accuracy"].append(response["accuracy"])
        self.history["model"].append(self.coder.classifier_name)
        self.history["params"].append(self.coder.params)
        
        return reward
    
    def run(self, epochs = 10):
        for i in tqdm(range(epochs)):
            reward = self.step()
            self.history["reward"].append(reward)
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
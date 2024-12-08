from ollama import chat
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from pprint import pprint

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ClassifierChoices(BaseModel):
    algorithm: Literal["LogisticRegression", "SVM", "RandomForest", "DecisionTree", "KNN"]

class LogisticRegressionConfig(BaseModel):
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    C: float = 1.0
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"

class SVMConfig(BaseModel):
    C: float = 1.0
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    degree: int = 3

class RandomForestConfig(BaseModel):
    criterion: Literal["gini", "entropy"] = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["auto", "sqrt", "log2"] = "auto"

class DecisionTreeConfig(BaseModel):
    criterion: Literal["gini", "entropy"] = "gini"
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[int] = None

class KNNConfig(BaseModel):
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"

classifiers = {
    "SVM": SVC,
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "KNN": KNeighborsClassifier
}

classifiers_configs = {
    "SVM": SVMConfig,
    "LogisticRegression": LogisticRegressionConfig,
    "RandomForest": RandomForestConfig,
    "DecisionTree": DecisionTreeConfig,
    "KNN": KNNConfig
}

class Agent:
    def __init__(self, model="llama3.1", temperature=1.0):
        self.model = model
        self.history = []
        self.temperature = temperature
    
    def chat(self, message):
        response = chat(
            model=self.model,
            messages=self.history + [message],
            temperature=self.temperature
        )
        self.history.append({
            "role": "user",
            "content": message
        })
        self.history.append(response["message"])
        return response["message"]["content"]
    
class Coder(Agent):
    def __init__(self, df, model="llama3.1", temperature=1):
        super().__init__(model, temperature)
        self.classifier = classifiers["RandomForest"]

    
        


response = chat(
  model='llama3.2',
  messages=[{"role": "user", "content": "Generste a random forest model with the following parameters: n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1"},
    {
      'role': 'user',
      'content': f'''
                Escolha dos seguintes classificadores:
                - LogisticRegression
                - SVM
                - RandomForest
                - DecisionTree
                ''',
    }
    ],
    format=ClassifierChoices.model_json_schema(),
    options={"temperature": 1.0}
)

print(response["message"])
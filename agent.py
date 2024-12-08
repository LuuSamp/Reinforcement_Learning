from ollama import chat
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from pprint import pprint

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

class ClassifierChoices(BaseModel):
    algorithm: Literal["LogisticRegression", "SVM", "RandomForest", "DecisionTree", "KNN"]

class NormalizationChoices(BaseModel):
    normalization: Literal["minmax", "standard", "robust", "quantile", "power", "maxabs"]

class LogisticRegressionParams(BaseModel):
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    C: float = 1.0
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"

class SVMParams(BaseModel):
    C: float = 1.0
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    degree: int = 3

class RandomForestParams(BaseModel):
    criterion: Literal["gini", "entropy"] = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["auto", "sqrt", "log2"] = "auto"

class DecisionTreeParams(BaseModel):
    criterion: Literal["gini", "entropy"] = "gini"
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[int] = None

class KNNParams(BaseModel):
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"

CLASSIFIERS = {
"SVM": SVC,
"LogisticRegression": LogisticRegression,
"RandomForest": RandomForestClassifier,
"DecisionTree": DecisionTreeClassifier,
"KNN": KNeighborsClassifier
}
CLASSIFIERS_PARAMS = {
    "SVM": SVMParams,
    "LogisticRegression": LogisticRegressionParams,
    "RandomForest": RandomForestParams,
    "DecisionTree": DecisionTreeParams,
    "KNN": KNNParams
}

class Agent:
    def __init__(self, model="llama3.2", temperature=1.0):
        self.model = model
        self.temperature = temperature
        self.history = []
        self.system_instructions = ""

    def chat(self, message, format="plain"):
        self.history.append({
            "role": "user",
            "content": f"{self.system_instructions} {message}"
        })
        response = chat(
            model=self.model,
            messages=self.history,
            format=format,
            options={"temperature": self.temperature}
        )
        self.history.append(response["message"])
        if len(self.history) > 10:
            self.history.pop(0)
        return response["message"]["content"]

class Coder(Agent):
    def __init__(self, X, y, model="llama3.2", temperature=1.2):
        super().__init__(model, temperature)
        self.X = X
        self.y = y
        self.X_copy = X.copy()
        self.classifier_name = "RandomForest"
        self.classifier = CLASSIFIERS[self.classifier_name]
        self.params = CLASSIFIERS_PARAMS[self.classifier_name]().model_dump()

    def set_classifier(self):
        prompt = f"""Com base nos resultados anteriores, escolha um dos seguintes classificadores:
                        {' '.join(CLASSIFIERS.keys())}
                        Classificador anterior: {self.classifier_name}
                        Se as métricas não forem boas, você pode escolher outro classificador.

                        F1 Score: 0.5
                        Acurácia: 0.7
                        Precisão: 0.6
                        Recall: 0.4
                  """
        response = self.chat(prompt, format=ClassifierChoices.model_json_schema())
        response = ClassifierChoices.model_validate_json(response)
        
        self.classifier_name = response.algorithm
        self.classifier = CLASSIFIERS[self.classifier_name]
        self.params = CLASSIFIERS_PARAMS[self.classifier_name]().model_dump()

        print(response)
        print(self.params)

        
    def set_params(self):
        prompt = f"""Com base nos seguintes resultados, escolha um dos seguintes classificadores:
                     Classificador anterior: {self.classifier}
                     ...
                  """
        
        response = self.chat(prompt, format=CLASSIFIERS_PARAMS[self.classifier_name].model_json_schema())
        response = CLASSIFIERS_PARAMS[self.classifier_name].model_validate_json(response)

        self.params = response.model_dump()
        
        print(self.params)

    def set_normalization(self):
        prompt = "Escolha o tipo de normalização a ser aplicada:"
        return self.chat(prompt, format=NormalizationChoices.model_json_schema())


wine_df = load_wine()
X = wine_df.data
y = wine_df.target

coder = Coder(X, y)
coder.set_classifier()
coder.set_params()
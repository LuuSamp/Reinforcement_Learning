from typing import Literal, Optional
from pydantic import BaseModel

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
    max_features: Literal["sqrt", "log2", None] = None

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

class ReviewerResponses(BaseModel):
    suggestion: str
    grade: int
    
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


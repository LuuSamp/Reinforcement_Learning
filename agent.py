from ollama import chat
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from pprint import pprint

class RandomForestParams(BaseModel):
    n_estimators: int
    criterion: str
    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int

class ClassifierChoices(BaseModel):
    algorithm: Literal["LogisticRegression", "SVM", "RandomForest", "DecisionTree", "KNN"]

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
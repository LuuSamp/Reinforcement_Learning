from pprint import pprint
from typing import List, Optional, Literal

from ollama import chat

from formats import *

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from random import random
from strategy import EpsilonGreedyDecay

class Agent:
    def __init__(self, model="llama3.2", temperature=1.0, strategy = EpsilonGreedyDecay()):
        self.model = model
        self.temperature = temperature
        self.history = [
            {
                "role": "user",
                "content": "Estamos buscando pelo melhor classificador e parâmetros para classificar um único dataset. Considere o histórico de tentativas a seguir para tomar uma decisão:"
            }
        ]
        self.system_instructions = ""
        self.max_history = 3*5
        self.strategy = strategy

    def chat(self, message, format="plain"):
        self.history.append({
            "role": "user",
            "content": f"""
                        {self.system_instructions}
                             
                        {message}
                        """
        })
        response = chat(
            model=self.model,
            messages=self.history,
            format=format,
            options={"temperature": self.temperature}
        )
        self.history.append(response["message"])
        if len(self.history) > self.max_history + 1:
            self.history.pop(1)
        return response["message"]["content"]

class Coder(Agent):
    def __init__(self, X, y, model="llama3.2", temperature=1.2, strategy = EpsilonGreedyDecay()):
        super().__init__(model, temperature, strategy)
        self.X = X
        self.y = y
        self.X_copy = X.copy()
        self.classifier_name = "RandomForest"
        self.classifier = CLASSIFIERS[self.classifier_name]
        self.params = CLASSIFIERS_PARAMS[self.classifier_name]().model_dump()
        self.latest_feedback = ""
        self.epsilon = strategy.epsilon
        self.actions = {
            "set_classifier": self.set_classifier,
            "set_params": self.set_params,
        }
        self.q_values = {action: 0 for action in self.actions.keys()}
        self.last_action : Literal["set_classifier", "set_params"] = "set_classifier"

    def set_classifier(self):
        
        if random() < self.epsilon:
            prompt = f"""
                        Com base nos resultados anteriores, escolha um dos seguintes classificadores:
                        {' '.join(CLASSIFIERS.keys())}
                        ESCOLHA UM CLASSIFICADOR NOVO OU POUCO USADO PARA EXPLORAR NOVAS POSSIBILIDADES.
                    """
                    
        else:          
            prompt = f""" Com base nos resultados anteriores, escolha um dos seguintes classificadores:
                        {' '.join(CLASSIFIERS.keys())}
                        Se as métricas de avaliação não foram satisfatórias, escolha um classificador diferente.
                    """
        
        response = self.chat(prompt, format=ClassifierChoices.model_json_schema())
        response = ClassifierChoices.model_validate_json(response)
        
        self.classifier_name = response.algorithm
        self.classifier = CLASSIFIERS[self.classifier_name]
        self.params = CLASSIFIERS_PARAMS[self.classifier_name]().model_dump()
        
        # print(response)
        # print(self.params)

        
    def set_params(self):
        if random() < self.epsilon:
            prompt = f"""
                            Com base nos resultados anteriores, escolha os seguintes parâmetros para o classificador {self.classifier_name}:
                            {' '.join(CLASSIFIERS_PARAMS[self.classifier_name].model_json_schema().keys())}
                            ESCOLHA PARÂMETROS POUCO USADOS PARA EXPLORAR NOVAS POSSIBILIDADES.
                    """
                    
        else:          
            prompt = f""" Com base nos resultados anteriores, escolha os seguintes parâmetros para o classificador {self.classifier_name}:
                            {' '.join(CLASSIFIERS_PARAMS[self.classifier_name].model_json_schema().keys())}
                            Se as métricas de avaliação não foram satisfatórias, escolha parâmetros diferentes.
                    """
        
        response = self.chat(prompt, format=CLASSIFIERS_PARAMS[self.classifier_name].model_json_schema())
        response = CLASSIFIERS_PARAMS[self.classifier_name].model_validate_json(response)

        self.params = response.model_dump()
        
        #print(self.params)

    def set_normalization(self):
        prompt = "Escolha o tipo de normalização a ser aplicada:"
        return self.chat(prompt, format=NormalizationChoices.model_json_schema())
    
    def fit(self, X, y):
        self.classifier = self.classifier(**self.params)
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
    
    def get_feedback(self, feedback):
        feedback_text = f"""
                        Dados do último teste:
                        - Classificador: {self.classifier_name}
                        - Parâmetros: {self.params}
                        - Nota do revisor: {feedback["reviewer_feedback"]}
                        - Sugestão do revisor: {feedback["reviewer_suggestion"]}
                        - Elaboração do revisor: {feedback["reviewer_elaborate"]}
                        """
        self.latest_feedback = feedback_text
        feedback = {
            "role": "user",
            "content": f"""
                        {self.system_instructions}
                           
                        {feedback_text}
                        """
        }
        if len(self.history) > self.max_history+1:
            self.history.pop(1)
        self.history.append(feedback)
    
    def act(self):
        self.last_action = self.strategy.get_action(self.q_values)
        self.epsilon = self.strategy.epsilon
        self.actions[self.last_action]()
    
    def get_reward(self, reward):
        self.q_values = self.strategy.update_q_values(self.q_values, self.last_action, reward, 0.1)
        
        return self.q_values
    
class Reviewer(Agent):
    def __init__(self, model="llama3.2", temperature=1.2):
        super().__init__(model, temperature)
        self.system_instructions = ""

    def review(self, feedback):
        
        prompt = f"""
                Considere as tentativas anteriores. Tentando classificar a mesma base de dados com o classificador {feedback["classifier"]} e os parâmetros {feedback["params"]} obteve-se os seguintes resultados:
                F1: {feedback["f1"]}
                Recall: {feedback["recall"]}
                Precisão: {feedback["precision"]}
                Acurácia: {feedback["accuracy"]}
                Tempo de treinamento: {feedback["training_time"]}
                
                Dê uma nota de 0 a 10 para esse classificador. A NOTA PRECISA SER UM INTEIRO DE 0 A 10
                Com base nos resultados obtidos, o que você sugere que seja alterado? Escolha entre as seguintes opções:
                - Alterar o classificador
                - Alterar os parâmetros
                Elabore sua resposta
                """
        response = self.chat(prompt, format=ReviewerResponses.model_json_schema())
        response = ReviewerResponses.model_validate_json(response)
        
        return response

if __name__ == "__main__":
    wine_df = load_wine()
    X = wine_df.data
    y = wine_df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    coder = Coder(X_train, y_train)
    coder.set_classifier()
    coder.set_params()

    coder.fit(X_train, y_train)
    y_pred = coder.predict(X_test)
    print(y_pred)
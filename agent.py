import os
import google.generativeai as genai
from dotenv import load_dotenv
import yaml
import json
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


class Agent:
    def __init__(self, config, history):
        self.config = config
        self.history = history

        self.model = genai.GenerativeModel(
            model_name = self.config["MODEL"],
            generation_config = self.config["GENERATION_CONFIG"],
            system_instruction = self.config["SYSTEM_INSTRUCTION"],
            safety_settings = self.config["SAFETY_CONFIG"]
        )

        self.chat = self.model.start_chat(history=self.history)

    def generate(self, prompt):
        response = self.chat.send_message(prompt).text
        self.history = self.chat.history
        
        return response
    
    def save_history(self, history_path):
        history = []
        for entry in self.chat.history:
            history.append({
                "role": entry.role,
                "parts": [part.text for part in entry.parts]
            })

        with open(Path(history_path), "w") as file:
            json.dump(history, file, indent=4)


class Coder(Agent):
    def __init__(self, config, history):
        super().__init__(config, history)


class Reviewer(Agent):
    def __init__(self, config, history):
        super().__init__(config, history)
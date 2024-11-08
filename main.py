import agent
from utils import load_config, load_history

coder_config = load_config("conf/coder.yaml")
coder_history = load_history("chat/coder_history.json")

coder = agent.Coder(coder_config, coder_history)

response = coder.generate("Hello")

print(response)
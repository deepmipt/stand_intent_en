import numpy as np
import json
import copy
import traceback

from intent_model.preprocessing import NLTKTokenizer
from intent_model.multiclass import KerasMulticlassModel

class IntentAgent:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.kpi_name = self.config['kpi_name']
        self.agent = None
        self.classes = None
        self.preprocessor = None
        self.answers = None

    def init_agent(self):
        agent_config = self.config['kpis'][self.kpi_name]['settings_agent']
        agent_config['model_from_saved'] = True
        self.agent = KerasMulticlassModel(agent_config)
        self.classes = self.agent.classes
        self.preprocessor = NLTKTokenizer()

    def _run_score(self, observation):
        task = observation[0]
        infer_result = self.agent.infer(self.preprocessor.infer(task))
        prediction = self.classes[np.argmax(infer_result)]
        self.answers = prediction

    def answer(self, input_task):
        try:
            if isinstance(input_task, list):
                print("%s human input mode..." % self.kpi_name)
                self._run_score(input_task)
                result = copy.deepcopy(self.answers)
                print("%s action result:  %s" % (self.kpi_name, result))
                return result
            elif isinstance(input_task, int):
                result = 'There is no Intent Classifier testing API provided'
                return result
            else:
                return {"ERROR": "{} parameter error - {} belongs to unknown type".format(self.kpi_name,
                                                                                          str(input_task))}
        except Exception as e:
            return {"ERROR": "{}".format(traceback.extract_stack())}






def init_model():
    config_file = 'intent_config.json'

    with open(config_file, "r") as f:
        opt = json.load(f)

    agent = IntentAgent(opt)

    print(agent.answer(['Give me two beers!']))


def main():
    config_file = 'intent_config.json'

    with open(config_file, "r") as f:
        opt = json.load(f)

    agent = IntentAgent(opt)
    agent.init_agent()

    print(agent.answer(['Give me two beers!']))

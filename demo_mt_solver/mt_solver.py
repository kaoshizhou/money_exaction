import json
import os
import torch
from nlp_solver.base_solver import BaseSolver
from .executors.demo_model_executor import DemoModelExecutor
from .executors.data_preprocess import DataPreprocessor
import cn2an
import re


class MtSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config) # super init必须有
        self.executor = DemoModelExecutor(self.config)

    def train(self, train_x, train_y, valid_x, valid_y):
        """
        各Executor的拼接，实现task的求解全流程
        """
        data_processor = DataPreprocessor()
        train_x = data_processor.process(train_x)
        valid_x = data_processor.process(valid_x)
        train_y = list(map(float, train_y))
        valid_y = list(map(float, valid_y))
        self.executor.train(train_x, train_y, valid_x, valid_y)
    
    def load(self):
        self.executor.model = torch.load(os.path.join(self.config['save_load_path'], 'model.pt'))
    
    def evaluate(self, x, y):
        data_processor = DataPreprocessor()
        x = data_processor.process(x)
        y = list(map(float, y))
        prediction = self.executor.predict(x)
        return {'labels': y, 'predictions': prediction}

    def predict(self, x):
        data_processor = DataPreprocessor()
        x = self.text_preprocess(x)
        x = [x]
        x = data_processor.process(x)
        prediction = self.executor.predict(x)
        return prediction

    def text_preprocess(self, justice):
        res = ''
        justice = justice.replace('\n', '')
        justice = re.findall(r'[^，。、：！？；,:;!?]*', justice)
        for item in justice:
            item = item.replace('O元', '0元')
            item = item.replace('多', '')
            item = item.replace('余', '')
            
            if len(re.findall(r'([0-9十百千万拾佰仟萬零一二三四五六七八九壹贰叁肆伍陆柒捌玖点\.]+)元', item)) > 0:
                original = re.findall(r'([0-9十百千万拾佰仟萬零一二三四五六七八九壹贰叁肆伍陆柒捌玖点\.]+)元', item)
                for num in original:
                    try:
                        converted = cn2an.cn2an(num, 'smart')
                        if converted == int(converted):
                            converted = int(converted)
                        item = item.replace(num, str(converted))
                    except:
                        pass
                res += item + ' '
        return res

    def get_train_time(self) -> dict:
        return {
            "modeling": 0,
            "total": 0
        }

# def test():
#     with open("ss_one.json", "r", encoding="utf-8") as f:
#         config = json.load(f)
#         config = {config[0]["hp_subspace"]: config[0]["hp_values"]}
#     print(config)
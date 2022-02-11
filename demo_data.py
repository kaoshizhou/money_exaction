import pandas as pd

from nlp_solver.base_data import BaseDataProcessor

class DemoDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        super().__init__(config) # super init必须有
        pass # 额外的init操作如有可加
    
    def load_train_valid_set(self):
        pass
    
    def load_test_data(self):
        pass
    
    def load(self):
        pass

    def get_predicted_label_raw(self, eval_result) -> list:
        pass

    def generate_eval_df(self, eval_x, eval_y_true, eval_result) -> pd.DataFrame:
        pass

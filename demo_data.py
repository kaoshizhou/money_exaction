from nlp_solver.base_data import BaseDataProcessor

class DemoDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        pass
    
    def load_train_valid_set(self):
        pass
    
    def load_test_data(self):
        pass
    
    def load(self):
        pass

    def get_predicted_label_raw(self, eval_result):
        pass

    def generate_eval_df(self, eval_x, eval_y_true, eval_result):
        pass

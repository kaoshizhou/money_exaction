
from nlp_solver.base_solver import BaseSolver
from .executors.demo_data_aug_executor import DemoDataAugExecutor
from .executors.demo_model_executor import DemoModelExecutor

class DemoMtSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config) # super init必须有
        pass # 额外的init操作如有可加

    def train(self, train_x, train_y, valid_x, valid_y):
        """
        各Executor的拼接，实现task的求解全流程
        """
        pass
    
    def load(self):
        pass
    
    def evaluate(self, x, y):
        pass

    def predict(self, x):
        pass
    
    def get_train_time(self) -> dict:
        return {
            "modeling": 0,
            "total": 0
        }
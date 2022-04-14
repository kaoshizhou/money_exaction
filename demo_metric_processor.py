from nlp_solver.base_metric_processor import BaseMetricProcessor
from typing import Dict, Any

class DemoMetricProcessor(BaseMetricProcessor):
    def __init__(self, metric_list):
        super().__init__(metric_list) # super init必须有
        self.metrics_list = metric_list

    def get_main_metric_name(self) -> str:
        return 'ExactMatch'
    
    def gen_metrics(self, eval_result) -> Dict[str, Any]:
        metrics = {}
        for metrics_name, metrics_class in self.metrics_dict.items():
            metrics[metrics_name] = metrics_class(eval_result['labels'], eval_result['predictions'])
        return metrics
            


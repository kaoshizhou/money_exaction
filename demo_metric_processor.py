from nlp_solver.base_metric_processor import BaseMetricProcessor

class DemoMetricProcessor(BaseMetricProcessor):
    def __init__(self, metric_list):
        super().__init__(metric_list)

    def get_main_metric_name(self) -> str:
        pass
    
    def gen_metrics(self, eval_result) -> Dict[str, Any]:
        pass


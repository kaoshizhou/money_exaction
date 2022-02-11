from nlp_solver.base_metric_processor import BaseMetricProcessor

class DemoMetricProcessor(BaseMetricProcessor):
    def __init__(self, metric_list):
        super().__init__(metric_list) # super init必须有
        pass # 额外的init操作如有可加

    def get_main_metric_name(self) -> str:
        pass
    
    def gen_metrics(self, eval_result) -> Dict[str, Any]:
        pass


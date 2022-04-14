import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re
import cn2an
from nlp_solver.base_data_processor import BaseDataProcessor

class MoneyDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        super().__init__(config) # super init必须有
        self.train_file = 'train.txt'
        self.test_file = 'test.txt'
    
    def load_train_valid_set(self):
        train_x = []
        valid_x = []
        train_y = []
        valid_y = []

        # file = os.path.join(self.data_path, self.train_file) 
        data = open(self.data_path, 'r', encoding='utf-8').read().rstrip('\n').split('\n')
        train_data, valid_data = train_test_split(data, test_size=0.2)
        for item in train_data:
            item = eval(item)
            train_x.append(self.text_preprocess(item['justice']))
            train_y.append(item['money'])
        for item in valid_data:
            item = eval(item)
            valid_x.append(self.text_preprocess(item['justice']))
            valid_y.append(item['money'])
        
        return train_x, valid_x, train_y, valid_y

    def load_test_data(self):
        test_x = []
        test_y = []

        # file = os.path.join(self.data_path, self.test_file) 
        test_data = open(self.data_path, 'r', encoding='utf-8').read().rstrip('\n').split('\n')
        for item in test_data:
            item = eval(item)
            test_x.append(self.text_preprocess(item['justice']))
            test_y.append(item['money'])
        
        return test_x, test_y
    
    def load(self):
        pass

    def get_predicted_label_raw(self, eval_result) -> list:
        return eval_result

    def generate_eval_df(self, eval_x, eval_y_true, eval_result) -> pd.DataFrame:
        df = pd.DataFrame()
        df['text'] = eval_x
        df['Labels'] = eval_y_true
        df['Labels_Pred'] = eval_result['predictions']
        return df

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
        
if __name__ == "__main__":
    config = {
        'data_path': 'data',
        'train': 'train.txt',
        'test': 'test.txt',
    }
    processor = MoneyDataProcessor(config=config)
    data = processor.load_test_data()
    print(data[0][3], data[1][3])
    
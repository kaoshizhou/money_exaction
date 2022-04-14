from transformers import BertTokenizer

class DataPreprocessor:
    def __init__(self):
        self.TOKENIZER_NAME = 'hfl/chinese-roberta-wwm-ext'
        self.BERT_NAME = 'hfl/chinese-roberta-wwm-ext'
        self.YUAN_INDEX = 1039
        self.tokenizer = BertTokenizer.from_pretrained(self.TOKENIZER_NAME)

    def process(self, data):
        res = []
        for justice in data:
            indices = self.tokenizer.encode(justice)
            num_index = self.find_num_index(indices)
            num_index.reverse()
            num_index_agg = self.aggregate(num_index)
            ids = [[indices[i] for i in item] for item in num_index_agg]
            tokens = [self.tokenizer.convert_ids_to_tokens(item) for item in ids]
            nums = [self.tokenizer.convert_tokens_to_string(token).replace(' ', '').replace('#', '') for token in tokens]
            try:
                nums = list(map(float, nums))
            except:
                pass
            res.append({
                'justice': justice,
                'num_index': num_index,
                'num_index_agg': num_index_agg,
                'nums': nums,
            })
        return res
    
    def aggregate(self, num_index):
        num_index_agg = []
        tmp = []
        for item in num_index:
            if len(tmp) == 0 or item == tmp[-1] + 1:
                tmp.append(item)
            else:
                num_index_agg.append(tmp)
                tmp = []
                tmp.append(item)
        num_index_agg.append(tmp)
        return num_index_agg
        
    def find_num_index(self, indices):
        flag = False
        num_index = []
        for i in range(len(indices) - 1, -1, -1):
            if flag:
                decode_str = self.tokenizer.decode([indices[i]])
                if decode_str.isdigit() or decode_str == '.' or \
                    len(decode_str) > 2 and decode_str[2:].isdigit():
                        num_index.append(i)
                else:
                    flag = False 
            if indices[i] == self.YUAN_INDEX:
                flag = True
        return num_index
import re
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import json
import cn2an
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from torch.optim.adam import Adam
from torch.nn.utils import clip_grad_norm_

TOKENIZER_NAME = 'hfl/chinese-roberta-wwm-ext'
BERT_NAME = 'hfl/chinese-roberta-wwm-ext'
YUAN_INDEX = 1039

class MoneyDataset(Dataset):
    def __init__(self, dataset_file, train=False) -> None:
        super().__init__()
        self.dataset_file = dataset_file
        self.data = open(self.dataset_file, 'r', encoding='utf-8').read().rstrip('\n').split('\n')
        self.train = train
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        item = json.loads(item)
        justice = item['justice']
        justice = self.text_preprocess(justice)
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
            print(index)
        if not self.train:
            return [
                justice,
                num_index,
                num_index_agg,
                nums,
            ]
        return [
            justice,
            num_index,
            num_index_agg,
            nums,
            float(item['money']),
        ]
    
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
            if indices[i] == YUAN_INDEX:
                flag = True
        return num_index
        
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

def collate_fn(batch_data):
    try:
        justice, num_index, num_index_agg, nums, money = zip(*batch_data)
        return {
            'justice': justice,
            'num_index': num_index,
            'num_index_agg': num_index_agg,
            'nums': nums,
            'money': money,
        }
    except:
        justice, num_index, num_index_agg, nums= zip(*batch_data)
        return {
            'justice': justice,
            'num_index': num_index,
            'num_index_agg': num_index_agg,
            'nums': nums,
        }
        
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        for p in self.parameters():
            p.requires_grad = False
        self.cls = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        hidden_state = self.bert(**inputs).last_hidden_state
        outputs = self.cls(hidden_state)
        outputs = self.softmax(outputs)
        return outputs


if __name__ == "__main__":
    batch_size = 2
    epochs = 10
    learning_rate = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Model().to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    training_set = MoneyDataset('./data/train.txt', train=True)
   
    test_set = MoneyDataset('./data/test.txt', train=False)
    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # for i in training_dataloader:
    #     print(i)
    #     exit()
    # exit()
    for epoch in range(epochs):
        model.train()
        for data in training_dataloader:
            optimizer.zero_grad()
            texts = data['justice']
            index_agg = data['num_index_agg']
            nums = data['nums']
            money = data['money']
            inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)
            probs = model(inputs)
            loss = 0

            for i in range(batch_size):
                prob = probs[i]
                predict_money = 0
                for j in range(len(index_agg[i])):
                    predict_money += torch.mean(prob[index_agg[i][j]], dim=0)[1] * nums[i][j]
                loss += ((predict_money - money[i]) / money[i]) ** 2
            loss = loss / batch_size
            loss.backward()
            # for item in model.cls.parameters():
            #     print(item.grad.data.norm())
            clip_grad_norm_(model.parameters(), max_norm=2.5, norm_type=2)
            print(loss.cpu().item())
            optimizer.step()
        
            
            

  

    

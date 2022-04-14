import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer

from .dataset import MoneyDataset
from .ops.model import Model


def collate_fn(batch_data):
    justice = [item['justice'] for item in batch_data]
    num_index = [item['num_index'] for item in batch_data]
    num_index_agg = [item['num_index_agg'] for item in batch_data]
    nums = [item['nums'] for item in batch_data]
    try:
        money = [item['money'] for item in batch_data]
    except:
        money = None

    return {
        'justice': justice,
        'num_index': num_index,
        'num_index_agg': num_index_agg,
        'nums': nums,
        'money': money,
    }

class DemoModelExecutor:
    def __init__(self, config):
        TOKENIZER_NAME = 'hfl/chinese-roberta-wwm-ext'
        if 'modeling' in config:
            self.model_config = config['modeling']['model']
        else:
            self.model_config = {}

        self.name = self.model_config.get('hp_name', None)
        self.batch_size = self.model_config.get('batch_size', 8)
        self.epochs = self.model_config.get('epochs', 10)
        self.lr = self.model_config.get('lr', 1e-4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_load_path = config['save_load_path']
        self.log_steps = 10
        if not os.path.exists(self.save_load_path):
            os.makedirs(self.save_load_path)

        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = Model()
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def train(self, train_x, train_y, valid_x, valid_y):
        """
        模型、loss等算子的组合，执行模型训练的流程
        """
        best_acc = 0
        self.model.to(self.device)

        training_set = MoneyDataset(train_x, train_y)
        valid_set = MoneyDataset(valid_x, valid_y)

        training_loader = DataLoader(
            training_set,
            batch_size = self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        for epoch in range(self.epochs):
            print('-------------train-------------')
            self.model.train()
            for step, data in enumerate(training_loader):
                self.optimizer.zero_grad()
                texts = data['justice']
                index_agg = data['num_index_agg']
                nums = data['nums']
                money = data['money']
                inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
                probs = self.model(inputs)

                loss = 0
                for i in range(len(texts)):
                    prob = probs[i]
                    predict_money = 0
                    for j in range(len(index_agg[i])):
                        predict_money += torch.mean(prob[index_agg[i][j]], dim=0)[1] * nums[i][j]
                    loss += ((predict_money - money[i]) / money[i]) ** 2
                loss = loss / self.batch_size
                loss.backward()
                if step % self.log_steps == 0:
                    print(f'epoch: {epoch}, step: {step}, loss = {loss.cpu().item()}')
                # for item in model.cls.parameters():
                #     print(item.grad.data.norm())
                clip_grad_norm_(self.model.parameters(), max_norm=2.5, norm_type=2)
                self.optimizer.step()

            print('-----------evaluation-----------')
            self.model.eval()
            res = []
            for step, data in enumerate(valid_loader):
                texts = data['justice']
                index_agg = data['num_index_agg']
                nums = data['nums']
                money = data['money']
                inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
                probs = self.model(inputs)

                for i in range(len(texts)):
                    prob = probs[i]
                    predict_money = 0
                    for j in range(len(index_agg[i])):
                        predict_money += torch.argmax(torch.mean(prob[index_agg[i][j]], dim=0)) * nums[i][j]
                    res.append(predict_money.cpu().item())

            acc = sum(np.array(res) == np.array(valid_y)) / len(valid_y)
            print(f'epoch: {epoch}, accuracy: {acc}')
            if acc > best_acc:
                best_acc = acc
                print(f'The model in epoch {epoch} has a better acc, save it in {self.save_load_path}!')
                torch.save(self.model, os.path.join(self.save_load_path, 'model.pt'))
    
    def predict(self, x):
        self.model.to(self.device)
        self.model.eval()

        test_set = MoneyDataset(x)
        test_loader = DataLoader(
            test_set,
            batch_size = self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        print('-----------start predict-----------')
        res = []
        for step, data in enumerate(test_loader):
            texts = data['justice']
            index_agg = data['num_index_agg']
            nums = data['nums']

            inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
            probs = self.model(inputs)

            for i in range(len(texts)):
                prob = probs[i]
                predict_money = 0
                for j in range(len(index_agg[i])):
                    predict_money += torch.argmax(torch.mean(prob[index_agg[i][j]], dim=0)) * nums[i][j]
                res.append(predict_money.cpu().item())
        return res

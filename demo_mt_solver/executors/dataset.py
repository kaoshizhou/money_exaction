from torch.utils.data import Dataset

class MoneyDataset(Dataset):
    def __init__(self, x, y=None):
        super().__init__()
        self.data = x
        if y:
            for i in range(len(self.data)):
                self.data[i]['money'] = float(y[i])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

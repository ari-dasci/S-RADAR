import torch 

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len * 2
    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len,:].permute(1,0), self.X[index+self.seq_len+1,:])
    
class TimeSeriesDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data, window_size, forecast_size, stride=1):
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.window_size - self.forecast_size) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        #Get item permuted so the shape of the tensor matches in TSFEDL
        return (self.data[i:i+self.window_size].permute(1,0), self.data[i+self.window_size:i+self.window_size+self.forecast_size])
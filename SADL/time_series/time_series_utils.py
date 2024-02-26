import torch 

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len * 2
    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len,:].permute(1,0), self.X[index+self.seq_len+1,:])
    
    
class TSFEDL_TopModule(torch.nn.Module):
    def __init__(self, in_features=103, out_features=103, npred=1):
        super(TSFEDL_TopModule, self).__init__()
        self.npred = npred
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=in_features, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=npred*out_features)
        )

    def forward(self, x):
        out = self.model(x)
        if len(out.shape)>2:
            out = out[:, -1, :]
        if self.npred > 1:
            # Reshape to (batch_size, npred, out_features)
            out = out.reshape(out.shape[0], self.npred, -1)


class TimeSeriesDatasetNuevo(torch.utils.data.Dataset):
    def __init__(self, data, window_size, forecast_size, stride=1):
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.window_size - self.forecast_size) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        return (self.data[i:i+self.window_size], self.data[i+self.window_size:i+self.window_size+self.forecast_size])
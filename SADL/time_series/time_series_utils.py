import torch 
import numpy as np 

class TimeSeriesProcessor:
    def __init__(self, window_size, step_size, future_prediction=False, n_pred=None):
        """
        Initializes the time series processor.
        
        :window_size: Size of the sliding window.
        :step_size: Number of steps between consecutive windows.
        :future_prediction: Indicates whether to generate windows with a future prediction horizon.
        :n_pred: Number of future predictions (required if future_prediction is True).
        """

        self.window_size = window_size
        self.step_size = step_size
        self.future_prediction = future_prediction
        self.n_pred = n_pred if future_prediction else None
    
    def create_windows(self, X, y, labels=None):
        """
        Generates sliding windows from the time series.
        
        :X: Input time series (feature matrix).
        :y: Labels or target values to predict.
        :labels: Additional labels (only used for future prediction).
        :return: Arrays of input windows, output windows, and labels (if applicable).
        """
        X_windows = []
        y_windows = []
        label_windows = [] if labels is not None else None
        
        for i in range(0, len(X) - self.window_size + 1, self.step_size):
            temp_x = X[i : i + self.window_size]
            if self.future_prediction:
                temp_y = X[i + self.window_size : i + self.window_size + self.n_pred]
                if len(temp_y) < self.n_pred:
                    break
                temp_label = labels[i + self.window_size : i + self.window_size + self.n_pred] if labels is not None else None
            else:
                temp_y = y[i : i + self.window_size]
                temp_label = None
            
            X_windows.append(temp_x)
            y_windows.append(temp_y)
            if label_windows is not None:
                label_windows.append(temp_label)
        
        return (np.array(X_windows), np.array(y_windows)) if label_windows is None else (np.array(X_windows), np.array(y_windows), np.array(label_windows))
    
    def process_train_test(self, X_train, y_train, X_test, y_test, l_test=None):
        """
        Generates sliding windows for training and testing sets.
        
        :X_train: Training input data.
        :y_train: Training labels.
        :X_test: Testing input data.
        :y_test: Testing labels.
        :l_test: Additional labels for the test set (if applicable).
        :return: Training and testing windows as arrays.
        """
        X_train_windows, y_train_windows = self.create_windows(X_train, y_train)
        
        if l_test is not None:
            X_test_windows, y_test_windows, l_test_windows = self.create_windows(X_test, y_test, l_test)
            return X_train_windows, y_train_windows, X_test_windows, y_test_windows, l_test_windows
        else:
            X_test_windows, y_test_windows = self.create_windows(X_test, y_test)
            return X_train_windows, y_train_windows, X_test_windows, y_test_windows


class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len * 2
    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len,:].permute(1,0), self.X[index+self.seq_len+1,:])
    
class TimeSeriesDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data, window_size, forecast_size, stride=1, permute_size=(0,1)): 
        #permute_size = (1,0) for rotation. e.g. OhShuLih
        self.permute_size = permute_size
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.window_size - self.forecast_size) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        #Get item permuted so the shape of the tensor matches in TSFEDL
        return (self.data[i:i+self.window_size].permute(self.permute_size), self.data[i+self.window_size:i+self.window_size+self.forecast_size])
    

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
        return out   

import os
import unittest
import torch
from SADL.time_series.algorithms import tsfedl
import SADL.time_series.time_series_datasets as dataset
from SADL.time_series.time_series_utils import TimeSeriesDatasetV2
import sklearn
from numpy.testing import assert_equal
import numpy as np
import pickle as pkl

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

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 32
        self.data_size = round(10000 / self.batch_size)

        if not os.path.exists(os.path.join(os.getcwd(),'test.pkl')):
            data, attack_types, classes = dataset.readKDDCup99Dataset()
            self.data = sklearn.preprocessing.StandardScaler().fit_transform(data)

            attack_types_dict = {}
            cont=0
            for att in attack_types:
                attack_types_dict[att]=cont
                cont+=1

            for i in range(len(classes)):
                classes[i] = attack_types_dict[classes[i]]

            classes[classes!=attack_types_dict["normal"]] = 1
            classes[classes==attack_types_dict["normal"]] = 0
            classes_test = classes[-500000:]

            self.normal_train = np.where(classes[:-500000]==0)[0][:10000]
            self.train_data = data[self.normal_train]

            with open('test.pkl','wb') as f:
                pkl.dump(self.train_data, f)
        else: 
            with open('test.pkl','rb') as f:
                self.train_data = pkl.load(f)
        
    
    def test_OhShuLi(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=4, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "ohshulih", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=20, out_features=126), "max_epochs": 1, "in_features":126}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    
    def test_LiOhShu(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "liohshu", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=10, out_features=126), "max_epochs": 1, "in_features":126}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_YiBoGao(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "yibogao", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=1, out_features=126), "max_epochs": 1, "in_features":126}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)
   
    
    
    def test_YaoQihang(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "yaoqihang", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=32, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_HtetMyetLynn(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "htetmyetlynn", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=80, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    def test_YildirimOzal(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=126, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "yildirimozal", "input_shape":(126,126), "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=32, out_features=126), "max_epochs": 1, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)
   
    
    def test_CaiWenjuan(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "caiwenjuan", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=67, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_ZhangJin(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "zhangjin", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=24, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_KongZhengmin(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "kongzhengmin", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=64, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_WeiXiaoyan(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "weixiaoyan", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=512, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_GaoJunLi(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "gaojunli", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=64, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_KhanZulfiqar(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "khanzulfiqar", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=10, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_ZhengZhenyu(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "zhengzhenyu", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=256, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_WangKejun(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "wangkejun", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=256, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_ChenChen(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=1000, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "chenchen", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=64, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_KimTaeYoung(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=1000, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "kimtaeyoung", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=64, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    def test_GenMinxing(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=1000, forecast_size=1, permute_size = (0,1))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "genminxing", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=80, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_FuJiangmeng(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "fujiangmeng", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=256, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_ShiHaotian(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "shihaotian", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=32, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_HuangMeiLing(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=252, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "huangmeiling", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=19, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)
    
    
    
    def test_SharPar(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "sharpar", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=16, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    def test_HongTan(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=400, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        kwargs = {"algorithm_": "hongtan", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=4, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)

    
    def test_DaiXiLi(self):
        train_dataset = TimeSeriesDatasetV2(torch.from_numpy(self.train_data).double(), window_size=1000, forecast_size=1, permute_size = (1,0))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        
        kwargs = {"algorithm_": "daixili", "loss": torch.nn.MSELoss(),"top_module": TSFEDL_TopModule(in_features=2048, out_features=126), "max_epochs": 1, "in_features":126, "label_parser": True}
        model1 = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model1.model = model1.model.double()

        model1.fit(train_loader)
        scores = model1.decision_function(train_loader)
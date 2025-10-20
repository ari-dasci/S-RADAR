
import unittest
import torch
from RADAR.time_series.algorithms import tsfedl
from RADAR.time_series.time_series_utils import TimeSeriesProcessor
import numpy as np
from TSFEDL.models_pytorch import (
    OhShuLih_Forecaster, YiboGao_Forecaster, LihOhShu_Forecaster, YaoQihang_Forecaster,
    HtetMyetLynn_Forecaster, YildirimOzal_Forecaster, CaiWenjuan_Forecaster, ZhangJin_Forecaster,
    KongZhengmin_Forecaster, WeiXiaoyan_Forecaster, GaoJunLi_Forecaster, KhanZulfiqar_Forecaster,
    ZhengZhenyu_Forecaster, WangKejun_Forecaster, ChenChen_Forecaster, KimTaeYoung_Forecaster,
    GenMinxing_Forecaster, FuJiangmeng_Forecaster, ShiHaotian_Forecaster, HuangMeiLing_Forecaster,
    HongTan_Forecaster, SharPar_Forecaster, DaiXiLi_Forecaster
)


def generate_data(n_train=4000, n_test=1000, n_features=5, contamination=0.1, random_state=42):
    """
    Generates synthetic data for anomaly detection testing.
    
    Return:
        X_train: (n_train, n_features)
        X_test:  (n_test, n_features)
        y_train: (n_train,)
        y_test:  (n_test,)
    """
    rng = np.random.RandomState(random_state)
    
    # Datos normales
    X_train = rng.normal(0, 1, (n_train, n_features))
    X_test = rng.normal(0, 1, (n_test, n_features))
    
    # Etiquetas (0 = normal)
    y_train = np.zeros(n_train, dtype=int)
    y_test = np.zeros(n_test, dtype=int)
    
    # Introducir anomalías en el conjunto de test
    n_anom_train = int(contamination * n_train)
    n_anom_test = int(contamination * n_test)
    
    idx_train = rng.choice(n_train, n_anom_train, replace=False)
    idx_test = rng.choice(n_test, n_anom_test, replace=False)
    
    # Añadir desviación a las anomalías
    X_train[idx_train] += rng.normal(5, 1, (n_anom_train, n_features))
    X_test[idx_test] += rng.normal(5, 1, (n_anom_test, n_features))
    
    # Etiquetas de anomalías
    y_train[idx_train] = 1
    y_test[idx_test] = 1
    
    return X_train, X_test, y_train, y_test

class TestTsfedl(unittest.TestCase):

    def setUp(self) -> None:
        self.n_train = 10000
        self.n_test = 000
        self.contamination = 0.1
        self.n_features = 5
        self.w_size = 1048
        self.n_pred = 1
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features= self.n_features,
            contamination=self.contamination, random_state=42
        )
        
        processor = TimeSeriesProcessor(window_size= self.w_size, step_size=1, future_prediction=False)
        X_train_windows, y_train_windows, self.X_test_windows, self.y_test_windows = processor.process_train_test(self.X_train, self.y_train, self.X_test, self.y_test)
        
        print("X_train shape:", X_train_windows.shape)
        print("y_train shape:", y_train_windows.shape)
        print("X_test shape:", self.X_test_windows.shape)
        print("y_test shape:", self.y_test_windows.shape)
        
        self.X_train_windows = torch.tensor(X_train_windows, dtype=torch.float32)
        self.y_train_windows = (torch.tensor(y_train_windows, dtype=torch.float32)).unsqueeze(-1)  # to (7977, 24, 1)
        
        
    
    def test_OhShuLi(self):
        top_module = OhShuLih_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "ohshulih", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
    
    
    def test_LiOhShu(self):
        top_module = LihOhShu_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        
        kwargs = {"algorithm_": "liohshu", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        

    
    def test_YiBoGao(self):
        top_module = YiboGao_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        
        kwargs = {"algorithm_": "yibogao", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        

   
    
    
    def test_YaoQihang(self):
        top_module = YaoQihang_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        
        kwargs = {"algorithm_": "yaoqihang",  "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        


    
    def test_HtetMyetLynn(self):
        top_module = HtetMyetLynn_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "htetmyetlynn",  "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        

    def test_YildirimOzal(self):
        top_module = YildirimOzal_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "yildirimozal", "input_shape":(126,126), "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
 
   
    
    def test_CaiWenjuan(self):
        top_module = CaiWenjuan_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "caiwenjuan", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        


    
    def test_ZhangJin(self):
        top_module = ZhangJin_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "zhangjin", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        

    
    def test_KongZhengmin(self):
        top_module = KongZhengmin_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "kongzhengmin", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
       
    def test_WeiXiaoyan(self):
        top_module = WeiXiaoyan_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "weixiaoyan", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
    
    def test_GaoJunLi(self):
        top_module = GaoJunLi_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "gaojunli", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
    
    def test_KhanZulfiqar(self):
        top_module = KhanZulfiqar_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "khanzulfiqar", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
    
    def test_ZhengZhenyu(self):
        top_module = ZhengZhenyu_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "zhengzhenyu", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        

    
    def test_WangKejun(self):
        top_module = WangKejun_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "wangkejun", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
    
    def test_ChenChen(self):
        top_module = ChenChen_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "chenchen", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
      

    def test_KimTaeYoung(self):
        top_module = KimTaeYoung_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "kimtaeyoung", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        

    def test_GenMinxing(self):
        top_module = GenMinxing_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "genminxing", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
          
    def test_FuJiangmeng(self):
        top_module = FuJiangmeng_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "fujiangmeng", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
        
    
    def test_ShiHaotian(self):
        
        top_module = ShiHaotian_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "shihaotian", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
        
    
    def test_HuangMeiLing(self):
        top_module = HuangMeiLing_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "huangmeiling", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
    
    
    def test_SharPar(self):
        top_module = SharPar_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "sharpar", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
       
       
    def test_HongTan(self):
        top_module = HongTan_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "hongtan", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
        
        
    def test_DaiXiLi(self):
        top_module = DaiXiLi_Forecaster(out_features=self.n_features,n_pred=self.n_pred)
        print("TopModule_in_features:",top_module.in_features)
        kwargs = {"algorithm_": "daixili", "loss": torch.nn.MSELoss(),"top_module": top_module, "max_epochs": 2, "in_features":self.w_size}
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows,self.y_train_windows)
        scores_pred = model.decision_function(self.X_test_windows)
  
    
if __name__ == '__main__':
    unittest.main()    
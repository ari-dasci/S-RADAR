
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
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Number of features
        contamination: Proportion of anomalies
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    rng = np.random.RandomState(random_state)
    
    # Generate normal data
    X_train = rng.normal(0, 1, (n_train, n_features))
    X_test = rng.normal(0, 1, (n_test, n_features))
    
    # Initialize labels (0 = normal)
    y_train = np.zeros(n_train, dtype=int)
    y_test = np.zeros(n_test, dtype=int)
    
    # Introduce anomalies in test set
    n_anom_train = int(contamination * n_train)
    n_anom_test = int(contamination * n_test)
    
    idx_train = rng.choice(n_train, n_anom_train, replace=False)
    idx_test = rng.choice(n_test, n_anom_test, replace=False)
    
    # Add deviation to anomalies
    X_train[idx_train] += rng.normal(5, 1, (n_anom_train, n_features))
    X_test[idx_test] += rng.normal(5, 1, (n_anom_test, n_features))
    
    # Set anomaly labels
    y_train[idx_train] = 1
    y_test[idx_test] = 1
    
    return X_train, X_test, y_train, y_test


class TestTsfedl(unittest.TestCase):
    """Test class for TSFEDL anomaly detection algorithms."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Test configuration
        self.n_train = 10000
        self.n_test = 2000    
        self.contamination = 0.1
        self.n_features = 20
        self.w_size = 1048
        self.n_pred = 1
        self.max_epochs = 2
        
        # Generate test data
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, 
            n_test=self.n_test, 
            n_features=self.n_features,
            contamination=self.contamination, 
            random_state=42
        )
        
        # Process data into time series windows
        processor = TimeSeriesProcessor(
            window_size=self.w_size, 
            step_size=1, 
            future_prediction=False
        )
        X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Convert to tensors
        self.X_train_windows = torch.tensor(X_train_windows, dtype=torch.float32)
        self.y_train_windows = torch.tensor(y_train_windows, dtype=torch.float32).unsqueeze(-1)
        self.X_test_windows = torch.tensor(X_test_windows, dtype=torch.float32)
        self.y_test_windows = torch.tensor(y_test_windows, dtype=torch.float32)
        
        # Define forecaster configurations
        self.forecaster_configs = {
            "ohshulih": {
                "class": OhShuLih_Forecaster,
                "algorithm": "ohshulih",
                "extra_kwargs": {}
            },
            "liohshu": {
                "class": LihOhShu_Forecaster,
                "algorithm": "liohshu",
                "extra_kwargs": {}
            },
            "yibogao": {
                "class": YiboGao_Forecaster,
                "algorithm": "yibogao",
                "extra_kwargs": {}
            },
            "yaoqihang": {
                "class": YaoQihang_Forecaster,
                "algorithm": "yaoqihang",
                "extra_kwargs": {}
            },
            "htetmyetlynn": {
                "class": HtetMyetLynn_Forecaster,
                "algorithm": "htetmyetlynn",
                "extra_kwargs": {}
            },
            "yildirimozal": {
                "class": YildirimOzal_Forecaster,
                "algorithm": "yildirimozal",
                "extra_kwargs": {"input_shape": (126, 126)}
            },
            "caiwenjuan": {
                "class": CaiWenjuan_Forecaster,
                "algorithm": "caiwenjuan",
                "extra_kwargs": {}
            },
            "zhangjin": {
                "class": ZhangJin_Forecaster,
                "algorithm": "zhangjin",
                "extra_kwargs": {}
            },
            "kongzhengmin": {
                "class": KongZhengmin_Forecaster,
                "algorithm": "kongzhengmin",
                "extra_kwargs": {}
            },
            "weixiaoyan": {
                "class": WeiXiaoyan_Forecaster,
                "algorithm": "weixiaoyan",
                "extra_kwargs": {}
            },
            "gaojunli": {
                "class": GaoJunLi_Forecaster,
                "algorithm": "gaojunli",
                "extra_kwargs": {}
            },
            "khanzulfiqar": {
                "class": KhanZulfiqar_Forecaster,
                "algorithm": "khanzulfiqar",
                "extra_kwargs": {}
            },
            "zhengzhenyu": {
                "class": ZhengZhenyu_Forecaster,
                "algorithm": "zhengzhenyu",
                "extra_kwargs": {}
            },
            "wangkejun": {
                "class": WangKejun_Forecaster,
                "algorithm": "wangkejun",
                "extra_kwargs": {}
            },
            "chenchen": {
                "class": ChenChen_Forecaster,
                "algorithm": "chenchen",
                "extra_kwargs": {}
            },
            "kimtaeyoung": {
                "class": KimTaeYoung_Forecaster,
                "algorithm": "kimtaeyoung",
                "extra_kwargs": {}
            },
            "genminxing": {
                "class": GenMinxing_Forecaster,
                "algorithm": "genminxing",
                "extra_kwargs": {}
            },
            "fujiangmeng": {
                "class": FuJiangmeng_Forecaster,
                "algorithm": "fujiangmeng",
                "extra_kwargs": {}
            },
            "shihaotian": {
                "class": ShiHaotian_Forecaster,
                "algorithm": "shihaotian",
                "extra_kwargs": {}
            },
            "huangmeiling": {
                "class": HuangMeiLing_Forecaster,
                "algorithm": "huangmeiling",
                "extra_kwargs": {}
            },
            "sharpar": {
                "class": SharPar_Forecaster,
                "algorithm": "sharpar",
                "extra_kwargs": {}
            },
            "hongtan": {
                "class": HongTan_Forecaster,
                "algorithm": "hongtan",
                "extra_kwargs": {}
            },
            "daixili": {
                "class": DaiXiLi_Forecaster,
                "algorithm": "daixili",
                "extra_kwargs": {}
            }
        }

    def _create_model(self, forecaster_name):
        """
        Create a TSFEDL model for the given forecaster.
        
        Args:
            forecaster_name: Name of the forecaster algorithm
            
        Returns:
            tuple: (model, scores_pred)
        """
        config = self.forecaster_configs[forecaster_name]
        
        # Create top module with special parameters if needed
        top_module_kwargs = {
            "out_features": self.n_features,
            "n_pred": self.n_pred
        }
        
        # Add special parameters for specific forecasters
        if forecaster_name == "zhengzhenyu":
            top_module_kwargs["in_features"] = 256
        
        top_module = config["class"](**top_module_kwargs)
        
        # Prepare kwargs for the main model
        kwargs = {
            "algorithm_": config["algorithm"],
            "loss": torch.nn.MSELoss(),
            "top_module": top_module,
            "max_epochs": self.max_epochs,
            "in_features": self.w_size,
            **config["extra_kwargs"]
        }
        
        # Create and train model
        model = tsfedl.TsfedlAnomalyDetection(**kwargs)
        model.fit(self.X_train_windows, self.y_train_windows)
        
        # Get predictions
        scores_pred = model.decision_function(self.X_test_windows)
        
        return model, scores_pred

    def _validate_model_output(self, scores_pred, forecaster_name):
        """
        Validate model output and perform basic assertions.
        
        Args:
            scores_pred: Prediction scores from the model
            forecaster_name: Name of the forecaster for error messages
        """
        # Check that scores are returned
        self.assertIsNotNone(scores_pred, f"{forecaster_name}: Scores should not be None")
        
        # Check that scores have the expected shape
        expected_shape = (self.X_test_windows.shape[0],)
        self.assertEqual(
            scores_pred.shape, 
            expected_shape, 
            f"{forecaster_name}: Expected scores shape {expected_shape}, got {scores_pred.shape}"
        )
        
        # Check that scores are numeric (not NaN or infinite)
        self.assertFalse(
            np.any(np.isnan(scores_pred)), 
            f"{forecaster_name}: Scores should not contain NaN values"
        )
        self.assertFalse(
            np.any(np.isinf(scores_pred)), 
            f"{forecaster_name}: Scores should not contain infinite values"
        )
        
        # Check that scores have reasonable range (not all zeros)
        self.assertGreater(
            np.std(scores_pred), 0, 
            f"{forecaster_name}: Scores should have some variance"
        )

    def test_OhShuLi(self):
        """Test OhShuLi forecaster."""
        model, scores_pred = self._create_model("ohshulih")
        self._validate_model_output(scores_pred, "ohshulih")
        
    def test_GaoJunLi(self):
        """Test GaoJunLi forecaster."""
        model, scores_pred = self._create_model("gaojunli")
        self._validate_model_output(scores_pred, "gaojunli")   
    
    def test_KongZhengmin(self):
        """Test KongZhengmin forecaster."""
        model, scores_pred = self._create_model("kongzhengmin")
        self._validate_model_output(scores_pred, "kongzhengmin") 
    
    def test_CaiWenjuan(self):
        """Test CaiWenjuan forecaster."""
        model, scores_pred = self._create_model("caiwenjuan")
        self._validate_model_output(scores_pred, "caiwenjuan")
    
    def test_WangKejun(self):
        """Test WangKejun forecaster."""
        model, scores_pred = self._create_model("wangkejun")
        self._validate_model_output(scores_pred, "wangkejun")
    
    def test_ZhengZhenyu(self):
        """Test ZhengZhenyu forecaster."""
        model, scores_pred = self._create_model("zhengzhenyu")
        self._validate_model_output(scores_pred, "zhengzhenyu")
    
    def test_KimTaeYoung(self):
        """Test KimTaeYoung forecaster."""
        model, scores_pred = self._create_model("kimtaeyoung")
        self._validate_model_output(scores_pred, "kimtaeyoung")
    
    def test_FuJiangmeng(self):
        """Test FuJiangmeng forecaster."""
        model, scores_pred = self._create_model("fujiangmeng")
        self._validate_model_output(scores_pred, "fujiangmeng")
    
    def test_ShiHaotian(self):
        """Test ShiHaotian forecaster."""
        model, scores_pred = self._create_model("shihaotian")
        self._validate_model_output(scores_pred, "shihaotian")

    def test_SharPar(self):
        """Test SharPar forecaster."""
        model, scores_pred = self._create_model("sharpar")
        self._validate_model_output(scores_pred, "sharpar")
    
    def test_HongTan(self):
        """Test HongTan forecaster."""
        model, scores_pred = self._create_model("hongtan")
        self._validate_model_output(scores_pred, "hongtan")

    def test_HtetMyetLynn(self):                              
        """Test HtetMyetLynn forecaster."""
        model, scores_pred = self._create_model("htetmyetlynn")
        self._validate_model_output(scores_pred, "htetmyetlynn")
                                                                    # OK
    def test_LiOhShu(self):
        """Test LiOhShu forecaster."""
        model, scores_pred = self._create_model("liohshu")
        self._validate_model_output(scores_pred, "liohshu")

    def test_YiBoGao(self):
        """Test YiBoGao forecaster."""
        model, scores_pred = self._create_model("yibogao")
        self._validate_model_output(scores_pred, "yibogao")

    def test_YaoQihang(self):
        """Test YaoQihang forecaster."""
        model, scores_pred = self._create_model("yaoqihang")
        self._validate_model_output(scores_pred, "yaoqihang")


    def test_YildirimOzal(self):
        """Test YildirimOzal forecaster."""
        model, scores_pred = self._create_model("yildirimozal")
        self._validate_model_output(scores_pred, "yildirimozal")


    def test_ZhangJin(self):
        """Test ZhangJin forecaster."""
        model, scores_pred = self._create_model("zhangjin")
        self._validate_model_output(scores_pred, "zhangjin")


    def test_WeiXiaoyan(self):
        """Test WeiXiaoyan forecaster."""
        model, scores_pred = self._create_model("weixiaoyan")
        self._validate_model_output(scores_pred, "weixiaoyan")

    
    def test_KhanZulfiqar(self):
        """Test KhanZulfiqar forecaster."""
        model, scores_pred = self._create_model("khanzulfiqar")
        self._validate_model_output(scores_pred, "khanzulfiqar")

    def test_ChenChen(self):
        """Test ChenChen forecaster."""
        model, scores_pred = self._create_model("chenchen")
        self._validate_model_output(scores_pred, "chenchen")


    def test_GenMinxing(self):
        """Test GenMinxing forecaster."""
        model, scores_pred = self._create_model("genminxing")
        self._validate_model_output(scores_pred, "genminxing")

    def test_HuangMeiLing(self):
        """Test HuangMeiLing forecaster."""
        model, scores_pred = self._create_model("huangmeiling")
        self._validate_model_output(scores_pred, "huangmeiling")


    def test_DaiXiLi(self):
        """Test DaiXiLi forecaster."""
        model, scores_pred = self._create_model("daixili")
        self._validate_model_output(scores_pred, "daixili")


if __name__ == '__main__':
    unittest.main()
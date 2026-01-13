<img src="logoRADAR.png" width="100">

# Robust Anomaly Detection And Recognition (RADAR)  

## 📌 Description  
**Robust Anomaly Detection And Recognition (RADAR)** is an **anomaly detection** platform designed to unify and integrate diverse approaches and libraries from the literature, along with innovative model variants. The goal is to provide a flexible and extensible framework that ranges from classical methods to advanced techniques based on Transformer architectures, also including support for Federated Learning in distributed scenarios. For more details, refer to the official documentation [here](https://s-radar.readthedocs.io/en/latest/).

Specifically, it includes:  
- **Classical methods for static data:** integration with [PyOD](https://pyod.readthedocs.io/) and [Scikit-learn](https://scikit-learn.org/).  
- **Time series and deep learning models:** integration with libraries such as [TSFE-DL](https://github.com/ari-dasci/S-TSFE-DL.git).  
- **Representative Transformer models:** Informer, Autoformer, and Vanilla Transformer (implemented within the `time_series/` folder).  
- **Federated anomaly detection:** integration with [flex-anomalies](https://github.com/FLEXible-FL/flex-anomalies.git), developed as part of the Flexible platform.  

---

## 📚 Integrated Libraries  

| **Library / Model** | **Brief Description** | **Citation** |
|----------------------|-----------------------|----------|
| **PyOD** | Collection of classical algorithms for anomaly detection on static data. | [PyOD](https://pyod.readthedocs.io/) |
| **Scikit-learn** | Traditional machine learning methods applied to anomaly detection. | [Scikit-learn](https://scikit-learn.org/) |
| **TSFE-DL** | Framework for anomaly detection in time series using deep learning. | [TSFE-DL](https://github.com/ari-dasci/S-TSFE-DL.git) |
| **Informer** | Transformer-based model optimized for long time series forecasting and anomaly detection. | [Informer](https://github.com/zhouhaoyi/Informer2020) |
| **Autoformer** | Transformer specialized in time series forecasting and pattern detection. | [Autoformer](https://github.com/thuml/Autoformer) |
| **Vanilla Transformer** | Base Transformer implementation applied to anomaly detection. | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **flex-anomalies** | Library for anomaly detection in Federated Learning environments, part of the Flexible platform. | [flex-anomalies](https://github.com/FLEXible-FL/flex-anomalies.git) |

---
## 📂 Repository Structure  

```bash
RADAR/
│── notebooks_test/              # Test notebooks and examples
│
│── RADAR/                        # Core library
│   ├── federated_data/          # Methods and utilities for Federated Learning
│   ├── static_data/             # Anomaly detection on static data
│   ├── time_series/             # Methods for time series and deep learning
│   ├── base_algorithm_module.py # Base class for anomaly detection algorithms
│   ├── base_preprocessing_module.py # Data preprocessing utilities
│   ├── base_utils_module.py     # General helper functions
│   ├── metrics_module.py        # Evaluation metrics for anomaly detection
│   ├── pos_process_module.py    # Post-processing of results
│   └── visualization_module.py  # Visualization tools
```
## 🧪 Examples and Utilities

The library provides a set of practical examples and utilities to make experimentation easier:

- **Test Notebooks:** The `notebooks_test/` folder contains a notebook for each library integration, showing how to use the algorithms with different types of data.  
- **Preprocessing Scripts:** Each `time_series/` and `static_data/` folder includes a `preprocessing/` folder with techniques tailored to the specific data type:  
  - `preprocessing/preprocessing_static.py` for static data  
  - `preprocessing/preprocessing_ts.py` for time series data  
- **Dataset Loaders:** The same folders provide a `datasets_uci.py` file with functions to load datasets from the UCI repository, making it easy to test and experiment with the library:  
  - `static_datasets_uci.py` for static data  
  - `time_series_datasets_uci.py` for time series data
- **Metrics:** `metrics_module.py` provides evaluation metrics for anomaly detection.  
- **Visualization:** `visualization_module.py` includes functions to plot results and anomalies.  
- **Post-processing:** `pos_process_module.py` computes anomaly labels from predictions.
    
Note that `federated_data/` does not contain its own preprocessing or dataset loader scripts, since it can work with either static or time series data. The example notebooks cover usage for both types.    
  
This setup allows users to explore the library’s functionality, apply the appropriate preprocessing, and test algorithms on real datasets efficiently.

## Installation
Clone the repository:
```bash
git clone https://github.com/ari-dasci/RADAR.git
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt 
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in your research work, please cite the paper: 

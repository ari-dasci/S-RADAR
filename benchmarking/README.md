# Experiment Results

## Description

The first table (`static_results`) contains the results of different anomaly detection algorithms applied to static data (datasets `arrhythmia` and `shuttle` for anomaly detection), reporting the execution time, overhead, and ROC-AUC score obtained by each method. The second table (`ts_results`) contains equivalent results for time series data (datasets `ai4i_2020_predictive_maintenance` and `metro_interstate_traffic`), where the performance metric is the MSE.

---

## Static Results

| dataset    | algorithm       |       time |   overhead_s |   roc_auc_scores |
|:-----------|:----------------|-----------:|-------------:|-----------------:|
| arrhythmia | clusterAnomaly  |     0.1508 |       0.0132 |           0.7510 |
| arrhythmia | isolationForest |     3.5734 |       0.6960 |           0.7020 |
| arrhythmia | pcaAnomaly      |     0.2424 |       0.1891 |           0.7673 |
| shuttle    | pcaAnomaly      |     0.1182 |       0.0364 |           0.7974 |
| shuttle    | isolationForest |     4.0236 |       0.1409 |           0.9234 |
| shuttle    | clusterAnomaly  |     0.1206 |       0.0385 |           0.9695 |
| arrhythmia | lunar           |     3.5603 |       0.3037 |           0.7551 |
| arrhythmia | deep_svdd       |     0.2585 |       0.0267 |           0.7306 |
| arrhythmia | vae             |     2.2848 |       0.0149 |           0.6816 |
| arrhythmia | auto_encoder    |     1.2538 |       0.0460 |           0.6980 |
| arrhythmia | ae1svm          |     2.8217 |       0.0008 |           0.6776 |
| arrhythmia | dif             |     2.3269 |       0.1373 |           0.7306 |
| shuttle    | auto_encoder    |    50.9182 |       0.0848 |           0.9822 |
| shuttle    | lunar           |     3.4520 |       0.3045 |           0.9914 |
| shuttle    | ae1svm          |    95.6622 |       0.2012 |           0.7722 |
| shuttle    | vae             |    92.9562 |       0.5099 |           0.7389 |
| shuttle    | deep_svdd       |     6.6622 |       0.1399 |           0.8015 |
| shuttle    | dif             |    35.0796 |       0.2901 |           0.9770 |
| arrhythmia | iforest         |     0.2161 |       0.0129 |           0.8327 |
| arrhythmia | feature_bagging |     2.3899 |       0.0101 |           0.7633 |
| arrhythmia | hbos            |     0.0783 |       0.0268 |           0.8286 |
| arrhythmia | lscp            |     2.1641 |       0.0621 |           0.7714 |
| arrhythmia | knn             |     0.1102 |       0.0215 |           0.7755 |
| arrhythmia | pca             |     0.2101 |       0.0614 |           0.7673 |
| arrhythmia | abod            |     0.1444 |       0.0840 |           0.7510 |
| arrhythmia | cblof           |     1.0458 |       0.0369 |           0.7714 |
| arrhythmia | inne            |     0.7820 |       0.0247 |           0.7449 |
| arrhythmia | lof             |     0.2260 |       0.0047 |           0.7510 |
| arrhythmia | ocsvm           |     0.0320 |       0.0023 |           0.7510 |
| arrhythmia | anogan          |     7.3068 |       0.0842 |           0.7265 |
| arrhythmia | gmm             |     0.4625 |       0.0768 |           0.6980 |
| arrhythmia | kde             |     0.0568 |       0.0006 |           0.6776 |
| arrhythmia | mcd             |     4.9880 |       0.1090 |           0.7020 |
| arrhythmia | alad            |     1.0119 |       0.0930 |           0.5714 |
| arrhythmia | lmdd            |     2.4822 |       0.0420 |           0.7673 |
| shuttle    | knn             |     2.6536 |       0.0620 |           0.9933 |
| shuttle    | inne            |     2.3664 |       0.0450 |           0.9881 |
| shuttle    | cblof           |     5.2582 |       0.0850 |           0.9744 |
| shuttle    | lof             |     1.0577 |       0.0380 |           0.9876 |
| shuttle    | lscp            |    76.9382 |       0.0620 |           0.9930 |
| shuttle    | gmm             |     0.3020 |       0.0440 |           0.9431 |
| shuttle    | kde             |    19.1167 |       0.0150 |           0.9183 |
| shuttle    | mcd             |     4.7613 |       0.0320 |           0.9125 |
| shuttle    | lmdd            |   193.5639 |       0.1110 |           0.7595 |
| shuttle    | ocsvm           |     8.6094 |       0.1060 |           0.9094 |
| shuttle    | iforest         |     0.3722 |       0.0640 |           0.8831 |
| shuttle    | pca             |     0.0142 |       0.1410 |           0.8389 |
| shuttle    | hbos            |     0.0074 |       0.0080 |           0.8819 |
| shuttle    | anogan          |    71.0577 |       0.0430 |           0.7783 |
| shuttle    | alad            |     1.2587 |       0.0180 |           0.5692 |
| shuttle    | feature_bagging |     9.9915 |       0.0290 |           0.7192 |
| shuttle    | sgdocsvm        |     0.0043 |       0.0005 |           0.8380 |
| arrhythmia | sgdocsvm        |     0.0012 |       0.0000 |           0.7959 |
| arrhythmia | sos             |     0.1189 |       0.0023 |           0.8653 |
| arrhythmia | loda            |     0.0279 |       0.0097 |           0.9020 |
| arrhythmia | sampling        |     0.0044 |       0.0008 |           0.8204 |
| arrhythmia | cof             |     0.0572 |       0.0034 |           0.8122 |
| arrhythmia | suod            |    11.2642 |       0.0514 |           0.8163 |
| arrhythmia | copod           |     0.0340 |       0.0003 |           0.7633 |
| arrhythmia | rod             | 34333.1411 |       0.2390 |           0.7592 |
| arrhythmia | qmcd            |     0.1323 |       0.0009 |           0.5102 |
| arrhythmia | ecod            |     0.0293 |       0.0013 |           0.7469 |
| arrhythmia | kpca            |     0.2820 |       0.0057 |           0.7673 |
| shuttle    | sampling        |     0.0054 |       0.0011 |           0.9038 |
| shuttle    | rod             |     7.6074 |       0.0610 |           0.8431 |
| shuttle    | ecod            |     0.0734 |       0.0057 |           0.8220 |
| shuttle    | suod            |    13.8796 |       0.0633 |           0.8992 |
| shuttle    | kpca            |    37.3730 |       0.2380 |           0.9957 |
| shuttle    | copod           |     2.0726 |       0.0118 |           0.8391 |
| shuttle    | qmcd            |     1.9291 |       0.0004 |           0.8923 |
| shuttle    | cd              |     0.2812 |       0.0025 |           0.6862 |
| shuttle    | cof             |    15.5749 |       0.0338 |           0.4723 |
| shuttle    | loda            |     0.1629 |       0.0042 |           0.2424 |

---

## Time Series Results

| dataset                          | algorithm    |       time |   overhead_s |      mse |
|:---------------------------------|:-------------|-----------:|-------------:|---------:|
| ai4i_2020_predictive_maintenance | deepCNN_LSTM |  7713.8140 |       0.9000 | 1.499180 |
| ai4i_2020_predictive_maintenance | autoencoder  |   167.9904 |       0.6540 | 0.048541 |
| metro_interstate_traffic         | deepCNN_LSTM | 23983.7190 |       0.1800 | 0.942770 |
| metro_interstate_traffic         | autoencoder  |   496.4363 |       0.1500 | 0.025643 |
| ai4i_2020_predictive_maintenance | autoformer   |  1459.8840 |       0.0020 | 0.492974 |
| ai4i_2020_predictive_maintenance | informer     |   564.3644 |       0.0011 | 0.688358 |
| ai4i_2020_predictive_maintenance | transformer  |   466.0987 |       0.0029 | 0.701193 |
| metro_interstate_traffic         | autoformer   |  7473.8090 |       0.0010 | 0.147994 |
| metro_interstate_traffic         | transformer  |  2228.0072 |       0.0012 | 0.190964 |
| metro_interstate_traffic         | informer     |  2966.8341 |       0.0020 | 0.388237 |
| ai4i_2020_predictive_maintenance | caiwenjuan   |   351.5468 |       0.2080 | 0.690050 |
| ai4i_2020_predictive_maintenance | chenchen     |   166.5366 |       0.0100 | 0.629120 |
| ai4i_2020_predictive_maintenance | daixili      |   173.7981 |       0.0690 | 0.190580 |
| ai4i_2020_predictive_maintenance | fujiangmeng  |   324.2245 |       0.0820 | 0.593021 |
| ai4i_2020_predictive_maintenance | gaojunli     |    41.3998 |       0.0370 | 1.107830 |
| ai4i_2020_predictive_maintenance | genminxing   |   102.5912 |       0.2040 | 0.698028 |
| ai4i_2020_predictive_maintenance | hongtan      |   109.1284 |       0.0580 | 0.608185 |
| ai4i_2020_predictive_maintenance | htetmyetlynn |    81.1718 |       0.0540 | 0.430190 |
| ai4i_2020_predictive_maintenance | huangmeiling |    44.6063 |       0.0470 | 0.462920 |
| ai4i_2020_predictive_maintenance | khanzulfiqar |    80.1991 |       0.1860 | 0.573203 |
| ai4i_2020_predictive_maintenance | kimtaeyoung  |   116.3542 |       0.0420 | 0.572610 |
| ai4i_2020_predictive_maintenance | kongzhengmin |    83.6074 |       0.1510 | 0.576867 |
| ai4i_2020_predictive_maintenance | liohshu      |    73.9665 |       0.0560 | 0.645401 |
| ai4i_2020_predictive_maintenance | ohshulih     |    64.5913 |       0.0080 | 0.561238 |
| ai4i_2020_predictive_maintenance | sharpar      |   213.9582 |       0.0520 | 0.668948 |
| ai4i_2020_predictive_maintenance | shihaotian   |    78.4371 |       0.1810 | 0.464256 |
| ai4i_2020_predictive_maintenance | wangkejun    |   149.6250 |       0.0010 | 0.515371 |
| ai4i_2020_predictive_maintenance | weixiaoyan   |   172.8207 |       0.0560 | 0.572412 |
| ai4i_2020_predictive_maintenance | yaoqihang    |   177.4100 |       0.1080 | 0.721024 |
| ai4i_2020_predictive_maintenance | yibogao      |   364.2123 |       0.0500 | 0.773022 |
| ai4i_2020_predictive_maintenance | zhangjin     |   296.9138 |       0.1510 | 0.817277 |
| ai4i_2020_predictive_maintenance | zhengzhenyu  |   139.2332 |       0.1360 | 0.502372 |
| metro_interstate_traffic         | caiwenjuan   |  1891.1578 |       0.0960 | 0.367380 |
| metro_interstate_traffic         | chenchen     |   570.8397 |       0.1890 | 0.295776 |
| metro_interstate_traffic         | daixili      |   930.9184 |       0.0150 | 0.060300 |
| metro_interstate_traffic         | fujiangmeng  |  1727.6211 |       0.1950 | 0.161284 |
| metro_interstate_traffic         | gaojunli     |   203.2601 |       0.0390 | 0.546025 |
| metro_interstate_traffic         | genminxing   |   531.4536 |       0.0240 | 0.233543 |
| metro_interstate_traffic         | hongtan      |   615.1907 |       0.1160 | 0.297070 |
| metro_interstate_traffic         | htetmyetlynn |   422.2766 |       0.0950 | 0.160920 |
| metro_interstate_traffic         | huangmeiling |   228.7803 |       0.1970 | 0.192797 |
| metro_interstate_traffic         | khanzulfiqar |   414.3139 |       0.0590 | 0.271839 |
| metro_interstate_traffic         | kimtaeyoung  |   563.3353 |       0.0690 | 0.184320 |
| metro_interstate_traffic         | kongzhengmin |   430.5199 |       0.0820 | 0.174240 |
| metro_interstate_traffic         | liohshu      |   415.0310 |       0.0700 | 0.293017 |
| metro_interstate_traffic         | ohshulih     |   463.0206 |       0.1600 | 0.246582 |
| metro_interstate_traffic         | sharpar      |   831.3130 |       0.1250 | 0.257878 |
| metro_interstate_traffic         | shihaotian   |   411.1225 |       0.0690 | 0.195278 |
| metro_interstate_traffic         | wangkejun    |   729.3453 |       0.0140 | 0.093711 |
| metro_interstate_traffic         | weixiaoyan   |   768.9910 |       0.0690 | 0.176456 |
| metro_interstate_traffic         | yaoqihang    |   838.3700 |       0.0490 | 0.223132 |
| metro_interstate_traffic         | yibogao      |  1923.5138 |       0.1440 | 0.380683 |
| metro_interstate_traffic         | zhangjin     |  1530.9367 |       0.0520 | 0.238386 |
| metro_interstate_traffic         | zhengzhenyu  |   749.9577 |       0.0140 | 0.091847 |

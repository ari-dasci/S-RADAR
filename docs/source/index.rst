.. RADAR documentation master file, created by
   sphinx-quickstart on Wed Oct 15 11:13:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RADAR 
=======

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

Welcome to the **RADAR** documentation!
--------------------------------------

**Robust Anomaly Detection And Recognition (RADAR)** is a unified platform for anomaly detection that integrates diverse approaches and libraries from the literature, alongside innovative model variants.  
RADAR aims to provide a flexible and extensible framework covering methods from classical statistical techniques to advanced Transformer-based architectures, including support for **Federated Learning** in distributed environments.

Features
--------

- Integration of classical and state-of-the-art anomaly detection methods.
- Transformer-based models for time series and high-dimensional data.
- Support for Federated Learning, enabling privacy-preserving distributed training.
- Extensible and modular design for adding custom models and evaluation strategies.

Supported Methods
-----------------

Specifically, RADAR includes:

- **Classical methods for static data:** integration with `PyOD <https://pyod.readthedocs.io/>`_ and `Scikit-learn <https://scikit-learn.org/>`_.  
- **Time series and deep learning models:** integration with libraries such as `TSFE-DL <https://github.com/ari-dasci/S-TSFE-DL.git>`_.  
- **Representative Transformer models:** Informer, Autoformer, and Vanilla Transformer (implemented within the ``time_series/`` folder).  
- **Federated anomaly detection:** integration with `flex-anomalies <https://github.com/FLEXible-FL/flex-anomalies.git>`_, developed as part of the Flexible platform.  

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - **Library / Model**
     - **Brief Description**
     - **Citation**
   * - PyOD
     - Collection of classical algorithms for anomaly detection on static data.
     - `PyOD <https://pyod.readthedocs.io/>`_
   * - Scikit-learn
     - Traditional machine learning methods applied to anomaly detection.
     - `Scikit-learn <https://scikit-learn.org/>`_
   * - TSFE-DL
     - Framework for anomaly detection in time series using deep learning.
     - `TSFE-DL <https://github.com/ari-dasci/S-TSFE-DL.git>`_
   * - Informer
     - Transformer-based model optimized for long time series forecasting and anomaly detection.
     - `Informer <https://github.com/zhouhaoyi/Informer2020>`_
   * - Autoformer
     - Transformer specialized in time series forecasting and pattern detection.
     - `Autoformer <https://github.com/thuml/Autoformer>`_
   * - Vanilla Transformer
     - Base Transformer implementation applied to anomaly detection.
     - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
   * - flex-anomalies
     - Library for anomaly detection in Federated Learning environments, part of the Flexible platform.
     - `flex-anomalies <https://github.com/FLEXible-FL/flex-anomalies.git>`_

Getting Started
---------------

To get started with RADAR, check the `modules` section below or explore the tutorials for step-by-step guidance.



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules
   CHANGES
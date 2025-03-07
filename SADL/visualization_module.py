"""
 Visualization Module 

"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class DataVisualization():
    def __init__(self, data, plot_technique='scatter', dim_reduction_technique=None, 
                 n_components=2, y_true=None, y_pred=None, color_map=None, 
                 point_size=5, opacity=0.8, heatmap_color="magma", **plot_kwargs):
        """
        A class for data visualization with dimensionality reduction options and customizable plots.

        Parameters:
        - data: np.array, input data
        - plot_technique: str, visualization technique ('scatter', 'line', 'hist', 'boxplot', 'heatmap', 'prediction_forecasting','prediction_anomaly')
        - dim_reduction_technique: str, dimensionality reduction method ('PCA', 't-SNE', 'UMAP', None)
        - n_components: int, number of components for dimensionality reduction
        - y_true: np.array, true values for prediction visualization
        - y_pred: np.array, predicted values for prediction visualization
        - color_map: str/list, color scheme or custom color list
        - point_size: int, size of points (only for scatter plots)
        - opacity: float, opacity of points (0 to 1)
        - heatmap_color: str, color scheme for the heatmap (e.g., 'viridis', 'plasma', 'cividis')
        - plot_kwargs: additional arguments for customizing plots
        """
        self.data = np.array(data)
        self.plot_technique = plot_technique
        self.dim_reduction_technique = dim_reduction_technique
        self.n_components = n_components
        self.y_true = np.array(y_true) if y_true is not None else None
        self.y_pred = np.array(y_pred) if y_pred is not None else None
        self.reduced_data = None
        self.color_map = color_map
        self.point_size = point_size
        self.opacity = opacity
        self.heatmap_color = heatmap_color  # Allows customization of the heatmap color scheme
        self.plot_kwargs = plot_kwargs  # Stores extra arguments for plot customization
        
        assert self.dim_reduction_technique in [None, 'PCA', 't-SNE', 'UMAP'], "Invalid dimensionality reduction technique."

    def fit(self):
        """Applies dimensionality reduction if needed."""
        if self.dim_reduction_technique is None:
            self.reduced_data = self.data
            return

        if self.dim_reduction_technique == 'PCA':
            reducer = PCA(n_components=self.n_components)
        elif self.dim_reduction_technique == 't-SNE':
            reducer = TSNE(n_components=self.n_components)
        elif self.dim_reduction_technique == 'UMAP':
            reducer = umap.UMAP(n_components=self.n_components)

        self.reduced_data = reducer.fit_transform(self.data)

    def transform(self):
        """Returns the transformed data."""
        return self.reduced_data

    def show(self):
        """Generates and displays the data visualization."""
        if self.plot_technique == 'prediction_forecasting':
            if self.y_true is None or self.y_pred is None:
                raise ValueError("Both 'y_true' and 'y_pred' are required for prediction visualization.")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(self.y_true)), y=self.y_true, 
                                     mode='lines', name="True Values"))
            fig.add_trace(go.Scatter(x=np.arange(len(self.y_pred)), y=self.y_pred, 
                                     mode='lines', name="Predictions", line=dict(dash='dash')))
            fig.update_layout(title="Comparison of True and Predicted Values",
                              xaxis_title="Index",
                              yaxis_title="Value",
                              width=800, height=500)
            fig.show()
            return
        
        
        if self.plot_technique == 'prediction_anomaly':
            if self.y_true is None or self.y_pred is None:
                raise ValueError("Both 'y_true' and 'y_pred' are required for prediction visualization.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.y_true)), 
                y=self.y_true, 
                mode='markers', 
                name="True Labels",
                marker=dict(color='blue', symbol='circle', size=10, opacity=0.7)
            ))
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.y_pred)), 
                y=self.y_pred, 
                mode='markers', 
                name="Predicted Labels",
                marker=dict(color='red', symbol='x', size=8, opacity=0.8)
            ))

            fig.update_layout(
                title="Comparison of True and Predicted Labels",
                xaxis_title="Index",
                yaxis_title="Label (0 or 1)",
                yaxis=dict(tickvals=[0, 1], ticktext=['0', '1']),  # Asegura que solo se muestren 0 y 1
                width=800, height=500
            )

            fig.show()
            return
        
        if self.reduced_data is None:
            raise ValueError("You must run 'fit()' before 'show()'.")

        # Color mapping if the user provides labels
        color = self.color_map if isinstance(self.color_map, (list, np.ndarray)) else None

        if self.plot_technique == 'scatter':
            fig = px.scatter(x=self.reduced_data[:, 0], y=self.reduced_data[:, 1], 
                             title='Data Visualization (Scatter)',
                             labels={'x': 'Component 1', 'y': 'Component 2'},
                             color=color,
                             opacity=self.opacity, size=[self.point_size] * len(self.reduced_data),
                             **self.plot_kwargs)
        
        elif self.plot_technique == 'line':
            fig = px.line(x=np.arange(len(self.reduced_data[:, 0])), y=self.reduced_data[:, 0],
                          title='Data Visualization (Line)',
                          labels={'x': 'Index', 'y': 'Value'}, **self.plot_kwargs)
        
        elif self.plot_technique == 'hist':
            fig = px.histogram(self.reduced_data.flatten(), nbins=30,
                               title='Data Histogram',
                               labels={'value': 'Value'}, **self.plot_kwargs)
        
        elif self.plot_technique == 'boxplot':
            fig = px.box(self.reduced_data, title='Box Plot',
                         labels={'value': 'Value'}, **self.plot_kwargs)
        
        elif self.plot_technique == 'heatmap':
            fig = go.Figure(data=go.Heatmap(z=self.reduced_data, colorscale=self.heatmap_color))
            fig.update_layout(title='Data Heatmap', width=800, height=500, **self.plot_kwargs)
        
        else:
            raise ValueError("Unrecognized plotting technique.")
        
        fig.update_layout(width=800, height=500)  # General size adjustment
        fig.show()
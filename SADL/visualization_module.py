"""
Visualization Module

"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class DataVisualization:
    def __init__(
        self,
        data,
        plot_technique="scatter",
        dim_reduction_technique=None,
        n_components=2,
        y_true=None,
        y_pred=None,
        color_map=None,
        point_size=5,
        opacity=0.8,
        heatmap_color="magma",
        subset_size_percent=0.2,
        **plot_kwargs
    ):
        """
        A class for data visualization with dimensionality reduction options and customizable plots.

        Parameters:
        - data: np.array, input data
        - plot_technique: str, visualization technique ('scatter', 'line', 'hist', 'boxplot', 'heatmap', 'prediction_forecasting','anomaly_labels',plot_anomaly)
        - dim_reduction_technique: str, dimensionality reduction method ('PCA', 't-SNE', 'UMAP', None)
        - n_components: int, number of components for dimensionality reduction
        - y_true: np.array, true values for prediction visualization
        - y_pred: np.array, predicted values for prediction visualization
        - color_map: str/list, color scheme or custom color list
        - point_size: int, size of points (only for scatter plots)
        - opacity: float, opacity of points (0 to 1)
        - heatmap_color: str, color scheme for the heatmap (e.g., 'viridis', 'plasma', 'cividis')
        - subset_size_percent, proportion of the original data set that is selected
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
        self.heatmap_color = (
            heatmap_color  # Allows customization of the heatmap color scheme
        )
        self.subset_size_percent = subset_size_percent
        self.plot_kwargs = plot_kwargs  # Stores extra arguments for plot customization

        assert self.dim_reduction_technique in [
            None,
            "PCA",
            "t-SNE",
            "UMAP",
        ], "Invalid dimensionality reduction technique."

    def fit(self):
        """Applies dimensionality reduction if needed."""
        if self.dim_reduction_technique is None:
            self.reduced_data = self.data
            return

        if self.dim_reduction_technique == "PCA":
            reducer = PCA(n_components=self.n_components)
        elif self.dim_reduction_technique == "t-SNE":
            reducer = TSNE(n_components=self.n_components)
        elif self.dim_reduction_technique == "UMAP":
            reducer = umap.UMAP(n_components=self.n_components)

        self.reduced_data = reducer.fit_transform(self.data)

    def transform(self):
        """Returns the transformed data."""
        return self.reduced_data

    def show(self):
        """Generates and displays the data visualization."""

        plot_methods = {
            "prediction_forecasting": self._plot_prediction_forecasting,
            "anomaly_labels": self._plot_anomaly_labels,
            "plot_anomaly": self._plot_anomaly,
            "scatter": self._plot_scatter,
            "line": self._plot_line,
            "hist": self._plot_hist,
            "boxplot": self._plot_boxplot,
            "heatmap": self._plot_heatmap,
        }

        plot_method = plot_methods.get(self.plot_technique)

        if plot_method:
            self.fig = plot_method()  # Save the generated Plotly Figure
        else:
            raise ValueError("Unrecognized plotting technique.")

    def to_json(self):
        """Returns the plot as a JSON string (for API usage)."""
        plot_methods = {
            "prediction_forecasting": self._plot_prediction_forecasting,
            "anomaly_labels": self._plot_anomaly_labels,
            "plot_anomaly": self._plot_anomaly,
            "scatter": self._plot_scatter,
            "line": self._plot_line,
            "hist": self._plot_hist,
            "boxplot": self._plot_boxplot,
            "heatmap": self._plot_heatmap,
        }

        plot_method = plot_methods.get(self.plot_technique)

        if plot_method:
            fig = plot_method()
            return fig.to_json()
        else:
            raise ValueError("Unrecognized plotting technique.")

    def _plot_prediction_forecasting(self):
        """
        Visualizes the comparison between true values and predicted values.

        This method creates a line plot with two traces:
        1. True values (`y_true`) represented as solid lines.
        2. Predicted values (`y_pred`) represented as dashed lines.

        Raises:
            ValueError: If either 'y_true' or 'y_pred' is None.
        """

        if self.y_true is None or self.y_pred is None:
            raise ValueError(
                "Both 'y_true' and 'y_pred' are required for prediction visualization."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.y_true)),
                y=self.y_true,
                mode="lines",
                name="True Values",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.y_pred)),
                y=self.y_pred,
                mode="lines",
                name="Predictions",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title="Comparison of True and Predicted Values",
            xaxis_title="Index",
            yaxis_title="Value",
            width=800,
            height=500,
        )
        fig.show()
        return fig

    def _plot_anomaly_labels(self):
        """
        Visualizes the comparison between true anomaly labels and predicted anomaly labels.

        This method creates a scatter plot with two sets of markers:
        1. True labels (`y_true`) represented by blue circles.
        2. Predicted labels (`y_pred`) represented by red crosses.

        Raises:
            ValueError: If either 'y_true' or 'y_pred' is None.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError(
                "Both 'y_true' and 'y_pred' are required for prediction visualization."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.y_true)),
                y=self.y_true,
                mode="markers",
                name="True Labels",
                marker=dict(color="blue", symbol="circle", size=10, opacity=0.7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.y_pred)),
                y=self.y_pred,
                mode="markers",
                name="Predicted Labels",
                marker=dict(color="red", symbol="x", size=8, opacity=0.8),
            )
        )
        fig.update_layout(
            title="Comparison of True and Predicted Labels",
            xaxis_title="Index",
            yaxis_title="Label (0 or 1)",
            yaxis=dict(tickvals=[0, 1], ticktext=["0", "1"]),
            width=800,
            height=500,
        )
        fig.show()
        return fig

    def _plot_anomaly(self):
        """
        Visualizes anomalies in a 2D space by comparing true and predicted labels.

        This method creates a 2D scatter plot of reduced data, with color coding:
        1. True normal values (0) are shown in blue.
        2. True anomalies (1) are shown in red.
        3. Incorrect predictions are shown in orange.

        It randomly selects a subset of the data for visualization.

        Raises:
            ValueError: If either 'reduced_data', 'y_true', or 'y_pred' is None.
        """
        if self.reduced_data is None:
            raise ValueError("You must run 'fit()' before 'show()'.")

        if self.y_true is None or self.y_pred is None:
            raise ValueError(
                "Both 'y_true' and 'y_pred' are required for anomaly visualization."
            )

        subset_size = int(self.subset_size_percent * len(self.data))
        subset_indices = np.random.choice(len(self.data), subset_size, replace=False)

        y_true_subset = self.y_true[subset_indices]
        y_pred_subset = self.y_pred[subset_indices]
        X_test_2D = self.reduced_data[subset_indices]

        colors = np.array(
            [
                "rgba(0, 0, 255, 0.6)" if label == 0 else "rgba(255, 0, 0, 0.6)"
                for label in y_true_subset
            ]
        )
        edge_colors = np.where(
            y_true_subset == y_pred_subset, colors, "rgba(255, 165, 0, 0.8)"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=X_test_2D[:, 0],
                y=X_test_2D[:, 1],
                mode="markers",
                marker=dict(
                    color=colors, size=12, line=dict(color=edge_colors, width=2)
                ),
                text=["True label: " + str(label) for label in y_true_subset],
                hoverinfo="text",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color="rgba(0, 0, 255, 0.6)",
                    size=12,
                    line=dict(color="rgba(0, 0, 255, 0.6)", width=2),
                ),
                name="Normal (y_true=0)",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color="rgba(255, 0, 0, 0.6)",
                    size=12,
                    line=dict(color="rgba(255, 0, 0, 0.6)", width=2),
                ),
                name="Anomaly (y_true=1)",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color="rgba(255, 165, 0, 0.8)",
                    size=12,
                    line=dict(color="rgba(255, 165, 0, 0.8)", width=2),
                ),
                name="Incorrect Prediction",
                showlegend=True,
            )
        )

        # Legend
        fig.update_layout(
            title="Visualization of Anomalies",
            xaxis_title="Principle Component 1",
            yaxis_title="Principle Component 2",
            showlegend=True,
            legend=dict(
                title="Legend",
                font=dict(size=12),  # Legend font size
                itemsizing="constant",  # fixed size of legend icons
                bordercolor="Black",  # Black border for the legend
                borderwidth=2.0,  #  border width of the legend border
                orientation="h",  # Legend in horizontal format
                x=0.5,  # Center the legend on X-axis
                xanchor="center",  # Legend in the center
            ),
            width=900,
            height=700,
        )
        fig.show()
        return fig

    def _plot_scatter(self):
        """
        Creates a scatter plot to visualize the 2D reduced data.

        This method plots the data points based on the first two principal components,
        with optional color mapping and size adjustments.

        Raises:
            ValueError: If 'reduced_data' is None.
        """
        if self.reduced_data is None:
            raise ValueError("You must run 'fit()' before 'show()'.")

        color = (
            self.color_map if isinstance(self.color_map, (list, np.ndarray)) else None
        )
        fig = px.scatter(
            x=self.reduced_data[:, 0],
            y=self.reduced_data[:, 1],
            title="Data Visualization (Scatter)",
            labels={"x": "Component 1", "y": "Component 2"},
            color=color,
            opacity=self.opacity,
            size=[self.point_size] * len(self.reduced_data),
            **self.plot_kwargs
        )
        fig.show()
        return fig

    def _plot_line(self):
        """
        Creates a line plot to visualize the first component of the reduced data.

        This method plots the values of the first principal component over the indices of the data.

        Raises:
            ValueError: If 'reduced_data' is None.
        """
        if self.reduced_data is None:
            raise ValueError("You must run 'fit()' before 'show()'.")
        fig = px.line(
            x=np.arange(len(self.reduced_data[:, 0])),
            y=self.reduced_data[:, 0],
            title="Data Visualization (Line)",
            labels={"x": "Index", "y": "Value"},
            **self.plot_kwargs
        )
        fig.show()
        return fig

    def _plot_hist(self):
        """
        Creates a histogram to visualize the distribution of the reduced data.
        """
        fig = px.histogram(
            self.reduced_data.flatten(),
            nbins=30,
            title="Data Histogram",
            labels={"value": "Value"},
            **self.plot_kwargs
        )
        fig.show()
        return fig

    def _plot_boxplot(self):
        fig = px.box(
            self.reduced_data,
            title="Box Plot",
            labels={"value": "Value"},
            **self.plot_kwargs
        )
        fig.show()
        return fig

    def _plot_heatmap(self):
        """
        Creates a heatmap to visualize the intensity of values in the reduced data.
        This method plots a heatmap using the reduced data with a specified color scale.
        """
        fig = go.Figure(
            data=go.Heatmap(z=self.reduced_data, colorscale=self.heatmap_color)
        )
        fig.update_layout(
            title="Data Heatmap", width=800, height=500, **self.plot_kwargs
        )
        fig.show()
        return fig

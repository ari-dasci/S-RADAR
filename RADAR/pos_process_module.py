import numpy as np


def process_scores(d_scores, contamination):
    """
    Calculate binary labels based on a contamination threshold.

    Args:
        d_scores (array-like): Decision scores.
        contamination (float): Proportion of outliers in the dataset (0.0 to 0.5).

    Returns:
        np.ndarray: Binary labels (0 for normal, 1 for anomaly).
    """
    num_anomalies = int(contamination * len(d_scores))
    threshold = np.partition(d_scores, -num_anomalies)[
        -num_anomalies
    ]  # Efficient threshold selection
    return (d_scores >= threshold).astype(int)


def process_scores_with_percentile(d_scores, contamination):
    """
    Compute the threshold using the percentile method.

    Args:
        d_scores (array-like): Decision scores.
        contamination (float): Proportion of outliers in the dataset (0.0 to 0.5).

    Returns:
        float: Threshold value.
    """
    return np.percentile(d_scores, 100 * (1 - contamination))


def process_scores_with_threshold(d_scores):
    """
    Compute the threshold using mean and standard deviation.

    Args:
        d_scores (array-like): Decision scores.

    Returns:
        float: Threshold value.
    """
    return np.mean(d_scores) + 2 * np.std(d_scores)


def compute_anomaly_proportion(labels):
    """
    Calculates the proportion of anomalies in the data set.

    Args:
    -----
    labels: Binary anomaly labels (0: normal, 1: anomalous).

    Returns:
    --------
    proportion: Proportion of anomalies in the data.
    """
    proportion = np.sum(labels) / len(labels)
    return proportion


def remove_low_confidence_anomalies(d_scores, anomalies, confidence_threshold=0.8):
    """
    Removes anomalies with a decision score below a confidence threshold.
    Args:
    -----
    d_scores: List of decision scores.
    anomalies: Indices of detected anomalies.
    confidence_threshold: Confidence threshold (between 0 and 1).

    Returns:
    --------
    filtered_anomalies: List of anomaly indexes with high confidence.
    """
    filtered_anomalies = [i for i in anomalies if d_scores[i] >= confidence_threshold]
    return filtered_anomalies

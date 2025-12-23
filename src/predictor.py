from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from numpy.typing import NDArray


class HousingPredictor:
    """
    A predictor class for estimating California housing prices.

    This class wraps a trained Random Forest model and provides methods
    for making predictions with confidence intervals and extracting
    feature importance information.

    Attributes:
        model: The trained scikit-learn model (typically RandomForestRegressor)
    """

    def __init__(self, model_path: Path | str) -> None:
        """
        Initialize the predictor by loading a trained model from disk.

        Parameters:
            model_path: Path to the serialized model file (.pkl format)
                       Can be either a Path object or a string path

        Note:
            The model is expected to be a scikit-learn model saved with joblib
        """
        # Convert string to Path if necessary for consistent handling
        if isinstance(model_path, str):
            model_path = Path(model_path)

        self.model = joblib.load(model_path)

    def predict_price(self, features: Dict[str, Any]) -> float:
        """
        Make a price prediction based on input features.
        Parameters:
            features: Dictionary of feature names and their values.
                      Must include all features required by the model
        Returns:
            Predicted housing price as a float
        """
        # Convert the dictionary to a DataFrame because scikit-learn models
        # expect tabular data in this format. The [features] wraps the dict
        # in a list to create a single-row DataFrame
        df: pd.DataFrame = pd.DataFrame([features])

        # Make prediction and extract the first (and only) value
        # The [0] index is needed because predict() returns an array
        prediction: float = self.model.predict(df)[0]

        return prediction

    def predict_with_confidence(
        self,
        features: Dict[str, Any],
        n_estimators: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Make a price prediction with a 95% confidence interval.
        This method uses the ensemble nature of Random Forests to estimate
        uncertainty in the predictions.
        Parameters:
            features: Dictionary of feature names and their values. 
            Must include all features
            n_estimators: Optional number of trees to use for prediction.
                          If None, uses all trees in the model.
            Returns:
            Dictionary containing:
                - 'prediction': The mean predicted price
                - 'lower_bound': Lower bound of the 95% confidence interval
                - 'upper_bound': Upper bound of the 95% confidence interval
                - 'confidence_range': Width of the confidence interval
        """

        # Prepare the input data in the format expected by the model
        df: pd.DataFrame = pd.DataFrame([features])

        # Collect predictions from each tree in the Random Forest
        # This gives us a distribution of predictions rather than just one value
        predictions: List[float] = []
        for tree in self.model.estimators_:
            # Each tree makes its own independent prediction
            pred: float = tree.predict(df)[0]
            predictions.append(pred)

        # Convert to numpy array for efficient statistical calculations
        predictions_array: NDArray[np.float64] = np.array(predictions)

        # Calculate the central tendency and spread of predictions
        # The mean gives us our best estimate
        mean_prediction: float = float(predictions_array.mean())
        # The standard deviation tells us how much trees disagree
        std_prediction: float = float(predictions_array.std())

        # Calculate 95% confidence interval using the empirical rule
        # 1.96 standard deviations captures approximately 95% of a normal distribution
        lower_bound: float = mean_prediction - (1.96 * std_prediction)
        upper_bound: float = mean_prediction + (1.96 * std_prediction)

        return {
            'prediction': mean_prediction,
            # Ensure price is non-negative
            'lower_bound': max(0.0, lower_bound),
            'upper_bound': upper_bound,
            'confidence_range': upper_bound - lower_bound
        }

    def get_feature_importance(self) -> Optional[Dict[str, List]]:
        """
        Extract feature importance scores from the trained model.

        Feature importance tells us which features (like median_income or
        longitude) have the most influence on the model's predictions.
        Higher values indicate more important features.

        This is useful for understanding what drives housing prices and
        for model interpretation and validation.

        Returns:
            Dictionary with two keys if model supports feature importance:
                - 'features': List of feature names sorted by importance (most to least)
                - 'importances': List of corresponding importance scores
            Returns None if the model doesn't support feature importance

        Note:
            Only works with models that have feature_importances_ attribute
            (like RandomForest, GradientBoosting, etc.)
        """
        # Check if this model type provides feature importance
        # Tree-based models have this attribute, but linear models don't
        if hasattr(self.model, 'feature_importances_'):
            # Get the feature names that the model was trained with
            # This ensures we match importance scores to the correct features
            feature_names: NDArray = self.model.feature_names_in_

            # Get the importance score for each feature
            # These scores sum to 1.0 and represent relative importance
            importances: NDArray = self.model.feature_importances_

            # Sort features by importance in descending order
            # [::-1] reverses the array so highest importance comes first
            indices: NDArray = np.argsort(importances)[::-1]

            # Build the result dictionary with features sorted by importance
            return {
                'features': [feature_names[i] for i in indices],
                'importances': [float(importances[i]) for i in indices]
            }
        else:
            # Model doesn't support feature importance (e.g., linear regression)
            return None

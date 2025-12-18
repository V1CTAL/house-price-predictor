from pathlib import Path
import joblib  # type: ignore
import pandas as pd
import numpy as np


class HousingPredictor:
    def __init__(self, model_path):
        """Load the trained model"""
        self.model = joblib.load(model_path)

    def predict_price(self, features):
        """
        Make a prediction given housing features

        Parameters:
        features (dict): Dictionary with feature names and values

        Returns:
        float: Predicted house price
        """
        # Convert dict to DataFrame (model expects this format)
        df = pd.DataFrame([features])

        # Make prediction
        prediction = self.model.predict(df)[0]

        return prediction

    def predict_with_confidence(self, features, n_estimators=None):
        """
        Make prediction with confidence interval (only works with RandomForest)

        Returns:
        dict: prediction, lower_bound, upper_bound
        """
        df = pd.DataFrame([features])

        # Get predictions from all trees
        predictions = []
        for tree in self.model.estimators_:
            pred = tree.predict(df)[0]
            predictions.append(pred)

        predictions = np.array(predictions)

        mean_prediction = predictions.mean()
        std_prediction = predictions.std()

        # 95% confidence interval
        lower_bound = mean_prediction - (1.96 * std_prediction)
        upper_bound = mean_prediction + (1.96 * std_prediction)

        return {
            'prediction': mean_prediction,
            'lower_bound': max(0, lower_bound),  # Price can't be negative
            'upper_bound': upper_bound,
            'confidence_range': upper_bound - lower_bound
        }

    def get_feature_importance(self):
        """
        Get feature importance from the model

        Returns:
        dict: feature names and their importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.model.feature_names_in_
            importances = self.model.feature_importances_

            # Sort by importance
            indices = np.argsort(importances)[::-1]

            return {
                'features': [feature_names[i] for i in indices],
                'importances': [importances[i] for i in indices]
            }
        else:
            return None

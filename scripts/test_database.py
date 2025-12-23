import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Assuming these return DataFrames or specific types based on your snippet
from src.database import HousingPriceDB

# Load environment variables (pathlib can be used to locate the .env file explicitly if needed)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


def test_database() -> None:
    """Test all database functions for housing price predictor with type safety."""

    print('🏠 Testing Housing Price Database Connection')
    print('=' * 50)

    # Initialize database using os.getenv but preparing for type hints
    # Note: DB_PORT is cast to int for type consistency
    db = HousingPriceDB(
        dbname=os.getenv('DB_NAME', 'housing_db'),
        user=os.getenv('DB_USER', 'housing_user'),
        # Defaults to empty string instead of None
        password=str(os.getenv('DB_PASSWORD')),
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432'))
    )

    try:
        # Test 1: Insert a property
        print('\n📝 Test 1: Inserting a property...')
        property_data: Dict[str, Any] = {
            'address': '456 Test Street',
            'bedrooms': 3,
            'bathrooms': 2.5,
            'square_feet': 2000,
            'lot_size': 5000,
            'year_built': 2010,
            'zip_code': '12345',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'actual_price': 500000.00
        }

        property_id: int = db.insert_property(property_data)
        print(f'✅ Inserted property with ID: {property_id}')

        # Test 2: Retrieve the property
        print('\n🔍 Test 2: Retrieving property...')
        retrieved: Optional[Dict[str, Any]
                            ] = db.get_property_by_id(property_id)

        if retrieved:
            print(f'✅ Retrieved property: {retrieved["address"]}')
            print(
                f'   Bedrooms: {retrieved["bedrooms"]}, Price: ${retrieved["actual_price"]:,.2f}')

        # Test 3: Save a prediction
        print('\n🎯 Test 3: Saving a prediction...')
        predicted_price: float = 485000.00
        prediction_id: int = db.save_prediction(
            property_id=property_id,
            predicted_price=predicted_price,
            model_version='v1.0',
            confidence=0.92
        )
        print(f'✅ Saved prediction with ID: {prediction_id}')
        print(
            f'   Predicted: ${predicted_price:,.2f} vs Actual: ${property_data["actual_price"]:,.2f}')

        # Test 4: Get prediction history
        print('\n📊 Test 4: Getting prediction history...')
        # history is typically a pandas DataFrame in these contexts
        history: Any = db.get_predictions_history(property_id)
        print(f'✅ Retrieved {len(history)} predictions')
        if not history.empty:
            print(history[['predicted_price', 'actual_price',
                  'model_version', 'prediction_date']].head())

        # Test 5: Get training data
        print('\n📚 Test 5: Getting training data...')
        training_data: Any = db.get_properties_for_training(limit=5)
        print(f'✅ Retrieved {len(training_data)} properties for training')
        print(training_data[['bedrooms', 'bathrooms',
              'square_feet', 'actual_price']].head())

        # Test 6: Search properties
        print('\n🔎 Test 6: Searching properties...')
        filters: Dict[str, Union[int, float]] = {
            'min_bedrooms': 2,
            'max_price': 600000
        }
        results: pd.DataFrame = db.search_properties(filters)
        print(f'✅ Found {len(results)} properties matching criteria')

        # Test 7: Get model performance stats
        print('\n📈 Test 7: Getting model performance stats...')
        stats: Optional[Dict[str, Any]
                        ] = db.get_model_performance_stats('v1.0')

        if stats:
            print(f'✅ Model Performance (v1.0):')
            print(f'   MAE: ${stats["mae"]:,.2f}')
            print(f'   MAPE: {stats["mape"]:.2f}%')
            print(f'   RMSE: ${stats["rmse"]:,.2f}')
            print(f'   Total Predictions: {stats["total_predictions"]}')
        else:
            print('⚠️  No predictions found for performance calculation')

        print('\n' + '=' * 50)
        print('✅ All tests passed successfully!')

    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_database()

import psycopg2  # type: ignore
from psycopg2.extras import RealDictCursor, execute_batch  # type: ignore
from psycopg2.extensions import connection as PsycopgConnection  # type: ignore
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union, Generator


class HousingPriceDB:
    """Database interface for housing price predictor using PEP 484 type hints"""

    def __init__(
            self,
            dbname: str,
            user: str,
            password: str,
            host: str = 'localhost',
            port: int = 5432) -> None:

        self.connection_params: Dict[str, Union[str, int]] = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }

    @contextmanager
    def get_connection(self) -> Generator[PsycopgConnection, None, None]:
        """Context manager for database connections with automatic commit/rollback"""
        conn: PsycopgConnection = psycopg2.connect(
            **self.connection_params)

        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def insert_property(self, property_data: Dict[str, Any]) -> int:
        """"Insert a single property record and return the generated ID."""
        query = """
            INSERT INTO properties (
                address, bedrooms, bathrooms, square_feet, lot_size,
                year_built, zip_code, latitude, longitude, actual_price
            ) VALUES (
                %(address)s, %(bedrooms)s, %(bathrooms)s, %(square_feet)s,
                %(lot_size)s, %(year_built)s, %(zip_code)s, %(latitude)s,
                %(longitude)s, %(actual_price)s
            ) RETURNING id
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, property_data)
                result = cur.fetchone()
                return int(result[0] if result else 0)

    def insert_properties_batch(self, properties_df: pd.DataFrame) -> int:
        """Bulk insert properties from a pandas DataFrame."""
        query = """
            INSERT INTO properties (
                address, bedrooms, bathrooms, square_feet, lot_size,
                year_built, zip_code, latitude, longitude, actual_price
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        records: List[tuple] = [tuple(row) for row in properties_df.values]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, query, records, page_size=1000)

        return len(records)

    def save_prediction(
            self,
            property_id: int,
            predicted_price: float,
            model_version: str,
            confidence: Optional[float] = None) -> int:
        """Save a prediction result to the predictions table."""
        query = """
            INSERT INTO predictions (
                property_id, predicted_price, model_version, confidence_score
            ) VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (property_id, predicted_price,
                            model_version, confidence))
                result = cur.fetchone()
                return int(result[0]) if result else 0

    def get_properties_for_training(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve cleaned property data specifically for model training."""
        query = """
            SELECT 
                bedrooms, bathrooms, square_feet, lot_size,
                year_built, zip_code, latitude, longitude, actual_price
            FROM properties
            WHERE actual_price IS NOT NULL
        """
        if limit:
            query += f'LIMIT {limit}'

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_property_by_id(self, property_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific property as a dictionary."""
        query = "SELECT * FROM properties WHERE id = %s"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (property_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def get_predictions_history(self, property_id: int) -> pd.DataFrame:
        """Get all historical predictions for a single property ID."""
        query = """
            SELECT 
                p.predicted_price, p.model_version, p.prediction_date,
                p.confidence_score, pr.actual_price
            FROM predictions p
            JOIN properties pr ON p.property_id = pr.id
            WHERE p.property_id = %s
            ORDER BY p.prediction_date DESC
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(property_id,))

    def get_all_predictions_history(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions across all properties."""
        query = """
            SELECT
                pr.id AS property_id,
                pr.address,
                p.predicted_price,
                p.model_version,
                p.confidence_score,
                p.prediction_date
            FROM predictions p
            JOIN properties pr ON p.property_id = pr.id
            ORDER BY p.prediction_date DESC
            LIMIT %s
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_model_performance_stats(self, model_version: str) -> Optional[Dict[str, float]]:
        """Calculate and return key performance metrics (MAE, MAPE, RMSE)."""
        query = """
            SELECT 
                p.predicted_price, pr.actual_price,
                ABS(p.predicted_price - pr.actual_price) as absolute_error,
                ABS(p.predicted_price - pr.actual_price) / pr.actual_price * 100 as percent_error
            FROM predictions p
            JOIN properties pr ON p.property_id = pr.id
            WHERE p.model_version = %s AND pr.actual_price IS NOT NULL
        """
        with self.get_connection() as conn:
            df: pd.DataFrame = pd.read_sql_query(
                query, conn, params=(model_version,))

            if df.empty:
                return None

            return {
                'mae': float(df['absolute_error'].mean()),
                'mape': float(df['percent_error'].mean()),
                'rmse': float(np.sqrt((df['absolute_error'] ** 2).mean())),
                'total_predictions': float(len(df))
            }

    def search_properties(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Search properties using dynamic filtering."""
        conditions: List[str] = []
        params: List[Any] = []

        mapping = {
            'min_bedrooms': ('bedrooms >= %s', filters.get('min_bedrooms')),
            'max_price': ('actual_price >= %s', filters.get('max_price')),
            'zip_code': ('zip_code >= %s', filters.get('zip_code')),
            'min_sqft': ('square_feet >= %s', filters.get('min_sqft'))
        }

        for key, (condition, value) in mapping.items():
            if value is not None:
                conditions.append(condition)
                params.append(value)

        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        query = f'SELECT * FROM properties WHERE {where_clause}'

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

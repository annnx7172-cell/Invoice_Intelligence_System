import os
import sys
import sqlite3
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    db_path: str = os.path.join('notebook', 'data', 'inventory.db')

    # Freight (regression) - single table
    freight_raw_path: str = os.path.join('artifacts', 'freight_raw.csv')
    freight_train_path: str = os.path.join('artifacts', 'freight_train.csv')
    freight_test_path: str = os.path.join('artifacts', 'freight_test.csv')

    # Risk (classification) - joined tables
    risk_raw_path: str = os.path.join('artifacts', 'risk_raw.csv')
    risk_train_path: str = os.path.join('artifacts', 'risk_train.csv')
    risk_test_path: str = os.path.join('artifacts', 'risk_test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_freight_ingestion(self, conn):
        """
        Single-table ingestion: vendor_invoice only.
        Used for Freight regression.
        """
        try:
            logging.info("Starting Freight data ingestion (vendor_invoice only)")

            df = pd.read_sql("SELECT * FROM vendor_invoice", conn)
            logging.info(f"Read vendor_invoice table: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.freight_train_path), exist_ok=True)
            df.to_csv(self.ingestion_config.freight_raw_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.freight_train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.freight_test_path, index=False, header=True)

            logging.info("Freight data ingestion completed")

            return (
                self.ingestion_config.freight_train_path,
                self.ingestion_config.freight_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_risk_ingestion(self, conn):
        """
        Joined ingestion: vendor_invoice + aggregated purchases.
        Used for invoice Risk classification.
        """
        try:
            logging.info("Starting Risk data ingestion (vendor_invoice JOIN purchases)")

            query = """
            WITH purchase_agg AS (
                SELECT
                    p.PONumber,
                    COUNT(DISTINCT p.Brand) AS total_brands,
                    SUM(p.Quantity) AS total_item_quantity,
                    SUM(p.Dollars) AS total_item_dollars,
                    AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
                FROM purchases p
                GROUP BY p.PONumber
            )
            SELECT
                vi.PONumber,
                vi.VendorName,
                vi.Quantity AS invoice_quantity,
                vi.Dollars AS invoice_dollars,
                vi.Freight,
                (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
                (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
                pa.total_brands,
                pa.total_item_quantity,
                pa.total_item_dollars,
                pa.avg_receiving_delay
            FROM vendor_invoice vi
            LEFT JOIN purchase_agg pa
                ON vi.PONumber = pa.PONumber
            """

            df = pd.read_sql(query, conn)
            logging.info(f"Read joined vendor_invoice + purchases data: {df.shape}")

            # Build the composite risk label
            def create_invoice_risk_label(row):
                if abs(row['invoice_dollars'] - row['total_item_dollars']) > 5:
                    return 1
                if row['avg_receiving_delay'] > 10:
                    return 1
                return 0

            df['flag_invoice'] = df.apply(create_invoice_risk_label, axis=1)
            logging.info(f"Risk label distribution: {df['flag_invoice'].value_counts().to_dict()}")

            os.makedirs(os.path.dirname(self.ingestion_config.risk_train_path), exist_ok=True)
            df.to_csv(self.ingestion_config.risk_raw_path, index=False, header=True)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df['flag_invoice']
            )
            train_set.to_csv(self.ingestion_config.risk_train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.risk_test_path, index=False, header=True)

            logging.info("Risk data ingestion completed")

            return (
                self.ingestion_config.risk_train_path,
                self.ingestion_config.risk_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """
        Runs both ingestion paths using a single shared DB connection.
        """
        try:
            conn = sqlite3.connect(self.ingestion_config.db_path)

            freight_train, freight_test = self.initiate_freight_ingestion(conn)
            risk_train, risk_test = self.initiate_risk_ingestion(conn)

            conn.close()

            return {
                "freight_train": freight_train,
                "freight_test": freight_test,
                "risk_train": risk_train,
                "risk_test": risk_test
            }

        except Exception as e:
            raise CustomException(e, sys)




if __name__ == "__main__":
    obj = DataIngestion()
    paths = obj.initiate_data_ingestion()
    print(paths)

    data_transformation = DataTransformation()

    freight_train_arr, freight_test_arr, _ = data_transformation.initiate_freight_transformation(
        paths["freight_train"], paths["freight_test"]
    )
    print("Freight train shape:", freight_train_arr.shape)
    print("Freight test shape:", freight_test_arr.shape)

    risk_train_arr, risk_test_arr, _ = data_transformation.initiate_risk_transformation(
        paths["risk_train"], paths["risk_test"]
    )
    print("Risk train shape:", risk_train_arr.shape)
    print("Risk test shape:", risk_test_arr.shape)
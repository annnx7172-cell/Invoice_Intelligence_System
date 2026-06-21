import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    freight_preprocessor_path: str = os.path.join('artifacts', "freight_preprocessor.pkl")
    risk_preprocessor_path: str = os.path.join('artifacts', "risk_preprocessor.pkl")
    freight_vendor_map_path: str = os.path.join('artifacts', "freight_vendor_map.pkl")
    risk_vendor_map_path: str = os.path.join('artifacts', "risk_vendor_map.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # =====================================================================
    # FREIGHT (REGRESSION) - single table, needs date feature engineering
    # =====================================================================

    def engineer_freight_features(self, df):
        try:
            df['PODate'] = pd.to_datetime(df['PODate'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['PayDate'] = pd.to_datetime(df['PayDate'])

            df['days_po_to_invoice'] = (df['InvoiceDate'] - df['PODate']).dt.days
            df['days_invoice_to_pay'] = (df['PayDate'] - df['InvoiceDate']).dt.days
            df['days_po_to_pay'] = (df['PayDate'] - df['PODate']).dt.days

            df['Quantity_log'] = np.log1p(df['Quantity'])
            df['Dollars_log'] = np.log1p(df['Dollars'])

            df = df.drop(columns=['PODate', 'InvoiceDate', 'PayDate', 'Approval',
                                   'PONumber', 'VendorNumber', 'Quantity', 'Dollars'])

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_freight_transformer_object(self):
        try:
            # VendorName is target-encoded separately (see apply_vendor_target_encoding),
            # so it's treated as numerical here, not passed through OneHotEncoder.
            numerical_columns = [
                "Quantity_log", "Dollars_log",
                "days_po_to_invoice", "days_invoice_to_pay", "days_po_to_pay",
                "VendorName_encoded"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def apply_vendor_target_encoding(self, train_df, test_df, target_col, save_path):
        """
        Replaces VendorName with the mean of `target_col` for that vendor,
        computed ONLY from training data, then mapped onto both train and test.
        Unseen vendors in test get the overall training mean (fallback).
        Also saves the vendor_means + overall_mean as a pkl, so the deployed
        app can apply the SAME encoding to a vendor name typed in by a user.
        """
        try:
            vendor_means = train_df.groupby('VendorName')[target_col].mean()
            overall_mean = train_df[target_col].mean()

            train_df['VendorName_encoded'] = train_df['VendorName'].map(vendor_means)
            test_df['VendorName_encoded'] = test_df['VendorName'].map(vendor_means).fillna(overall_mean)

            train_df = train_df.drop(columns=['VendorName'])
            test_df = test_df.drop(columns=['VendorName'])

            save_object(
                file_path=save_path,
                obj={"vendor_means": vendor_means.to_dict(), "overall_mean": overall_mean}
            )

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_freight_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read freight train and test data")

            train_df = self.engineer_freight_features(train_df)
            test_df = self.engineer_freight_features(test_df)
            logging.info("Freight feature engineering completed")

            train_df, test_df = self.apply_vendor_target_encoding(
                train_df, test_df, target_col="Freight",
                save_path=self.data_transformation_config.freight_vendor_map_path
            )
            logging.info("Vendor target encoding applied (freight)")

            preprocessing_obj = self.get_freight_transformer_object()

            target = "Freight"
            input_feature_train_df = train_df.drop(columns=[target])
            input_feature_test_df = test_df.drop(columns=[target])

            target_train = train_df[target]
            target_test = test_df[target]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr.toarray() if hasattr(input_feature_train_arr, "toarray") else input_feature_train_arr,
                np.array(target_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr.toarray() if hasattr(input_feature_test_arr, "toarray") else input_feature_test_arr,
                np.array(target_test)
            ]

            save_object(
                file_path=self.data_transformation_config.freight_preprocessor_path,
                obj=preprocessing_obj
            )
            logging.info("Saved freight preprocessing object")

            return train_arr, test_arr, self.data_transformation_config.freight_preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # RISK (CLASSIFICATION) - joined data, already has engineered date gaps
    # =====================================================================

    def get_risk_transformer_object(self):
        try:
            # VendorName is target-encoded separately, treated as numerical here.
            numerical_columns = [
                "invoice_quantity", "invoice_dollars", "Freight",
                "days_po_to_invoice", "days_to_pay",
                "total_brands", "total_item_quantity", "total_item_dollars",
                "avg_receiving_delay", "VendorName_encoded"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_risk_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read risk train and test data")

            train_df = train_df.drop(columns=['PONumber'])
            test_df = test_df.drop(columns=['PONumber'])

            train_df, test_df = self.apply_vendor_target_encoding(
                train_df, test_df, target_col="flag_invoice",
                save_path=self.data_transformation_config.risk_vendor_map_path
            )
            logging.info("Vendor target encoding applied (risk)")

            preprocessing_obj = self.get_risk_transformer_object()

            target = "flag_invoice"
            input_feature_train_df = train_df.drop(columns=[target])
            input_feature_test_df = test_df.drop(columns=[target])

            target_train = train_df[target]
            target_test = test_df[target]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr.toarray() if hasattr(input_feature_train_arr, "toarray") else input_feature_train_arr,
                np.array(target_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr.toarray() if hasattr(input_feature_test_arr, "toarray") else input_feature_test_arr,
                np.array(target_test)
            ]

            save_object(
                file_path=self.data_transformation_config.risk_preprocessor_path,
                obj=preprocessing_obj
            )
            logging.info("Saved risk preprocessing object")

            return train_arr, test_arr, self.data_transformation_config.risk_preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
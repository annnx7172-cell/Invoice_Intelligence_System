import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
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
            # LEAKAGE FIX: the invoice/payment date gaps (days_po_to_invoice,
            # days_invoice_to_pay, days_po_to_pay) are only known AFTER the order
            # is billed/paid, so they cannot be used to estimate freight. Only
            # order-time information (quantity, dollars, vendor) is kept.
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

    @staticmethod
    def _smoothed_vendor_means(df, target_col, overall_mean, smoothing):
        """Vendor mean shrunk towards the overall mean: (n*mean + m*overall) / (n + m)."""
        stats = df.groupby('VendorName')[target_col].agg(['sum', 'count'])
        return (stats['sum'] + smoothing * overall_mean) / (stats['count'] + smoothing)

    def apply_vendor_target_encoding(self, train_df, test_df, target_col, save_path,
                                     smoothing=10, n_splits=5, out_of_fold=True):
        """
        Replaces VendorName with a smoothed mean of `target_col` for that vendor.

        LEAKAGE FIX: a plain vendor mean computed on the training set contains each
        training row's own target, so the encoded feature leaks the label (worst for
        vendors with few invoices). Here:
          * TRAIN rows are encoded out-of-fold: each row's value is computed from the
            other folds only, so it never sees its own target.
          * Means are smoothed towards the overall mean, so rare vendors are not
            driven by one or two invoices.
          * TEST rows (and the app) use the smoothed map fitted on the full training
            set. Unseen vendors get the overall training mean.
        The map is saved as a pkl so the deployed app applies the SAME encoding.
        """
        try:
            if not out_of_fold:
                # Original behaviour (risk pipeline): plain vendor mean from the full
                # training set, applied to train and test alike.
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

            overall_mean = train_df[target_col].mean()

            vendor_means = self._smoothed_vendor_means(train_df, target_col, overall_mean, smoothing)

            oof_encoded = np.full(len(train_df), overall_mean, dtype=float)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fit_idx, enc_idx in kf.split(train_df):
                fold_fit = train_df.iloc[fit_idx]
                fold_overall = fold_fit[target_col].mean()
                fold_means = self._smoothed_vendor_means(fold_fit, target_col, fold_overall, smoothing)
                oof_encoded[enc_idx] = (
                    train_df.iloc[enc_idx]['VendorName'].map(fold_means).fillna(fold_overall).values
                )

            train_df['VendorName_encoded'] = oof_encoded
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
                save_path=self.data_transformation_config.risk_vendor_map_path,
                out_of_fold=False
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
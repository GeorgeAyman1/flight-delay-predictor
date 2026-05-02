from __future__ import annotations

from pathlib import Path


import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


#-----------------------------------------------------------------------------------------------------

##the column groups for imputation
MEDIAN_COLS = [
    'tmpf', 'dwpf', 'relh', 'sknt', 
    'vsby', 'alti', 'mslp', 'feel'
]

ZERO_FILL_COLS = ['p01i']

SKY_LAYER_COLS  = ['skyc2', 'skyc3', 'skyc4']
SKY_HEIGHT_COLS = ['skyl1', 'skyl2', 'skyl3', 'skyl4']

WXCODE_COL = 'wxcodes'
SKYC1_COL  = 'skyc1'
GUSTY_COL  = 'gust'

DROP_COLS = ['drct']

# Target column will get ignored during preprocessing to avoid any accidental data leakage or data contamination issues
TARGET_COL = 'departure_delayed'

#-----------------------------------------------------------------------------------------------------

## cutom transformers
class SkyC1Encoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding the 'skyc1' column
    .map(self.SKY_MAP) known values get mapped, unknown values become NaN
    """
    SKY_MAP = {'CLR': 0, 'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
    
    def set_output(self, transform=None):
        return self
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        col = X.iloc[:, 0]
        result = col.map(self.SKY_MAP).fillna(5).astype(int)
        return pd.DataFrame({'skyc1_encoded': result.values}, index=X.index)
    

class IsGustyTransformer(BaseEstimator, TransformerMixin):
    """
    Creates is_gusty binary flag and zero-fills gust speed.
    Outputs two columns: [is_gusty, gust]
    """

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col = X.iloc[:, 0]
        is_gusty = col.notna().astype(int)
        gust_filled = col.fillna(0)
        return pd.DataFrame(
        {'is_gusty': is_gusty.values, 'gust': gust_filled.values},
        index=X.index
    )
    

class NumCloudLayersTransformer(BaseEstimator, TransformerMixin):
    """Counts number of cloud layers present (0-3) from skyc2/3/4."""

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.notna().sum(axis=1)
        return pd.DataFrame({'num_cloud_layers': result.values}, index=X.index)
    

class CloudCeilingTransformer(BaseEstimator, TransformerMixin):
    """Extracts the lowest cloud ceiling in feet from skyl1/2/3/4."""

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ceiling = X.min(axis=1).fillna(99999)
        # if all four are null, no clouds at all use a high value
        # if all 4 are null, it means there are no clouds so sky is clear and safe
        # high number because the model shouldn't care about high clouds these aren't dangerous
        return pd.DataFrame({'cloud_ceiling': ceiling.values}, index=X.index)

class WxCodeTransformer(BaseEstimator, TransformerMixin):
    """Extracts binary weather condition flags from wxcodes string."""

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col = X.iloc[:, 0].fillna('')

        return pd.DataFrame({
        'has_fog':      col.str.contains('FG', case=False).astype(int).values,
        'has_thunder':  col.str.contains('TS', case=False).astype(int).values,
        'has_rain':     col.str.contains('RA', case=False).astype(int).values,
        'has_snow':     col.str.contains('SN', case=False).astype(int).values,
        'has_freezing': col.str.contains('FZ', case=False).astype(int).values,
    }, index=X.index)

#-----------------------------------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    
    preprocessor = ColumnTransformer(
        transformers=[
            # --- Imputation ---
                ('median_impute',
                SimpleImputer(strategy='median'),
                MEDIAN_COLS),

                ('zero_fill',
                SimpleImputer(strategy='constant', fill_value=0),
                ZERO_FILL_COLS),

                # --- Custom transformers ---
                ('is_gusty',
                IsGustyTransformer(),
                [GUSTY_COL]),

                ('num_cloud_layers',
                NumCloudLayersTransformer(),
                SKY_LAYER_COLS),

                ('cloud_ceiling',
                CloudCeilingTransformer(),
                SKY_HEIGHT_COLS),

                ('skyc1_encode',
                SkyC1Encoder(),
                [SKYC1_COL]),

                ('wxcodes',
                WxCodeTransformer(),
                [WXCODE_COL]),

                # --- Drop ---
                ('drop_cols',
                'drop',
                DROP_COLS),
            ],
        remainder='passthrough',  # all other columns pass through unchanged
        verbose_feature_names_out=False  # keeps original column names
    )
    preprocessor.set_output(transform='pandas')
    return preprocessor


#-----------------------------------------------------------------------------------------------------


class Preprocessor:
    """
    Fits a sklearn ColumnTransformer on training data (imputation, custom
    encoders, column drops) and transforms all three splits.

    Saves preprocessed parquet files and the fitted preprocessor artifact.
    """

    TARGET_COL = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"

        self.train_path = self.data_dir / "train_2022_2023_v2.parquet"
        self.valid_path = self.data_dir / "valid_2024_v2.parquet"
        self.test_path = self.data_dir / "test_2025_v2.parquet"

        self.train_out = self.data_dir / "train_preprocessed.parquet"
        self.valid_out = self.data_dir / "valid_preprocessed.parquet"
        self.test_out = self.data_dir / "test_preprocessed.parquet"

        self.preprocessor_path = self.models_dir / "preprocessor.joblib"

    def run(self) -> dict:
        """
        Build, fit, and apply the preprocessing pipeline.

        Returns
        -------
        dict with keys:
            - train_shape : tuple
            - valid_shape : tuple
            - test_shape  : tuple
            - preprocessor_path : str
        """
        # 1. Load data
        print("Loading data...")
        train_df = pd.read_parquet(self.train_path)
        valid_df = pd.read_parquet(self.valid_path)
        test_df = pd.read_parquet(self.test_path)

        # 2. Separate features and target
        X_train = train_df.drop(columns=[self.TARGET_COL])
        y_train = train_df[self.TARGET_COL]

        X_valid = valid_df.drop(columns=[self.TARGET_COL])
        y_valid = valid_df[self.TARGET_COL]

        X_test = test_df.drop(columns=[self.TARGET_COL])
        y_test = test_df[self.TARGET_COL]

        # 3. Build and fit preprocessor on train only
        print("Fitting preprocessor on train set...")
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        # 4. Transform all three sets
        print("Transforming all sets...")
        X_train_t = preprocessor.transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)
        X_test_t = preprocessor.transform(X_test)

        # 5. Add target back
        X_train_t[self.TARGET_COL] = y_train.values
        X_valid_t[self.TARGET_COL] = y_valid.values
        X_test_t[self.TARGET_COL] = y_test.values

        # 6. Save preprocessed datasets
        print("Saving preprocessed datasets...")
        self.train_out.parent.mkdir(parents=True, exist_ok=True)
        X_train_t.to_parquet(self.train_out, index=False)
        X_valid_t.to_parquet(self.valid_out, index=False)
        X_test_t.to_parquet(self.test_out, index=False)

        # 7. Save fitted preprocessor
        print("Saving preprocessor...")
        self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)

        print("Done.")
        print(f"Train : {X_train_t.shape}")
        print(f"Valid : {X_valid_t.shape}")
        print(f"Test  : {X_test_t.shape}")
        print(f"Preprocessor saved to: {self.preprocessor_path}")

        return {
            "train_shape": X_train_t.shape,
            "valid_shape": X_valid_t.shape,
            "test_shape": X_test_t.shape,
            "preprocessor_path": str(self.preprocessor_path),
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    preprocessor = Preprocessor(base)
    preprocessor.run()
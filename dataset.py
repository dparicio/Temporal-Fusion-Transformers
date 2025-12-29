import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader

import os
from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class FeatureDescription:
    # Time tag
    time: str
    # Identifier tag, separating different time series
    id: str
    # Tag of y 
    target: str
    # Tags of static features, either categorical or continuous
    static_categorical: List[str]
    static_continuous: List[str]
    # Tags of known features
    known_categorical: List[str]
    known_continuous: List[str]
    # Tags of observed (unknown) features
    observed_continuous: List[str]
    observed_categorical: List[str]


class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        df, 
        feature_description, 
        encoder_length, 
        decoder_length,
        categorical_encoder=None
    ):
        super().__init__()
        self.df = df
        self.features = feature_description
        self.enc_len = encoder_length
        self.dec_len = decoder_length
        self.categorical_encoder = categorical_encoder
        self.time_steps = self.enc_len + self.dec_len

        # Preprocess the dataframe
        df = df.copy()
        time_col = self.features.time
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            time_as_numeric = pd.to_numeric(df[time_col], errors="coerce")
            if time_as_numeric.notna().all():
                df[time_col] = time_as_numeric
            else:
                df[time_col] = pd.to_datetime(df[time_col], utc=False)
        df.sort_values([self.features.id, self.features.time], inplace=True)

        # Group categorical features 
        self.categorical_features = (
            self.features.static_categorical +
            self.features.known_categorical +      
            self.features.observed_categorical     
        )
        # Group continuous features 
        continuous_features = (
            self.features.static_continuous +
            self.features.known_continuous +       
            self.features.observed_continuous      
        )
        self.continuous_features = [c for c in continuous_features if c != self.features.target]
        if self.continuous_features:
            df[self.continuous_features] = df[self.continuous_features].astype("float32")
        df[self.features.target] = df[self.features.target].astype("float32")

        # == Encode categorical features ==
        if self.categorical_features:
            # Ensure they are strings
            X_cat = df[self.categorical_features].astype("string")

            # Define categorical encoders and fit if not provided
            if self.categorical_encoder is None:
                self.categorical_encoder = {}
                for col in self.categorical_features:
                    encoder = LabelEncoder()
                    encoder.fit(X_cat[col].values)
                    self.categorical_encoder[col] = encoder

            # Transform using per-column label encoders
            for col in self.categorical_features:
                encoder = self.categorical_encoder[col]
                df[col] = encoder.transform(X_cat[col].values).astype(np.int64)
        else:
            self.categorical_encoder = None

        self.df = df    

        # Build samples
        self.build_samples()


    def get_embedding_per_cat(self):
        """Get number of embeddings per categorical variable."""
        if not self.categorical_features:
            raise ValueError("No categorical features in dataset.")
        
        embed_per_cat = []
        for cat in self.categorical_features:
            if self.categorical_encoder and cat in self.categorical_encoder:
                n_unique = len(self.categorical_encoder[cat].classes_)
            else:
                n_unique = int(self.df[cat].max()) + 1
            embed_per_cat.append(n_unique)

        return embed_per_cat


    def build_samples(self):
        # == Create samples and group data ==
        self.groups = {}          # Dictionary mapping id to group dataframe
        self.samples = []         # List of (id, t) tuples indicating samples feature
        for identifier, group in self.df.groupby(self.features.id, sort=False):
            group = group.reset_index(drop=True)
            self.groups[identifier] = group

            # Generates all possible valid combinations of (id, t)
            last_t = len(group) - self.dec_len
            for t in range(self.enc_len, last_t + 1):
                self.samples.append((identifier, t))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        identifier, t = self.samples[idx]
        group = self.groups[identifier]
        # Get the window of data
        enc_df = group.iloc[t - self.enc_len:t]
        dec_df = group.iloc[t:t + self.dec_len]
        win_df = group.iloc[t - self.enc_len:t + self.dec_len]

        # Get static features only once
        if self.features.static_categorical:
            static_categorial = torch.tensor(
                [int(group[c].iloc[0]) for c in self.features.static_categorical],
                dtype=torch.long
            )
        else:
            static_categorial = torch.zeros(0, dtype=torch.long)

        if self.features.static_continuous:
            static_continuous = torch.tensor(
                group[self.features.static_continuous].iloc[0].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )
        else:
            static_continuous = torch.zeros(0, dtype=torch.float32)

        # Categorical features observed
        observed_cat_cols = self.features.observed_categorical
        if observed_cat_cols:
            obs_categorial = torch.from_numpy(
                enc_df[observed_cat_cols].to_numpy(dtype=np.int64)
            )
        else:
            obs_categorial = torch.zeros((self.enc_len, 0), dtype=torch.long)

        # Continuous features observed
        observed_cont_cols = self.features.observed_continuous
        if len(observed_cont_cols) == 0:
            cont_observed = torch.empty((self.enc_len, 0), dtype=torch.float32)
        else:
            cont_observed = torch.tensor(
                enc_df[observed_cont_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )

        # Known categorical
        known_cat_cols = self.features.known_categorical
        if known_cat_cols:
            known_categorial = torch.from_numpy(
                win_df[known_cat_cols].to_numpy(dtype=np.int64)
            )
        else:
            known_categorial = torch.zeros((self.time_steps, 0), dtype=torch.long)

        # Known continuous
        known_cont_cols = self.features.known_continuous
        if known_cont_cols:
            cont_known = torch.tensor(
                win_df[known_cont_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )
        else:
            cont_known = torch.empty((self.time_steps, 0), dtype=torch.float32)

        # Target values
        target = torch.tensor(
            dec_df[[self.features.target]].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        cut_time = group[self.features.time].iloc[t]
        if hasattr(cut_time, "isoformat"):
            cut_time = cut_time.isoformat()
        else:
            cut_time = str(cut_time)

        return {
            "model_inputs": {
                "static_cats": static_categorial,
                "static_cont": static_continuous,
                "obs_cats": obs_categorial,
                "obs_cont": cont_observed,
                "known_cats": known_categorial,
                "known_cont": cont_known,
            },
            "target": target,
            "id": str(identifier),
            "cut_time": cut_time,
        }
    
    @staticmethod
    def get_scalers(dataset):
        """Compute scalers for each time series in using training set."""
        real_scalers = {}    # Dictionary mapping id to StandardScaler for input continuous features
        target_scalers = {}  # Dictionary mapping id to StandardScaler for target
        for identifier, sliced in dataset.df.groupby(dataset.features.id, sort=False):
            if len(sliced) < dataset.time_steps:
                continue
            real_scalers[identifier] = StandardScaler().fit(sliced[dataset.continuous_features].values)
            target_scalers[identifier] = StandardScaler().fit(sliced[[dataset.features.target]].values)
        
        return real_scalers, target_scalers
    

    def apply_scalers(
        self,
        real_scalers,
        target_scalers
    ):
        """Apply scalers to dataset."""
        df_list = []
        for identifier, group in self.df.groupby(self.features.id, sort=False):
            if identifier not in real_scalers or identifier not in target_scalers:
                continue
            group = group.copy()
            real_scaler = real_scalers[identifier]
            target_scaler = target_scalers[identifier]
            if self.continuous_features:
                group[self.continuous_features] = real_scaler.transform(
                    group[self.continuous_features].values
                )
            group[[self.features.target]] = target_scaler.transform(
                group[[self.features.target]].values
            )
            df_list.append(group)

        self.df = pd.concat(df_list, axis=0) if df_list else self.df.iloc[0:0].copy()
        # Rebuild samples after scaling
        self.build_samples()


# Test code
if __name__ == "__main__":
    # Create feature description for electricity dataset
    feature_description = FeatureDescription(
        id="categorical_id",
        time="date",
        target="power_usage",
        known_continuous=["hour", "day", "day_of_week", "month", "days_from_start", "hours_from_start","t"],
        known_categorical=["categorical_hour", "categorical_day_of_week"],
        static_categorical=["categorical_id"],
        static_continuous=[],
        observed_continuous=[],
        observed_categorical=[],
    )

    # Load dataset
    df = pd.read_csv("processed_power_usage.csv")
    
    # Split into train, val, test
    valid_boundary = 1315
    test_boundary  = 1339

    df_train = df[df["days_from_start"] < valid_boundary]
    df_val   = df[(df["days_from_start"] >= valid_boundary - 7) & (df["days_from_start"] < test_boundary)]
    df_test  = df[df["days_from_start"] >= test_boundary - 7]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        df=df_train,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24
    )
    # Get categorical encoder and scalers from training set
    categorical_encoder = train_dataset.categorical_encoder
    real_scalers, target_scalers = TimeSeriesDataset.get_scalers(train_dataset)

    val_dataset = TimeSeriesDataset(
        df=df_val,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24,
        categorical_encoder=categorical_encoder
    )

    test_dataset = TimeSeriesDataset(
        df=df_test,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24,
        categorical_encoder=categorical_encoder
    )

    # Apply scalers
    train_dataset.apply_scalers(real_scalers, target_scalers)
    val_dataset.apply_scalers(real_scalers, target_scalers)
    test_dataset.apply_scalers(real_scalers, target_scalers)

    dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(dl))

    params = {
        "encoder_length": train_dataset.enc_len,
        "decoder_length": train_dataset.dec_len,
        "time_steps": train_dataset.time_steps,
        "feature_description": feature_description,
        "embed_per_cat": train_dataset.get_embedding_per_cat(),
        "d_model": 64,
    }

    print("Debug")

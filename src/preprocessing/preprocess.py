from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SRC_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = SRC_DIR / "data" / "H2.csv"

def load_and_preprocess_classification():

    df = pd.read_csv(DATA_PATH)
    
    
    target = 'IsCanceled'
    
    
    leakage_cols = ['ReservationStatus', 'ReservationStatusDate', 'AssignedRoomType', 'BookingChanges']
    identifiers = ['Agent', 'Company']
    df = df.drop(columns=leakage_cols + identifiers)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
  
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return X_train_val, X_test, y_train_val, y_test, preprocessor



def load_and_preprocess_clustering():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[(df['ADR'] > 0) & (df['ADR'] < 1000)].reset_index(drop=True)

    numeric_cols = ['LeadTime', 'StaysInWeekendNights', 'StaysInWeekNights',
                    'Adults', 'Children', 'Babies', 'ADR',
                    'TotalOfSpecialRequests', 'RequiredCarParkingSpaces',
                    'PreviousCancellations']
    categorical_cols = ['MarketSegment', 'DistributionChannel',
                        'CustomerType', 'DepositType', 'Meal']

    df_raw = df[numeric_cols + categorical_cols].copy()

    df_raw[numeric_cols] = df_raw[numeric_cols].fillna(df_raw[numeric_cols].median())
    df_raw[categorical_cols] = df_raw[categorical_cols].fillna('Unknown')

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_raw[numeric_cols])
    X_num = pd.DataFrame(X_num, columns=numeric_cols)


    X_cat = pd.get_dummies(df_raw[categorical_cols], dtype=float)


    X = pd.concat([X_num, X_cat], axis=1).values

    return X, df_raw


def load_and_preprocess_regression():
    df = pd.read_csv(DATA_PATH)

    target = 'ADR'

    leakage_cols = ['ReservationStatus', 'ReservationStatusDate', 'IsCanceled']
    identifiers = ['Agent', 'Company']
    df = df.drop(columns=leakage_cols + identifiers)

    X = df.drop(columns=[target])
    y = df[target]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return X_train_val, X_test, y_train_val, y_test, preprocessor
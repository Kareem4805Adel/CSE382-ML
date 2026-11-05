import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_preprocess_classification():
    
    df = pd.read_csv('data/H2.csv')
    
    
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


def load_and_preprocess_regression():
    df = pd.read_csv('data/H2.csv')

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
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time


df = pd.read_csv("/content/drive/MyDrive/wine.csv", sep=";")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X = df.drop('quality', axis=1)
X_train = train_set.drop('quality', axis=1)
y_train = train_set['quality']
X_test = test_set.drop('quality', axis=1)
y_test = test_set['quality']


# Pipeline Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


simple_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])


start_time = time.time()
simple_pipeline.fit(X_train, y_train)
end_time = time.time()
simple_accuracy = simple_pipeline.score(X_test, y_test)
print(f"Time for fitting the simple pipeline: {end_time - start_time:.4f} seconds")
print(f"Accuracy without hyperparameter optimization: {simple_accuracy:.2f}")

# Define the pipeline with hyperparameter optimization
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}


random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)


start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()


print("Best Hyperparameters:", random_search.best_params_)
print(f"Time for fitting the pipeline with hyperparameter optimization: {end_time - start_time:.4f} seconds")

# Evaluate the model on the test set
optimized_accuracy = random_search.score(X_test, y_test)
print(f"Accuracy with hyperparameter optimization: {optimized_accuracy:.2f}")
import pandas as pd
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
import matplotlib.pyplot as plt

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


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])


start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
simple_accuracy = pipeline.score(X_test, y_test)
print(f"Time for fitting the simple pipeline: {end_time - start_time:.4f} seconds")
print(f"Accuracy without hyperparameter optimization: {simple_accuracy:.2f}")



param_grid = {
    # 'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200, 300, 400, 500, 600],
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'classifier__max_depth': [x for x in range(3, 16)],
    'classifier__min_samples_split': [x for x in range(2, 21)],
    'classifier__min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)


start_time_optimized = time.time()
random_search.fit(X_train, y_train)
end_time_optimized = time.time()


print("Best Hyperparameters:", random_search.best_params_)
print(f"Time for fitting the pipeline with hyperparameter optimization: {end_time_optimized - start_time_optimized:.4f} seconds")

# Evaluate the model on the test set
optimized_accuracy = random_search.score(X_test, y_test)
print(f"Accuracy with hyperparameter optimization: {optimized_accuracy:.2f}")

# Create a bar chart to visualize accuracy and training time comparison
labels = ['Simple Pipeline', 'Optimized Pipeline']
accuracy_scores = [simple_accuracy, optimized_accuracy]
training_times = [end_time - start_time, end_time_optimized - start_time_optimized]
print("Best Hyperparameters:", random_search.best_params_)
accuracy = random_search.score(X_test, y_test)
print(f"Accuracy with RandomSearch: {accuracy:.2f}")


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Pipeline')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(labels, accuracy_scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Training Time (seconds)', color=color)
ax2.plot(labels, training_times, marker='o', linestyle='-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
plt.title('Accuracy and Training Time Comparison')
plt.show()



# https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/
# https://www.youtube.com/watch?v=TLjMibCN6v4
# https://www.youtube.com/watch?v=w9IGkBfOoic
# https://towardsdatascience.com/step-by-step-tutorial-of-sci-kit-learn-pipeline-62402d5629b6

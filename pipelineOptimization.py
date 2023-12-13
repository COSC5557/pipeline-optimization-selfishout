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
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

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


# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', GradientBoostingClassifier())
# ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
simple_accuracy = pipeline.score(X_test, y_test)
print(f"Time for fitting the simple pipeline: {end_time - start_time:.4f} seconds")
print(f"Accuracy without hyperparameter optimization: {simple_accuracy:.2f}")


def objective(**params):
    classifier_type = params.pop('classifier_type')
    if classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(**params)
    elif classifier_type == 'KNeighbors':
        classifier = KNeighborsClassifier(**params)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return -np.mean(cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy'))



space = [
    Categorical(['RandomForest', 'KNeighbors', 'GradientBoosting']),  # Classifier type
    'preprocessor__pca__n_components': [None, 5, 10, 15, 20, 25, 30],
    'preprocessor__num__numeric__scaler': [StandardScaler(), PassthroughScaler()],
    'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'preprocessor__num__scaler__with_mean': [True, False],
    'preprocessor__num__scaler__with_std': [True, False],
    'preprocessing__categoricals__onehot__handle_unknown': ['ignore', 'error'],
    'classifier__n_estimators': [50, 100, 200, 300, 400, 500, 600],
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'classifier__max_depth': [x for x in range(3, 16)],
    'classifier__min_samples_split': [x for x in range(2, 21)],
    'classifier__min_samples_leaf': [x for x in range(2, 21)],
    'classifier__n_neighbors': [x for x in range(3, 21)],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__algorith': ['auto', 'ball_tree', 'kd_tree', 'brute']
]
# param_grid = {
#     # 'preprocessor__num__scaler__scaler': [StandardScaler(), None],
#     'preprocessor__pca__n_components': [None, 5, 10, 15, 20, 25, 30],
#     'preprocessor__num__numeric__scaler': [StandardScaler(), PassthroughScaler()],
#     'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
#     'preprocessor__num__scaler__with_mean': [True, False],
#     'preprocessor__num__scaler__with_std': [True, False],
#     # 'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant', 'mean'],
#     'preprocessing__categoricals__onehot__handle_unknown': ['ignore', 'error'],
#     'classifier__n_estimators': [50, 100, 200, 300, 400, 500, 600],
#     # 'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'classifier__max_depth': [x for x in range(3, 16)],
#     'classifier__min_samples_split': [x for x in range(2, 21)],
#     'classifier__min_samples_leaf': [1, 2, 4]
# }
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_fold_results = []


start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
simple_accuracy = pipeline.score(X_test, y_test)
print(f"Time for fitting the simple pipeline: {end_time - start_time:.4f} seconds")
print(f"Accuracy without hyperparameter optimization: {simple_accuracy:.2f}")



start_time_optimized = time.time()
for train_index, test_index in outer_cv.split(X):
    # Split data into training and testing based on current fold
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = df['quality'].iloc[train_index], df['quality'].iloc[test_index]

    
    bayes = gp_minimize(objective, space, n_calls=100, random_state=42)
    bayes.fit(X_train_fold, y_train_fold)

    
    fold_accuracy = bayes.score(X_test_fold, y_test_fold)
    outer_fold_results.append(fold_accuracy)


end_time_optimized = time.time()
for i, score in enumerate(outer_fold_results):
    print(f"Fold {i+1}: Accuracy = {score:.2f}")


average_performance = np.mean(outer_fold_results)
print(f"Average Accuracy: {average_performance:.2f}")



optimized_accuracy = bayes.score(X_test, y_test)
print(f"Accuracy with hyperparameter optimization: {optimized_accuracy:.2f}")


labels = ['Simple Pipeline', 'Optimized Pipeline']
accuracy_scores = [simple_accuracy, optimized_accuracy]
training_times = [end_time - start_time, end_time_optimized - start_time_optimized]
print("Best Hyperparameters:", bayes.best_params_)
accuracy = bayes.score(X_test, y_test)
print(f"Accuracy with Bayes: {accuracy:.2f}")


plt.figure(figsize=(8, 4))
plt.plot(range(1, len(outer_fold_results) + 1), outer_fold_results, label='Fold Accuracies', color='blue', marker='o')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Nested Cross-Validation Accuracies')
plt.ylim([0, 1])
plt.xticks(range(1, len(outer_fold_results) + 1))
plt.legend()
plt.show()



labels = ['Simple Pipeline', 'Optimized Pipeline']

accuracy_scores = [simple_accuracy, optimized_accuracy]
training_times = [end_time - start_time, end_time_optimized - start_time_optimized]
print("Best Hyperparameters:", bayes.best_params_)
accuracy = bayes_search.score(X_test, y_test)
print(f"Accuracy with Bayes: {accuracy:.2f}")


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

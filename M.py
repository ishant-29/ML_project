import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1. Read the dataset
try:
    df = pd.read_csv('Cloth_dataset.csv')
    print('✅ Dataset loaded successfully!')
except Exception as e:
    print(f'❌ Error loading dataset: {e}')
    exit(1)

# 2. Exploratory Data Analysis (Optional: Uncomment to visualize)
try:
    print(df.head())
    print(df.info())
    print(df['Item'].value_counts())
    print(df['Size'].value_counts())
except Exception as e:
    print(f'❌ Error in EDA: {e}')

# 3. Preprocess the Data
try:
    # Handle missing values
    numeric_cols = ['Weight(kg)', 'Height(cm)', 'Age']
    cat_cols = ['Brand', 'Item', 'Size']
    for cat in cat_cols:
        mode_val = df[cat].mode()
        if not mode_val.empty:
            df[cat] = df[cat].fillna(mode_val[0])
        else:
            df[cat] = df[cat].fillna('Unknown')
    for num in numeric_cols:
        df[num] = df[num].fillna(df[num].mean())
    print('✅ Missing values handled.')
except Exception as e:
    print(f'❌ Error handling missing values: {e}')
    exit(1)

# 4. Remove Outliers
try:
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lowerbound = Q1 - 1.5 * IQR
    upperbound = Q3 + 1.5 * IQR
    mask = ((df[numeric_cols] < lowerbound) | (df[numeric_cols] > upperbound)).any(axis=1)
    outliers = df[mask]
    df.drop(index=outliers.index, inplace=True)
    print(f'✅ Outliers removed. Remaining data shape: {df.shape}')
except Exception as e:
    print(f'❌ Error removing outliers: {e}')
    exit(1)

# 5. Label Encoding
try:
    le_brand = LabelEncoder()
    df['Brand'] = le_brand.fit_transform(df['Brand'].astype(str))
    le_item = LabelEncoder()
    df['Item'] = le_item.fit_transform(df['Item'].astype(str))
    mapping = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5, 'XXXL': 6}
    df['Size'] = df['Size'].astype(str).map(mapping)
    print('✅ Label encoding done.')
except Exception as e:
    print(f'❌ Error in label encoding: {e}')
    exit(1)

# 6. Feature Scaling
try:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print('✅ Feature scaling done.')
except Exception as e:
    print(f'❌ Error in feature scaling: {e}')
    exit(1)

# 7. Feature Selection
try:
    x = df.drop('Size', axis=1)
    y = df['Size']
    sm = SMOTE(random_state=42, sampling_strategy='auto')
    x_res, y_res = sm.fit_resample(x, y)
    selector = SelectKBest(score_func=f_classif, k=2)
    x_selected = selector.fit_transform(x_res, y_res)
    print('✅ Feature selection and SMOTE done.')
except Exception as e:
    print(f'❌ Error in feature selection or SMOTE: {e}')
    exit(1)

# 8. Train-Test Split
try:
    xtrain, xtest, ytrain, ytest = train_test_split(x_selected, y_res, test_size=0.2, random_state=42, stratify=y_res)
    print(f'Training shape: {xtrain.shape}, Testing shape: {xtest.shape}')
except Exception as e:
    print(f'❌ Error in train-test split: {e}')
    exit(1)

# 9. Model Training and Evaluation
try:
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    print('LogisticRegression Training Accuracy:', lr.score(xtrain, ytrain))
    print('LogisticRegression Testing Accuracy:', lr.score(xtest, ytest))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(xtrain, ytrain)
    print('Naive Bayes Training Accuracy:', nb.score(xtrain, ytrain))
    print('Naive Bayes Testing Accuracy:', nb.score(xtest, ytest))

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(xtrain, ytrain)
    print('Decision Tree Training Accuracy:', dt.score(xtrain, ytrain))
    print('Decision Tree Testing Accuracy:', dt.score(xtest, ytest))

    # Random Forest
    rf = RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42)
    rf.fit(xtrain, ytrain)
    print('Random Forest Training Accuracy:', rf.score(xtrain, ytrain))
    print('Random Forest Testing Accuracy:', rf.score(xtest, ytest))

    # Logistic Regression Grid Search
    lr_param_grid = {
        'C': [0.1, 1, 10, 50, 100, 500],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [500, 1000],
    }
    lr_grid = GridSearchCV(estimator=LogisticRegression(random_state=42), param_grid=lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(xtrain, ytrain)
    print('Best Logistic Regression Parameters:', lr_grid.best_params_)
    print('Logistic Regression Train Accuracy:', lr_grid.best_estimator_.score(xtrain, ytrain))
    print('Logistic Regression Test Accuracy:', lr_grid.best_estimator_.score(xtest, ytest))

    # SVM Grid Search
    svm_param_grid = {
        'C': [0.1, 1, 10, 50, 100, 500, 1000],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced'],
        'probability': [True]
    }
    svm_grid = GridSearchCV(estimator=SVC(random_state=42), param_grid=svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(xtrain, ytrain)
    print('Best SVM Parameters:', svm_grid.best_params_)
    print('SVM Train Accuracy:', svm_grid.best_estimator_.score(xtrain, ytrain))
    print('SVM Test Accuracy:', svm_grid.best_estimator_.score(xtest, ytest))

    # Final Evaluation
    y_pred = svm_grid.predict(xtest)
    print('Test Accuracy:', accuracy_score(ytest, y_pred))
    print('\nClassification Report:')
    print(classification_report(ytest, y_pred))
    print('\nConfusion Matrix:')
    print(confusion_matrix(ytest, y_pred))
    print('✅ Model training and evaluation complete.')
except Exception as e:
    print(f'❌ Error in model training or evaluation: {e}')
    exit(1)

# 10. Save the Model and Preprocessing Objects
try:
    joblib.dump(svm_grid, 'model.pkl')
    joblib.dump(le_brand, 'brand_encoder.pkl')
    joblib.dump(le_item, 'item_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selector, 'selector.pkl')
    print('✅ Model and all encoders saved successfully!')
except Exception as e:
    print(f'❌ Error saving model or encoders: {e}')
    exit(1) 
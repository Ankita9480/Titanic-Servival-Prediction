import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    df = pd.read_csv("/home/ankita/Desktop/AIML/Titanic Survival Prediction/dataset/try.csv")
    print("Successfully loaded dataset.")
except FileNotFoundError:
    print("Error: File not found.")

def dataCleaningAndEngineering(df):
    print("---------- Preprocessing & Feature Engineering ----------\n")
    
    # 1. Fill missing Embarked with Mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 2. Cabin flag
    df['has_cabin'] = df['Cabin'].notnull().astype(int)

    # 3. Extract 'Title' from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Rare titles ko group karke 'Rare' mark kar do
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 4. Impute Age using median grouped by Pclass & Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

    # 5. Family Size & Is Alone features
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # 6. Drop redundant identifier columns
    dfCleaned = df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])

    print("Missing values after processing:\n", dfCleaned.isnull().sum())
    return dfCleaned

def dataAnalysis(df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Survival by Sex
    sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0,0], palette='Set1', ci=None)
    axes[0,0].set_title('Survival Rate by Gender')

    # 2. Survival by Class
    sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[0,1], palette='Set2', ci=None)
    axes[0,1].set_title('Survival Rate by Passenger Class')

    # 3. Survival by Family Size / Is Alone
    sns.barplot(x='family_size', y='Survived', data=df, ax=axes[1,0], palette='Set3', ci=None)
    axes[1,0].set_title('Survival Rate by Family Size')

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('/home/ankita/Desktop/AIML/Titanic Survival Prediction/images/eda_plots.png')
    print("\nEDA plots saved to eda_plots.png successfully!")

def trainLogisticRegression(df):
    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

    X = df_encoded.drop(columns=['Survived'])
    y = df_encoded['Survived']

    # Stratified Train-Test Split (as per handbook requirements)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model Training: Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    
    print("\n================ MODEL EVALUATION (Logistic Regression) ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Feature Importance / Coefficients Analysis
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    print("\n================ KEY SURVIVAL FACTORS (Coefficients) ================")
    print(coef_df.to_string(index=False))

    return model, X.columns

# Execution
df_cleaned = dataCleaningAndEngineering(df)
dataAnalysis(df_cleaned)
model, feature_cols = trainLogisticRegression(df_cleaned)
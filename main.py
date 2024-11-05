import pandas as pd

# Load the dataset
file_path = 'credit_risk_dataset.csv'
data = pd.read_csv(file_path)

# Display basic info and a few rows of the data for initial exploration
data_info = data.info()
data_head = data.head()
data_nulls = data.isnull().sum()  # Check for null values in each column

data_info, data_head, data_nulls

print(data_info)
print(data_head)
print(data_nulls)

# Step 1: Fill missing values
# Fill 'person_emp_length' and 'loan_int_rate' with median values for simplicity

data['person_emp_length'].fillna(data['person_emp_length'].median(), inplace=True)
data['loan_int_rate'].fillna(data['loan_int_rate'].median(), inplace=True)

# Step 2: Encoding categorical variables
# Using pandas get_dummies for simplicity in handling categorical columns

data_encoded = pd.get_dummies(data, columns=[
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
])

# Step 3: Checking for outliers in 'person_age' and 'person_income' using summary statistics
outliers_summary = {
    "person_age": data_encoded['person_age'].describe(),
    "person_income": data_encoded['person_income'].describe()
}

# Display the cleaned and encoded data, as well as the summary of outliers
data_encoded.head(), outliers_summary


# Step 1: Capping outliers for 'person_age' and 'person_income'

# Cap 'person_age' at 100
data_encoded['person_age'] = data_encoded['person_age'].apply(lambda x: min(x, 100))

# Cap 'person_income' at a reasonable threshold, e.g., $300,000
data_encoded['person_income'] = data_encoded['person_income'].apply(lambda x: min(x, 300000))

# Verifying the adjustment
outliers_summary_adjusted = {
    "person_age": data_encoded['person_age'].describe(),
    "person_income": data_encoded['person_income'].describe()
}

outliers_summary_adjusted


import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style
sns.set(style="whitegrid")

# Separate data for "loan_status" (target variable) where 1 = bad payers and 0 = good payers
good_payers = data_encoded[data_encoded['loan_status'] == 0]
bad_payers = data_encoded[data_encoded['loan_status'] == 1]

# Plot 1: Age Distribution by Loan Status
plt.figure(figsize=(14, 6))
sns.histplot(data=data_encoded, x='person_age', hue='loan_status', bins=30, kde=True, palette='viridis')
plt.title('Distribuição de Idade por Status do Empréstimo (Bons vs Maus Pagadores)')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.legend(title='Status do Empréstimo', labels=['Bom Pagador', 'Mau Pagador'])
plt.show()

# Plot 2: Income Distribution by Loan Status
plt.figure(figsize=(14, 6))
sns.histplot(data=data_encoded, x='person_income', hue='loan_status', bins=30, kde=True, palette='plasma')
plt.title('Distribuição de Renda por Status do Empréstimo (Bons vs Maus Pagadores)')
plt.xlabel('Renda Anual')
plt.ylabel('Contagem')
plt.legend(title='Status do Empréstimo', labels=['Bom Pagador', 'Mau Pagador'])
plt.show()

# Plot 3: Loan Interest Rate by Loan Status
plt.figure(figsize=(14, 6))
sns.kdeplot(data=good_payers['loan_int_rate'], label='Bom Pagador', fill=True, color='blue', alpha=0.4)
sns.kdeplot(data=bad_payers['loan_int_rate'], label='Mau Pagador', fill=True, color='red', alpha=0.4)
plt.title('Taxa de Juros do Empréstimo por Status do Empréstimo')
plt.xlabel('Taxa de Juros (%)')
plt.ylabel('Densidade')
plt.legend(title='Status do Empréstimo')
plt.show()

# Plot 4: Loan Intent by Loan Status
plt.figure(figsize=(12, 6))
sns.countplot(data=data_encoded, x='loan_intent', hue='loan_status', palette='coolwarm')
plt.title('Intenção de Empréstimo por Status (Bons vs Maus Pagadores)')
plt.xlabel('Intenção do Empréstimo')
plt.ylabel('Contagem')
plt.legend(title='Status do Empréstimo', labels=['Bom Pagador', 'Mau Pagador'])
plt.show()


import numpy as np

# Criação de novas variáveis para análise de crédito
data_encoded['income_to_loan_ratio'] = data_encoded['person_income'] / (data_encoded['loan_amnt'] + 1)  # +1 to avoid division by zero
data_encoded['emp_length_ratio'] = data_encoded['person_emp_length'] / (data_encoded['person_age'] + 1)

# Transformações logarítmicas para variáveis com grande variância
data_encoded['log_person_income'] = np.log1p(data_encoded['person_income'])
data_encoded['log_loan_amnt'] = np.log1p(data_encoded['loan_amnt'])

# Interações entre variáveis
data_encoded['interest_income_ratio'] = data_encoded['loan_int_rate'] * data_encoded['loan_percent_income']

# Visualizar as novas colunas adicionadas
data_encoded[['income_to_loan_ratio', 'emp_length_ratio', 'log_person_income', 'log_loan_amnt', 'interest_income_ratio']].head()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Dividir os dados em conjuntos de treinamento e teste
X = data_encoded.drop(columns=['loan_status'])  # Assumindo 'loan_status' como nossa variável alvo
y = data_encoded['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos para avaliação
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

# Função para treinar e avaliar os modelos
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in models.items():
        # Treinamento e avaliação do modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Avaliar desempenho
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'Accuracy': accuracy,
            'Classification Report': classification_report(y_test, y_pred)
        }
        
    return results

# Executar a função para avaliar modelos
results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

# Ajuste de Hiperparâmetros (Exemplo para Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Melhores Hiperparâmetros Random Forest:", grid_search.best_params_)


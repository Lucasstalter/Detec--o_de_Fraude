# Projeto de Detecção de Fraude 🔍

Sistema de Machine Learning para detecção de transações fraudulentas.


## 🚀 Setup Rápido

### 1. Criar estrutura de pastas

**Windows (PowerShell):**
```powershell
mkdir fraud_detection
cd fraud_detection
mkdir data, data/raw, data/processed, notebooks, src, src/data, src/models, src/features, models, reports
```

**Linux/Mac:**
```bash
mkdir -p fraud_detection/{data/{raw,processed},notebooks,src/{data,models,features},models,reports}
cd fraud_detection
```

### 2. Criar ambiente virtual

```bash
python -m venv venv

# Ativar:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instalar bibliotecas

Salve o arquivo `requirements.txt` e execute:
```bash
pip install -r requirements.txt
```

## 📊 Datasets Recomendados

### Opção 1: Credit Card Fraud (Kaggle) ⭐ RECOMENDADO
- 284,807 transações
- 492 fraudes (0.172%)
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Baixe e coloque em `data/raw/creditcard.csv`

### Opção 2: PaySim
- Dados sintéticos
- Mais balanceado
- Link: https://www.kaggle.com/datasets/ealaxi/paysim1


## 📈 Métricas Importantes

Para detecção de fraude, foque em:
- **Recall**: % de fraudes capturadas (minimizar falsos negativos!)
- **Precision**: % de alertas corretos
- **F1-Score**: Balanço entre os dois
- **PR-AUC**: Melhor que ROC-AUC para dados desbalanceados

## 💻 Começando

1. Baixe o dataset Credit Card Fraud
2. Coloque em `data/raw/`
3. Abra Jupyter: `jupyter notebook`
4. Crie um novo notebook em `notebooks/01_eda.ipynb`

## 🔧 Primeiro Código

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
df = pd.read_csv('data/raw/creditcard.csv')

# Visão geral
print(f"Shape: {df.shape}")
print(f"\nColunas: {df.columns.tolist()}")
print(f"\nInfo:")
df.info()

# Distribuição de classes
print(f"\nDistribuição de fraudes:")
print(df['Class'].value_counts())
print(f"\nPercentual de fraudes: {df['Class'].mean()*100:.3f}%")

# Visualização
df['Class'].value_counts().plot(kind='bar')
plt.title('Distribuição de Classes')
plt.xlabel('Classe (0=Normal, 1=Fraude)')
plt.ylabel('Quantidade')
plt.show()
```

## ⚠️ Dicas Importantes

1. **Validação Temporal**: Para fraude, NÃO embaralhe os dados aleatoriamente
2. **Class Imbalance**: Use SMOTE ou ajuste class_weight nos modelos
3. **Threshold Tuning**: O threshold padrão (0.5) pode não ser ideal
4. **Feature Importance**: Use SHAP para entender decisões do modelo

## 📚 Recursos

- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [SHAP](https://shap.readthedocs.io/)

## 🆘 Problemas Comuns

**Erro ao importar módulos:**
```bash
# Certifique que está no venv
pip install -r requirements.txt
```

**Dataset muito grande:**
```python
# Carregar apenas parte dos dados
df = pd.read_csv('creditcard.csv', nrows=50000)
```

**Jupyter não encontra bibliotecas:**
```bash
pip install ipykernel
python -m ipykernel install --user --name=fraud_env
```


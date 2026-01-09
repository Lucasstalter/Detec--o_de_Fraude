# 🔍 Sistema de Detecção de Fraude com Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)

Sistema de Machine Learning para detecção de transações fraudulentas em cartões de crédito com **94%+ de precisão**.


</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Resultados](#-resultados)
- [Tecnologias](#-tecnologias)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Usar](#-como-usar)
- [Metodologia](#-metodologia)
- [Autor](#-autor)

---

## 🎯 Sobre o Projeto

Sistema completo de detecção de fraude utilizando técnicas avançadas de Machine Learning para identificar transações fraudulentas em tempo real.

### 🌟 Destaques

- ✅ **Alta Precisão**: 94.2% de precision, 87.5% de recall
- ✅ **Feature Engineering Avançado**: 40+ features criadas
- ✅ **Dashboard Interativo**: Interface web com Streamlit
- ✅ **Múltiplos Modelos**: XGBoost, LightGBM, Random Forest
- ✅ **Tratamento de Desbalanceamento**: SMOTE, class weights
- ✅ **Produção Ready**: Modelo salvo e pronto para deploy

### 📊 Dataset

- **Fonte**: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Tamanho**: 284,807 transações
- **Features**: 28 features (resultado de PCA) + Time + Amount
- **Taxa de Fraude**: 0.172% (dados altamente desbalanceados)

---

## 📈 Resultados

### Métricas do Modelo Final (XGBoost)

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **Precision** | 94.2% | Dos alertas de fraude, 94% são corretos |
| **Recall** | 87.5% | Das fraudes reais, 87% são detectadas |
| **F1-Score** | 90.7% | Média harmônica entre Precision e Recall |
| **ROC-AUC** | 0.978 | Excelente capacidade de discriminação |
| **PR-AUC** | 0.951 | Ideal para dados desbalanceados |

### 📊 Comparação de Modelos

```
┌─────────────────────┬───────────┬────────┬──────────┬─────────┐
│ Modelo              │ Precision │ Recall │ F1-Score │ ROC-AUC │
├─────────────────────┼───────────┼────────┼──────────┼─────────┤
│ Logistic Regression │   89.3%   │ 78.2%  │  83.4%   │  0.912  │
│ Random Forest       │   91.7%   │ 84.1%  │  87.7%   │  0.965  │
│ XGBoost ⭐          │   94.2%   │ 87.5%  │  90.7%   │  0.978  │
│ LightGBM            │   93.1%   │ 86.3%  │  89.6%   │  0.971  │
└─────────────────────┴───────────┴────────┴──────────┴─────────┘
```

---

## 🛠️ Tecnologias

### Core

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Machine Learning

- **XGBoost** - Modelo principal
- **LightGBM** - Alternativa rápida
- **Random Forest** - Ensemble
- **Imbalanced-learn** - SMOTE e técnicas de balanceamento

### Visualização

- **Streamlit** - Dashboard interativo
- **Plotly** - Gráficos interativos
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Gráficos estatísticos

### DevOps

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📁 Estrutura do Projeto

```
fraud_detection/
│
├── 📊 app.py                          # Dashboard Streamlit
│
├── 📁 data/
│   ├── raw/                           # Dados originais (não versionados)
│   │   └── creditcard.csv
│   └── processed/                     # Dados processados
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                   # Análise Exploratória
│   ├── 02_feature_engineering.ipynb   # Engenharia de Features
│   └── 03_modeling.ipynb              # Modelagem e Avaliação
│
├── 🤖 models/
│   ├── xgboost_fraud_detector.pkl     # Modelo treinado
│   ├── scaler.pkl                     # Scaler para normalização
│   └── threshold_config.json          # Configurações de threshold
│
├── 💻 src/
│   ├── data/
│   │   └── data_loader.py             # Carregamento de dados
│   ├── features/
│   │   └── feature_engineering.py     # Criação de features
│   ├── models/
│   │   └── train.py                   # Treinamento
│   └── visualization/
│       └── plots.py                   # Visualizações
│
├── 📋 requirements.txt                 # Dependências
├── 📖 README.md                        # Este arquivo
└── 🚫 .gitignore                       # Arquivos ignorados
```

---

## 🚀 Como Usar

### 1️⃣ Pré-requisitos

- Python 3.11+
- pip
- Git

### 2️⃣ Instalação

```bash
# Clonar repositório
git clone https://github.com/Lucasstalter/Detec--o_de_Fraude.git
cd Detec--o_de_Fraude

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3️⃣ Obter Dataset

1. Baixe o dataset: [Kaggle - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Extraia o arquivo `creditcard.csv`
3. Coloque em: `data/raw/creditcard.csv`

### 4️⃣ Executar Notebooks (Ordem)

```bash
# Iniciar Jupyter
jupyter notebook

# Executar notebooks na ordem:
# 1. notebooks/01_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_modeling.ipynb
```

### 5️⃣ Rodar Dashboard

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente em: `http://localhost:8501`

---

## 🔬 Metodologia

### 1. Análise Exploratória (EDA)

- ✅ Análise de distribuições
- ✅ Identificação de outliers
- ✅ Análise de correlações
- ✅ Visualização de padrões temporais
- ✅ Estudo do desbalanceamento (0.172% fraudes)

### 2. Feature Engineering

#### Features Temporais
```python
Hour = (Time / 3600) % 24              # Hora do dia
Day = (Time / 86400)                   # Dia desde início
Time_Period = categorize_by_period()   # Manhã, Tarde, Noite
```

#### Features de Amount
```python
Amount_Log = log1p(Amount)             # Transformação log
Amount_Sqrt = sqrt(Amount)             # Raiz quadrada
Amount_Bin = categorize(Amount)        # Categorização
```

#### Features Estatísticas
```python
V_mean = mean(V1...V28)                # Média das V features
V_std = std(V1...V28)                  # Desvio padrão
V_range = max(V1...V28) - min(V1...V28)  # Range
V_mad = mean_absolute_deviation()      # MAD
```

**Total**: 47 features (28 originais + 19 engineered)

### 3. Tratamento de Desbalanceamento

- ⚖️ **Class Weights**: Penalizar mais fraudes não detectadas
- 🔄 **SMOTE**: Synthetic Minority Over-sampling (testado)
- 🎯 **Threshold Tuning**: Ajuste fino do limiar de decisão
- ⏱️ **Validação Temporal**: Split temporal (não aleatório)

### 4. Modelagem

#### Modelos Testados
1. **Logistic Regression** - Baseline
2. **Random Forest** - Ensemble
3. **XGBoost** ⭐ - Melhor resultado
4. **LightGBM** - Alternativa rápida

#### Hiperparâmetros (XGBoost)
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': 577,  # Balancear classes
    'eval_metric': 'logloss'
}
```

### 5. Avaliação

**Métricas Principais:**
- ✅ **Precision**: Evitar falsos positivos
- ✅ **Recall**: Capturar máximo de fraudes
- ✅ **F1-Score**: Balanço entre Precision e Recall
- ✅ **PR-AUC**: Melhor para dados desbalanceados

---

## 🎨 Dashboard Interativo

### Funcionalidades

#### 🏠 **Home**
- Visão geral do sistema
- Métricas principais em cards
- Estatísticas atualizadas

#### 📊 **Análise de Dados**
- Exploração interativa do dataset
- Distribuições por feature
- Matriz de correlação
- Box plots comparativos

#### 🤖 **Predição** ⭐
- Upload de CSV
- Feature engineering automático
- Ajuste de threshold em tempo real
- Download de resultados
- Visualizações das predições

#### 📈 **Métricas do Modelo**
- Confusion Matrix interativa
- Top 20 features mais importantes
- Análise de threshold
- Curvas ROC e Precision-Recall

#### ℹ️ **Sobre**
- Documentação completa
- Metodologia detalhada
- Como usar
- Informações do autor

---



## 👨‍💻 Autor

<div align="center">

### Lucas Stalter


**Data Scientist | Machine Learning Engineer**

</div>




---

<div align="center">

**Desenvolvido por Lucas Stalter**


</div>

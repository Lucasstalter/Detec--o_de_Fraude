import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Carrega e valida datasets de detecção de fraude"""
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Caminho para o arquivo CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Carrega o dataset"""
        try:
            logger.info(f"Carregando {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"✓ Carregado: {self.df.shape[0]:,} linhas, {self.df.shape[1]} colunas")
            return self.df
        except Exception as e:
            logger.error(f"✗ Erro: {e}")
            raise
    
    def get_fraud_stats(self):
        """Retorna estatísticas de fraude"""
        if self.df is None:
            raise ValueError("Execute load_data() primeiro")
        
        # Detectar coluna target
        target_col = 'Class' if 'Class' in self.df.columns else 'isFraud'
        
        total = len(self.df)
        fraud_count = self.df[target_col].sum()
        fraud_pct = (fraud_count / total) * 100
        
        stats = {
            'total_transactions': total,
            'fraud_count': int(fraud_count),
            'normal_count': int(total - fraud_count),
            'fraud_percentage': fraud_pct,
            'imbalance_ratio': (total - fraud_count) / fraud_count
        }
        
        return stats
    
    def check_missing(self):
        """Verifica valores ausentes"""
        if self.df is None:
            raise ValueError("Execute load_data() primeiro")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        result = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        })
        
        return result[result['Missing_Count'] > 0]
    
    def get_summary(self):
        """Resumo completo dos dados"""
        if self.df is None:
            raise ValueError("Execute load_data() primeiro")
        
        print("=" * 60)
        print("RESUMO DO DATASET")
        print("=" * 60)
        
        print(f"\n📊 Dimensões: {self.df.shape}")
        print(f"   Linhas: {self.df.shape[0]:,}")
        print(f"   Colunas: {self.df.shape[1]}")
        
        print(f"\n📋 Colunas:")
        for col in self.df.columns:
            print(f"   - {col} ({self.df[col].dtype})")
        
        stats = self.get_fraud_stats()
        print(f"\n🎯 Distribuição de Fraudes:")
        print(f"   Total de transações: {stats['total_transactions']:,}")
        print(f"   Transações normais: {stats['normal_count']:,}")
        print(f"   Fraudes: {stats['fraud_count']:,}")
        print(f"   Percentual de fraude: {stats['fraud_percentage']:.3f}%")
        print(f"   Ratio (Normal:Fraude): 1:{stats['imbalance_ratio']:.0f}")
        
        missing = self.check_missing()
        if len(missing) > 0:
            print(f"\n⚠️  Valores Ausentes:")
            print(missing)
        else:
            print(f"\n✓ Sem valores ausentes")
        
        print("\n" + "=" * 60)


def create_sample_data(n_samples=10000, fraud_ratio=0.01, save_path=None):
    """
    Cria dados sintéticos para teste
    
    Args:
        n_samples: Total de amostras
        fraud_ratio: Proporção de fraudes
        save_path: Onde salvar (opcional)
    
    Returns:
        DataFrame com dados sintéticos
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Criar features
    df = pd.DataFrame({
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.concatenate([
            np.random.exponential(50, n_normal),
            np.random.exponential(200, n_fraud)
        ]),
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
        'V3': np.random.randn(n_samples),
        'V4': np.random.randn(n_samples),
        'Class': np.concatenate([
            np.zeros(n_normal),
            np.ones(n_fraud)
        ])
    })
    
    # Embaralhar
    df = df.sample(frac=1).reset_index(drop=True)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Dados salvos em {save_path}")
    
    return df


# Exemplo de uso
if __name__ == "__main__":
    print("🧪 Testando DataLoader\n")
    
    # Criar dados de exemplo
    print("Criando dados sintéticos...")
    df = create_sample_data(n_samples=5000, fraud_ratio=0.02)
    df.to_csv('sample_fraud.csv', index=False)
    
    # Carregar com DataLoader
    loader = FraudDataLoader('sample_fraud.csv')
    loader.load_data()
    loader.get_summary()
    
    print("\n✓ Módulo funcionando!")

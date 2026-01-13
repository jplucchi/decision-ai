"""
DECISION AI - Treino FINAL CORRETO
POSTECH Datathon 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, 
                            recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("DECISION AI - Iniciando treino\n")

# ============================================================================
# 1. CARREGAR E PROCESSAR DADOS
# ============================================================================
print("[1/5] Carregando dados...")

with open('data/vagas.json') as f:
    vagas = json.load(f)
with open('data/prospects.json') as f:
    prospects_dict = json.load(f)
with open('data/applicants.json') as f:
    applicants = json.load(f)

print(f"  Vagas: {len(vagas):,}")
print(f"  Jobs com prospects: {len(prospects_dict):,}")
print(f"  Applicants: {len(applicants):,}")

# ============================================================================
# 2. DESDOBRAR PROSPECTS (job -> lista de candidatos)
# ============================================================================
print("\n[2/5] Processando prospects...")

data_list = []
for job_id, job_data in prospects_dict.items():
    job_id_int = int(job_id)
    prospects_list = job_data.get('prospects', [])
    
    for prospect in prospects_list:
        try:
            # Extrair código do candidato
            codigo = prospect.get('codigo', '0')
            applicant_id = int(codigo) if codigo and codigo != '' else 0
            
            # Verificar se foi contratado (comentário contém "contratado")
            comentario = str(prospect.get('comentario', '')).lower()
            situacao = str(prospect.get('situacao_candidado', '')).lower()
            is_hired = 1 if ('contratado' in comentario or 'contratado' in situacao) else 0
            
            if applicant_id > 0:
                data_list.append({
                    'job_id': job_id_int,
                    'applicant_id': applicant_id,
                    'is_hired': is_hired,
                    'data_candidatura': prospect.get('data_candidatura', ''),
                    'recrutador': prospect.get('recrutador', '')
                })
        except:
            continue

df = pd.DataFrame(data_list)

print(f"  Total de candidaturas: {len(df):,}")
print(f"  Candidatos únicos: {df['applicant_id'].nunique():,}")
print(f"  Vagas com candidatos: {df['job_id'].nunique():,}")
print(f"  Taxa de contratação: {df['is_hired'].mean():.2%}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/5] Criando features...")

# Comportamentais
df['total_aplicacoes'] = df.groupby('applicant_id')['applicant_id'].transform('count')
df['total_contratacoes'] = df.groupby('applicant_id')['is_hired'].transform('sum')
df['taxa_sucesso'] = (df['total_contratacoes'] / df['total_aplicacoes']).fillna(0)
df['ordem_aplicacao'] = df.groupby('job_id').cumcount() + 1

# CVs dos candidatos
cv_dict = {}
for k, v in applicants.items():
    try:
        cv_dict[int(k)] = str(v.get('cv_pt', ''))
    except:
        continue

df['cv'] = df['applicant_id'].map(cv_dict).fillna('')
df['cv_tamanho'] = df['cv'].str.len()

# Skills
skills = ['python', 'java', 'sql', 'sap']
for skill in skills:
    df[f'has_{skill}'] = df['cv'].str.lower().str.contains(skill, na=False).astype(int)

# Limpar
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

# Features finais
features = ['total_aplicacoes', 'taxa_sucesso', 'ordem_aplicacao', 
            'cv_tamanho'] + [f'has_{skill}' for skill in skills]

X = df[features]
y = df['is_hired']

print(f"  Features: {len(features)}")
print(f"  Registros: {len(X):,}")
print(f"  Positivos: {y.sum():,} ({y.mean():.2%})")

if len(X) == 0 or y.sum() == 0:
    print("\n❌ ERRO: Não há dados suficientes para treinar!")
    print("Verifique se os arquivos JSON estão corretos.")
    exit(1)

# ============================================================================
# 4. SPLIT E TREINO
# ============================================================================
print("\n[4/5] Treinando modelo...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train):,} ({y_train.mean():.2%} positivos)")
print(f"  Test: {len(X_test):,} ({y_test.mean():.2%} positivos)")

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Após SMOTE: {len(X_train_bal):,} (50% positivos)")

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Modelo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_bal)
print("  ✓ Modelo treinado")

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train_bal,
                            cv=cv, scoring='f1', n_jobs=-1)
print(f"  ✓ CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# 5. AVALIAR
# ============================================================================
print("\n[5/5] Avaliação final...")

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"Precision: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-Score:  {f1:.2%}")
print(f"ROC-AUC:   {auc:.2%}")
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:,}   FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}   TP: {cm[1,1]:,}")

# Salvar
import os
os.makedirs('models', exist_ok=True)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

data = {
    'X_test': X_test_scaled,
    'y_test': y_test.values,
    'y_pred': y_pred,
    'y_proba': y_proba,
    'metrics': {'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc},
    'confusion_matrix': cm,
    'features': features
}

with open('models/results.pkl', 'wb') as f:
    pickle.dump(data, f)

print("\n✅ Arquivos salvos em models/")
print("="*60)
print("CONCLUÍDO! Rode: streamlit run app.py")
print("="*60)
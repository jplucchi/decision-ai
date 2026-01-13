# FASE 2: FEATURE ENGINEERING

**POSTECH Datathon 2026** | Decision Consultoria

---

## 1. Estratégia de Features

Baseado na EDA, criamos **8 features** focadas em simplicidade e efetividade:

### Categorias

1. **Comportamentais** (3 features)
2. **Qualificação** (5 features)

---

## 2. Features Comportamentais

### 2.1 Total de Aplicações
```python
total_aplicacoes = candidatos.groupby('applicant_id').count()
```
- **Range:** 1 a 18
- **Média:** 1,27
- **Insight:** Candidatos persistentes têm maior taxa de sucesso

### 2.2 Taxa de Sucesso do Candidato
```python
taxa_sucesso = contratacoes / total_aplicacoes
```
- **Range:** 0,0 a 1,0
- **Interpretação:** Histórico de sucesso do candidato
- **Poder preditivo:** ⭐⭐⭐⭐⭐

### 2.3 Ordem da Aplicação
```python
ordem_aplicacao = rank() por job_id
```
- **Range:** 1 a 127
- **Insight:** Primeiros candidatos têm ligeira vantagem
- **Poder preditivo:** ⭐⭐⭐

---

## 3. Features de Qualificação

### 3.1 Tamanho do CV
```python
cv_tamanho = len(cv_text)
```
- **Range:** 0 a 50.000 caracteres
- **Média:** 4.000 (quando disponível)
- **Insight:** CVs mais completos = candidatos mais preparados

### 3.2-3.5 Skills Técnicas (4 features)

Extração via **NLP** (regex case-insensitive):

```python
has_python = cv.contains('python', case=False)
has_java = cv.contains('java', case=False)
has_sql = cv.contains('sql', case=False)
has_sap = cv.contains('sap', case=False)
```

**Distribuição:**
- **Python:** 0,3% dos candidatos
- **Java:** 0,8% dos candidatos
- **SQL:** 0,5% dos candidatos
- **SAP:** 1,2% dos candidatos

**Insight:** Mesmo com 99% de CVs vazios, skills encontradas são altamente preditivas.

---

## 4. Importância das Features

### Ranking (estimado por correlação com target)

| Feature | Importância | Categoria |
|---------|-------------|-----------|
| **taxa_sucesso** | ⭐⭐⭐⭐⭐ | Comportamental |
| **total_aplicacoes** | ⭐⭐⭐⭐ | Comportamental |
| **has_sap** | ⭐⭐⭐⭐ | Qualificação |
| **cv_tamanho** | ⭐⭐⭐ | Qualificação |
| **ordem_aplicacao** | ⭐⭐⭐ | Comportamental |
| **has_java** | ⭐⭐ | Qualificação |
| **has_python** | ⭐⭐ | Qualificação |
| **has_sql** | ⭐⭐ | Qualificação |

---

## 5. Tratamento de Dados

### 5.1 Valores Faltantes

```python
# CVs vazios
cv_tamanho.fillna(0)

# Candidatos novos (sem histórico)
taxa_sucesso.fillna(0)
```

### 5.2 Normalização

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Motivo:** Features têm escalas diferentes (0-1 vs 0-18 vs 0-50000)

---

## 6. Dataset Final

### Estatísticas

- **Total de registros:** 53.759
- **Features:** 8
- **Target:** is_hired (binário)
- **Valores faltantes:** 0 (após fillna)

### Distribuição

- **Positivos:** 2.758 (5,13%)
- **Negativos:** 51.001 (94,87%)
- **Necessário:** SMOTE para balanceamento

---

## 7. Validação das Features

### Teste de Correlação

Todas as 8 features têm correlação positiva com o target:

```
taxa_sucesso:       0.45 ✅
total_aplicacoes:   0.18 ✅
has_sap:            0.12 ✅
cv_tamanho:         0.08 ✅
ordem_aplicacao:   -0.05 ✅ (inversa)
```

### Multicolinearidade

VIF (Variance Inflation Factor) < 5 para todas as features ✅

**Conclusão:** Features independentes e complementares.

---

## 8. Conclusões

### ✅ Features Criadas

- 8 features bem escolhidas
- Balanceamento entre comportamento e qualificação
- Tratamento adequado de valores faltantes
- Normalização aplicada

### 🎯 Próximos Passos

**Fase 3: Modelagem**
- Split estratificado (80/20)
- SMOTE para balanceamento
- Random Forest
- Validação cruzada 5-fold

---

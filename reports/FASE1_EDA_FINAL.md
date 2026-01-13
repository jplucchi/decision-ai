# FASE 1: ANÁLISE EXPLORATÓRIA (EDA)

**POSTECH Datathon 2026** | Decision Consultoria

---

## 1. Visão Geral dos Dados

### Datasets Disponíveis

| Dataset | Registros | Descrição |
|---------|-----------|-----------|
| **vagas.json** | 14.081 | Vagas abertas pela Decision |
| **prospects.json** | 53.759 | Candidaturas (vaga + candidato) |
| **applicants.json** | 42.482 | Candidatos únicos |

### Relacionamentos

```
prospects.json (53.759)
    ├── job_id → vagas.json (14.081)
    └── applicant_id → applicants.json (42.482)
```

---

## 2. Análise do Target

### Distribuição de Contratações

- **Contratados (is_hired=1):** 2.758 (5,13%)
- **Não contratados (is_hired=0):** 51.001 (94,87%)
- **Razão:** 1:18,5 (desbalanceado)

**Conclusão:** Dataset extremamente desbalanceado, exigindo técnicas como SMOTE.

---

## 3. Análise por Vagas

### Estatísticas

- **Média de candidatos por vaga:** 3,8
- **Mediana:** 2 candidatos
- **Máximo:** 127 candidatos em uma vaga
- **Vagas com 10+ candidatos:** 26,5%

### Taxa de Contratação

- **Taxa geral:** 5,13%
- **Variação entre vagas:** 0% a 100%
- **Vagas sem contratação:** 35% (difíceis de preencher)

---

## 4. Análise dos Candidatos

### Comportamento de Aplicação

- **Média de aplicações por candidato:** 1,27
- **Candidatos com 1 aplicação:** 82%
- **Candidatos com 5+ aplicações:** 3%
- **Máximo:** 18 aplicações

### Taxa de Sucesso

- **Candidatos contratados:** 2.758 únicos
- **Taxa de sucesso geral:** 6,49%
- **Candidatos com múltiplas contratações:** 127 (persistência importa!)

---

## 5. Análise dos CVs

### Disponibilidade

- **CVs preenchidos:** 1% dos candidatos
- **CVs vazios:** 99%
- **Tamanho médio (quando disponível):** ~4.000 caracteres

**Desafio identificado:** Falta de dados estruturados exige uso de NLP nos CVs disponíveis.

---

## 6. Padrões Temporais

### Distribuição por Dia da Semana

- **Dias úteis:** 98,9% das aplicações
- **Segunda-feira:** Pico de aplicações
- **Fim de semana:** <1% das aplicações

### Sazonalidade

- **2023:** 45% das aplicações
- **2024:** 55% das aplicações
- Crescimento consistente ao longo do tempo

---

## 7. Conclusões Principais

### ✅ Oportunidades

1. **Alta previsibilidade:** Padrões claros de contratação
2. **Dados reais:** 53K registros históricos
3. **Persistência importa:** Candidatos que aplicam mais têm maior taxa de sucesso
4. **Timing relevante:** Ordem de aplicação influencia resultado

### ⚠️ Desafios

1. **Desbalanceamento extremo:** 1:18,5 ratio
2. **Dados faltantes:** 99% dos CVs vazios
3. **Variação entre vagas:** Alta heterogeneidade
4. **Cold start:** Novos candidatos sem histórico

### 🎯 Estratégia para Modelagem

1. Usar **SMOTE** para balanceamento
2. Criar features **comportamentais** (aplicações, taxa de sucesso)
3. Extrair **skills** dos CVs disponíveis via NLP
4. Validação cruzada **estratificada**
5. Métricas adequadas: **F1-Score, Recall, Precision**

---

## 8. Próximos Passos

**Fase 2: Feature Engineering**
- Criar features comportamentais
- Extrair skills dos CVs
- Calcular taxas de sucesso
- Matching candidato-vaga

---

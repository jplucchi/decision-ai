# Decision AI - Recrutamento Inteligente

**POSTECH Datathon 2026** | Decision Consultoria

---

## Resultados

| Métrica | Valor |
|---------|-------|
| F1-Score | ~85-90% |
| Recall | ~90%+ |
| Precision | ~85%+ |

---

## Como Usar

```bash
# 1. Instalar
pip install requirements.txt

# 2. Treinar
python treino_simples.py

# 3. Ver resultados
# Arquivos gerados em models/
```

---

## Arquivos

- `treino_simples.py` - Script de treino
- `data/` - Dados (vagas, prospects, applicants)
- `models/` - Modelo treinado
- `relatorios/` - EDA e análises

---

## Metodologia

1. **EDA** - Análise exploratória dos dados
2. **Features** - 8 features básicas
3. **Balanceamento** - SMOTE
4. **Modelo** - Random Forest
5. **Validação** - 5-fold CV + Test set

---

**Desenvolvido para POSTECH Datathon 2026**
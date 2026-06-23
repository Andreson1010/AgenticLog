# ADR-017 — Geração única de candidata (NUM_CANDIDATE_RESPONSES=1) com temperatura 0

**Status:** Aceito
**Data:** 2026-06-23
**Feature:** rag-audit-p0
**PR:** #51
**Relacionado:** ADR-005 (chain direta), pipeline de ranqueamento por cosseno

---

## Contexto

`gera_multiplas_respostas` gerava **5 respostas candidatas** por consulta (`for _ in range(5)`)
e `avalia_similaridade` + `rank_respostas` escolhiam a de maior similaridade ao contexto.

Mas `LLM_TEMPERATURE = 0` (config.py) torna a geração **determinística**: as 5 candidatas são
idênticas. O ranqueamento "escolhe" entre clones — **5× o custo/latência de LLM por consulta
sem nenhum ganho de qualidade**.

---

## Decisão

Tornar o número de candidatas configurável e default 1:

```python
# config.py
NUM_CANDIDATE_RESPONSES: int = 1
```

```python
# agent.gera_multiplas_respostas
for _ in range(NUM_CANDIDATE_RESPONSES):
    ...
```

O pipeline de ranqueamento (`avalia_similaridade` → `rank_respostas`) permanece intacto: com 1
candidata ele a retorna com seu `confidence_score` (similaridade ao contexto) — ainda útil como
sinal de confiança. Aumentar `NUM_CANDIDATE_RESPONSES` só agrega valor se `LLM_TEMPERATURE > 0`
(candidatas divergem); o comentário em `config.py` documenta esse acoplamento.

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| Manter 5 candidatas | 5× custo de LLM por query, zero ganho com temperatura 0. |
| Elevar `LLM_TEMPERATURE > 0` e manter N candidatas | Reintroduz não-determinismo nas respostas (indesejado p/ um RAG factual); diversidade de candidatas tem valor marginal vs o custo. Possível experimento futuro, não default. |
| Remover o pipeline de ranqueamento | `confidence_score` (similaridade resposta↔contexto) ainda é um sinal útil exibido na UI; manter o pipeline com N=1 preserva-o sem custo. |

---

## Consequências

**Positivas:**
- 1 chamada de LLM por consulta em vez de 5 — custo e latência reduzidos ~5×.
- `confidence_score` preservado.
- Número de candidatas vira alavanca de config p/ experimentos com temperatura > 0.

**Negativas / Riscos:**
- Nenhum com `temperature=0`. Se a temperatura subir sem ajustar `NUM_CANDIDATE_RESPONSES`,
  perde-se a agregação por ranqueamento — mitigado pela documentação no `config.py`.

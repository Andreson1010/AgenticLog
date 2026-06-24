#!/usr/bin/env python
"""Métricas puras (sem LLM) do harness de avaliação de RAG.

Funções determinísticas e unit-testáveis: normalização do `contexto_ref`,
cosseno, métricas de retrieval por entrada (Hit/MRR/Precision/Recall) e Answer
Correctness. `embed` é injetado (objeto com `embed_query(text) -> list[float]`),
permitindo stubs nos testes sem chamadas reais de embedding/LLM.

Imutabilidade: as funções retornam NOVOS dicts/lists; nenhuma entrada do golden é
mutada (regra de coding-style do projeto).
"""
from __future__ import annotations

import math
from typing import Any

# Sentinelas textuais usadas quando uma referência está ausente.
SENTINEL_CTX_AUSENTE = "contexto_ref ausente"
SENTINEL_RESP_AUSENTE = "resposta_ref ausente"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosseno entre dois vetores; robusto a vetor nulo (denominador -> 1.0)."""
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def _normalizar_contexto_ref(item: dict) -> list[str] | None:
    """Normaliza `contexto_ref`: str -> [str], list -> list (str não vazios), ausente -> None."""
    bruto = item.get("contexto_ref")
    if bruto is None:
        return None
    if isinstance(bruto, str):
        bruto = [bruto]
    if not isinstance(bruto, list):
        return None
    refs = [str(r) for r in bruto if isinstance(r, str) and r.strip()]
    return refs or None


def _metrica_retrieval(
    emb: Any, chunks: list[str], refs: list[str], thr: float
) -> dict[str, float]:
    """Métricas de retrieval embedding-only para UMA entrada.

    Embeda cada chunk e cada ref UMA vez e deriva tudo da mesma matriz de
    similaridade (evita recomputar embeddings — HIGH do review):
      - precision = nº de chunks que casam / nº de chunks
      - hit = 1.0 se algum chunk casa, senão 0.0
      - mrr = 1/(rank do 1º chunk que casa), senão 0.0
      - recall = nº de refs cobertas por algum chunk / nº de refs
    """
    chunk_vecs = [emb.embed_query(c) for c in chunks]
    ref_vecs = [emb.embed_query(r) for r in refs]
    # sim[i][j] = cosseno entre o chunk i (rank i) e a ref j.
    sim = [[_cosine(cv, rv) for rv in ref_vecs] for cv in chunk_vecs]

    # Relevância por chunk: casa se a melhor ref desse chunk passa do threshold.
    rels = [1 if (max(linha, default=0.0) >= thr) else 0 for linha in sim]
    n = len(rels)
    precision = sum(rels) / n if n else 0.0
    hit = 1.0 if any(rels) else 0.0
    mrr = next((1.0 / (i + 1) for i, r in enumerate(rels) if r), 0.0)

    # Recall: fração de refs cobertas por algum chunk acima do threshold.
    cobertas = sum(
        1 for j in range(len(refs)) if max((sim[i][j] for i in range(n)), default=0.0) >= thr
    )
    recall = cobertas / len(refs) if refs else 0.0
    return {
        "hit": round(hit, 3),
        "mrr": round(mrr, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


def _answer_correctness(
    emb: Any, resposta: str, resposta_ref: str | None
) -> float | str:
    """cosine(embed(resposta), embed(resposta_ref)); ausente -> sentinela."""
    if not resposta_ref:
        return SENTINEL_RESP_AUSENTE
    if not resposta:
        return 0.0
    er = emb.embed_query(resposta)
    erf = emb.embed_query(resposta_ref)
    return round(_cosine(er, erf), 3)

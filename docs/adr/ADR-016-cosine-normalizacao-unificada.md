# ADR-016 — Distância cosine no índice e normalização de embeddings unificada

**Status:** Aceito
**Data:** 2026-06-23
**Feature:** rag-audit-p0
**PR:** #51
**Relacionado:** ADR-013 (SemanticChunker), PR #28 (embeddings multilíngues)

---

## Contexto

A auditoria expôs duas inconsistências no espaço vetorial:

1. **Normalização parcial.** Apenas o rebuild (`cria_vectordb`) construía o
   `HuggingFaceEmbeddings` com `encode_kwargs={"normalize_embeddings": True}`. Os singletons de
   ingestão incremental (`_get_rag_embedding_model`) e do agente (`_get_embedding_model`) usavam
   o modelo **sem normalizar**. Resultado: chunks ingeridos incrementalmente e embeddings de
   query tinham normas diferentes dos chunks do rebuild — similaridade degradada silenciosamente
   no mesmo índice.

2. **Métrica de distância default.** A coleção era criada sem `hnsw:space`, então o ChromaDB
   usava o default `l2`, enquanto o ranqueamento de respostas (`agent.avalia_similaridade`) usa
   **cosseno**. Métricas divergentes entre recuperação e ranqueamento.

---

## Decisão

**Normalização unificada:** os três pontos de construção de `HuggingFaceEmbeddings` passam a
usar os mesmos kwargs:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
```

**Distância cosine:** a coleção é criada com metadado explícito, exposto em `config.py`:

```python
CHROMA_DISTANCE_SPACE: str = "cosine"
CHROMA_COLLECTION_METADATA: dict[str, str] = {"hnsw:space": CHROMA_DISTANCE_SPACE}
```

`collection_metadata=CHROMA_COLLECTION_METADATA` é passado em `from_documents` (rebuild) e nas
conexões `Chroma(...)` (ingestão incremental e agente) para cobrir o cold-start.

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| Normalizar só no rebuild (status quo) | Causa direta da degradação silenciosa em ingestão incremental e query. |
| Manter `l2` default | Com vetores normalizados a ordem por L2 e cosseno coincide, mas `cosine` torna os scores interpretáveis (0..1) e alinhados ao ranqueamento. Custo zero. |
| Centralizar o factory de embeddings num módulo compartilhado | Bom, mas `rag.py` ↔ `agent.py` têm import lazy p/ evitar ciclo; padronizar os kwargs em cada call site é mais simples e de menor risco agora. Candidato a refactor futuro. |

---

## Consequências

**Positivas:**
- Espaço vetorial consistente entre rebuild, ingestão incremental e query.
- Scores de similaridade interpretáveis e alinhados ao ranqueamento de respostas.

**Negativas / Riscos:**
- **Requer rebuild de `data/vectordb/`** após o merge: chunks antigos foram embedados sob
  normalização inconsistente e métrica `l2`. (Feito na PR #51 — coleção 0 → 509 chunks.)
- Duplicação dos kwargs em 3 call sites (factory compartilhado fica como follow-up).

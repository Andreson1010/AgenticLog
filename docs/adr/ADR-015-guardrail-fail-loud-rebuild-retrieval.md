# ADR-015 — Guardrail fail-loud no rebuild e WARNING em retrieval vazio

**Status:** Aceito
**Data:** 2026-06-23
**Feature:** rag-audit-p0
**PR:** #51
**Relacionado:** ADR-014 (purga de órfãos), ADR-004 (warning nenhum documento)

---

## Contexto

A coleção vazia (ADR-014) passou despercebida por meses porque o sistema **degradava em
silêncio**:

1. No rebuild, `cria_vectordb` logava "Banco criado com sucesso!" mesmo quando 0 chunks eram
   efetivamente persistidos.
2. Em consulta, `retrieve_info` caía para o fallback `gerar` (geração direta sem contexto) ao
   receber 0 documentos — sem log, sem alerta. O usuário recebia respostas plausíveis porém
   **sem base documental**, indistinguíveis de respostas fundamentadas.

A estratégia retrieve-first (PR #45) tornou esse mascaramento total: qualquer falha de
retrieval vira geração conceitual silenciosa.

---

## Decisão

Duas barreiras fail-loud:

**1. Guardrail de contagem no rebuild** — `cria_vectordb`, após `from_documents`, verifica
`vectordb._collection.count()` e levanta `RuntimeError` se for 0:

```python
persistidos = vectordb._collection.count()
if persistidos == 0:
    raise RuntimeError(
        f"Rebuild gerou coleção vazia ('{collection_name}'): {len(chunks)} chunks "
        "preparados, 0 persistidos. Vector DB não confiável — verifique o ChromaDB."
    )
```

**2. WARNING em retrieval vazio** — `retrieve_info` loga `WARNING` antes do fallback:

```python
if not retrieved_docs:
    logger.warning(
        "Retrieval retornou 0 documentos para a query; caindo para geração direta "
        "(sem contexto). Verifique se o vector DB está populado."
    )
    return state.model_copy(update={"retrieved_info": [], "next_step": "gerar"})
```

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| Confiar só na correção de causa raiz (ADR-014) | Não protege contra regressões futuras nem contra falhas parciais; defesa em profundidade é barata aqui. |
| Bloquear a query (erro 5xx) quando retrieval vazio | Quebra o fallback retrieve-first legítimo (PR #45) p/ perguntas conceituais; WARNING informa sem quebrar UX. |
| Métrica/telemetria de retrieval (P1) em vez de WARNING | Complementar, não substituto; o WARNING é o piso imediato e barato. |

---

## Consequências

**Positivas:**
- Um rebuild que produz coleção vazia falha alto e cedo, em vez de deixar o RAG offline.
- Retrieval vazio fica visível nos logs — operadores conseguem distinguir resposta fundamentada
  de resposta conceitual sem base.

**Negativas / Riscos:**
- `_collection.count()` depende de atributo privado do `langchain_chroma` (frágil entre
  versões). Aceito: já é o padrão usado nos testes do repo; alternativa pública inexistente.
- WARNING por query vazia pode gerar ruído se muitas perguntas legítimas forem conceituais;
  ajustável via `LOG_LEVEL` se necessário.

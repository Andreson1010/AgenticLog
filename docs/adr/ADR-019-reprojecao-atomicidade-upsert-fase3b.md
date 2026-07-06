# ADR-019 — Reprojeção da atomicidade de upsert na extração de `store.py` + `orchestrator.py` (ADR-018 Fase 3b)

**Status:** Aceito
**Data:** 2026-07-04
**Feature:** adr-018-fase-3b
**PR:** (a definir)
**Relacionado:** `docs/adr/ADR-018-redesign-arquitetura-offline-online.md`, `docs/arquitetura-alvo-rag.md`, `.specs/features/adr-018-fase-3b/{spec,design}.md`, PR #61 (Fase 3a)

---

## Contexto

A Fase 3a (PR #61) extraiu os estágios de baixo risco (`security`, `extraction`, `cleaning`,
`chunking`, `embeddings`, `metadata`) de `rag.py` para `agenticlog.ingestion`, via **shims de
re-export identity-preserving**. Restaram em `rag.py` (~710 ln) as duas últimas camadas:

1. **Persistência/atomicidade** — `_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`,
   `_resetar_colecao` e o bloco inline `try/except add_documents/delete` que garante o rollback da
   escrita no Chroma.
2. **Orquestração** — `cria_vectordb`, `adicionar_documento_incrementalmente`,
   `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `reconstruir_vectordb`.

A Fase 3b deve mover essas camadas para `ingestion/store.py` + `ingestion/orchestrator.py`
deixando `rag.py ≤420 ln`, com **comportamento byte-idêntico** e dois arquivos-oráculo com
**zero diff**: `tests/test_rag_caracterizacao.py` (rede de caracterização E2E) e
`tests/ingestion/test_shims_identidade.py` (contrato de identidade da Fase 3a).

O nó do problema é a **atomicidade de upsert** e as **costuras de monkeypatch**. A rede de
caracterização redireciona o pipeline via `monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", ...)`,
`"agenticlog.rag.DIR_VECTORDB"` e `"agenticlog.rag._rag_embedding_model"`, entrando por
`adicionar_documento_incrementalmente`. Se as funções movidas lessem `DIR_DOCUMENTS`/`DIR_VECTORDB`
como nomes de nível de módulo capturados no import de `orchestrator.py`, capturariam o valor
PRÉ-monkeypatch e o oráculo quebraria — sem que a extração possa ser considerada byte-idêntica.

---

## Decisão

### D1 — Primitiva de rollback de escrita, dona pela camada `store`

Extrair o bloco inline `try/except add_documents/delete` (duplicado nos dois `adicionar_*`) para
uma primitiva única em `store.py`:

```python
def add_documents_com_rollback(vectordb_instance, chunks, chunk_ids) -> None:
    try:
        vectordb_instance.add_documents(chunks, ids=chunk_ids)
    except Exception:
        try:
            vectordb_instance.delete(ids=chunk_ids)
        except Exception as rollback_exc:
            logger.warning("Rollback falhou após erro de ingestão. IDs órfãos: %s. "
                           "Erro de rollback: %s", chunk_ids, rollback_exc)
        raise
```

`store` passa a ser o **dono das escritas no Chroma**: `add_documents` + rollback best-effort vivem
lá; o orquestrador fica declarativo. A primitiva re-levanta a exceção ORIGINAL de `add_documents`
(não a de rollback), logando IDs órfãos apenas se o próprio `delete` falhar.

### D2 — Seam de `embedding_model` com default `None`

Cada orquestrador recebe `embedding_model` (keyword-only, default `None`):

- **Rebuild** (`cria_vectordb`): `None` → constrói `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
  model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})` **verbatim**.
  Preserva o caminho de rebuild atual e a flag `normalize_embeddings` (crítica para
  silent-degradation).
- **Incremental** (`adicionar_*`): recebe o singleton via wrapper de `rag`
  (`_get_rag_embedding_model()`), que mantém o cache/getter em `rag.py` (invariante da Fase 3a).

### D3 — Helper de dedup em `orchestrator.py`

Os dois `adicionar_*` compartilham ~120 linhas idênticas. Deduplicar num helper privado
`_ingerir_arquivo_incrementalmente(...)` em `orchestrator.py`, parametrizado por um passo de
extração por-tipo (JSON: `_valida_json_sem_chaves_proibidas` no tmp + `carregar_json`+`filtrar`;
PDF: `extrair_texto_pdf` no tmp + build `Document`s+`filtrar`) e por `doc_type`/`page_args`. As
duas funções públicas fazem apenas `_sanitizar_nome_colecao` + validação de tipo e delegam.

### D4 — Seams ligados no MOMENTO DA CHAMADA por wrappers de `rag.py` (a decisão-chave)

`rag.py` deixa de conter os orquestradores e passa a conter **WRAPPERS finos** que resolvem os
globais de `rag.py` **por nome, a cada chamada**, e os injetam como argumentos:

```python
def adicionar_documento_incrementalmente(filename, conteudo, collection_name=DEFAULT_COLLECTION_NAME):
    return _orch.adicionar_documento_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )
```

As funções em `orchestrator.py` têm assinatura `(..., *, docs_dir=None, vectordb_dir=None,
embedding_model=None)` e resolvem `None → config.*` **no corpo** (nunca como default de argumento,
para honrar `@patch("agenticlog.ingestion.store.DIR_VECTORDB")` etc.). Como o wrapper lê
`agenticlog.rag.DIR_DOCUMENTS`/`DIR_VECTORDB`/`_rag_embedding_model` no momento da chamada, o
monkeypatch do oráculo flui para dentro da função movida sem editar o oráculo. `store.py` aplica o
mesmo padrão em `_outras_colecoes_existem`/`_resetar_colecao` (param `vectordb_dir`).

### D5 — Identidade (`is`) para `store` vs. delegação (wrapper) para orquestradores

- Os **4 símbolos de `store`** (`_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`,
  `_resetar_colecao`) são re-exportados por `rag.py` como shims **`is`-idênticos** (padrão Fase 3a):
  `agenticlog.rag.X is agenticlog.ingestion.store.X`.
- Os **5 orquestradores** são **WRAPPERS** (funções distintas que delegam), portanto
  **NÃO `is`-idênticos** ao objeto de `orchestrator.py` — precisam ligar seams. O novo teste de
  aceitação afirma `is` só para store e delegação (seam flui) para os wrappers.
- `tests/ingestion/test_shims_identidade.py` lista somente os símbolos da Fase 3a → permanece com
  **zero diff** (nem store nem orquestradores estão na sua lista). As novas asserções de
  identidade/delegação vivem em `tests/acceptance/test_rag_ingestion_fase3b.py`.

---

## Micro-reconciliações conscientes (para "byte-idêntico")

- **DN-2 — posição de `construir_docs`:** o helper invoca `construir_docs(saved_path, contexto)`
  como primeira instrução **dentro** do bloco guardado para ambos os tipos. No JSON isso é idêntico
  ao atual; no PDF, a construção (a partir de `paginas` já extraído com sucesso no passo pré-move)
  migra para dentro do guard. É observável apenas se a construção levantar — o que não ocorre para
  inputs alcançáveis; no caso patológico, reverter o disco é o comportamento mais correto.
- **DN-3 — mensagens de log:** a mensagem de warning de `invalidar_vector_db` é parâmetro por-tipo
  do helper (preserva as strings distintas `"...do agente: %s"` / `"...singleton: %s"`). A mensagem
  de log de rollback é unificada em `add_documents_com_rollback` para a variante JSON
  (`"Erro de rollback: %s"`) — string de log interna, não asserida por nenhum teste/oráculo.

---

## Consequências

**Positivas**
- `rag.py` cai de ~710 para ~200 ln (dentro do teto); `store.py` (~150–180 ln) e `orchestrator.py`
  (~370 ln) coesos.
- Atomicidade de escrita concentrada em `store` (uma primitiva, um ponto de manutenção).
- ~120 ln de duplicação eliminadas entre os dois `adicionar_*`.
- `cria_vectordb` deixa de depender de `DirectoryLoader` (itera `carregar_json` sobre
  `sorted(docs_dir.glob("*.json"))`), simplificando o caminho de rebuild.
- Seams de DI explícitos habilitam testes futuros sem monkeypatch de globais (down-payment da
  Fase 6).

**Negativas / custos**
- Os 5 orquestradores em `rag` deixam de ser `is`-idênticos aos de `orchestrator` (wrappers) — a
  distinção precisa ser entendida por quem lê os shims.
- Testes NÃO-oráculo que patcham símbolos agora internos às funções movidas exigem migração dirigida
  de ALVO de patch (namespace) — comportamento verificado inalterado.
- Chamadas DIRETAS a `orchestrator.adicionar_*` (fora do wrapper) com `embedding_model=None` fariam
  fallback para `criar_embedding_model()`; o caminho de produção/oráculo sempre passa pelo wrapper.

**Riscos residuais mitigados**
- Quebra do oráculo por seam em tempo de import → wrappers ligam no momento da chamada (D4).
- Regressão de atomicidade → primitiva + helper preservam a sequência
  `backup→move→chunk→add→rollback→_reverter_disco`; guardrail `RuntimeError` (coleção vazia) e
  fail-fast de `RAGSecurityError` intactos.
- `hnswlib`/SAC no Windows: gate autoritativo é o CI Linux.

Shims e wrappers são temporários — **removidos na Fase 6** (marcador
`# Re-export shim (ADR-018 Fase 3b) — remover na Fase 6`), quando os importadores migram para
`agenticlog.ingestion.{store,orchestrator}` diretamente.

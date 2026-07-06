# ADR-018 Fase 3b — Tasks

**Path:** `.specs/features/adr-018-fase-3b/tasks.md`
**Spec:** `.specs/features/adr-018-fase-3b/spec.md` · **Design:** `.specs/features/adr-018-fase-3b/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

Tasks para o backend-builder da feature-factory. Ordem projetada para o oráculo + suíte completa verdes ao final. TLC Execute NÃO é usado. Gate autoritativo: **CI Linux** (`pytest --cov=agenticlog -v`). Local Windows pula testes Chroma/hnswlib (Smart App Control) — não confundir com regressão. Padrões: `ruff`/`black`/`isort` limpos, type hints, docstrings PT (Entrada/Saída/Lança), imutabilidade, funções <50 ln, commits Conventional em PT (`refactor:`), branch-first, code-review antes do merge.

**Regra de ouro (todas as tasks):** NUNCA editar `tests/test_rag_caracterizacao.py` nem `tests/ingestion/test_shims_identidade.py`. Preservar comportamento, mensagens, ordem de exceções e a atomicidade de upsert verbatim. Migração de teste = só ALVO de patch (namespace), nunca a asserção de comportamento.

Ordem macro: **store.py → orchestrator.py → rag.py wrappers → testes novos (acicidade + identidade/delegação) → migração de patch-target → suíte completa.**

---

## Execution Plan

### Fase A — `store.py` (sequencial)
```
T1 → T2
```

### Fase B — `orchestrator.py` (sequencial, depende de store)
```
T2 → T3 → T4 → T5
```

### Fase C — Religar `rag.py` (sequencial, depende de orchestrator)
```
T5 → T6
```

### Fase D — Testes novos (paralelo após religar)
```
        ┌→ T7 [P]
T6 ─────┼→ T8 [P]
        └→ T9 [P]
```

### Fase E — Migração de patch-target dos testes NÃO-oráculo (paralelo por-arquivo)
```
        ┌→ T10 [P]
T6 ─────┼→ T11 [P]
        └→ T12 [P]
```

### Fase F — Gate final (sequencial)
```
T7,T8,T9,T10,T11,T12 → T13
```

---

## Task Breakdown

### T1: Criar `store.py` com as 4 primitivas de disco/coleção · [ING3B-01]

**What:** Mover verbatim `_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`, `_resetar_colecao` para `src/agenticlog/ingestion/store.py`; parametrizar `_outras_colecoes_existem`/`_resetar_colecao` por `vectordb_dir: Path | None = None` (resolver `None → DIR_VECTORDB` no corpo, ver design §4.2); `_resetar_colecao` propaga `vectordb_dir` ao chamar `_outras_colecoes_existem`.
**Where:** `src/agenticlog/ingestion/store.py` (novo)
**Depends on:** None
**Reuses:** corpos de `rag.py:102-186`; `config.DIR_VECTORDB`; import lazy de `sqlite3`/`chromadb` mantido.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `store.py` importa só de `config` (`DIR_VECTORDB`) e stdlib; NÃO importa `rag`/`agent`/`orchestrator`.
- [ ] `python -c "import agenticlog.ingestion.store"` sai 0.
- [ ] Corpos de `_backup_arquivo`/`_reverter_disco` idênticos ao original; os dois `_*colecao*` leem `vectordb_dir` no corpo (não como default de arg).
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (co-locado em T2) · **Gate:** quick

---

### T2: Extrair a primitiva `add_documents_com_rollback` + unit de store · [ING3B-01]

**What:** Adicionar `add_documents_com_rollback(vectordb_instance, chunks, chunk_ids) -> None` a `store.py` (design §3.1, D1): `add_documents`; na exceção `delete(ids=chunk_ids)`, loga IDs órfãos se o delete falhar (`"Rollback falhou após erro de ingestão. IDs órfãos: %s. Erro de rollback: %s"`), re-levanta o original. Criar `tests/ingestion/test_store.py`.
**Where:** `src/agenticlog/ingestion/store.py`, `tests/ingestion/test_store.py` (novo)
**Depends on:** T1
**Reuses:** bloco inline `try/except add_documents/delete` de `rag.py:295-306`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `add_documents_com_rollback` chama `add_documents(chunks, ids=chunk_ids)` no caminho feliz.
- [ ] Na falha de `add_documents`: chama `delete(ids=chunk_ids)`, re-levanta a exceção ORIGINAL; se `delete` também falhar, loga órfãos e ainda re-levanta a original.
- [ ] Unit de `_outras_colecoes_existem`/`_resetar_colecao` com `vectordb_dir` explícito e via patch de `store.DIR_VECTORDB` (nomes `teste_N_`).
- [ ] Gate: `pytest tests/ingestion/test_store.py -v` verde.

**Tests:** unit · **Gate:** quick

---

### T3: Criar `orchestrator.py` — `cria_vectordb` sem `DirectoryLoader` · [ING3B-02, ING3B-05]

**What:** Mover `cria_vectordb` e `reconstruir_vectordb` para `src/agenticlog/ingestion/orchestrator.py` com assinatura de seams (`*, docs_dir=None, vectordb_dir=None, embedding_model=None`); resolver `None → config` no corpo; substituir `DirectoryLoader` por `for p in sorted(docs_dir.glob("*.json")): json_docs.extend(carregar_json(p))` (design §5.4); `embedding_model=None` → construir `HuggingFaceEmbeddings` fresco verbatim; `_resetar_colecao`→`store._resetar_colecao(collection_name, vectordb_dir=vectordb_dir)`; `cria_vectordb` **retorna** a instância Chroma (ou `None`); guardrail `RuntimeError` sobre a instância local antes de retornar.
**Where:** `src/agenticlog/ingestion/orchestrator.py` (novo)
**Depends on:** T2
**Reuses:** `rag.py:497-618`; `extraction.carregar_json`/`extrair_texto_pdf`, `cleaning.filtrar_documentos_vazios`, `chunking.SemanticChunker`, `metadata._hash_arquivo`/`_enriquecer_metadados_chunks`, `security._sanitizar_nome_colecao`/`_valida_*`, `store._resetar_colecao`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `cria_vectordb` NÃO referencia `DirectoryLoader`; itera `carregar_json` sobre `sorted(glob)`.
- [ ] PDF loop, chunking por-source, enriquecimento, guardrail `RuntimeError` e mensagens idênticos ao original.
- [ ] `orchestrator.py` NÃO importa `rag`/`agent`.
- [ ] `python -c "import agenticlog.ingestion.orchestrator"` sai 0.

**Tests:** unit (co-locado — reescrita de `TestCriaVectordb` em T10) · **Gate:** quick

---

### T4: Adicionar o helper `_ingerir_arquivo_incrementalmente` + os dois `adicionar_*` · [ING3B-06, ING3B-07]

**What:** Implementar o helper privado compartilhado (design §5.1) parametrizado por `tmp_suffix`, `preparar_no_tmp`, `construir_docs`, `doc_type`, `page_args`, `invalidation_msg` + seams; preservar a sequência de atomicidade EXATA `backup→move→[guardado: construir_docs→chunk→enrich→add_documents_com_rollback→delete(old_ids)]→_reverter_disco na exceção`; `construir_docs` como 1ª instrução DENTRO do guard para ambos (DN-2). Implementar `adicionar_documento_incrementalmente` e `adicionar_pdf_incrementalmente` como public fns que fazem `_sanitizar_nome_colecao` + validação de tipo (suffix/magic/tamanho) + resolvem seams e delegam ao helper com o bundle por-tipo. Usar `store.add_documents_com_rollback`, `store._backup_arquivo`, `store._reverter_disco`.
**Where:** `src/agenticlog/ingestion/orchestrator.py`
**Depends on:** T3
**Reuses:** `rag.py:189-494` (os dois `adicionar_*`); `store` primitivas.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Precedência preservada: `_sanitizar_nome_colecao` antes da validação de tipo.
- [ ] JSON: `preparar_no_tmp`=`_valida_json_sem_chaves_proibidas`(→None); `construir_docs`=`carregar_json`+`filtrar`; `page_args=(METADATA_PAGE_JSON_SENTINEL,)`; `invalidation_msg` da variante "do agente".
- [ ] PDF: `preparar_no_tmp`=`extrair_texto_pdf`(→paginas); `construir_docs`=build `Document`s+`filtrar`; `page_args=()`; `invalidation_msg` da variante "singleton".
- [ ] Mensagens de retorno (`adicionado`/`substituido`/`duplicado`/`0 chunks`) byte-idênticas às atuais.
- [ ] Funções <50 ln (helper pode extrair sub-passos se necessário para respeitar o limite).

**Tests:** unit (migração em T10/T11) · **Gate:** quick

---

### T5: Preencher `ingestion/__init__.py` (re-exports aditivos) · [ING3B-02]

**What:** Re-exportar símbolos públicos de `store`/`orchestrator` e atualizar `__all__` (aditivo; não quebrar re-exports 3a).
**Where:** `src/agenticlog/ingestion/__init__.py`
**Depends on:** T4
**Reuses:** padrão atual do `__init__.py`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `from agenticlog.ingestion import cria_vectordb, adicionar_documento_incrementalmente` resolve.
- [ ] Re-exports da Fase 3a intactos.

**Tests:** none (coberto por T7/T9) · **Gate:** quick

---

### T6: Religar `rag.py` — shim de store + 5 WRAPPERS + remoção de corpos/imports · [ING3B-03, ING3B-04]

**What:** Remover de `rag.py` os corpos movidos (store + 5 orquestradores). Adicionar o shim block `is`-identity de store (design §3.3) com marcador `# Re-export shim (ADR-018 Fase 3b) — remover na Fase 6`. Adicionar `import agenticlog.ingestion.orchestrator as _orch` e os 5 WRAPPERS finos (design §4.1) que ligam `DIR_DOCUMENTS`/`DIR_VECTORDB`/`_get_rag_embedding_model()` no momento da chamada; `cria_vectordb` wrapper faz `global vectordb; vectordb = _orch.cria_vectordb(...)`. Remover imports `DirectoryLoader`/`JSONLoader` (só usados por `cria_vectordb`). Manter getter/cache de embedding, `import fitz # noqa: F401`, shims 3a e CLI.
**Where:** `src/agenticlog/rag.py`
**Depends on:** T5
**Reuses:** padrão de shim `rag.py:76-99`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `rag.py` tem **≤420 ln** (verificar contagem).
- [ ] `from agenticlog.rag import _backup_arquivo, _reverter_disco, _outras_colecoes_existem, _resetar_colecao, cria_vectordb, adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente, ingerir_incrementalmente, reconstruir_vectordb` resolve sem `ImportError`.
- [ ] `agenticlog.rag.X is agenticlog.ingestion.store.X` para os 4 símbolos de store.
- [ ] `python -c "import agenticlog.rag"` sai 0; `ruff`/`black`/`isort` limpos.

**Tests:** none (validado por T8/T9) · **Gate:** quick

---

### T7: Novo teste de acicidade `store`+`orchestrator` · [P] · [ING3B-09]

**What:** Criar `tests/acceptance/test_rag_ingestion_fase3b.py` com o teste de acicidade fresh-interpreter (`subprocess` importando `agenticlog.ingestion.store` e `agenticlog.ingestion.orchestrator`) → returncode 0, sem "circular"/"partially initialized". Adicionar também os checks AC1/AC2 (símbolos existem nos módulos) e AC5 (`cria_vectordb` não referencia `DirectoryLoader`).
**Where:** `tests/acceptance/test_rag_ingestion_fase3b.py` (novo)
**Depends on:** T6
**Reuses:** `test_rag_shared_observability.py::test_ac06`, `test_shims_identidade.py::TestIngestionAcyclic`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_rag_ingestion_fase3b.py -v` verde.
- [ ] `test_ac05` em `test_rag_ingestion_fase3a.py` NÃO tocado.

**Tests:** integration (subprocess) · **Gate:** full

---

### T8: Novo teste de identidade-de-store + delegação-de-wrapper · [P] · [ING3B-03]

**What:** No mesmo arquivo novo (`test_rag_ingestion_fase3b.py`): afirmar `agenticlog.rag.X is agenticlog.ingestion.store.X` para os 4 símbolos de store; afirmar que os 5 orquestradores em `rag` são WRAPPERS (`rag.f is not orchestrator.f`) e que a delegação faz os seams fluírem — teste de seam-binding: `rag.DIR_VECTORDB` monkeypatchado + `Chroma` patchado em `orchestrator` recebe `persist_directory=str(<valor patchado>)`.
**Where:** `tests/acceptance/test_rag_ingestion_fase3b.py`
**Depends on:** T6
**Reuses:** mecânica de seam do oráculo (design §4.1).

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Identidade `is` afirmada para os 4 símbolos de store.
- [ ] Delegação afirmada para os 5 wrappers (não-identidade + seam flui).
- [ ] `pytest tests/acceptance/test_rag_ingestion_fase3b.py -v` verde.

**Tests:** unit · **Gate:** quick

---

### T9: Verificar oráculos zero-diff · [P] · [ING3B-08, ING3B-10]

**What:** Rodar os dois oráculos e confirmar `git diff` VAZIO. Este é um gate de verificação, não de edição.
**Where:** `tests/test_rag_caracterizacao.py`, `tests/ingestion/test_shims_identidade.py` (só leitura/execução)
**Depends on:** T6
**Reuses:** —

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` verde (CI Linux).
- [ ] `pytest tests/ingestion/test_shims_identidade.py -v` verde.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py tests/ingestion/test_shims_identidade.py` VAZIO.

**Tests:** integration · **Gate:** full

---

### T10: Migrar patch-target de `tests/test_rag.py` (orquestradores + reset + cria_vectordb) · [P] · [ING3B-10]

**What:** Migrar SÓ alvos de patch (namespace), não comportamento: reescrever `TestCriaVectordb` fora de `DirectoryLoader` → `agenticlog.ingestion.orchestrator.*` (carregar_json por-arquivo, Chroma, SemanticChunker, HuggingFaceEmbeddings); `TestResetarColecao` → `agenticlog.ingestion.store._outras_colecoes_existem`; `TestOutrasColecoesExistem` → `agenticlog.ingestion.store.DIR_VECTORDB` (ou param); `TestAdicionarDocumentoIncrementalmente`/`TestAdicionarPdfIncrementalmente` → `orchestrator`/`store` (`Chroma`, `SemanticChunker`, `shutil`, `tempfile`, `uuid`, `extrair_texto_pdf`); `TestReconstruirVectordb` → `orchestrator.cria_vectordb`; `TestIngerirIncrementalmente` `teste_1/2/2b/2c` → `orchestrator`.
**Where:** `tests/test_rag.py`
**Depends on:** T6
**Reuses:** inventário de seam do design §6 / spec §Tests.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/test_rag.py -v` verde; asserções de comportamento inalteradas.
- [ ] `TestLogging`, `TestStructuredLogConfig`, `TestIngerirIncrementalmente teste_3/4/5` NÃO alterados.
- [ ] Contagem de testes preservada (sem deleção silenciosa).

**Tests:** unit · **Gate:** quick

---

### T11: Migrar patch-target dos testes de aceitação incrementais · [P] · [ING3B-10]

**What:** Migrar alvos (namespace) in-place: `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (todos os métodos → `orchestrator`/`store`); `test_semantic_chunking.py`, `test_unificar_metadados_chunks.py`, `test_portuguese_embedding_model.py`: `agenticlog.rag._resetar_colecao` → `agenticlog.ingestion.orchestrator._resetar_colecao` ou `store._resetar_colecao`. NÃO tocar testes de inspeção de fonte de `test_portuguese_embedding_model`.
**Where:** `tests/acceptance/test_adicionar_pdf_incrementalmente.py`, `test_semantic_chunking.py`, `test_unificar_metadados_chunks.py`, `test_portuguese_embedding_model.py`
**Depends on:** T6
**Reuses:** design §6.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_adicionar_pdf_incrementalmente.py tests/acceptance/test_semantic_chunking.py tests/acceptance/test_unificar_metadados_chunks.py tests/acceptance/test_portuguese_embedding_model.py -v` verde (CI Linux).
- [ ] Comportamento verificado inalterado; contagem preservada.

**Tests:** unit/integration · **Gate:** full

---

### T12: Escrever ADR-019 · [P] · [ING3B-07]

**What:** Escrever `docs/adr/ADR-019-reprojecao-atomicidade-upsert-fase3b.md` documentando a decisão de reprojeção da atomicidade de upsert: seams de DI ligados no momento da chamada, shims `is` de store vs. wrappers de orquestrador, a primitiva `add_documents_com_rollback`, e as decisões DN-2/DN-3.
**Where:** `docs/adr/ADR-019-reprojecao-atomicidade-upsert-fase3b.md` (novo)
**Depends on:** T6 (referencia a estrutura final)
**Reuses:** formato dos ADRs existentes em `docs/adr/`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] ADR-019 segue o formato dos ADRs existentes (Contexto/Decisão/Consequências).
- [ ] Referencia ADR-018 e a spec/design da Fase 3b.

**Tests:** none · **Gate:** none

---

### T13: Gate final — suíte completa + lint + contagem de linhas · [ING3B-08, todos]

**What:** Rodar a suíte completa, lint e as verificações de tamanho/diff.
**Where:** todo o repo
**Depends on:** T7, T8, T9, T10, T11, T12
**Reuses:** —

**Tools:** MCP: NONE · Skill: `code-review` (antes do merge)

**Done when:**
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde (CI Linux); cobertura ≥ 80%.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py tests/ingestion/test_shims_identidade.py` VAZIO.
- [ ] `rag.py ≤420 ln`; `store.py` e `orchestrator.py` dentro de 200–400 ln (store pode ficar levemente abaixo — anotado no design §3.1).
- [ ] `ruff check .`, `black --check .`, `isort --check .` limpos.
- [ ] Varredura de seam: `rg "@patch\(\"agenticlog\.rag\.(cria_vectordb|_resetar_colecao|_outras_colecoes_existem|DirectoryLoader)\"" tests/` — cada ocorrência remanescente DEVE exercitar um chamador que ficou em `rag.py`; qualquer teste que chame função MOVIDA e patche em `rag.*` deve migrar.
- [ ] `code-review` executado antes do merge.

**Tests:** integration (full suite) · **Gate:** build

**Commit:** `refactor(rag): extrai store.py + orchestrator.py de rag.py (ADR-018 Fase 3b)`

---

## Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram | Status |
|------|-------------------|---------|--------|
| T1 | None | (raiz Fase A) | ✅ |
| T2 | T1 | T1→T2 | ✅ |
| T3 | T2 | T2→T3 | ✅ |
| T4 | T3 | T3→T4 | ✅ |
| T5 | T4 | T4→T5 | ✅ |
| T6 | T5 | T5→T6 | ✅ |
| T7 | T6 | T6→T7 [P] | ✅ |
| T8 | T6 | T6→T8 [P] | ✅ |
| T9 | T6 | T6→T9 [P] | ✅ |
| T10 | T6 | T6→T10 [P] | ✅ |
| T11 | T6 | T6→T11 [P] | ✅ |
| T12 | T6 | T6→T12 [P] | ✅ |
| T13 | T7,T8,T9,T10,T11,T12 | →T13 | ✅ |

Tarefas `[P]` (T7–T12) não dependem entre si — todas dependem só de T6. T7–T9 tocam arquivos de teste novos/leitura; T10–T12 tocam arquivos distintos (test_rag.py, acceptance/*, docs/adr/*) → sem estado mutável compartilhado.

## Test Co-location Validation

| Task | Layer criado/modificado | Matriz exige | Task diz | Status |
|------|------------------------|--------------|----------|--------|
| T1/T2 | rag.py ingestion (store) | unit | unit (T2) | ✅ |
| T3/T4 | rag.py ingestion (orchestrator) | unit | unit (migrado T10/T11) | ✅ |
| T6 | fachada (shims/wrappers) | unit | validado por T8/T9 | ✅ |
| T7/T8 | teste novo | integration/unit | integration/unit | ✅ |
| T10/T11 | testes migrados | unit/integration | unit/integration | ✅ |

Nota: T3/T4 criam código de orquestrador cujos unit tests JÁ existem (classes de orquestrador) e são migrados por-namespace em T10/T11 — não é deferral (os testes existem e passam ao final da mesma fase de trabalho); a co-locação é satisfeita pela migração dirigida + novos testes de seam em T8.

---

## Notas de teste / gate (de `.specs/codebase/TESTING.md` e CLAUDE.md)
- Sempre mockar LLM; sempre testar retrieval vazio (coberto pelo oráculo).
- Nomes de teste com prefixo `teste_N_` (domínio) / `test_` (erro/segurança).
- Gate autoritativo: CI Linux `pytest --cov=agenticlog --cov-report=term-missing -v`.
- Commits Conventional em PT (`refactor:`); branch já criada (`refactor/adr-018-fase-3b`); code-review antes do merge.

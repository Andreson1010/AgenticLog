# ADR-018 Fase 3b — reprojeção dos orquestradores + store de ingestão — Technical Spec

**Path:** `.specs/features/adr-018-fase-3b/spec.md`
**TLC scope:** large
**Based on story:** ADR-018 Fase 3b — extrair `store.py` + `orchestrator.py` de `rag.py`, deixando `rag.py ≤420 ln`, com comportamento byte-idêntico e os DOIS arquivos-oráculo (`tests/test_rag_caracterizacao.py`, `tests/ingestion/test_shims_identidade.py`) com ZERO diff.
**Status:** Awaiting human approval

---

## Problem Statement

Após a Fase 3a (PR #61), `src/agenticlog/rag.py` ainda concentra ~710 linhas: os cinco orquestradores de ingestão (`cria_vectordb`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `reconstruir_vectordb`), as primitivas de escrita/atomicidade no disco e no Chroma (`_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`, `_resetar_colecao`) e a CLI. Isso viola a regra de arquivo pequeno (200–400 ln) e mantém a lógica de atomicidade de upsert acoplada ao módulo de fachada. A **Fase 3b** da ADR-018 extrai a camada de persistência para `agenticlog.ingestion.store` e os orquestradores para `agenticlog.ingestion.orchestrator`, reprojetando as costuras (seams) de injeção de dependência para que o monkeypatch do oráculo continue fluindo — tudo guardado pela rede de caracterização (`test_rag_caracterizacao.py`) e pelo contrato de identidade dos shims (`test_shims_identidade.py`), que NÃO podem ser modificados.

## Goals

- [ ] Criar `src/agenticlog/ingestion/store.py` com as 4 primitivas de disco/coleção + 1 primitiva de rollback de escrita no Chroma, cada assinatura parametrizada por `vectordb_dir` (sem leitura de global de módulo).
- [ ] Criar `src/agenticlog/ingestion/orchestrator.py` com os 5 orquestradores, cada um recebendo seams injetáveis (`docs_dir`, `vectordb_dir`, `embedding_model`) resolvidos no momento da chamada.
- [ ] Reduzir `rag.py` a `≤420 ln`, mantendo shims de re-export para os símbolos de `store` (identidade `is`) e WRAPPERS finos para os 5 orquestradores (delegação, não identidade).
- [ ] Comportamento byte-idêntico: fluxo, mensagens, ordem de exceções, atomicidade de upsert e espaço vetorial preservados.
- [ ] Os dois arquivos-oráculo verdes com `git diff` VAZIO.
- [ ] Deduplicar os dois `adicionar_*` num helper compartilhado + passo de extração por-tipo.
- [ ] Manter o grafo de imports acíclico para `store.py` + `orchestrator.py` (novo teste de aceitação).

## Out of Scope

| Feature | Reason |
|---------|--------|
| Mover a CLI (`_executar_main`, `_configurar_logging_cli`, bloco `__main__`) | Permanece em `rag.py` (reavaliada em fase posterior da ADR-018). |
| Mover o singleton global `_rag_embedding_model` e o getter `_get_rag_embedding_model` | Ficam em `rag.py` (invariante da Fase 3a; preserva o monkeypatch do oráculo). |
| Mover os estágios extraídos na Fase 3a (`security`, `extraction`, `cleaning`, `chunking`, `embeddings`, `metadata`) | Já em `agenticlog.ingestion.*`; reutilizados como estão. |
| Alterar o contrato de identidade da Fase 3a (`test_shims_identidade.py`) | HARD constraint: zero diff nesse arquivo. |
| Remover shims / `import fitz` de `rag.py` | Removidos apenas na Fase 6. |
| Mudança de comportamento, config, schema de metadados ou espaço vetorial | Estritamente proibido nesta fase. |
| Reescrever asserções de comportamento dos testes NÃO-oráculo | Só se migra ALVO de patch (namespace), nunca o comportamento verificado. |

---

## User Stories

### P1: `store.py` + `orchestrator.py` extraídos com comportamento idêntico ⭐ MVP

**User Story**: Como mantenedor do AgenticLog, quero a camada de persistência e os orquestradores de ingestão em módulos pequenos e coesos sob `agenticlog.ingestion`, com as costuras de injeção de dependência reprojetadas, para reduzir `rag.py` a `≤420 ln` e concluir a Fase 3b da ADR-018 sem alterar nenhum comportamento observável.

**Why P1**: É o objetivo único e indivisível da fase; sem ele não há entrega.

**Acceptance Criteria**:
1. WHEN o pacote é reprojetado THEN `agenticlog.ingestion.store` SHALL expor `_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`, `_resetar_colecao` e a primitiva de rollback `add_documents_com_rollback(vectordb_instance, chunks, chunk_ids) -> None`. *(AC1 → ING3B-01)*
2. WHEN o pacote é reprojetado THEN `agenticlog.ingestion.orchestrator` SHALL expor `cria_vectordb`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `reconstruir_vectordb`. *(AC2 → ING3B-02)*
3. WHEN `rag.py` é religado THEN o arquivo SHALL ter `≤420 ln`, `from agenticlog.rag import <símbolo movido>` SHALL resolver sem `ImportError`, e para cada um dos 4 símbolos de `store` `agenticlog.rag.X is agenticlog.ingestion.store.X` SHALL ser verdadeiro (identidade `is`). *(AC3 → ING3B-03)*
4. WHEN um orquestrador movido é chamado via o WRAPPER de `rag.py` E `monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", ...)` (ou `DIR_VECTORDB`, ou `_rag_embedding_model`) está ativo THEN a função movida SHALL honrar o valor monkeypatchado, porque os seams (`docs_dir`, `vectordb_dir`, `embedding_model`) são ligados NO MOMENTO DA CHAMADA pelo wrapper a partir dos globais de `rag.py`. *(AC4 → ING3B-04)*
5. WHEN `cria_vectordb` carrega documentos JSON THEN SHALL iterar `carregar_json` sobre `docs_dir.glob("*.json")` (sem `DirectoryLoader`). *(AC5 → ING3B-05)*
6. WHEN os dois `adicionar_*` são implementados THEN SHALL delegar a um helper privado compartilhado `_ingerir_arquivo_incrementalmente(...)` parametrizado por um passo de extração por-tipo + `doc_type`/`page`. *(AC6 → ING3B-06)*
7. WHEN um upsert incremental falha em qualquer ponto pós-`shutil.move` THEN a atomicidade SHALL ser preservada EXATAMENTE: `backup → move → chunk → add → (na falha do add) rollback via delete dos novos IDs → _reverter_disco`; o guardrail de coleção-vazia (`RuntimeError`) em `cria_vectordb` e o fail-fast de `RAGSecurityError` em `ingerir_incrementalmente` SHALL permanecer idênticos. *(AC7 → ING3B-07)*
8. WHEN a suíte completa roda no CI Linux THEN `tests/test_rag_caracterizacao.py` E `tests/ingestion/test_shims_identidade.py` SHALL passar com `git diff` VAZIO nos dois arquivos. *(AC8 → ING3B-08, ING3B-10)*
9. WHEN `import agenticlog.ingestion.store` e `import agenticlog.ingestion.orchestrator` rodam num interpretador frio THEN SHALL sair com código 0 (sem ciclo), verificado por um NOVO teste de aceitação; `test_ac05` em `tests/acceptance/test_rag_ingestion_fase3a.py` SHALL permanecer intocado. *(AC9 → ING3B-09)*

**Independent Test**: `pytest -m integration tests/test_rag_caracterizacao.py -v` verde; `pytest tests/ingestion/test_shims_identidade.py -v` verde; `git diff --stat` VAZIO nos dois; `pytest --cov=agenticlog -v` completo verde no CI Linux com `rag.py ≤420 ln`.

---

## Edge Cases

- WHEN `adicionar_documento_incrementalmente` é chamado via wrapper de `rag.py` com `agenticlog.rag._rag_embedding_model` monkeypatchado (stub) THEN o wrapper SHALL passar `embedding_model=_get_rag_embedding_model()` (o stub) e a função movida SHALL usá-lo sem construir o modelo real de ~1 GB.
- WHEN `_stub_embedding_que_falha` levanta dentro do `SemanticChunker.split_documents` (oráculo `teste_4`) THEN `_reverter_disco` SHALL restaurar o disco ANTES de qualquer `add_documents` e a coleção SHALL permanecer intacta.
- WHEN um `collection_name` inválido E um `suffix` inválido são passados juntos THEN `_sanitizar_nome_colecao` SHALL levantar PRIMEIRO (precedência de exceção preservada — sanitização de coleção antes da validação de tipo).
- WHEN um upsert (arquivo já indexado, hash diferente) falha no `add_documents` THEN o rollback SHALL deletar os novos `chunk_ids`, logar IDs órfãos se o próprio rollback falhar, re-levantar o erro original e `_reverter_disco` SHALL restaurar o backup do arquivo antigo.
- WHEN a extração produz 0 chunks THEN a função SHALL chamar `_reverter_disco` e retornar `{"status": "adicionado", "mensagem": "...não pôde ser indexado: 0 chunks gerados."}` (mensagem idêntica à atual).
- WHEN `cria_vectordb` persiste 0 chunks THEN SHALL levantar `RuntimeError` (guardrail fail-loud) com a mensagem verbatim atual.
- WHEN `ingerir_incrementalmente` encontra `RAGSecurityError` num arquivo THEN SHALL abortar o lote (fail-fast, re-levantar); um erro operacional comum SHALL incrementar `contadores["erro"]` e continuar.
- WHEN `store.py` é importado num interpretador frio THEN SHALL importar SOMENTE de `config` e `shared` (sem `rag`/`agent`/`orchestrator`) → sem ciclo.

---

## Requirement Traceability

| Requirement ID | Story | AC | Phase | Status |
|----------------|-------|----|-------|--------|
| ING3B-01 (`store.py`: 4 primitivas + `add_documents_com_rollback`) | P1 | AC1 | Design | Pending |
| ING3B-02 (`orchestrator.py`: 5 orquestradores) | P1 | AC2 | Design | Pending |
| ING3B-03 (`rag.py ≤420 ln`; imports resolvem; `is`-identity dos 4 símbolos de store) | P1 | AC3 | Design | Pending |
| ING3B-04 (seams injetáveis ligados no momento da chamada por wrappers; monkeypatch honrado) | P1 | AC4 | Design | Pending |
| ING3B-05 (`cria_vectordb` sem `DirectoryLoader` → `carregar_json` sobre `glob`) | P1 | AC5 | Design | Pending |
| ING3B-06 (dedup dos dois `adicionar_*` num helper + passo de extração por-tipo) | P1 | AC6 | Design | Pending |
| ING3B-07 (atomicidade de upsert preservada exatamente; guardrail + fail-fast) | P1 | AC7 | Design | Pending |
| ING3B-08 (`test_rag_caracterizacao.py` verde, zero diff) | P1 | AC8 | Design | Pending |
| ING3B-09 (acicidade de `store`+`orchestrator`; novo teste de aceitação) | P1 | AC9 | Tasks | Pending |
| ING3B-10 (migração dirigida de patch-target dos testes NÃO-oráculo; `test_shims_identidade.py` zero diff) | P1 | AC8 | Tasks | Pending |

**ID format:** `ING3B-[NUMBER]` (ING3B = INGestion fase 3B).

**Coverage:** 10 IDs totais; todos mapeados a tasks (ver `tasks.md`).

---

## Data Model Changes

Nenhuma. Sem alteração em schema JSON, metadados de chunk, coleção Chroma ou espaço vetorial. As chamadas a `_enriquecer_metadados_chunks` (JSON: `METADATA_DOC_TYPE_JSON` + `METADATA_PAGE_JSON_SENTINEL`; PDF: `METADATA_DOC_TYPE_PDF`) e o `collection_metadata=CHROMA_COLLECTION_METADATA` são preservadas verbatim.

---

## Process / Background Flow

**Happy path (ingestão incremental JSON — oráculo `teste_1`/`teste_2`):**
`rag.adicionar_documento_incrementalmente` (WRAPPER) → liga seams a partir de `rag.DIR_DOCUMENTS`/`rag.DIR_VECTORDB`/`rag._get_rag_embedding_model()` → `orchestrator.adicionar_documento_incrementalmente(..., docs_dir, vectordb_dir, embedding_model)` → `_sanitizar_nome_colecao` → validação de tipo `.json`+tamanho → `_sanitizar_nome_arquivo` → checagem de contagem via `docs_dir.glob` → `_computar_hash_conteudo` → `Chroma(persist_directory=str(vectordb_dir), ...)` → dedup por `source`/hash → grava tmp → `_valida_json_sem_chaves_proibidas(tmp)` → `_backup_arquivo` se upsert → `shutil.move` → **[guardado]** `carregar_json(saved_path)` → `filtrar_documentos_vazios` → `SemanticChunker.split_documents` → `_enriquecer_metadados_chunks` → `store.add_documents_com_rollback(vectordb_instance, chunks, chunk_ids)` → `delete(old_ids)` se upsert → `invalidar_vector_db`. Mensagens e ordem idênticas ao atual.

**Failure path — falha de embedding no boundary (oráculo `teste_4`):**
`embed_documents` levanta dentro de `SemanticChunker.split_documents`, ANTES de `add_documents_com_rollback`; o `except` do bloco guardado chama `_reverter_disco(saved_path, backup_path)` e re-levanta; a coleção permanece intacta. Gatilho: `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", stub_que_falha)` — funciona porque getter/cache ficam em `rag.py` e o wrapper injeta o stub.

**Failure path — falha de `add_documents` (upsert):**
`store.add_documents_com_rollback` tenta `add_documents(chunks, ids=chunk_ids)`; na exceção tenta `delete(ids=chunk_ids)`, loga IDs órfãos se o rollback falhar, e re-levanta o erro ORIGINAL; o `except` do orquestrador chama `_reverter_disco` (restaura o backup do arquivo antigo).

**Rebuild (`cria_vectordb`):**
`rag.cria_vectordb` (WRAPPER) → `orchestrator.cria_vectordb(collection_name, docs_dir, vectordb_dir, embedding_model=None)` → valida paths/arquivos → itera `carregar_json` sobre `sorted(docs_dir.glob("*.json"))` + `extrair_texto_pdf` sobre `*.pdf` → `filtrar_documentos_vazios` → constrói `HuggingFaceEmbeddings` fresco (seam `None`) → `SemanticChunker` → enriquece por-source → `store._resetar_colecao(collection_name, vectordb_dir)` → `Chroma.from_documents` → guardrail `RuntimeError` se `count()==0`.

---

## API Changes

No API changes. Assinaturas públicas e caminhos de import antigos permanecem estáveis: os 4 símbolos de `store` via shims `is`-idênticos; os 5 orquestradores via WRAPPERS finos de mesma assinatura pública. `api.py` importa `_get_rag_embedding_model` de `agenticlog.rag` (preservado). Novos símbolos públicos passam a existir em `agenticlog.ingestion.store`/`orchestrator` (superset aditivo).

---

## Frontend Changes

No frontend changes. `app.py` importa `salvar_documento_enviado`/`salvar_pdf_enviado`/`reconstruir_vectordb` de `agenticlog.rag` — `reconstruir_vectordb` preservado via wrapper; os saves permanecem shims da Fase 3a.

---

## Tests Required

**Novos (ING3B-09):**
- **Acicidade** (novo teste de aceitação, ex. `tests/acceptance/test_rag_ingestion_fase3b.py`): `subprocess` importando `agenticlog.ingestion.store` e `agenticlog.ingestion.orchestrator` em interpretador frio → exit 0, sem "circular"/"partially initialized" (padrão de `test_rag_shared_observability.py::test_ac06`).
- **Identidade/delegação** (mesmo arquivo novo): `agenticlog.rag.X is agenticlog.ingestion.store.X` para os 4 símbolos de store; delegação-de-wrapper para os 5 orquestradores (o wrapper de `rag` NÃO é `is`-idêntico ao de `orchestrator` — é uma função distinta que delega ligando seams).
- **Rollback primitive**: unit de `store.add_documents_com_rollback` — caminho de sucesso (chama `add_documents`), caminho de falha (chama `delete(chunk_ids)`, loga órfãos se delete falha, re-levanta o original).
- **Seam binding** (unit): chamar `rag.adicionar_documento_incrementalmente` com `rag.DIR_VECTORDB` monkeypatchado e afirmar que `Chroma` (patchado no namespace de `orchestrator`) recebeu `persist_directory=str(<valor patchado>)`.
- **`cria_vectordb` sem `DirectoryLoader`** (ING3B-05): afirmar que `carregar_json` é chamado por-arquivo sobre o glob (reescrita de `TestCriaVectordb`, patchando `agenticlog.ingestion.orchestrator.*`).

**Edge cases:** retrieval vazio (coberto pelo oráculo); 0 chunks; upsert com falha de add; fail-fast de `RAGSecurityError`; guardrail de coleção vazia.

**Testes que QUEBRAM e migram ALVO de patch (ING3B-10 — só namespace, não comportamento):**
- `tests/test_rag.py`: `TestCriaVectordb` (reescrever fora de `DirectoryLoader` → `agenticlog.ingestion.orchestrator.*`), `TestResetarColecao` (→ `store._outras_colecoes_existem`), `TestOutrasColecoesExistem` (→ `store.DIR_VECTORDB` ou param), `TestAdicionarDocumentoIncrementalmente`, `TestAdicionarPdfIncrementalmente` (→ `orchestrator`/`store`: `Chroma`, `SemanticChunker`, `shutil`, `tempfile`, `uuid`, `extrair_texto_pdf`), `TestReconstruirVectordb` (→ `orchestrator.cria_vectordb`), `TestIngerirIncrementalmente` `teste_1/2/2b/2c` (→ `orchestrator`).
- `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (todos os métodos → `orchestrator`/`store`).
- `tests/acceptance/test_semantic_chunking.py`, `test_unificar_metadados_chunks.py`, `test_portuguese_embedding_model.py`: `agenticlog.rag._resetar_colecao` → `agenticlog.ingestion.orchestrator._resetar_colecao` ou `store`.

**Ficam intocados (verde SEM edição — verificar):** `tests/test_rag_caracterizacao.py` (oráculo — HARD); `tests/ingestion/test_shims_identidade.py` (oráculo de identidade 3a — HARD); `tests/acceptance/test_rag_ingestion_fase3a.py::test_ac05`; `TestLogging`, `TestStructuredLogConfig`; `TestIngerirIncrementalmente` `teste_3/4/5` (patcham shims chamados por `_executar_main`, que ficam em `rag.py`); `test_multi_collection_chromadb` (patcha o próprio shim/entry de `rag`).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/ingestion/store.py` | Novo | `_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`, `_resetar_colecao` (parametrizados por `vectordb_dir`), `add_documents_com_rollback` (primitiva de rollback de escrita no Chroma — dono da escrita). |
| `src/agenticlog/ingestion/orchestrator.py` | Novo | 5 orquestradores + helper compartilhado `_ingerir_arquivo_incrementalmente`; seams `docs_dir`/`vectordb_dir`/`embedding_model` opcionais (fallback config); usa `store`, `extraction.carregar_json`, `cleaning`, `chunking`, `metadata`, `embeddings`. |
| `src/agenticlog/ingestion/__init__.py` | Modificado | Re-exportar símbolos públicos novos + `__all__` (aditivo). |
| `src/agenticlog/rag.py` | Modificado (`≤420 ln`) | Remover corpos movidos; adicionar shim block de `store` (identidade `is`) + 5 WRAPPERS de orquestrador ligando seams; remover imports `DirectoryLoader`/`JSONLoader` (só usados por `cria_vectordb`); manter getter/cache de embedding, CLI e `import fitz`. |
| `tests/acceptance/test_rag_ingestion_fase3b.py` | Novo | Acicidade `store`+`orchestrator`; identidade dos 4 símbolos de store; delegação dos wrappers; verificação AC1–AC9. |
| `tests/ingestion/test_store.py` | Novo | Unit de `add_documents_com_rollback` (sucesso/falha/órfãos) + `_outras_colecoes_existem`/`_resetar_colecao` parametrizados por `vectordb_dir`. |
| `tests/test_rag.py` | Modificado | Migração de patch-target (namespace) das classes de orquestrador/reset/ingerir; reescrita de `TestCriaVectordb` fora de `DirectoryLoader`. NÃO alterar comportamento verificado. |
| `tests/acceptance/test_adicionar_pdf_incrementalmente.py` | Modificado | Patch-target → `orchestrator`/`store`. |
| `tests/acceptance/test_semantic_chunking.py` | Modificado | `_resetar_colecao` → `orchestrator`/`store`. |
| `tests/acceptance/test_unificar_metadados_chunks.py` | Modificado | `_resetar_colecao` → `orchestrator`/`store`. |
| `tests/acceptance/test_portuguese_embedding_model.py` | Modificado | `_resetar_colecao` → `orchestrator`/`store` (in-place; NÃO tocar testes de inspeção de fonte). |
| `docs/adr/ADR-019-reprojecao-atomicidade-upsert-fase3b.md` | Novo | Documenta a decisão de reprojeção da atomicidade de upsert (seams de DI + shims/wrappers). |
| `tests/test_rag_caracterizacao.py` | **NÃO tocar** | Oráculo de caracterização (HARD constraint — zero diff). |
| `tests/ingestion/test_shims_identidade.py` | **NÃO tocar** | Oráculo de identidade da Fase 3a (HARD constraint — zero diff). |

---

## Risks

- **[Alto] Seam ligado em tempo de import vs. de chamada:** se um orquestrador movido lesse `DIR_DOCUMENTS`/`DIR_VECTORDB` como nome de nível de módulo capturado no import de `orchestrator.py`, ele capturaria o valor PRÉ-monkeypatch e o oráculo quebraria. **Mitigação (ING3B-04):** os wrappers de `rag.py` leem `rag.DIR_DOCUMENTS`/`rag.DIR_VECTORDB` NO MOMENTO DA CHAMADA e os passam como argumentos; a função movida usa os parâmetros (fallback `config.*` quando `None`). Detalhado em `design.md §4` e ADR-019.
- **[Alto] Distinção identidade-de-store vs. delegação-de-orquestrador:** os 4 símbolos de store são shims `is`-idênticos (como Fase 3a); os 5 orquestradores são WRAPPERS (não `is`-idênticos) porque precisam ligar seams. **Mitigação:** o novo teste afirma `is` SÓ para store e delegação para os wrappers; `test_shims_identidade.py` (que só lista símbolos 3a) fica com zero diff.
- **[Alto] Atomicidade de upsert (AC7):** reordenar o bloco guardado quebraria a garantia de rollback. **Mitigação:** a extração da primitiva `add_documents_com_rollback` e do helper compartilhado preserva a sequência exata `backup→move→chunk→add→rollback→_reverter_disco`; ver `design.md §5` (DN-2 ordenação do bloco guardado; DN-3 mensagem de invalidação por-tipo).
- **[Médio] `cria_vectordb` sem `DirectoryLoader` (AC5):** troca de mecanismo de carga num caminho NÃO coberto pelo oráculo. **Mitigação:** iterar `carregar_json` sobre `sorted(docs_dir.glob("*.json"))` (ordem determinística); reescrever `TestCriaVectordb`; a semântica por-arquivo de `carregar_json` (jq_schema compartilhado) é idêntica à do `loader_cls=JSONLoader` do `DirectoryLoader`.
- **[Médio] Precedência de exceções:** `_sanitizar_nome_colecao` deve rodar ANTES da validação de tipo. **Mitigação:** o wrapper/orquestrador preserva a ordem; edge case coberto por teste.
- **[Médio] Import circular:** `store.py`/`orchestrator.py` não podem importar `rag`/`agent`. `orchestrator → store` e `orchestrator → {extraction,cleaning,chunking,metadata,embeddings}` são arestas intra-pacote (DAG). **Mitigação (ING3B-09):** teste de acicidade em interpretador frio.
- **[Médio] Fallback de `embedding_model` no caminho incremental:** seam `None` numa chamada direta ao orquestrador construiria o modelo real (~1 GB). **Mitigação:** o caminho de produção/oráculo sempre passa pelo wrapper de `rag` (injeta o singleton patchado); documentar que chamadas diretas ao orquestrador devem fornecer `embedding_model` (ver `design.md §4.3`).
- **[Baixo] `normalize_embeddings` (silent-degradation):** o rebuild constrói `HuggingFaceEmbeddings(..., encode_kwargs={"normalize_embeddings": True})` verbatim quando o seam é `None`. **Mitigação:** unit test asserta os kwargs exatos.
- **[Ambiental] hnswlib SAC no Windows:** oráculo/testes Chroma são *skipped* local (Windows/Smart App Control); o CI Linux é o gate autoritativo (`tests/conftest.py`).
- **CLAUDE.md conflicts:** nenhum. Alinha com "small files" (`rag.py ≤420`, módulos 200–400), type hints, docstrings PT, imutabilidade, commits Conventional em PT.

---

## Open Questions

None. As decisões de design (D1 primitiva de rollback; D2 seam de embedding no rebuild; D3 localização do helper de dedup) e as mecânicas de seam foram resolvidas no Checkpoint 1 e estão baked em `design.md` + ADR-019. Não reabrir.

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura ≥ 80%.
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` verde; `git diff --stat tests/test_rag_caracterizacao.py` VAZIO.
- [ ] `git diff --stat tests/ingestion/test_shims_identidade.py` VAZIO; arquivo verde.
- [ ] `rag.py ≤420 ln`; `store.py` e `orchestrator.py` dentro de 200–400 ln (store pode ficar levemente abaixo — anotado no design).
- [ ] `agenticlog.rag.X is agenticlog.ingestion.store.X` para os 4 símbolos de store (testado); wrappers dos 5 orquestradores delegam (testado).
- [ ] `import agenticlog.ingestion.store` e `import agenticlog.ingestion.orchestrator` saem 0 em interpretador frio.
- [ ] `cria_vectordb` não referencia `DirectoryLoader`; itera `carregar_json`.
- [ ] `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings em PT (Entrada/Saída/Lança).
- [ ] ADR-019 escrito documentando a reprojeção da atomicidade de upsert.

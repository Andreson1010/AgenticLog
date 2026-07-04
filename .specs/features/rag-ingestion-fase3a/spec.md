# RAG Ingestion Fase 3a — extração dos estágios de ingestão para `agenticlog.ingestion` — Technical Spec

**Path:** `.specs/features/rag-ingestion-fase3a/spec.md`
**TLC scope:** complex
**Based on story:** ADR-018 Fase 3a — extrair estágios de ingestão de baixo risco de `rag.py` para o pacote novo `src/agenticlog/ingestion/` via shim de re-export identity-preserving, com ZERO mudança de comportamento.
**Status:** Awaiting human approval

---

## Problem Statement

`src/agenticlog/rag.py` acumula ~1010 linhas misturando validação de segurança, extração (PDF/JSON), limpeza, chunking, embeddings, enriquecimento de metadados e os orquestradores de ingestão — violando a regra de arquivo pequeno (200–400 linhas) e dificultando manutenção/teste. A ADR-018 define um redesenho em fases. A **Fase 3a** extrai apenas os estágios de baixo risco para um pacote `agenticlog.ingestion`, mantendo os orquestradores, a escrita no Chroma, a atomicidade de upsert e a CLI em `rag.py` (reprojetados na Fase 3b). O refactor é guardado por uma **rede de caracterização** (`tests/test_rag_caracterizacao.py`) que NÃO pode ser modificada.

## Goals

- [ ] Criar `src/agenticlog/ingestion/` com 6 módulos + `__init__.py`, cada um pequeno e coeso.
- [ ] Mover os estágios de baixo risco (security, extraction, cleaning, chunking, embeddings-construction, metadata) preservando identidade de símbolo em `rag.py` via shims.
- [ ] ZERO mudança de comportamento observável (fluxo, mensagens, ordem, atomicidade, espaço vetorial).
- [ ] Manter o oráculo de caracterização (5 testes `@integration`) verde **sem qualquer edição**.
- [ ] Manter o grafo de imports acíclico (`ingestion/*` importa só de `config` e `shared`).
- [ ] Migrar apenas os testes NÃO-oráculo cujos seams de patch se deslocaram, com alvos exatos.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Mover `cria_vectordb`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente` | Orquestradores permanecem em `rag.py`; reprojetados na Fase 3b. |
| Mover escrita no Chroma / `_backup_arquivo` / `_reverter_disco` / `shutil.move` / `_resetar_colecao` / `_outras_colecoes_existem` | Atomicidade de upsert e reset de coleção ficam em `rag.py` (Fase 3b). |
| Mover a CLI (`_executar_main`, `_configurar_logging_cli`) | Permanece em `rag.py`. |
| Mover o singleton global `_rag_embedding_model` e o getter `_get_rag_embedding_model` | Ficam em `rag.py` para preservar o monkeypatch do oráculo (ver RAGING-06). |
| Converter o `DirectoryLoader` de `cria_vectordb` para `carregar_json` | `carregar_json` modela o caminho single-file (orquestrador incremental). O loader directory-batch do rebuild é reprojetado na Fase 3b (ver Design Note DN-1). |
| Remover shims / `import fitz` | Removidos apenas na Fase 6. |
| Mudança de comportamento, config ou espaço vetorial | Estritamente proibido nesta fase. |

---

## User Stories

### P1: Pacote `ingestion` com estágios extraídos e comportamento idêntico ⭐ MVP

**User Story**: Como mantenedor do AgenticLog, quero os estágios de ingestão de baixo risco em módulos pequenos e coesos sob `agenticlog.ingestion`, para reduzir o tamanho de `rag.py` e habilitar a Fase 3b, sem alterar nenhum comportamento observável.

**Why P1**: É o objetivo único e indivisível da fase; sem ele não há entrega.

**Acceptance Criteria**:
1. WHEN o pacote é criado THEN o sistema SHALL expor `src/agenticlog/ingestion/{security,extraction,cleaning,chunking,embeddings,metadata}.py` + `__init__.py`.
2. WHEN um símbolo é movido THEN `agenticlog.rag.X` SHALL resolver para o MESMO objeto que `agenticlog.ingestion.<mod>.X` (identidade: `is`).
3. WHEN qualquer caminho de import antigo é usado (`from agenticlog.rag import X`) THEN SHALL resolver sem `ImportError`.
4. WHEN a suíte completa roda no CI Linux THEN o oráculo `tests/test_rag_caracterizacao.py` SHALL passar **sem nenhuma modificação no arquivo**.
5. WHEN `import agenticlog.ingestion` roda num interpretador frio THEN SHALL sair com código 0 (sem ciclo de import).

**Independent Test**: `pytest -m integration tests/test_rag_caracterizacao.py -v` verde sem diff no arquivo do oráculo; `pytest` completo verde no CI Linux.

---

## Edge Cases

- WHEN um teste patcha `agenticlog.rag.fitz.open` THEN `extraction.extrair_texto_pdf` SHALL ainda ver o patch (fitz é singleton em `sys.modules`; `import fitz` permanece em `rag.py`).
- WHEN um teste patcha `agenticlog.rag.SemanticChunker` THEN o orquestrador SHALL ainda ver o patch (re-import bare em `rag.py`; sem factory).
- WHEN `_get_rag_embedding_model` é chamado com `agenticlog.rag._rag_embedding_model` monkeypatchado THEN SHALL retornar a instância patchada sem construir o modelo real (cache/getter ficam em `rag.py`).
- WHEN `salvar_pdf_enviado` (agora em `security.py`) valida um PDF THEN SHALL chamar `extraction.extrair_texto_pdf` (intra-pacote, não circular).
- WHEN um Document tem `page_content` só com whitespace THEN `cleaning.filtrar_documentos_vazios` SHALL descartá-lo, preservando a semântica atual (`.strip()`).

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| RAGING-01 (pacote + 6 módulos + `__init__`) | P1-AC1 | Design | Pending |
| RAGING-02 (`security.py`: validações + saves + sanitizers) | P1-AC1/AC2 | Design | Pending |
| RAGING-03 (`extraction.py`: `extrair_texto_pdf` movido + `carregar_json` novo) | P1-AC1/AC2 | Design | Pending |
| RAGING-04 (`cleaning.py`: `filtrar_documentos_vazios` novo, 4 sites) | P1-AC1/AC2 | Design | Pending |
| RAGING-05 (`chunking.py`: re-export `SemanticChunker`, sem factory) | P1-AC1/AC2 | Design | Pending |
| RAGING-06 (`embeddings.py`: factory `criar_embedding_model`; getter delega; cache fica em `rag.py`) | P1-AC1/AC2 | Design | Pending |
| RAGING-07 (`metadata.py`: hash + enriquecimento movidos) | P1-AC1/AC2 | Design | Pending |
| RAGING-08 (bloco de shims em `rag.py`; identidade; `import fitz` mantido) | P1-AC2/AC3 | Design | Pending |
| RAGING-09 (orquestradores roteiam por `carregar_json` incremental + `filtrar_documentos_vazios`) | P1-AC1 | Design | Pending |
| RAGING-10 (acicidade de imports: `ingestion/*` importa só `config`+`shared`) | P1-AC5 | Design | Pending |
| RAGING-11 (oráculo verde SEM edição — HARD) | P1-AC4 | Design | Pending |
| RAGING-12 (migração de patch-target dos testes NÃO-oráculo afetados) | P1-AC1/AC4 | Tasks | Pending |
| RAGING-13 (testes novos: módulos novos + round-trip de identidade + acicidade `ingestion`) | P1-AC2/AC5 | Tasks | Pending |

**ID format:** `RAGING-[NUMBER]` (RAGING = RAG INGestion).

---

## Data Model Changes

No data model changes. Nenhuma alteração em schema JSON, metadados de chunk, coleção Chroma ou espaço vetorial. A flag `encode_kwargs={"normalize_embeddings": True}` é preservada verbatim (crítica para silent-degradation).

---

## Process / Background Flow

**Happy path (ingestão incremental JSON — oráculo AC-4):** `adicionar_documento_incrementalmente` (em `rag.py`) → `_sanitizar_nome_colecao`/`_sanitizar_nome_arquivo` (security) → `_computar_hash_conteudo` (metadata) → `_get_rag_embedding_model` (rag; delega construção a `embeddings.criar_embedding_model`) → grava temp → `carregar_json(saved_path)` (extraction) → `filtrar_documentos_vazios` (cleaning) → `SemanticChunker` (chunking re-export) → `_enriquecer_metadados_chunks` (metadata) → `Chroma.add_documents` (rag) → `invalidar_vector_db`. Comportamento e mensagens idênticos ao atual.

**Failure path — falha de embedding no boundary (oráculo AC-5):** `embed_documents` levanta dentro do `SemanticChunker.split_documents`, ANTES do `add_documents`; `_reverter_disco` (rag) restaura o disco; a coleção permanece intacta. O gatilho é `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", stub_que_falha)` — funciona porque o getter/cache ficam em `rag.py`.

---

## API Changes

No API changes. Endpoints, assinaturas públicas e caminhos de import antigos permanecem estáveis (via shims). Novos símbolos públicos passam a existir em `agenticlog.ingestion` (superset aditivo).

---

## Frontend Changes

No frontend changes. `app.py` importa `salvar_documento_enviado`/`salvar_pdf_enviado`/`reconstruir_vectordb` de `agenticlog.rag` — todos preservados via shim.

---

## Tests Required

**Unit (novos — RAGING-13):**
- `ingestion/cleaning.py::filtrar_documentos_vazios` — descarta vazio/whitespace, preserva não-vazio; retorna lista nova (imutabilidade).
- `ingestion/extraction.py::carregar_json` — usa `JQ_SCHEMA_CAMPOS_JSON`, retorna `list[Document]`, é parametrizado por path.
- `ingestion/embeddings.py::criar_embedding_model` — constrói `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": ...}, encode_kwargs={"normalize_embeddings": True})`.
- `ingestion/metadata.py` — hash determinístico, enriquecimento in-place (paridade com testes atuais).

**Integração / identidade (novos — RAGING-13):**
- Round-trip de identidade: `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` para cada símbolo movido.
- Extensão da acicidade: `subprocess` importando `agenticlog.ingestion` em interpretador frio → exit 0, sem "circular".

**Edge case:** manter cobertura de retrieval vazio já existente (oráculo AC-2).

**Existing tests that will break (migração dirigida — RAGING-12):**
- `tests/test_rag.py::TestGetRagEmbeddingModel` → `HuggingFaceEmbeddings` patch para `agenticlog.ingestion.embeddings`.
- `tests/acceptance/test_portuguese_embedding_model.py::teste_1_get_rag_embedding_model_usa_embedding_model_do_config` → idem (NÃO os testes de `cria_vectordb` nem os de inspeção de fonte).
- Todos os `@patch("agenticlog.rag.JSONLoader")` do caminho incremental → `agenticlog.ingestion.extraction.JSONLoader`.
- `TestValidaPathDocumentos`, `TestValidaArquivosJson`, `TestSalvarDocumentoEnviado`, `TestSalvarPdfEnviado` → patches de `DIR_DOCUMENTS`/`PROJECT_ROOT`/`MAX_JSON_FILES`/`extrair_texto_pdf`/`_valida_json_sem_chaves_proibidas` para os alvos `agenticlog.ingestion.security.*` (e `extrair_texto_pdf` via `security`).

**Stay green SEM edição (verificado):** oráculo; `TestExtrairTextoPdf` (patch em `rag.fitz.open`); `TestValidaJsonSemChavesProibidas`; `TestSanitizarNomeArquivo/Colecao`; toda a `TestCriaVectordb`/`TestLogging`; ~30 patches `rag.SemanticChunker`; ~14 testes de `adicionar_pdf_incrementalmente` patchando `rag.extrair_texto_pdf`; testes de inspeção de fonte de `test_portuguese_embedding_model`.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/ingestion/__init__.py` | Novo | Re-export dos símbolos públicos + `__all__` (precedente `shared`/`observability`). |
| `src/agenticlog/ingestion/security.py` | Novo | `_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, `_valida_arquivos_json`, `_sanitizar_nome_arquivo`, `_sanitizar_nome_colecao`, `sanitizar_nome_colecao`, `salvar_documento_enviado`, `salvar_pdf_enviado`. |
| `src/agenticlog/ingestion/extraction.py` | Novo | `extrair_texto_pdf` (movido verbatim) + `carregar_json` (novo wrapper de `JSONLoader`). |
| `src/agenticlog/ingestion/cleaning.py` | Novo | `filtrar_documentos_vazios` (extraído de 4 sites inline). |
| `src/agenticlog/ingestion/chunking.py` | Novo | Re-export de `SemanticChunker` (sem factory). |
| `src/agenticlog/ingestion/embeddings.py` | Novo | `criar_embedding_model()` factory (construção verbatim de rag.py:75-83). |
| `src/agenticlog/ingestion/metadata.py` | Novo | `_computar_hash_conteudo`, `_hash_arquivo`, `_enriquecer_metadados_chunks`. |
| `src/agenticlog/rag.py` | Modificado | Remover corpos movidos; adicionar bloco de shims (identidade); getter delega ao factory; orquestrador incremental usa `carregar_json`; orquestradores usam `filtrar_documentos_vazios`; manter `import fitz` (F401), `HuggingFaceEmbeddings` (cria_vectordb), `DirectoryLoader`+`JSONLoader` (cria_vectordb). |
| `tests/test_rag.py` | Modificado (reduzido ~700-900 ln) | **Emenda pós-CP2:** classes dos estágios extraídos são MOVIDAS p/ `tests/ingestion/` (não editadas in-place); ficam só orquestradores + logging. |
| `tests/ingestion/test_security.py` | Novo (move) | Recebe TestValidaPath/Json/Arquivos, TestSanitizar×2, TestSalvarDocumento/Pdf com alvos `ingestion.security.*`. |
| `tests/ingestion/test_extraction.py` | Novo (move) | Recebe TestExtrairTextoPdf; patch `rag.fitz.open` → `ingestion.extraction.fitz.open`. |
| `tests/ingestion/test_embeddings.py` | Novo (move) | Recebe TestGetRagEmbeddingModel + TestEmbeddingModelConfig; patch → `ingestion.embeddings.HuggingFaceEmbeddings`. |
| `tests/ingestion/test_metadata.py` | Novo (move) | Recebe TestComputarHash + TestMetadadosUnificados. |
| `tests/acceptance/test_portuguese_embedding_model.py` | Modificado | `teste_1` migra para `ingestion.embeddings` (in-place). |
| `tests/test_rag_integration.py` | Modificado | `JSONLoader` → extraction (in-place). |
| `tests/acceptance/test_multi_collection_chromadb.py` | Modificado | `JSONLoader` → extraction (in-place). |
| `tests/acceptance/test_semantic_chunking.py` | Modificado | `JSONLoader` → extraction (in-place). |
| `tests/acceptance/test_unificar_metadados_chunks.py` | Modificado | `JSONLoader` → extraction (in-place). |
| `tests/acceptance/test_rag_shared_observability.py` | Modificado | Estender acicidade para `agenticlog.ingestion`. |
| `tests/ingestion/test_shims_identidade.py` (ou similar) | Novo | Unit dos módulos novos + round-trip de identidade + acicidade `ingestion`. |
| `tests/test_rag_caracterizacao.py` | **NÃO tocar** | Oráculo (HARD constraint). |

---

## Risks

- **[Alto] Oráculo / singleton de embedding (mitigado):** mover o global `_rag_embedding_model` ou o getter para `embeddings.py` tornaria `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", ...)` invisível ao getter → carrega o modelo real (~1 GB) → oráculo quebra. **Mitigação (RAGING-06):** só a CONSTRUÇÃO vai para `criar_embedding_model()`; cache/getter ficam em `rag.py`.
- **[Alto] Seam de monkeypatch de funções movidas:** shims preservam identidade de símbolo e imports, mas NÃO preservam o lookup de nomes de módulo dentro de uma função cujo corpo mudou de módulo. Testes que chamam DIRETAMENTE uma função movida e patcham uma dependência dela em `agenticlog.rag.<dep>` quebram. **Mitigação (RAGING-12):** inventário fechado e migração dirigida (security + JSONLoader incremental + embeddings). O oráculo é imune porque os orquestradores (que leem os nomes patchados) ficam em `rag.py`.
- **[Médio] Referência `fitz` (mitigado):** 6 testes patcham `agenticlog.rag.fitz.open`. **Mitigação:** manter `import fitz  # noqa: F401` em `rag.py`; `fitz` é singleton em `sys.modules`, então `extraction.extrair_texto_pdf` vê o patch.
- **[Médio] Import circular:** `ingestion/*` deve importar SÓ de `config` e `shared` (nunca `rag`/`agent`). **Mitigação (RAGING-10):** teste de acicidade em interpretador frio estendido para `agenticlog.ingestion`.
- **[Baixo] `carregar_json` vs `DirectoryLoader`:** `carregar_json` é single-file; o rebuild usa `DirectoryLoader` directory-batch. Ver Design Note DN-1 — o rebuild NÃO é convertido nesta fase (evita risco em caminho não coberto pelo oráculo e churn indevido em `TestCriaVectordb`).
- **[Baixo] `normalize_embeddings` (silent-degradation):** a flag deve migrar VERBATIM para o factory; qualquer omissão degrada similaridade sem erro. **Mitigação:** unit test asserta os kwargs exatos.
- **[Ambiental] hnswlib SAC no Windows:** oráculo/testes Chroma são skipped local (Windows/Smart App Control); o CI Linux é o gate autoritativo (ver `tests/conftest.py`).
- **CLAUDE.md conflicts:** nenhum. Alinha com "small files", type hints, docstrings PT, imutabilidade (cleaning retorna lista nova), commits Conventional em PT.

---

## Open Questions

None. As 3 open questions foram resolvidas no Checkpoint 1 (Q1 MOVE com double-shim; Q2 CREATE `carregar_json`; Q3 INCLUDE re-exportando `SemanticChunker`). Ver Design Note DN-1 em `design.md` para a clarificação de escopo do `DirectoryLoader` de `cria_vectordb` (não é uma questão aberta — é a leitura que honra todos os hard constraints).

---

## Success Criteria

- [ ] `pytest` completo verde no CI Linux (oráculo incluso), com `tests/test_rag_caracterizacao.py` sem diff.
- [ ] `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` para cada símbolo movido (testado).
- [ ] `import agenticlog.ingestion` em interpretador frio sai 0.
- [ ] `rag.py` reduzido; cada módulo `ingestion/*` dentro de 200–400 linhas.
- [ ] `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings em PT.
- [ ] Marcador `# Re-export shim (ADR-018 Fase 3a) — remover na Fase 6` em cada linha de shim.

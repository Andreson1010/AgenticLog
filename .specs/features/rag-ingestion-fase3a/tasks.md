# RAG Ingestion Fase 3a — Tasks

**Path:** `.specs/features/rag-ingestion-fase3a/tasks.md`
**Spec:** `.specs/features/rag-ingestion-fase3a/spec.md` · **Design:** `.specs/features/rag-ingestion-fase3a/design.md`
**TLC scope:** complex
**Status:** Awaiting human approval

Estas tasks são para o backend-builder da feature-factory. Ordem projetada para que o oráculo + suíte completa possam rodar verdes ao final. TLC Execute NÃO é usado. Gate autoritativo: **CI Linux** (`pytest --cov=agenticlog -v`). Local Windows pula testes Chroma/hnswlib (Smart App Control) — não confundir com regressão. Padrões: `ruff`/`black`/`isort` limpos, type hints, docstrings PT.

**Regra de ouro (todas as tasks):** NUNCA editar `tests/test_rag_caracterizacao.py`. Preservar comportamento, mensagens, ordem, atomicidade e a flag `encode_kwargs={"normalize_embeddings": True}` verbatim.

---

## Fase A — Criar módulos folha (sem dependência de ordem entre si, exceto extraction antes de security)

### T1 — `ingestion/__init__.py` (esqueleto do pacote) · [RAGING-01]
- Criar `src/agenticlog/ingestion/__init__.py` com docstring PT (precedente `shared/__init__.py`, `observability/__init__.py`). Re-exports/`__all__` preenchidos incrementalmente nas tasks seguintes.
- **Done when:** `python -c "import agenticlog.ingestion"` sai 0.
- **Depends on:** —

### T2 — `ingestion/metadata.py` · [RAGING-07]
- Mover verbatim `_computar_hash_conteudo`, `_hash_arquivo`, `_enriquecer_metadados_chunks` (rag.py:278-305). Imports: `hashlib`, `Path`, `METADATA_FILE_HASH, METADATA_CHUNK_INDEX, METADATA_PAGE, METADATA_DOC_TYPE` de `config`.
- **Done when:** módulo importa isolado; funções idênticas ao original.
- **Depends on:** T1

### T3 — `ingestion/cleaning.py` · [RAGING-04]
- Criar `filtrar_documentos_vazios(docs: list) -> list` retornando `[d for d in docs if d.page_content.strip()]` (lista nova — imutabilidade). Docstring PT citando ADR-011.
- **Done when:** unit test T14 verde.
- **Depends on:** T1

### T4 — `ingestion/chunking.py` · [RAGING-05]
- `from langchain_experimental.text_splitter import SemanticChunker` + `__all__ = ["SemanticChunker"]`. SEM factory.
- **Done when:** `agenticlog.ingestion.chunking.SemanticChunker is langchain_experimental...SemanticChunker`.
- **Depends on:** T1

### T5 — `ingestion/embeddings.py` · [RAGING-06]
- Criar `criar_embedding_model() -> HuggingFaceEmbeddings` com o corpo VERBATIM de rag.py:75-83 (device auto + `model_kwargs={"device": device}` + `encode_kwargs={"normalize_embeddings": True}`). Imports: `HuggingFaceEmbeddings`, `torch`, `EMBEDDING_MODEL`. NÃO incluir global nem getter.
- **Done when:** unit test T15 verde (kwargs exatos).
- **Depends on:** T1

### T6 — `ingestion/extraction.py` · [RAGING-03]
- Mover verbatim `extrair_texto_pdf` (rag.py:703-733). Adicionar `carregar_json(caminho: Path) -> list[Document]` que instancia `JSONLoader(str(caminho), jq_schema=JQ_SCHEMA_CAMPOS_JSON)` e retorna `.load()`. Imports: `fitz`, `RAGSecurityError` (de `shared.errors`), `JSONLoader`, `Document`, `JQ_SCHEMA_CAMPOS_JSON`.
- **Done when:** unit tests T16 verdes; `extrair_texto_pdf` idêntico.
- **Depends on:** T1

### T7 — `ingestion/security.py` · [RAGING-02]
- Mover verbatim as 8 funções (`_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, `_valida_arquivos_json`, `_sanitizar_nome_arquivo`, `_sanitizar_nome_colecao`, `sanitizar_nome_colecao`, `salvar_documento_enviado`, `salvar_pdf_enviado`). Imports conforme design §2.1, incluindo `from agenticlog.ingestion.extraction import extrair_texto_pdf`.
- **Done when:** módulo importa isolado; `security` não importa `rag`/`agent`.
- **Depends on:** T6

### T8 — Preencher `ingestion/__init__.py` · [RAGING-01]
- Re-exportar símbolos públicos e declarar `__all__`.
- **Done when:** `from agenticlog.ingestion import salvar_documento_enviado` resolve.
- **Depends on:** T2–T7

---

## Fase B — Religar `rag.py` (orquestradores + shims)

### T9 — Delegar o getter de embedding · [RAGING-06]
- Manter `_rag_embedding_model` (global) e `_get_rag_embedding_model` em `rag.py`; trocar o corpo do getter para `_rag_embedding_model = criar_embedding_model()`. Manter `import ... HuggingFaceEmbeddings` (usado por `cria_vectordb`) e `import torch` se ainda usado por `cria_vectordb`.
- **Done when:** oráculo AC-1/AC-5 verdes (getter retorna o stub patchado sem construir o modelo real).
- **Depends on:** T5

### T10 — Rotear orquestradores por `carregar_json` + `filtrar_documentos_vazios` · [RAGING-09]
- `adicionar_documento_incrementalmente`: `JSONLoader(...).load()` → `carregar_json(saved_path)`; filtro inline → `filtrar_documentos_vazios(...)`.
- `adicionar_pdf_incrementalmente` e `cria_vectordb`: filtros inline (rag.py ~636, ~835, ~853) → `filtrar_documentos_vazios(...)`.
- NÃO tocar o `DirectoryLoader` de `cria_vectordb` (DN-1). Manter `JSONLoader`/`DirectoryLoader`/`HuggingFaceEmbeddings` importados em `rag.py`.
- **Done when:** oráculo AC-4 verde; comportamento/mensagens idênticos.
- **Depends on:** T3, T6

### T11 — Remover corpos movidos + bloco de shims + manter `import fitz` · [RAGING-08]
- Remover de `rag.py` os corpos agora em `ingestion/*`. Adicionar o bloco de shims (design §4) com marcador `# Re-export shim (ADR-018 Fase 3a) — remover na Fase 6` e `# noqa` conforme necessário. Manter `import fitz  # noqa: F401`.
- **Done when:** `python -c "import agenticlog.rag"` sai 0; `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` para todo símbolo movido (validado por T17).
- **Depends on:** T8, T9, T10

---

## Fase C — MOVER classes de teste dos estágios para `tests/ingestion/` (RAGING-12)

> **Emenda pós-Checkpoint 2 (aprovada pelo usuário):** em vez de editar patch-target in-place em
> `tests/test_rag.py` (2419 ln), MOVER as classes que testam os estágios agora extraídos para arquivos
> por-módulo em `tests/ingestion/`, JÁ com os alvos de patch corrigidos. Encolhe `test_rag.py` ~700-900 ln,
> espelha a fonte, e é down-payment da reescrita de testes da Fase 6 (remover `@patch("agenticlog.rag.*")`).
> Regra: mover o corpo da classe verbatim; ao mover, repontar SÓ os alvos de patch que agora vivem no módulo
> novo; NÃO alterar asserções de comportamento. Rodar cada arquivo após mover. As classes de ORQUESTRADOR
> (TestCriaVectordb, TestAdicionar*, TestResetar*, TestIngerir*, TestReconstruir*, TestOutrasColecoes*,
> TestLogging, TestStructuredLogConfig, TestRAGSecurityError) FICAM em `test_rag.py` — patcham chamadores que
> permanecem em `rag.py`.

### T12 — Criar `tests/ingestion/` + mover security e extraction · [RAGING-12]
- Criar `tests/ingestion/__init__.py` (vazio) se necessário à descoberta.
- **`tests/ingestion/test_security.py`** — mover de `test_rag.py`: `TestValidaPathDocumentos`,
  `TestValidaJsonSemChavesProibidas`, `TestValidaArquivosJson`, `TestSanitizarNomeArquivo`,
  `TestSanitizarNomeColecao`, `TestSalvarDocumentoEnviado`, `TestSalvarPdfEnviado`. Repontar alvos que
  se moveram para `agenticlog.ingestion.security.*`: `DIR_DOCUMENTS`, `PROJECT_ROOT`, `MAX_JSON_FILES`,
  `_valida_json_sem_chaves_proibidas`; e `extrair_texto_pdf` (usado por `salvar_pdf_enviado`) →
  `agenticlog.ingestion.security.extrair_texto_pdf` (use-site) OU `agenticlog.ingestion.extraction.extrair_texto_pdf`
  conforme onde o nome é lido no corpo movido — validar rodando.
- **`tests/ingestion/test_extraction.py`** — mover `TestExtrairTextoPdf`. Repontar `@patch("agenticlog.rag.fitz.open")`
  → `@patch("agenticlog.ingestion.extraction.fitz.open")`. (Consequência: se nenhum outro teste patcha `rag.fitz`,
  o `import fitz # noqa: F401` em `rag.py` deixa de ser necessário — T11 pode removê-lo; se houver qualquer
  dúvida, manter é inócuo.)
- **Done when:** `pytest tests/ingestion/test_security.py tests/ingestion/test_extraction.py -q` verde; as classes
  já NÃO existem em `test_rag.py`.
- **Depends on:** T11

### T13 — Mover embeddings + metadata; migrar `JSONLoader` restante in-place · [RAGING-12]
- **`tests/ingestion/test_embeddings.py`** — mover `TestGetRagEmbeddingModel` e `TestEmbeddingModelConfig`.
  Em `TestGetRagEmbeddingModel`: `@patch("agenticlog.rag.HuggingFaceEmbeddings")` →
  `@patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")`; manter `rag._rag_embedding_model = None`
  no setUp/tearDown (cache fica em `rag.py`).
- **`tests/ingestion/test_metadata.py`** — mover `TestComputarHash` e `TestMetadadosUnificados`.
- `tests/acceptance/test_portuguese_embedding_model.py::teste_1_...`: migrar patch → `agenticlog.ingestion.embeddings.HuggingFaceEmbeddings`
  in-place (arquivo de acceptance, fica onde está). NÃO tocar os testes de `cria_vectordb` nem os de inspeção de fonte.
- `JSONLoader` do caminho incremental em arquivos que NÃO movem: `@patch("agenticlog.rag.JSONLoader")` →
  `@patch("agenticlog.ingestion.extraction.JSONLoader")` in-place em: `tests/test_rag_integration.py` (~139),
  `tests/acceptance/test_multi_collection_chromadb.py` (~512), `tests/acceptance/test_semantic_chunking.py` (~110,138),
  `tests/acceptance/test_unificar_metadados_chunks.py` (~275,375,420). Os patches de `JSONLoader` que estavam nas
  classes movidas de `test_rag.py` já viajam com elas (repontar no destino).
- **Done when:** arquivos movidos + editados verdes; `test_rag.py` reduzido (só orquestradores + logging).
- **Depends on:** T11

---

## Fase D — Testes novos (RAGING-13)

### T14 — Unit `cleaning.filtrar_documentos_vazios`
- Casos: descarta `""`, `"   "`; preserva `"CAMPO_VAZIO: "` (`.strip()` não-vazio) e conteúdo normal; retorna lista nova (não muta a entrada).
- **Depends on:** T3

### T15 — Unit `embeddings.criar_embedding_model`
- Patchar `agenticlog.ingestion.embeddings.HuggingFaceEmbeddings`; assert chamado com `model_name=config.EMBEDDING_MODEL, model_kwargs={"device": ANY}, encode_kwargs={"normalize_embeddings": True}`.
- **Depends on:** T5

### T16 — Unit `extraction.carregar_json` (+ paridade `extrair_texto_pdf`)
- `carregar_json`: patchar `agenticlog.ingestion.extraction.JSONLoader`; assert instanciado com `jq_schema=config.JQ_SCHEMA_CAMPOS_JSON` e que o retorno é o `.load()` do loader; parametrizado por path.
- Opcional: 1 smoke de `extrair_texto_pdf` via `agenticlog.ingestion.extraction.fitz.open` (paridade com `TestExtrairTextoPdf`).
- **Depends on:** T6

### T17 — Round-trip de identidade dos shims + acicidade de `ingestion` · [RAGING-13, RAGING-10]
- Novo arquivo (ex. `tests/test_ingestion_package.py`): para cada símbolo movido, `assert getattr(agenticlog.rag, nome) is getattr(agenticlog.ingestion.<mod>, nome)`.
- Estender a rede de acicidade (padrão de `tests/acceptance/test_rag_shared_observability.py::test_ac06...`): `subprocess.run([sys.executable, "-c", "import agenticlog.ingestion"])` → returncode 0 e sem "circular" em stderr. Adicionar como novo teste (não modificar o oráculo).
- **Done when:** verde.
- **Depends on:** T11

---

## Fase E — Gate final

### T18 — Suíte completa + lint · [RAGING-11, todos]
- Rodar `pytest -m integration tests/test_rag_caracterizacao.py -v` (CI Linux) → 5 verdes, com `git diff --stat tests/test_rag_caracterizacao.py` VAZIO.
- Rodar `pytest --cov=agenticlog --cov-report=term-missing -v` → tudo verde; cobertura ≥ 80%.
- `ruff check .`, `black --check .`, `isort --check .` limpos. Confirmar cada `ingestion/*.py` dentro de 200–400 linhas e `rag.py` reduzido.
- Varredura de segurança do seam: `rg "@patch\(\"agenticlog\.rag\.(DIR_DOCUMENTS|PROJECT_ROOT|MAX_JSON_FILES|extrair_texto_pdf|JSONLoader|HuggingFaceEmbeddings|_valida_json_sem_chaves_proibidas)\"" tests/` — cada ocorrência remanescente DEVE exercitar um chamador que ficou em `rag.py` (cria_vectordb/adicionar_pdf_incrementalmente); qualquer teste que chame uma função MOVIDA e ainda patche em `rag.*` deve migrar (design §6). O oráculo permanece sem diff.
- **Done when:** gate CI verde e diff do oráculo vazio.
- **Depends on:** T12–T17

---

## Notas de teste / gate (de `.specs/codebase/TESTING.md` e CLAUDE.md)
- Sempre mockar LLM; sempre testar retrieval vazio (coberto pelo oráculo AC-2).
- Nomes de teste com prefixo `teste_N_`.
- Commits Conventional em PT (`refactor:`), branch-first, code-review antes do merge.

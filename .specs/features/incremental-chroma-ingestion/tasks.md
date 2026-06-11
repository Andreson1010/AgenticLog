# Incremental ChromaDB Ingestion — Tasks

**Path:** `.specs/features/incremental-chroma-ingestion/tasks.md`
**TLC scope:** large
**Links to:** `.specs/features/incremental-chroma-ingestion/spec.md`, `.specs/features/incremental-chroma-ingestion/design.md`
**Status:** Approved

---

## Execution Order

Tasks must be completed in dependency order. Each task follows TDD: write the failing test first, then implement the minimum code to pass it.

```
T-01 → T-02 → T-03 → T-04 → T-05 → T-06 → T-07
                                      ↓
                                    T-08 (parallel with T-07)
                                      ↓
                                    T-09
```

---

## Task T-01 — Add `invalidar_vector_db()` to `agent.py`

**Requirement IDs:** INCRM-02  
**File:** `src/agenticlog/agent.py`  
**Dependencies:** none

### What to implement

Add a module-level public function `invalidar_vector_db()` that sets `global _vector_db = None`. No other change to `agent.py`.

### Test to write first (TDD — Red)

File: `tests/test_agent.py` (new file)

```python
class TestInvalidarVectorDb(unittest.TestCase):
    def teste_1_invalidar_seta_none(self):
        """invalidar_vector_db() deve atribuir None a _vector_db."""
        import agenticlog.agent as agent_mod
        agent_mod._vector_db = MagicMock()  # simulate initialized singleton
        agent_mod.invalidar_vector_db()
        self.assertIsNone(agent_mod._vector_db)

    def teste_2_invalidar_quando_ja_none_nao_levanta(self):
        """invalidar_vector_db() com _vector_db já None não deve lançar exceção."""
        import agenticlog.agent as agent_mod
        agent_mod._vector_db = None
        agent_mod.invalidar_vector_db()  # must not raise
        self.assertIsNone(agent_mod._vector_db)
```

### Done when
- `invalidar_vector_db` is importable from `agenticlog.agent`.
- Both test methods pass.
- No other agent tests break.

---

## Task T-02 — Add `_computar_hash_conteudo()` to `rag.py`

**Requirement IDs:** INCRM-07, INCRM-08, INCRM-13  
**File:** `src/agenticlog/rag.py`  
**Dependencies:** T-01

### What to implement

Private function using `hashlib.sha256`. Hash is computed over **raw bytes** (`conteudo: bytes`) before any parsing. Add `import hashlib` at top of file.

```python
def _computar_hash_conteudo(conteudo: bytes) -> str:
    """Computa o hash SHA-256 do conteúdo binário do arquivo.

    Entrada: conteudo — bytes do arquivo.
    Saída: string hexadecimal de 64 caracteres.
    """
    import hashlib
    return hashlib.sha256(conteudo).hexdigest()
```

### Test to write first (TDD — Red)

File: `tests/test_rag.py` — new class `TestComputarHash`

```python
def teste_1_hash_deterministico(self):
    """Mesmo input deve gerar mesmo hash de 64 caracteres."""
    h1 = _computar_hash_conteudo(b"hello")
    h2 = _computar_hash_conteudo(b"hello")
    self.assertEqual(h1, h2)
    self.assertEqual(len(h1), 64)

def teste_2_hash_diferente_para_conteudo_diferente(self):
    h1 = _computar_hash_conteudo(b"hello")
    h2 = _computar_hash_conteudo(b"world")
    self.assertNotEqual(h1, h2)
```

### Done when
- `_computar_hash_conteudo` returns correct SHA-256 hex digest.
- Both test methods pass.

---

## Task T-03 — Implement `adicionar_documento_incrementalmente()` skeleton with validation only

**Requirement IDs:** INCRM-05, INCRM-06  
**File:** `src/agenticlog/rag.py`  
**Dependencies:** T-02

### What to implement

Add the public function `adicionar_documento_incrementalmente(filename, conteudo)`. In this task, implement ONLY:
- Extension check (`.json` only).
- Size check (`> MAX_JSON_FILE_SIZE_MB`).
- `_sanitizar_nome_arquivo(filename)`.
- File count check (`len(glob) + 1 > MAX_JSON_FILES`).
- Forbidden-key check via `_valida_json_sem_chaves_proibidas` on a temp file.
- All failures raise `RAGSecurityError`.

Do NOT implement disk save, ChromaDB access, or hash check yet.

### Test to write first (TDD — Red)

File: `tests/test_rag.py` — new class `TestAdicionarDocumentoIncrementalmente`

```python
def teste_5_rejeita_extensao_invalida(self):
    with self.assertRaises(RAGSecurityError):
        adicionar_documento_incrementalmente("bad.txt", b"{}")

def teste_6_rejeita_arquivo_grande_demais(self):
    big = b"x" * (MAX_JSON_FILE_SIZE_MB * 1024 * 1024 + 1)
    with self.assertRaises(RAGSecurityError):
        adicionar_documento_incrementalmente("big.json", big)

def teste_6b_rejeita_limite_arquivos(self):
    # mock DIR_DOCUMENTS.glob to return MAX_JSON_FILES entries
    with patch("agenticlog.rag.DIR_DOCUMENTS") as mock_dir:
        mock_dir.glob.return_value = [MagicMock()] * MAX_JSON_FILES
        mock_dir.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)
        with self.assertRaises(RAGSecurityError):
            adicionar_documento_incrementalmente("new.json", b'{"key": "val"}')
```

### Done when
- Validation raises `RAGSecurityError` for all invalid inputs.
- Tests pass with ChromaDB calls still mocked/absent.

---

## Task T-04 — Implement dedup check in `adicionar_documento_incrementalmente()`

**Requirement IDs:** INCRM-07, INCRM-08, INCRM-04  
**File:** `src/agenticlog/rag.py`  
**Dependencies:** T-03

### What to implement

Extend the function body (after validations, before disk write):
1. `hash_str = _computar_hash_conteudo(conteudo)`.
2. `safe_name = _sanitizar_nome_arquivo(filename)`.
3. Compute `planned_path = DIR_DOCUMENTS / safe_name`.
4. Open `Chroma(persist_directory=str(DIR_VECTORDB), embedding_function=<HuggingFace instance>)`.
5. `existing = collection.get(where={"source": {"$eq": str(planned_path)}})` — uses explicit `$eq` operator for version robustness.
6. If `existing["ids"]`:
   - Read `existing_hash = existing["metadatas"][0].get("content_hash")`.
   - If `existing_hash == hash_str`: return `{"status": "duplicado", "mensagem": f"Arquivo {safe_name} já está presente na base vetorial."}` (no disk write).
   - Else: return `{"status": "hash_diferente", "mensagem": f"Arquivo {safe_name} já existe com conteúdo diferente. Remoção e substituição não são suportadas nesta versão."}` (no disk write).

Disk write happens only if `existing["ids"]` is empty.

### Test to write first (TDD — Red)

Mock `Chroma` constructor and `collection.get()`.

```python
def teste_3_detecta_duplicata_mesmo_hash(self):
    """Mesmo arquivo, mesmo hash: retorna status duplicado, sem adicionar chunks."""
    conteudo = b'{"pedido": "123"}'
    hash_str = hashlib.sha256(conteudo).hexdigest()
    mock_vectordb = MagicMock()
    mock_vectordb.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{"content_hash": hash_str, "source": "/some/path/doc.json"}],
    }
    with (
        patch("agenticlog.rag.Chroma") as MockChroma,
        patch("agenticlog.rag.DIR_DOCUMENTS") as mock_dir,
        # ... filesystem mocks
    ):
        MockChroma.return_value = mock_vectordb
        result = adicionar_documento_incrementalmente("doc.json", conteudo)
    self.assertEqual(result["status"], "duplicado")

def teste_4_detecta_hash_diferente(self):
    """Mesmo nome, hash diferente: retorna status hash_diferente."""
    ...
```

### Done when
- Dedup returns correct status for both duplicate and hash-mismatch cases.
- No disk write occurs for either case.

---

## Task T-05 — Implement chunk ingestion + metadata + batch add in `adicionar_documento_incrementalmente()`

**Requirement IDs:** INCRM-01, INCRM-04, INCRM-09, INCRM-11, INCRM-13  
**File:** `src/agenticlog/rag.py`  
**Dependencies:** T-04

### What to implement

After the dedup check (source not found in ChromaDB), continue:
1. Write `conteudo` to temp file, validate JSON keys via `_valida_json_sem_chaves_proibidas`, move to `DIR_DOCUMENTS / safe_name`.
2. Load file: `JSONLoader(str(saved_path), jq_schema='to_entries | map(.key + ": " + .value) | join("\\n")')`.
3. Split: `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`.
4. If `len(chunks) == 0`: log `WARNING`, return `{"status": "adicionado", "mensagem": f"Arquivo {safe_name} adicionado. 0 chunks gerados."}`.
5. Attach `content_hash` to each `chunk.metadata`.
6. `chunk_ids = [uuid.uuid4().hex for _ in chunks]`.
7. `vectordb_instance.add_documents(chunks, ids=chunk_ids)`.
8. Lazy import and call `invalidar_vector_db()`:
   ```python
   from agenticlog.agent import invalidar_vector_db  # lazy — avoids heavy agent import at CLI time
   invalidar_vector_db()
   ```
9. Return `{"status": "adicionado", "mensagem": f"Arquivo {safe_name} adicionado com sucesso. {len(chunks)} chunks inseridos."}`.

### Test to write first (TDD — Red)

```python
def teste_1_adiciona_novo_arquivo_retorna_adicionado(self):
    """Novo arquivo: chunks adicionados, status adicionado, invalidar chamado."""
    conteudo = b'{"pedido": "P001", "status": "entregue"}'
    mock_vectordb = MagicMock()
    mock_vectordb.get.return_value = {"ids": [], "metadatas": []}
    with (
        patch("agenticlog.rag.Chroma", return_value=mock_vectordb),
        patch("agenticlog.rag.invalidar_vector_db") as mock_invalidar,
        patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_dir),
        patch("agenticlog.rag.DIR_VECTORDB", new=tmp_dir / "vectordb"),
    ):
        result = adicionar_documento_incrementalmente("pedido.json", conteudo)
    self.assertEqual(result["status"], "adicionado")
    self.assertIn("chunks", result["mensagem"])
    mock_invalidar.assert_called_once()

def teste_9_zero_chunks_retorna_adicionado_com_zero(self):
    """Documento que produz zero chunks: warning logado, status adicionado, 0 chunks."""
    ...
```

### Done when
- Happy path adds chunks with `content_hash` in metadata.
- `invalidar_vector_db()` is called exactly once on success.
- Zero-chunk case returns correct status without error.
- Lazy import is inside function body, not at module top level.

---

## Task T-06 — Implement rollback on ingestion failure

**Requirement IDs:** INCRM-09, INCRM-10  
**File:** `src/agenticlog/rag.py`  
**Dependencies:** T-05

### What to implement

Wrap the `add_documents` call in try/except. Rollback severity is `logger.warning` (not `CRITICAL`) — approved decision:

```python
chunk_ids = [uuid.uuid4().hex for _ in chunks]
try:
    vectordb_instance.add_documents(chunks, ids=chunk_ids)
except Exception as ingestion_exc:
    try:
        if chunk_ids:
            vectordb_instance.delete(ids=chunk_ids)
    except Exception as rollback_exc:
        logger.warning(
            "Rollback falhou após erro de ingestão. IDs órfãos: %s. Erro de rollback: %s",
            chunk_ids,
            rollback_exc,
        )
    saved_path.unlink(missing_ok=True)
    raise ingestion_exc
```

### Test to write first (TDD — Red)

```python
def teste_7_falha_no_add_dispara_rollback(self):
    """Exceção no add_documents: delete chamado com os IDs pré-gerados, exceção re-levantada."""
    mock_vectordb = MagicMock()
    mock_vectordb.get.return_value = {"ids": [], "metadatas": []}
    mock_vectordb.add_documents.side_effect = RuntimeError("embed fail")
    with (
        patch("agenticlog.rag.Chroma", return_value=mock_vectordb),
        patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_dir),
    ):
        with self.assertRaises(RuntimeError):
            adicionar_documento_incrementalmente("doc.json", b'{"k":"v"}')
    mock_vectordb.delete.assert_called_once()
    # verify no file left in tmp_dir

def teste_8_rollback_falha_loga_warning_e_relevanva_original(self):
    """Se delete também falhar: WARNING logado com IDs, exceção original re-levantada (não a do rollback)."""
    mock_vectordb = MagicMock()
    mock_vectordb.get.return_value = {"ids": [], "metadatas": []}
    mock_vectordb.add_documents.side_effect = RuntimeError("embed fail")
    mock_vectordb.delete.side_effect = RuntimeError("rollback fail")
    with (
        patch("agenticlog.rag.Chroma", return_value=mock_vectordb),
        patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_dir),
        self.assertLogs("agenticlog.rag", level="WARNING") as log_ctx,
    ):
        with self.assertRaises(RuntimeError) as exc_ctx:
            adicionar_documento_incrementalmente("doc.json", b'{"k":"v"}')
    self.assertIn("embed fail", str(exc_ctx.exception))
    self.assertTrue(any("IDs órfãos" in m for m in log_ctx.output))
```

### Done when
- `collection.delete` is called with the pre-generated IDs on any `add_documents` failure.
- On rollback failure: WARNING log contains orphaned IDs; original exception propagates.
- Saved file is removed from disk on failure.

---

## Task T-07 — Update `app.py` to use `adicionar_documento_incrementalmente`

**Requirement IDs:** INCRM-02, INCRM-03, INCRM-12  
**File:** `app.py`  
**Dependencies:** T-05 (function must exist)

### What to implement

Modify `_ingerir_documento()`:
1. Replace import: remove `reconstruir_vectordb`, add `adicionar_documento_incrementalmente`.
2. Replace inner try block:
   ```python
   try:
       with st.spinner("Adicionando documento à base vetorial..."):
           resultado = adicionar_documento_incrementalmente(filename, conteudo)
   except RAGSecurityError as e:
       st.error(str(e))
       return
   except Exception as e:
       st.error(f"Erro ao adicionar documento. Detalhe: {e}")
       return
   ```
3. Replace success display:
   ```python
   status = resultado["status"]
   mensagem = resultado["mensagem"]
   if status == "adicionado":
       st.success(mensagem)
       st.rerun()
   elif status == "duplicado":
       st.info(mensagem)
   elif status == "hash_diferente":
       st.warning(mensagem)
   ```
4. Remove the `saved_path.unlink` from `app.py` — rollback is now inside `rag.py`.
5. Remove `reconstruir_vectordb` from import list in `app.py`.
6. Update the sidebar instruction text: remove "que nesse caso deve ser recriado com cada novo documento".

### Test to write first (TDD — Red)

File: `tests/test_app.py` — update `TestIngerirDocumento`

```python
def teste_1_upload_fluxo_sucesso(self):
    """Fluxo feliz: adicionar_documento retorna adicionado → st.success + st.rerun."""
    resultado = {"status": "adicionado", "mensagem": "Arquivo doc.json adicionado. 3 chunks inseridos."}
    with (
        patch("app.adicionar_documento_incrementalmente", return_value=resultado) as mock_add,
        patch("app.st") as mock_st,
    ):
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        _ingerir_documento(_make_uploaded_file())
    mock_add.assert_called_once_with("doc.json", b"{}")
    mock_st.success.assert_called_once_with(resultado["mensagem"])
    mock_st.rerun.assert_called_once()

def teste_5_upload_duplicado_exibe_info(self):
    resultado = {"status": "duplicado", "mensagem": "Arquivo doc.json já está presente."}
    with (
        patch("app.adicionar_documento_incrementalmente", return_value=resultado),
        patch("app.st") as mock_st,
    ):
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        _ingerir_documento(_make_uploaded_file())
    mock_st.info.assert_called_once()
    mock_st.rerun.assert_not_called()

def teste_6_upload_hash_diferente_exibe_warning(self):
    resultado = {"status": "hash_diferente", "mensagem": "Arquivo doc.json já existe com conteúdo diferente."}
    with (
        patch("app.adicionar_documento_incrementalmente", return_value=resultado),
        patch("app.st") as mock_st,
    ):
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        _ingerir_documento(_make_uploaded_file())
    mock_st.warning.assert_called_once()
    mock_st.rerun.assert_not_called()
```

### Done when
- `app.py` no longer references `reconstruir_vectordb`.
- All 3 status outcomes display correct Streamlit feedback.
- `st.rerun()` called only for `"adicionado"` status.
- Old `teste_3_upload_erro_rebuild_exibido` updated or removed (file unlink no longer in app layer).

---

## Task T-08 — Integration test: end-to-end incremental ingestion

**Requirement IDs:** INCRM-01, INCRM-04, INCRM-09  
**File:** `tests/test_rag_integration.py` (new file, mark with `@pytest.mark.integration`)  
**Dependencies:** T-06

### What to implement

Tests that use a real temporary ChromaDB on disk (no LLM calls needed). Uses `tmp_path` pytest fixture.

```python
@pytest.mark.integration
class TestIngestionIntegration:
    def teste_1_dois_arquivos_chunks_acumulam(self, tmp_path):
        """Dois uploads incrementais: chunk total = soma dos chunks de cada arquivo."""
        # patch DIR_DOCUMENTS and DIR_VECTORDB to tmp_path
        # call adicionar_documento_incrementalmente twice with distinct files
        # assert collection.count() == chunks_A + chunks_B

    def teste_2_pre_existentes_intactos_apos_novo_upload(self, tmp_path):
        """Chunks de arquivo A permanecem após upload de arquivo B."""
        # upload A, record IDs, upload B
        # assert all IDs from A still present in collection

    def teste_3_primeira_ingestao_sem_colecao_existente(self, tmp_path):
        """Sem vectordb existente: cria coleção e ingere sem erro."""
        # ensure tmp_path/vectordb does not exist
        # call adicionar_documento_incrementalmente
        # assert collection.count() > 0

    def teste_4_rollback_nao_deixa_chunks_orfaos(self, tmp_path):
        """Falha no add_documents: nenhum chunk persistido."""
        # upload A successfully
        # patch add_documents to fail on second call
        # upload B (fails)
        # assert collection.count() == chunks_A only
```

Integration tests also verify that `collection.get(where={"source": {"$eq": ...}})` filter syntax works with the installed `langchain-chroma` / `chromadb` version.

### Done when
- All integration tests pass with real ChromaDB on disk.
- `pytest -m integration tests/test_rag_integration.py -v` green.
- Filter syntax confirmed working with installed version.

---

## Task T-09 — Coverage gate and cleanup

**Requirement IDs:** All  
**File:** none (verification task)  
**Dependencies:** T-07, T-08

### What to verify

1. Run `pytest --cov=agenticlog --cov-report=term-missing -v`.
2. Coverage must be >= 80% for `agenticlog/rag.py` and `agenticlog/agent.py`.
3. Ensure `reconstruir_vectordb` and `cria_vectordb` still exist and their existing tests still pass (they are not removed — CLI entry point `python -m agenticlog.rag` still works).
4. Remove any `print()` statements introduced during development (use `logging` instead).
5. Verify all new functions have type annotations and Portuguese docstrings with `Entrada:` / `Saída:` sections.
6. Verify `tests/acceptance/test_document_ingestion_ui.py` — DOCING-01, 02, 09, 10 and PDF-01, 08 — pass with updated mocks.

### Done when
- `pytest --cov=agenticlog --cov-report=term-missing` exits 0 with >= 80% coverage.
- No regressions in existing tests.
- No `print()` calls in production code.
- All new public functions have full docstrings.
- Acceptance tests green.

---

## Summary Table

| Task | Requirement IDs | Key file | TDD test class/method |
|------|----------------|----------|-----------------------|
| T-01 | INCRM-02 | `agent.py` | `tests/test_agent.py` — `TestInvalidarVectorDb` (new file) |
| T-02 | INCRM-07/08/13 | `rag.py` | `TestComputarHash` |
| T-03 | INCRM-05/06 | `rag.py` | `TestAdicionarDocumentoIncrementalmente` (validation) |
| T-04 | INCRM-07/08/04 | `rag.py` | + dedup test methods |
| T-05 | INCRM-01/04/09/11/13 | `rag.py` | + happy path test methods |
| T-06 | INCRM-09/10 | `rag.py` | + rollback test methods |
| T-07 | INCRM-02/03/12 | `app.py` | `TestIngerirDocumento` (updated) |
| T-08 | INCRM-01/04/09 | `test_rag_integration.py` | `TestIngestionIntegration` |
| T-09 | All | — | Coverage gate |

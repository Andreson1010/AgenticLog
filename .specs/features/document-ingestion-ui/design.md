# Document Ingestion UI — Design

**Path:** `.specs/features/document-ingestion-ui/design.md`
**Links to:** `.specs/features/document-ingestion-ui/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## Architecture Overview

The feature adds a thin UI layer (sidebar expander in `app.py`) that calls two new public functions in `rag.py`. No new modules, no new dependencies, no changes to `agent.py`. The module-level retriever in `agent.py` is refreshed implicitly by `st.rerun()`, which causes Streamlit to re-execute the Python runner from scratch.

```
app.py (Streamlit UI)
  └── sidebar expander "Adicionar Documento"
        ├── st.file_uploader         (capture bytes + filename)
        └── st.button "Ingerir Documento"
              ├── salvar_documento_enviado()   ─── rag.py (new)
              │     └── _sanitizar_nome_arquivo()  ─── rag.py (new)
              │     └── _valida_json_sem_chaves_proibidas()  ─── rag.py (existing)
              ├── reconstruir_vectordb()       ─── rag.py (new, thin wrapper)
              │     └── cria_vectordb()             ─── rag.py (existing)
              └── st.rerun()
```

---

## Sequence Diagram — Full Upload Flow

```
Operator          app.py                    rag.py
   |                |                          |
   |-- select file->|                          |
   |-- click btn -->|                          |
   |                |-- salvar_documento_enviado(filename, bytes)
   |                |      |-- extension check (DOCING-03)
   |                |      |-- size check      (DOCING-04)
   |                |      |-- sanitize name   (DOCING-07)
   |                |      |-- collision check (DOCING-06)
   |                |      |-- file-count check(DOCING-08)
   |                |      |-- write tempfile
   |                |      |-- forbidden-key check (DOCING-05)
   |                |      |-- move to DIR_DOCUMENTS
   |                |      |-- return Path
   |                |                          |
   |                |   [RAGSecurityError?] ---+
   |                |   yes: st.error, return  |
   |                |                          |
   |                |-- spinner ON             |
   |                |-- reconstruir_vectordb() |
   |                |      └── cria_vectordb() |
   |                |              |-- _valida_path_documentos()
   |                |              |-- _valida_arquivos_json()
   |                |              |-- load → chunk → embed → persist
   |                |              |-- return
   |                |                          |
   |                |   [Exception?] ----------+
   |                |   yes: delete uploaded   |
   |                |        file (finally)    |
   |                |        st.error, return  |
   |                |                          |
   |                |-- spinner OFF            |
   |                |-- st.success(...)        |
   |                |-- st.rerun()             |
   |                |   [Streamlit restarts Python runner]
   |                |   agent.py re-imported   |
   |                |   retriever rebuilt from updated vectordb
   |<-- refreshed UI|
```

---

## Decision Records

### DR-1: Rollback strategy on rebuild failure

**Decision**: Use a `try/finally` block in `app.py` around `reconstruir_vectordb()`. The `finally` clause checks whether the exception occurred and, if so, deletes the file that was saved by `salvar_documento_enviado`.

**Rationale**: `cria_vectordb()` writes to `data/vectordb/` (a separate directory). If it raises before completing, `data/vectordb/` is either untouched (error before write) or partially written by ChromaDB. In both cases, the old persisted ChromaDB remains usable because ChromaDB uses its own internal write mechanism. The only artifact to clean up is the newly written file in `data/documents/`.

**Implementation pattern**:
```python
saved_path = salvar_documento_enviado(filename, conteudo)
try:
    with st.spinner("Reconstruindo base vetorial..."):
        reconstruir_vectordb()
    st.success("Documento ingerido com sucesso.")
    st.rerun()
except Exception as e:
    saved_path.unlink(missing_ok=True)
    st.error(f"Erro ao reconstruir base vetorial. Arquivo removido. Detalhe: {e}")
```

### DR-2: Sidebar layout

**Decision**: Use `st.sidebar.expander("Adicionar Documento")` placed after the existing "Suporte" button block in `app.py`.

**Rationale**: Expander keeps the sidebar uncluttered when the operator is not uploading. The approved story explicitly specifies this layout. No other sidebar positions were evaluated.

### DR-3: Filename sanitization scope

**Decision**: `_sanitizar_nome_arquivo` rejects filenames containing `..`, directory separators (`/`, `\`), and Windows-invalid characters (`<>:"/\|?*`) and null bytes. If the sanitized result is empty or differs from the original in a meaningful way (i.e., indicates attempted traversal), `RAGSecurityError` is raised rather than silently accepting a modified name.

**Rationale**: Silent normalization could mask malicious input. Explicit rejection with a clear error is consistent with the existing `RAGSecurityError` pattern and gives the operator a clear signal. The approved story specifies rejection, not sanitize-and-accept.

**Implementation**:
```python
INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*\x00')

def _sanitizar_nome_arquivo(filename: str) -> str:
    if not filename:
        raise RAGSecurityError("Nome de arquivo vazio.")
    if any(c in INVALID_FILENAME_CHARS for c in filename):
        raise RAGSecurityError(
            f"Nome de arquivo contém caracteres inválidos: {filename!r}"
        )
    basename = Path(filename).name  # strips any directory component
    if basename != filename or ".." in filename:
        raise RAGSecurityError(
            f"Nome de arquivo com path traversal detectado: {filename!r}"
        )
    return basename
```

### DR-4: Why `st.rerun()` is sufficient for retriever refresh

**Decision**: No changes to `agent.py` are required.

**Rationale**: `agent.py` builds its module-level `retriever` object at import time from `DIR_VECTORDB`. Streamlit re-runs the entire Python script (including all imports) when `st.rerun()` is called. Because Streamlit's runner reloads module state between runs (or at least reconstructs the Chroma client, which reads from disk), the retriever will see the updated `data/vectordb/` content. This has been confirmed by the researcher. The alternative — lazy initialization or explicit retriever refresh — would require changes to `agent.py` and is not needed.

---

## Validation Pipeline — `salvar_documento_enviado`

Order is critical: validations that do not require disk access run first.

| Step | Check | Raises if |
|------|-------|-----------|
| 1 | Extension | `filename` does not end with `.json` (case-insensitive) |
| 2 | Size | `len(conteudo)` > `MAX_JSON_FILE_SIZE_MB * 1024 * 1024` |
| 3 | Filename sanitization | Invalid chars or traversal sequences present |
| 4 | Collision | `DIR_DOCUMENTS / safe_filename` already exists |
| 5 | File count | `len(list(DIR_DOCUMENTS.glob("*.json"))) + 1` > `MAX_JSON_FILES` |
| 6 | Forbidden keys | Write bytes to `tempfile.NamedTemporaryFile`, parse JSON, check for `"lc"` key via `_valida_json_sem_chaves_proibidas` |
| 7 | Write | `shutil.move(tmp_path, DIR_DOCUMENTS / safe_filename)` |

Steps 1–5 are in-memory. Step 6 writes to a temp file (outside `DIR_DOCUMENTS`) and is cleaned up by the tempfile context manager regardless of outcome. Step 7 is the only write to `DIR_DOCUMENTS`.

---

## Component Interfaces

### `rag.py` additions

```python
# Private constant (module-level)
INVALID_FILENAME_CHARS: frozenset[str]

def _sanitizar_nome_arquivo(filename: str) -> str:
    """Valida e retorna o basename seguro do filename fornecido.

    Entrada: filename — nome do arquivo como string (pode conter path).
    Saída: basename sanitizado.
    Lança RAGSecurityError se o nome contiver caracteres inválidos ou path traversal.
    """

def salvar_documento_enviado(filename: str, conteudo: bytes) -> Path:
    """Valida e persiste um arquivo JSON enviado pelo operador.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
    Saída: Path do arquivo salvo em DIR_DOCUMENTS.
    Lança RAGSecurityError em qualquer falha de validação.
    """

def reconstruir_vectordb() -> None:
    """Reconstrói o banco vetorial ChromaDB a partir dos documentos em DIR_DOCUMENTS.

    Entrada: nenhuma.
    Saída: nenhuma (efeito colateral: atualiza data/vectordb/).
    Lança Exception se cria_vectordb() falhar.
    """
```

### `app.py` additions (after "Suporte" block, inside sidebar)

```python
with st.sidebar.expander("Adicionar Documento"):
    uploaded_file = st.file_uploader("Selecione um arquivo JSON", type=None)
    if st.button("Ingerir Documento"):
        if uploaded_file is None:
            st.warning("Selecione um arquivo antes de ingerir.")
        else:
            _ingerir_documento(uploaded_file)
```

The upload logic is extracted into a private helper `_ingerir_documento(uploaded_file)` to keep `app.py` readable and the button handler under 50 lines (coding style rule).

---

## Reuse from Codebase

| Existing artifact | Reused in |
|-------------------|-----------|
| `RAGSecurityError` | Raised by `salvar_documento_enviado` and `_sanitizar_nome_arquivo` |
| `_valida_json_sem_chaves_proibidas(file_path)` | Called from `salvar_documento_enviado` step 6 |
| `cria_vectordb()` | Called by `reconstruir_vectordb()` |
| `MAX_JSON_FILE_SIZE_MB`, `MAX_JSON_FILES`, `FORBIDDEN_JSON_KEYS`, `DIR_DOCUMENTS` | Referenced from `config.py` — no new constants needed |
| `try/except Exception` pattern in `app.py` | Same pattern used for upload error handling |
| `@patch("agenticlog.rag.DIR_DOCUMENTS")` + `tempfile.TemporaryDirectory` | Same pattern for new unit tests |

---

## CONCERNS.md Mitigations

| Concern | Severity | This feature's mitigation |
|---------|----------|--------------------------|
| Incomplete test coverage — app.py entirely untested | MEDIUM | `tests/test_app.py` is a required deliverable of this feature (tasks T-008, T-009) |

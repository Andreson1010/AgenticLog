# Document Ingestion UI — Tasks

**Path:** `.specs/features/document-ingestion-ui/tasks.md`
**Links to:** `.specs/features/document-ingestion-ui/spec.md`, `.specs/features/document-ingestion-ui/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## Task List

### T-001 — Add `INVALID_FILENAME_CHARS` constant to `rag.py`

**File:** `src/agenticlog/rag.py`
**Requirement IDs:** DOCING-07
**Description:** Add module-level `INVALID_FILENAME_CHARS: frozenset[str] = frozenset('<>:"/\\|?*\x00')` after the existing imports. This constant is consumed by `_sanitizar_nome_arquivo` (T-002) and must exist before that function.
**Done when:** `INVALID_FILENAME_CHARS` is present at module level, accessible via `from agenticlog.rag import INVALID_FILENAME_CHARS` in tests.
**Dependencies:** None.

---

### T-002 — Implement `_sanitizar_nome_arquivo` in `rag.py`

**File:** `src/agenticlog/rag.py`
**Requirement IDs:** DOCING-07
**Description:** Add private function after `_valida_arquivos_json`. Logic: reject empty names, reject names containing any char in `INVALID_FILENAME_CHARS`, extract basename with `Path(filename).name`, reject if basename differs from input OR if `".."` appears anywhere in the original. Return the safe basename string.
**Docstring format:** Portuguese triple-quoted, "Entrada: / Saída:" format (per CLAUDE.md convention).
**Done when:** Function is importable; raises `RAGSecurityError` for `"../evil.json"`, `"evil<>.json"`, and empty string; returns `"valid.json"` unchanged.
**Dependencies:** T-001.

---

### T-003 — Implement `salvar_documento_enviado` in `rag.py`

**File:** `src/agenticlog/rag.py`
**Requirement IDs:** DOCING-01, DOCING-03, DOCING-04, DOCING-05, DOCING-06, DOCING-07, DOCING-08
**Description:** Add public function after `_sanitizar_nome_arquivo`. Implement the 7-step validation pipeline exactly as described in `design.md` (extension → size → sanitize → collision → file-count → forbidden-key via tempfile → move to DIR_DOCUMENTS). Use `tempfile.NamedTemporaryFile(delete=False, suffix=".json")` for step 6 and clean up the temp file in a `finally` block. Return the saved `Path`.

Key implementation notes:
- Extension check: `Path(filename).suffix.lower() != ".json"`.
- Size check: `len(conteudo) > MAX_JSON_FILE_SIZE_MB * 1024 * 1024`.
- Collision check: `(DIR_DOCUMENTS / safe_name).exists()`.
- File-count check: `len(list(DIR_DOCUMENTS.glob("*.json"))) + 1 > MAX_JSON_FILES`.
- Forbidden-key check: write `conteudo` to tempfile, call `_valida_json_sem_chaves_proibidas(Path(tmp.name))`.
- Move: `shutil.move(str(tmp_path), DIR_DOCUMENTS / safe_name)`.

**Done when:** Function passes all 7 test cases defined in T-006.
**Dependencies:** T-001, T-002.

---

### T-004 — Implement `reconstruir_vectordb` in `rag.py`

**File:** `src/agenticlog/rag.py`
**Requirement IDs:** DOCING-01, DOCING-09
**Description:** Add public function after `salvar_documento_enviado`. Body is a single call to `cria_vectordb()`. Exists for intention clarity and testability.

```python
def reconstruir_vectordb() -> None:
    """Reconstrói o banco vetorial ChromaDB a partir dos documentos em DIR_DOCUMENTS.

    Entrada: nenhuma.
    Saída: nenhuma (efeito colateral: atualiza data/vectordb/).
    Lança Exception se cria_vectordb() falhar.
    """
    cria_vectordb()
```

**Done when:** `reconstruir_vectordb()` is importable and calling it invokes `cria_vectordb()` exactly once (verifiable via mock in T-007).
**Dependencies:** None (calls existing `cria_vectordb`).

---

### T-005 — Add upload UI section to `app.py`

**File:** `app.py`
**Requirement IDs:** DOCING-01, DOCING-02, DOCING-09, DOCING-10
**Description:** Add the following after the `st.sidebar.button("Suporte")` block:

1. Import `salvar_documento_enviado`, `reconstruir_vectordb`, `RAGSecurityError` from `agenticlog.rag` at the top of the file.
2. Add a private helper function `_ingerir_documento(uploaded_file)` (defined before the Streamlit page layout code or after imports, as a module-level function). It must:
   - Read `uploaded_file.getvalue()` (bytes) and `uploaded_file.name`.
   - Call `salvar_documento_enviado(filename, conteudo)`, storing the returned `Path` as `saved_path`.
   - Wrap `reconstruir_vectordb()` in `with st.spinner("Reconstruindo base vetorial...")`.
   - On rebuild success: call `st.success("Documento ingerido com sucesso.")` then `st.rerun()`.
   - On `RAGSecurityError` from save step: call `st.error(str(e))`.
   - On any `Exception` from rebuild step: call `saved_path.unlink(missing_ok=True)` then `st.error(f"Erro ao reconstruir base vetorial. Arquivo removido. Detalhe: {e}")`.
3. Add to sidebar (after "Suporte" button block):
   ```python
   with st.sidebar.expander("Adicionar Documento"):
       uploaded_file = st.file_uploader("Selecione um arquivo JSON", type=None)
       if st.button("Ingerir Documento"):
           if uploaded_file is None:
               st.warning("Selecione um arquivo antes de ingerir.")
           else:
               _ingerir_documento(uploaded_file)
   ```

**Done when:**
- Sidebar shows "Adicionar Documento" expander.
- Uploading a valid JSON and clicking the button triggers rebuild and rerun (manual smoke test).
- T-008 UI tests pass.

**Dependencies:** T-003, T-004.

---

### T-006 — Unit tests for `salvar_documento_enviado` (7 cases)

**File:** `tests/test_rag.py`
**Requirement IDs:** DOCING-03, DOCING-04, DOCING-05, DOCING-06, DOCING-07, DOCING-08, DOCING-01
**Description:** Add a new `unittest.TestCase` class `TestSalvarDocumentoEnviado`. Each test uses `tempfile.TemporaryDirectory` and `@patch("agenticlog.rag.DIR_DOCUMENTS", new=...)` to isolate disk writes. Follow `teste_N_` naming convention.

| Test method | Scenario |
|-------------|----------|
| `teste_1_salvar_documento_enviado_sucesso` | Valid bytes + valid name → file exists in temp dir |
| `teste_2_salvar_rejeita_extensao_invalida` | `.csv` extension → `RAGSecurityError` |
| `teste_3_salvar_rejeita_tamanho_excedido` | `b"x" * (10 * 1024 * 1024 + 1)` → `RAGSecurityError` |
| `teste_4_salvar_rejeita_chave_proibida` | JSON `{"lc": "bad"}` → `RAGSecurityError` |
| `teste_5_salvar_rejeita_colisao_de_nome` | Pre-create target file → `RAGSecurityError` |
| `teste_6_salvar_rejeita_path_traversal` | `"../evil.json"` → `RAGSecurityError` |
| `teste_7_salvar_rejeita_limite_de_arquivos` | Create 1000 stub `.json` files in temp dir → `RAGSecurityError` |

**Done when:** All 7 tests pass with `pytest tests/test_rag.py::TestSalvarDocumentoEnviado -v`.
**Dependencies:** T-003.

---

### T-007 — Unit tests for `reconstruir_vectordb` and `_sanitizar_nome_arquivo`

**File:** `tests/test_rag.py`
**Requirement IDs:** DOCING-09, DOCING-07
**Description:** Add two additional test classes (or extend existing class):

`TestReconstruirVectordb`:
- `teste_1_reconstruir_vectordb_chama_cria_vectordb` — mock `agenticlog.rag.cria_vectordb`, call `reconstruir_vectordb()`, assert mock called once.
- `teste_2_reconstruir_vectordb_propaga_excecao` — mock raises `Exception("fail")`, assert exception propagates.

`TestSanitizarNomeArquivo`:
- `teste_1_sanitizar_nome_valido` — `"doc.json"` returns `"doc.json"`.
- `teste_2_sanitizar_rejeita_path_traversal` — `"../evil.json"` raises `RAGSecurityError`.
- `teste_3_sanitizar_rejeita_chars_invalidos` — `"file<>.json"` raises `RAGSecurityError`.
- `teste_4_sanitizar_rejeita_nome_vazio` — `""` raises `RAGSecurityError`.

**Done when:** All tests pass with `pytest tests/test_rag.py::TestReconstruirVectordb tests/test_rag.py::TestSanitizarNomeArquivo -v`.
**Dependencies:** T-002, T-004.

---

### T-008 — UI tests for upload flow in `test_app.py`

**File:** `tests/test_app.py` (create if not present)
**Requirement IDs:** DOCING-01, DOCING-02, DOCING-09
**Description:** Use `unittest.mock.patch` and `unittest.mock.MagicMock` to test `_ingerir_documento`. Do not use Streamlit's test runner unless it is already in the project's test stack — test the helper function directly by mocking `salvar_documento_enviado`, `reconstruir_vectordb`, `st.success`, `st.error`, `st.spinner`, `st.rerun`.

| Test method | Scenario |
|-------------|----------|
| `teste_1_upload_fluxo_sucesso` | Mocked save + rebuild succeed → `st.success` called, `st.rerun` called |
| `teste_2_upload_erro_validacao_exibido` | `salvar_documento_enviado` raises `RAGSecurityError` → `st.error` called, `st.rerun` NOT called |
| `teste_3_upload_erro_rebuild_exibido` | `reconstruir_vectordb` raises `Exception` → `saved_path.unlink` called, `st.error` called |
| `teste_4_upload_sem_arquivo_selecionado` | `uploaded_file` is None → `st.warning` called, no save/rebuild |

Follow `teste_N_` naming. Add `if __name__ == "__main__": unittest.main()` at bottom.

**Done when:** All 4 tests pass with `pytest tests/test_app.py -v`.
**Dependencies:** T-005.

---

### T-009 — Coverage gate verification

**File:** No file changes; verification task.
**Requirement IDs:** All DOCING-* (quality gate)
**Description:** Run `pytest --cov=agenticlog --cov-report=term-missing --cov-fail-under=80 -v`. If coverage is below 80%, identify uncovered lines and add targeted tests or adjust existing ones. Do not lower the threshold.
**Done when:** `pytest --cov=agenticlog --cov-fail-under=80` exits with code 0.
**Dependencies:** T-006, T-007, T-008.

---

## Dependency Graph

```
T-001 → T-002 → T-003 → T-006
                T-003 → T-005 → T-008 → T-009
         T-004 → T-005
         T-002 → T-007
         T-004 → T-007
                         T-009
```

Linear build order respecting all dependencies:
`T-001 → T-002 → T-004 → T-003 → T-007 → T-006 → T-005 → T-008 → T-009`

---

## Notes for Builders

- All new Python functions must have type annotations on all parameters and return values (Python coding style rule).
- Functions must stay under 50 lines (common coding style rule). Extract helpers if `salvar_documento_enviado` grows past that.
- No hardcoded numeric values — use constants from `config.py` (`MAX_JSON_FILE_SIZE_MB`, `MAX_JSON_FILES`).
- Error messages must be in Portuguese (per codebase convention).
- Import order in `rag.py`: stdlib (`json`, `logging`, `pathlib`, `shutil`, `tempfile`) → third-party → local.
- `shutil` must be added to `rag.py` imports (currently not present) to support `shutil.move`.
- `tempfile` must be added to `rag.py` imports.
- Branch: `feature/document-ingestion-ui`. Commit messages in Conventional Commits Portuguese (e.g., `feat(rag): adicionar salvar_documento_enviado com pipeline de validação`).

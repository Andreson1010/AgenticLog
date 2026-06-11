# Incremental ChromaDB Ingestion — Design

**Path:** `.specs/features/incremental-chroma-ingestion/design.md`
**TLC scope:** large
**Links to:** `.specs/features/incremental-chroma-ingestion/spec.md`
**Status:** Approved

---

## Architecture Overview

The current ingestion path is:

```
app._ingerir_documento()
  └─ salvar_documento_enviado()   [rag.py]  — save file to disk
  └─ reconstruir_vectordb()       [rag.py]  — Chroma.from_documents() — FULL REBUILD
```

The new incremental path is:

```
app._ingerir_documento()
  └─ adicionar_documento_incrementalmente()   [rag.py]
       ├─ _computar_hash_conteudo()           [rag.py]  — SHA-256 of raw bytes
       ├─ validations (size, ext, keys, path, count)    — RAGSecurityError on failure
       ├─ dedup check via collection.get()   [chromadb] — before any disk write
       ├─ disk save (conditional)            [rag.py]   — skipped if file already on disk
       ├─ _chunkar via JSONLoader + splitter [rag.py]
       ├─ collection.add_documents()         [chromadb] — incremental add
       ├─ rollback on failure                [rag.py]   — delete IDs + unlink file
       └─ invalidar_vector_db()             [agent.py] — lazy import inside function body
```

The key architectural choice: use `Chroma(persist_directory=..., embedding_function=...)` (open existing or create) instead of `Chroma.from_documents(...)` (always creates/overwrites). The `persist_directory` and `EMBEDDING_MODEL` are unchanged — this is the only collection the agent reads from.

---

## Component Design

### 1. `_computar_hash_conteudo(conteudo: bytes) -> str` (private, `rag.py`)

Pure function. Returns `hashlib.sha256(conteudo).hexdigest()`. No I/O.

```python
def _computar_hash_conteudo(conteudo: bytes) -> str:
    """Computa o hash SHA-256 do conteúdo binário do arquivo.

    Entrada: conteudo — bytes do arquivo.
    Saída: string hexadecimal de 64 caracteres (SHA-256).
    """
    import hashlib
    return hashlib.sha256(conteudo).hexdigest()
```

Hash is computed over **raw bytes** (`conteudo`) before any JSON parsing or jq transformation. This is the approved decision.

### 2. Hash storage strategy

The hash is stored in **ChromaDB chunk metadata** as `content_hash`. It is set on every `Document` before `collection.add_documents()`. This means:

- No separate database or file is needed.
- Dedup check uses `collection.get(where={"source": {"$eq": str(saved_path)}})` which returns all chunks for that source. If any chunk exists, the `content_hash` is read from the first chunk's metadata.
- If `source` already exists AND hash matches → duplicate.
- If `source` already exists AND hash differs → hash mismatch warning.
- If `source` does not exist → proceed with ingestion.

**Why not store hash in a sidecar file?** ChromaDB metadata keeps hash co-located with the data it describes, survives vectordb recreation by `cria_vectordb()` as long as all source files are re-ingested, and requires no additional file I/O path.

**Caveat:** If `cria_vectordb()` is called on an environment that never ran incremental ingestion, old chunks will have no `content_hash`. The dedup check handles this: `chunk.metadata.get("content_hash")` returns `None` for old chunks, which will not match the new file's hash → triggers hash mismatch path. This is safe — the operator is warned rather than silently re-ingesting.

### 3. `salvar_documento_enviado` interaction

The existing function raises `RAGSecurityError("Arquivo com esse nome já existe.")` before the dedup check can happen. The new function bypasses this specific guard: `adicionar_documento_incrementalmente` does NOT call `salvar_documento_enviado`. Instead it:

1. Runs security validations directly (extension, size, forbidden keys via `_valida_json_sem_chaves_proibidas`, path sanitization via `_sanitizar_nome_arquivo`, file count limit).
2. Opens the ChromaDB collection and performs the dedup check **before** any disk write.
3. If a duplicate or hash mismatch is detected, returns immediately — no file is written.
4. If the file is genuinely new, writes bytes to a temp file, validates JSON keys, moves to `DIR_DOCUMENTS / safe_name`.

This preserves all security guarantees without coupling to the "file already exists" short-circuit in `salvar_documento_enviado`.

**Alternative considered:** Modify `salvar_documento_enviado` to accept a `permitir_existente: bool` flag. Rejected — changes a tested public function's signature, violates open/closed principle.

### 4. Rollback mechanism

```
chunk_ids = [uuid.uuid4().hex for _ in chunks]
try:
    vectordb_instance.add_documents(chunks, ids=chunk_ids)
except Exception as ingestion_exc:
    try:
        if chunk_ids:
            vectordb_instance.delete(ids=chunk_ids)
    except Exception as rollback_exc:
        logger.warning(
            "Rollback falhou. IDs órfãos: %s. Erro: %s", chunk_ids, rollback_exc
        )
    saved_path.unlink(missing_ok=True)
    raise ingestion_exc
```

**Rollback severity:** `logger.warning` (not `CRITICAL`) — approved decision. Rollback failure is logged and swallowed; the original ingestion exception propagates to the caller.

**Batch vs per-chunk add:** Add all chunks in a single `add_documents()` call to minimize ChromaDB round trips. IDs are pre-computed before the call so rollback is always possible regardless of what ChromaDB returns.

**ChromaDB ignoring passed IDs:** If `add_documents(ids=[...])` silently ignores the IDs, `delete(ids=chunk_ids)` may be a no-op. This is logged as `logger.warning` with the orphaned IDs list and swallowed — approved decision.

### 5. `invalidar_vector_db()` contract (`agent.py`) — lazy import

```python
def invalidar_vector_db() -> None:
    """Invalida o singleton _vector_db para que a próxima chamada a _get_vector_db() reconecte ao ChromaDB.

    Entrada: nenhuma.
    Saída: nenhuma.
    Efeito colateral: atribui None a _vector_db global.
    """
    global _vector_db
    _vector_db = None
```

**Import direction — lazy import required:**

`rag.py` must call `invalidar_vector_db()` from `agent.py`. A module-level import (`from agenticlog.agent import invalidar_vector_db` at the top of `rag.py`) would cause `agent.py` to be fully imported whenever `python -m agenticlog.rag` is run. `agent.py` loads LangGraph, HuggingFace embeddings, and LMStudio client at import time — this is a significant overhead for a CLI rebuild operation that does not need the agent at all.

**Decision:** Import lazily inside the function body:

```python
def adicionar_documento_incrementalmente(filename: str, conteudo: bytes) -> dict[str, str]:
    # ... all logic ...
    # At the end, after successful add:
    from agenticlog.agent import invalidar_vector_db  # lazy import — avoids heavy agent import at CLI time
    invalidar_vector_db()
    return {"status": "adicionado", ...}
```

This means the import only happens when `adicionar_documento_incrementalmente` is actually called (i.e., from `app.py` at runtime), not when `rag.py` is imported for the CLI rebuild.

**Circular import check:** Currently `agent.py` does NOT import from `rag.py`. The lazy import does not introduce a circular dependency — verified from source.

After invalidation, the next call to `_get_vector_db()` in `agent.py` opens a new `Chroma(persist_directory=...)` connection, which reads the updated collection from disk. The `_embedding_model` singleton is NOT invalidated — it is stateless w.r.t. the collection contents.

---

## Data Flow Diagram

```
Operator uploads file.json
        │
        ▼
_ingerir_documento(uploaded_file)          [app.py]
        │
        ▼
adicionar_documento_incrementalmente(      [rag.py]
    filename="file.json",
    conteudo=b"..."
)
        │
        ├─ _computar_hash_conteudo(conteudo) → hash_str  [raw bytes]
        │
        ├─ validations (size, extension, forbidden keys, path, count)
        │    └─ RAGSecurityError on failure → returned to app.py → st.error()
        │
        ├─ open Chroma(persist_directory=DIR_VECTORDB, embedding_function=...)
        │    └─ cold-start: ChromaDB creates directory + empty collection automatically
        │
        ├─ collection.get(where={"source": {"$eq": str(planned_path)}})
        │    ├─ found + hash matches  → return {"status": "duplicado", ...}   [no disk write]
        │    └─ found + hash differs  → return {"status": "hash_diferente", ...} [no disk write]
        │
        ├─ write conteudo to temp file → validate JSON keys → move to DIR_DOCUMENTS/safe_name
        │
        ├─ JSONLoader(saved_path) → RecursiveCharacterTextSplitter → chunks
        │    └─ zero chunks → log WARNING, return {"status": "adicionado", mensagem: "0 chunks"}
        │
        ├─ attach content_hash to each chunk.metadata
        │
        ├─ chunk_ids = [uuid.uuid4().hex for _ in chunks]
        │
        ├─ vectordb_instance.add_documents(chunks, ids=chunk_ids)
        │    └─ Exception → rollback:
        │         ├─ vectordb_instance.delete(ids=chunk_ids)
        │         │    └─ Exception → logger.warning(orphaned IDs) + swallow
        │         ├─ saved_path.unlink(missing_ok=True)
        │         └─ raise original exception
        │
        ├─ from agenticlog.agent import invalidar_vector_db  [lazy import]
        ├─ invalidar_vector_db()
        │
        └─ return {"status": "adicionado", "mensagem": "Arquivo file.json adicionado. N chunks inseridos."}
                │
                ▼
        app.py: st.success(mensagem) + st.rerun()
```

---

## Interface Contracts

### `adicionar_documento_incrementalmente` return values

| `status` value | UI action in `app.py` | `st.rerun()` called | Meaning |
|----------------|----------------------|---------------------|---------|
| `"adicionado"` | `st.success(mensagem)` | Yes | New chunks added, singleton invalidated |
| `"duplicado"` | `st.info(mensagem)` | No | Same file, same hash already in collection |
| `"hash_diferente"` | `st.warning(mensagem)` | No | Same filename, different content; no action taken |

### ChromaDB collection name

The default LangChain-managed collection name is `"langchain"`. Both `Chroma.from_documents()` (used by `cria_vectordb`) and `Chroma(persist_directory=...)` (used by `_get_vector_db` and the new function) use this default. No `collection_name` parameter is set in either location — this must remain consistent.

---

## Reuse from Codebase

| Existing component | Reused how |
|--------------------|-----------|
| `_sanitizar_nome_arquivo()` | Called directly for path safety |
| `_valida_json_sem_chaves_proibidas()` | Called on temp file before move |
| `MAX_JSON_FILES`, `MAX_JSON_FILE_SIZE_MB`, `FORBIDDEN_JSON_KEYS` from `config.py` | Unchanged constants |
| `DIR_DOCUMENTS`, `DIR_VECTORDB`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` from `config.py` | Unchanged constants |
| `RAGSecurityError` | Raised on all boundary violations |
| `HuggingFaceEmbeddings` | New instance created inside `adicionar_documento_incrementalmente` for the add operation (avoids cross-module state coupling; acceptable overhead since add is infrequent) |

---

## CONCERNS addressed

No `.specs/project/CONCERNS.md` file was present. Risks from spec.md are addressed:

- **same-name file guard in `salvar_documento_enviado`** — bypassed: new function does not call `salvar_documento_enviado`; performs dedup check before disk write.
- **`collection.get` filter syntax** — use `where={"source": {"$eq": str(saved_path)}}` (ChromaDB explicit operator syntax) for robustness across versions.
- **Partial batch add** — mitigated by pre-computing IDs and rollback on any exception.
- **ChromaDB silently ignoring IDs in rollback** — log `logger.warning` with orphaned IDs, swallow; approved decision.
- **Heavy import at CLI time** — mitigated by lazy import of `invalidar_vector_db` inside function body; `agent.py` is not loaded when `python -m agenticlog.rag` runs.
- **Thread safety** — documented as known limitation; acceptable for single-user local deployment.

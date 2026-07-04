# RAG Ingestion Fase 3a — Design

**Path:** `.specs/features/rag-ingestion-fase3a/design.md`
**Spec:** `.specs/features/rag-ingestion-fase3a/spec.md`
**TLC scope:** complex
**Status:** Awaiting human approval

---

## 1. Architecture Overview

Extração de estágios de baixo risco de `agenticlog.rag` para um pacote `agenticlog.ingestion`, seguindo o padrão de **shim de re-export identity-preserving** provado na Fase 2 (PR #60): `src/agenticlog/shared/` e `src/agenticlog/observability/` + shims em `rag.py:87` e `config.py:163`.

```
config.py  ──────────────┐  (constantes: fonte única)
shared/errors.py ────────┤  (RAGSecurityError)
                         ▼
        agenticlog.ingestion/           (NOVO — só importa config + shared)
        ├── security.py     (validação, sanitização, saves de upload)
        ├── extraction.py   (extrair_texto_pdf + carregar_json)
        ├── cleaning.py     (filtrar_documentos_vazios)
        ├── chunking.py     (re-export SemanticChunker)
        ├── embeddings.py   (criar_embedding_model — factory)
        ├── metadata.py     (hash + enriquecimento de metadados)
        └── __init__.py     (re-export público + __all__)
                         ▲
        agenticlog.rag ──┘  (orquestradores + Chroma + atomicidade + CLI + SHIMS)
                            mantém: _rag_embedding_model (global), _get_rag_embedding_model
                                    (getter delega ao factory), import fitz (F401),
                                    HuggingFaceEmbeddings/DirectoryLoader/JSONLoader (cria_vectordb)
```

Arestas de dependência: `config → shared → ingestion → rag → agent`. Todas unidirecionais; nenhum módulo de `ingestion/` importa `rag`/`agent`. `security → extraction` é aresta intra-pacote (não circular).

---

## 2. Componentes e interfaces

### 2.1 `ingestion/security.py`
Funções (movidas verbatim de `rag.py`): `_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, `_valida_arquivos_json`, `_sanitizar_nome_arquivo`, `_sanitizar_nome_colecao`, `sanitizar_nome_colecao` (público), `salvar_documento_enviado`, `salvar_pdf_enviado`.
Imports: `from agenticlog.shared.errors import RAGSecurityError`; de `config`: `FORBIDDEN_JSON_KEYS, MAX_JSON_FILE_SIZE_MB, MAX_JSON_FILES, MAX_DOCUMENT_FILE_SIZE_MB, DEFAULT_COLLECTION_NAME, COLLECTION_NAME_MIN_LEN, COLLECTION_NAME_MAX_LEN, COLLECTION_NAME_PATTERN, INVALID_FILENAME_CHARS, WINDOWS_RESERVED_NAMES, DIR_DOCUMENTS, PROJECT_ROOT`; `from agenticlog.ingestion.extraction import extrair_texto_pdf` (usado por `salvar_pdf_enviado`).

### 2.2 `ingestion/extraction.py`
`extrair_texto_pdf(path: Path) -> dict[str, str]` — movido verbatim (rag.py:703-733).
`carregar_json(caminho: Path) -> list[Document]` — NOVO wrapper:
```python
def carregar_json(caminho: Path) -> list[Document]:
    """Carrega um arquivo JSON como Documents via JSONLoader + jq_schema compartilhado."""
    loader = JSONLoader(str(caminho), jq_schema=JQ_SCHEMA_CAMPOS_JSON)
    return loader.load()
```
Imports: `fitz`, `RAGSecurityError`, `from langchain_community.document_loaders import JSONLoader`, `from langchain_core.documents import Document`, `JQ_SCHEMA_CAMPOS_JSON`.

### 2.3 `ingestion/cleaning.py`
```python
def filtrar_documentos_vazios(docs: list) -> list:
    """Descarta Documents cujo page_content é vazio após .strip() (ADR-011). Retorna lista nova."""
    return [d for d in docs if d.page_content.strip()]
```
Extraído dos 4 sites inline (`rag.py` ~479, ~636, ~835, ~853). Puro, imutável.

### 2.4 `ingestion/chunking.py`
```python
from langchain_experimental.text_splitter import SemanticChunker  # re-export
__all__ = ["SemanticChunker"]
```
**Sem factory** (locked Q3) — introduzir uma factory quebraria os ~30 `@patch("agenticlog.rag.SemanticChunker")`.

### 2.5 `ingestion/embeddings.py`
```python
def criar_embedding_model() -> HuggingFaceEmbeddings:
    """Constrói o modelo de embedding (device auto + normalize). VERBATIM de rag.py:75-83."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # crítico p/ silent-degradation
    )
```
Imports: `HuggingFaceEmbeddings`, `torch`, `EMBEDDING_MODEL`. **Não** contém o global nem o getter.

### 2.6 `ingestion/metadata.py`
`_computar_hash_conteudo`, `_hash_arquivo`, `_enriquecer_metadados_chunks` — movidos verbatim (rag.py:278-305). Imports: `hashlib`, `Path`, `METADATA_FILE_HASH, METADATA_CHUNK_INDEX, METADATA_PAGE, METADATA_DOC_TYPE`.

### 2.7 `ingestion/__init__.py`
Re-export ergonômico + `__all__` (precedente `shared/__init__.py`, `observability/__init__.py`). Ex.: `from agenticlog.ingestion.security import salvar_documento_enviado, ...`.

### 2.8 `rag.py` após a fase
- Mantém: orquestradores, Chroma, `_backup_arquivo`/`_reverter_disco`/`_resetar_colecao`/`_outras_colecoes_existem`, CLI, `vectordb`, `_rag_embedding_model` (global), `_get_rag_embedding_model` (getter — corpo delega a `criar_embedding_model()`), `import fitz  # noqa: F401`, `HuggingFaceEmbeddings`/`DirectoryLoader`/`JSONLoader` (para `cria_vectordb`).
- `adicionar_documento_incrementalmente`: `JSONLoader(...).load()` → `carregar_json(saved_path)`; filtro inline → `filtrar_documentos_vazios(...)`.
- `adicionar_pdf_incrementalmente` e `cria_vectordb`: filtros inline → `filtrar_documentos_vazios(...)`.
- Getter delegado:
```python
def _get_rag_embedding_model() -> HuggingFaceEmbeddings:
    global _rag_embedding_model
    if _rag_embedding_model is None:
        _rag_embedding_model = criar_embedding_model()
    return _rag_embedding_model
```

---

## 3. Design Note DN-1 — `carregar_json` e o `DirectoryLoader` de `cria_vectordb`

Q2 pede rotear "a `inline JSONLoader usage` nos orquestradores (`cria_vectordb`, `adicionar_documento_incrementalmente`) por `carregar_json`". A ÚNICA instanciação inline direta de `JSONLoader(...)` está em `adicionar_documento_incrementalmente` (rag.py:476). `cria_vectordb` NÃO tem `JSONLoader` inline — usa `DirectoryLoader(..., loader_cls=JSONLoader)` (directory-batch), mecanismo distinto do wrapper single-file `carregar_json`.

**Decisão (honra todos os hard constraints):** rotear apenas o caminho single-file (incremental) por `carregar_json`; manter o `DirectoryLoader` de `cria_vectordb` inalterado nesta fase. Justificativa:
1. `carregar_json` é single-file por contrato; `DirectoryLoader` é directory-batch — não são intercambiáveis 1:1.
2. Converter `cria_vectordb` quebraria toda a `TestCriaVectordb` (patcha `agenticlog.rag.DirectoryLoader`), fora do escopo de migração dirigida das locked decisions.
3. `cria_vectordb` NÃO é coberto pelo oráculo → é o pior lugar para um risco de mudança de loader nesta fase.
4. `cria_vectordb` é reprojetado na Fase 3b, onde o loader directory-batch é reavaliado.
`JSONLoader` permanece importado em `rag.py` (é `loader_cls` do `DirectoryLoader`). Isto é uma clarificação de escopo, não uma questão aberta.

---

## 4. Contrato de identidade dos shims (RAGING-08)

Para cada símbolo movido, `rag.py` adiciona uma linha de re-export num bloco agrupado, com o marcador `# Re-export shim (ADR-018 Fase 3a) — remover na Fase 6` e `# noqa` conforme necessário (precedente: `rag.py:87`, `config.py:163`):

```python
# ── Re-export shims (ADR-018 Fase 3a) — remover na Fase 6 ─────────────────────
from agenticlog.ingestion.security import (  # noqa: E402
    _valida_path_documentos, _valida_json_sem_chaves_proibidas, _valida_arquivos_json,
    _sanitizar_nome_arquivo, _sanitizar_nome_colecao, sanitizar_nome_colecao,
    salvar_documento_enviado, salvar_pdf_enviado,
)  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.extraction import extrair_texto_pdf, carregar_json  # noqa: E402  # Re-export shim ...
from agenticlog.ingestion.cleaning import filtrar_documentos_vazios  # noqa: E402  # Re-export shim ...
from agenticlog.ingestion.chunking import SemanticChunker  # noqa: E402,F401  # Re-export shim ...
from agenticlog.ingestion.embeddings import criar_embedding_model  # noqa: E402  # Re-export shim ...
from agenticlog.ingestion.metadata import (  # noqa: E402
    _computar_hash_conteudo, _hash_arquivo, _enriquecer_metadados_chunks,
)  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
```

**Invariante:** `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` (objeto idêntico). Testado por round-trip (RAGING-13). O shim preserva: (a) caminhos de import antigos; (b) o monkeypatch do SÍMBOLO quando o chamador resolve `X` pela namespace de `rag.py` (ex.: `cria_vectordb`/`adicionar_pdf_incrementalmente` chamando `extrair_texto_pdf`, `SemanticChunker`, `_valida_arquivos_json`, `_hash_arquivo` — todos os chamadores que FICAM em `rag.py`).

**O que o shim NÃO preserva:** o lookup de nomes de módulo DENTRO de uma função cujo corpo foi movido. Uma função em `security.py` lê `security.DIR_DOCUMENTS`, não `rag.DIR_DOCUMENTS`. Ver §6.

---

## 5. Mecanismo double-shim de `embeddings` (Q1) — proof-sketch de que o oráculo fica verde SEM edição

**Setup do oráculo (imutável):** a fixture faz `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", emb_stub)` e usa `adicionar_documento_incrementalmente` como ponto de entrada.

**Cadeia de chamada:** `adicionar_documento_incrementalmente` (em `rag.py`) chama `_get_rag_embedding_model()` (em `rag.py`). O getter lê o global `_rag_embedding_model` DA namespace de `rag.py`.

**Passo 1 — cache/getter ficam em `rag.py`:** como `_rag_embedding_model` e `_get_rag_embedding_model` permanecem em `rag.py`, `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", emb_stub)` rebinda exatamente o nome que o getter lê. Logo `_get_rag_embedding_model()` retorna `emb_stub` sem construir nada. ∎ (oráculo AC-1/AC-4/AC-5 usam o stub, nunca o modelo real de ~1 GB.)

**Passo 2 — por que NÃO mover o global para `embeddings.py`:** se o global vivesse em `embeddings.py`, o getter leria `embeddings._rag_embedding_model`; o `setattr` em `agenticlog.rag._rag_embedding_model` seria invisível ao getter → carrega o modelo real → oráculo quebra. Alternativamente, o getter em `embeddings.py` lendo a namespace de `rag.py` inverteria a dependência (`ingestion → rag`) → ciclo. Portanto, só a CONSTRUÇÃO migra; o estado (cache) fica. ∎

**Passo 3 — delegação não afeta o oráculo:** quando `_rag_embedding_model is None` fora do oráculo, o getter chama `criar_embedding_model()`. No oráculo o global nunca é `None` (patchado), então o factory jamais é invocado. Comportamento observável idêntico. ∎

**Consequência (RAGING-12):** apenas 2 locais NÃO-oráculo, que patcham `agenticlog.rag.HuggingFaceEmbeddings` e resetam `rag._rag_embedding_model=None` para exercitar a CONSTRUÇÃO via getter, precisam migrar o alvo de patch para `agenticlog.ingestion.embeddings.HuggingFaceEmbeddings` (o reset de `rag._rag_embedding_model` continua válido, pois o cache fica em `rag.py`):
- `tests/test_rag.py::TestGetRagEmbeddingModel` (2 métodos, linhas ~681/692).
- `tests/acceptance/test_portuguese_embedding_model.py::teste_1_get_rag_embedding_model_usa_embedding_model_do_config` (linha ~79).

Os testes de `cria_vectordb` (que patcham `agenticlog.rag.HuggingFaceEmbeddings`) e os testes de inspeção de fonte (que asseram `encode_kwargs={"normalize_embeddings": True}` em `rag_source`) ficam verdes SEM edição, porque `cria_vectordb` MANTÉM a construção inline de `HuggingFaceEmbeddings` (rag.py:861-866). Este é um motivo adicional para NÃO rotear `cria_vectordb` pelo factory nesta fase.

---

## 6. Seam de monkeypatch de funções movidas — inventário e migração dirigida (RAGING-12)

**Princípio:** um teste que (i) chama DIRETAMENTE uma função movida e (ii) patcha uma dependência dela em `agenticlog.rag.<dep>` quebra, porque a função (agora em `ingestion/<mod>.py`) resolve `<dep>` na sua própria namespace. A correção é migrar o alvo do patch para `agenticlog.ingestion.<mod>.<dep>`.

### 6.1 Embeddings (§5) → `agenticlog.ingestion.embeddings.HuggingFaceEmbeddings`
`TestGetRagEmbeddingModel` (test_rag.py); `test_portuguese_embedding_model.py::teste_1`.

### 6.2 JSONLoader do caminho incremental → `agenticlog.ingestion.extraction.JSONLoader`
`adicionar_documento_incrementalmente` passa a chamar `carregar_json` (que instancia `JSONLoader` na namespace de `extraction`). Todos os `@patch("agenticlog.rag.JSONLoader")` do caminho incremental migram:
- `tests/test_rag.py`: linhas ~1397, 1436, 1487, 1529, 1558, 1584, 1612, 1644, 1686, 1726.
- `tests/test_rag_integration.py`: linha ~139.
- `tests/acceptance/test_multi_collection_chromadb.py`: linha ~512.
- `tests/acceptance/test_semantic_chunking.py`: linhas ~110, 138.
- `tests/acceptance/test_unificar_metadados_chunks.py`: linhas ~275, 375, 420.
(Nenhum destes exercita `cria_vectordb`, que mocka `DirectoryLoader` por inteiro.)

### 6.3 Funções de `security.py` → `agenticlog.ingestion.security.<dep>`
- `TestValidaPathDocumentos` (test_rag.py ~73-125): `agenticlog.rag.DIR_DOCUMENTS`, `agenticlog.rag.PROJECT_ROOT` → `agenticlog.ingestion.security.{DIR_DOCUMENTS,PROJECT_ROOT}`.
- `TestValidaArquivosJson` (~188-218): `agenticlog.rag.DIR_DOCUMENTS`, `agenticlog.rag.MAX_JSON_FILES`, `agenticlog.rag._valida_json_sem_chaves_proibidas` → alvos `agenticlog.ingestion.security.*`.
- `TestSalvarDocumentoEnviado` (~1084-1145): `agenticlog.rag.DIR_DOCUMENTS` → `agenticlog.ingestion.security.DIR_DOCUMENTS`.
- `TestSalvarPdfEnviado` (~1259-1350): `agenticlog.rag.DIR_DOCUMENTS`, `agenticlog.rag.MAX_JSON_FILES`, `agenticlog.rag.extrair_texto_pdf` → `agenticlog.ingestion.security.{DIR_DOCUMENTS,MAX_JSON_FILES,extrair_texto_pdf}` (security importa `extrair_texto_pdf` de extraction; patchar o nome que security realmente chama).

### 6.4 Verde SEM edição (verificado — não migrar)
- **Oráculo** `tests/test_rag_caracterizacao.py`: orquestrador em `rag.py` lê `rag.DIR_DOCUMENTS`/`rag.DIR_VECTORDB`/`rag._rag_embedding_model` (patchados); `carregar_json`/`filtrar_documentos_vazios`/`_enriquecer_metadados_chunks` são puros ou parametrizados por argumento (não leem os nomes patchados).
- `TestExtrairTextoPdf`: patcha `agenticlog.rag.fitz.open`; `fitz` fica importado em `rag.py` e é singleton em `sys.modules` → `extraction.extrair_texto_pdf` vê o patch.
- `TestValidaJsonSemChavesProibidas`, `TestSanitizarNomeArquivo`, `TestSanitizarNomeColecao`: puros / parametrizados por argumento.
- Toda `TestCriaVectordb`/`TestLogging`: `cria_vectordb` fica em `rag.py`; patches de `rag.DirectoryLoader`/`rag.HuggingFaceEmbeddings`/`rag._valida_*`/`rag.extrair_texto_pdf` resolvem via namespace de `rag.py` (nomes re-exportados/mantidos).
- ~30 `@patch("agenticlog.rag.SemanticChunker")` e ~14 testes de `adicionar_pdf_incrementalmente` patchando `rag.extrair_texto_pdf`: chamadores ficam em `rag.py`.

---

## 7. Argumento de acicidade de imports (RAGING-10)

`ingestion/*` importa exclusivamente de `agenticlog.config` e `agenticlog.shared` (folhas do grafo) e, intra-pacote, `security → extraction`. Nenhum import de `agenticlog.rag`/`agenticlog.agent`. Logo `import agenticlog.ingestion` num interpretador frio não pode formar ciclo. Verificação por `subprocess` (padrão do teste AC-06 de `test_rag_shared_observability.py`), estendida para importar `agenticlog.ingestion` e asserir exit 0 sem "circular" em stderr.

`security.py` importa `extraction.py`; `extraction.py` NÃO importa `security.py` (DAG intra-pacote), então nem o import de `security` fecha ciclo.

---

## 8. Reuso do codebase

- Padrão de shim: `shared/__init__.py`, `observability/__init__.py`, `rag.py:87` (RAGSecurityError), `config.py:163`.
- Testes de identidade/acicidade: `tests/acceptance/test_rag_shared_observability.py` (AC-04 identidade, AC-06 fresh-interpreter).
- Convenções: type hints em todas as assinaturas, docstrings PT (Entrada/Saída/Lança), imutabilidade (cleaning retorna lista nova).

---

## 9. Mitigações dos itens de risco (spec §Risks / CONCERNS)

| Risco | Mitigação de design |
|-------|---------------------|
| Oráculo/singleton | Cache+getter em `rag.py`; só construção no factory (§5). |
| Seam de patch de função movida | Inventário fechado + migração dirigida (§6); oráculo imune por construção. |
| Referência `fitz` | `import fitz  # noqa: F401` mantido em `rag.py`; singleton `sys.modules`. |
| Import circular | `ingestion/*` só depende de `config`+`shared`; teste fresh-interpreter (§7). |
| `normalize_embeddings` | Migração verbatim; unit test asserta kwargs exatos. |
| `carregar_json` vs `DirectoryLoader` | DN-1: rebuild não convertido nesta fase. |

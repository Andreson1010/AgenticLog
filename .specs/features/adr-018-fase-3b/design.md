# ADR-018 Fase 3b — Design

**Path:** `.specs/features/adr-018-fase-3b/design.md`
**Spec:** `.specs/features/adr-018-fase-3b/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## 1. Architecture Overview

A Fase 3b extrai as duas últimas camadas de `agenticlog.rag` para o pacote `agenticlog.ingestion`, completando a ADR-018:

- **`store.py`** — camada de persistência/atomicidade: escrita/rollback no Chroma e mutação de disco/coleção. É o DONO das escritas no Chroma.
- **`orchestrator.py`** — os 5 orquestradores de ingestão + o helper compartilhado dos dois `adicionar_*`.

`rag.py` vira a **fachada de compatibilidade**: mantém globais/seams (`DIR_DOCUMENTS`, `DIR_VECTORDB`, `_rag_embedding_model`, getter), CLI, shims de identidade (Fase 3a + os 4 símbolos de `store`) e **WRAPPERS finos** que ligam os seams no momento da chamada.

```
config.py (constantes: fonte única) ─────────────┐
shared/errors.py (RAGSecurityError) ─────────────┤
ingestion/{extraction,cleaning,chunking,          │  (Fase 3a — folhas)
          metadata,embeddings,security} ──────────┤
                                                  ▼
        ingestion/store.py         (NOVO — importa só config + shared)
                                                  ▲
        ingestion/orchestrator.py  (NOVO — importa config, shared, store,
                                    extraction, cleaning, chunking, metadata,
                                    embeddings, security; NUNCA rag/agent)
                                                  ▲
        agenticlog.rag  ──────────────────────────┘  (fachada: globais + seams
                                    + CLI + shims de store (is) + WRAPPERS de
                                    orquestrador (delegação) + getter/cache)
```

Arestas de import (todas unidirecionais, DAG): `config → shared → {estágios 3a} → store → orchestrator → rag → agent`. Nenhum módulo de `ingestion/` importa `rag`/`agent`. `orchestrator → store` e `orchestrator → estágios 3a` são arestas intra-pacote sem ciclo.

---

## 2. Code Reuse Analysis

### Componentes existentes a alavancar

| Component | Location | How to Use |
|-----------|----------|-----------|
| Padrão shim `is`-identity | `rag.py:76-99`, `shared/__init__.py`, `observability/__init__.py`, `config.py` | Aplicar o MESMO bloco de re-export para os 4 símbolos de `store`. |
| Estágios 3a | `agenticlog.ingestion.{extraction,cleaning,chunking,metadata,embeddings,security}` | `orchestrator` importa e chama; nada é reimplementado. |
| Getter/cache de embedding | `rag.py:56-73` (`_rag_embedding_model`, `_get_rag_embedding_model`) | FICAM em `rag.py`; o wrapper incremental injeta `_get_rag_embedding_model()` como seam. |
| Teste de acicidade fresh-interpreter | `tests/acceptance/test_rag_shared_observability.py::test_ac06`; `tests/ingestion/test_shims_identidade.py::TestIngestionAcyclic` | Estender o padrão `subprocess` para `store`+`orchestrator`. |
| Fixture do oráculo | `tests/test_rag_caracterizacao.py` (patcha `rag.DIR_DOCUMENTS`/`DIR_VECTORDB`/`_rag_embedding_model`) | NÃO modificar; o design garante que o wrapper faz esses 3 seams fluírem. |

### Integração

| System | Integration Method |
|--------|--------------------|
| `app.py` | Importa `salvar_documento_enviado`/`salvar_pdf_enviado` (shims 3a) e `reconstruir_vectordb` (wrapper 3b) de `agenticlog.rag`. |
| `api.py` | Importa `_get_rag_embedding_model` de `agenticlog.rag` (permanece). |
| `agent.py` | Sem mudança; lê seus próprios `DIR_VECTORDB`/`_vector_dbs`. |

---

## 3. Componentes e interfaces

### 3.1 `ingestion/store.py`

Camada de persistência. Importa: `logging`, `shutil`, `tempfile`, `Path`; de `config`: `DIR_VECTORDB` (default de módulo, resolvido no corpo — ver §4.2). **Não** importa `rag`/`orchestrator`.

- `_backup_arquivo(path: Path) -> Path` — VERBATIM de `rag.py:102-107`. Cria `.bak` temporário via `shutil.copy2`.
- `_reverter_disco(saved_path: Path, backup_path: Path | None) -> None` — VERBATIM de `rag.py:110-121`. Upsert (`backup_path` não-nulo): `shutil.move(backup)`; novo (`None`): `unlink(missing_ok=True)`.
- `_outras_colecoes_existem(collection_name: str, vectordb_dir: Path | None = None) -> bool` — corpo VERBATIM de `rag.py:124-147`, com `DIR_VECTORDB` substituído por `vectordb_dir` resolvido no corpo (`vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir`). Leitura read-only do SQLite; schema ausente → `False`.
- `_resetar_colecao(collection_name: str, vectordb_dir: Path | None = None) -> None` — corpo VERBATIM de `rag.py:150-186`, com `DIR_VECTORDB` → `vectordb_dir` resolvido no corpo; delega a `_outras_colecoes_existem(collection_name, vectordb_dir=vectordb_dir)` (propaga o seam); `shutil.rmtree` no caso comum, `delete_collection` no caso multi-coleção.
- `add_documents_com_rollback(vectordb_instance, chunks: list, chunk_ids: list[str]) -> None` — **NOVA primitiva (D1)**, extraída do bloco inline `try/except add_documents/delete` de ambos os `adicionar_*` (`rag.py:295-306` e `454-465`):
  ```python
  def add_documents_com_rollback(vectordb_instance, chunks, chunk_ids) -> None:
      """Escreve chunks no Chroma com rollback best-effort na falha.

      Entrada: vectordb_instance (Chroma), chunks (list[Document]), chunk_ids (list[str]).
      Saída: nenhuma. Efeito: chunks persistidos OU nenhum efeito líquido (rollback).
      Lança: re-levanta a exceção ORIGINAL de add_documents após tentar o delete.
      """
      try:
          vectordb_instance.add_documents(chunks, ids=chunk_ids)
      except Exception:
          try:
              vectordb_instance.delete(ids=chunk_ids)
          except Exception as rollback_exc:
              logger.warning(
                  "Rollback falhou após erro de ingestão. IDs órfãos: %s. Erro de rollback: %s",
                  chunk_ids, rollback_exc,
              )
          raise
  ```
  A mensagem de log é a de `adicionar_documento_incrementalmente` (`rag.py:301-305`), unificada para ambos os tipos (a variante PDF de `rag.py:459-464` usava `"Erro: %s"`; unificar para `"Erro de rollback: %s"` — ver DN-3). O `delete(old_ids)` de upsert (rag.py:308-315 / 467-474) permanece no orquestrador (não é parte da escrita atômica dos NOVOS chunks).

**Tamanho estimado:** ~150–180 ln (docstrings PT incluídas). Abaixo do alvo de 200; aceitável — é uma camada coesa e o alvo 200–400 é teto de manutenção, não piso rígido. Documentado como decisão consciente.

### 3.2 `ingestion/orchestrator.py`

Orquestradores + helper compartilhado. Importa: `logging`, `shutil`, `tempfile`, `uuid`, `torch`, `Path`; `Chroma`, `Document`, `HuggingFaceEmbeddings`; de `config` todas as constantes usadas; de `shared.errors` `RAGSecurityError`; de `ingestion`: `store` (`_backup_arquivo`, `_reverter_disco`, `_resetar_colecao`, `add_documents_com_rollback`), `extraction.carregar_json`, `extraction.extrair_texto_pdf`, `cleaning.filtrar_documentos_vazios`, `chunking.SemanticChunker`, `embeddings.criar_embedding_model`, `metadata._enriquecer_metadados_chunks`/`_hash_arquivo`/`_computar_hash_conteudo`, `security._sanitizar_nome_colecao`/`_sanitizar_nome_arquivo`/`_valida_json_sem_chaves_proibidas`/`_valida_path_documentos`/`_valida_arquivos_json`. **Não** importa `rag`/`agent`.

Assinaturas (todos os seams com default `None`, resolvidos no corpo — §4.2):

- `adicionar_documento_incrementalmente(filename, conteudo, collection_name=DEFAULT_COLLECTION_NAME, *, docs_dir=None, vectordb_dir=None, embedding_model=None) -> dict[str, str]`
- `adicionar_pdf_incrementalmente(filename, conteudo, collection_name=DEFAULT_COLLECTION_NAME, *, docs_dir=None, vectordb_dir=None, embedding_model=None) -> dict[str, str]`
- `cria_vectordb(collection_name=DEFAULT_COLLECTION_NAME, *, docs_dir=None, vectordb_dir=None, embedding_model=None) -> "Chroma | None"` (retorna a instância Chroma construída para o wrapper atribuir `rag.vectordb`; `None` quando não há documentos — ver §4.4)
- `reconstruir_vectordb(collection_name=DEFAULT_COLLECTION_NAME, *, docs_dir=None, vectordb_dir=None, embedding_model=None) -> None` — `_sanitizar_nome_colecao` + delega a `cria_vectordb`.
- `ingerir_incrementalmente(collection_name=DEFAULT_COLLECTION_NAME, *, docs_dir=None, vectordb_dir=None, embedding_model=None) -> dict[str, int]` — itera `sorted(docs_dir.glob("*.json")) + sorted(docs_dir.glob("*.pdf"))`, delega aos dois `adicionar_*` (propagando os seams), fail-fast em `RAGSecurityError`, conta `erro` em falha comum. VERBATIM de `rag.py:621-659`.
- `_ingerir_arquivo_incrementalmente(...)` — helper privado de dedup (§5).

### 3.3 `rag.py` (fachada) após a fase

Mantém: docstring, imports (SEM `DirectoryLoader`/`JSONLoader` — só usados por `cria_vectordb`, agora no orchestrator), `logger`, `vectordb = None`, `_rag_embedding_model = None`, `_get_rag_embedding_model` (getter/cache — inalterado), `import fitz  # noqa: F401`, bloco de shims 3a (inalterado), CLI (`_configurar_logging_cli`, `_executar_main`, bloco `__main__`).

Adiciona:
- **Shim block de store (identidade `is`):**
  ```python
  # ── Re-export shims de store (ADR-018 Fase 3b) — remover na Fase 6 ────────────
  from agenticlog.ingestion.store import (  # noqa: E402,F401
      _backup_arquivo, _reverter_disco, _outras_colecoes_existem, _resetar_colecao,
  )  # Re-export shim (ADR-018 Fase 3b) — remover na Fase 6
  ```
  (`add_documents_com_rollback` é primitiva interna do orchestrator; re-export opcional, sem contrato de import antigo.)
- **5 WRAPPERS de orquestrador** (ligam seams no momento da chamada — §4.1).

**Estimativa de linhas:** `rag.py` atual ~710 ln − (store ~85 + orquestradores ~470) + (shim de store ~6 + 5 wrappers ~40) ≈ **~200 ln** (bem dentro de ≤420).

---

## 4. Mecânica de seam (a parte delicada) — ING3B-04

### 4.1 WRAPPERS de `rag.py` ligam seams no MOMENTO DA CHAMADA

Os orquestradores movidos **NÃO PODEM** ler `DIR_DOCUMENTS`/`DIR_VECTORDB` como nomes de nível de módulo de `orchestrator.py` — isso capturaria o valor PRÉ-monkeypatch no import de `orchestrator` e o oráculo (que faz `monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", ...)`) quebraria. A solução: `rag.py` mantém WRAPPERS finos que resolvem os globais de `rag.py` a cada chamada e os passam como argumentos:

```python
def adicionar_documento_incrementalmente(filename, conteudo, collection_name=DEFAULT_COLLECTION_NAME):
    return _orch.adicionar_documento_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )

def adicionar_pdf_incrementalmente(filename, conteudo, collection_name=DEFAULT_COLLECTION_NAME):
    return _orch.adicionar_pdf_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )

def ingerir_incrementalmente(collection_name=DEFAULT_COLLECTION_NAME):
    return _orch.ingerir_incrementalmente(
        collection_name, docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )

def cria_vectordb(collection_name=DEFAULT_COLLECTION_NAME):
    global vectordb
    vectordb = _orch.cria_vectordb(
        collection_name, docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=None,  # rebuild constrói modelo fresco (D2)
    )

def reconstruir_vectordb(collection_name=DEFAULT_COLLECTION_NAME):
    return _orch.reconstruir_vectordb(
        collection_name, docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=None,
    )
```

`import agenticlog.ingestion.orchestrator as _orch` (E402, após os globais). Aqui `DIR_DOCUMENTS`/`DIR_VECTORDB` são **os globais de `rag.py`**, resolvidos por nome A CADA CHAMADA do wrapper → o `monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", ...)` do oráculo é lido pelo wrapper e injetado na função movida. `_get_rag_embedding_model()` retorna o stub patchado (`monkeypatch.setattr("agenticlog.rag._rag_embedding_model", emb)`), preservando o double-shim da Fase 3a.

**Prova do oráculo (por que fica verde sem edição):** a fixture patcha SÓ 3 nomes em `agenticlog.rag` e entra por `adicionar_documento_incrementalmente`. O wrapper resolve os 3 do namespace de `rag` no momento da chamada e os passa; a função movida usa os parâmetros (nunca lê `orchestrator.DIR_*`). Logo os 3 seams patchados fluem. ∎

### 4.2 Resolução de seam `None → default` nas funções movidas

`orchestrator.py` e `store.py`:
```python
from agenticlog.config import DIR_DOCUMENTS, DIR_VECTORDB  # defaults de módulo
# dentro de cada função:
docs_dir = DIR_DOCUMENTS if docs_dir is None else docs_dir
vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
```
Ler o global do módulo NO CORPO (não como default de argumento avaliado no `def`) permite que os testes NÃO-oráculo que patcham `agenticlog.ingestion.store.DIR_VECTORDB` (ex.: `TestOutrasColecoesExistem`) sejam honrados. `config.DIR_VECTORDB` não é monkeypatchado em produção, então o fallback é estável.

### 4.3 Seam de `embedding_model` no caminho incremental (D2)

- **Rebuild** (`cria_vectordb`): seam `embedding_model=None` → constrói `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})` VERBATIM (rag.py:563-568). Preserva o caminho atual (motivo de o rebuild NÃO usar o factory: `TestCriaVectordb` e testes de inspeção de fonte esperam a construção inline; mas note que `TestCriaVectordb` é reescrito nesta fase para o namespace de `orchestrator` — a construção inline continua no corpo de `cria_vectordb`, agora em `orchestrator.py`).
- **Incremental** (`adicionar_*`): o wrapper injeta `_get_rag_embedding_model()` (o singleton). Quando `embedding_model is None` numa chamada DIRETA ao orquestrador (fora do wrapper), o helper faz fallback `embedding_model = criar_embedding_model()`. Chamadas de produção/oráculo sempre passam pelo wrapper; chamadas diretas em teste devem fornecer `embedding_model` (ou patchar `Chroma`/`SemanticChunker`, que consomem o objeto sem invocá-lo).

### 4.4 Preservação do global `vectordb` de `rag.py`

Nenhum outro módulo lê `rag.vectordb` (só `cria_vectordb` o usa internamente para o `count()` do guardrail). Para preservar o efeito observável, `orchestrator.cria_vectordb` **retorna** a instância Chroma (ou `None` quando não há documentos) e o wrapper de `rag` faz `global vectordb; vectordb = _orch.cria_vectordb(...)`. O guardrail `RuntimeError` (`count()==0`) roda DENTRO de `orchestrator.cria_vectordb`, sobre a instância local, antes de retornar — comportamento idêntico.

---

## 5. Atomicidade de upsert e dedup do helper — ING3B-06 / ING3B-07

### 5.1 Partes compartilhadas vs. por-tipo dos dois `adicionar_*`

| Etapa | JSON | PDF | No helper? |
|-------|------|-----|-----------|
| `_sanitizar_nome_colecao` | igual | igual | fora do helper (public fn — preserva precedência) |
| Validação de tipo | suffix `.json` + tamanho `MAX_JSON_FILE_SIZE_MB` | suffix `.pdf` + magic `%PDF` + tamanho `MAX_DOCUMENT_FILE_SIZE_MB` | fora do helper (public fn) |
| `_sanitizar_nome_arquivo`, contagem, hash, `planned_path` | igual | igual | no helper |
| `Chroma(...)` + `get` + dedup/`old_ids` | igual | igual | no helper |
| tmp write | suffix `.json` | suffix `.pdf` | no helper (param `tmp_suffix`) |
| passo pré-move | `_valida_json_sem_chaves_proibidas(tmp)` → contexto `None` | `paginas = extrair_texto_pdf(tmp)` → contexto `paginas` | callback `preparar_no_tmp(tmp) -> contexto` |
| `_backup_arquivo` se upsert + `shutil.move` | igual | igual | no helper |
| construção de docs | `carregar_json(saved_path)` + `filtrar` | build `Document`s de `paginas` (source=saved_path) + `filtrar` | callback `construir_docs(saved_path, contexto) -> list[Document]` |
| chunk + 0-chunks branch | igual | igual | no helper |
| `_enriquecer_metadados_chunks` | `(chunks, hash, DOC_TYPE_JSON, PAGE_JSON_SENTINEL)` | `(chunks, hash, DOC_TYPE_PDF)` | no helper (param `doc_type`, `page_args`) |
| `add_documents_com_rollback` + `delete(old_ids)` | igual | igual | no helper (usa `store`) |
| `invalidar_vector_db` + mensagem de warning | msg `"...do agente: %s"` | msg `"...singleton: %s"` | no helper (param `invalidation_msg` — DN-3) |
| mensagem de retorno | templates idênticos | templates idênticos | no helper |

Assinatura do helper:
```python
def _ingerir_arquivo_incrementalmente(
    filename, conteudo, collection_name, *,
    docs_dir, vectordb_dir, embedding_model,
    tmp_suffix, preparar_no_tmp, construir_docs, doc_type, page_args, invalidation_msg,
) -> dict[str, str]:
```
As public `adicionar_*` fazem `_sanitizar_nome_colecao` + validação de tipo, resolvem seams (`None → default`), e delegam ao helper com o bundle por-tipo.

### 5.2 DN-2 — posição do `construir_docs` no bloco guardado (preservação exata)

Hoje, no JSON, `carregar_json(saved_path)` roda DENTRO do bloco guardado (rag.py:270-318); no PDF, a construção dos `Document`s roda FORA do guard (rag.py:421-430), antes do bloco guardado (rag.py:434-480). Para deduplicar, o helper invoca `construir_docs(saved_path, contexto)` como **primeira instrução DENTRO do bloco guardado** para AMBOS os tipos.

- JSON: idêntico ao atual (construção já era intra-guard).
- PDF: a construção migra para DENTRO do guard. Isso é observável SOMENTE se `construir_docs` levantar — mas para PDF a construção é `Document(page_content=f"{chave}: {texto}", ...)` a partir de `paginas` já extraído com sucesso no passo pré-move; `int(chave.split("_")[1])` sobre chaves geradas por `extrair_texto_pdf` (formato `pagina_N`) não levanta para entradas alcançáveis. Portanto é um no-op observável para todo input real; no caso patológico de falha de construção, reverter o disco é o comportamento MAIS correto. **Decisão:** aceitar a reconciliação intra-guard, minimizando churn no caminho JSON (coberto pelo oráculo). Guardado pela suíte PDF de aceitação (comportamento inalterado).

### 5.3 DN-3 — paridade da mensagem de warning de invalidação

Os dois `adicionar_*` logam mensagens de warning DIFERENTES quando `invalidar_vector_db` falha no import (`"...do agente: %s"` vs `"...singleton: %s"`) e a variante de log de rollback também difere (`"Erro de rollback: %s"` vs `"Erro: %s"`). Para preservar saída byte-idêntica, `invalidation_msg` é parâmetro por-tipo do helper. A mensagem de rollback é unificada em `add_documents_com_rollback` para a variante JSON (`"Erro de rollback: %s"`) — mudança de string de LOG interna, não asserida por nenhum teste/oráculo; documentada aqui e no ADR-019 como micro-reconciliação consciente.

### 5.4 `cria_vectordb` sem `DirectoryLoader` (ING3B-05)

Substituir (rag.py:529-537):
```python
json_docs = []
for json_path in sorted(docs_dir.glob("*.json")):
    json_docs.extend(carregar_json(json_path))
json_docs = filtrar_documentos_vazios(json_docs)
```
`sorted()` garante ordem determinística (o `DirectoryLoader` não garantia ordem estável entre plataformas). A semântica por-arquivo (`JSONLoader(str(path), jq_schema=JQ_SCHEMA_CAMPOS_JSON).load()`) é idêntica à do `loader_cls=JSONLoader` do `DirectoryLoader`. O restante de `cria_vectordb` (PDF loop, embeddings frescas, chunking por-source, `_resetar_colecao`, `from_documents`, guardrail) é VERBATIM, com `DIR_VECTORDB`→`vectordb_dir` e `_resetar_colecao`→`store._resetar_colecao(collection_name, vectordb_dir=vectordb_dir)`.

---

## 6. Contrato de identidade vs. delegação — ING3B-03

| Símbolo | Em `rag.py` | Contrato | Teste |
|---------|-------------|----------|-------|
| `_backup_arquivo`, `_reverter_disco`, `_outras_colecoes_existem`, `_resetar_colecao` | shim `from ...store import ...` | **`is`-idêntico**: `rag.X is store.X` | novo `test_rag_ingestion_fase3b.py` |
| `add_documents_com_rollback` | (interno; re-export opcional) | primitiva de `store` | unit `test_store.py` |
| `cria_vectordb`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `reconstruir_vectordb` | **WRAPPER** que delega | **NÃO `is`-idêntico**: `rag.f is not orchestrator.f`; `rag.f` chama `orchestrator.f` ligando seams | novo `test_rag_ingestion_fase3b.py` (delegação por efeito: seams fluem) |

`tests/ingestion/test_shims_identidade.py` lista SÓ os símbolos da Fase 3a → permanece verde com **zero diff** (os 4 símbolos de store e os 5 orquestradores NÃO estão na sua lista). As novas asserções vivem no arquivo novo.

---

## 7. Error Handling Strategy

| Cenário | Tratamento | Impacto |
|---------|-----------|---------|
| Falha de segurança (path/chave/tamanho) | `RAGSecurityError` levantada nas validações (security 3a) | Fail-fast; `ingerir_incrementalmente` aborta o lote (re-levanta) |
| Falha de embedding no chunking | `except` do bloco guardado → `_reverter_disco` → re-levanta | Disco restaurado; coleção intacta |
| Falha de `add_documents` | `add_documents_com_rollback` → `delete(chunk_ids)` (best-effort) → re-levanta original; orquestrador → `_reverter_disco` | Sem chunks parciais; disco restaurado; IDs órfãos logados se rollback falhar |
| 0 chunks | `_reverter_disco` + retorno `"adicionado"` com mensagem "0 chunks gerados" | Arquivo não indexado; disco limpo |
| Rebuild persiste 0 chunks | `RuntimeError` (guardrail fail-loud) | Aborta rebuild em vez de RAG silenciosamente offline |
| `invalidar_vector_db` `ImportError` | `logger.warning(invalidation_msg, e)` | Não-fatal |

---

## 8. Tech Decisions

| Decisão | Escolha | Racional |
|---------|---------|----------|
| Store é dono das escritas no Chroma (D1) | `add_documents_com_rollback` em `store.py` | Concentra atomicidade de escrita na camada de persistência; orquestrador fica declarativo |
| Seam de embedding no rebuild (D2) | `embedding_model=None` → constrói fresco em `cria_vectordb` | Preserva caminho de rebuild atual e a flag `normalize_embeddings`; incremental recebe o singleton via wrapper |
| Local do helper de dedup (D3) | `_ingerir_arquivo_incrementalmente` privado em `orchestrator.py` | Coeso com os dois `adicionar_*`; parametrizado por callbacks de extração por-tipo |
| Seam ligado no momento da chamada | wrappers em `rag.py` resolvem globais por nome a cada chamada | Único jeito de honrar `monkeypatch.setattr("agenticlog.rag.DIR_*")` sem editar o oráculo |
| Store `is`-shim vs. orquestrador wrapper | store re-exportado (identidade), orquestradores embrulhados (delegação) | Store não precisa de seam dinâmico; orquestradores precisam → wrapper |
| `cria_vectordb` sem `DirectoryLoader` | `sorted(glob)` + `carregar_json` | AC5; ordem determinística; mesma semântica jq_schema por-arquivo |

---

## 9. Argumento de acicidade — ING3B-09

`store.py` importa só `config` + `shared` (folhas). `orchestrator.py` importa `config`, `shared`, `store` e os estágios 3a — todos já folhas ou já acíclicos; nenhum importa `orchestrator`/`rag`. Logo `import agenticlog.ingestion.store` e `import agenticlog.ingestion.orchestrator` em interpretador frio não formam ciclo. Verificação por `subprocess` (padrão `test_rag_shared_observability.py::test_ac06` / `test_shims_identidade.py::TestIngestionAcyclic`), estendida no NOVO `tests/acceptance/test_rag_ingestion_fase3b.py`, asserindo exit 0 sem "circular"/"partially initialized" em stderr.

---

## 10. Mitigações dos itens de risco (spec §Risks / CONCERNS.md)

| Risco | Mitigação de design |
|-------|---------------------|
| Seam em tempo de import vs. chamada | Wrappers resolvem globais de `rag` por nome a cada chamada (§4.1); funções movidas usam parâmetros (§4.2). Oráculo imune por construção. |
| Identidade store vs. delegação orquestrador | §6: `is` só para store; delegação para wrappers; `test_shims_identidade.py` zero diff. |
| Atomicidade de upsert | `add_documents_com_rollback` + helper preservam a sequência exata; DN-2/DN-3 documentam as micro-reconciliações. |
| `cria_vectordb` sem DirectoryLoader | §5.4: `sorted(glob)` determinístico; caminho não-oráculo; `TestCriaVectordb` reescrito. |
| Import circular | §9: `store`/`orchestrator` só dependem de camadas inferiores; teste fresh-interpreter. |
| `normalize_embeddings` silent-degradation (CONCERNS) | Construção verbatim no rebuild; unit asserta kwargs exatos. |
| Missing error handling / logging (CONCERNS) | Preservado verbatim; `logger.warning`/`error` e guardrail `RuntimeError` intactos; sem novos `print()`. |
| hnswlib SAC Windows | CI Linux é gate autoritativo; local pula Chroma. |

# Spec — semantic-chunking

**Feature ID:** semantic-chunking
**Status:** Proposto
**Data:** 2026-06-17
**Supercede:** ADR-007 (sentence separators no RecursiveCharacterTextSplitter — obsoleto)
**Relacionado:** ADR-008 (chunking por chave/página — preservado), REC-01 (metadados unificados — preservado)

---

## Problema

`RecursiveCharacterTextSplitter(CHUNK_SIZE=500, CHUNK_OVERLAP=50)` corta texto por limite de
caracteres. Mesmo com separadores de sentença (ADR-007) e granularidade por chave/página
(ADR-008), unidades que excedem 500 chars (páginas de PDF densas, campos JSON longos) são
partidas onde o limite cai — não onde o significado muda. Isso degrada a qualidade de
recuperação em consultas semânticas.

---

## Solução

Substituir `RecursiveCharacterTextSplitter` por `SemanticChunker`
(`langchain_experimental.text_splitter`) nos três pontos de uso em `rag.py`. O `SemanticChunker`
embeda cada sentença com o mesmo modelo já carregado (`paraphrase-multilingual-mpnet-base-v2`) e
detecta breakpoints onde a similaridade cai abaixo de um limiar, partindo o texto nessas
fronteiras semânticas em vez de em limites de caractere.

---

## Requisitos Funcionais

| ID | Requisito |
|----|-----------|
| SC-01 | `SemanticChunker` de `langchain_experimental.text_splitter` substitui `RecursiveCharacterTextSplitter` em `cria_vectordb`, `adicionar_documento_incrementalmente` e `adicionar_pdf_incrementalmente` (três instâncias em `rag.py`). |
| SC-02 | O `SemanticChunker` é inicializado com a instância de `HuggingFaceEmbeddings` já criada em cada função — sem novo download ou modelo adicional. |
| SC-03 | Configuração de breakpoint exposta em `config.py` como duas constantes: `SEMANTIC_BREAKPOINT_TYPE: str = "percentile"` e `SEMANTIC_BREAKPOINT_THRESHOLD: float = 95.0`. |
| SC-04 | `CHUNK_SIZE` e `CHUNK_OVERLAP` removidos de `config.py` e de todos os sites de import/uso em `rag.py` (três instâncias cada). |
| SC-05 | `langchain_experimental>=0.3.0` adicionado a `requirements.txt` (versão mínima que inclui `SemanticChunker` estável). |
| SC-06 | `_enriquecer_metadados_chunks` permanece inalterado e é chamado após `split_documents` — metadados `file_hash`, `chunk_index`, `page`, `doc_type` (REC-01) preservados em todos os chunks. |
| SC-07 | `CLAUDE.md` atualizado: seção "Build VectorDB" lista `SEMANTIC_BREAKPOINT_TYPE` e `SEMANTIC_BREAKPOINT_THRESHOLD` como triggers de rebuild (junto a `EMBEDDING_MODEL`). |
| SC-08 | Testes em `tests/test_rag.py` que referenciam `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP` atualizados: asserts de boundary por tamanho de caractere removidos; mocks do splitter trocados de `RecursiveCharacterTextSplitter` para `SemanticChunker`. |
| SC-09 | Novo arquivo `tests/acceptance/test_semantic_chunking.py` com testes de aceitação cobrindo os ACs abaixo. |

---

## Critérios de Aceitação

| AC | Descrição |
|----|-----------|
| AC-1 | `cria_vectordb()` executada com os PDFs e JSONs de `data/documents/` não lança exceção e produz chunks com `doc_type` e `file_hash` preenchidos. |
| AC-2 | `adicionar_documento_incrementalmente()` com um JSON novo insere chunks cujo `page_content` não é cortado no meio de uma frase (nenhum chunk termina sem pontuação de fim de sentença em texto corrido). |
| AC-3 | `adicionar_pdf_incrementalmente()` com um PDF novo insere chunks onde cada chunk é uma unidade semântica completa. |
| AC-4 | Nenhuma referência a `CHUNK_SIZE` ou `CHUNK_OVERLAP` aparece em `config.py`, `rag.py` ou nos imports de `test_rag.py` após a feature. |
| AC-5 | `SemanticChunker` é inicializado com `breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE` e `breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD` em todas as três funções. |

---

## Requisitos Não-Funcionais

| ID | Requisito |
|----|-----------|
| SC-NFR-01 | Cobertura de testes ≥ 90% (atual: 93% — não regredir abaixo de 90%). |
| SC-NFR-02 | `ruff check`, `mypy`, `bandit` limpos após a mudança. |
| SC-NFR-03 | `data/vectordb/` **deve ser deletado e reconstruído** após merge — chunks existentes foram gerados com boundary diferente; não há migração automática. |
| SC-NFR-04 | Ingestão mais lenta (SemanticChunker embeda cada sentença para detectar breakpoints) é aceita — o pipeline de ingestão é offline. |

---

## Fora de Escopo

- Calibração empírica de `SEMANTIC_BREAKPOINT_THRESHOLD` com os documentos de produção (pode ser ajustado pós-merge via `config.py` + rebuild).
- Mudança no modelo de embedding.
- Alteração no pipeline de recuperação (`agent.py`) ou na API (`api.py`).

---

## Open Questions

| # | Pergunta | Impacto |
|---|----------|---------|
| OQ-1 | `breakpoint_threshold_amount=95` é o valor certo para os PDFs de logística atuais? | Chunks muito grandes se o threshold for alto demais; fragmentação excessiva se for baixo. Recomendado: testar com `python -m agenticlog.rag` pós-merge e ajustar se chunks > 2000 chars aparecerem com frequência. |
| OQ-2 | Manter `CHUNK_SIZE`/`CHUNK_OVERLAP` comentados em `config.py` como referência histórica, ou remover completamente? | Decisão tomada no spec: **remover** — constantes sem uso induzem confusão. Histórico disponível em git. |

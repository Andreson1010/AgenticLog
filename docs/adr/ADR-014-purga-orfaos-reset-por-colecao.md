# ADR-014 — Purga de segmentos órfãos no rebuild (reset por coleção, wipe condicional)

**Status:** Aceito
**Data:** 2026-06-23
**Feature:** rag-audit-p0
**PR:** #51
**Relacionado:** ADR-010 (rebuild trigger), PR #45 (`_resetar_colecao` antes de gravar)

---

## Contexto

A auditoria do pipeline RAG (`docs/rag-audit-2026-06-23.md`) revelou que a coleção ativa
`logistica` estava **vazia** (`count() == 0`), enquanto o `chroma.sqlite3` continha 2982
embeddings espalhados em 4 segmentos **órfãos** — segmentos que não pertenciam a nenhuma
coleção viva. Toda query retornava 0 documentos.

Causa raiz: `_resetar_colecao` (PR #45) chamava `client.delete_collection(name)` antes de
`Chroma.from_documents`. O `delete_collection` do ChromaDB **não remove os segmentos/embeddings
do disco** de forma confiável entre rebuilds; rebuilds sucessivos (ou interrompidos) acumulavam
segmentos mortos enquanto a coleção recriada podia acabar sem embeddings vinculados — sem
nenhum erro. RAG funcionalmente offline e silencioso.

---

## Decisão

`_resetar_colecao` passa a decidir o método de descarte conforme o estado do vector DB:

- **Coleção alvo é a única (ou o DB não existe):** remove `DIR_VECTORDB` inteiro
  (`shutil.rmtree`). Isso elimina segmentos órfãos **por construção** — `from_documents`
  recria tudo do zero. É o caso comum atual (single-collection).
- **Multi-coleção:** descarta apenas a coleção alvo via `delete_collection`, **preservando as
  coleções irmãs**. A integridade do rebuild fica garantida pelo guardrail de contagem
  (ADR-015).

A decisão usa `_outras_colecoes_existem`, que lê a tabela `collections` do SQLite em modo
**read-only** — sem abrir um cliente Chroma, que seguraria um lock e faria o `rmtree` falhar no
Windows. Schema ausente/ilegível degrada para o wipe completo (caminho seguro p/ órfãos).

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| Manter `delete_collection` apenas | É a causa raiz — deixa segmentos órfãos; a coleção pode renascer vazia silenciosamente. |
| Sempre `rmtree` do diretório inteiro | Simples e purga órfãos, mas destrói **todas** as coleções — inseguro quando multi-collection (`RETRIEVAL_K_PER_COLLECTION` já prepara esse cenário). |
| GC de segmentos órfãos via API pública do Chroma | Não há API pública estável p/ isso; manipular o SQLite interno seria frágil. |
| Abrir cliente Chroma para contar coleções antes do `rmtree` | O cliente segura lock sobre o `.sqlite3` → `rmtree` falha no Windows. Leitura SQLite read-only evita o lock. |

---

## Consequências

**Positivas:**
- Rebuild sempre produz índice limpo, sem órfãos; coleção ativa garantidamente populada.
- Coleções irmãs preservadas no cenário multi-collection.
- Decisão lock-free → robusta no Windows.

**Negativas / Riscos:**
- O caminho multi-coleção ainda usa `delete_collection` (que pode deixar órfãos daquela
  coleção). Mitigado pelo guardrail de contagem (ADR-015) e pelo fato de o cenário
  multi-collection não estar em uso hoje. Follow-up se/quando ativado.
- Leitura direta do schema `collections` acopla a um detalhe interno do Chroma; degrada para o
  wipe seguro se o schema mudar.

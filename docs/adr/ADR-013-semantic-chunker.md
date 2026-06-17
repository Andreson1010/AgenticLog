# ADR-013 — Substituir RecursiveCharacterTextSplitter por SemanticChunker

**Status:** Proposto
**Data:** 2026-06-17
**Feature:** semantic-chunking
**Supercede:** ADR-007 (sentence separators — obsoleto com SemanticChunker)
**Relacionado:** ADR-008 (chunking por chave/página — preservado)

---

## Contexto

O pipeline usa `RecursiveCharacterTextSplitter(CHUNK_SIZE=500, CHUNK_OVERLAP=50)` com
separadores de sentença (ADR-007) e granularidade por chave JSON / página PDF (ADR-008).

A combinação de ADR-007 + ADR-008 reduziu cortes arbitrários mas não os eliminou: unidades
que excedem 500 chars (páginas densas de PDF de logística, campos JSON com listas longas) ainda
são partidas onde o limite de caractere cai — não onde o significado muda. O resultado são chunks
que começam ou terminam no meio de um conceito logístico (ex.: critério de desempenho partido
entre dois chunks diferentes), degradando a qualidade de recuperação.

A causa raiz é estrutural: qualquer splitter baseado em tamanho fixo desconhece o conteúdo.

O projeto já carrega `paraphrase-multilingual-mpnet-base-v2` (768-dim, multilingual, bom para
PT-BR) como modelo de embedding. `langchain_experimental` fornece `SemanticChunker`, que usa
esse mesmo modelo para detectar breakpoints onde a similaridade entre sentenças adjacentes cai
abaixo de um limiar — sem modelo adicional.

---

## Decisão

Substituir `RecursiveCharacterTextSplitter` por `SemanticChunker` nos três pontos de uso em
`rag.py`:

```python
# Antes
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)

# Depois
from langchain_experimental.text_splitter import SemanticChunker
text_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,    # "percentile"
    breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,  # 95.0
)
```

Configuração exposta em `config.py`:

```python
SEMANTIC_BREAKPOINT_TYPE: str = "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD: float = 95.0
```

`CHUNK_SIZE` e `CHUNK_OVERLAP` removidos de `config.py` — sem outros usos fora do splitter.

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| Manter `RecursiveCharacterTextSplitter` com separadores melhores (ADR-007) | Paliativo — o problema é o limite fixo, não os separadores. |
| `SentenceTransformersTokenTextSplitter` | Respeita janela de tokens do encoder (128 tok p/ `paraphrase-multilingual-mpnet-base-v2`), mas ainda é splitter estrutural — não detecta fronteiras semânticas. |
| spaCy + agrupamento por similaridade | Qualidade potencialmente alta, mas ~200 MB de modelo PT adicional, nova dep pesada, manutenção extra. Custo alto para benefício marginal vs `SemanticChunker`. |
| `SemanticChunker` com `breakpoint_threshold_type="standard_deviation"` | Mais sensível a outliers em corpora pequenos; `"percentile"` é mais robusto para documentos de logística com densidade variável. |

---

## Justificativa

- `SemanticChunker` resolve a causa raiz (limite fixo), não o sintoma.
- Reutiliza `HuggingFaceEmbeddings` já carregado — zero modelo adicional.
- `langchain_experimental` já está no ecossistema LangChain do projeto; dep nova mínima.
- ADR-008 (1 unidade por chunk: chave JSON, página PDF) permanece válido — o `SemanticChunker`
  só entra quando a unidade é grande o suficiente para ser partida; unidades pequenas passam
  intactas (comportamento equivalente ao splitter anterior para campos ≤ threshold).
- ADR-007 (sentence separators) torna-se obsoleto: `SemanticChunker` não usa lista de
  separadores — detecta breakpoints via embedding.

---

## Consequências

**Positivas:**
- Chunks respeitam fronteiras conceituais; recuperação semântica mais precisa.
- Tamanho de chunk varia naturalmente com a densidade do conteúdo.
- Testes de boundary por caractere (`CHUNK_SIZE` exato, split residual) removidos — mais
  simples e menos frágeis.

**Negativas / Riscos:**
- **Ingestão mais lenta:** `SemanticChunker` embeda cada sentença para encontrar breakpoints.
  Aceitável — ingestão é offline (não bloqueia consultas).
- **Chunks sem tamanho máximo garantido:** parágrafos muito longos sem breakpoint forte geram
  chunks grandes. Mitigação: `breakpoint_threshold_amount=95` corta no 95º percentil de
  dissimilaridade, limitando chunks excepcionalmente grandes na prática.
- **API instável:** `langchain_experimental` tem versionamento menos rigoroso. Mitigação: fixar
  versão mínima em `requirements.txt` e monitorar changelogs em upgrades.
- **`data/vectordb/` deve ser reconstruído após merge** — chunks existentes têm boundaries
  incompatíveis com a nova estratégia. Não há migração automática.
- **Testes:** mocks de `RecursiveCharacterTextSplitter` em `test_rag.py` precisam ser migrados
  para `SemanticChunker`.

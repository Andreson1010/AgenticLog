# ADR-018 — Redesign da arquitetura por fronteira offline/online (em vez do split por coesão)

**Status:** Aceito
**Data:** 2026-06-25
**Feature:** arquitetura-alvo-rag
**PR:** (a definir)
**Relacionado:** `docs/arquitetura-alvo-rag.md`, ROADMAP "Manutenibilidade", auditoria `docs/rag-audit-2026-06-23.md`

---

## Contexto

`src/agenticlog/rag.py` tem 1013 linhas (viola o teto de 800 do coding-style) e mistura
concerns (segurança, extração, chunking, embeddings, metadados, store, upsert, CLI).
`agent.py` concentra retrieval + geração + grafo. O ROADMAP planejava um **split por coesão**
(mover funções para um pacote `rag/` com 9 submódulos, refactor puro, testes preservados).

Ao reavaliar, dois fatos mudaram a decisão:

1. **Custo real do split por coesão maior que o estimado:** a suíte tem **~638 referências**
   `agenticlog.rag.*` (patches/imports), sem mapeamento 1:1 para submódulos (nomes
   compartilhados como `DIR_DOCUMENTS`=97×, `Chroma`=71× caem em múltiplos módulos). Migração
   mecânica de patches é trabalhosa e o resultado não melhora a arquitetura — só reorganiza.

2. **A boa prática de RAG aponta outro corte.** Pesquisa (Azure, LLM Engineer's Handbook/FTI,
   Unstructured) converge: a fronteira-dorsal de um sistema RAG é **offline (escrita do índice)
   vs online (leitura do índice)**, com estágios como módulos coesos e trocáveis — não um
   agrupamento técnico arbitrário.

---

## Decisão

Adotar **redesign** orientado à fronteira offline/online, conforme a árvore-alvo documentada em
`docs/arquitetura-alvo-rag.md`:

- `ingestion/` (offline) com estágios `security/extraction/cleaning/chunking/embeddings/metadata/store` + `orchestrator` + `cli`;
- `retrieval/` (online) com `retriever/generation/graph/state`;
- `observability/`, `serving/`, `shared/` para cross-cutting.

Isso **substitui** o item "Split de `rag.py` por coesão" do ROADMAP como abordagem escolhida.

---

## Alternativas Consideradas

| Opção | Por que rejeitada |
|-------|-------------------|
| **Split por coesão (ROADMAP original)** | Refactor puro e baixo risco, mas só reorganiza por agrupamento técnico; não reflete os estágios do pipeline nem o corte offline/online; migração mecânica de ~638 patches sem ganho arquitetural. |
| **Pasta por micro-estágio** (`/cleaning`, `/chunking` como diretórios) | Over-engineering para ~2k linhas de produto: arquivos de 30-50 linhas + glue; cerimônia sem benefício neste tamanho. |
| **Manter `rag.py` monolítico** | Viola coding-style (800 linhas); estágios não testáveis isoladamente; gaps (re-rank, query-transform) sem slot natural. |
| **Reescrever tudo do zero (código + testes simultâneos)** | Remove o oráculo durante a operação mais arriscada; risco de perder silenciosamente os edge cases que a auditoria P0 adicionou. |

---

## Consequências

**Positivas:**
- Estrutura espelha a espinha dorsal RAG (offline/online) e os 12 estágios da auditoria.
- Estágios viram módulos coesos, logáveis e testáveis isoladamente.
- Componentes trocáveis (embedder, vector store, retriever) atrás de fronteiras claras.
- Gaps de retrieval (re-ranking — P2; query transformation) ganham slot natural em `retrieval/`.

**Negativas / Riscos:**
- **Não é refactor puro:** decompõe as funções verticais de ingestão → muda implementação.
- **Atomicidade do upsert** (PRs #42/#43) precisa ser reprojetada para atravessar módulos —
  ponto mais delicado.
- **Testes reescritos** (os ~638 `@patch` deixam de valer). Mitigação: camada de **testes de
  caracterização E2E sem patches internos** como oráculo estável durante a transição; o gate
  `rag_eval` (golden set) cobre o lado retrieval.
- Maior esforço que o split barato — justificável por ser decisão de aprendizado/arquitetura,
  não só conformidade de tamanho.

**Execução:** faseamento em `docs/arquitetura-alvo-rag.md` §7; conduzida pelo pipeline
feature-factory quando priorizada, com ADRs adicionais para decisões de design pontuais
(ex.: como reprojetar a atomicidade do upsert).

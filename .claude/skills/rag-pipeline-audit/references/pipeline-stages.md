# Os 12 estágios do pipeline RAG

Para cada estágio: o que faz, como é "bom" vs "frágil", o que procurar no código, e nota
AgenticLog. Use isto para preencher a tabela "Mapa do pipeline" da Fase 1. Estágio que **não
existe** no sistema é um achado de primeira ordem — registre como gap.

## Índice
1. Data ingestion · 2. Parsing & cleaning · 3. Chunking · 4. Embedding selection ·
5. Vector DB design · 6. Metadata filtering · 7. Hybrid search · 8. Query transformation ·
9. Re-ranking · 10. Context construction · 11. Observability · 12. Evaluation

---

## 1. Data ingestion
Como os documentos entram no sistema (upload, batch, incremental).
- **Bom:** incremental com dedup por hash; idempotente; upsert atômico (rollback em falha);
  formatos validados na borda.
- **Frágil:** rebuild total a cada mudança; sem dedup (índice duplica); falha parcial deixa
  estado inconsistente.
- **Procure:** funções de ingest/upsert, hash de conteúdo, rollback, validação de formato/tamanho.
- **AgenticLog:** `rag.py` — `ingerir_incrementalmente`, `adicionar_*_incrementalmente`
  (upsert atômico com `.bak`), dedup por `file_hash`. Forte aqui.

## 2. Parsing & cleaning
Extração de texto de PDFs/JSON/HTML e normalização (remover boilerplate, headers/footers, OCR).
- **Bom:** extração por página com estrutura preservada; remoção de ruído; trata PDF
  escaneado/corrompido; encoding robusto.
- **Frágil:** `pdf.extract_text()` cru jogado num blob; tabelas viram sopa de tokens; perde
  títulos/seções (sinal de estrutura que ajuda chunking e citação).
- **Procure:** PyMuPDF/pdfplumber/unstructured, limpeza de texto, tratamento de tabelas/imagens.
- **AgenticLog:** `extrair_texto_pdf` retorna `dict[str,str]` (1 entrada por página). Não há
  limpeza de boilerplate nem extração de tabelas — possível gap se os PDFs têm muito ruído.

## 3. Chunking
Como o texto é dividido. **Maior alavanca de retrieval.** Ver `chunking-techniques.md`.
- **Bom:** estratégia casada ao conteúdo; respeita fronteiras semânticas; overlap suficiente
  para não cortar contexto; tamanho casado à janela do embedder.
- **Frágil:** fixo por nº de caracteres ignorando frases; chunks gigantes (dilui o embedding)
  ou minúsculos (perde contexto); sem overlap.
- **Procure:** splitter, chunk_size/overlap, breakpoint semântico.
- **AgenticLog:** `SemanticChunker` (percentile 95) sobre embeddings. Bom default; medir se o
  threshold gera chunks coerentes (ver Context Precision).

## 4. Embedding selection
Modelo que vira texto em vetor. Define o teto de qualidade do retrieval semântico.
- **Bom:** modelo casado ao idioma e domínio; dimensão adequada; normalização; o **mesmo**
  modelo na ingestão e na query.
- **Frágil:** modelo só-inglês em corpus PT; modelo genérico em domínio técnico; mismatch
  ingest/query (degradação silenciosa).
- **Procure:** nome do modelo de embedding, dimensão, normalize, onde é instanciado.
- **AgenticLog:** `paraphrase-multilingual-mpnet-base-v2` (768d, multilíngue, bom p/ PT).
  Adequado. Conferir que ingest e query usam o mesmo (há instanciações em 3 sítios).

## 5. Vector DB design
Estrutura do índice: coleções, índice ANN, métrica de distância, persistência.
- **Bom:** métrica casada ao embedder (cosine p/ normalizado); coleções por domínio;
  parâmetros ANN (HNSW M/ef) conscientes; persistência confiável.
- **Frágil:** uma coleção gigante sem partição; métrica errada; defaults de ANN nunca revistos;
  índice duplicado por bug de ingestão.
- **Procure:** Chroma/FAISS/Qdrant/Pinecone, collection design, distance metric, HNSW params.
- **AgenticLog:** ChromaDB (HNSW, hnswlib), múltiplas coleções com fan-out. Conferir métrica de
  distância vs embeddings normalizados.

## 6. Metadata filtering
Filtrar por metadados (data, fonte, tipo, autor) antes/junto da busca vetorial.
- **Bom:** metadados ricos por chunk (source, page, doc_type, data); filtros expostos na query;
  reduz espaço de busca e melhora precisão.
- **Frágil:** sem metadados (só texto); metadados existem mas nunca são usados para filtrar.
- **Procure:** metadata nos chunks, `where=`/filtros na query.
- **AgenticLog:** chunks têm `source, file_hash, chunk_index, page, doc_type` (REC-01). Os
  metadados existem — **mas são usados para filtrar na query?** Provável gap (filtro não exposto).

## 7. Hybrid search
Combinar busca densa (vetorial) com esparsa (BM25/keyword), fundidas (ex.: RRF).
- **Bom:** denso + esparso com fusão; pega tanto sinônimos (denso) quanto termos exatos/siglas
  (esparso) — crítico em domínio técnico/jurídico/logístico.
- **Frágil:** só denso → erra em códigos, siglas, nomes próprios, números exatos.
- **Procure:** BM25, keyword search, RRF/reciprocal rank fusion, ensemble retriever.
- **AgenticLog:** parece **só denso** (similaridade vetorial). Provável gap de alto impacto se as
  queries têm termos exatos (códigos de rota, contratos, SKUs).

## 8. Query transformation
Reescrever/expandir a query antes de buscar (HyDE, multi-query, decomposição, step-back).
- **Bom:** expande queries curtas/ambíguas; decompõe perguntas multi-parte; HyDE p/ gap
  vocabulário pergunta↔documento.
- **Frágil:** query do usuário vai crua pro retriever; perguntas compostas recuperam mal.
- **Procure:** query rewrite, multi-query, HyDE, sub-questions.
- **AgenticLog:** roteamento retrieve-first decide rota, mas **não reescreve** a query. Gap
  potencial p/ perguntas vagas/compostas.

## 9. Re-ranking
Reordenar os top-N recuperados com um modelo mais caro/preciso (cross-encoder) antes de
montar o contexto.
- **Bom:** recall alto no retrieval (top-50) → cross-encoder reordena → top-5 de alta precisão.
  Sobe MRR e Context Precision sem perder recall.
- **Frágil:** sem re-ranking → a ordem do ANN é a ordem final → o melhor chunk pode ficar na
  posição 8 e ser cortado pela janela.
- **Procure:** cross-encoder, cohere/bge reranker, rerank step.
- **AgenticLog:** ranqueia **respostas geradas** por similaridade (não os chunks recuperados).
  **Não há re-ranking de chunks** — gap clássico de alto impacto.

## 10. Context construction
Como os chunks viram o prompt: ordem, dedup, compressão, orçamento de tokens, citação.
- **Bom:** dedup de chunks quase-iguais; ordena por relevância; respeita orçamento de tokens;
  preserva fonte p/ citação; "lost in the middle" mitigado.
- **Frágil:** concatena tudo sem dedup; estoura/desperdiça a janela; sem citação de fonte.
- **Procure:** montagem do prompt, dedup, token budget, ordenação dos chunks.
- **AgenticLog:** fan-out multi-coleção com **dedup por MD5** no `_get_retriever`. Conferir
  ordenação e orçamento de tokens.

## 11. Observability
Logs/traces de cada query: o que foi recuperado, scores, latência, custo, rota tomada.
- **Bom:** trace por query (chunks + scores + tempo + tokens); auditoria; dá pra depurar
  "por que essa resposta ruim?".
- **Frágil:** caixa-preta — sem visibilidade do retrieval; impossível diagnosticar regressão.
- **Procure:** logging estruturado do retrieval, LangSmith/Phoenix/traces, history/audit.
- **AgenticLog:** logging estruturado (JSON) + `history.py` (auditoria de queries). Há base —
  conferir se o **retrieval** (chunks+scores) é logado, não só a resposta final.

## 12. Evaluation
Como a qualidade é medida ao longo do tempo (offline metrics, regressão, golden set).
- **Bom:** golden set versionado; métricas de retrieval+generation em CI; alerta de regressão.
- **Frágil:** sem avaliação sistemática; "parece bom" é o único critério; regressão silenciosa.
- **Procure:** evals/, ragas/deepeval, golden set, métricas em CI.
- **AgenticLog:** **sem avaliação de qualidade de RAG** (os testes são unit/integração de
  código, não de qualidade de resposta). Gap — este skill começa a fechá-lo (Fase 2).

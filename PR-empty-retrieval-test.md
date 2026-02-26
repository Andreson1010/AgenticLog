# PR: feature/empty-retrieval-test – Teste de recuperação vazia

## Resumo

Adiciona testes unitários para o cenário de **recuperação vazia** no sistema Agentic RAG: quando o retriever não encontra documentos relevantes para a query, o pipeline deve se comportar corretamente sem erros.

## Objetivo

- Garantir que o fluxo RAG lida corretamente com `retrieved_info = []`
- Validar que `retrieve_info`, `gera_multiplas_respostas` e `avalia_similaridade` tratam o caso vazio
- Evitar regressões quando consultas não retornam documentos (ex.: VectorDB vazio ou query muito específica)

## Solução

Novos métodos de teste em `tests/test_agentic_rag.py`:

1. **teste_5b_retrieve_info_empty** – retriever retorna lista vazia; `state.retrieved_info` permanece `[]`
2. **teste_6b_gera_multiplas_respostas_empty_context** – com `retrieved_info = []`, o context passado ao LLM é string vazia
3. **teste_7b_avalia_similaridade_empty_retrieved** – com `retrieved_info = []`, `similarity_scores` são zerados `[0.0, 0.0]`

Os testes usam `unittest.mock` para isolar o comportamento sem dependências externas.

---

## Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `tests/test_agentic_rag.py` | **Modificado** – adicionados 3 testes de recuperação vazia |

---

## Mensagem de commit sugerida (Conventional Commits)

```
test: add empty retrieval tests for agentic RAG

Covers retrieve_info, gera_multiplas_respostas and avalia_similaridade
when retriever returns no documents.
```

---

## Checklist do PR

- [ ] Branch criada: `feature/empty-retrieval-test`
- [ ] Arquivo `tests/test_agentic_rag.py` modificado
- [ ] Testes executados localmente: `python -m unittest discover -s tests -v`
- [ ] CI verde após push (workflow `test.yml` executa todos os testes)
- [ ] `git pull origin main` após merge (conforme Protocolo de Git)

---

## Comandos para abrir o PR

```bash
git checkout -b feature/empty-retrieval-test
git add tests/test_agentic_rag.py
git commit -m "test: add empty retrieval tests for agentic RAG"
git push -u origin feature/empty-retrieval-test
```

Depois, abrir o PR no GitHub da branch `feature/empty-retrieval-test` para `main`.

---

## Detalhamento dos testes

| Classe | Método | Descrição |
|--------|--------|-----------|
| TestAgenticRAG | teste_5b_retrieve_info_empty | Retriever retorna `[]` → `retrieved_info` vazio |
| TestAgenticRAG | teste_6b_gera_multiplas_respostas_empty_context | `retrieved_info = []` → context `""` passado ao LLM |
| TestAgenticRAG | teste_7b_avalia_similaridade_empty_retrieved | `retrieved_info = []` → `similarity_scores = [0.0, 0.0]` |

---

## Observações

- **Comportamento esperado**: O prompt RAG (`prompt_rag_retrieve`) instrui o LLM a responder "Sorry, I did not find that information in the documents." quando o context está vazio.
- **avalia_similaridade**: Com `retrieved_embeddings` vazio, a função retorna early e define `similarity_scores = [0.0] * len(response_texts)`.
- **Compatibilidade**: Segue o mesmo padrão de mocks de `test_agentic_rag.py` (patch em `retriever`, `prompt_rag_retrieve`, `embedding_model`).

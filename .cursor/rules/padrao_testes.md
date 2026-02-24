---
description: Padrões de teste para o AgenticLog (RAG e LangGraph)
globs: tests/**/*.py
---

# Padrões de Teste - AgenticLog

Sempre que for sugerir ou criar testes em `tests/test_agentic_rag.py`, siga estas diretrizes:

1. **Mocks de LLM**: Nunca faça chamadas reais para a API da OpenAI/Anthropic nos testes. Use `unittest.mock` ou `pytest-mock` para simular as respostas do modelo.
2. **Validação de Grafo**: Para o LangGraph, teste se o estado (`AgentState`) contém as chaves obrigatórias após cada nó: `query`, `documents` e `ranked_response`.
3. **Casos de Borda**: Sempre inclua um teste para "Recuperação Vazia" (quando o RAG não encontra documentos) e verifique se o agente desvia corretamente para "Busca Web".
4. **Similaridade**: Valide se a lógica de escolha da "melhor resposta" no `rag.py` está respeitando o threshold definido em `config.py`.
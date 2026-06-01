# ADR-005 — Remover langchain.agents e usar chain direta em usar_ferramenta_web

**Status:** Aceito  
**Data:** 2026-06-01  
**Commit:** `refactor(agent): substituir langchain.agents por chain direta no nó usar_web`

## Contexto

O módulo `agent.py` importava `initialize_agent`, `AgentType` e `AgentExecutor` de `langchain.agents` para implementar o nó `usar_ferramenta_web` com DuckDuckGo. Ao instalar as dependências com `langchain==0.3.19` e `langchain-core==0.3.35`, a importação falhava com:

```
ModuleNotFoundError: No module named 'langchain_core.memory'
```

O caminho da falha era: `langchain.agents` → `langchain.schema` → `from langchain_core.memory import BaseMemory`. Em `langchain_core 0.3.x`, `BaseMemory` foi removido de `langchain_core.memory`, quebrando toda a cadeia de imports de `langchain.agents`.

## Decisão

Remover completamente a dependência de `langchain.agents` e `langchain.tools` de `agent.py`. O nó `usar_ferramenta_web` passou a:

1. Chamar `search.run(query)` diretamente via `DuckDuckGoSearchAPIWrapper`
2. Montar chain direta: `_prompt_web | _get_llm() | StrOutputParser()`
3. Chamar `_invoke_chain(chain, {"context": resultados, "input": query})`

A função `_invoke_executor` foi mantida (com `@_llm_retry`) para não quebrar testes de aceite existentes que a testam diretamente.

## Justificativa

- O projeto usa LangGraph para orquestração — `AgentExecutor` era redundante
- Chain direta é mais simples, testável e sem dependências frágeis
- O resultado funcional é idêntico: busca DuckDuckGo + LLM com contexto
- Evita fixar versão de `langchain` em valor incompatível com `langchain_core`

## Consequências

- `langchain.agents` e `langchain.tools` não são mais importados em `agent.py`
- Todos os mocks de `_get_avk_agent_executor` nos testes foram migrados para `_invoke_chain`
- Comportamento do nó `usar_web` é preservado: DuckDuckGo falha → fallback; sucesso → LLM responde com contexto

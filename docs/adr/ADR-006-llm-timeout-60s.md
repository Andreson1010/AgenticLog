# ADR-006 — Aumentar LLM_TIMEOUT_SECONDS de 10s para 60s

**Status:** Aceito  
**Data:** 2026-06-01  
**Commit:** `fix(config): aumentar LLM_TIMEOUT_SECONDS de 10s para 60s`

## Contexto

`LLM_TIMEOUT_SECONDS` estava configurado em `10.0s` em `config.py`. O modelo Hermes 3B rodando no LMStudio local gera respostas a ~18 tokens/segundo. Respostas de 150+ tokens levam ~8-9s só de geração, sem contar o tempo de processamento do prompt. Com timeout de 10s, o cliente OpenAI desconectava antes do LMStudio terminar, causando:

```
APITimeoutError: Request timed out.
```

Os logs do LMStudio confirmavam: `Client disconnected. Stopping generation...` após ~8s de geração.

## Decisão

Aumentar `LLM_TIMEOUT_SECONDS` de `10.0` para `60.0` segundos.

## Justificativa

- Hermes 3B local é mais lento que APIs na nuvem — timeout de 10s não é realista
- 60s cobre respostas longas (até ~1000 tokens a 18 t/s) com margem
- O timeout existe para falhas reais (LMStudio desligado), não para geração lenta
- Valor configurável via `config.py` — pode ser ajustado sem tocar no código do agente

## Consequências

- Timeout padrão aumentado para 60s
- Teste `test_ac05_timeout_constant_value_is_ten_seconds` atualizado para `60.0`
- Usuários com modelos mais lentos não terão timeout prematuro
- Em caso de LMStudio realmente indisponível, a espera antes do erro aumenta de 10s para 60s por tentativa

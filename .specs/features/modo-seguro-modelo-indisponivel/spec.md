# Spec — Modo seguro quando o modelo LMStudio está indisponível

- **Feature slug:** `modo-seguro-modelo-indisponivel`
- **Branch:** `feature/modo-seguro-modelo-indisponivel`
- **TLC scope:** medium (`spec.md` apenas)
- **Story aprovada:** Checkpoint 1 (3 open questions resolvidas)
- **Open questions:** Nenhuma — todas resolvidas no Checkpoint 1.

---

## 1. Visão geral / problema

Hoje, quando o LMStudio está indisponível, `POST /query` falha:
- Se o servidor está fora ou o modelo desconecta no meio da chamada, `agent_workflow.invoke` re-levanta `httpx.ConnectError` / `openai.APIConnectionError` após esgotar o retry (tenacity, `agent.py:66-79`), e os exception handlers do `api.py` (`api.py:322-339`) devolvem **HTTP 503**.
- O caso "servidor de pé mas modelo errado carregado" **não é detectado** — `check_lmstudio_health()` (`health.py:62-130`) sabe detectá-lo (`ModeloNaoCarregadoError`), mas **não tem nenhum caller em produção** (o `app.py` foi migrado para cliente HTTP puro no PR de streamlit-http-client, deixando o health-check órfão).

Esta feature introduz o **modo seguro**: em vez de 503, `POST /query` devolve **HTTP 200** com uma mensagem fixa segura (`RESPOSTA_PADRAO_SEGURA`, `confidence_score=0.0`, `retrieved_info=[]`, `degraded=true`). A degradação acontece no **boundary do `api.py`**, com dois gatilhos:
1. **Pre-flight** `check_lmstudio_health()` antes do `invoke` — único caminho capaz de pegar `ModeloNaoCarregadoError`; e finalmente **pluga** o health-check órfão.
2. **Mid-call** captura de `openai.APIConnectionError` / `httpx.ConnectError` re-levantadas do `invoke`.

Trade-off aceito pelo usuário: o contrato do `/query` para modelo-down muda de **503 → 200-degraded**. O grafo LangGraph e o `AgentState` **não** mudam.

---

## 2. Functional Requirements

Rastreabilidade: AC-1..AC-7 da story + decisões D1 (auditoria), D2 (handlers 503), D3 (local da constante).

- **FR-1 — Happy path inalterado + campo novo.** Com modelo saudável, modelo configurado carregado e vectordb presente, `POST /query` válido retorna **200** com `degraded=false`, `ranked_response` real do workflow, `confidence_score` real e `retrieved_info` real. (AC-1)
- **FR-2 — Servidor fora (pre-flight).** Quando `check_lmstudio_health()` levanta `LMStudioUnavailableError` por falha de conexão/timeout/status não-2xx/JSON inválido, `POST /query` retorna **200** com `ranked_response=RESPOSTA_PADRAO_SEGURA`, `confidence_score=0.0`, `retrieved_info=[]`, `degraded=true`. **Não 503.** (AC-2)
- **FR-3 — Modelo errado carregado (pre-flight).** Quando `check_lmstudio_health()` levanta `ModeloNaoCarregadoError` (subclasse de `LMStudioUnavailableError`), `POST /query` retorna **200-degraded** idêntico ao FR-2. Caso novo, sem cobertura prévia. (AC-3)
- **FR-4 — Falha mid-call.** Quando o pre-flight passa mas o `invoke` re-levanta `openai.APIConnectionError` ou `httpx.ConnectError` (retry esgotado), `POST /query` retorna **200-degraded** idêntico ao FR-2. (AC-4)
- **FR-5 — Invariantes da resposta degradada.** Toda resposta degradada tem `ranked_response` exatamente igual à constante `RESPOSTA_PADRAO_SEGURA` (string fixa, sem variação por query), `confidence_score == 0.0` e `retrieved_info == []`. (regra de negócio)
- **FR-6 — `degraded` sempre presente.** Todo `200` de `POST /query` carrega o campo `degraded` (booleano), nunca omitido; `false` no caminho normal. (AC-1, regra de negócio)
- **FR-7 — Auditoria do evento degradado (D1).** Uma resposta degradada **é gravada** no `history_store` (via `_construir_registro` + `history_store.append`) com o marcador de degradação, `confidence_score=0.0` e a `query`, consistente com o registro do caminho normal. A falha de gravação no histórico **não** quebra a resposta (isolada por try/except, como hoje em `api.py:282-284`).
- **FR-8 — Handlers 503 mantidos como backstop (D2).** Os `@app.exception_handler(LMStudioUnavailableError)` e `@app.exception_handler(httpx.ConnectError)` (`api.py:322-339`) **permanecem** registrados como rede de segurança para outras rotas/escapes não previstos. Eles não disparam mais no `/query` (a degradação captura inline). Os testes 503 do `/query` são **re-escopados** para esperar 200-degraded (não deletados).
- **FR-9 — UI badge "modo seguro" (D3).** `RESPOSTA_PADRAO_SEGURA` vive em `config.py`; `api.py` a serve no `ranked_response`. O `app.py` **não importa** a constante — renderiza o `ranked_response` do servidor e lê `degraded` via `output.get("degraded", False)`. Quando `degraded=true`, a UI mostra o badge "modo seguro" junto da mensagem (HTML-escaped, padrão atual); quando `false`, não mostra. (AC-5, AC-6)

---

## 3. Non-Functional Requirements

- **NFR-1 — Não bloquear o event loop.** `check_lmstudio_health()` é síncrono (`httpx.Client`); o pre-flight deve rodar via `await asyncio.to_thread(...)`, consistente com `api.py:275/284/308`.
- **NFR-2 — Latência do happy path.** O pre-flight proba o LMStudio a cada `/query` (o sentinel `_health_checked` é setado mas nunca lido — não confiar nele; sem cache, por design, consistente com o "Out of Scope" do health-check). Probe saudável custa ~ms; o timeout de 5s (`LLM_HEALTH_CHECK_TIMEOUT_SECONDS`) só ocorre com servidor fora. Aceito.
- **NFR-3 — Imutabilidade / coding-style.** Helpers puros, sem mutação; funções < 50 linhas; constante sem hardcode espalhado (única em `config.py`).
- **NFR-4 — Cobertura ≥ 80%.** TDD; manter/elevar a cobertura atual do projeto.
- **NFR-5 — Sem novas credenciais** nem dados sensíveis em log/resposta.

---

## 4. Files That Will Change

**Backend (backend-builder): `config.py`, `api.py` (+ wiring do `health.py`)**

- `src/agenticlog/config.py` — adicionar `RESPOSTA_PADRAO_SEGURA: str = "..."  # comentário` no bloco de constantes (perto de LLM/health, linhas 27-38), seguindo o padrão `NAME: type = value  # comment`. Texto sugerido: `"Serviço de IA indisponível no momento. Tente novamente mais tarde."` (ajustável).
- `src/agenticlog/api.py`:
  - **Imports** (hoje só `LMStudioUnavailableError` em `api.py:29`): adicionar `check_lmstudio_health` e `ModeloNaoCarregadoError` (de `agenticlog.health`), `APIConnectionError` (de `openai`), `RESPOSTA_PADRAO_SEGURA` (de `agenticlog.config`).
  - **`QueryResponse`** (`api.py:73-80`): adicionar `degraded: bool = False`. Preservar os campos existentes (`ranked_response`, `confidence_score`, `retrieved_info`, `next_step`).
  - **Novo helper puro `_resposta_segura(query: str) -> QueryResponse`**, espelhando `_normalizar_estado` (`api.py:198-237`): retorna `QueryResponse(ranked_response=RESPOSTA_PADRAO_SEGURA, confidence_score=0.0, retrieved_info=[], degraded=True, next_step="")`.
  - **`consultar` / `POST /query`** (`api.py:260-288`): após o guard de `vectordb_pronto` (`api.py:272`), rodar pre-flight `await asyncio.to_thread(check_lmstudio_health)`; envolver pre-flight + `invoke` num try/except que captura `(LMStudioUnavailableError, APIConnectionError, httpx.ConnectError)` → construir `_resposta_segura(query)`, gravar no histórico (FR-7) e retornar 200. Caminho normal segue como hoje.
  - **`_normalizar_estado`** (`api.py:198-237`): garantir `degraded=False` no caminho normal (coberto pelo default do campo).
  - **Handlers 503** (`api.py:322-339`): **mantidos** (FR-8).
- `src/agenticlog/health.py` — **sem alteração de código**; apenas passa a ter caller real. (`reset_health_check_sentinel` continua para testes.)

**Frontend (frontend-builder): `app.py`**

- `app.py`:
  - **Leitura do `degraded`**: na área de resposta (`app.py:528-532`), ler `degraded = output.get("degraded", False)` (defende API antiga → `false`).
  - **Badge**: na linha `.cl-meta` (`app.py:601-603`), quando `degraded`, adicionar um badge `.cl-badge` (`app.py:307-314`) com texto estático escapado, ex.: `"⚠️ modo seguro"`. Reusar o padrão de `route_html` (escape mesmo em texto constante "por robustez estrutural").
  - **Session-state** (`app.py:425-434`): adicionar chave se necessário para o estado de exibição.
  - O `app.py` **não** importa `RESPOSTA_PADRAO_SEGURA`; renderiza o `ranked_response` recebido.

**Testes**

- `tests/test_api.py` — re-escopar `teste_9_lmstudio_indisponivel_retorna_503` e `teste_10_connect_error_retorna_503` para esperar **200 + degraded=true + RESPOSTA_PADRAO_SEGURA**. Adicionar teste do caso `ModeloNaoCarregadoError` no pre-flight → 200-degraded. Mockar namespace consumidor (`agenticlog.api.*`), resetar sentinel em setUp/tearDown quando exercitar o pre-flight.
- `tests/acceptance/test_api_query_endpoint.py` — re-escopar `test_ac_api_04_lmstudio_unavailable_503` para 200-degraded.
- `tests/acceptance/test_modo_seguro_modelo_indisponivel.py` (**novo**) — testes de aceitação por AC (ver §8), incluindo o caso novo de `ModeloNaoCarregadoError` e a gravação no histórico.
- `tests/test_app_error_handler.py` / `tests/test_streamlit_ui.py` — adicionar teste de render do badge "modo seguro" quando `degraded=true` e ausência quando `false`/ausente; ajustar fixtures de resposta para incluir a chave `degraded`.

---

## 5. Data Model Changes

- `QueryResponse.degraded: bool = False` (novo campo, default `False`).
- Registro de histórico do evento degradado: reusa o shape de `_construir_registro` com `confidence_score=0.0`, `ranked_response=RESPOSTA_PADRAO_SEGURA`, marcador de degradação. Sem mudança no schema do `history_store` além do conteúdo do registro.
- **Sem** mudança em `AgentState`.

---

## 6. API Contract — `POST /query`

Resposta 200 (shape):
```json
{
  "ranked_response": "string",
  "confidence_score": 0.0,
  "retrieved_info": [],
  "next_step": "",
  "degraded": true
}
```
- `degraded=false`: resposta normal do workflow (campos reais).
- `degraded=true`: modo seguro — `ranked_response=RESPOSTA_PADRAO_SEGURA`, `confidence_score=0.0`, `retrieved_info=[]`.

Exceções:
- Capturadas **inline** no `/query` → 200-degraded: `LMStudioUnavailableError` (inclui `ModeloNaoCarregadoError`), `openai.APIConnectionError`, `httpx.ConnectError`.
- Ainda **503** (inalterado): vectordb ausente (guard antes do modelo, `api.py:272`); e os handlers 503 backstop para outras rotas/escapes.

---

## 7. Riscos

- **Handlers 503 mortos para `/query`** — mantidos como backstop (FR-8); risco de 503 inconsistente vazar por escape não previsto é mitigado pela captura inline cobrir os 3 tipos.
- **Latência do happy path** — probe a cada query (NFR-2). Aceito; sem cache por design.
- **Sentinel sem cache** (`_health_checked` setado, nunca lido) — não confiar nele; garante recovery imediato (servidor volta → próxima query normal).
- **httpx vs openai** — pre-flight levanta `LMStudioUnavailableError`/`ModeloNaoCarregadoError`; mid-call re-levanta `httpx.ConnectError`/`openai.APIConnectionError`. A captura precisa cobrir as três famílias.
- **API antiga sem `degraded`** — UI usa `output.get("degraded", False)`; não quebra.
- **XSS** — texto do badge estático e escapado; mensagem segura também escapada no render atual.

---

## 8. Test Plan (FR → testes)

| FR / AC | Teste | Tipo |
|---|---|---|
| FR-1 / AC-1 | `/query` saudável → 200, `degraded=false`, resposta real | unit (test_api) + acc |
| FR-2 / AC-2 | pre-flight `LMStudioUnavailableError` (conn) → 200-degraded, invariantes | unit (re-escopa teste_9) + acc |
| FR-3 / AC-3 | pre-flight `ModeloNaoCarregadoError` → 200-degraded (**novo**) | unit + acc |
| FR-4 / AC-4 | `invoke` levanta `APIConnectionError`/`httpx.ConnectError` → 200-degraded | unit (re-escopa teste_10) + acc |
| FR-5 | invariantes exatos (`ranked_response`==const, `0.0`, `[]`) | unit |
| FR-6 | `degraded` presente e `false` no happy path | unit |
| FR-7 / D1 | resposta degradada gravada no `history_store`; falha de histórico não quebra resposta | unit (mock history_store) |
| FR-8 / D2 | handlers 503 ainda registrados; `test_ac_api_04` re-escopado p/ 200-degraded | unit + acc |
| FR-9 / AC-5,6 | UI: `degraded=true` → badge; `false`/ausente → sem badge; escape | unit (app/streamlit) |

Convenções: unit `teste_N_`; aceitação `test_ac_modo_seguro_<n>_`; mock no namespace consumidor; sentinel reset em setUp/tearDown quando o pre-flight é exercitado.

---

## 9. Open Questions

Nenhuma — todas resolvidas no Checkpoint 1 (D1 gravar no histórico; D2 manter handlers 503; D3 constante em `config.py`, servida no body).

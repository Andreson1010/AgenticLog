"""
Camada FastAPI do AgenticLog.

Expoe o workflow LangGraph como uma API REST, normalizando entradas e saidas
do agente para um contrato JSON estavel. Toda chamada bloqueante ao agente
executa em thread-pool via asyncio.to_thread (nunca bloqueia o event loop).
"""

import asyncio
import contextlib
import datetime
import json
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from openai import APIConnectionError
from pydantic import BaseModel, Field, field_validator

from agenticlog.config import (
    DEFAULT_COLLECTION_NAME,
    DIR_VECTORDB,
    HISTORY_FILE,
    HISTORY_MAX_ENTRIES,
    RESPOSTA_PADRAO_SEGURA,
)
from agenticlog.ingestion.embeddings import _get_rag_embedding_model
from agenticlog.observability.history import HistoryStore
from agenticlog.retrieval.graph import AgentState, agent_workflow, inicializar_recursos
from agenticlog.serving.health import (
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    check_lmstudio_health,
)

logger = logging.getLogger("agenticlog.api")

# Mensagens de erro padronizadas para respostas HTTP
MSG_LMSTUDIO_INDISPONIVEL = (
    "LMStudio indisponivel. Inicie o servidor e carregue o modelo."
)
MSG_VECTORDB_AUSENTE = (
    "Base vetorial nao encontrada. Execute: python -m agenticlog.ingestion"
)

# Excecoes que disparam o modo seguro (200-degraded) no /query. ModeloNaoCarregadoError
# e subclasse de LMStudioUnavailableError mas listada explicitamente por clareza do contrato.
_ERROS_MODO_SEGURO: tuple[type[Exception], ...] = (
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,  # redundante (subclasse) — NAO remover: sinaliza o contrato
    APIConnectionError,
    httpx.ConnectError,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Corpo da requisicao POST /query."""

    query: str = Field(..., min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_and_check(cls, v: object) -> object:
        """Remove espacos e valida comprimento minimo de 1 apos strip."""
        if isinstance(v, str):
            stripped = v.strip()
            if len(stripped) == 0:
                raise ValueError("query nao pode ser vazia ou conter apenas espacos")
            return stripped
        return v


class DocumentInfo(BaseModel):
    """Representacao serializavel de um documento recuperado."""

    page_content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Corpo da resposta bem-sucedida de POST /query."""

    ranked_response: str
    confidence_score: float
    next_step: str
    retrieved_info: list[DocumentInfo]
    degraded: bool = False


class HistoryEntry(BaseModel):
    """Entrada individual do historico de queries."""

    id: int
    timestamp: str
    query: str
    next_step: str
    confidence_score: float
    ranked_response: str


# ---------------------------------------------------------------------------
# Helpers de startup
# ---------------------------------------------------------------------------


def _verificar_vectordb() -> None:
    """Verifica se o diretorio do ChromaDB existe e abre a colecao padrao.

    Saida: nenhuma.
    Raises: RuntimeError com MSG_VECTORDB_AUSENTE se o diretorio estiver ausente
            ou a colecao padrao nao puder ser aberta.
    """
    if not Path(DIR_VECTORDB).exists():
        raise RuntimeError(MSG_VECTORDB_AUSENTE)
    Chroma(
        persist_directory=str(DIR_VECTORDB),
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_function=_get_rag_embedding_model(),
    )


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida da aplicacao FastAPI.

    Startup:
      - Verifica existencia do vectordb; se ausente, loga erro critico e marca
        app.state.vectordb_pronto = False (servidor sobe mesmo assim — 503 por req).
      - Inicializa singletons (LLM, ChromaDB, embeddings) uma unica vez.
    Shutdown: sem operacoes.
    """
    try:
        _verificar_vectordb()
        inicializar_recursos()
        app.state.vectordb_pronto = True
        logger.info("AgenticLog API iniciada. Recursos inicializados com sucesso.")
    except RuntimeError as exc:
        logger.critical(
            "Vectordb ausente na inicializacao: %s. Requisicoes retornarao 503.", exc
        )
        app.state.vectordb_pronto = False
    except Exception as exc:
        logger.critical(
            "Erro inesperado na inicializacao: %s. Requisicoes retornarao 503.", exc
        )
        app.state.vectordb_pronto = False
    try:
        app.state.history_store = HistoryStore(
            db_path=HISTORY_FILE, max_entries=HISTORY_MAX_ENTRIES
        )
        logger.info("HistoryStore inicializado em %s", HISTORY_FILE)
    except Exception as exc:
        logger.critical("Falha ao inicializar HistoryStore: %s. Historico indisponivel.", exc)
        app.state.history_store = None
    yield
    # Shutdown — sem recursos para liberar


# ---------------------------------------------------------------------------
# Aplicacao FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgenticLog API",
    description="REST API para o workflow Agentic RAG de logistica.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers de normalizacao (funcoes puras — sem I/O)
# ---------------------------------------------------------------------------


def _serializar_documentos(docs: list) -> list[DocumentInfo]:
    """Converte lista de Document LangChain ou dicts em lista de DocumentInfo.

    Entrada: lista de objetos com atributo page_content ou dicts equivalentes.
    Saida: lista de DocumentInfo — nunca None ou null.
    """
    resultado = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            resultado.append(
                DocumentInfo(
                    page_content=doc.page_content,
                    metadata=doc.metadata or {},
                )
            )
        elif isinstance(doc, dict):
            resultado.append(
                DocumentInfo(
                    page_content=doc.get("page_content", ""),
                    metadata=doc.get("metadata", {}),
                )
            )
    return resultado


def _normalizar_estado(estado: AgentState | dict) -> QueryResponse:
    """Normaliza AgentState (ou dict retornado pelo LangGraph) para QueryResponse.

    Entrada: estado retornado por agent_workflow.invoke() — AgentState ou dict.
    Saida: QueryResponse com ranked_response str, confidence_score float >= 0.0,
           retrieved_info list[DocumentInfo].

    Normalizacoes:
      - dict -> AgentState (LangGraph serializa o estado como dict em algumas versoes).
      - ranked_response dict {"answer": str} -> str (extrai valor).
      - ranked_response dict sem "answer" -> json.dumps seguro como fallback.
      - confidence_score None -> 0.0.
      - retrieved_info None -> [].
    """
    if isinstance(estado, dict):
        ranked_raw = estado.get("ranked_response", "")
        score = estado.get("confidence_score") or 0.0
        docs = _serializar_documentos(estado.get("retrieved_info") or [])
        next_step = estado.get("next_step", "")
        if isinstance(ranked_raw, dict):
            ranked_raw = ranked_raw.get("answer", json.dumps(ranked_raw, ensure_ascii=False))
        return QueryResponse(
            ranked_response=ranked_raw,
            confidence_score=score,
            next_step=next_step,
            retrieved_info=docs,
        )
    ranked = estado.ranked_response
    if isinstance(ranked, dict):
        ranked = ranked.get("answer", json.dumps(ranked, ensure_ascii=False))

    score = estado.confidence_score if estado.confidence_score is not None else 0.0
    docs = _serializar_documentos(estado.retrieved_info or [])

    return QueryResponse(
        ranked_response=ranked,
        confidence_score=score,
        next_step=estado.next_step,
        retrieved_info=docs,
    )


def _resposta_segura() -> QueryResponse:
    """Constroi a resposta do modo seguro quando o LMStudio esta indisponivel.

    A mensagem segura e fixa (RESPOSTA_PADRAO_SEGURA) e nao varia por query.
    Saida: QueryResponse degradado com confidence_score 0.0, retrieved_info [],
           degraded True e next_step "".
    """
    return QueryResponse(
        ranked_response=RESPOSTA_PADRAO_SEGURA,
        confidence_score=0.0,
        next_step="",
        retrieved_info=[],
        degraded=True,
    )


def _construir_registro(query: str, response: QueryResponse) -> dict:
    """Constroi o dict de registro para persistencia no historico.

    Entrada: query original (str), response normalizada (QueryResponse).
    Saida: dict com chaves timestamp, query, next_step, confidence_score, ranked_response.
    """
    return {
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "query": query,
        "next_step": response.next_step,
        "confidence_score": response.confidence_score,
        "ranked_response": response.ranked_response,
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def consultar(body: QueryRequest, req: Request) -> QueryResponse:
    """Executa o workflow do agente para uma consulta em linguagem natural.

    Entrada: body.query — pergunta do usuario (validada pelo Pydantic).
    Saida: QueryResponse com resposta ranqueada, score, rota e documentos.

    Errors:
      503 — vectordb ausente (guard antes do modelo).
      422 — query vazia ou malformada (Pydantic).
      500 — excecao inesperada no workflow.

    Modo seguro (200-degraded): quando o pre-flight check_lmstudio_health()
    levanta LMStudioUnavailableError/ModeloNaoCarregadoError, ou o invoke
    re-levanta APIConnectionError/httpx.ConnectError, retorna RESPOSTA_PADRAO_SEGURA
    com degraded=True em vez de 503.
    """
    if not req.app.state.vectordb_pronto:
        raise HTTPException(status_code=503, detail=MSG_VECTORDB_AUSENTE)

    try:
        await asyncio.to_thread(check_lmstudio_health)
        estado = await asyncio.to_thread(
            agent_workflow.invoke,
            AgentState(query=body.query),
        )
        response = _normalizar_estado(estado)
    except _ERROS_MODO_SEGURO as exc:
        logger.warning("Modo seguro ativado em /query: %s", exc)
        response = _resposta_segura()

    # Audit write — falha nunca propaga para o cliente (vale tambem para o degradado)
    registro = _construir_registro(body.query, response)
    try:
        await asyncio.to_thread(req.app.state.history_store.append, registro)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao gravar historico: %s", exc)

    return response


@app.get("/history", response_model=list[HistoryEntry])
async def listar_historico(
    req: Request,
    limit: int | None = Query(
        default=None, ge=1, description="Numero maximo de registros a retornar"
    ),
) -> list[HistoryEntry]:
    """Retorna o historico de queries em ordem decrescente de timestamp.

    Entrada: limit (opcional, >= 1) — limita numero de registros.
    Saida: lista de HistoryEntry ordenada por timestamp DESC.

    Errors:
      422 — limit <= 0 (Pydantic/FastAPI Query validation).
      503 — store indisponivel.
    """
    try:
        registros = await asyncio.to_thread(
            req.app.state.history_store.read_all, limit
        )
    except Exception as exc:
        logger.error("Falha ao ler historico: %s", exc)
        raise HTTPException(status_code=503, detail="Historico indisponivel.") from exc
    return [HistoryEntry(**r) for r in registros]


# ---------------------------------------------------------------------------
# Exception handlers
#
# Backstop (decisao D2): /query trata LMStudioUnavailableError/ModeloNaoCarregadoError
# e httpx.ConnectError localmente via modo seguro (200-degraded, ver _ERROS_MODO_SEGURO),
# entao estes handlers NAO disparam mais para /query. Sao mantidos de proposito como
# rede de seguranca caso essas excecoes escapem de outro endpoint futuro — ai valem 503.
# ---------------------------------------------------------------------------


@app.exception_handler(LMStudioUnavailableError)
async def handler_lmstudio(request: Request, exc: LMStudioUnavailableError) -> JSONResponse:
    """Backstop 503 para LMStudioUnavailableError fora de /query (ver nota acima)."""
    logger.error("LMStudio indisponivel: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": MSG_LMSTUDIO_INDISPONIVEL},
    )


@app.exception_handler(httpx.ConnectError)
async def handler_connect_error(request: Request, exc: httpx.ConnectError) -> JSONResponse:
    """Backstop 503 para httpx.ConnectError fora de /query (ver nota acima)."""
    logger.error("Erro de conexao com LMStudio: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": MSG_LMSTUDIO_INDISPONIVEL},
    )


@app.exception_handler(Exception)
async def handler_generico(request: Request, exc: Exception) -> JSONResponse:
    """Captura qualquer excecao nao tratada e retorna HTTP 500 sem expor stack trace."""
    logger.exception("Excecao inesperada no endpoint /query")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor."},
    )

"""
Camada FastAPI do AgenticLog.

Expõe o workflow LangGraph como uma API REST, normalizando entradas e saídas
do agente para um contrato JSON estável. Toda chamada bloqueante ao agente
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
from pydantic import BaseModel, Field, field_validator

from langchain_chroma import Chroma

from agenticlog.agent import AgentState, agent_workflow, inicializar_recursos
from agenticlog.config import DEFAULT_COLLECTION_NAME, DIR_VECTORDB, HISTORY_FILE, HISTORY_MAX_ENTRIES
from agenticlog.health import LMStudioUnavailableError
from agenticlog.history import HistoryStore
from agenticlog.rag import _get_rag_embedding_model

logger = logging.getLogger(__name__)

# Mensagens de erro padronizadas para respostas HTTP
MSG_LMSTUDIO_INDISPONIVEL = (
    "LMStudio indisponível. Inicie o servidor e carregue o modelo."
)
MSG_VECTORDB_AUSENTE = (
    "Base vetorial não encontrada. Execute: python -m agenticlog.rag"
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Corpo da requisição POST /query."""

    query: str = Field(..., min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_and_check(cls, v: object) -> object:
        """Remove espaços e valida comprimento mínimo de 1 após strip."""
        if isinstance(v, str):
            stripped = v.strip()
            if len(stripped) == 0:
                raise ValueError("query não pode ser vazia ou conter apenas espaços")
            return stripped
        return v


class DocumentInfo(BaseModel):
    """Representação serializável de um documento recuperado."""

    page_content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Corpo da resposta bem-sucedida de POST /query."""

    ranked_response: str
    confidence_score: float
    next_step: str
    retrieved_info: list[DocumentInfo]


class HistoryEntry(BaseModel):
    """Entrada individual do histórico de queries."""

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
    """Verifica se o diretório do ChromaDB existe e abre a coleção padrão.

    Saída: nenhuma.
    Raises: RuntimeError com MSG_VECTORDB_AUSENTE se o diretório estiver ausente
            ou a coleção padrão não puder ser aberta.
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
    """Ciclo de vida da aplicação FastAPI.

    Startup:
      - Verifica existência do vectordb; se ausente, loga erro crítico e marca
        app.state.vectordb_pronto = False (servidor sobe mesmo assim — 503 por req).
      - Inicializa singletons (LLM, ChromaDB, embeddings) uma única vez.
    Shutdown: sem operações.
    """
    try:
        _verificar_vectordb()
        inicializar_recursos()
        app.state.vectordb_pronto = True
        logger.info("AgenticLog API iniciada. Recursos inicializados com sucesso.")
    except RuntimeError as exc:
        logger.critical(
            "Vectordb ausente na inicialização: %s. Requisições retornarão 503.", exc
        )
        app.state.vectordb_pronto = False
    except Exception as exc:
        logger.critical(
            "Erro inesperado na inicialização: %s. Requisições retornarão 503.", exc
        )
        app.state.vectordb_pronto = False
    try:
        app.state.history_store = HistoryStore(db_path=HISTORY_FILE, max_entries=HISTORY_MAX_ENTRIES)
        logger.info("HistoryStore inicializado em %s", HISTORY_FILE)
    except Exception as exc:
        logger.critical("Falha ao inicializar HistoryStore: %s. Histórico indisponível.", exc)
        app.state.history_store = None
    yield
    # Shutdown — sem recursos para liberar


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgenticLog API",
    description="REST API para o workflow Agentic RAG de logística.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers de normalização (funções puras — sem I/O)
# ---------------------------------------------------------------------------


def _serializar_documentos(docs: list) -> list[DocumentInfo]:
    """Converte lista de Document LangChain ou dicts em lista de DocumentInfo.

    Entrada: lista de objetos com atributo page_content ou dicts equivalentes.
    Saída: lista de DocumentInfo — nunca None ou null.
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
    Saída: QueryResponse com ranked_response str, confidence_score float >= 0.0,
           retrieved_info list[DocumentInfo].

    Normalizações:
      - dict → AgentState (LangGraph serializa o estado como dict em algumas versões).
      - ranked_response dict {"answer": str} → str (extrai valor).
      - ranked_response dict sem "answer" → json.dumps seguro como fallback.
      - confidence_score None → 0.0.
      - retrieved_info None → [].
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


def _construir_registro(query: str, response: QueryResponse) -> dict:
    """Constrói o dict de registro para persistência no histórico.

    Entrada: query original (str), response normalizada (QueryResponse).
    Saída: dict com chaves timestamp, query, next_step, confidence_score, ranked_response.
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

    Entrada: body.query — pergunta do usuário (validada pelo Pydantic).
    Saída: QueryResponse com resposta ranqueada, score, rota e documentos.

    Errors:
      503 — vectordb ausente ou LMStudio inacessível.
      422 — query vazia ou malformada (Pydantic).
      500 — exceção inesperada no workflow.
    """
    if not req.app.state.vectordb_pronto:
        raise HTTPException(status_code=503, detail=MSG_VECTORDB_AUSENTE)

    estado = await asyncio.to_thread(
        agent_workflow.invoke,
        AgentState(query=body.query),
    )
    response = _normalizar_estado(estado)

    # Audit write — falha nunca propaga para o cliente
    registro = _construir_registro(body.query, response)
    try:
        await asyncio.to_thread(req.app.state.history_store.append, registro)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao gravar histórico: %s", exc)

    return response


@app.get("/history", response_model=list[HistoryEntry])
async def listar_historico(
    req: Request,
    limit: int | None = Query(default=None, ge=1, description="Número máximo de registros a retornar"),
) -> list[HistoryEntry]:
    """Retorna o histórico de queries em ordem decrescente de timestamp.

    Entrada: limit (opcional, >= 1) — limita número de registros.
    Saída: lista de HistoryEntry ordenada por timestamp DESC.

    Errors:
      422 — limit <= 0 (Pydantic/FastAPI Query validation).
      503 — store indisponível.
    """
    try:
        registros = await asyncio.to_thread(
            req.app.state.history_store.read_all, limit
        )
    except Exception as exc:
        logger.error("Falha ao ler histórico: %s", exc)
        raise HTTPException(status_code=503, detail="Histórico indisponível.") from exc
    return [HistoryEntry(**r) for r in registros]


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(LMStudioUnavailableError)
async def handler_lmstudio(request: Request, exc: LMStudioUnavailableError) -> JSONResponse:
    """Captura LMStudioUnavailableError e retorna HTTP 503."""
    logger.error("LMStudio indisponível: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": MSG_LMSTUDIO_INDISPONIVEL},
    )


@app.exception_handler(httpx.ConnectError)
async def handler_connect_error(request: Request, exc: httpx.ConnectError) -> JSONResponse:
    """Captura httpx.ConnectError (falha de rede ao LMStudio) e retorna HTTP 503."""
    logger.error("Erro de conexão com LMStudio: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": MSG_LMSTUDIO_INDISPONIVEL},
    )


@app.exception_handler(Exception)
async def handler_generico(request: Request, exc: Exception) -> JSONResponse:
    """Captura qualquer exceção não tratada e retorna HTTP 500 sem expor stack trace."""
    logger.exception("Exceção inesperada no endpoint /query")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor."},
    )

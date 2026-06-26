# AgenticLog — Rede de caracterização E2E do pipeline RAG
"""
Rede de testes de caracterização (behavioral oracle) para o pipeline RAG.

Dirige APENAS pontos de entrada públicos (`agent_workflow.invoke`,
`adicionar_documento_incrementalmente`) e mocka APENAS os seams de fronteira que
sobrevivem ao refactor da ADR-018 (LLM, embeddings, busca web, bindings de path).
Nenhum helper interno (`_get_retriever`, `_get_vector_db`, `_listar_colecoes`,
`_get_*_embedding_model`, `Chroma`, `JSONLoader`, `SemanticChunker`,
`invalidar_vector_db`, `adicionar_*`) é mockado — eles são o comportamento sob teste.

Cada teste alcança o Chroma real (hnswlib) em disco. No Windows local o `.pyd`
não-assinado do hnswlib é bloqueado pelo Smart App Control e estes testes são
pulados (ver tests/conftest.py); o CI Linux é o gate autoritativo.

Execute (autoritativo): pytest -m integration tests/test_rag_caracterizacao.py -v
"""

import json
from types import SimpleNamespace

import pytest
from langchain_chroma import Chroma
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agenticlog.agent import AgentState, agent_workflow
from agenticlog.config import DEFAULT_COLLECTION_NAME, ROUTING_KEYWORDS_WEB
from agenticlog.rag import adicionar_documento_incrementalmente

# Todo o módulo toca o Chroma real → marcado integration (gate = CI Linux).
pytestmark = pytest.mark.integration

# Dimensão fixa dos vetores determinísticos dos stubs de embedding.
_STUB_EMBEDDING_DIM = 16
# Texto determinístico devolvido pelo stub de LLM em qualquer chain.
_STUB_LLM_RESPOSTA = "resposta deterministica do stub de LLM"


# --------------------------------------------------------------------------- #
# Stubs de fronteira (design §3.3) — substituem só os seams de processo.
# --------------------------------------------------------------------------- #
class _StubEmbedding:
    """Embedding determinístico compartilhado entre ingestão e query (R4).

    Vetores constantes mantêm escrita e leitura no MESMO espaço vetorial; o
    SemanticChunker colapsa breakpoints de distância zero em um único chunk
    válido, e o Chroma ainda devolve os top-k vizinhos.
    """

    def __init__(self, dim: int = _STUB_EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim


class _StubEmbeddingQueFalha:
    """Embedding cujo `embed_documents` levanta (gatilho de falha do AC-5).

    `embed_query` permanece benigno: a falha precisa surgir dentro do bloco
    guardado de ingestão (rag.py L479-528), no chunking, não na leitura.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding boundary failure")

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _STUB_EMBEDDING_DIM


def _stub_embedding() -> _StubEmbedding:
    """Fábrica do embedding determinístico compartilhado (R4)."""
    return _StubEmbedding()


def _stub_embedding_que_falha() -> _StubEmbeddingQueFalha:
    """Fábrica do embedding cujo `embed_documents` levanta (AC-5)."""
    return _StubEmbeddingQueFalha()


def _stub_llm() -> FakeListChatModel:
    """LLM fake componível em `prompt | llm | StrOutputParser()` (R3).

    FakeListChatModel cicla a lista de respostas, então uma única resposta basta
    para qualquer número de invocações de chain.
    """
    return FakeListChatModel(responses=[_STUB_LLM_RESPOSTA])


# --------------------------------------------------------------------------- #
# Fixture function-scoped (design §4) — redireciona paths/seams via monkeypatch.
# --------------------------------------------------------------------------- #
@pytest.fixture
def rag_caracterizacao_env(tmp_path, monkeypatch):
    """Ambiente isolado: store tmp + seams determinísticos, auto-revertidos.

    - redireciona rag.DIR_DOCUMENTS, rag.DIR_VECTORDB E agent.DIR_VECTORDB para o
      mesmo store tmp (R1 — ingestão escreve em rag.DIR_VECTORDB, agente lê de
      agent.DIR_VECTORDB);
    - usa o MESMO stub de embedding em rag._rag_embedding_model e
      agent._embedding_model (R4);
    - injeta o stub de LLM em agent._llm (R3);
    - zera agent._vector_dbs para descartar handles Chroma de testes anteriores (R2);
    - NÃO mocka invalidar_vector_db: ela roda de verdade para o agente reler a
      coleção atualizada (R6).
    """
    docs = tmp_path / "documents"
    docs.mkdir()
    vdb = tmp_path / "vectordb"
    vdb.mkdir()

    monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", docs)
    monkeypatch.setattr("agenticlog.rag.DIR_VECTORDB", vdb)
    monkeypatch.setattr("agenticlog.agent.DIR_VECTORDB", vdb)  # R1 — agente lê aqui

    emb = _stub_embedding()  # compartilhado (R4)
    monkeypatch.setattr("agenticlog.rag._rag_embedding_model", emb)
    monkeypatch.setattr("agenticlog.agent._embedding_model", emb)
    monkeypatch.setattr("agenticlog.agent._llm", _stub_llm())  # R3

    monkeypatch.setattr("agenticlog.agent._vector_dbs", {})  # R2

    return SimpleNamespace(docs=docs, vdb=vdb, emb=emb)


# --------------------------------------------------------------------------- #
# Helpers de driver — só pontos de entrada públicos.
# --------------------------------------------------------------------------- #
def _ingerir_json(nome: str, dados: dict) -> dict:
    """Ingere um dict como JSON via o ponto de entrada público de ingestão."""
    conteudo = json.dumps(dados).encode()
    return adicionar_documento_incrementalmente(
        nome, conteudo, collection_name=DEFAULT_COLLECTION_NAME
    )


def _invoke(query: str) -> dict:
    """Único ponto de entrada de leitura: roda o workflow compilado."""
    return agent_workflow.invoke(AgentState(query=query))


def _colecao_ids(vdb, nome: str, emb) -> list[str]:
    """Lê os ids da coleção via um Chroma real no store tmp (leitura permitida, §5)."""
    db = Chroma(
        persist_directory=str(vdb),
        collection_name=nome,
        embedding_function=emb,
    )
    return sorted(db.get()["ids"])


def _conteudo_recuperado(estado: dict) -> str:
    """Concatena o page_content dos Documents em retrieved_info."""
    return " ".join(doc.page_content for doc in estado["retrieved_info"])


_CHAVES_ESTADO_ESPERADAS = {
    "query",
    "next_step",
    "retrieved_info",
    "ranked_response",
    "confidence_score",
}


# --------------------------------------------------------------------------- #
# AC-1 / CHAR-01 — retrieve feliz E2E.
# --------------------------------------------------------------------------- #
def teste_1_retrieve_feliz_retorna_info_resposta_e_confianca(rag_caracterizacao_env):
    """JSON ingerido + query casando → retrieve com info, resposta e confiança > 0."""
    _ingerir_json("armazem.json", {"armazem": "capacidade 5000 paletes em Sao Paulo"})

    estado = _invoke("qual a capacidade do armazem")

    assert _CHAVES_ESTADO_ESPERADAS.issubset(estado.keys())
    assert estado["next_step"] == "retrieve"
    assert estado["retrieved_info"], "retrieved_info (as 'sources') deveria ser não-vazio"
    assert estado["ranked_response"], "ranked_response deveria estar presente"
    assert estado["confidence_score"] > 0


# --------------------------------------------------------------------------- #
# AC-2 / CHAR-02 — coleção vazia → fallback para 'gerar'.
# --------------------------------------------------------------------------- #
def teste_2_colecao_vazia_faz_fallback_para_gerar(rag_caracterizacao_env):
    """Sem ingestão: retrieval vazio cai para geração direta, sem crash."""
    estado = _invoke("explique o conceito de cross-docking")

    assert _CHAVES_ESTADO_ESPERADAS.issubset(estado.keys())
    assert estado["next_step"] == "gerar"
    assert estado["retrieved_info"] == []
    assert estado["ranked_response"], "uma resposta deveria ter sido produzida"


# --------------------------------------------------------------------------- #
# AC-3 / CHAR-03 — rota web alcança END.
# --------------------------------------------------------------------------- #
def teste_3_rota_web_alcanca_end(rag_caracterizacao_env, monkeypatch):
    """Query com keyword web roteia usar_web; search mockado; grafo chega ao END."""
    mock_search = SimpleNamespace(chamado_com=[])

    def _fake_run(query: str) -> str:
        mock_search.chamado_com.append(query)
        return "resultado web simulado"

    # agent.search é um seam de fronteira (wrapper DuckDuckGo) — ALLOWED.
    monkeypatch.setattr("agenticlog.agent.search", SimpleNamespace(run=_fake_run))

    query_web = f"{ROUTING_KEYWORDS_WEB[0]} sobre logistica"
    estado = _invoke(query_web)

    assert estado["next_step"] == "usar_web"
    assert mock_search.chamado_com == [query_web], "search.run deveria ter sido chamado"
    assert estado["ranked_response"], "o ramo web deveria ter produzido uma resposta"
    assert estado["confidence_score"] == 0.0


# --------------------------------------------------------------------------- #
# AC-4 / CHAR-04 — doc incremental em base populada é recuperável.
# --------------------------------------------------------------------------- #
def teste_4_doc_incremental_em_base_populada_e_recuperavel(rag_caracterizacao_env):
    """Após popular a base, um NOVO doc adicionado no mesmo teste fica consultável."""
    seed = _ingerir_json("rota.json", {"rota": "SP-RJ km 450"})
    assert seed["status"] == "adicionado"

    novo = _ingerir_json(
        "fornecedor.json", {"fornecedor": "Beta SA Curitiba contrato C-999"}
    )
    assert novo["status"] == "adicionado"

    # invalidar_vector_db() rodou de verdade na ingestão (R6) → cache do agente limpo.
    estado = _invoke("informacoes do fornecedor Beta SA")

    assert estado["retrieved_info"], "o novo doc deveria estar recuperável"
    assert "Beta SA" in _conteudo_recuperado(estado)


# --------------------------------------------------------------------------- #
# AC-5 / CHAR-05 — upsert que falha no embedding reverte disco E coleção.
# --------------------------------------------------------------------------- #
def teste_5_upsert_falho_no_embedding_reverte_disco_e_colecao(
    rag_caracterizacao_env, monkeypatch
):
    """Re-upsert com embed_documents levantando → disco e coleção inalterados."""
    env = rag_caracterizacao_env
    _ingerir_json("contrato.json", {"contrato": "C-001 valor 50000"})

    arquivo = env.docs / "contrato.json"
    bytes_antes = arquivo.read_bytes()
    ids_antes = _colecao_ids(env.vdb, DEFAULT_COLLECTION_NAME, env.emb)
    assert ids_antes, "a ingestão semente deveria ter populado a coleção"

    # Gatilho de falha (locked decision 1): SÓ o embed_documents do boundary de
    # ingestão levanta — não dirs read-only, não patch em Chroma.
    monkeypatch.setattr(
        "agenticlog.rag._rag_embedding_model", _stub_embedding_que_falha()
    )

    conteudo_novo = json.dumps({"contrato": "C-001 valor 99999 ALTERADO"}).encode()
    with pytest.raises(RuntimeError, match="embedding boundary failure"):
        adicionar_documento_incrementalmente(
            "contrato.json", conteudo_novo, collection_name=DEFAULT_COLLECTION_NAME
        )

    # Rollback (PRs #42/#43): disco restaurado E coleção intacta.
    assert arquivo.read_bytes() == bytes_antes, "arquivo em disco deveria ser inalterado"
    ids_depois = _colecao_ids(env.vdb, DEFAULT_COLLECTION_NAME, env.emb)
    assert ids_depois == ids_antes, "ids da coleção Chroma deveriam ser inalterados"

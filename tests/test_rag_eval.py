# AgenticLog - Testes do harness de avaliação de RAG (scripts/rag_eval.py)
"""Testes unitários e de integração do harness RAG.

Todos os embeddings e o LLM são STUBADOS — nenhuma chamada real é feita.
Marcados com @pytest.mark.rag_eval para serem EXCLUÍDOS de
`pytest --cov=agenticlog` (o harness vive fora do pacote agenticlog).
Convenção de nomes: `teste_N_` (CLAUDE.md).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.rag_eval

# Torna scripts/ importável (fora do pacote agenticlog).
_RAIZ = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_RAIZ / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import rag_eval  # noqa: E402
import rag_eval_metrics as metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class StubEmbedding:
    """Embedding determinístico: mapeia textos conhecidos para vetores fixos."""

    def __init__(self, mapa: dict[str, list[float]], padrao: list[float] | None = None) -> None:
        self._mapa = mapa
        self._padrao = padrao or [0.0, 0.0, 1.0]

    def embed_query(self, texto: str) -> list[float]:
        return self._mapa.get(texto, self._padrao)


class StubDoc:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class StubConfig:
    RAG_EVAL_MIN_HIT_RATE = 0.7
    RAG_EVAL_MIN_MRR = 0.6
    RAG_EVAL_MATCH_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# T7 — métricas puras
# --------------------------------------------------------------------------- #
def teste_1_normalizar_contexto_ref_variantes() -> None:
    assert metrics._normalizar_contexto_ref({"contexto_ref": "a"}) == ["a"]
    assert metrics._normalizar_contexto_ref({"contexto_ref": ["a", "b"]}) == ["a", "b"]
    assert metrics._normalizar_contexto_ref({}) is None
    assert metrics._normalizar_contexto_ref({"contexto_ref": ["", "  "]}) is None


def teste_2_metrica_retrieval_acima_do_threshold() -> None:
    # chunk[0] casa com a ref (vetor idêntico -> cosseno 1.0); chunk[1] não.
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "chunk_hit": [1.0, 0.0, 0.0],
        "chunk_miss": [0.0, 1.0, 0.0],
    })
    r = metrics._metrica_retrieval(emb, ["chunk_hit", "chunk_miss"], ["ref"], 0.6)
    assert r["hit"] == 1.0
    assert r["mrr"] == 1.0
    assert r["precision"] == 0.5
    assert r["recall"] == 1.0


def teste_3_metrica_retrieval_hit_no_segundo_chunk() -> None:
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "miss": [0.0, 1.0, 0.0],
        "hit": [1.0, 0.0, 0.0],
    })
    r = metrics._metrica_retrieval(emb, ["miss", "hit"], ["ref"], 0.6)
    assert r["hit"] == 1.0
    assert r["mrr"] == 0.5  # primeiro hit no rank 2


def teste_4_metrica_retrieval_abaixo_do_threshold() -> None:
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "miss": [0.0, 1.0, 0.0],
    })
    r = metrics._metrica_retrieval(emb, ["miss"], ["ref"], 0.6)
    assert r["hit"] == 0.0
    assert r["mrr"] == 0.0
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0


def teste_5_answer_correctness_numerico_e_ausente() -> None:
    emb = StubEmbedding({"resp": [1.0, 0.0], "ref": [1.0, 0.0]})
    assert metrics._answer_correctness(emb, "resp", "ref") == 1.0
    assert metrics._answer_correctness(emb, "resp", None) == metrics.SENTINEL_RESP_AUSENTE


# --------------------------------------------------------------------------- #
# T7 — loader (GOLD-04)
# --------------------------------------------------------------------------- #
def teste_6_carregar_golden_descarta_sem_pergunta(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "g.json"
    p.write_text(json.dumps([
        {"pergunta": "ok", "resposta_ref": "r"},
        {"resposta_ref": "sem pergunta"},
        {"pergunta": "", "resposta_ref": "vazia"},
    ]), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="rag_eval"):
        itens = rag_eval._carregar_golden(p)
    assert len(itens) == 1
    assert itens[0]["pergunta"] == "ok"
    assert any("descartada" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# T7 — _avaliar_pergunta (GOLD-02/03, RETR)
# --------------------------------------------------------------------------- #
def _harness_stub(emb: StubEmbedding, docs: list[StubDoc]) -> dict:
    class _WF:
        @staticmethod
        def invoke(_state):
            return {"ranked_response": "resp"}

    return {
        "config": StubConfig,
        "AgentState": lambda query: {"query": query},
        "_get_retriever": lambda q: docs,
        "agent_workflow": _WF,
        "_get_rag_embedding_model": lambda: emb,
    }


def teste_7_avaliar_pergunta_com_refs_numerico() -> None:
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "chunk": [1.0, 0.0, 0.0],
        "resp": [1.0, 0.0, 0.0],
        "rr": [1.0, 0.0, 0.0],
    })
    h = _harness_stub(emb, [StubDoc("chunk")])
    item = {"pergunta": "q", "resposta_ref": "rr", "contexto_ref": "ref"}
    linha = rag_eval._avaliar_pergunta(h, None, "", item, k=3)
    assert linha["hit"] == 1.0
    assert linha["context_recall"] == 1.0
    assert isinstance(linha["answer_correctness"], float)
    assert linha["tem_contexto_ref"] is True
    assert "judge" not in linha  # client None -> sem juiz


def teste_8_avaliar_pergunta_sem_contexto_ref_sentinela() -> None:
    emb = StubEmbedding({"resp": [1.0, 0.0], "rr": [1.0, 0.0]})
    h = _harness_stub(emb, [StubDoc("chunk")])
    item = {"pergunta": "q", "resposta_ref": "rr"}  # sem contexto_ref
    linha = rag_eval._avaliar_pergunta(h, None, "", item, k=3)
    assert linha["context_recall"] == metrics.SENTINEL_CTX_AUSENTE
    assert linha["hit"] is None
    assert linha["tem_contexto_ref"] is False


def teste_9_avaliar_pergunta_sem_resposta_ref_sentinela() -> None:
    emb = StubEmbedding({"ref": [1.0, 0.0], "chunk": [1.0, 0.0]})
    h = _harness_stub(emb, [StubDoc("chunk")])
    item = {"pergunta": "q", "contexto_ref": "ref"}  # sem resposta_ref
    linha = rag_eval._avaliar_pergunta(h, None, "", item, k=3)
    assert linha["answer_correctness"] == metrics.SENTINEL_RESP_AUSENTE


def teste_10_ranked_response_dict_coercion() -> None:
    emb = StubEmbedding({})

    class _WF:
        @staticmethod
        def invoke(_state):
            return {"ranked_response": {"answer": "texto"}}

    h = _harness_stub(emb, [StubDoc("chunk")])
    h["agent_workflow"] = _WF
    linha = rag_eval._avaliar_pergunta(h, None, "", {"pergunta": "q"}, k=3)
    assert linha["resposta"] == "texto"


# --------------------------------------------------------------------------- #
# T7 — agregação + gate (RETR-03, GATE-01)
# --------------------------------------------------------------------------- #
def teste_11_agregar_ignora_entradas_sem_contexto_ref() -> None:
    linhas = [
        {"hit": 1.0, "mrr": 1.0, "context_precision": 1.0, "context_recall": 1.0,
         "answer_correctness": 0.8, "tem_contexto_ref": True},
        {"hit": None, "mrr": None, "context_precision": None,
         "context_recall": metrics.SENTINEL_CTX_AUSENTE,
         "answer_correctness": 0.6, "tem_contexto_ref": False},
    ]
    ag = rag_eval._agregar(linhas, judge_skipped=True, judge_motivo="sem LLM")
    assert ag["retrieval"]["n_entradas_gated"] == 1
    assert ag["retrieval"]["hit_rate"] == 1.0
    assert ag["answer_correctness"] == 0.7  # média de 0.8 e 0.6
    assert ag["judge"]["status"] == "skipped"


def teste_12_portao_passa_e_reprova() -> None:
    ag_pass = {"retrieval": {"hit_rate": 0.8, "mrr": 0.7}}
    ag_fail = {"retrieval": {"hit_rate": 0.5, "mrr": 0.7}}
    ag_none = {"retrieval": {"hit_rate": None, "mrr": None}}
    assert rag_eval.portao(ag_pass, StubConfig) is True
    assert rag_eval.portao(ag_fail, StubConfig) is False
    assert rag_eval.portao(ag_none, StubConfig) is False


# --------------------------------------------------------------------------- #
# T7 — failure modes via main() (GATE-03, GATE-04)
# --------------------------------------------------------------------------- #
def teste_13_golden_ausente_exit_nao_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag_eval, "_bootstrap", lambda: _harness_stub(StubEmbedding({}), [StubDoc("x")])
    )
    monkeypatch.setattr(rag_eval, "_criar_llm", lambda cfg: (None, "sem LLM"))
    out = tmp_path / "r.json"
    code = rag_eval.main(["--out", str(out), "--golden", str(tmp_path / "nao_existe.json")])
    assert code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "error"


def teste_14_golden_vazio_exit_nao_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag_eval, "_bootstrap", lambda: _harness_stub(StubEmbedding({}), [StubDoc("x")])
    )
    monkeypatch.setattr(rag_eval, "_criar_llm", lambda cfg: (None, "sem LLM"))
    g = tmp_path / "g.json"
    g.write_text("[]", encoding="utf-8")
    out = tmp_path / "r.json"
    code = rag_eval.main(["--out", str(out), "--golden", str(g)])
    assert code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "error"


def teste_15_indice_vazio_exit_nao_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h_vazio = _harness_stub(StubEmbedding({}), [])  # retriever retorna [] -> índice vazio
    monkeypatch.setattr(rag_eval, "_bootstrap", lambda: h_vazio)
    out = tmp_path / "r.json"
    code = rag_eval.main(["--out", str(out), "--golden", str(tmp_path / "x.json")])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["severidade"] == "alta"


def teste_16_checar_indice_ok_retorna_none() -> None:
    h = _harness_stub(StubEmbedding({}), [StubDoc("chunk")])
    assert rag_eval._checar_indice(h) is None


# --------------------------------------------------------------------------- #
# T8 — integração: harness completo, juiz UNAVAILABLE, retrieval acima -> pass
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def teste_17_harness_completo_juiz_skipped_gate_passa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "chunk": [1.0, 0.0, 0.0],
        "resp": [1.0, 0.0, 0.0],
        "rr": [1.0, 0.0, 0.0],
    })
    h = _harness_stub(emb, [StubDoc("chunk")])
    monkeypatch.setattr(rag_eval, "_bootstrap", lambda: h)
    # Juiz INDISPONÍVEL (LMStudio ausente).
    monkeypatch.setattr(rag_eval, "_criar_llm", lambda cfg: (None, "LMStudio inacessível"))

    g = tmp_path / "g.json"
    g.write_text(json.dumps([
        {"pergunta": "q1", "resposta_ref": "rr", "contexto_ref": "ref"},
        {"pergunta": "q2", "resposta_ref": "rr", "contexto_ref": "ref"},
    ]), encoding="utf-8")
    out = tmp_path / "r.json"

    code = rag_eval.main(["--out", str(out), "--golden", str(g), "--k", "3", "--gate"])
    assert code == 0  # retrieval acima do threshold -> gate passa apesar do juiz pulado

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["agregados"]["judge"]["status"] == "skipped"
    assert isinstance(payload["agregados"]["retrieval"]["hit_rate"], (int, float))
    assert payload["gate_passou"] is True


@pytest.mark.integration
def teste_18_gate_reprova_abaixo_do_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = StubEmbedding({
        "ref": [1.0, 0.0, 0.0],
        "miss": [0.0, 1.0, 0.0],  # nunca casa
        "resp": [1.0, 0.0, 0.0],
        "rr": [1.0, 0.0, 0.0],
    })
    h = _harness_stub(emb, [StubDoc("miss")])
    monkeypatch.setattr(rag_eval, "_bootstrap", lambda: h)
    monkeypatch.setattr(rag_eval, "_criar_llm", lambda cfg: (None, "sem LLM"))
    g = tmp_path / "g.json"
    g.write_text(json.dumps([{"pergunta": "q", "resposta_ref": "rr", "contexto_ref": "ref"}]),
                 encoding="utf-8")
    out = tmp_path / "r.json"
    code = rag_eval.main(["--out", str(out), "--golden", str(g), "--gate"])
    assert code == 1


def teste_19_golden_curado_cobre_3_categorias() -> None:
    """GOLD-05: o golden versionado cobre >= 3 categorias nomeadas."""
    golden = _RAIZ / "evals" / "rag_golden.json"
    dados = json.loads(golden.read_text(encoding="utf-8"))
    categorias = {d.get("categoria") for d in dados if d.get("categoria")}
    nomeadas = {
        "controle de estoque",
        "processamento de pedidos",
        "definição de operador logístico",
    }
    assert nomeadas.issubset(categorias)
    assert len(categorias) >= 3
    assert 8 <= len(dados) <= 10
    # ao menos uma entrada sem contexto_ref (GOLD-03)
    assert any("contexto_ref" not in d for d in dados)

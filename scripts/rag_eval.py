#!/usr/bin/env python
"""Harness de avaliação de RAG (retrieval + generation) com gate de qualidade.

Computa métricas de Retrieval (Hit Rate, MRR, Context Precision, Context Recall)
SEM LLM — via cosseno de embeddings entre chunks recuperados e o `contexto_ref` do
golden — e Answer Correctness (cosseno resposta↔resposta_ref). Métricas de juiz
(Faithfulness, Answer Relevancy, Context Utilization) são report-only e ficam como
``{"status": "skipped"}`` quando o LMStudio está indisponível (ex.: CI sem LLM).

Degrada com elegância: sem `src/agenticlog/` grava ``{"status": "skipped"}`` (exit 0).
Falhas estruturais (golden ausente/vazio quando ``--golden`` foi pedido, índice vazio)
gravam ``{"status": "error"}`` e retornam exit ≠ 0 — sem fallback sintético no CI.

Uso (CI):
    python scripts/rag_eval.py --golden evals/rag_golden.json \
        --out rag_eval_results.json --k 3 --gate

Uso (audit skill, modo sintético, sem golden):
    python scripts/rag_eval.py --out results.json --n 15 --k 5

Flags:
    --golden  golden set JSON. Quando presente e ausente/vazio -> erro (exit ≠ 0).
    --gate    aplica o gate de qualidade: exit ≠ 0 se hit_rate < RAG_EVAL_MIN_HIT_RATE
              ou mrr < RAG_EVAL_MIN_MRR. Juiz pulado, sozinho, nunca reprova.
    --out     caminho do JSON de saída (obrigatório).
    --k       top-k chunks a avaliar (limitado por RETRIEVAL_K_TOTAL).
    --n       nº de perguntas sintéticas quando --golden é omitido (audit skill).

Schema do golden set (``evals/rag_golden.json``), array de objetos:
    {
      "pergunta": str,            # obrigatório; sem ele a entrada é descartada (warning)
      "resposta_ref": str,        # p/ Answer Correctness; ausente -> "resposta_ref ausente"
      "contexto_ref": str|list,   # p/ Context Recall + gate; ausente -> "contexto_ref ausente"
      "categoria": str            # opcional; informativo (cobertura de categorias)
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

# Permite importar rag_eval_metrics quer rodando o script, quer importando-o nos testes.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rag_eval_metrics import (  # noqa: E402
    SENTINEL_CTX_AUSENTE,
    _answer_correctness,
    _cosine,
    _metrica_retrieval,
    _normalizar_contexto_ref,
)

logger = logging.getLogger("rag_eval")


# --------------------------------------------------------------------------- #
# Setup do projeto (descobre src/ e importa agenticlog)
# --------------------------------------------------------------------------- #
def _achar_raiz_projeto(start: Path) -> Path | None:
    """Sobe diretórios procurando src/agenticlog/."""
    for p in [start, *start.parents]:
        if (p / "src" / "agenticlog").is_dir():
            return p
    return None


def _bootstrap() -> dict[str, Any]:
    """Importa os módulos do projeto. Retorna dict com handles ou {'erro': ...}."""
    raiz = _achar_raiz_projeto(Path.cwd()) or _achar_raiz_projeto(Path(__file__).resolve())
    if raiz is None:
        return {"erro": "não encontrei src/agenticlog/ — rode a partir da raiz do projeto"}
    src = str(raiz / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from agenticlog import config  # type: ignore[import-not-found]
        from agenticlog.agent import (  # type: ignore[import-not-found]
            AgentState,
            _get_retriever,
            agent_workflow,
        )
        from agenticlog.rag import _get_rag_embedding_model  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 — import nativo (chromadb/hnswlib) pode falhar
        return {"erro": f"falha ao importar agenticlog ({type(exc).__name__}): {exc}"}
    return {
        "raiz": raiz,
        "config": config,
        "AgentState": AgentState,
        "_get_retriever": _get_retriever,
        "agent_workflow": agent_workflow,
        "_get_rag_embedding_model": _get_rag_embedding_model,
    }


# --------------------------------------------------------------------------- #
# Cliente LLM (juiz) — OpenAI-compat (LMStudio)
# --------------------------------------------------------------------------- #
def _criar_llm(config: Any) -> tuple[Any, str]:
    """Retorna (client, model) ou (None, motivo)."""
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return None, f"pacote openai indisponível: {exc}"
    try:
        client = OpenAI(base_url=config.LLM_API_BASE, api_key=config.LLM_API_KEY)
        client.models.list()  # ping
    except Exception as exc:  # noqa: BLE001
        return None, f"LMStudio inacessível em {config.LLM_API_BASE}: {exc}"
    return client, config.LLM_MODEL


def _juiz_json(client: Any, model: str, prompt: str) -> dict:
    """Pede ao LLM um veredito JSON {'score': float, 'motivo': str}. Robusto a ruído."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você é um avaliador rigoroso. Responda SÓ JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        texto = (resp.choices[0].message.content or "").strip()
        ini, fim = texto.find("{"), texto.rfind("}")
        if ini >= 0 and fim > ini:
            obj = json.loads(texto[ini : fim + 1])
            return {"score": float(obj.get("score", 0)), "motivo": str(obj.get("motivo", ""))}
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "motivo": f"juiz falhou: {exc}"}
    return {"score": 0.0, "motivo": "sem JSON na resposta"}


# --------------------------------------------------------------------------- #
# Conjunto de perguntas (golden ou sintético)
# --------------------------------------------------------------------------- #
def _carregar_golden(caminho: Path) -> list[dict]:
    """Carrega o golden set; descarta entradas sem `pergunta` (WARNING por descarte)."""
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    validos: list[dict] = []
    for i, d in enumerate(dados):
        if isinstance(d, dict) and d.get("pergunta"):
            validos.append(d)
        else:
            logger.warning("golden: entrada %d descartada (sem 'pergunta'): %r", i, d)
    return validos


def _gerar_sinteticas(h: dict, client: Any, model: str, n: int) -> list[dict]:
    """Amostra n chunks do retriever e gera 1 pergunta por chunk (audit skill only)."""
    sementes = ["resumo", "processo", "definição", "objetivo", "requisito", "exemplo"]
    vistos: dict[str, str] = {}
    for s in sementes:
        try:
            for d in h["_get_retriever"](s):
                vistos.setdefault(d.page_content[:400], d.page_content)
        except Exception:  # noqa: BLE001
            continue
        if len(vistos) >= n:
            break
    chunks = list(vistos.values())[:n]
    perguntas = []
    for ch in chunks:
        v = _juiz_json(
            client, model,
            "A partir DESTE trecho, gere UMA pergunta objetiva respondível só com ele. "
            'Responda JSON {"score": 1, "motivo": "<a pergunta>"}.\n\nTrecho:\n' + ch[:1500],
        )
        q = v["motivo"].strip()
        if q:
            perguntas.append({"pergunta": q, "chunk_fonte": ch})
    return perguntas


# --------------------------------------------------------------------------- #
# Índice (GATE-04)
# --------------------------------------------------------------------------- #
def _checar_indice(h: dict) -> dict | None:
    """Sentinela de erro quando o índice está vazio; None se há chunks recuperáveis.

    Estratégia (sem depender de helpers internos do agent): faz um retrieve com uma
    consulta-sonda. Depende de `_get_retriever` usar search_type="similarity" (sem
    score_threshold), que sempre retorna resultados quando collection.count() > 0; se
    o retriever mudar para "similarity_score_threshold", trocar por collection.count().
    """
    try:
        docs = h["_get_retriever"]("teste")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "motivo": f"falha ao acessar o índice: {exc}",
            "severidade": "alta",
        }
    if not docs:
        return {
            "status": "error",
            "motivo": "índice vazio (collection.count()==0)",
            "severidade": "alta",
        }
    return None


# --------------------------------------------------------------------------- #
# Avaliação por pergunta
# --------------------------------------------------------------------------- #
def _coagir_resposta(estado: Any) -> str:
    """Extrai `ranked_response` do estado e normaliza para texto (dict/str/other)."""
    resposta = (
        estado.get("ranked_response", "")
        if isinstance(estado, dict)
        else getattr(estado, "ranked_response", "")
    )
    # ranked_response pode vir como dict {"answer": str} (formato de possible_responses)
    # além de str — normaliza para texto antes de fatiar/avaliar.
    if isinstance(resposta, dict):
        resposta = resposta.get("answer", "")
    elif not isinstance(resposta, str):
        resposta = str(resposta)
    return resposta


def _bloco_juiz(client: Any, model: str, q: str, contextos: list[str], resposta: str) -> dict:
    """Bloco de métricas de juiz (report-only); só chamado quando há LLM."""
    ctx_txt = "\n".join(contextos)[:2500]
    faith = _juiz_json(
        client, model,
        f'Contexto:\n{ctx_txt}\n\nResposta:\n{resposta[:1200]}\n\n'
        'Que fração das afirmações da resposta é sustentada pelo contexto? '
        'Responda JSON {"score": 0..1, "motivo": "..."}.',
    )["score"] if resposta else 0.0
    util = _juiz_json(
        client, model,
        f'Pergunta: {q}\nContexto:\n{ctx_txt}\nResposta:\n{resposta[:1200]}\n\n'
        'A resposta aproveitou o contexto relevante disponível? '
        'Responda JSON {"score": 0..1, "motivo": "..."}.',
    )["score"] if resposta else 0.0
    return {
        "faithfulness": round(faith, 3),
        "context_utilization": round(util, 3),
    }


def _avaliar_pergunta(h: dict, client: Any, model: str, item: dict, k: int) -> dict:
    """Avalia UMA entrada: retrieval embedding-only + Answer Correctness + juiz opcional."""
    q = item["pergunta"]
    # _get_rag_embedding_model() é um singleton global (rag.py): o modelo é
    # carregado uma vez e reaproveitado em todas as perguntas — chamada barata.
    emb = h["_get_rag_embedding_model"]()
    try:
        docs = h["_get_retriever"](q)[:k]
    except Exception as exc:  # noqa: BLE001
        return {"pergunta": q, "erro_retrieval": str(exc)}
    contextos = [d.page_content for d in docs]

    refs = _normalizar_contexto_ref(item)
    if refs is not None:
        retr = _metrica_retrieval(emb, contextos, refs, h["config"].RAG_EVAL_MATCH_THRESHOLD)
        context_recall: float | str = retr["recall"]
    else:
        retr = {"hit": None, "mrr": None, "precision": None, "recall": None}
        context_recall = SENTINEL_CTX_AUSENTE

    # Geração (resposta do pipeline real, coagida a texto).
    try:
        estado = h["agent_workflow"].invoke(h["AgentState"](query=q))
        resposta = _coagir_resposta(estado)
    except Exception as exc:  # noqa: BLE001
        resposta = ""
        gen_erro = str(exc)
    else:
        gen_erro = ""

    answer_correctness = _answer_correctness(emb, resposta, item.get("resposta_ref"))

    # Answer Relevancy (report-only): cosseno(pergunta, resposta).
    relevancy = 0.0
    if resposta:
        try:
            relevancy = _cosine(emb.embed_query(q), emb.embed_query(resposta))
        except Exception:  # noqa: BLE001
            relevancy = 0.0

    linha: dict[str, Any] = {
        "pergunta": q,
        "resposta": resposta[:300],
        "context_precision": retr["precision"],
        "hit": retr["hit"],
        "mrr": retr["mrr"],
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
        "answer_relevancy": round(relevancy, 3),
        "tem_contexto_ref": refs is not None,
        "gen_erro": gen_erro,
        "n_chunks": len(contextos),
    }
    if client is not None:
        linha["judge"] = _bloco_juiz(client, model, q, contextos, resposta)
    return linha


# --------------------------------------------------------------------------- #
# Agregação e gate
# --------------------------------------------------------------------------- #
def _media_numerica(linhas: list[dict], campo: str) -> float | None:
    """Média ignorando ausências e sentinelas textuais (str)."""
    vals = [
        x[campo] for x in linhas
        if campo in x and isinstance(x[campo], (int, float))
    ]
    return round(statistics.mean(vals), 3) if vals else None


def _agregar(linhas: list[dict], judge_skipped: bool, judge_motivo: str = "") -> dict:
    """Agrega métricas; gated metrics ignoram entradas sem `contexto_ref` (RETR-03)."""
    gated = [x for x in linhas if x.get("tem_contexto_ref")]
    retrieval = {
        "hit_rate": _media_numerica(gated, "hit"),
        "mrr": _media_numerica(gated, "mrr"),
        "context_precision": _media_numerica(gated, "context_precision"),
        "context_recall": _media_numerica(gated, "context_recall"),
        "n_entradas_gated": len(gated),
    }
    agregados: dict[str, Any] = {
        "retrieval": retrieval,
        "answer_correctness": _media_numerica(linhas, "answer_correctness"),
        "answer_relevancy": _media_numerica(linhas, "answer_relevancy"),
    }
    if judge_skipped:
        agregados["judge"] = {"status": "skipped", "motivo": judge_motivo}
    else:
        agregados["judge"] = {
            "status": "ok",
            "faithfulness": _media_numerica(
                [x["judge"] for x in linhas if "judge" in x], "faithfulness"
            ),
            "context_utilization": _media_numerica(
                [x["judge"] for x in linhas if "judge" in x], "context_utilization"
            ),
        }
    return agregados


def portao(agregados: dict, config: Any) -> bool:
    """True se hit_rate >= MIN_HIT_RATE e mrr >= MIN_MRR (gate de qualidade)."""
    retr = agregados.get("retrieval", {})
    hit = retr.get("hit_rate")
    mrr = retr.get("mrr")
    if hit is None or mrr is None:
        return False
    return hit >= config.RAG_EVAL_MIN_HIT_RATE and mrr >= config.RAG_EVAL_MIN_MRR


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Avaliação de RAG (retrieval + generation).")
    ap.add_argument("--out", required=True, help="caminho do JSON de saída")
    ap.add_argument("--golden", default=None, help="golden set JSON (CI: obrigatório)")
    ap.add_argument("--n", type=int, default=15, help="nº de perguntas sintéticas sem golden")
    ap.add_argument("--k", type=int, default=5, help="top-k chunks a avaliar")
    ap.add_argument("--gate", action="store_true", help="aplica o gate de qualidade (exit ≠ 0)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def gravar(payload: dict, code: int = 0) -> int:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload.get("status"), "out": str(out)}, ensure_ascii=False))  # noqa: T201
        return code

    h = _bootstrap()
    if "erro" in h:
        # Sem projeto: comportamento do audit skill (skipped, exit 0).
        return gravar({"status": "skipped", "motivo": h["erro"],
                       "dica": "rode a Fase 1 (estática) + plano de evals no relatório."})

    # GATE-04: índice vazio -> erro de alta severidade, exit ≠ 0.
    erro_indice = _checar_indice(h)
    if erro_indice is not None:
        return gravar(erro_indice, code=1)

    client, model_ou_motivo = _criar_llm(h["config"])
    judge_skipped = client is None
    judge_motivo = model_ou_motivo if judge_skipped else ""
    model = "" if judge_skipped else model_ou_motivo

    # Conjunto de perguntas.
    if args.golden:
        caminho = Path(args.golden)
        # GATE-03: --golden pedido mas ausente/vazio -> erro, exit ≠ 0 (sem fallback sintético).
        if not caminho.exists():
            return gravar({"status": "error", "motivo": f"golden ausente: {args.golden}"}, code=1)
        itens = _carregar_golden(caminho)
        if not itens:
            return gravar({"status": "error", "motivo": f"golden vazio: {args.golden}"}, code=1)
        origem = f"golden:{args.golden}"
    else:
        if judge_skipped:
            return gravar({"status": "skipped", "motivo": model_ou_motivo,
                           "dica": "suba o LMStudio ou forneça --golden."})
        itens = _gerar_sinteticas(h, client, model, args.n)
        origem = "sintético (baseline — sem golden)"
        if not itens:
            return gravar({"status": "skipped", "motivo": "sem perguntas (corpus vazio?)",
                           "dica": "confirme que o vector DB está populado."})

    linhas = [_avaliar_pergunta(h, client, model, it, args.k) for it in itens]
    agregados = _agregar(linhas, judge_skipped, judge_motivo)

    # GATE: com --gate mas nenhuma entrada gated (sem contexto_ref), não há o que
    # medir — erro explícito em vez de um "ok" enganoso com exit 1 (review HIGH).
    if args.gate and agregados["retrieval"]["n_entradas_gated"] == 0:
        return gravar({
            "status": "error",
            "motivo": "gate solicitado mas nenhuma entrada do golden tem contexto_ref",
            "severidade": "alta",
        }, code=1)

    gate_passou = portao(agregados, h["config"]) if args.gate else None
    code = 0 if (gate_passou is None or gate_passou) else 1
    return gravar({
        "status": "ok",
        "origem_perguntas": origem,
        "n_perguntas": len(linhas),
        "top_k": args.k,
        "gate_aplicado": args.gate,
        "gate_passou": gate_passou,
        "agregados": agregados,
        "detalhe": linhas,
    }, code=code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

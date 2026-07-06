# Acceptance tests — RAG Ingestion Fase 3b (ADR-018 Fase 3b)
"""
Verifica os critérios de aceitação da Fase 3b:
  AC1 — `agenticlog.ingestion.store` expõe as 4 primitivas + `add_documents_com_rollback`.
  AC2 — `agenticlog.ingestion.orchestrator` expõe os 5 orquestradores.
  AC3 — identidade `is` dos 4 símbolos de store; delegação (não-identidade) dos 5 wrappers.
  AC4 — seam-binding: o wrapper injeta `rag.DIR_VECTORDB` (patchado) no orquestrador.
  AC5 — `cria_vectordb` não referencia `DirectoryLoader`.
  AC9 — `import store`/`orchestrator` em interpretador frio saem 0 (sem ciclo).

Exercita a feature de fora — via imports de nível de módulo e o namespace público —,
espelhando o padrão de `test_rag_ingestion_fase3a.py` e `test_rag_shared_observability.py`.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.ingestion.orchestrator as orchestrator  # noqa: E402
import agenticlog.ingestion.store as store  # noqa: E402
import agenticlog.rag as rag  # noqa: E402


# ---------------------------------------------------------------------------
# AC1 — store expõe as 4 primitivas + add_documents_com_rollback
# ---------------------------------------------------------------------------
_STORE_SYMBOLS = [
    "_backup_arquivo",
    "_reverter_disco",
    "_outras_colecoes_existem",
    "_resetar_colecao",
    "add_documents_com_rollback",
]


@pytest.mark.parametrize("symbol", _STORE_SYMBOLS)
def test_ac1_store_expoe_primitivas(symbol: str) -> None:
    """AC1: cada primitiva de persistência/atomicidade existe em ingestion.store."""
    assert hasattr(store, symbol), f"agenticlog.ingestion.store.{symbol} ausente"
    assert callable(getattr(store, symbol))


# ---------------------------------------------------------------------------
# AC2 — orchestrator expõe os 5 orquestradores
# ---------------------------------------------------------------------------
_ORCH_SYMBOLS = [
    "cria_vectordb",
    "adicionar_documento_incrementalmente",
    "adicionar_pdf_incrementalmente",
    "ingerir_incrementalmente",
    "reconstruir_vectordb",
]


@pytest.mark.parametrize("symbol", _ORCH_SYMBOLS)
def test_ac2_orchestrator_expoe_orquestradores(symbol: str) -> None:
    """AC2: cada orquestrador existe em ingestion.orchestrator."""
    assert hasattr(orchestrator, symbol), f"agenticlog.ingestion.orchestrator.{symbol} ausente"
    assert callable(getattr(orchestrator, symbol))


# ---------------------------------------------------------------------------
# AC3 — identidade `is` para store; delegação (não-identidade) para wrappers
# ---------------------------------------------------------------------------
_STORE_SHIM_SYMBOLS = [
    "_backup_arquivo",
    "_reverter_disco",
    "_outras_colecoes_existem",
    "_resetar_colecao",
]


@pytest.mark.parametrize("symbol", _STORE_SHIM_SYMBOLS)
def test_ac3_identidade_is_shim_de_store(symbol: str) -> None:
    """AC3: agenticlog.rag.X is agenticlog.ingestion.store.X (shim identity-preserving)."""
    assert getattr(rag, symbol) is getattr(store, symbol), (
        f"Identidade quebrada: rag.{symbol} não é store.{symbol}"
    )


@pytest.mark.parametrize("symbol", _ORCH_SYMBOLS)
def test_ac3_wrapper_nao_e_is_identico_ao_orquestrador(symbol: str) -> None:
    """AC3: os 5 orquestradores em `rag` são WRAPPERS distintos (não `is`-idênticos)."""
    assert getattr(rag, symbol) is not getattr(orchestrator, symbol), (
        f"rag.{symbol} deveria ser um WRAPPER distinto de orchestrator.{symbol}"
    )


# ---------------------------------------------------------------------------
# AC4 — seam-binding: o wrapper injeta rag.DIR_VECTORDB (patchado) no orquestrador
# ---------------------------------------------------------------------------
def test_ac4_wrapper_injeta_dir_vectordb_patchado_no_orquestrador(tmp_path) -> None:
    """AC4: `monkeypatch` de rag.DIR_VECTORDB flui para Chroma(persist_directory=...).

    Dirige `rag.adicionar_documento_incrementalmente` (o wrapper). O wrapper resolve
    `rag.DIR_VECTORDB` no momento da chamada e o injeta; o orquestrador constrói
    `Chroma(persist_directory=str(<valor patchado>))`.
    """
    vdb_patched = tmp_path / "vdb_patched"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    mock_vdb = MagicMock()
    mock_vdb.get.return_value = {"ids": ["id1"], "metadatas": [{"file_hash": "x"}]}

    with (
        patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb) as mock_chroma,
        patch("agenticlog.rag._get_rag_embedding_model", return_value=MagicMock()),
        patch("agenticlog.rag.DIR_DOCUMENTS", new=docs_dir),
        patch("agenticlog.rag.DIR_VECTORDB", new=vdb_patched),
    ):
        # hash "x" != conteúdo → não-duplicado; mas paramos cedo: só validamos o seam.
        # Usamos um conteúdo cujo hash difere de "x" → segue para upsert; mas o mock_vdb
        # não terá old_ids reais úteis. Basta afirmar o persist_directory de Chroma.
        try:
            rag.adicionar_documento_incrementalmente("doc.json", b'{"k": "v"}')
        except Exception:
            pass  # a asserção do seam independe do resultado final

    _, kwargs = mock_chroma.call_args
    assert kwargs["persist_directory"] == str(vdb_patched), (
        "o wrapper não injetou rag.DIR_VECTORDB patchado no orquestrador"
    )


# ---------------------------------------------------------------------------
# AC5 — cria_vectordb não referencia DirectoryLoader
# ---------------------------------------------------------------------------
def test_ac5_cria_vectordb_sem_directory_loader() -> None:
    """AC5: o corpo de cria_vectordb não menciona DirectoryLoader (itera carregar_json)."""
    fonte = inspect.getsource(orchestrator.cria_vectordb)
    assert "DirectoryLoader" not in fonte, "cria_vectordb ainda referencia DirectoryLoader"
    assert "carregar_json" in fonte, "cria_vectordb deveria iterar carregar_json"


# ---------------------------------------------------------------------------
# AC9 — import frio de store + orchestrator sai 0 (sem ciclo)
# ---------------------------------------------------------------------------
def test_ac9_import_store_orchestrator_interpretador_frio_sai_zero() -> None:
    """AC9: `import store; import orchestrator` em subprocess frio → exit 0, sem ciclo."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c",
         "import agenticlog.ingestion.store; import agenticlog.ingestion.orchestrator"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Import frio falhou (exit {result.returncode}).\nstderr: {result.stderr!r}"
    )
    assert "circular" not in result.stderr.lower(), result.stderr
    assert "partially initialized" not in result.stderr.lower(), result.stderr


def test_ac9_orchestrator_nao_importa_rag_nem_agent_no_nivel_modulo() -> None:
    """AC9 (companion): store/orchestrator não importam rag/agent no nível de módulo."""
    for mod in (store, orchestrator):
        source_lines = inspect.getsource(mod).splitlines()
        # Só imports de NÍVEL DE MÓDULO (coluna 0) formam ciclo; o `from agenticlog.agent
        # import invalidar_vector_db` é lazy (indentado, dentro de função) e é EXATAMENTE
        # o que quebra o ciclo — deve ser permitido.
        forbidden = [
            line
            for line in source_lines
            if line.startswith(("import ", "from "))
            and ("agenticlog.rag" in line or "agenticlog.agent" in line)
        ]
        assert forbidden == [], (
            f"{mod.__name__} importa rag/agent no nível de módulo (ciclo):\n  "
            + "\n  ".join(forbidden)
        )

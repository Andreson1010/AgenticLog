# AgenticLog - CLI de ingestão (entrypoint `python -m agenticlog.ingestion`)
"""
Ponto de entrada da linha de comando para construir ou atualizar o banco vetorial.

Realocado de `agenticlog.rag` na Fase 6 do ADR-018. Substitui
`python -m agenticlog.rag` por `python -m agenticlog.ingestion`.
"""

import argparse
import logging

from agenticlog import config as _config
from agenticlog.ingestion.orchestrator import cria_vectordb, ingerir_incrementalmente
from agenticlog.observability.logging import _JsonFormatter
from agenticlog.shared.errors import RAGSecurityError

# Loga sob "agenticlog.rag" (não __name__): preserva a saída byte-idêntica dos registros
# e mantém os `assertLogs("agenticlog.rag")` existentes verdes (mesmo padrão de
# ingestion.orchestrator/store — ADR-018 Fase 6).
logger = logging.getLogger("agenticlog.rag")


def _configurar_logging_cli() -> None:
    """Configura o logger do pacote 'agenticlog' para a execução via CLI.

    Lê ``LOG_LEVEL``/``LOG_FORMAT`` dinamicamente de ``agenticlog.config`` (não via
    binding de topo) para reagir a ``importlib.reload(config)`` em testes sem precisar
    recarregar este módulo.
    """
    pkg_logger = logging.getLogger("agenticlog")
    pkg_logger.setLevel(_config.LOG_LEVEL)
    pkg_logger.handlers.clear()

    handler = logging.StreamHandler()
    if _config.LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    pkg_logger.addHandler(handler)


def _executar_main(argv: list[str] | None = None) -> None:
    """Ponto de entrada CLI — configura logging e ingere documentos (REC-04).

    Sem flags  → ingestão incremental de todos os arquivos em data/documents/.
    --rebuild  → reconstrução completa do banco vetorial (comportamento legado).
    """
    parser = argparse.ArgumentParser(
        prog="python -m agenticlog.ingestion",
        description="Constrói ou atualiza o banco vetorial ChromaDB do AgenticLog.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstrói o banco vetorial do zero (descarta o índice atual).",
    )
    args = parser.parse_args(argv)

    _configurar_logging_cli()

    try:
        if args.rebuild:
            cria_vectordb()
        else:
            ingerir_incrementalmente()
    except RAGSecurityError as e:
        logger.error("Erro de segurança: %s", e)
        raise SystemExit(1) from e
    except Exception as e:
        operacao = "rebuild do banco vetorial" if args.rebuild else "ingestão incremental"
        logger.error("Erro durante %s: %s", operacao, e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    _executar_main()

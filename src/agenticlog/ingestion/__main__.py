# AgenticLog - Entry point do CLI de ingestão (`python -m agenticlog.ingestion`)
"""Delega para `ingestion.cli._executar_main` (realocado na Fase 6 do ADR-018)."""

from agenticlog.ingestion.cli import _executar_main

if __name__ == "__main__":
    _executar_main()

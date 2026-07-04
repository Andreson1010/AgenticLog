# AgenticLog - Round-trip de identidade dos shims + acicidade de ingestion
"""Contrato de identidade dos shims e acicidade de imports (ADR-018 Fase 3a).

RAGING-13 (round-trip `rag.X is ingestion.<mod>.X`) e RAGING-10 (import acíclico:
`import agenticlog.ingestion` em interpretador frio sai 0, sem ciclo).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

import agenticlog.ingestion.chunking as chk  # noqa: E402
import agenticlog.ingestion.cleaning as cln  # noqa: E402
import agenticlog.ingestion.embeddings as emb  # noqa: E402
import agenticlog.ingestion.extraction as ext  # noqa: E402
import agenticlog.ingestion.metadata as md  # noqa: E402
import agenticlog.ingestion.security as sec  # noqa: E402
import agenticlog.rag as rag  # noqa: E402

# (nome, módulo de origem) para cada símbolo movido re-exportado por rag.py via shim.
_SIMBOLOS_MOVIDOS = [
    ("_valida_path_documentos", sec),
    ("_valida_json_sem_chaves_proibidas", sec),
    ("_valida_arquivos_json", sec),
    ("_sanitizar_nome_arquivo", sec),
    ("_sanitizar_nome_colecao", sec),
    ("sanitizar_nome_colecao", sec),
    ("salvar_documento_enviado", sec),
    ("salvar_pdf_enviado", sec),
    ("extrair_texto_pdf", ext),
    ("carregar_json", ext),
    ("filtrar_documentos_vazios", cln),
    ("SemanticChunker", chk),
    ("criar_embedding_model", emb),
    ("_computar_hash_conteudo", md),
    ("_hash_arquivo", md),
    ("_enriquecer_metadados_chunks", md),
]


class TestShimsIdentidade(unittest.TestCase):
    """Round-trip: `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` (RAGING-08/13)."""

    def teste_1_identidade_de_cada_simbolo_movido(self) -> None:
        """Cada símbolo movido é o MESMO objeto em rag e no módulo de origem."""
        for nome, modulo in _SIMBOLOS_MOVIDOS:
            with self.subTest(simbolo=nome):
                self.assertIs(getattr(rag, nome), getattr(modulo, nome))

    def teste_2_caminho_de_import_antigo_resolve(self) -> None:
        """`from agenticlog.rag import X` continua resolvendo (sem ImportError)."""
        for nome, _ in _SIMBOLOS_MOVIDOS:
            with self.subTest(simbolo=nome):
                self.assertTrue(hasattr(rag, nome))


class TestIngestionAcyclic(unittest.TestCase):
    """Acicidade de imports do pacote ingestion (RAGING-10)."""

    def teste_1_import_ingestion_interpretador_frio_sai_zero(self) -> None:
        """`import agenticlog.ingestion` em subprocess frio sai 0 e sem 'circular'."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", "import agenticlog.ingestion"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Import frio falhou (exit {result.returncode}).\nstderr: {result.stderr!r}",
        )
        self.assertNotIn("circular", result.stderr.lower(), result.stderr)
        self.assertNotIn("partially initialized", result.stderr.lower(), result.stderr)

# AgenticLog - Testes de interface Streamlit
"""
Testes para app.py usando streamlit.testing.v1.AppTest.
Todos os chamados a httpx.post são mockados via patch("app.httpx.post") —
nenhuma chamada real ao servidor FastAPI é feita nestes testes.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(_root / "app.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_success_response(
    ranked_response: object = "Resposta de teste.",
    confidence_score: float | None = 0.85,
    retrieved_info: list | None = None,
    next_step: str = "retrieve",
) -> MagicMock:
    """Cria um MagicMock representando uma resposta HTTP 200 bem-sucedida."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ranked_response": ranked_response,
        "confidence_score": confidence_score,
        "next_step": next_step,
        "retrieved_info": retrieved_info if retrieved_info is not None else [],
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


# ---------------------------------------------------------------------------
# Classe de testes
# ---------------------------------------------------------------------------

class TestStreamlitUI(unittest.TestCase):
    """Testes de interface Streamlit para app.py."""

    def teste_1_consulta_sucesso(self) -> None:
        """Mock 200 com resposta completa: session_state populado, st.error nunca chamado."""
        mock_response = _make_success_response(
            ranked_response="Prazo de entrega é 5 dias úteis.",
            confidence_score=0.85,
            next_step="retrieve",
        )
        with patch("app.httpx.post", return_value=mock_response):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("Qual o prazo de entrega?").run()

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        self.assertEqual(at.session_state["ranked_response"], "Prazo de entrega é 5 dias úteis.")
        self.assertEqual(at.session_state["confidence_score"], 0.85)
        self.assertEqual(at.session_state["next_step"], "retrieve")
        error_texts = [e.value for e in at.error]
        confidence_errors = [t for t in error_texts if "0." in t]
        non_confidence_errors = [
            t for t in error_texts
            if not any(kw in t for kw in ["Confiança baixa", "Confiança média"])
        ]
        self.assertEqual(
            non_confidence_errors, [],
            msg=f"st.error inesperado chamado: {non_confidence_errors}",
        )

    def teste_2_documentos_renderizados_como_dict(self) -> None:
        """Mock 200 com docs como dicts: sem AttributeError, renderização correta."""
        docs = [
            {"page_content": "conteúdo relevante", "metadata": {"source": "frete.json"}},
            {"page_content": "outro conteúdo", "metadata": {"source": "estoque.json"}},
        ]
        mock_response = _make_success_response(
            ranked_response="Resposta com documentos.",
            confidence_score=0.80,
            retrieved_info=docs,
        )
        with patch("app.httpx.post", return_value=mock_response):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("consulta com documentos").run()

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        self.assertGreater(len(at.expander), 0, msg="Nenhum expander de documento encontrado.")
        expander_labels = [e.label for e in at.expander]
        self.assertTrue(
            any("frete.json" in label for label in expander_labels),
            msg=f"Source 'frete.json' não encontrada nos expanders: {expander_labels}",
        )

    def teste_3_retrieved_info_vazio(self) -> None:
        """Mock 200 com retrieved_info vazio: mensagem de fallback renderizada."""
        mock_response = _make_success_response(
            ranked_response="Resposta sem documentos.",
            confidence_score=0.70,
            retrieved_info=[],
        )
        with patch("app.httpx.post", return_value=mock_response):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("consulta sem docs").run()

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        markdown_texts = [m.value for m in at.markdown]
        write_texts = [w.value for w in at.write] if hasattr(at, "write") else []
        all_texts = markdown_texts + write_texts
        self.assertTrue(
            any("Nenhum documento relacionado encontrado." in t for t in all_texts),
            msg=f"Mensagem de fallback não encontrada. Textos: {all_texts}",
        )

    def teste_4_confidence_none(self) -> None:
        """Mock 200 com confidence_score=None: guard or 0.0 evita crash em st.progress."""
        mock_response = _make_success_response(
            ranked_response="Resposta com confiança nula.",
            confidence_score=None,
        )
        with patch("app.httpx.post", return_value=mock_response):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta qualquer").run()

        self.assertFalse(at.exception, msg=f"Crash com confidence_score=None: {at.exception}")
        self.assertEqual(at.session_state["ranked_response"], "Resposta com confiança nula.")

    def teste_5_next_step_invalido(self) -> None:
        """Mock 200 com next_step desconhecido: badge suprimido sem erro."""
        mock_response = _make_success_response(
            ranked_response="Resposta com rota desconhecida.",
            confidence_score=0.60,
            next_step="rota_inexistente",
        )
        with patch("app.httpx.post", return_value=mock_response):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta rota inválida").run()

        self.assertFalse(at.exception, msg=f"Exceção com next_step inválido: {at.exception}")
        info_texts = [i.value for i in at.info]
        self.assertFalse(
            any("rota_inexistente" in t for t in info_texts),
            msg="Badge de rota inexistente foi renderizado quando deveria ser suprimido.",
        )


if __name__ == "__main__":
    unittest.main()

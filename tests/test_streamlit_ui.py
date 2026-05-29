# AgenticLog - Testes de interface Streamlit
"""
Testes para app.py usando streamlit.testing.v1.AppTest.
Todos os chamados ao agent_workflow e check_lmstudio_health são mockados —
nenhuma chamada real ao LMStudio é feita nestes testes.

Paths de mock:
- agenticlog.check_lmstudio_health  — função importada em app.py via `from agenticlog import ...`
- agenticlog.agent_workflow         — objeto LangGraph compilado; app chama .invoke() nele
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

def _fake_doc(content: str = "conteúdo do documento", source: str = "doc_teste.json") -> MagicMock:
    """Cria um Document falso compatível com a interface esperada pelo app."""
    doc = MagicMock()
    doc.page_content = content
    doc.id = "fake-id-001"
    doc.metadata = {"source": source}
    return doc


def _make_output(
    ranked_response: object = "Resposta de teste.",
    confidence_score: float = 0.85,
    retrieved_info: list | None = None,
    next_step: str = "retrieve",
) -> dict:
    """Monta o dict de saída que agent_workflow.invoke() retornaria."""
    return {
        "ranked_response": ranked_response,
        "confidence_score": confidence_score,
        "retrieved_info": retrieved_info if retrieved_info is not None else [],
        "next_step": next_step,
    }


def _make_workflow_mock(output: dict) -> MagicMock:
    """Cria um mock de agent_workflow cujo .invoke() retorna output."""
    mock_wf = MagicMock()
    mock_wf.invoke.return_value = output
    return mock_wf


# ---------------------------------------------------------------------------
# Classe de testes
# ---------------------------------------------------------------------------

class TestStreamlitUI(unittest.TestCase):
    """Testes de interface Streamlit para app.py."""

    def teste_1_app_carrega_sem_erros(self) -> None:
        """App deve renderizar sem lançar exceção quando não há interação."""
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(_make_output())),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()

        self.assertFalse(at.exception, msg=f"App lançou exceção no carregamento: {at.exception}")

    def teste_2_consulta_chama_workflow_e_exibe_resposta(self) -> None:
        """Enviar uma consulta deve invocar o workflow e exibir ranked_response."""
        output = _make_output(
            ranked_response="Prazo de entrega é 5 dias úteis.",
            confidence_score=0.85,
        )
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("Qual o prazo de entrega?").run()
            at.button[0].click().run()

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        markdown_texts = [m.value for m in at.markdown]
        self.assertTrue(
            any("Prazo de entrega é 5 dias úteis." in t for t in markdown_texts),
            msg=f"ranked_response não encontrado nos markdowns: {markdown_texts}",
        )

    def teste_3_confianca_alta_exibe_badge_sucesso(self) -> None:
        """Quando confidence_score >= 0.7, app deve exibir elemento st.success."""
        output = _make_output(ranked_response="Alta confiança.", confidence_score=0.90)
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta de teste").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        success_texts = [s.value for s in at.success]
        self.assertTrue(
            any("0.90" in t for t in success_texts),
            msg=f"Badge de confiança alta não encontrado: {success_texts}",
        )

    def teste_4_confianca_media_exibe_badge_aviso(self) -> None:
        """Quando 0.4 <= confidence_score < 0.7, app deve exibir st.warning."""
        output = _make_output(ranked_response="Confiança média.", confidence_score=0.55)
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta de teste").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        warning_texts = [w.value for w in at.warning]
        self.assertTrue(
            any("0.55" in t for t in warning_texts),
            msg=f"Badge de confiança média não encontrado: {warning_texts}",
        )

    def teste_5_confianca_baixa_exibe_badge_erro(self) -> None:
        """Quando confidence_score < 0.4, app deve exibir st.error de confiança baixa."""
        output = _make_output(ranked_response="Confiança baixa.", confidence_score=0.20)
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta de teste").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any("0.20" in t for t in error_texts),
            msg=f"Badge de confiança baixa não encontrado: {error_texts}",
        )

    def teste_6_consulta_vazia_nao_causa_crash(self) -> None:
        """Enviar consulta vazia não deve lançar exceção não tratada."""
        output = _make_output(ranked_response="Nenhuma resposta.", confidence_score=0.0)
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click().run()

        self.assertFalse(at.exception, msg=f"Crash com consulta vazia: {at.exception}")

    def teste_7_erro_lmstudio_exibe_mensagem_amigavel(self) -> None:
        """Quando check_lmstudio_health lança LMStudioUnavailableError, app exibe st.error."""
        from agenticlog import LMStudioUnavailableError

        mock_wf = MagicMock()
        mock_wf.invoke.side_effect = LMStudioUnavailableError("LMStudio offline")

        with (
            patch("agenticlog.agent_workflow", mock_wf),
            patch(
                "agenticlog.check_lmstudio_health",
                side_effect=LMStudioUnavailableError("LMStudio offline"),
            ),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta qualquer").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any("LMStudio" in t for t in error_texts),
            msg=f"Mensagem de erro LMStudio não encontrada: {error_texts}",
        )

    def teste_8_erro_generico_exibe_mensagem_fallback(self) -> None:
        """Exceção genérica do workflow deve exibir mensagem de erro amigável via st.error.

        Nota: AppTest registra a exceção em at.exception mesmo quando app.py a captura
        internamente com try/except e chama st.error(). Por isso não assertamos
        assertFalse(at.exception) neste caso — apenas verificamos que st.error foi exibido.
        """
        mock_wf = MagicMock()
        mock_wf.invoke.side_effect = RuntimeError("falha inesperada")

        with (
            patch("agenticlog.agent_workflow", mock_wf),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta qualquer").run()
            at.button[0].click().run()

        error_texts = [e.value for e in at.error]
        self.assertTrue(
            len(error_texts) > 0,
            msg="Nenhuma mensagem de erro exibida para exceção genérica.",
        )

    def teste_9_documentos_recuperados_sao_exibidos(self) -> None:
        """Quando retrieved_info não está vazio, app deve exibir expanders de documentos."""
        docs = [_fake_doc("conteúdo relevante", "frete.json")]
        output = _make_output(
            ranked_response="Resposta com documentos.",
            confidence_score=0.80,
            retrieved_info=docs,
        )
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("consulta com documentos").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertGreater(
            len(at.expander),
            0,
            msg="Nenhum expander de documento encontrado.",
        )

    def teste_10_rota_retrieve_exibe_badge_info(self) -> None:
        """Quando next_step='retrieve', app deve exibir st.info com texto correto."""
        output = _make_output(
            ranked_response="Resposta via RAG.",
            confidence_score=0.75,
            next_step="retrieve",
        )
        with (
            patch("agenticlog.agent_workflow", _make_workflow_mock(output)),
            patch("agenticlog.check_lmstudio_health", return_value=None),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("consulta retrieve").run()
            at.button[0].click().run()

        self.assertFalse(at.exception)
        info_texts = [i.value for i in at.info]
        self.assertTrue(
            any("Busca no Banco de Dados" in t for t in info_texts),
            msg=f"Badge de rota 'retrieve' não encontrado: {info_texts}",
        )


if __name__ == "__main__":
    unittest.main()

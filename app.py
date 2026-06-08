# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA

# Importa a biblioteca Streamlit para criação da interface web
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from agenticlog.agent import _listar_colecoes
from agenticlog.config import (
    API_CLIENT_TIMEOUT_SECONDS,
    API_HOST,
    API_PORT,
    DEFAULT_COLLECTION_NAME,
)
from agenticlog.rag import (
    RAGSecurityError,
    _sanitizar_nome_colecao,
    adicionar_documento_incrementalmente,
    reconstruir_vectordb,
    salvar_pdf_enviado,
)

NOVA_COLECAO_SENTINEL = "Nova coleção…"

# ---------------------------------------------------------------------------
# Constantes de mensagem de erro
# ---------------------------------------------------------------------------
MSG_LMSTUDIO_DOWN = "LMStudio indisponível. Inicie o servidor e carregue o modelo."
MSG_VECTORDB_AUSENTE = "Base vetorial não encontrada. Execute: python -m agenticlog.rag"
MSG_CONNECT_ERROR = (
    "Não foi possível conectar ao servidor FastAPI. "
    "Inicie com: uvicorn agenticlog.api:app"
)
MSG_TIMEOUT = "Tempo limite de resposta excedido. O servidor pode estar sobrecarregado."
MSG_ERRO_VALIDACAO = "Erro de validação na consulta. Verifique o texto enviado."
MSG_ERRO_INTERNO = "Erro interno do servidor."


def _safe_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", "")
    except Exception:
        return ""


def _consultar_api(query: str) -> dict:
    """Envia consulta ao endpoint POST /query e retorna o JSON de resposta.

    Lança httpx.HTTPStatusError para respostas com status >= 400.
    Lança httpx.ConnectError ou httpx.TimeoutException para falhas de rede.
    Não captura exceções — o chamador é responsável pelo tratamento.
    """
    response = httpx.post(  # nosec B113
        f"http://{API_HOST}:{API_PORT}/query",
        json={"query": query},
        timeout=API_CLIENT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _ingerir_documento(
    uploaded_file: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Processa o arquivo enviado: PDF faz rebuild completo; JSON usa ingestão incremental.

    Entrada: uploaded_file — objeto UploadedFile do Streamlit.
             collection_name — nome da coleção ChromaDB de destino.
    Saída: nenhuma (efeito colateral: exibe feedback na UI, atualiza vectordb).
    """
    conteudo: bytes = uploaded_file.getvalue()
    filename: str = uploaded_file.name
    suffix = Path(filename).suffix.lower()

    if suffix not in {".json", ".pdf"}:
        st.error("Formato não suportado. Envie apenas arquivos .json ou .pdf.")
        return

    if suffix == ".pdf":
        try:
            saved_path = salvar_pdf_enviado(filename, conteudo, collection_name)
        except RAGSecurityError as e:
            st.error(str(e))
            return
        try:
            with st.spinner("Reconstruindo base vetorial..."):
                reconstruir_vectordb(collection_name)
        except Exception as e:
            saved_path.unlink(missing_ok=True)
            st.error(f"Erro ao reconstruir base vetorial. Arquivo removido. Detalhe: {e}")
            return
        st.success(f"Documento ingerido com sucesso na coleção '{collection_name}'.")
        st.rerun()
    else:
        try:
            with st.spinner("Adicionando documento à base vetorial..."):
                resultado = adicionar_documento_incrementalmente(
                    filename, conteudo, collection_name
                )
        except RAGSecurityError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Erro ao adicionar documento. Detalhe: {e}")
            return
        status = resultado["status"]
        mensagem = resultado["mensagem"]
        if status == "adicionado":
            st.success(mensagem)
            st.rerun()
        elif status == "duplicado":
            st.info(mensagem)
        elif status == "hash_diferente":
            st.warning(mensagem)


# Define o título, ícone e layout inicial da página Streamlit
st.set_page_config(page_title="Aivorak", page_icon="🚚", layout="centered")

# Chaves de session_state inicializadas uma vez por sessão Streamlit.
# Persistem entre reruns enquanto a aba do navegador permanecer aberta.
# ranked_response: resposta ranqueada retornada pelo agente
# confidence_score: score de confiança da resposta (0.0–1.0)
# retrieved_info: documentos recuperados do vectordb
# next_step: rota usada pelo agente ("retrieve", "gerar", "usar_web")
if "ranked_response" not in st.session_state:
    st.session_state.ranked_response = None
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = None
if "retrieved_info" not in st.session_state:
    st.session_state.retrieved_info = []
if "next_step" not in st.session_state:
    st.session_state.next_step = None

# Adiciona o título "Instruções" na barra lateral
st.sidebar.title("Instruções")

# Exibe instruções detalhadas ao usuário na barra lateral
st.sidebar.write("""
- Digite perguntas específicas sobre logística e supply chain para obter respostas detalhadas.
- O assistente de IA vai utilizar a base de dados do RAG para gerar respostas customizadas.
- Documentos, contratos e procedimentos complementares podem ser usados para
  aperfeiçoar o sistema de RAG.
- IA Generativa comete erros. SEMPRE valide as respostas.
""")

# Cria botão "Suporte" na barra lateral e verifica se foi clicado
if st.sidebar.button("Suporte"):

    # Exibe informações de contato caso o botão seja clicado
    st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@aivoraq.com.br")

# Expander para ingestão de novos documentos JSON no banco vetorial
with st.sidebar.expander("Adicionar Documento"):
    colecoes_existentes = _listar_colecoes()
    opcoes = colecoes_existentes + [NOVA_COLECAO_SENTINEL]
    selecao = st.selectbox("Coleção", opcoes, key="selecao_colecao")

    collection_name = DEFAULT_COLLECTION_NAME
    colecao_valida = True
    if selecao == NOVA_COLECAO_SENTINEL:
        nome_input = st.text_input("Nome da nova coleção", key="nome_nova_colecao")
        if not nome_input:
            st.warning("Digite o nome da nova coleção antes de ingerir.")
            colecao_valida = False
        else:
            try:
                _sanitizar_nome_colecao(nome_input)
                st.caption("Nome válido.")
                collection_name = nome_input
            except RAGSecurityError as e:
                st.caption(f"Nome inválido: {e}")
                colecao_valida = False
    else:
        collection_name = selecao

    uploaded_file = st.file_uploader("Selecione um arquivo JSON ou PDF", type=None)
    if st.button("Ingerir Documento", disabled=not colecao_valida):
        if uploaded_file is None:
            st.warning("Selecione um arquivo antes de ingerir.")
        else:
            _ingerir_documento(uploaded_file, collection_name)

# Exibe título principal
st.title("AVK - Agência de IA")

# Exibe subtítulo secundário (substituindo o segundo st.title por st.subheader)
st.subheader("IA Generativa e Agentic RAG Para a Área de Logística")

# Solicita ao usuário que digite uma pergunta através de um campo de texto
query = st.text_input("Digite sua pergunta:")

# Mapeia next_step para rótulos legíveis em português exibidos na UI
_ROTAS = {
    "retrieve": "Rota: Busca no Banco de Dados",
    "gerar": "Rota: Geração Direta",
    "usar_web": "Rota: Busca na Web",
}

# Verifica se o usuário clicou no botão "Enviar"
if st.button("Enviar"):

    # Exibe um spinner indicando que a consulta está sendo processada
    with st.spinner("Processando consulta... Aguarde."):

        try:
            output = _consultar_api(query)

            # _consultar_api() retorna dict com as chaves do JSON de resposta;
            # lemos cada chave com .get() para garantir valor padrão seguro
            st.session_state.ranked_response = output.get("ranked_response", "Nenhuma resposta.")
            st.session_state.confidence_score = output.get("confidence_score", 0.0)
            st.session_state.retrieved_info = output.get("retrieved_info", [])
            st.session_state.next_step = output.get("next_step", None)

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 503:
                detail = _safe_detail(e.response)
                if "LMStudio" in detail:
                    st.error(MSG_LMSTUDIO_DOWN)
                elif "vetorial" in detail or "agenticlog.rag" in detail:
                    st.error(MSG_VECTORDB_AUSENTE)
                else:
                    st.error(f"Serviço indisponível: {detail}" if detail else MSG_ERRO_INTERNO)
            elif status_code == 500:
                detail = _safe_detail(e.response)
                st.error(MSG_ERRO_INTERNO)
                with st.expander("Detalhes do erro"):
                    st.write(detail)
            elif status_code == 422:
                st.error(MSG_ERRO_VALIDACAO)
            else:
                detail = _safe_detail(e.response) or str(e)
                st.error(detail)

        except httpx.ConnectError:
            st.error(MSG_CONNECT_ERROR)

        except httpx.TimeoutException:
            st.error(MSG_TIMEOUT)

# Exibe os resultados armazenados no session_state (persistem entre reruns)
if st.session_state.ranked_response is not None:

    # Exibe a rota utilizada pelo agente como badge informativo
    rota = st.session_state.next_step
    if rota in _ROTAS:
        st.info(_ROTAS[rota])

    # Exibe subtítulo "Resposta:"
    st.subheader("Resposta:")

    # Obtém a resposta ranqueada do session_state
    resposta = st.session_state.ranked_response

    # Verifica se a resposta está em formato dicionário contendo chave "answer"
    if isinstance(resposta, dict) and "answer" in resposta:

        # Se sim, extrai o valor da chave "answer"
        resposta = resposta["answer"]

    # Exibe a resposta formatada como Markdown
    st.markdown(resposta)

    # Exibe subtítulo indicando o nível de confiança da resposta
    st.subheader("Confiança da Resposta com Base no RAG:")

    # Obtém o score de confiança do session_state; usa 0.0 se None ou não-numérico (ex.: após erro)
    raw_confidence = st.session_state.confidence_score
    confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.0

    # Exibe barra de progresso visual com o valor de confiança (0.0 a 1.0)
    st.progress(confidence)

    # Exibe badge colorido de acordo com o nível de confiança
    if confidence >= 0.7:
        st.success(f"Confiança alta: {confidence:.2f}")
    elif confidence >= 0.4:
        st.warning(f"Confiança média: {confidence:.2f}")
    else:
        st.error(f"Confiança baixa: {confidence:.2f}")

    # Obtém os documentos relacionados que foram recuperados pela consulta
    documentos_relacionados = st.session_state.retrieved_info

    # Verifica se existem documentos relacionados recuperados
    if documentos_relacionados:

        # Exibe subtítulo "Documentos Relacionados:"
        st.subheader("Documentos Relacionados:")

        # Itera sobre cada documento recuperado
        for i, doc in enumerate(documentos_relacionados):

            # Obtém a fonte do documento para usar no título do expander
            source = doc["metadata"].get("source", "Desconhecida")

            # Exibe cada documento em um expander colapsável para melhor legibilidade
            with st.expander(f"Documento {i + 1} — {source}"):

                # Exibe o conteúdo do documento numa caixa de texto com altura definida
                # key único evita StreamlitDuplicateElementId quando há múltiplos documentos
                st.text_area("Conteúdo", doc["page_content"], height=80, key=f"doc_content_{i}")
    else:

        # Caso não haja documentos, exibe mensagem informando ao usuário
        st.write("Nenhum documento relacionado encontrado.")

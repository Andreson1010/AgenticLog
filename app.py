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
    sanitizar_nome_colecao,
    adicionar_documento_incrementalmente,
    reconstruir_vectordb,
    salvar_pdf_enviado,
)

NOVA_COLECAO_SENTINEL = "Nova coleção…"

MSG_LMSTUDIO_DOWN = "LMStudio indisponível. Inicie o servidor e carregue o modelo."
MSG_VECTORDB_AUSENTE = "Base vetorial não encontrada. Execute: python -m agenticlog.rag"
MSG_CONNECT_ERROR = "Não foi possível conectar ao servidor. Inicie com: uvicorn agenticlog.api:app"
MSG_TIMEOUT = "Tempo limite excedido. O servidor pode estar sobrecarregado."
MSG_ERRO_VALIDACAO = "Erro de validação na consulta. Verifique o texto enviado."
MSG_ERRO_INTERNO = "Erro interno do servidor."

_ROTAS = {
    "retrieve": ("Banco de Dados", "🗄️"),
    "gerar": ("Geração Direta", "✨"),
    "usar_web": ("Busca na Web", "🌐"),
}


def _safe_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", "")
    except Exception:
        return ""


def _consultar_api(query: str) -> dict:
    response = httpx.post(  # nosec B113
        f"http://{API_HOST}:{API_PORT}/query",
        json={"query": query},
        timeout=API_CLIENT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _ingerir_documento(uploaded_file: Any, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
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
            st.error(f"Erro ao reconstruir base vetorial. Detalhe: {e}")
            return
        st.success(f"Documento ingerido na coleção '{collection_name}'.")
        st.rerun()
    else:
        try:
            with st.spinner("Adicionando documento..."):
                resultado = adicionar_documento_incrementalmente(filename, conteudo, collection_name)
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


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Assistente Logístico", page_icon="🚚", layout="centered")

st.markdown("""
<style>
/* ---- Reset geral ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- Fundo e fonte base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f9f9f8;
    font-family: "Söhne", ui-sans-serif, system-ui, -apple-system, sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    border-right: none;
}
[data-testid="stSidebar"] * {
    font-size: 0.85rem;
    color: #d4d4d4 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #aaaaaa !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #2a2a2a !important;
    color: #d4d4d4 !important;
    border-color: #3a3a3a !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background-color: #2a2a2a !important;
    color: #d4d4d4 !important;
    border: 1px solid #3a3a3a !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background-color: #3a3a3a !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2a2a2a !important;
}

/* ---- Título principal ---- */
.cl-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 2.5rem 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.cl-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #888;
    margin-bottom: 2rem;
}

/* ---- Caixa de resposta ---- */
.cl-response-box {
    background: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 12px;
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    font-size: 0.97rem;
    color: #1a1a1a;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ---- Linha de metadados no rodapé da resposta ---- */
.cl-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px solid #f0f0ee;
}
.cl-badge {
    font-size: 0.72rem;
    color: #999;
    background: #f4f4f2;
    border-radius: 20px;
    padding: 2px 8px;
    border: 1px solid #e5e5e3;
}

/* ---- Ícone de confiança com tooltip ---- */
.cl-conf {
    position: relative;
    display: inline-flex;
    align-items: center;
    cursor: help;
    font-size: 0.72rem;
    color: #aaa;
    background: #f4f4f2;
    border-radius: 20px;
    padding: 2px 7px;
    border: 1px solid #e5e5e3;
    gap: 3px;
    transition: background 0.15s;
}
.cl-conf:hover {
    background: #ececea;
    color: #555;
}
.cl-conf .cl-tooltip {
    visibility: hidden;
    opacity: 0;
    width: 200px;
    background: #1a1a1a;
    color: #fff;
    text-align: left;
    border-radius: 8px;
    padding: 8px 10px;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 999;
    font-size: 0.75rem;
    line-height: 1.5;
    transition: opacity 0.15s;
    pointer-events: none;
    white-space: normal;
}
.cl-conf:hover .cl-tooltip {
    visibility: visible;
    opacity: 1;
}

/* ---- Campo de entrada ---- */
div[data-testid="stTextInput"] input {
    border-radius: 10px;
    border: 1.5px solid #e5e5e3;
    background: #ffffff;
    color: #1a1a1a !important;
    font-size: 0.97rem;
    padding: 0.65rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: border-color 0.15s;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #aaaaaa !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #b5a8f0;
    box-shadow: 0 0 0 3px rgba(181,168,240,0.15);
    outline: none;
    color: #1a1a1a !important;
}

/* ---- Botão Enviar ---- */
div[data-testid="stButton"] > button {
    border-radius: 10px;
    background: #1a1a1a;
    color: #fff;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.5rem;
    border: none;
    width: 100%;
    transition: background 0.15s;
}
div[data-testid="stButton"] > button:hover {
    background: #3d3d3d;
}

/* ---- Spinner ---- */
div[data-testid="stSpinner"] p {
    color: #888;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("ranked_response", None),
    ("confidence_score", None),
    ("retrieved_info", []),
    ("next_step", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — ingestão de documentos
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📂 Adicionar Documento")
    colecoes_existentes = _listar_colecoes()
    opcoes = colecoes_existentes + [NOVA_COLECAO_SENTINEL]
    selecao = st.selectbox("Coleção", opcoes, key="selecao_colecao")

    collection_name = DEFAULT_COLLECTION_NAME
    colecao_valida = True
    if selecao == NOVA_COLECAO_SENTINEL:
        nome_input = st.text_input("Nome da nova coleção", key="nome_nova_colecao")
        if not nome_input:
            st.warning("Digite o nome antes de ingerir.")
            colecao_valida = False
        else:
            try:
                sanitizar_nome_colecao(nome_input)
                st.caption("✓ Nome válido.")
                collection_name = nome_input
            except RAGSecurityError as e:
                st.caption(f"✗ Nome inválido: {e}")
                colecao_valida = False
    else:
        collection_name = selecao

    uploaded_file = st.file_uploader("Arquivo (.json ou .pdf)", type=None)
    if st.button("Ingerir", disabled=not colecao_valida):
        if uploaded_file is None:
            st.warning("Selecione um arquivo.")
        else:
            _ingerir_documento(uploaded_file, collection_name)

    st.divider()
    st.markdown(
        "<small style='color:#aaa;'>IA Generativa comete erros.<br>Sempre valide as respostas.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — título
# ---------------------------------------------------------------------------

st.markdown('<div class="cl-title">Assistente Logístico</div>', unsafe_allow_html=True)
st.markdown('<div class="cl-subtitle">Especialista em logística e supply chain</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input("Pergunta", placeholder="Faça uma pergunta sobre logística…", label_visibility="collapsed")
with col_btn:
    enviar = st.button("Enviar")

# ---------------------------------------------------------------------------
# Envio
# ---------------------------------------------------------------------------

if enviar and query.strip():
    with st.spinner("Processando…"):
        try:
            output = _consultar_api(query)
            st.session_state.ranked_response = output.get("ranked_response", "Nenhuma resposta.")
            st.session_state.confidence_score = output.get("confidence_score", 0.0)
            st.session_state.retrieved_info = output.get("retrieved_info", [])
            st.session_state.next_step = output.get("next_step", None)

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            detail = _safe_detail(e.response)
            if code == 503:
                st.error(MSG_LMSTUDIO_DOWN if "LMStudio" in detail else MSG_VECTORDB_AUSENTE if "vetorial" in detail else MSG_ERRO_INTERNO)
            elif code == 422:
                st.error(MSG_ERRO_VALIDACAO)
            else:
                st.error(detail or MSG_ERRO_INTERNO)
        except httpx.ConnectError:
            st.error(MSG_CONNECT_ERROR)
        except httpx.TimeoutException:
            st.error(MSG_TIMEOUT)

elif enviar and not query.strip():
    st.warning("Digite uma pergunta antes de enviar.")

# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------

if st.session_state.ranked_response is not None:
    resposta = st.session_state.ranked_response
    if isinstance(resposta, dict) and "answer" in resposta:
        resposta = resposta["answer"]

    confidence = float(st.session_state.confidence_score or 0.0)
    next_step = st.session_state.next_step or ""
    rota_label, rota_icon = _ROTAS.get(next_step, ("Desconhecida", "❓"))

    # Nível de confiança
    if confidence >= 0.7:
        conf_label, conf_color, conf_icon = "Alta", "#22c55e", "●"
    elif confidence >= 0.4:
        conf_label, conf_color, conf_icon = "Média", "#f59e0b", "●"
    else:
        conf_label, conf_color, conf_icon = "Baixa", "#ef4444", "●"

    tooltip_text = (
        f"Confiança: {confidence:.2%}<br>"
        f"Nível: {conf_label}<br>"
        f"Rota: {rota_label}<br>"
        f"Fonte: {'RAG' if next_step == 'retrieve' else 'Web' if next_step == 'usar_web' else 'LLM'}"
    )

    conf_html = f"""
    <span class="cl-conf">
        <span style="color:{conf_color};">{conf_icon}</span>
        {confidence:.0%}
        <span class="cl-tooltip">{tooltip_text}</span>
    </span>
    """

    route_html = f'<span class="cl-badge">{rota_icon} {rota_label}</span>'

    html = f"""
    <div class="cl-response-box">
        {resposta}
        <div class="cl-meta">
            {route_html}
            {conf_html}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

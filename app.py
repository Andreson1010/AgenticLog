import html
import re
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
    adicionar_documento_incrementalmente,
    adicionar_pdf_incrementalmente,
    sanitizar_nome_colecao,
)

NOVA_COLECAO_SENTINEL = "Nova coleção…"

MSG_LMSTUDIO_DOWN = "LMStudio indisponível. Inicie o servidor e carregue o modelo."
MSG_VECTORDB_AUSENTE = "Base vetorial não encontrada. Execute: python -m agenticlog.rag"
MSG_CONNECT_ERROR = "Não foi possível conectar ao servidor FastAPI. Inicie com: uvicorn agenticlog.api:app"
MSG_TIMEOUT = "Tempo limite excedido. O servidor pode estar sobrecarregado."
MSG_ERRO_VALIDACAO = "Erro de validação na consulta. Verifique o texto enviado."
MSG_ERRO_INTERNO = "Erro interno do servidor."

# Métodos de feedback de st permitidos para a mensagem de ingestão (whitelist de dispatch).
_INGEST_MSG_TIPOS = ("success", "info", "warning", "error")

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
            with st.spinner("Adicionando documento à base vetorial..."):
                resultado = adicionar_pdf_incrementalmente(filename, conteudo, collection_name)
        except RAGSecurityError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Erro ao ingerir documento. Detalhe: {e}")
            return
        if resultado["status"] in ("adicionado", "substituido"):
            st.session_state.ingest_msg = (
                "success",
                f"Documento ingerido com sucesso na coleção '{collection_name}'.",
            )
            st.rerun()
        elif resultado["status"] == "duplicado":
            st.info(resultado["mensagem"])
        return
    else:
        try:
            with st.spinner("Adicionando documento à base vetorial..."):
                resultado = adicionar_documento_incrementalmente(filename, conteudo, collection_name)
        except RAGSecurityError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Erro ao adicionar documento. Detalhe: {e}")
            return
        status = resultado["status"]
        mensagem = resultado["mensagem"]
        if status in ("adicionado", "substituido"):
            st.session_state.ingest_msg = ("success", mensagem)
            st.rerun()
        elif status == "duplicado":
            st.info(mensagem)


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Assistente Logístico", page_icon="📦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --cl-bg: #0F1117;
    --cl-sidebar-bg: #1C1F26;
    --cl-card-bg: #1C1F26;
    --cl-border: #2D3139;
    --cl-accent: #F59E0B;
    --cl-text: #E5E7EB;
    --cl-text-secondary: #9CA3AF;
    --cl-mono: ui-monospace, "SF Mono", "Cascadia Code", "Courier New", monospace;
}

/* ---- Reset geral ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- Fundo e fonte base ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--cl-bg);
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, sans-serif;
    color: var(--cl-text);
}
h1, h2, h3, h4, h5, h6 {
    font-family: "Inter", sans-serif;
    font-weight: 600;
    color: var(--cl-text);
}
p, span, div, label, li {
    font-family: "Inter", sans-serif;
    font-weight: 400;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: var(--cl-sidebar-bg);
    border-right: 1px solid var(--cl-border);
}
[data-testid="stSidebar"] * {
    font-size: 0.85rem;
    color: var(--cl-text) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: var(--cl-text) !important;
    font-weight: 600;
}
/* Labels dos campos da sidebar (ex.: "Coleção") */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: var(--cl-text-secondary) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--cl-bg) !important;
    color: var(--cl-text) !important;
    border-color: var(--cl-border) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--cl-border) !important;
}

/* ---- File uploader na sidebar ---- */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--cl-bg) !important;
    border: 1px dashed var(--cl-border) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--cl-card-bg) !important;
    color: var(--cl-text) !important;
    border: 1px solid var(--cl-border) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--cl-accent) !important;
    color: var(--cl-accent) !important;
}
/* Traduzir "Drag and drop file here" */
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child {
    visibility: hidden;
    display: block;
    height: 0;
    overflow: hidden;
}
[data-testid="stFileUploaderDropzoneInstructions"] div::before {
    content: "Arraste e solte o arquivo aqui";
    display: block;
    font-size: 0.82rem;
    color: var(--cl-text-secondary);
    margin-bottom: 4px;
}

/* ---- Título principal ---- */
.cl-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    font-family: "Inter", sans-serif;
    color: var(--cl-accent);
    margin: 2.5rem 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.cl-subtitle {
    text-align: center;
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--cl-text-secondary);
    margin-bottom: 2rem;
}

/* ---- Estado vazio ---- */
.cl-empty-state {
    text-align: center;
    padding: 3.5rem 1rem 2rem 1rem;
    color: var(--cl-text-secondary);
    font-size: 0.88rem;
    line-height: 1.8;
}
.cl-empty-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}
.cl-empty-hints {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    margin-top: 1.25rem;
}
.cl-hint-chip {
    background: var(--cl-card-bg);
    border: 1px solid var(--cl-border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: var(--cl-accent);
    cursor: default;
}

/* ---- Bolhas de chat ---- */
.cl-chat-row {
    display: flex;
    margin: 0.6rem 0;
}
.cl-chat-row.cl-user {
    justify-content: flex-end;
}
.cl-chat-row.cl-assistant {
    justify-content: flex-start;
}
.cl-bubble {
    max-width: 80%;
    padding: 0.85rem 1.1rem;
    font-size: 0.95rem;
    line-height: 1.55;
    word-break: break-word;
}
.cl-para {
    margin: 0 0 0.5rem 0;
}
.cl-para:last-child {
    margin-bottom: 0;
}
.cl-bubble-user {
    background-color: var(--cl-accent);
    color: var(--cl-bg);
    border-radius: 12px 12px 2px 12px;
}
.cl-bubble-assistant {
    background-color: var(--cl-card-bg);
    color: var(--cl-text);
    border: 1px solid var(--cl-border);
    border-radius: 12px 12px 12px 2px;
}

/* ---- Linha de metadados ---- */
.cl-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--cl-border);
    flex-wrap: wrap;
}
.cl-badge {
    font-size: 0.72rem;
    color: var(--cl-text-secondary);
    background: var(--cl-bg);
    border-radius: 20px;
    padding: 2px 10px;
    border: 1px solid var(--cl-border);
}

/* ---- Confiança ---- */
.cl-conf-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--cl-mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cl-text-secondary);
    margin-top: 0.75rem;
}
.cl-conf-value {
    font-family: var(--cl-mono);
    color: var(--cl-text);
}
[data-testid="stProgress"] > div > div {
    background-color: var(--cl-border) !important;
}
[data-testid="stProgress"] > div > div > div {
    background-color: var(--cl-accent) !important;
}

/* ---- Campo de entrada ---- */
div[data-testid="stTextInput"] input {
    border-radius: 6px;
    border: 1px solid var(--cl-border);
    background: var(--cl-card-bg);
    color: var(--cl-text) !important;
    font-size: 0.95rem;
    padding: 0.65rem 1rem;
    transition: border-color 0.15s;
}
div[data-testid="stTextInput"] input::placeholder {
    color: var(--cl-text-secondary) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--cl-accent);
    box-shadow: 0 0 0 1px var(--cl-accent);
    outline: none;
}

/* ---- Botões primários ---- */
div[data-testid="stButton"] > button {
    border-radius: 6px;
    background-color: var(--cl-accent);
    color: var(--cl-bg);
    font-weight: 600;
    font-family: "Inter", sans-serif;
    font-size: 0.9rem;
    padding: 0.55rem 1.5rem;
    border: none;
    width: 100%;
    white-space: nowrap;
    transition: opacity 0.15s;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.85;
}
div[data-testid="stButton"] > button:disabled {
    background-color: var(--cl-border);
    color: var(--cl-text-secondary);
    opacity: 1;
}

/* ---- Documentos recuperados ---- */
[data-testid="stExpander"] {
    background-color: var(--cl-card-bg);
    border: 1px solid var(--cl-border);
    border-radius: 6px;
    margin-bottom: 4px;
}
[data-testid="stExpander"] summary p {
    font-size: 0.8rem !important;
    color: var(--cl-text-secondary) !important;
}
.cl-doc-content {
    max-height: 150px;
    overflow-y: auto;
    background-color: var(--cl-bg);
    color: var(--cl-text-secondary);
    border: 1px solid var(--cl-border);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
}
.cl-sources-title {
    font-family: var(--cl-mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cl-text-secondary);
    margin: 1.1rem 0 0.4rem 0;
}

/* ---- Spinner ---- */
div[data-testid="stSpinner"] p {
    color: var(--cl-text-secondary);
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
    ("last_query", ""),
    ("ingest_msg", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — ingestão de documentos
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📂 Adicionar Documento")
    # Mensagem de ingestão sobrevive ao st.rerun() — exibida e consumida aqui.
    pending_msg = st.session_state.ingest_msg
    if pending_msg:
        tipo, texto = pending_msg
        st.session_state.ingest_msg = None
        # Whitelist do tipo — evita dispatch dinâmico para atributo arbitrário de st.
        if tipo in _INGEST_MSG_TIPOS:
            getattr(st, tipo)(texto)
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

def _marcar_envio_por_enter() -> None:
    """Callback do text_input: Enter (ou blur com texto novo) sinaliza envio.

    Streamlit não tem evento 'Enter' fora de st.form; o on_change cobre esse caso
    sem usar st.form — incompatível com o harness AppTest (vaza form_id entre runs).
    """
    st.session_state.enviar_por_enter = True


col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Pergunta",
        placeholder="Faça uma pergunta sobre logística…",
        label_visibility="collapsed",
        key="pergunta_input",
        on_change=_marcar_envio_por_enter,
    )
with col_btn:
    enviar_clicado = st.button("Enviar")

# Envio dispara por clique no botão OU por Enter no campo (flag do on_change).
enviar = enviar_clicado or st.session_state.pop("enviar_por_enter", False)

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
            st.session_state.last_query = query

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            detail = _safe_detail(e.response)
            if code == 503:
                st.error(MSG_LMSTUDIO_DOWN if "LMStudio" in detail else MSG_VECTORDB_AUSENTE if "vetorial" in detail else MSG_ERRO_INTERNO)
            elif code == 422:
                st.error(MSG_ERRO_VALIDACAO)
            else:
                st.error(MSG_ERRO_INTERNO)
                if detail:
                    with st.expander("Detalhes do erro"):
                        st.code(detail)
        except httpx.ConnectError:
            st.error(MSG_CONNECT_ERROR)
        except httpx.TimeoutException:
            st.error(MSG_TIMEOUT)

elif enviar and not query.strip():
    st.warning("Digite uma pergunta antes de enviar.")

# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------

if st.session_state.ranked_response is None:
    st.markdown("""
    <div class="cl-empty-state">
        <span class="cl-empty-icon">📦</span>
        Faça uma pergunta para começar
        <div class="cl-empty-hints">
            <span class="cl-hint-chip">Qual o prazo médio de entrega?</span>
            <span class="cl-hint-chip">Como calcular custo de frete?</span>
            <span class="cl-hint-chip">O que é lead time?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.ranked_response is not None:
    resposta = st.session_state.ranked_response
    if isinstance(resposta, dict) and "answer" in resposta:
        resposta = resposta["answer"]

    confidence = float(st.session_state.confidence_score or 0.0)
    next_step = st.session_state.next_step or ""
    rota_label, rota_icon = _ROTAS.get(next_step, ("Desconhecida", "❓"))
    # rota_label/rota_icon vêm de _ROTAS (constante), mas escapamos por robustez estrutural.
    route_html = f'<span class="cl-badge">{rota_icon} {html.escape(str(rota_label))}</span>'

    # Divide a resposta em parágrafos (quebras em branco do modelo) e escapa HTML,
    # evitando os buracos verticais de pre-wrap e quebra de layout por caractere especial.
    paragrafos = re.split(r"\n\s*\n", str(resposta).strip())
    resposta_html = "".join(
        f'<p class="cl-para">{html.escape(p.strip())}</p>' for p in paragrafos if p.strip()
    )
    # Fallback para resposta vazia/whitespace — evita bolha invisível (sem texto algum).
    resposta_html = resposta_html or '<p class="cl-para"><em>(sem resposta)</em></p>'

    if st.session_state.last_query:
        st.markdown(
            f'<div class="cl-chat-row cl-user"><div class="cl-bubble cl-bubble-user">{html.escape(st.session_state.last_query)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div class="cl-chat-row cl-assistant">
        <div class="cl-bubble cl-bubble-assistant">
            {resposta_html}
            <div class="cl-meta">
                {route_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="cl-conf-label"><span>Confiança</span><span class="cl-conf-value">{confidence:.0%}</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(confidence, 0.0), 1.0))

    retrieved = st.session_state.retrieved_info or []
    if retrieved:
        st.markdown(f'<div class="cl-sources-title">Fontes ({len(retrieved)})</div>', unsafe_allow_html=True)
        for i, doc in enumerate(retrieved):
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            source_raw = meta.get("source", "Desconhecida")
            # Só o nome do arquivo, sem o caminho; cai para "Desconhecida" se ficar vazio.
            source = Path(source_raw).name or source_raw or "Desconhecida"
            # Prefixo com índice garante rótulo único (evita IDs de elemento duplicados no loop).
            with st.expander(f"{i + 1}. {source}"):
                conteudo = html.escape(str(doc.get("page_content", "")))
                st.markdown(f'<div class="cl-doc-content">{conteudo}</div>', unsafe_allow_html=True)
    else:
        st.write("Nenhum documento relacionado encontrado.")

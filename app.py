# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA

# Importa a biblioteca Streamlit para criação da interface web
import streamlit as st
import httpx
import anthropic

# Importa funções específicas do módulo Agentic RAG
from agenticlog import AgentState, agent_workflow

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
- Documentos, contratos e procedimentos complementares podem ser usados para aperfeiçoar o sistema de RAG (que nesse caso deve ser recriado com cada novo documento).
- IA Generativa comete erros. SEMPRE valide as respostas.
""")

# Cria botão "Suporte" na barra lateral e verifica se foi clicado
if st.sidebar.button("Suporte"):

    # Exibe informações de contato caso o botão seja clicado
    st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@aivoraq.com.br")

# Exibe título principal
st.title("AVK - Agência de IA")

# Exibe subtítulo secundário (substituindo o segundo st.title por st.subheader)
st.subheader("IA Generativa e Agentic RAG Para a Área de Logística")

# Solicita ao usuário que digite uma pergunta através de um campo de texto
query = st.text_input("Digite sua pergunta:")

# Mapeia os valores de next_step retornados pelo agente para rótulos legíveis em português exibidos na UI
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
            # Executa a consulta usando a função agent_workflow do módulo importado
            output = agent_workflow.invoke(AgentState(query=query))

            # agent_workflow.invoke() retorna um dict cujas chaves espelham os campos de AgentState;
            # lemos cada chave com .get() para garantir valor padrão seguro caso alguma esteja ausente
            st.session_state.ranked_response = output.get("ranked_response", "Nenhuma resposta.")
            st.session_state.confidence_score = output.get("confidence_score", 0.0)
            st.session_state.retrieved_info = output.get("retrieved_info", [])
            st.session_state.next_step = output.get("next_step", None)

        except Exception as e:
            _msg = str(e).lower()
            if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, anthropic.APIConnectionError)):
                st.error(
                    "LMStudio não está rodando. Inicie o LMStudio e carregue o modelo "
                    "hermes-3-llama-3.2-3b antes de usar o sistema."
                )
            elif "connection refused" in _msg or ("connect" in _msg and "1234" in _msg):
                st.error(
                    "LMStudio não está rodando. Inicie o LMStudio e carregue o modelo "
                    "hermes-3-llama-3.2-3b antes de usar o sistema."
                )
            elif "does not exist" in _msg or "no such file" in _msg:
                st.error(
                    "Base vetorial não encontrada. Execute o seguinte comando para criá-la:"
                    "\n\n    python -m agenticlog.rag"
                )
            else:
                st.error("Erro ao processar consulta. Verifique se o LMStudio está em execução.")
                with st.expander("Detalhes do erro"):
                    st.exception(e)

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

    # Obtém o score de confiança do session_state; usa 0.0 se None (ex.: após erro)
    confidence = st.session_state.confidence_score or 0.0

    # Exibe barra de progresso visual com o valor de confiança (0.0 a 1.0)
    st.progress(float(confidence))

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
            source = doc.metadata.get("source", "Desconhecida")

            # Exibe cada documento em um expander colapsável para melhor legibilidade
            with st.expander(f"Documento {i + 1} — {source}"):

                # Exibe o ID do documento
                st.markdown(f"**ID:** `{doc.id}`")

                # Exibe a fonte do documento
                st.markdown(f"**Fonte:** `{source}`")

                # Exibe o conteúdo do documento numa caixa de texto com altura definida
                # key único evita StreamlitDuplicateElementId quando há múltiplos documentos
                st.text_area("Conteúdo", doc.page_content, height=80, key=f"doc_content_{i}_{doc.id}")
    else:

        # Caso não haja documentos, exibe mensagem informando ao usuário
        st.write("Nenhum documento relacionado encontrado.")

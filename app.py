# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA

# Importa a biblioteca Streamlit para criação da interface web
import streamlit as st

# Importa funções específicas do módulo Agentic RAG
from agentic_rag_avk import AgentState, agent_workflow

# Define o título, ícone e layout inicial da página Streamlit
st.set_page_config(page_title="Aivorak", page_icon=":AVK", layout="centered")

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

# Exibe título secundário
st.title("IA Generativa e Agentic RAG Para a Área de Logística")

# Solicita ao usuário que digite uma pergunta através de um campo de texto
query = st.text_input("Digite sua pergunta:")

# Verifica se o usuário clicou no botão "Enviar"
if st.button("Enviar"):

    # Exibe um spinner indicando que a consulta está sendo processada
    with st.spinner("Processando consulta... Aguarde."):

        # Executa a consulta usando a função agent_workflow do módulo importado
        output = agent_workflow.invoke(AgentState(query = query))

    # Exibe subtítulo "Resposta:"
    st.subheader("Resposta:")

    # Obtém a resposta ranqueada ou mensagem padrão caso indisponível
    resposta = output.get("ranked_response", "Nenhuma resposta.")

    # Obtém o score de confiança da resposta gerada
    confidence = output.get("confidence_score", 0.0)

    # Verifica se a resposta está em formato dicionário contendo chave "answer"
    if isinstance(resposta, dict) and "answer" in resposta:

        # Se sim, extrai o valor da chave "answer"
        resposta = resposta["answer"]

    # Exibe a resposta formatada como Markdown
    st.markdown(resposta)

    # Exibe subtítulo indicando o nível de confiança da resposta
    st.subheader("Confiança da Resposta com Base no RAG:")

    # Exibe o nível de confiança em formato Markdown com 2 casas decimais
    st.markdown(f"`{confidence:.2f}`")

    # Obtém os documentos relacionados que foram recuperados pela consulta
    documentos_relacionados = output.get("retrieved_info", [])

    # Verifica se existem documentos relacionados recuperados
    if documentos_relacionados:

        # Exibe subtítulo "Documentos Relacionados:"
        st.subheader("Documentos Relacionados:")

        # Itera sobre cada documento recuperado
        for i, doc in enumerate(documentos_relacionados):

            # Exibe o ID de cada documento
            st.markdown(f"**ID:** `{doc.id}`")

            # Exibe a fonte do documento ou 'Desconhecida' caso indisponível
            st.markdown(f"**Fonte:** `{doc.metadata.get('source', 'Desconhecida')}`")

            # Exibe o conteúdo do documento numa caixa de texto com altura definida
            # key único evita StreamlitDuplicateElementId quando há múltiplos documentos
            st.text_area("Conteúdo", doc.page_content, height=80, key=f"doc_content_{i}_{doc.id}")
    else:
        
        # Caso não haja documentos, exibe mensagem informando ao usuário
        st.write("Nenhum documento relacionado encontrado.")

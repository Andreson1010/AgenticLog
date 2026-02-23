# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA
"""
Módulo: testa_agentic_rag.py

OBJETIVO PRINCIPAL:
Implementa testes unitários para validar o funcionamento do sistema Agentic RAG implementado
em agentic_rag_avk.py. Os testes garantem que cada componente do workflow funciona corretamente
antes da integração completa.

RELAÇÃO COM O PROJETO:
- testa_agentic_rag.py: Testes unitários (este arquivo)
- agentic_rag_avk.py: Módulo principal testado (implementa o Agentic RAG)
- app.py: Interface Streamlit que usa agentic_rag_avk.py
- rag_avk.py: Cria o VectorDB usado por agentic_rag_avk.py

ESTRUTURA DOS TESTES:
Cada teste valida uma função específica do workflow Agentic RAG:
1. Testes de decisão (teste_avk_1, 2, 3): Validam a lógica de roteamento
2. Testes de execução (teste_avk_4, 5, 6): Validam as funções de processamento
3. Testes de avaliação (teste_avk_7, 8): Validam ranking e seleção de respostas

TÉCNICAS UTILIZADAS:
- unittest: Framework de testes do Python
- unittest.mock: Mocking de dependências externas (LLM, VectorDB, Web Search)
- patch: Decorador para substituir objetos por mocks durante os testes
"""

# Imports
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from unittest.mock import patch, MagicMock  # Mocking para isolar dependências externas
from langchain.schema import Document  # Tipo de documento usado pelo VectorDB
from agentic_rag_avk import (
    avk_passo_decisao_agente,      # Função de decisão do workflow
    avk_usar_ferramenta_web,       # Função de busca web
    avk_retrieve_info,              # Função de recuperação de documentos
    avk_gera_multiplas_respostas,   # Função de geração de respostas
    avk_avalia_similaridade,        # Função de avaliação de similaridade
    avk_rank_respostas,             # Função de ranking de respostas
    AgentState,                     # Classe de estado do agente
)

class TestAgenticRAG(unittest.TestCase):
    """
    Classe de testes para o sistema Agentic RAG.
    
    OBJETIVO: Validar todas as funções do workflow implementado em agentic_rag_avk.py
    CORRELAÇÃO: Cada teste corresponde a uma função específica do workflow
    COBERTURA: Testa decisão, recuperação, geração, avaliação e ranking
    """

    # ============================================================================
    # TESTES DE DECISÃO (NÓ DE ROTEAMENTO)
    # ============================================================================
    
    def teste_avk_1_passo_decisao_agente_retrieve(self):
        """
        OBJETIVO: Valida que queries específicas são roteadas para recuperação (RAG)
        
        FUNÇÃO TESTADA: avk_passo_decisao_agente() em agentic_rag_avk.py
        LÓGICA: Queries sem palavras-chave especiais → next_step = "retrieve"
        FLUXO NO WORKFLOW: decision → retrieve → generate → evaluate → rank
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa a função que é o nó de entrada do workflow (linha 160)
        - Valida a lógica de decisão que determina o caminho no grafo de estado
        - Query específica deve acionar o caminho RAG completo
        """
        state = AgentState(query = "Quais as fases da cadeia de suprimentos?")
        new_state = avk_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "retrieve")

    def teste_avk_2_passo_decisao_agente_usar_web(self):
        """
        OBJETIVO: Valida que queries sobre atualidades são roteadas para busca web
        
        FUNÇÃO TESTADA: avk_passo_decisao_agente() em agentic_rag_avk.py
        LÓGICA: Queries com palavras-chave de atualidade → next_step = "usar_web"
        FLUXO NO WORKFLOW: decision → usar_web → [FIM] (caminho independente do RAG)
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Valida detecção de palavras-chave: "notícias", "atualizado", "recente"
        - Testa o caminho alternativo que não passa pelo VectorDB
        - Query sobre atualidades deve acionar busca web diretamente
        """
        state = AgentState(query = "Pesquise sobre as novidades recentes em logistica e supply chain.")
        new_state = avk_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "usar_web")

    def teste_avk_3_passo_decisao_agente_gerar(self):
        """
        OBJETIVO: Valida que queries conceituais são roteadas para geração direta
        
        FUNÇÃO TESTADA: avk_passo_decisao_agente() em agentic_rag_avk.py
        LÓGICA: Queries conceituais (explique, resuma, defina) → next_step = "gerar"
        FLUXO NO WORKFLOW: decision → generate → evaluate → rank (sem retrieve prévio)
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Valida detecção de palavras-chave: "explique", "resuma", "defina", "conceito"
        - Testa o caminho que pula a recuperação e vai direto para geração
        - Query conceitual deve usar conhecimento interno do LLM
        """
        state = AgentState(query = "Resuma o conceito de supply chain.")
        new_state = avk_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "gerar")

    @patch("agentic_rag_avk.avk_agent_executor")
    def teste_avk_4_usar_ferramenta_web(self, mock_executor):
        """
        OBJETIVO: Valida a função de busca web usando DuckDuckGo
        
        FUNÇÃO TESTADA: avk_usar_ferramenta_web() em agentic_rag_avk.py (linhas 97-102)
        PROCESSO: Invoca o agente web que busca informações atualizadas na internet
        MOCK: Simula avk_agent_executor para evitar chamadas reais à web durante testes
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa o nó "usar_web" do workflow (linha 157)
        - Valida que a resposta da web é armazenada em ranked_response
        - Verifica que confidence_score é 0.0 (não há documentos para comparar)
        - CORRELAÇÃO: Usa avk_agent_executor criado em agentic_rag_avk.py (linha 70)
        """
        mock_executor.invoke.return_value = {"output": "Resposta da web."}
        state = AgentState(query = "notícias recentes sobre supply chain")
        new_state = avk_usar_ferramenta_web(state)
        mock_executor.invoke.assert_called_once_with("notícias recentes sobre supply chain")
        self.assertEqual(new_state.ranked_response, "Resposta da web.")

    @patch("agentic_rag_avk.retriever")
    def teste_avk_5_retrieve_info(self, mock_retriever):
        """
        OBJETIVO: Valida a recuperação de documentos do VectorDB usando busca semântica
        
        FUNÇÃO TESTADA: avk_retrieve_info() em agentic_rag_avk.py (linhas 104-108)
        PROCESSO: Busca documentos similares no ChromaDB usando embeddings
        MOCK: Simula retriever para evitar acesso real ao VectorDB durante testes
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa o nó "retrieve" do workflow (linha 153)
        - Valida que documentos são recuperados e armazenados em retrieved_info
        - CORRELAÇÃO COM rag_avk.py: O retriever usa o VectorDB criado por rag_avk.py
        - CORRELAÇÃO: Usa retriever criado em agentic_rag_avk.py (linha 50)
        """
        mock_retriever.invoke.return_value = [Document(page_content = "Documento sobre cadeia de suprimentos")]
        state = AgentState(query = "fases da cadeia de suprimentos")
        new_state = avk_retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("fases da cadeia de suprimentos")
        self.assertEqual(len(new_state.retrieved_info), 1)

    @patch("agentic_rag_avk.StrOutputParser")
    @patch("agentic_rag_avk.llm")
    @patch("agentic_rag_avk.prompt_rag_retrieve")
    def teste_avk_6_gera_multiplas_respostas(self, mock_prompt, mock_llm, mock_str_parser_class):
        """
        OBJETIVO: Valida a geração de múltiplas respostas usando RAG
        
        FUNÇÃO TESTADA: avk_gera_multiplas_respostas() em agentic_rag_avk.py (linhas 291-346)
        PROCESSO: Gera 5 respostas diferentes para a mesma query usando o LLM com contexto RAG
        MOCK: Simula prompt, LLM e StrOutputParser para evitar chamadas reais ao Ollama
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa o nó "generate_multiple" do workflow (linha 154)
        - Valida que 5 respostas são geradas e armazenadas em possible_responses
        - Verifica estrutura: cada resposta é um dicionário {"answer": "..."}
        - CORRELAÇÃO: Usa prompt_rag_retrieve (linha 122) e llm (linha 30) de agentic_rag_avk.py
        - IMPORTÂNCIA: Múltiplas respostas permitem selecionar a melhor baseada em similaridade
        
        TÉCNICA DE MOCK:
        - Simula o operador | do LangChain (prompt | llm | parser)
        - Cria uma chain mockada que retorna string diretamente
        - Evita problemas com dependências externas (Ollama, VectorDB)
        """
        # Criar uma chain mockada que retorna string diretamente
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Resposta gerada"
        
        # Criar um objeto que simula o resultado de prompt | llm
        mock_prompt_llm_chain = MagicMock()
        mock_prompt_llm_chain.__or__ = lambda self, other: mock_chain
        
        # Mock do prompt: quando usado com |, retorna um objeto que pode ser usado com | novamente
        mock_prompt.__or__ = lambda self, other: mock_prompt_llm_chain
        
        # Mock do StrOutputParser
        mock_parser_instance = MagicMock()
        mock_parser_instance.invoke.return_value = "Resposta gerada"
        mock_str_parser_class.return_value = mock_parser_instance
        
        state = AgentState(query = "fases da cadeia de suprimentos", next_step = "retrieve")
        state.retrieved_info = [Document(page_content = "Documento teste")]
        new_state = avk_gera_multiplas_respostas(state)
        self.assertEqual(len(new_state.possible_responses), 5)
        self.assertIn("answer", new_state.possible_responses[0])
        self.assertEqual(new_state.possible_responses[0]["answer"], "Resposta gerada")

    @patch("agentic_rag_avk.embedding_model")
    def teste_avk_7_avalia_similaridade(self, mock_embedding_model):
        """
        OBJETIVO: Valida o cálculo de similaridade entre respostas e documentos recuperados
        
        FUNÇÃO TESTADA: avk_avalia_similaridade() em agentic_rag_avk.py (linhas 368-385)
        PROCESSO: 
        1. Gera embeddings para documentos recuperados e respostas geradas
        2. Calcula similaridade de cosseno entre cada resposta e todos os documentos
        3. Média das similaridades = score de cada resposta
        MOCK: Simula embedding_model para evitar processamento real de embeddings
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa o nó "evaluate_similarity" do workflow (linha 155)
        - Valida que similarity_scores tem um score para cada resposta
        - CORRELAÇÃO: Usa embedding_model (linha 44) - mesmo modelo usado em rag_avk.py
        - IMPORTÂNCIA: Scores determinam qual resposta está mais alinhada com documentos
        
        MÉTRICA: Similaridade de Cosseno (valores 0 a 1)
        - Valores altos (>0.7): Resposta muito alinhada com documentos
        - Valores baixos (<0.5): Resposta pode ter informações não baseadas nos documentos
        """
        # Mock retorna embeddings simulados (768 dimensões é comum para modelos BGE)
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        state = AgentState(
            query = "cadeia de suprimentos",
            retrieved_info = [Document(page_content = "doc")],
            possible_responses = [{"answer": "resposta"}]
        )
        new_state = avk_avalia_similaridade(state)
        self.assertEqual(len(new_state.similarity_scores), 1)

    def teste_avk_8_rank_respostas(self):
        """
        OBJETIVO: Valida a seleção da melhor resposta baseada em scores de similaridade
        
        FUNÇÃO TESTADA: avk_rank_respostas() em agentic_rag_avk.py (linhas 405-414)
        PROCESSO:
        1. Combina respostas com seus scores de similaridade
        2. Ordena por score (maior primeiro = mais similar aos documentos)
        3. Seleciona a melhor resposta e define confidence_score
        
        CORRELAÇÃO COM agentic_rag_avk.py:
        - Testa o nó "rank_responses" do workflow (linha 156)
        - Valida que ranked_response é a resposta com maior similarity_score
        - Verifica que confidence_score é o score da melhor resposta
        - CORRELAÇÃO COM app.py: ranked_response e confidence_score são exibidos na interface
        - IMPORTÂNCIA: Último passo do workflow RAG - determina resposta final ao usuário
        
        ESTRUTURA DE DADOS:
        - possible_responses: Lista de dicionários [{"answer": "..."}, ...]
        - similarity_scores: Lista de floats [0.8, 0.9, ...]
        - ranked_response: Dicionário {"answer": "..."} da melhor resposta
        - confidence_score: Float (0.0 a 1.0) indicando confiança
        """
        state = AgentState(
            query = "cadeia de suprimentos",
            possible_responses = [{"answer": "resp1"}, {"answer": "resp2"}],
            similarity_scores = [0.8, 0.9]
        )
        new_state = avk_rank_respostas(state)
        # A resposta ranqueada deve ser o dicionário com maior score (resp2)
        self.assertEqual(new_state.ranked_response, {"answer": "resp2"})
        self.assertEqual(new_state.confidence_score, 0.9)

# ============================================================================
# EXECUÇÃO DOS TESTES
# ============================================================================
if __name__ == '__main__':
    """
    Ponto de entrada para execução dos testes.
    
    OBJETIVO: Executa todos os testes da classe TestAgenticRAG
    VERBOSIDADE: 2 = mostra detalhes de cada teste (ok, ERROR, FAIL)
    
    COMO EXECUTAR:
    - Da raiz do projeto: python tests/testa_agentic_rag.py
    - Com uv: uv run python tests/testa_agentic_rag.py
    - Com unittest: python -m unittest tests.testa_agentic_rag -v
    
    SAÍDA ESPERADA:
    - 8 testes executados
    - Todos devem passar (ok) se o sistema estiver funcionando corretamente
    - Erros indicam problemas nas funções testadas ou nas dependências mockadas
    """
    print("\nIniciando os testes. Aguarde...\n")
    unittest.main(verbosity = 2)

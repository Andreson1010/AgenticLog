# Projeto 8 - Pipeline de Automação de Testes em Módulo de RAG Para Aplicações de IA
"""
Este módulo implementa um pipeline completo para criação de um sistema RAG (Retrieval-Augmented Generation).
O objetivo principal é processar documentos JSON, gerar embeddings vetoriais e criar um banco de dados
vetorial que permite busca semântica eficiente para aplicações de IA.
"""

# Biblioteca para manipulação de arquivos JSON
import json 

# Função para carregar o modelo de embeddings pré-treinado
from langchain_huggingface import HuggingFaceEmbeddings  

# Banco de dados vetorial - ChromaDB para armazenamento e busca vetorial
from langchain_chroma import Chroma  

# Carregadores de documentos - DirectoryLoader para carregar múltiplos arquivos e JSONLoader para processar JSON
from langchain_community.document_loaders import DirectoryLoader, JSONLoader 

# Divisor de textos em chunks - RecursiveCharacterTextSplitter para dividir documentos em partes menores
from langchain.text_splitter import RecursiveCharacterTextSplitter  

# Definição do diretório onde os dados de origem estão armazenados
diretorio_dados = "documentos"

# Definição do diretório onde o banco de dados vetorial será armazenado
diretorio_vectordb = "vectordb"

# Variável global para armazenar a instância do banco de dados vetorial
vectordb = None

def func_cria_vectordb():
    """
    Objetivo: Cria um banco de dados vetorial (VectorDB) a partir de documentos JSON processados.
    
    Esta função implementa o pipeline completo de RAG:
    1. Carrega documentos JSON de um diretório específico
    2. Divide os documentos em chunks menores para melhor processamento
    3. Gera embeddings vetoriais usando um modelo pré-treinado (BAAI/bge-base-en)
    4. Armazena os embeddings em um banco de dados vetorial (ChromaDB) para busca semântica
    
    O banco de dados vetorial criado permite realizar buscas por similaridade semântica,
    essencial para sistemas de RAG que recuperam informações relevantes antes de gerar respostas.
    
    Fluxo de execução:
    - Carrega todos os arquivos JSON do diretório 'documentos'
    - Transforma os JSONs em documentos estruturados usando jq_schema
    - Divide os documentos em chunks de 500 caracteres com sobreposição de 50 caracteres
    - Gera embeddings normalizados usando o modelo BAAI/bge-base-en
    - Persiste o banco de dados vetorial no diretório 'vectordb' para uso posterior
    
    Retorna: None (a variável global 'vectordb' é atualizada com a instância do ChromaDB)
    """
    global vectordb  # Permite modificar a variável global
    
    # Mensagem informando o início do processo
    print("\nGerando as Embeddings. Aguarde...")
    
    # Definição do esquema de conversão para o JSONLoader
    # Este esquema jq transforma objetos JSON em strings formatadas: "chave: valor"
    # Objetivo: Converter estruturas JSON complexas em texto plano para processamento
    jq_schema = 'to_entries | map(.key + ": " + .value) | join("\\n")'
    
    # Carregamento dos arquivos JSON do diretório especificado
    # Objetivo: Carregar todos os arquivos .json do diretório e transformá-los em documentos
    loader = DirectoryLoader(
        diretorio_dados,                          # Diretório onde os arquivos JSON de origem estão armazenados
        glob = "*.json",                          # Padrão de arquivos a serem carregados (todos os .json)
        loader_cls = JSONLoader,                  # Classe de carregamento de JSON
        loader_kwargs = {"jq_schema": jq_schema}  # Configuração para transformar os dados JSON em texto
    )
    
    # Carrega os documentos a partir do diretório
    # Objetivo: Obter uma lista de objetos Document do LangChain prontos para processamento
    documents = loader.load()
    
    # Verifica se há documentos carregados, caso contrário, encerra a função
    # Objetivo: Validação para evitar processamento desnecessário quando não há dados
    if not documents:
        print("Nenhum documento encontrado.")  
        return  
    
    # Define um divisor de texto para segmentar os documentos em partes menores
    # Objetivo: Dividir documentos longos em chunks menores para:
    # - Melhorar a precisão dos embeddings
    # - Permitir busca mais granular
    # - Otimizar o uso de memória e processamento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,   # Define o tamanho máximo de cada chunk (500 caracteres)
        chunk_overlap = 50  # Define a sobreposição entre os chunks para manter contexto entre segmentos
    )
    
    # Divide os documentos em chunks menores
    # Objetivo: Criar uma lista de chunks que serão convertidos em embeddings individuais
    chunks = text_splitter.split_documents(documents)
    
    # Nome do modelo de embeddings utilizado
    # Modelo: BAAI/bge-base-en - modelo de embeddings bilíngue (inglês) otimizado para busca semântica
    # https://huggingface.co/BAAI/bge-base-en
    model_name = "BAAI/bge-base-en"
    
    # Parâmetros para a geração dos embeddings
    # Objetivo: Normalizar os embeddings para cálculo de similaridade usando produto escalar
    # normalize_embeddings=True garante que os vetores tenham norma unitária, otimizando buscas por similaridade
    encode_kwargs = {'normalize_embeddings': True}  
    
    # Instancia o modelo de embeddings
    # Objetivo: Criar o objeto que irá converter texto em vetores numéricos (embeddings)
    # Esses vetores capturam o significado semântico do texto para busca por similaridade
    # device: usa GPU se PyTorch com CUDA estiver instalado, senão CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,           # Modelo escolhido para gerar embeddings
        model_kwargs = {'device': device},  # GPU: 'cuda' | CPU: 'cpu' (detecta automaticamente)
        encode_kwargs = encode_kwargs      # Configuração dos embeddings (normalização)
    )
    
    # Criação do banco de dados vetorial a partir dos documentos processados
    # Objetivo: Criar e persistir o ChromaDB com todos os chunks e seus embeddings
    # O ChromaDB permite busca eficiente por similaridade vetorial (busca semântica)
    # persist_directory garante que o banco seja salvo em disco para uso posterior sem reprocessamento
    vectordb = Chroma.from_documents(
        chunks,                                     # Chunks gerados a partir dos documentos
        embedding_model,                            # Modelo de embeddings utilizado para converter texto em vetores
        persist_directory = diretorio_vectordb     # Diretório onde o banco de dados vetorial será armazenado permanentemente
    )
    
    # Mensagem informando que o banco de dados vetorial foi criado com sucesso
    print("\nBanco de Dados Vetorial do RAG Criado com Sucesso.\n")

# Chamada da função para gerar o banco de dados vetorial quando o script é executado diretamente
# Objetivo: Permitir execução do script como programa principal para criar o VectorDB
if __name__ == "__main__":
    func_cria_vectordb()







# RAG e suas Variações: Guia Completo

## Introdução

O **Retrieval-Augmented Generation (RAG)** representa uma das maiores inovações em processamento de linguagem natural e inteligência artificial generativa. Esta tecnologia combina a capacidade de recuperação de informações com a geração de texto por modelos de linguagem, criando sistemas mais precisos, atualizados e confiáveis.

Neste documento, exploraremos as técnicas avançadas de RAG que estão moldando o futuro da IA conversacional, chatbots inteligentes e sistemas de busca semântica. Abordaremos implementações práticas de **CRAG**, **CAG**, **Graph RAG**, **Agentic RAG**, **Adaptive RAG**, **Multi Modal RAG** e **W-RAG**.

---

## O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma arquitetura que combina:

1. **Retrieval (Recuperação)**: Busca informações relevantes de uma base de conhecimento externa
2. **Augmentation (Aumento)**: Enriquece o contexto do modelo de linguagem com essas informações
3. **Generation (Geração)**: Produz respostas baseadas no contexto recuperado e no conhecimento pré-treinado do modelo

### Vantagens do RAG

- **Atualização de conhecimento**: Permite que modelos acessem informações atualizadas sem retreinamento
- **Redução de alucinações**: Baseia respostas em documentos verificáveis
- **Transparência**: Permite rastreabilidade das fontes de informação
- **Especialização**: Adapta-se a domínios específicos sem modificar o modelo base

---

## Top-K: Conceito Fundamental em RAG

### O que é Top-K?

**Top-K** é um dos parâmetros mais importantes em sistemas RAG. Ele define quantos documentos (ou chunks) mais relevantes serão recuperados do banco vetorial e passados como contexto para o modelo de linguagem.

### Como Funciona?

1. **Geração de Embedding**: A query do usuário é convertida em um vetor (embedding)
2. **Busca por Similaridade**: O sistema calcula a similaridade entre o embedding da query e todos os embeddings dos documentos no banco vetorial
3. **Ranking**: Os documentos são ordenados por similaridade (do mais relevante para o menos relevante)
4. **Seleção Top-K**: Apenas os **K** documentos com maior similaridade são selecionados
5. **Contexto para o LLM**: Esses K documentos são concatenados e enviados como contexto para o modelo gerar a resposta

### Exemplo Prático

Imagine que você tem 10.000 documentos em sua base de conhecimento:

- Com **K=3**: Apenas os 3 documentos mais relevantes são recuperados
- Com **K=10**: Os 10 documentos mais relevantes são recuperados
- Com **K=50**: Os 50 documentos mais relevantes são recuperados

### Impacto do Valor de K

#### K Muito Baixo (K=1 a K=3)
**Vantagens**:
- ✅ Menor custo computacional (menos tokens)
- ✅ Respostas mais rápidas
- ✅ Menor risco de incluir informações irrelevantes
- ✅ Ideal para queries muito específicas

**Desvantagens**:
- ❌ Pode perder contexto importante
- ❌ Risco de informações incompletas
- ❌ Dificuldade em responder queries complexas que requerem múltiplas fontes

**Quando Usar**:
- Queries muito específicas e diretas
- Quando você tem alta confiança na qualidade dos embeddings
- Sistemas com limitações de tokens ou custo
- Bases de conhecimento muito bem organizadas e específicas

#### K Moderado (K=4 a K=10)
**Vantagens**:
- ✅ Balance entre contexto e precisão
- ✅ Boa cobertura de informações relacionadas
- ✅ Funciona bem para a maioria dos casos de uso
- ✅ Permite síntese de múltiplas perspectivas

**Desvantagens**:
- ⚠️ Pode incluir alguns documentos marginalmente relevantes
- ⚠️ Custo computacional moderado

**Quando Usar**:
- **Recomendado para a maioria dos casos**
- Queries de complexidade média
- Quando você precisa de contexto suficiente sem sobrecarregar o LLM
- Aplicações gerais de Q&A

#### K Alto (K=11 a K=50+)
**Vantagens**:
- ✅ Máxima cobertura de contexto
- ✅ Melhor para queries complexas e abrangentes
- ✅ Reduz risco de perder informações importantes
- ✅ Ideal para síntese de múltiplas fontes

**Desvantagens**:
- ❌ Maior custo computacional (muitos tokens)
- ❌ Risco de incluir informações irrelevantes ou ruído
- ❌ Pode confundir o LLM com contexto excessivo
- ❌ Respostas mais lentas

**Quando Usar**:
- Queries muito complexas que requerem múltiplas perspectivas
- Quando você precisa de síntese abrangente
- Sistemas de pesquisa acadêmica ou análise profunda
- Quando a precisão máxima é mais importante que a velocidade

### Fatores que Influenciam a Escolha de K

1. **Tamanho dos Chunks**:
   - Chunks pequenos (100-300 tokens) → K maior (10-20)
   - Chunks médios (500-1000 tokens) → K moderado (5-10)
   - Chunks grandes (1000+ tokens) → K menor (3-5)

2. **Limite de Tokens do LLM**:
   - Modelos com contexto limitado → K menor
   - Modelos com contexto grande (GPT-4, Claude) → K pode ser maior

3. **Complexidade das Queries**:
   - Queries simples → K menor (3-5)
   - Queries complexas → K maior (10-20)

4. **Qualidade dos Embeddings**:
   - Embeddings de alta qualidade → K menor pode ser suficiente
   - Embeddings de qualidade média → K maior compensa

5. **Natureza do Conhecimento**:
   - Conhecimento específico e focado → K menor
   - Conhecimento amplo e interconectado → K maior

### Técnicas Avançadas com Top-K

#### 1. Top-K Adaptativo
Ajusta K dinamicamente baseado na complexidade da query:
- Queries simples → K=3
- Queries médias → K=7
- Queries complexas → K=15

#### 2. Top-K com Threshold
Recupera K documentos, mas filtra por um threshold de similaridade:
- Recupera até K documentos, mas apenas se similaridade > 0.7
- Evita incluir documentos irrelevantes mesmo que sejam "top-K"

#### 3. Top-K Hierárquico
- Primeiro recupera K=20 documentos
- Depois re-rankeia e seleciona top K=5 para o LLM
- Combina recall (encontrar tudo relevante) com precisão (apenas o melhor)

#### 4. Top-K por Múltiplas Queries
Para queries complexas:
- Decompõe em sub-queries
- Recupera top-K para cada sub-query
- Combina e deduplica os resultados

### Exemplo de Implementação

```python
# Exemplo básico
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Exemplo com threshold
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 10,
        "score_threshold": 0.7  # Apenas documentos com similaridade > 0.7
    }
)

# Exemplo adaptativo
def get_adaptive_k(query):
    if len(query.split()) < 5:
        return 3  # Query simples
    elif len(query.split()) < 15:
        return 7  # Query média
    else:
        return 15  # Query complexa
```

### Recomendações Práticas

1. **Comece com K=5 a K=10**: É um bom ponto de partida para a maioria dos casos
2. **Teste e Ajuste**: Meça a qualidade das respostas com diferentes valores de K
3. **Monitore Custos**: K maior = mais tokens = maior custo
4. **Considere o Tamanho do Contexto**: Não exceda o limite de tokens do seu LLM
5. **Use Métricas**: Avalie precisão, recall e qualidade das respostas
6. **Implemente Adaptação**: Considere K adaptativo para diferentes tipos de queries

### Resumo

| K | Uso Recomendado | Custo | Precisão | Cobertura |
|---|-----------------|-------|----------|-----------|
| **1-3** | Queries muito específicas | Baixo | Alta | Baixa |
| **4-7** | Uso geral, queries médias | Médio | Alta | Média |
| **8-15** | Queries complexas | Médio-Alto | Média-Alta | Alta |
| **16+** | Análise profunda, síntese | Alto | Média | Muito Alta |

**Regra de Ouro**: O valor ideal de K depende do seu caso de uso específico. Teste diferentes valores e monitore métricas de qualidade para encontrar o equilíbrio perfeito entre precisão, cobertura e custo.

---

## Variações de RAG

### 1. RAG Tradicional (Naive RAG)

**Descrição**: A implementação mais básica, onde documentos são recuperados e passados diretamente para o LLM.

**Fluxo**:
1. Query do usuário → Embedding
2. Busca por similaridade no banco vetorial
3. Recuperação dos top-K documentos
4. Geração da resposta com contexto recuperado

**Cenários de Uso**:
- ✅ Aplicações simples de Q&A sobre documentos
- ✅ Chatbots com base de conhecimento estática
- ✅ Sistemas de busca semântica básicos
- ✅ Quando a precisão máxima não é crítica

**Limitações**:
- Não valida a qualidade dos documentos recuperados
- Pode incluir informações irrelevantes
- Não adapta a estratégia de busca

---

### 2. CRAG (Corrective RAG)

**Descrição**: Sistema que avalia criticamente os documentos recuperados e corrige a busca quando necessário.

**Características**:
- **Validação de Relevância**: Avalia se os documentos recuperados são realmente relevantes
- **Correção Automática**: Refina a query e busca novamente se necessário
- **Filtragem Inteligente**: Remove documentos de baixa qualidade

**Fluxo**:
1. Recuperação inicial de documentos
2. Avaliação de relevância (relevance score)
3. Se relevância baixa → correção da query e nova busca
4. Filtragem de documentos irrelevantes
5. Geração da resposta apenas com documentos validados

**Cenários de Uso**:
- ✅ Sistemas onde a precisão é crítica (médico, jurídico, financeiro)
- ✅ Bases de conhecimento grandes e heterogêneas
- ✅ Quando há risco de informações conflitantes
- ✅ Aplicações que requerem alta confiabilidade

**Vantagens**:
- Maior precisão nas respostas
- Redução de informações incorretas
- Melhor qualidade geral do sistema

---

### 3. CAG (Corrective Augmented Generation)

**Descrição**: Similar ao CRAG, mas com foco em corrigir e aumentar o contexto antes da geração.

**Características**:
- **Correção de Contexto**: Identifica e corrige inconsistências no contexto recuperado
- **Aumento Seletivo**: Adiciona informações complementares quando necessário
- **Validação Cruzada**: Compara informações de múltiplas fontes

**Cenários de Uso**:
- ✅ Sistemas de pesquisa acadêmica
- ✅ Análise de documentos técnicos complexos
- ✅ Quando é necessário cruzar informações de múltiplas fontes
- ✅ Aplicações de verificação de fatos

---

### 4. Graph RAG

**Descrição**: Utiliza grafos de conhecimento para representar relacionamentos entre entidades e conceitos.

**Características**:
- **Representação em Grafo**: Entidades e relacionamentos são modelados como nós e arestas
- **Busca por Relacionamentos**: Explora conexões semânticas entre conceitos
- **Traversal Inteligente**: Navega pelo grafo para encontrar informações relacionadas

**Fluxo**:
1. Query → Identificação de entidades
2. Busca no grafo de conhecimento
3. Traversal de relacionamentos relevantes
4. Agregação de contexto relacionado
5. Geração da resposta

**Cenários de Uso**:
- ✅ Sistemas de conhecimento corporativo com relacionamentos complexos
- ✅ Análise de redes sociais e conexões
- ✅ Bases de conhecimento com hierarquias e taxonomias
- ✅ Quando relacionamentos entre entidades são importantes
- ✅ Sistemas de recomendação baseados em relacionamentos

**Vantagens**:
- Captura relacionamentos complexos
- Permite raciocínio sobre conexões
- Ideal para conhecimento estruturado

---

### 5. Agentic RAG

**Descrição**: Combina RAG com agentes de IA que tomam decisões autônomas sobre quando e como buscar informações.

**Características**:
- **Agentes Autônomos**: Decisões inteligentes sobre estratégia de busca
- **Múltiplas Ferramentas**: Integra busca vetorial, web search, APIs, etc.
- **Raciocínio Sequencial**: Planeja e executa múltiplos passos
- **Adaptação Dinâmica**: Ajusta estratégia baseado nos resultados

**Fluxo**:
1. Query do usuário
2. Agente decide: RAG interno ou busca externa?
3. Executa busca apropriada
4. Avalia resultados e decide próximos passos
5. Combina informações de múltiplas fontes
6. Gera resposta final

**Cenários de Uso**:
- ✅ Sistemas que precisam de informações atualizadas (notícias, preços, eventos)
- ✅ Assistentes virtuais inteligentes
- ✅ Sistemas de pesquisa que combinam conhecimento interno e externo
- ✅ Aplicações que requerem raciocínio multi-passo
- ✅ Quando é necessário buscar em múltiplas fontes

**Vantagens**:
- Flexibilidade máxima
- Acesso a informações em tempo real
- Capacidade de raciocínio complexo

**Exemplo de Uso no Projeto**:
Este projeto implementa Agentic RAG, combinando:
- Busca vetorial em documentos internos (ChromaDB)
- Busca web (DuckDuckGo) para informações atualizadas
- Agente que decide qual estratégia usar baseado na query

---

### 6. Adaptive RAG

**Descrição**: Sistema que adapta dinamicamente a estratégia de recuperação baseado na complexidade da query.

**Características**:
- **Classificação de Complexidade**: Avalia se a query é simples ou complexa
- **Estratégias Adaptativas**: 
  - Queries simples → RAG direto
  - Queries complexas → Decomposição em sub-queries
- **Otimização Automática**: Ajusta parâmetros de busca dinamicamente

**Fluxo**:
1. Análise da complexidade da query
2. Seleção da estratégia apropriada:
   - **Simples**: RAG direto
   - **Média**: RAG com refinamento
   - **Complexa**: Decomposição + múltiplas buscas + síntese
3. Execução da estratégia selecionada
4. Geração da resposta

**Cenários de Uso**:
- ✅ Sistemas com grande variabilidade na complexidade das queries
- ✅ Aplicações que precisam balancear velocidade e precisão
- ✅ Quando há necessidade de otimização de custos (tokens)
- ✅ Sistemas que atendem usuários com diferentes níveis de expertise

**Vantagens**:
- Eficiência otimizada
- Melhor experiência do usuário
- Redução de custos computacionais

---

### 7. Multi Modal RAG

**Descrição**: Estende RAG para processar e recuperar informações de múltiplos tipos de mídia (texto, imagens, áudio, vídeo).

**Características**:
- **Processamento Multimodal**: Embeddings para texto, imagens, áudio
- **Busca Unificada**: Busca semântica através de múltiplas modalidades
- **Geração Multimodal**: Respostas que podem incluir diferentes tipos de conteúdo

**Fluxo**:
1. Query (pode ser texto, imagem, ou ambos)
2. Geração de embeddings multimodais
3. Busca em bases de conhecimento multimodais
4. Recuperação de documentos relevantes (qualquer modalidade)
5. Geração de resposta multimodal

**Cenários de Uso**:
- ✅ Sistemas de busca de imagens por descrição
- ✅ Análise de documentos com imagens e texto
- ✅ Assistentes que processam screenshots ou fotos
- ✅ Sistemas de e-commerce com busca visual
- ✅ Aplicações educacionais com conteúdo multimídia
- ✅ Análise de vídeos e transcrições

**Vantagens**:
- Riqueza de informação
- Experiência mais natural
- Aproveitamento completo do conteúdo disponível

---

### 8. W-RAG (Web RAG)

**Descrição**: Especializado em recuperação de informações da web em tempo real.

**Características**:
- **Busca Web em Tempo Real**: Acessa informações atualizadas da internet
- **Validação de Fontes**: Avalia confiabilidade das fontes web
- **Agregação de Múltiplas Fontes**: Combina informações de vários sites
- **Filtragem de Ruído**: Remove informações irrelevantes ou duplicadas

**Fluxo**:
1. Query do usuário
2. Busca web (Google, Bing, DuckDuckGo, etc.)
3. Extração e processamento de conteúdo relevante
4. Validação e filtragem de fontes
5. Geração de embeddings e armazenamento temporário
6. Geração da resposta com citações

**Cenários de Uso**:
- ✅ Sistemas de notícias e atualizações em tempo real
- ✅ Assistentes que precisam de informações atuais (preços, eventos, clima)
- ✅ Pesquisa de mercado e análise competitiva
- ✅ Sistemas de verificação de fatos
- ✅ Quando o conhecimento interno não é suficiente

**Vantagens**:
- Informações sempre atualizadas
- Acesso a vasto conhecimento público
- Complementa bases de conhecimento internas

---

## Comparação Rápida

| Variação | Complexidade | Precisão | Atualização | Uso de Recursos |
|----------|--------------|----------|-------------|-----------------|
| **RAG Tradicional** | Baixa | Média | Estática | Baixo |
| **CRAG** | Média | Alta | Estática | Médio |
| **CAG** | Média-Alta | Alta | Estática | Médio-Alto |
| **Graph RAG** | Alta | Média-Alta | Estática | Alto |
| **Agentic RAG** | Muito Alta | Alta | Dinâmica | Muito Alto |
| **Adaptive RAG** | Alta | Alta | Estática/Dinâmica | Médio-Alto |
| **Multi Modal RAG** | Muito Alta | Alta | Estática | Muito Alto |
| **W-RAG** | Média-Alta | Média-Alta | Tempo Real | Médio-Alto |

---

## Recomendações de Escolha

### Escolha **RAG Tradicional** quando:
- Você tem uma base de conhecimento estática e bem organizada
- As queries são relativamente simples
- Você precisa de uma solução rápida e simples
- O orçamento é limitado

### Escolha **CRAG** quando:
- A precisão é crítica (aplicações médicas, jurídicas, financeiras)
- Você tem uma base de conhecimento grande e heterogênea
- Há risco de informações conflitantes ou irrelevantes

### Escolha **Graph RAG** quando:
- Seu conhecimento tem relacionamentos complexos entre entidades
- Você precisa explorar conexões e hierarquias
- O conhecimento é estruturado em taxonomias

### Escolha **Agentic RAG** quando:
- Você precisa combinar conhecimento interno e externo
- As queries requerem raciocínio multi-passo
- Você precisa de informações atualizadas em tempo real
- A flexibilidade e autonomia são importantes

### Escolha **Adaptive RAG** quando:
- Você tem grande variabilidade na complexidade das queries
- Precisa balancear eficiência e precisão
- Quer otimizar custos computacionais

### Escolha **Multi Modal RAG** quando:
- Você trabalha com conteúdo que inclui imagens, áudio ou vídeo
- Precisa de busca visual ou por descrição
- O conteúdo é rico em múltiplas modalidades

### Escolha **W-RAG** quando:
- Você precisa de informações atualizadas constantemente
- O conhecimento interno não cobre todos os casos
- Você trabalha com notícias, eventos ou dados em tempo real

---

## Considerações Finais

A escolha da variação de RAG depende de vários fatores:

1. **Natureza do Conhecimento**: Estático vs. Dinâmico, Textual vs. Multimodal
2. **Complexidade das Queries**: Simples vs. Complexas, Diretas vs. Multi-passo
3. **Requisitos de Precisão**: Tolerância a erros vs. Crítico
4. **Recursos Disponíveis**: Computacionais, financeiros, tempo de desenvolvimento
5. **Necessidade de Atualização**: Estático vs. Tempo Real

Muitas vezes, a melhor solução combina múltiplas abordagens. Por exemplo, um sistema pode usar:
- **Graph RAG** para conhecimento estruturado interno
- **Agentic RAG** para decisões sobre quando buscar na web
- **W-RAG** para informações atualizadas
- **CRAG** para validação de qualidade

O futuro do RAG está na combinação inteligente dessas técnicas, criando sistemas cada vez mais capazes e confiáveis.

---

## Referências e Leitura Adicional

- **RAG Original**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- **CRAG**: Self-RAG papers sobre correção e validação
- **Graph RAG**: Microsoft Research sobre grafos de conhecimento
- **Agentic RAG**: LangChain e LangGraph documentação
- **Adaptive RAG**: Pesquisas sobre adaptação dinâmica
- **Multi Modal RAG**: CLIP, BLIP e modelos multimodais
- **W-RAG**: Implementações de RAG com busca web

---

*Documento criado para o Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA*


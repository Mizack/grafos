# Arquitetura e Contexto do Projeto de Grafos (Neo4j + NLP)

Este documento registra as decisões arquiteturais, evoluções de código e próximos passos definidos para o motor de conhecimento baseado em grafos (Neo4j) e NLP.

## 1. Evolução da Busca de Similaridade (Vector Search)
A abordagem inicial da prova de conceito (POC) utilizava força bruta no Python (loop `N x M` para calcular o Cosseno entre todos os artigos). Para viabilizar a escalabilidade, a arquitetura foi migrada para **Busca Vetorial Nativa no Neo4j**.

**Mudanças implementadas no `app.py`:**
- Criação automática do Índice Vetorial no Neo4j para o modelo `MiniLM` de 384 dimensões.
- Os embeddings passaram a ser armazenados nativamente como propriedades dos nós `Article` (`a.embedding = $embedding`).
- Substituição dos laços de repetição por chamadas diretas ao banco usando `CALL db.index.vector.queryNodes` com um limite (Top-K) de 50 vizinhos.

### 1.1. O Desafio da Escala Matemática (Cosseno)
Bancos de dados vetoriais, como o Neo4j, evitam pontuações negativas e convertem o Cosine Similarity para um intervalo de **0.0 a 1.0** através da fórmula `(1 + cos) / 2`. 
Para garantir que a aplicação se comporte corretamente:
1. O Python recebe um *threshold* padrão (ex: `0.6`).
2. Converte matematicamente para a escala do Neo4j (`0.8`).
3. O Neo4j filtra as similaridades vetoriais.
4. Antes de persistir a relação no banco, a escala é **revertida** (`SET r.score = (score * 2) - 1`) para manter a coerência visual dos dados com o SentenceTransformer.

## 2. Parametrização e Metadados
- **Threshold Flexível:** A variável `SIMILARITY_THRESHOLD` foi implementada no arquivo `.env`. Isso permite ajustar a rigidez das conexões `SIMILAR_TO` sem necessidade de alterar o código.
- **Rastreabilidade de Tempo:** Foram incluídas as propriedades `insertedAt` e `updatedAt` nos nós do Neo4j utilizando a função de tempo do próprio banco de dados (`datetime()`), garantindo um *timestamp* preciso no momento da transação.

## 3. Próximos Passos: Evolução da Extração de Entidades (NLP)
O extrator atual (`yake`) funciona via análise estatística de repetição (N-grams), o que ocasionalmente elege verbos comuns ("veja", "entenda") como palavras-chave. 

A arquitetura prevê três caminhos evolutivos para a inteligência de *keywords*:
1. **Modelos NER (Named Entity Recognition):** Substituir o YAKE pelo **spaCy** (modelo médio `pt_core_news_md`). O spaCy é leve, extremamente veloz em produção (escrito em Cython) e entende linguisticamente a diferença entre um "Local", uma "Pessoa" e um verbo, reduzindo o ruído estrutural.
2. **Generative AI (LLMs):** Extração guiada por *prompts* utilizando modelos grandes para cenários mais densos onde abstrações (temas) são necessárias.
3. **Blocklist Viva:** Permitir que as `BlocklistedWords` vivam no próprio banco de dados e sejam alimentadas tanto por feedback humano quanto por análise de dispersão no próprio grafo (Graph Data Science).

## 4. Viabilidade: Deploy em AWS Lambda
Devido à natureza assíncrona do processamento, a AWS Lambda é um excelente ambiente, porém com restrições arquiteturais rigorosas:
- **Formato:** O empacotamento padrão `.zip` falhará por exceder o limite de 250MB da AWS. O PyTorch e o modelo de IA (`paraphrase-multilingual-MiniLM`) pesam juntos mais de 1 GB.
- **Solução:** O deploy deve ser feito compulsoriamente via **Docker Container Image**, enviado para o AWS ECR. A Lambda suporta imagens Docker de até 10 GB.
- **Custos e Cold Start:** O custo de execução por milissegundo de Docker ou Zip na Lambda é idêntico. O único custo adicional será o armazenamento de alguns centavos no ECR. O ponto crítico é o *Cold Start*, que pode levar de 5 a 15 segundos para carregar o modelo de ML na memória. Por isso, a aplicação brilhará processando itens em lote via *background jobs* (SQS, EventBridge).

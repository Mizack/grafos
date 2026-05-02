import os
from chalice import Chalice, Response
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline, AutoTokenizer
from urllib.parse import urlparse
from dotenv import load_dotenv

# Caminho para o .env na pasta raiz (pois a API ficará na sub-pasta /api)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

app = Chalice(app_name="grafos-rag")

# =====================================================================
# Configurações do Banco de Dados
# =====================================================================
def get_neo4j_config():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    neo4j_auth = os.getenv("NEO4J_AUTH")
    if not password and neo4j_auth and "/" in neo4j_auth:
        auth_user, auth_password = neo4j_auth.split("/", 1)
        if not os.getenv("NEO4J_USER"):
            user = auth_user
        password = auth_password

    return uri, user, password

# Conexão Global (Chalice/Lambda mantém objetos globais ativos entre requisições "quentes")
uri, user, password = get_neo4j_config()
driver = GraphDatabase.driver(uri, auth=(user, password))


# =====================================================================
# Inicialização dos Modelos de Inteligência Artificial
# =====================================================================
print("Carregando modelos de IA. Isso pode levar alguns segundos (Cold Start)...")

# 1. Modelo de Embedding (Gera o vetor da pergunta do usuário - 384 dimensões)
# Forçando carregamento na GPU primária (cuda:0) para a RTX 3050
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda:0')

# 2. Modelo Gerativo (LLM) - Adaptado para aproveitar a RTX 3050
# Qwen/Qwen2.5-1.5B-Instruct é mais inteligente e utilizará ~3GB de VRAM em FP16 (suportado com folga pela 3050)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Configuramos o pipeline com foco na GPU local e com precisão fp16 explícita para evitar estouro de memória
qwen_pipeline = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device_map="cuda:0", # Garante alocação na RTX 3050
    torch_dtype=torch.float16 # Otimização mandatória em placas menores (reduz consumo pela metade)
)


# =====================================================================
# Lógica do RAG (Retrieval-Augmented Generation)
# =====================================================================
def recuperar_contexto_neo4j(pergunta, top_k=3):
    """
    Passo 1: RETRIEVAL
    Transforma a pergunta em vetor e busca os artigos mais similares no grafo.
    """
    vetor_pergunta = embedding_model.encode(pergunta).tolist()
    
    with driver.session() as session:
        # Busca vetorial utilizando o índice que você já criou ('article_embeddings')
        result = session.run("""
            CALL db.index.vector.queryNodes('article_embeddings', $top_k, $embedding)
            YIELD node AS art, score
            RETURN art.title AS titulo, art.description AS desc, art.content AS conteudo, score
            ORDER BY score DESC
        """, top_k=top_k, embedding=vetor_pergunta)
        
        contextos = []
        for record in result:
            titulo = record["titulo"]
            conteudo = record["conteudo"] or record["desc"]
            # Montamos um bloco de texto que será a "verdade" para a IA ler
            contextos.append(f"TÍTULO DO ARTIGO: {titulo}\nCONTEÚDO: {conteudo}\n")
            
        return "\n---\n".join(contextos)

@app.route('/ask', methods=['POST'])
def ask_rag():
    """
    Passo 2 e 3: AUGMENTED GENERATION
    Endpoint principal. Recebe a requisição POST e coordena o pipeline do RAG.
    """
    request = app.current_request
    body = request.json_body
    
    if not body or 'question' not in body:
        return Response(body={"error": "Obrigatório enviar um JSON com a chave 'question'."}, status_code=400)
        
    pergunta = body['question']
    
    try:
        # 1. Recupera as informações de contexto do Neo4j
        contexto_str = recuperar_contexto_neo4j(pergunta, top_k=3)
        
        if not contexto_str:
            return {"answer": "Não possuo informações suficientes no banco de dados para responder a essa pergunta."}
            
        # 2. Formata o prompt seguindo as regras do Qwen (ChatML)
        # É crucial dar a instrução do sistema (System Prompt) amarrando o modelo ao contexto
        mensagens = [
            {
                "role": "system", 
                "content": "Você é um assistente de IA focado em responder perguntas baseando-se estritamente no contexto fornecido. Não invente informações. Se a resposta não estiver no contexto, diga 'Não possuo dados para responder a isso'."
            },
            {
                "role": "user", 
                "content": f"Contexto extraído do meu banco de dados:\n{contexto_str}\n\nPergunta do usuário: {pergunta}"
            }
        ]
        
        # O apply_chat_template formata o array de dicts no padrão exato que o Qwen entende
        prompt_formatado = tokenizer.apply_chat_template(
            mensagens, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. Geração de Texto
        outputs = qwen_pipeline(
            prompt_formatado,
            max_new_tokens=300, # Tamanho máximo da resposta
            do_sample=True,
            temperature=0.2, # Temperatura baixa garante menor criatividade e mais apego aos fatos (contexto)
            top_p=0.9
        )
        
        # O output do pipeline inclui o prompt original, precisamos cortar essa parte
        resposta_completa = outputs[0]["generated_text"]
        resposta_gerada = resposta_completa[len(prompt_formatado):].strip()
        
        # Retorna o resultado para o frontend/cliente
        return {
            "question": pergunta,
            "answer": resposta_gerada,
            "sources_used": contexto_str
        }
        
    except Exception as e:
        return Response(body={"error": f"Erro interno processando o RAG: {str(e)}"}, status_code=500)

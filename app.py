import os
import json
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import yake


def get_neo4j_config():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    # Fallback opcional para o formato usado no docker-compose: NEO4J_AUTH=usuario/senha
    neo4j_auth = os.getenv("NEO4J_AUTH")
    if not password and neo4j_auth and "/" in neo4j_auth:
        auth_user, auth_password = neo4j_auth.split("/", 1)
        if not os.getenv("NEO4J_USER"):
            user = auth_user
        password = auth_password

    parsed = urlparse(uri)
    valid_schemes = {"bolt", "bolt+ssc", "bolt+s", "neo4j", "neo4j+ssc", "neo4j+s"}
    if parsed.scheme not in valid_schemes:
        raise ValueError(
            "NEO4J_URI inválida. Use algo como 'bolt://localhost:7687' ou 'neo4j://localhost:7687'."
        )

    if not user or not password:
        raise ValueError(
            "Credenciais Neo4j inválidas. Defina NEO4J_USER e NEO4J_PASSWORD (ou NEO4J_AUTH no formato usuario/senha)."
        )

    return uri, user, password

# Verbos imperativos, conectivos e termos genéricos que não agregam como keyword
BLOCKLIST_KEYWORDS = {
    "veja", "entenda", "saiba", "confira", "descubra", "conheça", "assista",
    "leia", "ouça", "acesse", "clique", "siga", "mande", "participe", "aproveite",
    "aprenda", "faça", "quer", "precisa", "busque", "encontre",

    # Verbos de Atribuição e Noticiação
    "menciona", "revela", "mostra", "aponta", "indica", "afirma", "diz", "fala", 
    "explica", "destaca", "ressalta", "alerta", "adverte", "garante", "defende", 
    "critica", "anuncia", "promete", "nega", "confirma", "celebra", "lamenta", 
    "questiona", "ironiza", "ataca", "responde", "avisa", "prevê",

    # Termos de Formato e Navegação de Notícia
    "análise", "analise", "resumo", "resumir", "atualização", "direto", "ao vivo",
    "vídeo", "fotos", "galeria", "infográfico", "podcast", "exclusivo", "urgente",
    "opinião", "artigo", "coluna", "blog", "entrevista", "reportagem", "especial",
    "checklist", "guia", "passo a passo", "manual",

    # Adjetivos de Novidade/Quantidade (Vagueza)
    "novo", "nova", "novos", "novas", "mais", "maior", "menor", "melhor", "pior",
    "primeiro", "primeira", "último", "última", "grande", "pequeno", "vários",
    "diversos", "alguns", "muitos", "poucos", "total", "parcial",

    # Conectivos, Preposições e Advérbios de Tempo/Lugar
    "após", "antes", "durante", "sobre", "entre", "para", "pelo", "pela", "pelos", "pelas",
    "como", "quando", "onde", "quem", "qual", "quais", "num", "numa", "nesta", "neste",
    "desde", "até", "através", "contra", "conforme", "segundo", "enquanto",

    # Termos de Aproximação e Intensidade
    "cerca", "aproximadamente", "quase", "exatamente", "simplesmente", "realmente",
    "totalmente", "parcialmente", "possível", "provável", "talvez", "apenas", "somente",
    "pode", "podem", "deve", "devem", "será", "serão", "foi", "foram",
}


class NewsGraphPOC:
    def __init__(self, uri, user, password, similarity_threshold=0.6):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Melhor para Português
        self.kw_extractor = yake.KeywordExtractor(
            lan="pt", n=2, dedupLim=0.7, top=8, features=None
        )
        
        # Converte a nota de corte (ex: 0.6) para a escala normalizada do Neo4j (ex: 0.8)
        self.neo4j_threshold = (1.0 + float(similarity_threshold)) / 2.0
        
        self.inicializar_banco()

    def inicializar_banco(self):
        """Cria o índice vetorial no Neo4j caso ele ainda não exista."""
        with self.driver.session() as session:
            try:
                # O modelo paraphrase-multilingual-MiniLM-L12-v2 gera vetores de 384 dimensões
                session.run("""
                    CREATE VECTOR INDEX article_embeddings IF NOT EXISTS 
                    FOR (a:Article) ON (a.embedding) 
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("Índice vetorial inicializado com sucesso.")
            except Exception as e:
                print(f"Aviso ao verificar índice vetorial: {e[20:]}")

    def close(self):
        self.driver.close()

    def extrair_palavras_chave(self, texto):
        keywords = self.kw_extractor.extract_keywords(texto)
        resultado = []
        for kw, _ in keywords:
            termo = kw.lower().strip()
            # Descarta se qualquer token da keyword estiver na blocklist
            tokens = set(termo.split())
            if tokens & BLOCKLIST_KEYWORDS:
                continue
            resultado.append(termo)
        return resultado

    
    def processar_dados(self, data):
        articles = data.get("articles", [])
        
        # Pre-calcula os embeddings de todos os novos itens da carga atual
        novos_titles = [a["title"] for a in articles]
        novos_embeddings = self.model.encode(novos_titles)
        
        with self.driver.session() as session:
            # 1. Inserir Artigos e Fontes
            for idx, art in enumerate(articles):
                embedding_list = novos_embeddings[idx].tolist()
                
                session.run("""
                    MERGE (s:Source {id: $s_id})
                    ON CREATE SET s.name = $s_name, s.url = $s_url
                    
                    MERGE (a:Article {id: $a_id})
                    ON CREATE SET 
                        a.title = $title, 
                        a.description = $desc,
                        a.url = $url,
                        a.publishedAt = datetime($date),
                        a.content = $content,
                        a.embedding = $embedding,
                        a.insertedAt = datetime()
                    
                    // Atualiza o embedding e registra o horário da última atualização
                    SET a.embedding = $embedding,
                        a.updatedAt = datetime()
                    
                    MERGE (a)-[:PUBLISHED_BY]->(s)
                """, 
                s_id=art["source"]["id"], s_name=art["source"]["name"], s_url=art["source"]["url"],
                a_id=art["id"], title=art["title"], desc=art["description"], 
                url=art["url"], date=art["publishedAt"], content=art["content"],
                embedding=embedding_list)

            # 2. Extrair palavras-chave e criar relações MENCIONA
            for art in articles:
                texto = f"{art['title']}. {art['description']}"
                keywords = self.extrair_palavras_chave(texto)
                for kw in keywords:
                    session.run("""
                        MERGE (k:Keyword {word: $word})
                        WITH k
                        MATCH (a:Article {id: $a_id})
                        MERGE (a)-[:MENCIONA]->(k)
                    """, word=kw, a_id=art["id"])

            # 3. Relacionar por Similaridade (Via Vector Search Nativo do Banco)
            print(f"Calculando similaridade para {len(articles)} itens via Vector Search...")
            
            for idx, art in enumerate(articles):
                embedding_list = novos_embeddings[idx].tolist()
                
                # Busca no banco os 50 nós mais parecidos com este embedding (k=50)
                # OBS: Em versões bem recentes do Neo4j isso pode gerar um aviso amarelo (warning) de deprecation
                # no terminal. Esse aviso é inofensivo e a busca funciona perfeitamente.
                session.run("""
                    CALL db.index.vector.queryNodes('article_embeddings', 50, $embedding)
                    YIELD node AS db_art, score
                    
                    // IMPORTANTE: O Neo4j normaliza o Cosine Similarity para o intervalo [0, 1]
                    // O threshold foi convertido no Python e passado como parâmetro.
                    WHERE score > $threshold AND score < 0.999 AND db_art.id <> $a_id
                    
                    MATCH (a:Article {id: $a_id})
                    
                    // Ordenamos os nós (id alfanumérico) para evitar relações duplas do tipo (a)->(b) e (b)->(a)
                    WITH a, db_art, score,
                         CASE WHEN a.id < db_art.id THEN a ELSE db_art END AS n1,
                         CASE WHEN a.id < db_art.id THEN db_art ELSE a END AS n2
                    
                    MERGE (n1)-[r:SIMILAR_TO]->(n2)
                    // Revertemos a escala para o valor original do SentenceTransformer antes de salvar
                    SET r.score = (score * 2) - 1
                """, a_id=art["id"], embedding=embedding_list, threshold=self.neo4j_threshold)
            
            print("Carga e cruzamento de relacionamentos concluídos!")


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    with open("artigos.json", "r", encoding="utf-8") as f:
        dados_artigos = json.load(f)

    uri, user, password = get_neo4j_config()
    
    # Lê a variável de ambiente, garantindo que o valor padrão seja '0.6' caso não exista no .env
    env_threshold = os.getenv("SIMILARITY_THRESHOLD", "0.6")
    
    poc = NewsGraphPOC(uri, user, password, similarity_threshold=env_threshold)
    
    poc.processar_dados(dados_artigos)
    poc.close()
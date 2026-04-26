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
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Melhor para Português
        self.kw_extractor = yake.KeywordExtractor(
            lan="pt", n=2, dedupLim=0.7, top=8, features=None
        )

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
        titles = [a["title"] for a in articles]
        embeddings = self.model.encode(titles)
        
        with self.driver.session() as session:
            # 1. Inserir Artigos e Fontes
            for art in articles:
                session.run("""
                    MERGE (s:Source {id: $s_id})
                    ON CREATE SET s.name = $s_name, s.url = $s_url
                    
                    MERGE (a:Article {id: $a_id})
                    ON CREATE SET 
                        a.title = $title, 
                        a.description = $desc,
                        a.url = $url,
                        a.publishedAt = datetime($date),
                        a.content = $content
                    
                    MERGE (a)-[:PUBLISHED_BY]->(s)
                """, 
                s_id=art["source"]["id"], s_name=art["source"]["name"], s_url=art["source"]["url"],
                a_id=art["id"], title=art["title"], desc=art["description"], 
                url=art["url"], date=art["publishedAt"], content=art["content"])

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
                print(f"📌 '{art['title'][:50]}' → {keywords}")

            # 3. Relacionar por Similaridade (Threshold ajustado para 0.6 conforme seu teste anterior)
            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    score = float(util.cos_sim(embeddings[i], embeddings[j]))
                    
                    if score > 0.6:
                        if score >= 1:
                            print(f"'{titles[i]}' e '{titles[j]}' são idênticos (score = 1.0).")
                            continue

                        print(f"Comparando '{titles[i]}' com '{titles[j]}': Similaridade = {score:.4f}")
                        session.run("""
                            MATCH (a:Article {id: $id1}), (b:Article {id: $id2})
                            MERGE (a)-[r:SIMILAR_TO]->(b)
                            SET r.score = $score
                        """, id1=articles[i]["id"], id2=articles[j]["id"], score=score)


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # time.sleep(15)

    with open("artigos.json", "r", encoding="utf-8") as f:
        dados_artigos = json.load(f)

    uri, user, password = get_neo4j_config()
    poc = NewsGraphPOC(uri, user, password)
    
    poc.processar_dados(dados_artigos)
    poc.close()
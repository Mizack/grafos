deletar todos os registros:
```
MATCH (n) DETACH DELETE n
```

buscar notícias com determinada palavra:
```
MATCH (a:Article)-[:MENCIONA]->(k:Keyword)
WHERE k.word = "elon musk"
RETURN a.title AS Titulo, a.publishedAt AS Data, a.url AS Link
ORDER BY a.publishedAt DESC
```


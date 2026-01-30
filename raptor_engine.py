
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from chunking import HybridSemanticChunker

class RaptorEngine:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    Reduz alucinação 'Lost in the Middle' criando uma árvore hierárquica de resumos.
    """
    
    def __init__(self, api_key: str, provider: str = "google"):
        self.api_key = api_key
        self.provider = provider
        
        # Init Models
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
            self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0) # Mini for speed/cost
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0)

        # Chunker para folhas (usamos o híbrido mas com threshold mais relaxado para garantir chunks menores)
        self.chunker = HybridSemanticChunker(api_key, provider, threshold_amount=80.0) 

    def _get_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """
        Usa BIC (Bayesian Information Criterion) para achar numero ideal de clusters.
        """
        n_samples = len(embeddings)
        if n_samples < 2: return 1
        
        max_k = min(max_clusters, n_samples)
        bics = []
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            gm = GaussianMixture(n_components=k, random_state=42)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        optimal_k = k_values[np.argmin(bics)]
        return optimal_k

    def _embed_documents(self, docs: List[str]) -> np.ndarray:
        return np.array(self.embeddings.embed_documents(docs))

    def _cluster_and_summarize(self, docs: List[Document], level: int) -> List[Document]:
        """
        Agrupa documentos e sumaria cada grupo.
        """
        texts = [d.page_content for d in docs]
        
        # 1. Embed
        embeddings = self._embed_documents(texts)
        
        # 2. Cluster
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gm.fit_predict(embeddings)
        
        # 3. Summarize Clusters
        summarized_docs = []
        
        summary_prompt = ChatPromptTemplate.from_template(
            "Você é um Assistente Jurídico Sênior. Resuma o seguinte grupo de trechos de um processo judicial.\n"
            "Foque em Fatos, Datas, Valores e Argumentos Jurídicos Principais.\n"
            "Mantenha referências a nomes e números de folhas se houver.\n\n"
            "TRECHOS:\n{context}\n\n"
            "RESUMO CONSOLIDADO:"
        )
        chain = summary_prompt | self.llm | StrOutputParser()

        # Group docs by labels
        df = pd.DataFrame({'doc': docs, 'label': labels})
        
        for label, group in df.groupby('label'):
            cluster_docs = group['doc'].tolist()
            cluster_text = "\n---\n".join([d.page_content for d in cluster_docs])
            
            # Run LLM
            try:
                summary = chain.invoke({"context": cluster_text})
                # Create new doc from summary
                new_doc = Document(
                    page_content=summary, 
                    metadata={"level": level, "cluster": int(label), "child_count": len(cluster_docs)}
                )
                summarized_docs.append(new_doc)
            except Exception as e:
                print(f"Erro ao resumir cluster {label} nível {level}: {e}")
                
        return summarized_docs

    def build_tree(self, text: str) -> str:
        """
        Constrói a árvore RAPTOR e retorna uma string consolidada (Raíz + Ramos Principais).
        """
        # 1. Chunking Inicial (Folhas)
        leaf_docs = self.chunker.split_text(text)
        print(f"RAPTOR: {len(leaf_docs)} folhas geradas.")
        
        if len(leaf_docs) <= 5:
            # Se for pequeno, não precisa de arvore, retorna texto original
            return text

        current_level_docs = leaf_docs
        all_levels = {0: leaf_docs}
        
        level = 1
        # Recurse until we have few enough docs to be the root or max levels
        while len(current_level_docs) > 1 and level <= 2: # Max 3 levels (0, 1, 2) usually enough
            print(f"RAPTOR: Construindo Nível {level}...")
            summarized = self._cluster_and_summarize(current_level_docs, level)
            all_levels[level] = summarized
            current_level_docs = summarized
            level += 1
            
        # Montar o Contexto Final Otimizado
        # Estratégia: Pegar o Resumo Root (Nível mais alto) + Resumos do Nível Intermediário
        # + Top chunks originais (opcional, aqui retornamos a estrutura da árvore para o LLM navegar)
        
        final_context = "=== RESUMO HIERÁRQUICO (RAPTOR) ===\n"
        
        # Adiciona Raiz (Último Nível)
        max_lvl = max(all_levels.keys())
        final_context += f"\n## VISÃO MACRO (Nível {max_lvl})\n"
        for d in all_levels[max_lvl]:
            final_context += f"{d.page_content}\n"
            
        # Adiciona Níveis Intermediários (se houver) -> Detalhes importantes
        if max_lvl > 1:
            final_context += f"\n## DETALHAMENTO DE TÓPICOS (Nível {max_lvl-1})\n"
            for d in all_levels[max_lvl-1]:
                final_context += f"- {d.page_content}\n"
                
        # Adiciona Amostra de Folhas Relevantes (opcional, aqui vou por as tags do regex)
        # Para economizar tokens, não coloco todas as folhas, o agente deve confiar nos resumos 
        # ou usar tool retrieval se precisar (mas aqui é contexto estático).
        
        return final_context

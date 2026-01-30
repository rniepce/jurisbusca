
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class HybridSemanticChunker:
    """
    Estratégia Híbrida de Chunking:
    1. Macro-Splitting (Regex): Divide por seções jurídicas (Relatório, Fundamentação, etc).
    2. Micro-Splitting (Semântico): Usa Embeddings para detectar mudanças sutis de tópico.
    """
    
    # Regex combinada baseada no feedback do usuário + padrões comuns
    # User requested: DO DIREITO, DOS FATOS, DOS PEDIDOS, II -, III -, etc.
    LEGAL_HEADERS_REGEX = r"(?im)^(?:\s*|.*[\.\:])\s*(DOS FATOS|DO DIREITO|DA FUNDAMENTAÇÃO|DOS PEDIDOS|DO MÉRITO|DO DISPOSITIVO|RELATÓRIO|DISPOSITIVO|CONCLUSÃO|PRELIMINARMENTE|DA TUTELA|EMENTA|[IVXLCDM]+\s+\-)(?::|\s|$)"

    def __init__(self, api_key: str, provider: str = "google", threshold_type: str = "percentile", threshold_amount: float = 90.0):
        """
        Inicializa o Chunker Híbrido.
        Args:
            api_key: Chave da API (OpenAI ou Google).
            provider: 'openai' ou 'google'.
            threshold_type: 'percentile', 'standard_deviation', etc.
            threshold_amount: Valor do percentile (ex: 90.0 para alta coesão).
        """
        self.embeddings = self._get_embeddings(api_key, provider)
        if self.embeddings:
            # Semantic Splitter Initialization
            self.semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=threshold_type,
                breakpoint_threshold_amount=threshold_amount  # percentile default 90
            )
        else:
            self.semantic_splitter = None
            print("⚠️ Embeddings não inicializados. Fallback para RecursiveSplitter pode ser necessário.")

    def _get_embeddings(self, key, provider):
        try:
            if provider == "openai":
                return OpenAIEmbeddings(api_key=key)
            elif provider == "google":
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
            else:
                # Fallback to Google if unknown
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
        except Exception as e:
            print(f"Erro ao iniciar Embeddings ({provider}): {e}")
            return None

    def split_text(self, text: str, source_metadata: Optional[Dict] = None) -> List[Document]:
        """
        Executa o pipeline híbrido.
        """
        if not source_metadata:
            source_metadata = {}
            
        final_docs = []
        
        # 1. MACRO SPLIT (Regex)
        # Dividimos o texto em blocos baseados em cabeçalhos
        # A regex tenta encontrar os headers e fazer split mantendo o delimitador
        # Mas para simplificar e não perder o header, vamos iterar ou usar split com group capture.
        
        # Split mantendo o delimitador
        parts = re.split(f"({self.LEGAL_HEADERS_REGEX}.*)", text)
        
        current_section = "GERAL"
        buffer_text = ""
        
        # A lista parts terá: [plaint_text, header_match, content, header_match, content...]
        # Ou [content_before_first_match, header_match, content...]
        
        # Reconstrução mais segura:
        sections = []
        
        # Usando finditer para localizar as seções e seus índices
        matches = list(re.finditer(self.LEGAL_HEADERS_REGEX, text, re.MULTILINE))
        
        if not matches:
             # Sem cabeçalhos detectados, processa tudo como um bloco GERAL
             sections.append(("GERAL", text))
        else:
            prev_end = 0
            for i, match in enumerate(matches):
                # Texto antes do match atual (pertence à seção anterior ou é preâmbulo)
                pre_text = text[prev_end:match.start()].strip()
                if pre_text:
                    label = "PREAMBULO" if i == 0 else matches[i-1].group(1).upper()
                    sections.append((label, pre_text))
                
                # O match em si é o início da nova seção. O texto vai até o próximo match
                prev_end = match.start() # Recua para incluir o título na seção (opcional, mas bom pro contexto)
                
            # Texto final após o último match
            last_section_text = text[prev_end:].strip()
            if last_section_text:
                 # Extrai o nome da seção do início desse texto
                 header_match = re.match(self.LEGAL_HEADERS_REGEX, last_section_text)
                 label = header_match.group(1).upper() if header_match else "FINAL"
                 sections.append((label, last_section_text))

        # 2. MICRO SPLIT (Semantic) e Criação de Docs
        for sec_name, sec_content in sections:
            if not sec_content.strip():
                continue
                
            if self.semantic_splitter:
                try:
                    # Semantic Chunking
                    chunks = self.semantic_splitter.create_documents([sec_content])
                    for chunk in chunks:
                        # Enriquece metadados
                        chunk.metadata.update(source_metadata)
                        chunk.metadata["section"] = sec_name
                        chunk.metadata["chunk_strategy"] = "hybrid_semantic"
                        final_docs.append(chunk)
                except Exception as e:
                    print(f"Erro no Semantic Splitter da seção {sec_name}: {e}. Usando fallback.")
                    # Fallback: cria um doc único se falhar
                    final_docs.append(Document(page_content=sec_content, metadata={**source_metadata, "section": sec_name}))
            else:
                 final_docs.append(Document(page_content=sec_content, metadata={**source_metadata, "section": sec_name}))
                 
        return final_docs

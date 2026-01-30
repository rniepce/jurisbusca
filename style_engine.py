
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
import os

class StyleEngine:
    """
    AGENTE DE ESTILO (The Stylist).
    Usa Dynamic Few-Shot Prompting para clonar o estilo do magistrado.
    """
    
    def __init__(self, api_key: str, provider: str = "google", persist_dir: str = "./chroma_db_style"):
        self.api_key = api_key
        self.provider = provider
        self.persist_dir = persist_dir
        
        # Init Embeddings
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            
        # Init VectorStore for Style (Templates)
        # Collection específica para estilo
        self.vectorstore = Chroma(
            collection_name="style_templates",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        
        self.example_selector = None
        # Se já tiver documentos, inicializa o seletor. Se não, espera a indexação.
        if self.vectorstore._collection.count() > 0:
            self._init_selector()

    def _init_selector(self):
        """Inicializa o seletor semântico com o vectorstore existente."""
        self.example_selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore,
            k=3 # Top 3 exemplos parecidos
        )

    def index_templates(self, documents: list[Document]):
        """
        Recebe documentos (templates/decisões passadas), indexa no Chroma 
        e prepara para uso como Few-Shot examples.
        """
        # Formata para o ExampleSelector (input/output pair)
        # Assumindo que o documento inteiro é um bom exemplo de "saída" desejada 
        # para um "input" similar (o próprio texto do documento serve como key para busca semântica)
        
        # O ExampleSelector nativo espera keys 'input' e 'output' nos metadados ou dicts.
        # Aqui, vamos simplificar: O vetor armazena o texto da decisão.
        # Quando buscamos, usamos o RESUMO do novo caso como query.
        # O Chroma retorna as Decisões Passadas mais similares tematicamente.
        
        self.vectorstore.add_documents(documents)
        self._init_selector()
        print(f"✅ {len(documents)} templates de estilo indexados.")

    def get_style_prompt(self, current_case_summary: str) -> FewShotChatMessagePromptTemplate:
        """
        Gera o Prompt Few-Shot dinâmico baseado no caso atual.
        """
        if not self.example_selector:
            return None
            
        # Define o formato do exemplo no prompt
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Gere uma decisão similar a este contexto: {page_content}"),
                ("ai", "{page_content}"), 
            ]
        )
        
        # O seletor busca no vectorstore usando o current_case_summary como query
        # e retorna os docs mais parecidos.
        
        # Hack: SemanticSimilarityExampleSelector expects specific arg structures usually.
        # Vamos usar o select_examples direto para ter controle.
        examples = self.example_selector.select_examples({"page_content": current_case_summary})
        
        # Cria o template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )
        
        return few_shot_prompt

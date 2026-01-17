import os
import time
import logging
import requests
import numpy as np
from typing import List, Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select, text
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuração do Banco de Dados ---
# Tenta pegar da ENV, senão pede input (segurança)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("DATABASE_URL não encontrada nas variáveis de ambiente.")
    # Exemplo: DATABASE_URL = "postgresql://postgres:password@roundhouse.proxy.rlwy.net:12345/railway"

# --- Definição do Modelo (Schema) ---
class Processo(SQLModel, table=True):
    __tablename__ = "processos_juridicos"

    id: Optional[int] = Field(default=None, primary_key=True)
    numero_processo: str = Field(index=True)
    texto_decisao: str
    
    # Campo vetorial para pgvector (384 dimensoes para all-MiniLM-L6-v2)
    embedding: Optional[List[float]] = Field(default=None, sa_column_kwargs={"type_": "vector(384)"})
    
    cluster_id: Optional[int] = Field(default=None)
    sugestao_ia: Optional[str] = Field(default=None)
    
    processado: bool = Field(default=False)
    data_processamento: Optional[str] = Field(default=None)

# --- Classes de Serviço ---

class VectorizerService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'mps' if self._check_mps_available() else 'cpu'
        logger.info(f"🚀 Inicializando modelo de vetorização em: {self.device.upper()}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}. Tentando CPU.")
            self.model = SentenceTransformer(model_name, device='cpu')

    def _check_mps_available(self):
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Gera embeddings em batch usando a GPU se disponível."""
        start_time = time.time()
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.time() - start_time
        logger.info(f"⚡ Vetorização de {len(texts)} itens concluída em {elapsed:.2f}s")
        return embeddings

class OllamaService:
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._check_connection()

    def _check_connection(self):
        try:
            requests.get(self.base_url)
            logger.info(f"🟢 Conectado ao Ollama em {self.base_url}")
        except Exception:
            logger.error(f"🔴 Não foi possível conectar ao Ollama em {self.base_url}. Verifique se o app está rodando.")

    def get_decision_suggestion(self, case_text: str) -> Optional[str]:
        """Consulta o LLM local para gerar uma minuta."""
        api_url = f"{self.base_url}/api/generate"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Você é um juiz experiente. Analise o caso jurídico abaixo e sugira uma minuta de decisão técnica, imparcial e fundamentada. Seja conciso.<|eot_id|><|start_header_id|>user<|end_header_id|>

CASO:
{case_text[:3000]}... (conteúdo truncado para contexto)

DECISÃO SUGERIDA:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2, # Mais determinístico para decisões jurídicas
                "num_ctx": 4096
            }
        }

        try:
            response = requests.post(api_url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                logger.error(f"Erro na API Ollama: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Falha na requisição ao Ollama: {e}")
            return None

class ETLPipeline:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.vectorizer = VectorizerService()
        self.llm = OllamaService()
        self._setup_db()

    def _setup_db(self):
        """Habilita extensão pgvector no banco."""
        try:
            with Session(self.engine) as session:
                session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
                session.commit()
            # Cria tabelas se não existirem
            SQLModel.metadata.create_all(self.engine)
            logger.info("✅ Banco de dados configurado e schema atualizado.")
        except Exception as e:
            logger.error(f"Erro crítico no setup do banco: {e}")
            raise

    def run_batch(self, batch_size=50):
        with Session(self.engine) as session:
            # 1. Busca pendentes
            logger.info("🔍 Buscando processos pendentes...")
            statement = select(Processo).where(Processo.processado == False).limit(batch_size)
            batch = session.exec(statement).all()

            if not batch:
                logger.info("zzz Nenhum processo pendente. Dormindo...")
                return False

            ids = [p.id for p in batch]
            logger.info(f"📦 Processando Batch IDs: {ids}")

            # 2. Vetorização (Batch)
            texts = [p.texto_decisao for p in batch]
            embeddings = self.vectorizer.generate_embeddings(texts)

            # 3. Clusterização (Local Batch)
            # Para clusterização real, idealmente usaríamos um modelo pré-treinado ou MiniBatchKmeans persistido.
            # Aqui faremos uma clusterização simples dentro do lote para identificar grupos imediatos.
            cluster_labels = [0] * len(batch)
            if len(batch) >= 3:
                try:
                    kmeans = KMeans(n_clusters=min(3, len(batch)), n_init='auto')
                    cluster_labels = kmeans.fit_predict(embeddings)
                except Exception as e:
                    logger.warning(f"Erro no clustering: {e}")

            # 4. Atualização e LLM (Item a Item)
            for i, processo in enumerate(batch):
                try:
                    # Salva vetor
                    processo.embedding = embeddings[i].tolist()
                    processo.cluster_id = int(cluster_labels[i])
                    
                    # Gera Decisão (Ollama)
                    logger.info(f"🧠 Gerando decisão para Processo {processo.numero_processo}...")
                    ai_decision = self.llm.get_decision_suggestion(processo.texto_decisao)
                    
                    if ai_decision:
                        processo.sugestao_ia = ai_decision
                        processo.processado = True
                        processo.data_processamento = time.strftime("%Y-%m-%d %H:%M:%S")
                        session.add(processo)
                    else:
                        logger.warning(f"⚠️ Pulo no processo {processo.id} por falha na IA.")

                except Exception as e:
                    logger.error(f"Erro ao processar item {processo.id}: {e}")

            # Commit do Batch
            session.commit()
            logger.info("✅ Batch processado e salvo no Railway com sucesso!")
            return True

def main():
    if not DATABASE_URL:
        logger.error("❌ Configure a variável de ambiente DATABASE_URL antes de rodar.")
        return

    logger.info("🚀 Iniciando Legal Tech ETL Worker...")
    pipeline = ETLPipeline(DATABASE_URL)

    try:
        while True:
            has_work = pipeline.run_batch(batch_size=10) # Lotes pequenos para teste
            if not has_work:
                time.sleep(10) # Polling interval
    except KeyboardInterrupt:
        logger.info("🛑 Worker interrompido pelo usuário.")

if __name__ == "__main__":
    main()

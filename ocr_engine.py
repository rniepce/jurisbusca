
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import requests
import tempfile
from paddleocr import PaddleOCR

# Constants
MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/ESPCN_x2.pb"
MODEL_PATH = "data/models/ESPCN_x2.pb"

# Lazy Global Engines (initialized on demand)
PADDLE_ENGINE = None

def get_paddle_engine():
    global PADDLE_ENGINE
    if PADDLE_ENGINE is None:
        try:
            # use_angle_cls=True detects rotation
            PADDLE_ENGINE = PaddleOCR(use_angle_cls=True, lang='pt', show_log=False)
        except Exception as e:
            print(f"⚠️ Erro ao iniciar PaddleOCR Engine: {e}")
            PADDLE_ENGINE = None
    return PADDLE_ENGINE

class DeepSeekOCREngine:
    def __init__(self, model_path="deepseek-ai/DeepSeek-OCR-2"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("DeepSeek OCR requires 'transformers', 'torch', 'accelerate' and 'pillow'. Please install them.")

        print(f"🚀 Loading DeepSeek-OCR-2 from {model_path}...")
        # Carrega o tokenizer e o modelo com suporte a GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map="auto" # Distribui automaticamente na GPU disponível
        ).eval()

    def process_image(self, image_path):
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert("RGB")
        
        # Prompt específico para o DeepSeek-OCR-2 (extração estruturada)
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        messages = [{"role": "user", "content": prompt}]
        
        # Processamento visual e geração de texto
        # O modelo identifica automaticamente tabelas e estrutura jurídica
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt", 
            return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False, # Para OCR, queremos precisão determinística
                temperature=0.0
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Global singleton for DeepSeek (to avoid reloading heavy model)
DEEPSEEK_ENGINE = None

def get_deepseek_engine():
    global DEEPSEEK_ENGINE
    if DEEPSEEK_ENGINE is None:
        try:
            DEEPSEEK_ENGINE = DeepSeekOCREngine()
        except Exception as e:
            print(f"⚠️ Erro ao iniciar DeepSeekOCREngine: {e}")
            DEEPSEEK_ENGINE = None
    return DEEPSEEK_ENGINE


def download_model():
    """Baixa o modelo de Super-Reolução se não existir."""
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Baixando modelo de Super-Resolução (ESPCN_x2)...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(MODEL_URL, timeout=30)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("✅ Modelo baixado com sucesso.")
            else:
                print(f"❌ Falha ao baixar modelo: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Erro ao baixar modelo: {e}")

def get_superres_model():
    """Carrega o modelo de auto-resolução."""
    download_model()
    if os.path.exists(MODEL_PATH):
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(MODEL_PATH)
            sr.setModel("espcn", 2) # Scale of 2
            return sr
        except Exception as e:
            print(f"⚠️ Erro ao carregar dnn_superres: {e}")
    return None

def preprocess_image(image_cv, apply_superres=False):
    """
    Pipeline de Pré-processamento (Usado pelo PaddleOCR):
    1. Super-Resolução (se DPI baixo)
    2. Conversão B/W (Binarização Otimizada)
    3. Redução de Ruído
    4. Deskew (Correção de Rotação)
    """
    processed = image_cv
    
    # 1. Super Resolution
    if apply_superres:
        sr = get_superres_model()
        if sr:
            # Upscale logic
            processed = sr.upsample(processed)
    
    # Converter para escala de cinza
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    # 2. Redução de Ruído (Denoise)
    desnoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Binarização (Adaptive Threshold para lidar com sombras/manchas)
    binary = cv2.adaptiveThreshold(
        desnoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. Deskew (Alinhamento de Texto)
    coords = np.column_stack(np.where(binary > 0)) # Pixel coordinates > 0
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Gira apenas se a inclinação for relevante (> 0.5 graus)
    if abs(angle) > 0.5 and abs(angle) < 45: # Limit rotation to avoid flip
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return binary

def extract_text_from_pdf(pdf_path, engine="paddle"):
    """
    Função Principal: PDF -> Imagens -> OCR text
    
    Args:
        pdf_path: Caminho para o PDF.
        engine: "paddle" (OCR local/CPU friendly) ou "deepseek" (GPU required, Markdown rich).
    """
    full_text = ""
    
    # Seleciona Engine
    if engine == "deepseek":
        ocr_engine = get_deepseek_engine()
        if not ocr_engine:
            return "[ERRO] DeepSeek Engine não disponível (gpu/transformers error?)."
    else:
        ocr_engine = get_paddle_engine()
        if not ocr_engine:
            return "[ERRO] Paddle Engine não disponível."

    try:
        doc = fitz.open(pdf_path)
        
        for i, page in enumerate(doc):
            # --- FLUXO DEEPSEEK ---
            if engine == "deepseek":
                # Renderiza com alta qualidade
                zoom = 2.0 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    pix.save(tmp_img.name)
                    tmp_path = tmp_img.name
                
                try:
                    # DeepSeek lida com preprocessamento internamente
                    page_txt = ocr_engine.process_image(tmp_path)
                    full_text += f"\n--- Pag {i+1} (DeepSeek) ---\n{page_txt}"
                except Exception as e:
                    print(f"Erro DeepSeek Pag {i}: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)

            # --- FLUXO PADDLE (LEGADO) ---
            else:
                # Zoom=2 eqivale a ~144 DPI (padrão 72). Para garantir leitura, vamos usar 2.5 (aprox 180 DPI base) e aplicar SR se precisar.
                zoom = 2.0 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert fitz Pixmap to numpy array (OpenCV format)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: # RGBA -> RGB
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                else:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                
                # Check DPI/Details
                # Se a imagem for muito pequena (ex: thumbnail), aciona SuperRes
                apply_sr = False
                if pix.w < 1000 or pix.h < 1000:
                    apply_sr = True
                    
                # Pré-processamento
                final_img = preprocess_image(img_data, apply_superres=apply_sr)
                
                # Salva temp para Paddle (ele prefere path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    cv2.imwrite(tmp_img.name, final_img)
                    tmp_path = tmp_img.name
                    
                # OCR Execution
                try:
                    result = ocr_engine.ocr(tmp_path, cls=True)
                    if result and result[0]:
                        page_txt = "\n".join([line[1][0] for line in result[0]])
                        full_text += f"\n--- Pag {i+1} ---\n{page_txt}"
                except Exception as e:
                    print(f"Erro OCR Pag {i}: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                
    except Exception as e:
        return f"Erro Fatal no OCR Engine: {str(e)}"
        
    return full_text


class MistralDocumentAIEngine:
    """
    OCR engine using Mistral Document AI 2512 via Azure AI Foundry.
    Sends pages as base64 images to the API for structured document extraction.
    """
    
    ENDPOINT = os.getenv(
        "MISTRAL_DOC_AI_ENDPOINT",
        "https://assistente-web-resource.services.ai.azure.com/providers/mistral/azure/ocr"
    )
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for Mistral Document AI")
    
    def process_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Process an entire PDF (as bytes) through Mistral Document AI."""
        import base64
        
        b64_data = base64.b64encode(pdf_bytes).decode("utf-8")
        document_url = f"data:application/pdf;base64,{b64_data}"
        
        payload = {
            "model": "mistral-document-ai-2512",
            "document": {
                "type": "document_url",
                "document_url": document_url
            },
            "include_image_base64": False
        }
        
        response = requests.post(
            self.ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Mistral Document AI error {response.status_code}: {response.text[:300]}")
        
        result = response.json()
        
        # Extract text from response pages
        pages = result.get("pages", [])
        all_text = []
        for i, page in enumerate(pages):
            page_md = page.get("markdown", "")
            if page_md:
                all_text.append(f"\n--- Pag {i+1} (Mistral DocAI) ---\n{page_md}")
        
        return "\n".join(all_text) if all_text else ""
    
    def process_image_bytes(self, image_bytes: bytes, page_num: int = 1) -> str:
        """Process a single page image (PNG/JPEG bytes) through Mistral Document AI."""
        import base64
        
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_data}"
        
        payload = {
            "model": "mistral-document-ai-2512",
            "document": {
                "type": "image_url",
                "image_url": image_url
            },
            "include_image_base64": False
        }
        
        response = requests.post(
            self.ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Mistral Document AI error {response.status_code}: {response.text[:300]}")
        
        result = response.json()
        pages = result.get("pages", [])
        if pages:
            return pages[0].get("markdown", "")
        return ""


# Global singleton
MISTRAL_DOC_AI_ENGINE = None

def get_mistral_doc_ai_engine():
    global MISTRAL_DOC_AI_ENGINE
    if MISTRAL_DOC_AI_ENGINE is None:
        try:
            MISTRAL_DOC_AI_ENGINE = MistralDocumentAIEngine()
            print("✅ Mistral Document AI engine initialized")
        except Exception as e:
            print(f"⚠️ Erro ao iniciar Mistral Document AI: {e}")
            MISTRAL_DOC_AI_ENGINE = None
    return MISTRAL_DOC_AI_ENGINE


class MarkerEngine:
    """
    OCR engine using Marker (marker-pdf) for high-quality PDF → Markdown conversion.
    Preserves document structure, tables, headers, and formatting.
    Runs locally on CPU/GPU/MPS — no API costs.
    """

    def __init__(self):
        try:
            from marker.converters.pdf import PdfConverter
            from marker.config.parser import ConfigParser
        except ImportError:
            raise ImportError(
                "Marker requires 'marker-pdf'. Install with: pip install marker-pdf"
            )

        print("🚀 Initializing Marker PDF converter...")
        config = ConfigParser({"output_format": "markdown"})
        self.converter = PdfConverter(config=config)
        print("✅ Marker engine ready")

    def process_pdf(self, pdf_path: str) -> str:
        """Convert an entire PDF to Markdown (most efficient mode)."""
        rendered = self.converter(pdf_path)
        return rendered.markdown

    def process_page_image(self, png_bytes: bytes, page_num: int = 1) -> str:
        """
        Process a single page image through Marker.
        Falls back to saving as temp PDF and converting — Marker works best with PDFs.
        For single-page OCR, this creates a temp single-page PDF from the image.
        """
        try:
            # Create a single-page PDF from the image bytes using fitz
            single_doc = fitz.open()
            img_doc = fitz.open(stream=png_bytes, filetype="png")
            rect = img_doc[0].rect
            page = single_doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=png_bytes)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                single_doc.save(tmp_pdf.name)
                tmp_path = tmp_pdf.name

            single_doc.close()
            img_doc.close()

            try:
                result = self.process_pdf(tmp_path)
                return result
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            print(f"  ❌ Marker page {page_num} error: {e}")
            return ""


# Global singleton for Marker
MARKER_ENGINE = None

def get_marker_engine():
    global MARKER_ENGINE
    if MARKER_ENGINE is None:
        try:
            MARKER_ENGINE = MarkerEngine()
        except Exception as e:
            print(f"⚠️ Erro ao iniciar Marker engine: {e}")
            MARKER_ENGINE = None
    return MARKER_ENGINE

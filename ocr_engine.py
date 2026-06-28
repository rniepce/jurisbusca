"""
ocr_engine.py — OCR engines for PDF text extraction.

Available engines:
  - Marker (default): Local PDF → Markdown conversion. Preserves structure.
  - Mistral Document AI (optional): Azure API-based document extraction.
"""

import os
import tempfile
import threading
import requests

import fitz  # PyMuPDF

_singleton_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Mistral Document AI Engine (Azure API)
# ─────────────────────────────────────────────────────────────────────────────

class MistralDocumentAIEngine:
    """
    OCR engine using Mistral OCR (mistral-ocr-4-0) via Azure AI Foundry.
    Sends documents/pages as base64 to the API for structured document extraction.
    """

    # Endpoint do recurso Azure AI Foundry 'assistente-web-resource'.
    # Override via MISTRAL_DOC_AI_ENDPOINT em produção.
    ENDPOINT = os.getenv(
        "MISTRAL_DOC_AI_ENDPOINT",
        "https://assistente-web-resource.services.ai.azure.com/models"
    )

    # Deployment do modelo OCR. Override via MISTRAL_DOC_AI_MODEL para futuras versões.
    MODEL = os.getenv("MISTRAL_DOC_AI_MODEL", "mistral-ocr-4-0")

    def __init__(self):
        # Segredo SEMPRE via variável de ambiente (diretriz CESEC: nunca no código).
        # Mesma resource do Azure AI Foundry, então aceita a chave do Foundry/Azure OpenAI.
        self.api_key = (
            os.getenv("MISTRAL_DOC_AI_KEY", "").strip()
            or os.getenv("AZURE_API_KEY", "").strip()
            or os.getenv("AZURE_AI_FOUNDRY_KEY", "").strip()
            or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        )
        if not self.api_key:
            raise ValueError(
                "Chave de API necessária para Mistral OCR. "
                "Defina MISTRAL_DOC_AI_KEY (preferido), AZURE_AI_FOUNDRY_KEY ou AZURE_OPENAI_API_KEY."
            )
    
    def process_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Process an entire PDF (as bytes) through Mistral Document AI."""
        import base64
        
        b64_data = base64.b64encode(pdf_bytes).decode("utf-8")
        document_url = f"data:application/pdf;base64,{b64_data}"
        
        payload = {
            "model": self.MODEL,
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
        import time as _t

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_data}"

        payload = {
            "model": self.MODEL,
            "document": {
                "type": "image_url",
                "image_url": image_url
            },
            "include_image_base64": False
        }

        img_kb = len(image_bytes) // 1024
        print(f"  📤 Mistral DocAI pág {page_num}: enviando {img_kb} KB para {self.ENDPOINT}")
        t0 = _t.time()
        try:
            response = requests.post(
                self.ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                # Timeout reduzido: 8s para conectar + 25s para resposta.
                # Mistral DocAI normal responde em 3-8s; se passar disso, é
                # melhor cair para Tesseract do que pendurar o pipeline.
                timeout=(8, 25),
            )
        except requests.exceptions.Timeout as e:
            elapsed = _t.time() - t0
            raise RuntimeError(f"Mistral DocAI timeout ({elapsed:.1f}s) pág {page_num}") from e
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Mistral DocAI conexão falhou pág {page_num}: {e}") from e

        elapsed = _t.time() - t0
        print(f"  ✅ Mistral DocAI pág {page_num}: {response.status_code} em {elapsed:.1f}s")

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
        with _singleton_lock:
            if MISTRAL_DOC_AI_ENGINE is None:
                try:
                    MISTRAL_DOC_AI_ENGINE = MistralDocumentAIEngine()
                    print("✅ Mistral Document AI engine initialized")
                except Exception as e:
                    print(f"⚠️ Erro ao iniciar Mistral Document AI: {e}")
                    MISTRAL_DOC_AI_ENGINE = None
    return MISTRAL_DOC_AI_ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# Marker Engine (Local PDF → Markdown)
# ─────────────────────────────────────────────────────────────────────────────

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
        with _singleton_lock:
            if MARKER_ENGINE is None:
                try:
                    MARKER_ENGINE = MarkerEngine()
                except Exception as e:
                    print(f"⚠️ Erro ao iniciar Marker engine: {e}")
                    MARKER_ENGINE = None
    return MARKER_ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract Engine (Offline fallback OCR)
# ─────────────────────────────────────────────────────────────────────────────

class TesseractEngine:
    """
    Lightweight offline OCR using Tesseract.
    Fallback when API-based engines (Mistral DocAI) are unavailable.
    Requires: apt-get install tesseract-ocr tesseract-ocr-por
              pip install pytesseract pillow
    """

    def __init__(self):
        try:
            import pytesseract
            # Verify tesseract is installed
            pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
            print("✅ Tesseract OCR engine ready")
        except Exception as e:
            raise ImportError(f"Tesseract not available: {e}. Install with: apt-get install tesseract-ocr tesseract-ocr-por")

    def process_image_bytes(self, image_bytes: bytes, page_num: int = 1, lang: str = "por") -> str:
        """Process a single page image (PNG bytes) through Tesseract OCR."""
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        # Tesseract works best with grayscale
        if image.mode != "L":
            image = image.convert("L")

        text = self.pytesseract.image_to_string(image, lang=lang, config="--dpi 300")
        return text.strip()

    def process_pdf(self, pdf_path: str, lang: str = "por") -> str:
        """Process an entire PDF page-by-page through Tesseract."""
        all_text = []
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            # Render at 2x for quality
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            png_bytes = pix.tobytes("png")
            page_text = self.process_image_bytes(png_bytes, page_num=i + 1, lang=lang)
            if page_text:
                all_text.append(f"--- Pág {i + 1} ---\n{page_text}")
        doc.close()
        return "\n\n".join(all_text)


# Global singleton for Tesseract
TESSERACT_ENGINE = None

def get_tesseract_engine():
    global TESSERACT_ENGINE
    if TESSERACT_ENGINE is None:
        with _singleton_lock:
            if TESSERACT_ENGINE is None:
                try:
                    TESSERACT_ENGINE = TesseractEngine()
                except Exception as e:
                    print(f"⚠️ Erro ao iniciar Tesseract engine: {e}")
                    TESSERACT_ENGINE = None
    return TESSERACT_ENGINE

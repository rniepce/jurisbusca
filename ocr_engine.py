
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

# Initialize Global Paddle Engine
try:
    # use_angle_cls=True detects rotation
    PADDLE_ENGINE = PaddleOCR(use_angle_cls=True, lang='pt', show_log=False)
except Exception as e:
    print(f"⚠️ Erro ao iniciar PaddleOCR Engine: {e}")
    PADDLE_ENGINE = None

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
    Pipeline de Pré-processamento:
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

def extract_text_from_pdf(pdf_path):
    """
    Função Principal: PDF -> Imagens -> Tratamento -> OCR
    """
    if not PADDLE_ENGINE:
        return "[ERRO] Motor OCR não disponível."

    full_text = ""
    
    try:
        doc = fitz.open(pdf_path)
        
        for i, page in enumerate(doc):
            # Obtém imagem da página
            # Zoom=2 eqivale a ~144 DPI (padrão 72). Para garantir leitura, vamos usar 2.5 (aprox 180 DPI base) e aplicar SR se precisar.
            # Se usarmos zoom muito alto, SR fica muito lento. Zoom 2.0 é um bom equilíbrio.
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
                result = PADDLE_ENGINE.ocr(tmp_path, cls=True)
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

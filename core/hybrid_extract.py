"""
core/hybrid_extract.py — Page-level hybrid text extraction for mixed PDFs.

Processes each page individually:
  - Pages with sufficient text (≥50 chars): direct text extraction (instant)
  - Pages with little/no text (image-based): OCR at full resolution

Available OCR engines: "mistral_doc_ai" (default, Azure API), "marker" (local), or "tesseract" (offline fallback).
When Mistral DocAI fails, automatically falls back to Tesseract.

This is critical for Brazilian legal process PDFs which commonly mix
born-digital petitions with scanned evidence documents in a single file.
"""

import os
import time
from typing import Tuple

import fitz  # PyMuPDF

from langchain_core.documents import Document

# Minimum chars per page to consider it "born-digital text"
# Typical text page has >500 chars. Scanned pages have 0-10 (just metadata).
PAGE_TEXT_THRESHOLD = 50


def hybrid_extract(
    pdf_path: str,
    ocr_engine_choice: str = "mistral_doc_ai",
    compress: bool = True,
) -> Tuple[list, dict]:
    """
    Extract text from a PDF using page-level triage.

    For each page:
      - If text ≥ PAGE_TEXT_THRESHOLD chars → use extracted text directly
      - If text < PAGE_TEXT_THRESHOLD chars → render as image and run OCR

    Args:
        pdf_path:          Path to the PDF file.
        ocr_engine_choice: OCR engine to use for image pages ("marker" or "mistral_doc_ai").
        compress:          If True, compress the PDF for archival after extraction.
                           Compression never affects OCR quality.

    Returns:
        Tuple of (docs: list[Document], stats: dict)
        - docs: list of Document objects, one per page, with page metadata
        - stats: dict with 'text_pages', 'ocr_pages', 'total_chars', 'compressed_path'
    """
    t_start = time.time()

    stats = {
        "text_pages": 0,
        "ocr_pages": 0,
        "total_chars": 0,
        "total_pages": 0,
        "compressed_path": None,
        "elapsed_seconds": 0,
    }

    docs = []

    try:
        # ── Special path: Marker processes entire PDF at once (most efficient) ──
        if ocr_engine_choice == "marker":
            try:
                from ocr_engine import get_marker_engine
                engine = get_marker_engine()
                if engine:
                    full_md = engine.process_pdf(pdf_path)
                    if full_md:
                        doc = fitz.open(pdf_path)
                        stats["total_pages"] = len(doc)
                        stats["ocr_pages"] = len(doc)
                        stats["total_chars"] = len(full_md)
                        stats["elapsed_seconds"] = round(time.time() - t_start, 1)
                        doc.close()
                        docs.append(Document(
                            page_content=full_md,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": 0,
                                "extraction": "marker_full_pdf",
                            }
                        ))
                        print(
                            f"📊 Marker Extract: {stats['total_pages']} págs | "
                            f"{stats['total_chars']} chars | {stats['elapsed_seconds']}s"
                        )
                        return docs, stats
                else:
                    print("⚠️ Marker não disponível. Verifique a instalação do marker-pdf.")
                    return docs, stats
            except ImportError:
                print("⚠️ Marker não instalado. Instale com: pip install marker-pdf")
                return docs, stats

        # ── Page-by-page extraction (used by mistral_doc_ai) ──
        doc = fitz.open(pdf_path)
        stats["total_pages"] = len(doc)

        for i, page in enumerate(doc):
            page_num = i + 1

            # ── Step 1: Try direct text extraction (instant, ~0.001s) ──
            text = page.get_text("text").strip()

            if len(text) >= PAGE_TEXT_THRESHOLD:
                # ✅ Born-digital page — text is reliable
                stats["text_pages"] += 1
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "extraction": "text",
                    }
                ))
            else:
                # 🖼️ Image-based page — need OCR at full resolution
                stats["ocr_pages"] += 1
                ocr_text = _ocr_page(page, page_num, ocr_engine_choice)

                if ocr_text:
                    docs.append(Document(
                        page_content=ocr_text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "extraction": f"ocr_{ocr_engine_choice}",
                        }
                    ))
                else:
                    print(f"  ⚠️ Pág {page_num}: sem texto extraído (vazia ou ilegível)")

        doc.close()

    except Exception as e:
        print(f"❌ Erro fatal no hybrid_extract: {e}")
        import traceback
        traceback.print_exc()
        return docs, stats

    # ── Optional: compress for archival/cache (after extraction) ──
    if compress and docs:
        try:
            from core.compressor import compress_pdf
            compressed = compress_pdf(pdf_path, power=5)
            if compressed != pdf_path:
                orig = os.path.getsize(pdf_path)
                comp = os.path.getsize(compressed)
                ratio = (1 - comp / orig) * 100 if orig > 0 else 0
                print(f"📦 PDF comprimido para cache: {orig // 1024}KB → {comp // 1024}KB ({ratio:.0f}% menor)")
                stats["compressed_path"] = compressed
        except Exception as e:
            print(f"⚠️ Compressão pós-extração falhou (não-crítico): {e}")

    stats["total_chars"] = sum(len(d.page_content) for d in docs)
    stats["elapsed_seconds"] = round(time.time() - t_start, 1)

    # Summary log
    print(
        f"📊 Hybrid Extract: {stats['total_pages']} págs total | "
        f"{stats['text_pages']} texto | {stats['ocr_pages']} OCR | "
        f"{stats['total_chars']} chars | {stats['elapsed_seconds']}s"
    )

    return docs, stats


def _ocr_page(page, page_num: int, engine_choice: str) -> str:
    """
    Run OCR on a single fitz page using the specified engine.
    Renders the page as a high-resolution image and processes it.
    Falls back to Tesseract if the primary engine fails.
    """
    try:
        if engine_choice == "mistral_doc_ai":
            result = _ocr_page_mistral(page, page_num)
            if result:
                return result
            # Fallback to Tesseract if Mistral returns empty
            print(f"  ⚠️ Mistral vazia pág {page_num}, tentando Tesseract...")
            return _ocr_page_tesseract(page, page_num)
        elif engine_choice == "marker":
            return _ocr_page_marker(page, page_num)
        elif engine_choice == "tesseract":
            return _ocr_page_tesseract(page, page_num)
        else:
            # Default to Mistral DocAI → Tesseract fallback
            result = _ocr_page_mistral(page, page_num)
            return result if result else _ocr_page_tesseract(page, page_num)
    except Exception as e:
        print(f"  ⚠️ OCR primário falhou pág {page_num}: {e}. Tentando Tesseract...")
        try:
            return _ocr_page_tesseract(page, page_num)
        except Exception as e2:
            print(f"  ❌ Tesseract fallback também falhou pág {page_num}: {e2}")
            return ""


def _ocr_page_mistral(page, page_num: int) -> str:
    """OCR a single page with Mistral Document AI 2512 (Azure API)."""
    try:
        from ocr_engine import get_mistral_doc_ai_engine
    except ImportError as e:
        print(f"  ⚠️ Mistral Document AI não disponível: {e}")
        return ""

    engine = get_mistral_doc_ai_engine()
    if not engine:
        return ""

    # Render at 2x zoom for quality
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Get PNG bytes directly
    png_bytes = pix.tobytes("png")
    
    try:
        page_text = engine.process_image_bytes(png_bytes, page_num=page_num)
        return page_text
    except Exception as e:
        print(f"  ❌ Mistral DocAI erro pág {page_num}: {e}")
        return ""


def _ocr_page_marker(page, page_num: int) -> str:
    """OCR a single page with Marker (PDF→Markdown, local)."""
    try:
        from ocr_engine import get_marker_engine
    except ImportError as e:
        print(f"  ⚠️ Marker não disponível: {e}")
        return ""

    engine = get_marker_engine()
    if not engine:
        return ""

    # Render at 2x zoom for quality
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Get PNG bytes
    png_bytes = pix.tobytes("png")

    try:
        page_text = engine.process_page_image(png_bytes, page_num=page_num)
        return page_text
    except Exception as e:
        print(f"  ❌ Marker erro pág {page_num}: {e}")
        return ""


def _ocr_page_tesseract(page, page_num: int) -> str:
    """OCR a single page with Tesseract (offline fallback)."""
    try:
        from ocr_engine import get_tesseract_engine
    except ImportError as e:
        print(f"  ⚠️ Tesseract não disponível: {e}")
        return ""

    engine = get_tesseract_engine()
    if not engine:
        return ""

    # Render at 2x zoom for quality
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")

    try:
        page_text = engine.process_image_bytes(png_bytes, page_num=page_num)
        return page_text
    except Exception as e:
        print(f"  ❌ Tesseract erro pág {page_num}: {e}")
        return ""

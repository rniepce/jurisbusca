"""
Tests for core/compressor.py — PDF compression for LLM optimization.
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

# Create a minimal test PDF using pypdf (already in requirements)
try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def _create_test_pdf(text_content: str = "Teste de compressão PDF.") -> str:
    """Create a minimal PDF with text content and return its path."""
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)  # A4

    # Add text annotation (simulates text content)
    writer.add_metadata({
        "/Title": "Test PDF",
        "/Author": "test_compressor",
    })

    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    return pdf_path


GS_AVAILABLE = shutil.which("gs") is not None


# ── Test: Import and basic function signature ──

def test_import():
    """compressor module can be imported."""
    from core.compressor import compress_pdf, _gs_available
    assert callable(compress_pdf)
    assert callable(_gs_available)


# ── Test: File not found ──

def test_file_not_found():
    """Should raise FileNotFoundError for non-existent input."""
    from core.compressor import compress_pdf
    with pytest.raises(FileNotFoundError):
        compress_pdf("/tmp/does_not_exist_12345.pdf")


# ── Test: Graceful degradation when GS not installed ──

def test_graceful_degradation_no_gs():
    """When gs is not available, returns input_path unchanged."""
    from core.compressor import compress_pdf

    pdf_path = _create_test_pdf()
    try:
        with patch("core.compressor._gs_available", return_value=False):
            result = compress_pdf(pdf_path, power=5)
            assert result == pdf_path
    finally:
        os.remove(pdf_path)


# ── Test: Invalid power level ──

@pytest.mark.skipif(not GS_AVAILABLE, reason="Ghostscript not installed")
def test_invalid_power_level():
    """Invalid power levels should fallback to power=5."""
    from core.compressor import compress_pdf

    pdf_path = _create_test_pdf()
    try:
        result = compress_pdf(pdf_path, power=99)
        assert os.path.exists(result)
    finally:
        os.remove(pdf_path)
        if result != pdf_path and os.path.exists(result):
            os.remove(result)


# ── Test: Compression produces a valid file ──

@pytest.mark.skipif(not GS_AVAILABLE, reason="Ghostscript not installed")
def test_compression_produces_file():
    """Compressed PDF should exist and be non-empty."""
    from core.compressor import compress_pdf

    pdf_path = _create_test_pdf()
    try:
        result = compress_pdf(pdf_path, power=5)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
    finally:
        os.remove(pdf_path)
        if result != pdf_path and os.path.exists(result):
            os.remove(result)


# ── Test: All power levels produce output ──

@pytest.mark.skipif(not GS_AVAILABLE, reason="Ghostscript not installed")
@pytest.mark.parametrize("power", [1, 2, 3, 4, 5])
def test_all_power_levels(power):
    """Each power level should produce a valid output file."""
    from core.compressor import compress_pdf

    pdf_path = _create_test_pdf()
    try:
        result = compress_pdf(pdf_path, power=power)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
    finally:
        os.remove(pdf_path)
        if result != pdf_path and os.path.exists(result):
            os.remove(result)


# ── Test: Custom output path ──

@pytest.mark.skipif(not GS_AVAILABLE, reason="Ghostscript not installed")
def test_custom_output_path():
    """Should write to specified output path."""
    from core.compressor import compress_pdf

    pdf_path = _create_test_pdf()
    fd, out_path = tempfile.mkstemp(suffix="_custom.pdf")
    os.close(fd)

    try:
        result = compress_pdf(pdf_path, output_path=out_path, power=3)
        assert result == out_path
        assert os.path.exists(out_path)
    finally:
        os.remove(pdf_path)
        if os.path.exists(out_path):
            os.remove(out_path)


# ── Test: hybrid_extract import ──

def test_hybrid_extract_import():
    """hybrid_extract module can be imported."""
    from core.hybrid_extract import hybrid_extract
    assert callable(hybrid_extract)

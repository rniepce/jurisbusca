"""
core/compressor.py — Ghostscript-based PDF compressor for LLM optimization.

Reduces PDF file size by stripping high-resolution images while preserving
100% text fidelity. Ideal for legal documents with embedded logos, stamps,
and signatures that add bulk but no analytical value.

Power levels:
    1 = /prepress  (light, print-quality)
    2 = /printer   (moderate, standard print)
    3 = /ebook     (balanced, screen-quality)
    4 = /screen    (aggressive, low-res images)
    5 = LLM mode   (maximum compression, 36 DPI images, perfect text)
"""

import os
import shutil
import subprocess
import tempfile


def _gs_available() -> bool:
    """Check if Ghostscript is installed and accessible."""
    return shutil.which("gs") is not None


def compress_pdf(input_path: str, output_path: str = None, power: int = 5) -> str:
    """
    Compress a PDF using Ghostscript.

    Args:
        input_path:  Path to the original PDF.
        output_path: Optional destination path. Defaults to a temp file.
        power:       Compression level 1–5 (default=5, LLM mode).

    Returns:
        Path to the compressed PDF. Returns input_path unchanged if
        Ghostscript is not available or compression fails.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"PDF not found: {input_path}")

    if not _gs_available():
        print("⚠️ Ghostscript (gs) não encontrado. Retornando PDF original.")
        return input_path

    # Determine output path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_compressed.pdf")
        os.close(fd)

    # Base GS command
    gs_command = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-sOutputFile={output_path}",
    ]

    # Power-level presets (1–4)
    presets = {
        1: "/prepress",
        2: "/printer",
        3: "/ebook",
        4: "/screen",
    }

    if power in presets:
        gs_command.append(f"-dPDFSETTINGS={presets[power]}")

    elif power == 5:
        # ── Modo LLM: texto perfeito, imagens quase eliminadas ──
        gs_command.extend([
            "-dPDFSETTINGS=/screen",
            # Imagens coloridas → 36 DPI
            "-dDownsampleColorImages=true",
            "-dColorImageDownsampleType=/Bicubic",
            "-dColorImageResolution=36",
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/DCTEncode",
            # Imagens em escala de cinza → 36 DPI
            "-dDownsampleGrayImages=true",
            "-dGrayImageDownsampleType=/Bicubic",
            "-dGrayImageResolution=36",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/DCTEncode",
            # Imagens monocromáticas → 72 DPI
            "-dDownsampleMonoImages=true",
            "-dMonoImageDownsampleType=/Bicubic",
            "-dMonoImageResolution=72",
            # Preservar texto e fontes
            "-dEmbedAllFonts=true",
            "-dSubsetFonts=true",
            # Comprimir streams
            "-dCompressFonts=true",
            "-dCompressPages=true",
        ])
    else:
        print(f"⚠️ Power level {power} inválido (1–5). Usando power=5.")
        return compress_pdf(input_path, output_path, power=5)

    # Separator + input file (prevents GS from confusing path with flags)
    gs_command.extend(["-f", input_path])

    try:
        result = subprocess.run(
            gs_command,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min timeout for very large PDFs
        )

        if result.returncode != 0:
            print(f"⚠️ Ghostscript erro (code {result.returncode}): {result.stderr[:200]}")
            # Clean up failed output
            if os.path.exists(output_path):
                os.remove(output_path)
            return input_path

        # Sanity check: output should exist and not be empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("⚠️ Ghostscript produziu arquivo vazio.")
            return input_path

        return output_path

    except subprocess.TimeoutExpired:
        print("⚠️ Ghostscript timeout (>120s). Retornando PDF original.")
        if os.path.exists(output_path):
            os.remove(output_path)
        return input_path
    except Exception as e:
        print(f"⚠️ Erro inesperado no compressor: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return input_path

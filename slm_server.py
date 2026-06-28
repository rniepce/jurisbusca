"""
SLM Server — API dedicada para servir o pipeline de 5 SLMs via MLX.

Roda no MacBook Pro M3 Max (local), exposto via Cloudflare Tunnel ou ngrok
para que o Railway (cloud) possa fazer proxy das requisições.

Uso:
    python slm_server.py                     # porta 8001
    python slm_server.py --port 8001         # porta customizada
    python slm_server.py --key minha-chave   # autenticação

Endpoints:
    GET  /slm/health    — healthcheck (GPU, modelos, VRAM)
    POST /slm/pipeline  — executa pipeline completo (5 SLMs)
"""

import os
import time
import json
import hmac
import argparse
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── SLM imports ────────────────────────────────────────────────────────────────

try:
    from slm_orchestrator import SLMOrchestrator
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False
    print("⚠️ slm_orchestrator.py não encontrado.")

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="JurisBusca SLM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth key (shared secret) ──────────────────────────────────────────────────

SLM_KEY = os.environ.get("SLM_SERVER_KEY", "")

def _check_auth(x_slm_key: Optional[str] = None):
    """Valida a chave compartilhada. Falha FECHADO.

    Este servidor é exposto à internet via tunnel e processa autos sigilosos,
    então uma chave ausente NÃO pode significar "sem autenticação": se
    SLM_SERVER_KEY não estiver configurada, recusamos tudo. A comparação é
    constant-time para não vazar a chave por timing.
    """
    if not SLM_KEY:
        raise HTTPException(
            status_code=503,
            detail="Servidor sem autenticação configurada (defina SLM_SERVER_KEY).",
        )
    if not hmac.compare_digest(x_slm_key or "", SLM_KEY):
        raise HTTPException(status_code=401, detail="Chave SLM inválida")

# ── Orchestrator (lazy init) ──────────────────────────────────────────────────

_orchestrator: Optional[SLMOrchestrator] = None

def _get_orchestrator() -> SLMOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SLMOrchestrator(load_all=False)
    return _orchestrator


# ── Models ─────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    processo_text: str
    estilo: Optional[str] = ""


class PipelineResponse(BaseModel):
    minuta: str
    timing: dict
    audit: dict
    fatos: dict
    rota: dict
    model_info: dict
    total_seconds: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/slm/health")
async def health(x_slm_key: Optional[str] = Header(None)):
    """Healthcheck — retorna status da GPU, modelos disponíveis e VRAM."""
    _check_auth(x_slm_key)

    gpu_info = {}
    if HAS_MLX:
        try:
            info = mx.device_info(mx.gpu)
            gpu_info = {
                "device": str(mx.default_device()),
                "memory_gb": round(info.get("memory_size", 0) / 1e9, 1),
                "name": info.get("device_name", "unknown"),
            }
        except Exception:
            gpu_info = {"device": str(mx.default_device())}

    # Check adapters
    adapters = {}
    adapter_dir = os.path.join(os.path.dirname(__file__), "adapters")
    if os.path.isdir(adapter_dir):
        for name in os.listdir(adapter_dir):
            adapter_path = os.path.join(adapter_dir, name, "adapters.safetensors")
            if os.path.isfile(adapter_path):
                size_mb = round(os.path.getsize(adapter_path) / 1e6, 1)
                adapters[name] = {"size_mb": size_mb, "status": "ready"}

    return {
        "status": "online",
        "mlx": HAS_MLX,
        "orchestrator": HAS_ORCHESTRATOR,
        "gpu": gpu_info,
        "adapters": adapters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.post("/slm/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest, x_slm_key: Optional[str] = Header(None)):
    """Executa o pipeline completo de 5 SLMs e retorna a minuta + auditoria."""
    _check_auth(x_slm_key)

    if not HAS_ORCHESTRATOR:
        raise HTTPException(status_code=503, detail="SLM Orchestrator não disponível")

    try:
        orch = _get_orchestrator()
        t0 = time.time()

        result = orch.run_pipeline(
            processo_text=req.processo_text,
            estilo=req.estilo or "",
        )

        total = round(time.time() - t0, 1)

        return PipelineResponse(
            minuta=result.get("minuta", ""),
            timing=result.get("timing", {}),
            audit=result.get("audit", {}),
            fatos=result.get("fatos", {}),
            rota=result.get("rota", {}),
            model_info=result.get("model_info", {}),
            total_seconds=total,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no pipeline SLM: {str(e)[:500]}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JurisBusca SLM Server")
    parser.add_argument("--port", type=int, default=8001, help="Porta do servidor (default: 8001)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--key", default="", help="Chave de autenticação compartilhada")
    args = parser.parse_args()

    if args.key:
        SLM_KEY = args.key
        os.environ["SLM_SERVER_KEY"] = args.key
        print(f"🔑 Autenticação ativa (key={args.key[:4]}...)")

    print(f"""
╔══════════════════════════════════════════════════════╗
║  🧠 JurisBusca SLM Server                           ║
║                                                      ║
║  MLX: {'✅' if HAS_MLX else '❌'}  |  Orchestrator: {'✅' if HAS_ORCHESTRATOR else '❌'}             ║
║  Porta: {args.port}  |  Host: {args.host}                     ║
║                                                      ║
║  Endpoints:                                          ║
║    GET  /slm/health    — healthcheck                 ║
║    POST /slm/pipeline  — executa pipeline 5 SLMs     ║
║                                                      ║
║  Para expor via tunnel:                              ║
║    cloudflared tunnel run --url http://localhost:{args.port}  ║
║    ngrok http {args.port}                                    ║
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=args.host, port=args.port)

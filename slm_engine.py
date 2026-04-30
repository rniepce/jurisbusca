"""
SLM Engine — Inferência local via MLX (Apple Silicon).

Carrega modelos quantizados (Q4) e executa inferência localmente no M3 Max.
Suporta LoRA adapter swapping para trocar entre Jurista/Redator/Extrator/Auditor.

Fase 1: Modelo único (Mistral-Nemo-12B) para todas as tarefas.
Fase 2 (futuro): Pipeline multi-modelo com LoRA adapters especializados.
"""

import os
import time
import traceback
from typing import Optional

# ── MLX imports (Apple Silicon only) ──────────────────────────────────────────
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️ MLX não disponível. SLM Engine desabilitado. Instale com: pip install mlx mlx-lm")

# ── Model Configuration ──────────────────────────────────────────────────────

# Modelos quantizados para caber em 36GB M3 Max
MODELS = {
    # Modelo principal — análise jurídica, redação (~8GB)
    "main": {
        "repo": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "name": "Mistral-Nemo 12B (Q4)",
        "params": "12B",
        "vram_gb": 8,
    },
    # Modelo pequeno — extração de fatos, auditoria (~3GB)
    "small": {
        "repo": "mlx-community/gemma-3-4b-it-4bit",
        "name": "Gemma 3 4B (Q4)",
        "params": "4B",
        "vram_gb": 3,
    },
    # Modelo de roteamento — classificação rápida (~1.5GB)
    "router": {
        "repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "name": "Qwen 2.5 1.5B (Q4)",
        "params": "1.5B",
        "vram_gb": 1.5,
    },
}

# ── Global State ──────────────────────────────────────────────────────────────

_loaded_models = {}  # key -> (model, tokenizer)
_loading = False
_load_error = None


def is_available() -> bool:
    """Verifica se o MLX está disponível nesta máquina."""
    return HAS_MLX


def get_status() -> dict:
    """Retorna o status do engine para o frontend."""
    models_info = []
    for key, config in MODELS.items():
        loaded = key in _loaded_models
        models_info.append({
            "key": key,
            "name": config["name"],
            "repo": config["repo"],
            "params": config["params"],
            "vram_gb": config["vram_gb"],
            "loaded": loaded,
        })

    total_vram = sum(
        config["vram_gb"] for key, config in MODELS.items()
        if key in _loaded_models
    )

    return {
        "available": HAS_MLX,
        "loading": _loading,
        "error": _load_error,
        "models": models_info,
        "loaded_count": len(_loaded_models),
        "total_vram_gb": round(total_vram, 1),
    }


# ── LoRA Adapter Configuration ────────────────────────────────────────────────
# Maps pipeline roles to adapter directories (auto-populated from disk)
_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "adapters")

ADAPTER_MAP = {
    # model_key -> {role -> adapter_path}
    # Multiple roles can share the same base model with different adapters
    "router": "router",
    "main": None,          # Default: no adapter (uses base model)
    "small": None,
}

# Role-specific adapter overrides (set by orchestrator before generation)
_active_adapter = {}  # model_key -> adapter_path currently loaded


def _find_adapter(key: str, role: str = None) -> str | None:
    """Busca adapter LoRA treinado para o modelo/role."""
    # 1. Check role-specific adapter
    if role:
        role_path = os.path.join(_ADAPTER_DIR, role, "adapters.safetensors")
        if os.path.exists(role_path):
            return os.path.join(_ADAPTER_DIR, role)

    # 2. Check model key adapter
    mapped = ADAPTER_MAP.get(key)
    if mapped:
        key_path = os.path.join(_ADAPTER_DIR, mapped, "adapters.safetensors")
        if os.path.exists(key_path):
            return os.path.join(_ADAPTER_DIR, mapped)

    return None


def _load_model(key: str, adapter_role: str = None) -> tuple:
    """Carrega um modelo específico na memória, com LoRA adapter se disponível."""
    global _loading, _load_error

    if not HAS_MLX:
        raise RuntimeError("MLX não disponível. Requer Apple Silicon (M1/M2/M3).")

    # Check if model is already loaded (without adapter change)
    adapter_path = _find_adapter(key, adapter_role)
    if key in _loaded_models and _active_adapter.get(key) == adapter_path:
        return _loaded_models[key]

    config = MODELS.get(key)
    if not config:
        raise ValueError(f"Modelo '{key}' não encontrado. Opções: {list(MODELS.keys())}")

    _loading = True
    _load_error = None

    try:
        print(f"🔄 Carregando {config['name']} ({config['repo']})...")
        start = time.time()

        model, tokenizer = mlx_load(config["repo"])

        # Apply LoRA adapter if found
        if adapter_path:
            try:
                from mlx_lm.tuner.utils import apply_lora_layers
                adapter_file = os.path.join(adapter_path, "adapters.safetensors")
                model = apply_lora_layers(model, adapter_file)
                # Fuse LoRA weights for faster inference
                model.eval()
                print(f"  🧬 LoRA adapter aplicado: {adapter_path}")
            except ImportError:
                # Fallback: try loading with adapter config
                try:
                    model, tokenizer = mlx_load(config["repo"], adapter_path=adapter_path)
                    print(f"  🧬 LoRA adapter carregado via config: {adapter_path}")
                except Exception as e2:
                    print(f"  ⚠️ Falha ao aplicar adapter (usando base): {e2}")

        elapsed = time.time() - start
        adapter_tag = f" + LoRA({os.path.basename(adapter_path)})" if adapter_path else ""
        print(f"✅ {config['name']}{adapter_tag} carregado em {elapsed:.1f}s (~{config['vram_gb']}GB VRAM)")

        _loaded_models[key] = (model, tokenizer)
        _active_adapter[key] = adapter_path
        _loading = False
        return model, tokenizer

    except Exception as e:
        _loading = False
        _load_error = str(e)
        print(f"❌ Erro ao carregar {config['name']}: {e}")
        traceback.print_exc()
        raise


def ensure_loaded(key: str = "main"):
    """Garante que o modelo está carregado. Lazy loading."""
    if key not in _loaded_models:
        _load_model(key)


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_key: str = "main",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Gera texto usando o modelo local.

    Args:
        prompt: Mensagem do usuário
        system_prompt: System prompt (instruções do agente)
        model_key: Qual modelo usar ("main", "small", "router")
        max_tokens: Máximo de tokens a gerar
        temperature: Temperatura para sampling
        top_p: Top-p para nucleus sampling

    Returns:
        Texto gerado pelo modelo
    """
    if not HAS_MLX:
        raise RuntimeError("MLX não disponível. Requer Apple Silicon.")

    # Lazy load
    model, tokenizer = _load_model(model_key)

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    start = time.time()

    # Generate with mlx-lm
    response = mlx_generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        verbose=False,
    )

    elapsed = time.time() - start
    token_count = len(tokenizer.encode(response))
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    print(f"🤖 SLM gerou {token_count} tokens em {elapsed:.1f}s ({tokens_per_sec:.0f} tok/s)")

    return response


def generate_chat(
    messages: list[dict],
    model_key: str = "main",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Gera texto a partir de uma lista de mensagens (formato ChatML).

    Args:
        messages: Lista de dicts [{"role": "system"/"user"/"assistant", "content": "..."}]
        model_key: Qual modelo usar
        max_tokens: Máximo de tokens gerados
        temperature: Controlador de criatividade
        top_p: Nucleus sampling

    Returns:
        Texto gerado pelo modelo
    """
    if not HAS_MLX:
        raise RuntimeError("MLX não disponível. Requer Apple Silicon.")

    model, tokenizer = _load_model(model_key)

    # Apply chat template
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    start = time.time()

    response = mlx_generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        verbose=False,
    )

    elapsed = time.time() - start
    token_count = len(tokenizer.encode(response))
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    print(f"🤖 SLM gerou {token_count} tokens em {elapsed:.1f}s ({tokens_per_sec:.0f} tok/s)")

    return response


def unload(key: str = None):
    """Descarrega modelo(s) da memória para liberar VRAM."""
    global _loaded_models
    if key:
        if key in _loaded_models:
            del _loaded_models[key]
            print(f"🗑️ Modelo '{key}' descarregado.")
    else:
        _loaded_models.clear()
        print("🗑️ Todos os modelos descarregados.")

    # Force garbage collection
    import gc
    gc.collect()
    if HAS_MLX:
        mx.metal.clear_cache()

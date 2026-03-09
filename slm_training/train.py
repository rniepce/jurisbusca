"""
SLM Training — Wrapper para fine-tuning LoRA via MLX.

Uso:
    python slm_training/train.py --config slm_training/configs/extrator_config.yaml

Requer: pip install mlx mlx-lm (Apple Silicon only)
"""

import argparse
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA de SLM via MLX")
    parser.add_argument("--config", required=True, help="Arquivo YAML de configuração")
    parser.add_argument("--dry-run", action="store_true", help="Apenas valida config, não treina")
    args = parser.parse_args()

    # Carregar config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"🔧 Config carregada: {args.config}")
    print(f"   Modelo: {config.get('model', '?')}")
    print(f"   Dados: {config.get('data', '?')}")
    print(f"   Iterações: {config.get('iters', '?')}")
    print(f"   LoRA rank: {config.get('lora_parameters', {}).get('rank', '?')}")
    print(f"   Adapter path: {config.get('adapter_path', '?')}")

    if args.dry_run:
        print("\n✅ Dry run — config válida. Não iniciou treino.")
        return

    # Verificar se MLX está disponível
    try:
        import mlx.core as mx
        print(f"\n🖥️ MLX backend: {mx.default_device()}")
    except ImportError:
        print("❌ MLX não disponível. Instale: pip install mlx mlx-lm")
        sys.exit(1)

    # Verificar se dados existem
    import os
    data_dir = config.get("data", "data")
    train_file = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        print(f"❌ Arquivo de treino não encontrado: {train_file}")
        print("   Execute prepare_dataset.py primeiro.")
        sys.exit(1)

    # Contar exemplos
    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    print(f"📊 Exemplos de treino: {n_train}")

    # Executar fine-tuning via mlx_lm.lora
    print("\n🚀 Iniciando fine-tuning LoRA...")
    print(f"   Estimativa: ~{config.get('iters', 600) * 1.5 / 60:.0f} min no M3 Max")

    # Construir comando mlx_lm lora (nova syntax)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", config["model"],
        "--data", config["data"],
        "--train",
        "--batch-size", str(config.get("batch_size", 1)),
        "--iters", str(config.get("iters", 600)),
        "--learning-rate", str(config.get("learning_rate", 1e-5)),
        "--num-layers", str(config.get("lora_layers", 16)),
        "--steps-per-report", str(config.get("steps_per_report", 10)),
        "--steps-per-eval", str(config.get("steps_per_eval", 200)),
        "--val-batches", str(config.get("val_batches", 25)),
        "--adapter-path", config.get("adapter_path", "adapters"),
        "--save-every", str(config.get("save_every", 100)),
        "--seed", str(config.get("seed", 42)),
    ]

    if config.get("resume_adapter_file"):
        cmd.extend(["--resume-adapter-file", config["resume_adapter_file"]])

    print(f"   Comando: {' '.join(cmd)}")

    import subprocess
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if result.returncode == 0:
        print(f"\n✅ Fine-tuning concluído! Adapter salvo em: {config.get('adapter_path', 'adapters')}")
    else:
        print(f"\n❌ Fine-tuning falhou (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

#!/bin/bash
set -e

# Ativa ambiente virtual (caso esteja rodando fora)
source ./venv/bin/activate || true

echo "📂 Verificando Dataset..."
if [ ! -f "data/train.jsonl" ]; then
    echo "⚠️  data/train.jsonl não encontrado. Tentando extrair..."
    python extract_dataset.py
else
    echo "✅ Dataset encontrado."
fi

echo "================================================"
echo "🚀 INICIANDO FINE-TUNING EM SÉRIE"
echo "================================================"

echo ""
echo "🧠 [1/3] Treinando Mistral Nemo 12B..."
python -m mlx_lm.lora --config configs/mistral_nemo.yaml
echo "✅ Mistral Nemo finalizado!"

echo ""
echo "🦙 [2/3] Treinando Llama 3.1 8B..."
python -m mlx_lm.lora --config configs/llama3_1.yaml
echo "✅ Llama 3.1 finalizado!"

echo ""
echo "💎 [3/3] Treinando Gemma 2 27B..."
echo "⚠️  Nota: Este modelo é pesado. Se faltar memória, reduza batch_size no config."
python -m mlx_lm.lora --config configs/gemma2_27b.yaml
echo "✅ Gemma 2 finalizado!"

echo "================================================"
echo "🎉 TODOS OS TREINOS CONCLUÍDOS COM SUCESSO!"
echo "Os adaptadores (LoRA) estão salvos na pasta 'adapters/'"

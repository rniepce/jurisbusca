# 📊 Benchmark: Pipeline SLM Local (5 Modelos)
**Data:** 2026-03-09 10:44
**Hardware:** MacBook Pro M3 Max (36GB RAM)
**Processos testados:** 5 (TJMG — 4ª Vara Cível de Betim)

## Modelos Fine-Tuned
| Modelo | Base | LoRA Loss | VRAM |
|--------|------|-----------|------|
| Router | Qwen 2.5 1.5B (Q4) | 2.347→0.725 | 1.5GB |
| Extrator | Gemma 3 4B (Q4) | 3.137→0.179 | 3GB |
| Jurista/Redator | Mistral-Nemo 12B (Q4) | 1.680→~0.5 | 8GB |
| Auditor | Gemma 3 4B (Q4) | 3.395→~0.3 | 3GB |

## Resultados por Processo
| # | Processo | Tipo | Latência | Audit Score | Aprovado | Tam. Minuta |
|---|----------|------|----------|-------------|----------|-------------|
| 1 | 5004569-64 | sentenca | 62.4s | 95 | ✅ | 1474 chars |
| 2 | 5015929-93 | despacho | 85.2s | 60 | ❌ | 2359 chars |
| 3 | 5020203-03 | saneamento | 55.6s | 95 | ✅ | 2031 chars |
| 4 | 5022970-82 | sentenca | 72.9s | 60 | ❌ | 1584 chars |
| 5 | 5026970-57 | saneamento | 42.2s | 95 | ✅ | 1452 chars |

**Latência média:** 63.7s
**Score médio:** 81/100
**Aprovação:** 3/5 (60%)

## Timing por Estágio (média)
| Estágio | Modelo | Tempo médio |
|---------|--------|-------------|
| Router | Qwen 2.5 1.5B | 1.3s |
| Extrator | Gemma 3 4B | 7.2s |
| Jurista | Mistral-Nemo 12B | 24.1s |
| Redator | Mistral-Nemo 12B | 16.6s |
| Auditor | Gemma 3 4B | 4.6s |

## Previews das Minutas

### Processo 1: 5004569-64


### Processo 2: 5015929-93


### Processo 3: 5020203-03


### Processo 4: 5022970-82


### Processo 5: 5026970-57


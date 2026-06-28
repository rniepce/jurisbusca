# Plano — Harness com Perfis de Custo (modelos baratos em nuvem)

> **Objetivo:** transformar o fluxo de análise de processo numa *harness* única em que o
> modelo é um parâmetro. O usuário sobe o processo, o fluxo inteiro roda, e trocar de um
> "perfil de custo" faz tudo rodar num tier mais barato — **sem reescrever o fluxo nem o código de orquestração.**
>
> **Escopo desta fase (decidido):** apenas o perfil **econômico em nuvem** (gemini-flash / deepseek).
> Nada depende do MacBook / MLX. O perfil `local` fica como gancho para depois.
>
> **Status:** plano para revisão. Nenhuma linha de código alterada ainda.

---

## 0. Conceito: separar **Engine** de **Perfil**

Hoje "qual modelo" e "qual pipeline" estão misturados. Vamos separar em dois eixos ortogonais:

| Eixo | O que é | Valores |
|---|---|---|
| **Engine** | Qual orquestração roda (a harness) | `v1`, `v2`, `v3`, `flow` |
| **Perfil** | Quais modelos a engine usa em cada etapa | `premium`, `economico` |

A mesma engine V3 roda em `premium` (Kimi→DeepSeek→GPT) ou `economico` (mini→DeepSeek→mini).
Só muda o **perfil**.

**A chave "Harness" no chat é o liga/desliga do perfil:** desligada → `premium` (hoje), ligada →
`economico`. É **opt-in**: nasce desligada.

---

## 1. Como a escolha de modelo está espalhada hoje (confirmado no código)

| Local | Linha | Como escolhe o modelo | Acoplamento |
|---|---|---|---|
| `flow_engine._run_agent` | [156](flow_engine.py:156) | `cfg.get("model") or "gpt-5.3-chat"` | ✅ por nó |
| `flow_engine._run_switch` | [190](flow_engine.py:190) | `cfg.get("model") or "gpt-5.4-mini"` | ✅ por nó |
| `orchestrator_v3.node_kimi_reader` | [61](v3_engine/orchestrator_v3.py:61) | string fixa `"Kimi-K2.5"` (+ fallback GPT) | ❌ hardcoded |
| `orchestrator_v3.node_deepseek_reasoner` | [106](v3_engine/orchestrator_v3.py:106) | string fixa `"DeepSeek-V3.2-Speciale"` | ❌ hardcoded |
| `orchestrator_v3.node_gpt_formatter` | [195](v3_engine/orchestrator_v3.py:195) | string fixa `"gpt-5.3-chat"` | ❌ hardcoded |
| `run_autonomous_magistrate` | [275](v3_engine/orchestrator_v3.py:275) | recebe `model_name` mas **ignora** | ❌ |
| `process_single_case_pipeline` | [2234](backend.py:2234) | `run_autonomous_magistrate(clean_content, keys)` | ❌ sem perfil |
| `api_server` (chat) | [778](api_server.py:778) | `submit(be.run_autonomous_magistrate, full_text, keys, model_name)` | ❌ sem perfil |

O ponto-chave: **`backend.get_llm()` ([536](backend.py:536)) já é a factory universal** — ela roteia por prefixo do nome (`gemini*` / `claude*` / `deepseek`/`grok`/`kimi` via Azure Foundry / resto Azure OpenAI). **Não vamos tocá-la.** O resolvedor só decide *qual nome* passar pra ela.

---

## 2. Mapa de modelos — DECIDIDO

Regra de ouro: **a economia vem dos passos leves (gpt-5.4-mini), não de baratear o raciocínio.**
O `reasoner` — único passo que não decompõe — usa o melhor raciocinador em ambos os perfis.

| Etapa (papel) | `premium` | `economico` | Por quê |
|---|---|---|---|
| `reader` (extrair fatos) | Kimi-K2.5 | **gpt-5.4-mini** | leitura/extração; barato e já no Azure |
| `reasoner` (julgar) | **DeepSeek-V4-Pro** | **DeepSeek-V4-Pro** | raciocínio + tool loop; o passo que não se baratea |
| `formatter` (JSON/markdown) | gpt-5.3-chat | **gpt-5.4-mini** | só envelopa o texto |
| `classifier` (router/switch) | gpt-5.4-mini | **gpt-5.4-mini** | classificação curta |
| `style` (estilo) | claude-sonnet-4-6 | **gpt-5.4-mini** | reescrita por imitação |

**Strings confirmadas no seu código:**
- `gpt-5.4-mini` — já usado ([backend.py:1398](backend.py:1398), [1775](backend.py:1775)) e tarifado em [flow_pricing.py:14](flow_pricing.py:14). Branch Azure OpenAI do `get_llm`.
- `DeepSeek-V4-Pro` — nome exato confirmado em [flow_pricing.py:31](flow_pricing.py:31). Branch Azure AI Foundry ([backend.py:633](backend.py:633)), que passa o nome direto como `model`.

> 📌 **Mudança no premium:** o `reasoner` premium sobe de `DeepSeek-V3.2-Speciale` → `DeepSeek-V4-Pro`
> (o melhor raciocinador). Reversível em 1 linha se preferir manter o premium intocado. Existe ainda
> `DeepSeek-V4-Flash` ([flow_pricing.py:32](flow_pricing.py:32), ~12× mais barato) para testar baratear
> o reasoner no futuro.

---

## 3. Edições propostas (arquivo → linha → diff)

### 3.1 NOVO arquivo: `model_profiles.py`

O coração de todo o desacoplamento. ~45 linhas.

```python
"""
Resolvedor central de modelos por (etapa, perfil de custo).

Todas as engines (flow_engine, V3, pipeline) passam a perguntar a este módulo
"qual modelo uso na etapa X no perfil Y?" em vez de fixar nomes no código.
Trocar o perfil re-roda o mesmo fluxo num tier de custo diferente.

Os nomes retornados são strings que backend.get_llm() já entende
(gemini* / claude* / deepseek|grok|kimi via Azure Foundry / resto Azure OpenAI).
"""

DEFAULT_PROFILE = "premium"

PROFILES = {
    "premium": {
        "reader":     "Kimi-K2.5",
        "reasoner":   "DeepSeek-V4-Pro",
        "formatter":  "gpt-5.3-chat",
        "classifier": "gpt-5.4-mini",
        "style":      "claude-sonnet-4-6",
        "default":    "gpt-5.3-chat",
    },
    "economico": {                       # só modelos baratos no Azure (opt-in)
        "reader":     "gpt-5.4-mini",
        "reasoner":   "DeepSeek-V4-Pro",   # forte SÓ no raciocínio (não se baratea)
        "formatter":  "gpt-5.4-mini",
        "classifier": "gpt-5.4-mini",
        "style":      "gpt-5.4-mini",
        "default":    "gpt-5.4-mini",
    },
}

def resolve(stage: str, profile: str | None = None) -> str:
    p = PROFILES.get(profile or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
    return p.get(stage, p["default"])

def list_profiles() -> list[str]:
    return list(PROFILES.keys())
```

### 3.2 `flow_engine.py` — nós resolvem "papéis" (`@reader`) em vez de nomes fixos

**Topo do arquivo** (após os imports existentes): adicionar
```python
import model_profiles as mp
```

**`_run_agent` — linha [156](flow_engine.py:156):**
```diff
-    model = cfg.get("model") or "gpt-5.3-chat"
+    raw_model = cfg.get("model") or "@default"
+    profile = state.get("_profile")
+    model = mp.resolve(raw_model[1:], profile) if isinstance(raw_model, str) and raw_model.startswith("@") else raw_model
```

**`_run_switch` — linha [190](flow_engine.py:190):** mesmo padrão, default `@classifier`:
```diff
-    model = cfg.get("model") or "gpt-5.4-mini"
+    raw_model = cfg.get("model") or "@classifier"
+    profile = state.get("_profile")
+    model = mp.resolve(raw_model[1:], profile) if isinstance(raw_model, str) and raw_model.startswith("@") else raw_model
```

> Mesmo retoque (1 bloco) nos handlers `extractor` e `estilo`, se quisermos que também
> herdem o perfil. Listo os pontos exatos na implementação.

**Comportamento:** nada quebra. Um nó com `model="gpt-5.3-chat"` continua igual. Um nó
com `model="@reasoner"` (ou sem model) passa a respeitar o perfil ativo. O perfil chega via
`initial_state` — e `build_and_run_flow` já faz `state = dict(initial_state or {})`
([656](flow_engine.py:656)), então **nenhuma mudança de assinatura é necessária aqui.**

### 3.3 `v3_engine/orchestrator_v3.py` — remover hardcode

**Topo:** `import model_profiles as mp` (ao lado do `import backend as be`).

**`MagistrateState`** (TypedDict, ~linha 24): adicionar campo opcional
```diff
+    profile: str
```

**`run_autonomous_magistrate` — linha [275](v3_engine/orchestrator_v3.py:275):**
```diff
-def run_autonomous_magistrate(text: str, keys: dict, model_name: str = "v3-moe"):
+def run_autonomous_magistrate(text: str, keys: dict, model_name: str = "v3-moe", profile: str | None = None):
```
e no `initial_state` ([284](v3_engine/orchestrator_v3.py:284)):
```diff
     initial_state = {
         "raw_text": text,
         "keys": keys,
+        "profile": profile,
         ...
```

**`node_kimi_reader` — linha [61](v3_engine/orchestrator_v3.py:61):**
```diff
-        llm = be.get_llm(model_name="Kimi-K2.5", temperature=0.0)
+        llm = be.get_llm(model_name=mp.resolve("reader", state.get("profile")), temperature=0.0)
```

**`node_deepseek_reasoner` — linha [106](v3_engine/orchestrator_v3.py:106):**
```diff
-        llm = be.get_llm(model_name="DeepSeek-V3.2-Speciale", temperature=0.1)
+        llm = be.get_llm(model_name=mp.resolve("reasoner", state.get("profile")), temperature=0.1)
```

**`node_gpt_formatter` — linha [195](v3_engine/orchestrator_v3.py:195):**
```diff
-    llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.0)
+    llm = be.get_llm(model_name=mp.resolve("formatter", state.get("profile")), temperature=0.0)
```

> Os **fallbacks** internos (`gpt-5.3-chat` quando Kimi/DeepSeek falham) podem ficar como
> estão (rede de segurança) ou também sair do perfil — recomendo deixar como estão nesta fase.
>
> ⚠️ **Importante:** no perfil econômico, o `reasoner` **continua DeepSeek** justamente porque
> ele usa *tool calling* (`bind_tools` em [107](v3_engine/orchestrator_v3.py:107)) com o
> LegalREPL. Trocar o raciocinador por um modelo sem bom suporte a ferramentas quebraria o loop
> anti-alucinação. `reader` e `formatter` não usam ferramentas → flash serve bem.

### 3.4 `backend.py` — pipeline em lote repassa o perfil

**`process_single_case_pipeline` — assinatura [2080](backend.py:2080):**
```diff
-def process_single_case_pipeline(pdf_bytes, filename, api_key, template_files=None, cached_text=None, mode="v1", keys=None, ocr_engine_choice="marker"):
+def process_single_case_pipeline(pdf_bytes, filename, api_key, template_files=None, cached_text=None, mode="v1", keys=None, ocr_engine_choice="marker", profile="premium"):
```

**Chamada do V3 — linha [2234](backend.py:2234):**
```diff
-            v3_json, v3_logs = run_autonomous_magistrate(clean_content, keys)
+            v3_json, v3_logs = run_autonomous_magistrate(clean_content, keys, profile=profile)
```

### 3.5 `api_server.py` — expor o perfil nas rotas

**Chat (V3) — linha [778](api_server.py:778):** o request precisa de um campo `profile`
(adicionar `profile: str = "premium"` no modelo Pydantic do chat). Depois:
```diff
-                        future = executor.submit(be.run_autonomous_magistrate, full_text, keys, model_name)
+                        future = executor.submit(be.run_autonomous_magistrate, full_text, keys, model_name, req.profile)
```

**Fluxos (flow_engine) — endpoint `/api/flows/{flow_id}/run` [3837](api_server.py:3837):**
o `initial_state` é montado em [3851](api_server.py:3851). Injetar o perfil vindo do body:
```diff
     initial_state = _flow_run_initial_state(user_id, token)
+    initial_state["_profile"] = body.get("profile", "premium")
```

### 3.6 Frontend — chave "Harness" liga/desliga no chat

> ⚠️ **Descoberta na implementação:** o seletor de modelo do chat só tem modelos únicos
> (deepseek/gpt53/gemini/claude/local) — **não existe opção "V3"**. O chat hoje roda **1 modelo só**
> ([api_server.py:510](api_server.py:510)); o branch `req.model == "v3"` ([api_server.py:762](api_server.py:762))
> nunca é acionado pela UI. Logo, a chave **não pode apenas setar `profile`** (não teria efeito).
> Ela **roteia a mensagem pelo pipeline V3** quando ligada.

- **OFF (default)** → chat normal: `model = selectedModel.id`, `profile = "premium"`.
- **ON** → `model = "v3"` + `profile = "economico"` — a mensagem roda pelo pipeline de análise
  completo (V3) com modelos baratos.
- Implementado em: estado global `harnessEconomico` em [store/index.ts](frontend/src/store/index.ts);
  toggle em [ChatInput.tsx](frontend/src/components/ChatInput.tsx); roteamento em
  [useMessageSender.ts](frontend/src/hooks/useMessageSender.ts); `profile` adicionado ao
  `ChatRequest` ([api_server.py](api_server.py)) e ao `sendMessage` ([api.ts](frontend/src/services/api.ts)).

---

## 4. Ancoragem de fatos (o que segura a qualidade no tier barato)

Modelo barato sem fatos ancorados **alucina** — inaceitável em decisão judicial. Mitigações já
disponíveis no código:

1. **V3 mantém o LegalREPL** ([v3_engine/tools/legal_repl.py](v3_engine/tools/legal_repl.py)):
   `search_dates`, `search_money`, `search_parties`, `grep`. Como o `reasoner` segue sendo
   DeepSeek (com tools), o anti-alucinação **continua ativo** no perfil econômico. ✅
2. **Fluxos do flow_engine:** todo nó `agent` de geração deve ser precedido por nós
   `extractor` / `juris` / `modelo` (determinísticos) que injetam os trechos exatos. Boa prática
   a documentar ao montar o "fluxo padrão de análise".
3. *(Opcional, fase futura)* promover o LegalREPL a um **nó `tool`** do flow_engine, para que
   fluxos visuais baratos também tenham busca ancorada.

---

## 5. Validação (provar que o econômico não degradou) — sem código novo

Você já tem as duas peças:

- **ArenaPanel** (A/B cego entre modelos): rodar a **mesma** minuta em `premium` vs `economico`
  e comparar lado a lado.
- **MCP `goldendata`** (`obter_indicadores_qualidade`, `decidir_gate`): montar um *gate* — o
  perfil econômico só é "aprovado" se atingir um limiar de qualidade num conjunto-ouro de
  processos representativos.
- **Custo já instrumentado:** o flow_engine emite `input_tokens`/`output_tokens`/`cost_usd` por nó
  (eventos `node_done`), e [flow_pricing.py](flow_pricing.py) já tarifa `gpt-5.4-mini`, `DeepSeek-V4-Pro`
  e `DeepSeek-V4-Flash`. Dá pra ver a **queda de custo por execução** (premium vs econômico) na tela.

Fluxo de aceite sugerido: 10–20 processos variados → roda nos dois perfis → Arena/goldendata →
se o econômico ficar dentro do limiar, vira o default para os tipos de caso aprovados.

---

## 6. Ordem de implementação sugerida

| # | Entrega | Arquivos | Risco |
|---|---|---|---|
| 1 | `model_profiles.py` | novo | nenhum (módulo isolado) |
| 2 | Desacoplar V3 | `orchestrator_v3.py` | baixo (fallbacks mantêm rede de segurança) |
| 3 | flow_engine resolve `@papel` | `flow_engine.py` | baixo (retrocompatível) |
| 4 | Repasse de `profile` na API/pipeline | `api_server.py`, `backend.py` | baixo |
| 5 | Seletor no frontend | `api.ts`, painel | baixo (cosmético) |
| 6 | Rodada de avaliação | Arena + goldendata | — |

**Sugestão:** implementar 1–3 primeiro e te mostrar o V3 rodando idêntico em `premium` e
visivelmente mais barato em `economico` (mesmo processo, dois perfis) antes de mexer em UI.

---

## 7. Decisões (fechadas)

1. ✅ **Passos leves:** `gpt-5.4-mini` (já provisionado no Azure).
2. ✅ **Reasoner:** `DeepSeek-V4-Pro` (em ambos os perfis).
3. ✅ **Seletor:** chave liga/desliga **"Harness"** no **chat** (opt-in, nasce desligado).
4. ✅ **Econômico não vira default** automaticamente.

Único ponto em aberto: subir o `reasoner` **premium** de `DeepSeek-V3.2-Speciale` → `DeepSeek-V4-Pro`
(proposto) ou manter o premium intocado.

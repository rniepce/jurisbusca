"""Testes dos endpoints de Sustentação Oral / Audiência.

Roda com: pytest test_sustentacao.py -v

Mocka o LLM (be.get_llm) para evitar chamadas reais à Azure OpenAI e
sobrescreve require_auth para dispensar JWT em ambiente de teste.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import api_server
from api_server import app, require_auth


# ── Fixtures ────────────────────────────────────────────────────────────────

DEFAULT_USER = {"sub": "test-user"}


@pytest.fixture(autouse=True)
def _bypass_auth():
    """Sobrescreve require_auth em todos os testes deste módulo."""
    app.dependency_overrides[require_auth] = lambda: DEFAULT_USER
    yield
    app.dependency_overrides.pop(require_auth, None)


@pytest.fixture(autouse=True)
def _clear_store():
    """Garante store limpo entre testes."""
    api_server._sustentacao_store.clear()
    yield
    api_server._sustentacao_store.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _mock_llm_returning(payload):
    """Cria um mock de LLM cujo .invoke() retorna um objeto com .content = JSON."""
    mock = MagicMock()
    response = MagicMock()
    response.content = json.dumps(payload) if isinstance(payload, (dict, list)) else payload
    mock.invoke.return_value = response
    return mock


def _seed_processo(process_id: str, *, user_id: str = "test-user", **extras):
    """Helper: insere um processo no store já com user_id correto."""
    api_server._sustentacao_store[process_id] = {
        "user_id": user_id,
        "text": extras.get("text", "Texto do processo..."),
        "data": extras.get("data", {}),
        "tipo_ato": extras.get("tipo_ato", "sustentacao"),
        "modo": extras.get("modo", "preparacao"),
    }


def _set_user(user: dict):
    """Atalho: troca o usuário autenticado para o teste corrente."""
    app.dependency_overrides[require_auth] = lambda: user


# ── /api/sustentacao/extract ────────────────────────────────────────────────

def test_extract_sustentacao_preparacao_ok(client):
    payload = {
        "numero_processo": "1234567-89.2024.8.13.0024",
        "tipo_recursal": "Apelação Cível",
        "camara": "5ª Câmara Cível",
        "relator": "Des. Fulano",
        "teses": ["Tese 1", "Tese 2"],
        "preliminares": [],
        "sintese_decisao_1grau": "Improcedente",
        "pontos_criticos": ["Verificar prescrição"],
        "pre_juizo": "Tendência ao desprovimento.",
    }
    with patch("api_server.be.get_llm", return_value=_mock_llm_returning(payload)):
        res = client.post("/api/sustentacao/extract", json={
            "text": "Texto longo do processo...",
            "tipo_ato": "sustentacao",
            "modo": "preparacao",
        })
    assert res.status_code == 200
    body = res.json()
    assert "process_id" in body
    assert body["tipo_ato"] == "sustentacao"
    assert body["modo"] == "preparacao"
    # Store guarda user_id (tenant isolation)
    stored = api_server._sustentacao_store[body["process_id"]]
    assert stored["user_id"] == "test-user"


def test_extract_audiencia_realizacao_ok(client):
    payload = {
        "numero_processo": "987",
        "vara": "1ª Vara Cível",
        "autor": "João",
        "reu": "Empresa X",
        "pontos_controvertidos": ["Existência da dívida"],
        "depoentes": [{"id": "autor", "nome": "João", "tipo": "autor"}],
        "perguntas_planejadas": [{"depoente_id": "autor", "perguntas": ["P1"]}],
    }
    with patch("api_server.be.get_llm", return_value=_mock_llm_returning(payload)):
        res = client.post("/api/sustentacao/extract", json={
            "text": "...",
            "tipo_ato": "audiencia",
            "modo": "realizacao",
        })
    assert res.status_code == 200
    assert res.json()["tipo_ato"] == "audiencia"


def test_extract_strips_markdown_fences(client):
    """LLM frequentemente envolve JSON em ```json ... ``` — deve ser tolerado."""
    raw = "```json\n" + json.dumps({"numero_processo": "X", "teses": []}) + "\n```"
    mock = MagicMock()
    response = MagicMock()
    response.content = raw
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/extract", json={
            "text": "...", "tipo_ato": "sustentacao", "modo": "preparacao",
        })
    assert res.status_code == 200
    assert res.json()["data"]["numero_processo"] == "X"


def test_extract_empty_text_rejected(client):
    res = client.post("/api/sustentacao/extract", json={
        "text": "   ", "tipo_ato": "sustentacao", "modo": "preparacao",
    })
    assert res.status_code == 400


def test_extract_invalid_combo_falls_back(client):
    """tipo_ato/modo inválidos caem no default (sustentacao/preparacao)."""
    with patch("api_server.be.get_llm", return_value=_mock_llm_returning({"teses": []})):
        res = client.post("/api/sustentacao/extract", json={
            "text": "...", "tipo_ato": "invalido", "modo": "outro",
        })
    assert res.status_code == 200
    assert res.json()["tipo_ato"] == "sustentacao"
    assert res.json()["modo"] == "preparacao"


def test_extract_invalid_json_returns_502(client):
    """Se o LLM retorna JSON malformado, devolve 502."""
    mock = MagicMock()
    response = MagicMock()
    response.content = "isso não é JSON {[}"
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/extract", json={
            "text": "...", "tipo_ato": "sustentacao", "modo": "preparacao",
        })
    assert res.status_code == 502


# ── /api/sustentacao/chat ───────────────────────────────────────────────────

def test_chat_ok(client):
    _seed_processo("pid-1")
    mock = MagicMock()
    response = MagicMock()
    response.content = "Resposta sobre o processo."
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/chat", json={
            "process_id": "pid-1",
            "messages": [{"role": "user", "content": "Qual o pedido?"}],
        })
    assert res.status_code == 200
    assert res.json()["reply"] == "Resposta sobre o processo."


def test_chat_unknown_process(client):
    res = client.post("/api/sustentacao/chat", json={
        "process_id": "nao-existe",
        "messages": [{"role": "user", "content": "?"}],
    })
    assert res.status_code == 404


def test_chat_no_messages(client):
    _seed_processo("pid-2")
    res = client.post("/api/sustentacao/chat", json={"process_id": "pid-2", "messages": []})
    assert res.status_code == 400


# ── /api/sustentacao/analisar-voto ─────────────────────────────────────────

def test_analisar_voto_ok(client):
    _seed_processo("pid-3", data={"teses": ["Prescrição", "Dano moral"]})
    payload = {
        "resultado": "favoravel",
        "resumo": "Voto provê o recurso.",
        "por_tese": [
            {"tese": "Prescrição", "posicao": "favoravel", "justificativa": "Reconhecida"},
            {"tese": "Dano moral", "posicao": "favoravel", "justificativa": "Majorado"},
        ],
    }
    with patch("api_server.be.get_llm", return_value=_mock_llm_returning(payload)):
        res = client.post("/api/sustentacao/analisar-voto", json={
            "process_id": "pid-3", "documento_text": "Voto do relator...",
        })
    assert res.status_code == 200
    body = res.json()
    assert body["resultado"] == "favoravel"
    assert len(body["por_tese"]) == 2


def test_analisar_voto_aceita_fences_markdown(client):
    """LLM pode envolver a resposta em ```json ... ```."""
    _seed_processo("pid-3b", data={"teses": ["X"]})
    raw = "```json\n" + json.dumps({"resultado": "parcial", "resumo": "OK", "por_tese": []}) + "\n```"
    mock = MagicMock()
    response = MagicMock()
    response.content = raw
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/analisar-voto", json={
            "process_id": "pid-3b", "documento_text": "Voto...",
        })
    assert res.status_code == 200
    assert res.json()["resultado"] == "parcial"


def test_analisar_voto_sem_teses(client):
    """Se o processo não tem teses extraídas, retorna 400."""
    _seed_processo("pid-4", data={"teses": []})
    res = client.post("/api/sustentacao/analisar-voto", json={
        "process_id": "pid-4", "documento_text": "Voto...",
    })
    assert res.status_code == 400


def test_analisar_voto_processo_inexistente(client):
    res = client.post("/api/sustentacao/analisar-voto", json={
        "process_id": "nao-existe", "documento_text": "Voto...",
    })
    assert res.status_code == 404


def test_analisar_voto_documento_vazio(client):
    _seed_processo("pid-5", data={"teses": ["T"]})
    res = client.post("/api/sustentacao/analisar-voto", json={
        "process_id": "pid-5", "documento_text": "  ",
    })
    assert res.status_code == 400


# ── /api/sustentacao/analisar-sentenca ─────────────────────────────────────

def test_analisar_sentenca_ok(client):
    _seed_processo("pid-6", data={"pontos_controvertidos": ["Existência da dívida", "Valor"]})
    payload = {
        "resultado": "procedente",
        "resumo": "Procedente em parte.",
        "por_ponto": [
            {"ponto": "Existência da dívida", "decisao": "Reconhecida", "fundamento": "Doc fls 12", "alerta": None},
            {"ponto": "Valor", "decisao": "Reduzido", "fundamento": "Cálculo", "alerta": "Verificar fls 30"},
        ],
    }
    with patch("api_server.be.get_llm", return_value=_mock_llm_returning(payload)):
        res = client.post("/api/sustentacao/analisar-sentenca", json={
            "process_id": "pid-6", "documento_text": "Minuta de sentença...",
        })
    assert res.status_code == 200
    assert res.json()["por_ponto"][1]["alerta"] == "Verificar fls 30"


def test_analisar_sentenca_aceita_fences_markdown(client):
    _seed_processo("pid-6b", data={"pontos_controvertidos": ["X"]})
    raw = "```json\n" + json.dumps({"resultado": "improcedente", "resumo": "...", "por_ponto": []}) + "\n```"
    mock = MagicMock()
    response = MagicMock()
    response.content = raw
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/analisar-sentenca", json={
            "process_id": "pid-6b", "documento_text": "Sentença...",
        })
    assert res.status_code == 200
    assert res.json()["resultado"] == "improcedente"


def test_analisar_sentenca_sem_pontos(client):
    _seed_processo("pid-7", data={"pontos_controvertidos": []})
    res = client.post("/api/sustentacao/analisar-sentenca", json={
        "process_id": "pid-7", "documento_text": "...",
    })
    assert res.status_code == 400


# ── Tenant isolation ────────────────────────────────────────────────────────

def test_chat_tenant_isolation(client):
    """Usuário B não consegue acessar processo do usuário A — recebe 404 (não vaza existência)."""
    _seed_processo("pid-A", user_id="user-A", text="Processo do A")

    _set_user({"sub": "user-B"})
    res = client.post("/api/sustentacao/chat", json={
        "process_id": "pid-A",
        "messages": [{"role": "user", "content": "Resumo?"}],
    })
    assert res.status_code == 404


def test_analisar_voto_tenant_isolation(client):
    _seed_processo("pid-A2", user_id="user-A", data={"teses": ["T"]})

    _set_user({"sub": "user-B"})
    res = client.post("/api/sustentacao/analisar-voto", json={
        "process_id": "pid-A2", "documento_text": "Voto...",
    })
    assert res.status_code == 404


def test_analisar_sentenca_tenant_isolation(client):
    _seed_processo("pid-A3", user_id="user-A", data={"pontos_controvertidos": ["P"]})

    _set_user({"sub": "user-B"})
    res = client.post("/api/sustentacao/analisar-sentenca", json={
        "process_id": "pid-A3", "documento_text": "Sentença...",
    })
    assert res.status_code == 404


def test_owner_pode_acessar(client):
    """Caminho positivo: dono do processo acessa sem problema."""
    _seed_processo("pid-mine", user_id="test-user")
    mock = MagicMock()
    response = MagicMock()
    response.content = "ok"
    mock.invoke.return_value = response
    with patch("api_server.be.get_llm", return_value=mock):
        res = client.post("/api/sustentacao/chat", json={
            "process_id": "pid-mine",
            "messages": [{"role": "user", "content": "?"}],
        })
    assert res.status_code == 200

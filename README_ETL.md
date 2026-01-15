# 🚀 Sistema Híbrido de ETL Jurízico (MacBook Pro M3 + Railway)

Este guia explica como rodar o worker de processamento local (`legal_etl.py`) que utiliza a GPU do seu Mac para processar dados do Railway.

## Pré-requisitos

1.  **Python 3.10+** instalado.
2.  **Ollama** instalado e rodando.

## Instalação

1.  **Instale as dependências Python:**
    ```bash
    pip install -r requirements_etl.txt
    ```
    *Nota: Isso instalará o `torch` com suporte a MPS (Metal Performance Shaders) para usar a GPU do Mac.*

2.  **Baixe o modelo Llama 3 no Ollama:**
    ```bash
    ollama run llama3
    ```
    *Mantenha o Ollama rodando em segundo plano.*

## Configuração

Você precisa da URL de conexão do seu banco de dados PostgreSQL no Railway.
1.  Vá no Dashboard do Railway.
2.  Clique no serviço Postgres -> Aba "Variables".
3.  Copie o valor de `DATABASE_URL` (começa com `postgresql://...`).

## Executando o Worker

Abra o terminal na pasta do projeto e rode:

```bash
# Substitua a URL abaixo pela sua URL real do Railway
export DATABASE_URL="postgresql://postgres:password@roundhouse.proxy.rlwy.net:12345/railway"

# Inicia o script
python legal_etl.py
```

## O que ele faz?

1.  **Conecta** no Railway.
2.  **Busca** 50 processos pendentes (`processado=False`).
3.  **Vetoriza** os textos usando a GPU do Mac (super rápido).
4.  **Clusteriza** os casos similares.
5.  **Gera Minuta** usando o Llama 3 localmente.
6.  **Salva** tudo de volta no Railway.
7.  Dorme por 10 segundos e repete.

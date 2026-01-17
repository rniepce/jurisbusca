# 🤖 Assistente Rafa

Ferramenta de busca semântica para documentos jurídicos (sentenças, modelos), permitindo encontrar conceitos por similaridade (vetorização) em vez de palavras-chave exatas.

Permite o uso de **IA Local (HuggingFace)** ou **OpenAI**.

## 🚀 Como Rodar Localmente

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute a aplicação:**
    ```bash
    python -m streamlit run app.py
    ```
3.  Acesse `http://localhost:8501`.

---

## ☁️ Deploy no Railway

O projeto já está configurado para deploy no [Railway](https://railway.app).

### Passos para Deploy

1.  Faça um fork ou push deste repositório para o seu GitHub.
2.  No Railway, crie um "New Project" > "Deploy from GitHub repo".
3.  O deploy vai iniciar automaticamente.

### 💾 Persistência de Dados (Volumes)

Para que seus documentos não sumam quando o aplicativo reiniciar, configure um **Volume**:

1.  No painel do Railway, adicione um Volume (New > Volume).
2.  Conecte o Volume ao serviço do `jurisbusca`.
3.  Vá em **Settings** > **Service Domains / Volumes** e defina o **Mount Path** como:
    ```
    /chroma_data
    ```
4.  Vá em **Variables** e adicione a variável de ambiente:
    ```
    CHROMA_DB_PATH=/chroma_data
    ```

O sistema reiniciará e seus dados estarão seguros.

### 🔑 Configuração da OpenAI (Opcional)

Para usar modelos de embedding mais avançados:
- No painel lateral do app, insira sua **OpenAI API Key**.
- Caso deixe em branco, o sistema usará o modelo local (gratuito).

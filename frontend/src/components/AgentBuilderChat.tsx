import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FaXmark, FaPaperPlane, FaWandMagicSparkles } from 'react-icons/fa6';
import { sendMessage } from '../services/api';
import './AgentBuilderChat.css';

const ENGINEER_SYSTEM_PROMPT = `# PROMPT (VERSÃO 3.2): ENGENHEIRO DE PROMPT PARA CRIAÇÃO DE ASSISTENTES VIRTUAIS

## 1. FINALIDADE

Este prompt transforma o Assistente Virtual em um **Engenheiro de Prompt Experiente**. Sua finalidade é entrevistar o usuário para entender uma necessidade complexa e, a partir dessa interação, construir um prompt detalhado, estruturado e otimizado que permitirá a um modelo de linguagem executar a tarefa desejada com máxima eficiência, precisão e segurança.

---

## 2. PERSONA

Assuma a persona de um **Engenheiro de Prompt Sênior**, um especialista em traduzir necessidades de negócio e fluxos de trabalho humanos em instruções claras e eficazes para modelos de linguagem (LLMs).

* **Seu Estilo:** Analítico, metódico, colaborativo e pedagógico. Você não apenas executa, mas também guia o usuário, explicando a importância de cada etapa para que ele entenda o valor da estruturação. **Sua abordagem inclui:**
    * **Gestão de Ambiguidade:** Se uma resposta do usuário for vaga ou incompleta, sua função é pedir esclarecimentos com perguntas adicionais ou oferecer exemplos para ajudar a refinar o conceito antes de prosseguir.
* **Seu Objetivo:** Fazer as perguntas certas para extrair todos os requisitos implícitos e explícitos do usuário e, com eles, construir um prompt de alta qualidade que seja reutilizável, auditável e eficaz.

---

## 3. OBJETIVOS DO ENGENHEIRO DE PROMPT

* Identificar com clareza o **problema central** que o usuário quer resolver.
* Mapear as **tarefas e subtarefas** que o assistente final deverá executar.
* Definir quais **dados de entrada, contexto e métricas** são necessários.
* Estruturar um **fluxo de trabalho lógico (Chain-of-Thought)** para o assistente final.
* Elaborar e entregar um **prompt completo e pronto para uso** como produto final desta interação.

---

## 4. FLUXO DE TRABALHO (WORKFLOW INTERATIVO)

> **Pense passo a passo (Chain-of-Thought)**

### ETAPA 1: BOAS-VINDAS E DIAGNÓSTICO INICIAL

1.  **Mensagem Introdutória:** No primeiro contato com o usuário, apresente-se e explique o processo.
    > "Olá! Sou seu Engenheiro de Prompt. Minha função é ajudar você a construir um assistente virtual sob medida para suas necessidades. Juntos, vamos definir o objetivo, as tarefas e as regras para que ele funcione da melhor forma possível.
    >
    > **Um detalhe importante: nosso processo é flexível. A qualquer momento, você pode dizer 'ajustar' ou 'mudar a resposta anterior' para modificarmos algo que já definimos.**
    >
    > Para começarmos, **descreva de forma geral a tarefa principal que você deseja que o seu futuro assistente realize.** O que ele deve fazer? Qual problema ele deve resolver?"

2.  **Aguarde a resposta do usuário.** A partir da descrição inicial, passe para a próxima etapa.

---

### ETAPA 2: ENTREVISTA GUIADA PARA ESTRUTURAÇÃO

*Com base na resposta do usuário, conduza uma entrevista para detalhar cada componente do prompt final. Faça uma pergunta de cada vez para não sobrecarregar o usuário.*

1.  **PERSONA DO ASSISTENTE FINAL:**
    * **Pergunta:** "Entendido. Agora, vamos definir a **Persona** do seu assistente. Que 'papel' ou 'profissão' ele deve assumir para executar essa tarefa? (Exemplos: 'Analista Financeiro Cauteloso', 'Advogado especialista em Direito do Consumidor', 'Revisor de Textos Criativo', 'Assistente Executivo Proativo')."

2.  **OBJETIVOS DO ASSISTENTE FINAL:**
    * **Pergunta:** "Qual seria o principal **objetivo** desse assistente? O que é mais importante para você? (Exemplos: 'ganhar tempo', 'evitar erros', 'padronizar documentos', 'analisar dados para encontrar insights')."

3.  **CONTEXTO DE USO E PÚBLICO-ALVO:**
    * **Pergunta:** "Para quem este assistente se destina? Quem são os usuários finais? Qual o nível de conhecimento deles sobre o assunto?"

4.  **DADOS DE ENTRADA (INPUTS):**
    * **Pergunta:** "Quais **documentos, textos, dados ou informações** você (ou outro usuário) precisará fornecer ao assistente para que ele possa começar a trabalhar?"

5.  **TAREFAS E FLUXO DE TRABALHO (WORKFLOW):**
    * **Pergunta:** "Ótimo. Agora, vamos detalhar o **passo a passo**. Depois que você fornecer os dados, quais são as tarefas exatas que o assistente deve executar em ordem?"

6.  **RESULTADO FINAL (OUTPUT):**
    * **Pergunta:** "E qual é o **formato do resultado final** que você espera receber do assistente?"

7.  **MÉTRICAS DE SUCESSO:**
    * **Pergunta:** "Como você medirá o sucesso ou a qualidade do resultado? O que separaria uma resposta 'excelente' de uma apenas 'aceitável'?"

8.  **RESTRIÇÕES E REGRAS:**
    * **Pergunta:** "Por último, mas muito importante: existem **regras ou restrições**? O que o assistente **NÃO** deve fazer em nenhuma hipótese?"

---

### ETAPA 3: CONSOLIDAÇÃO E VALIDAÇÃO

1.  **Resumo da Coleta:** Após o usuário responder a todas as perguntas, consolide as informações em um resumo claro e peça confirmação.

---

### ETAPA 4: GERAÇÃO DO PROMPT E CONFIRMAÇÃO FINAL

1.  Com a confirmação do usuário, redija o prompt completo em um bloco de código markdown.
2.  Apresente o prompt final para aprovação.

---

### ETAPA 5: ENTREGA FINAL E ENCERRAMENTO

1.  Após a aprovação, entregue o prompt em formato Markdown.

IMPORTANTE: Sempre responda em português do Brasil.`;

const AgentBuilderChat = ({ onClose, onPromptReady }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const [finalPrompt, setFinalPrompt] = useState(null);
    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
        }
    }, [input]);

    // Send initial greeting on mount
    useEffect(() => {
        const initChat = async () => {
            setIsLoading(true);
            try {
                const result = await sendMessage({
                    message: 'Olá, quero criar um novo agente.',
                    model: 'v0',
                    llm: 'gpt-5.3-chat',
                    agentPrompt: ENGINEER_SYSTEM_PROMPT,
                    conversationId: null,
                    uploadedText: null,
                    styleDossier: null,
                    useRag: false,
                });
                setConversationId(result.conversation_id);
                let content = result.response;
                if (typeof content === 'object' && content !== null) {
                    content = content.text || content.content || JSON.stringify(content);
                }
                setMessages([{ role: 'assistant', content: String(content) }]);
            } catch (err) {
                setMessages([{
                    role: 'assistant',
                    content: `⚠️ Erro ao iniciar o Engenheiro de Prompt: ${err.message}`
                }]);
            } finally {
                setIsLoading(false);
            }
        };
        initChat();
    }, []);

    // Detect code blocks in assistant messages (potential final prompt)
    const extractCodeBlock = useCallback((text) => {
        // Match ```text, ```markdown, or ``` followed by content
        const regex = /```(?:text|markdown|md)?\n([\s\S]*?)```/;
        const match = text.match(regex);
        return match ? match[1].trim() : null;
    }, []);

    // Check latest assistant message for a code block
    useEffect(() => {
        if (messages.length === 0) return;
        const lastMsg = messages[messages.length - 1];
        if (lastMsg.role !== 'assistant') return;

        const codeBlock = extractCodeBlock(lastMsg.content);
        if (codeBlock && codeBlock.length > 100) {
            setFinalPrompt(codeBlock);
        }
    }, [messages, extractCodeBlock]);

    const handleSend = async () => {
        const text = input.trim();
        if (!text || isLoading) return;

        setMessages(prev => [...prev, { role: 'user', content: text }]);
        setInput('');
        setIsLoading(true);

        try {
            const result = await sendMessage({
                message: text,
                model: 'v0',
                llm: 'gpt-5.3-chat',
                agentPrompt: ENGINEER_SYSTEM_PROMPT,
                conversationId,
                uploadedText: null,
                styleDossier: null,
                useRag: false,
            });
            setConversationId(result.conversation_id);
            let content = result.response;
            if (typeof content === 'object' && content !== null) {
                content = content.text || content.content || JSON.stringify(content);
            }
            setMessages(prev => [...prev, { role: 'assistant', content: String(content) }]);
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `⚠️ Erro: ${err.message}`
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="agent-builder-container">
            {/* Header */}
            <div className="agent-builder-header">
                <div className="agent-builder-header-left">
                    <div className="agent-builder-header-icon">
                        <FaWandMagicSparkles />
                    </div>
                    <div>
                        <h3>Engenheiro de Prompt</h3>
                        <p>Construa seu agente personalizado</p>
                    </div>
                </div>
                <button className="agent-builder-close-btn" onClick={onClose} aria-label="Fechar">
                    <FaXmark />
                </button>
            </div>

            {/* Messages */}
            <div className="agent-builder-messages">
                {messages.map((msg, i) => (
                    <div key={i} className={`builder-msg ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}

                {isLoading && (
                    <div className="builder-typing">
                        <div className="builder-typing-dot" />
                        <div className="builder-typing-dot" />
                        <div className="builder-typing-dot" />
                    </div>
                )}

                {/* Create Agent button — appears when a final prompt is detected */}
                {finalPrompt && !isLoading && (
                    <button
                        className="builder-create-btn"
                        onClick={() => onPromptReady(finalPrompt)}
                        id="btn-create-from-builder"
                    >
                        <FaWandMagicSparkles /> Criar Agente
                    </button>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="agent-builder-input-area">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Descreva o que você precisa..."
                    rows={1}
                    disabled={isLoading}
                    id="agent-builder-input"
                />
                <button
                    className="agent-builder-send-btn"
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    aria-label="Enviar"
                    id="agent-builder-send"
                >
                    <FaPaperPlane />
                </button>
            </div>
        </div>
    );
};

export default AgentBuilderChat;

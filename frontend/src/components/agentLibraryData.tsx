import React from 'react';
import {
    FaScaleBalanced, FaMagnifyingGlass, FaFileLines, FaGavel,
    FaBrain, FaWandMagicSparkles,
} from 'react-icons/fa6';
import type { FlowConfig } from '../services/api';

// ─────────────────────────────────────────────────────────────────
// AGENT TEMPLATES (com prompts completos)
// ─────────────────────────────────────────────────────────────────

export interface AgentTemplate {
    id: string;
    name: string;
    description: string;
    icon: React.ReactNode;
    color: string;
    tags: string[];
    prompt: string;
}

export const AGENT_TEMPLATES: AgentTemplate[] = [
    {
        id: 'redator',
        name: 'Redator Jurídico',
        description: 'Especialista em redigir decisões, acórdãos e peças processuais com linguagem jurídica precisa e fundamentação adequada.',
        icon: <FaScaleBalanced />,
        color: '#3b82f6',
        tags: ['Redação', 'Decisões'],
        prompt: `Você é um redator jurídico sênior do TJMG. Sua função é redigir minutas de decisões, acórdãos e despachos com:

1. **Linguagem jurídica precisa**: use terminologia técnica correta, evitando coloquialismos
2. **Estrutura formal**: Relatório → Fundamentação → Dispositivo
3. **Fundamentação completa**: cite dispositivos legais (CF, CC, CPC, leis especiais), súmulas, e quando pertinente, precedentes vinculantes
4. **Clareza e objetividade**: parágrafos curtos, conexão lógica entre argumentos
5. **Adequação ao tipo**: sentenças exigem fundamentação completa; despachos de mero expediente são concisos

Ao receber um processo:
- Identifique o tipo (cível, criminal, tributário, trabalhista, etc.)
- Localize: partes, pedido principal, fundamentação invocada, provas relevantes
- Construa uma minuta seguindo o padrão TJMG, com cabeçalho, ementa (se acórdão), relatório, fundamentação, dispositivo e fecho

Se faltarem informações críticas, sinalize com [⚠️ DADO AUSENTE: descrição] no texto.`,
    },
    {
        id: 'jurisprudencia',
        name: 'Pesquisador de Jurisprudência',
        description: 'Busca e analisa precedentes relevantes, identificando teses aplicáveis ao caso concreto em bases jurídicas.',
        icon: <FaMagnifyingGlass />,
        color: '#8B5CF6',
        tags: ['Pesquisa', 'Jurisprudência'],
        prompt: `Você é um pesquisador especializado em jurisprudência do TJMG, STJ, STF e tribunais superiores.

Ao receber um caso ou tese jurídica:

1. **Identifique a questão central**: qual é o ponto controvertido principal?
2. **Mapeie palavras-chave**: termos técnicos, dispositivos legais, súmulas potencialmente aplicáveis
3. **Localize entendimentos relevantes**:
   - Precedentes vinculantes (CF art. 927 do CPC)
   - Súmulas STF/STJ
   - Acordãos recentes do TJMG na mesma matéria
   - Divergências entre câmaras ou turmas
4. **Construa um parecer estruturado**:
   - Tese dominante
   - Tese minoritária (se houver)
   - Posicionamento do TJMG
   - Recomendação final

Sempre cite: número do processo, data do julgamento, relator, órgão julgador. Quando houver divergência, explicite-a e indique qual posição predomina quantitativamente.`,
    },
    {
        id: 'revisor',
        name: 'Revisor de Documentos',
        description: 'Revisa minutas para identificar inconsistências, erros formais e lacunas de fundamentação antes da publicação.',
        icon: <FaFileLines />,
        color: '#10b981',
        tags: ['Revisão', 'Qualidade'],
        prompt: `Você é um revisor sênior de minutas judiciais com 20 anos de experiência. Sua função é auditar minutas ANTES da publicação.

Para cada minuta recebida, produza um **DASHBOARD DE CONFORMIDADE** com 5 seções:

## 1. ✅ INTEGRIDADE FACTUAL
- Os fatos descritos batem com o que está nos autos?
- Datas, valores e nomes estão corretos?
- Há contradição entre o relatório e o dispositivo?

## 2. ⚖️ FUNDAMENTAÇÃO LEGAL
- Dispositivos citados existem e estão em vigor?
- A interpretação dada é compatível com a jurisprudência?
- Foram aplicadas súmulas vinculantes pertinentes?

## 3. 📐 ESTRUTURA FORMAL
- Cabeçalho, relatório, fundamentação e dispositivo presentes?
- Numeração de parágrafos consistente?
- Linguagem formal mantida?

## 4. 🎯 ADERÊNCIA AO PEDIDO
- Todos os pedidos foram apreciados?
- Há decisão "extra petita" ou "ultra petita"?
- O dispositivo responde claramente ao pedido?

## 5. 🚨 ALERTAS CRÍTICOS
- Riscos de nulidade
- Súmulas/precedentes vinculantes ignorados
- Inconsistências graves

Use ✅ aprovado, ⚠️ atenção, ❌ erro grave em cada item.`,
    },
    {
        id: 'audiencias',
        name: 'Assistente de Audiências',
        description: 'Apoio em tempo real durante sessões e audiências — resume alegações, sinaliza pontos controvertidos e sugere perguntas.',
        icon: <FaGavel />,
        color: '#f59e0b',
        tags: ['Audiências', 'Tempo real'],
        prompt: `Você é um assistente jurídico de apoio em tempo real durante audiências e sessões de julgamento.

Durante a audiência, você deve:

1. **Resumir alegações**: para cada manifestação oral, gere um resumo em 2-3 bullets do ponto central, fundamentação invocada e pedido implícito
2. **Sinalizar pontos controvertidos**: identifique divergências entre as partes e marque com 🔴 questões factuais vs ⚖️ questões jurídicas
3. **Sugerir perguntas**: liste perguntas que o magistrado pode fazer à parte, testemunha ou perito para esclarecer pontos obscuros
4. **Identificar fundamentos legais aplicáveis**: durante a manifestação, indique dispositivos, súmulas ou precedentes que estão sendo invocados ou que poderiam fortalecer/contestar a tese
5. **Alertar sobre prazos e nulidades**: se algo na audiência puder gerar nulidade processual (cerceamento de defesa, suspeição, etc.) sinalize imediatamente

Formato de saída: blocos curtos, marcados com emojis, prontos para leitura rápida durante a sessão. Evite parágrafos longos.`,
    },
    {
        id: 'sintetizador',
        name: 'Sintetizador de Processos',
        description: 'Resume processos extensos em pontos-chave: partes, pedidos, provas e histórico processual de forma estruturada.',
        icon: <FaBrain />,
        color: '#ec4899',
        tags: ['Resumo', 'Análise'],
        prompt: `Você é um sintetizador de processos judiciais. Sua função é transformar autos extensos em resumos estruturados e navegáveis.

Para qualquer processo recebido, produza:

## IDENTIFICAÇÃO
- **Número CNJ**:
- **Vara/Comarca**:
- **Data de distribuição**:
- **Valor da causa**:

## PARTES
- **Autor(es)**:
- **Réu(s)**:
- **Terceiros interessados**:

## OBJETO
- **Pedido principal** (1 frase):
- **Pedidos subsidiários**:
- **Causa de pedir** (resumo em 3-5 linhas):

## HISTÓRICO PROCESSUAL
Linha do tempo das peças relevantes (data → tipo → resumo da decisão/manifestação):
- DD/MM/AAAA — Petição inicial
- DD/MM/AAAA — Contestação
- DD/MM/AAAA — Decisão saneadora
- (e assim por diante)

## PROVAS PRODUZIDAS
- Documentais: quantidade e principais
- Testemunhais: arroladas/ouvidas
- Periciais: realizadas/pendentes

## SITUAÇÃO ATUAL
- Fase processual
- Próximo ato esperado

Seja conciso. Use bullets. Não invente — se algo não estiver claro no processo, escreva [NÃO IDENTIFICADO].`,
    },
    {
        id: 'instrucao',
        name: 'Instrutor de Processo',
        description: 'Analisa a fase instrutória e identifica quais provas faltam, quais já foram produzidas e o que pode ser deferido ou indeferido.',
        icon: <FaWandMagicSparkles />,
        color: '#06b6d4',
        tags: ['Instrução', 'Provas'],
        prompt: `Você é especialista em direito processual, com foco em fase de instrução probatória.

Ao receber um processo na fase instrutória, analise e produza:

## 1. PONTOS CONTROVERTIDOS
Liste os fatos relevantes ainda controvertidos (não acordados pelas partes ou não demonstrados documentalmente).

## 2. PROVAS REQUERIDAS PELAS PARTES
Para cada prova requerida (testemunhal, pericial, depoimento pessoal):
- **Quem requereu**:
- **Sobre qual ponto**:
- **Análise de pertinência**: relevante / impertinente / protelatória
- **Recomendação**: ✅ deferir / ❌ indeferir / 🔄 deferir parcialmente
- **Justificativa**: 2-3 linhas com fundamento jurídico (CPC art. 369-380)

## 3. PROVAS QUE DEVERIAM SER PRODUZIDAS DE OFÍCIO
Identifique fatos relevantes que carecem de prova e sugira como o juízo pode supri-las (perícia técnica, prova emprestada, busca por documentos públicos).

## 4. DECISÃO SANEADORA PROPOSTA
Minute uma decisão saneadora contendo:
- Pontos controvertidos fixados
- Provas deferidas e indeferidas (com fundamento)
- Designação de audiência (se necessário)
- Distribuição dinâmica do ônus probatório (se cabível)

Cite sempre: CPC art. 357 e correlatos. Evite jargão excessivo — seja didático.`,
    },
];

// ─────────────────────────────────────────────────────────────────
// FLOW TEMPLATES (com configs reais executáveis)
// ─────────────────────────────────────────────────────────────────

export interface FlowTemplate {
    id: string;
    name: string;
    description: string;
    steps: string[];
    color: string;
    tags: string[];
    config: FlowConfig;
}

// Helpers
const _start = (id = 'start', x = 60, y = 240) => ({ id, type: 'start', position: { x, y }, data: {} });
const _end = (id = 'end', x = 1200, y = 240) => ({ id, type: 'end', position: { x, y }, data: {} });
const _agent = (id: string, label: string, model: string, prompt: string, x: number, y: number) => ({
    id, type: 'agent', position: { x, y },
    data: { label, model, prompt, knowledge: '', knowledge_files: '', output_var: id },
});
const _juris = (id: string, label: string, query: string, x: number, y: number) => ({
    id, type: 'juris', position: { x, y }, data: { label, query, top_k: '5', output_var: id },
});
const _switch = (id: string, label: string, categories: string, x: number, y: number) => ({
    id, type: 'switch', position: { x, y },
    data: { label, categories, model: 'gpt-5.4-mini', output_var: id },
});
const _extractor = (id: string, label: string, fields: string, x: number, y: number) => ({
    id, type: 'extractor', position: { x, y },
    data: { label, fields, model: 'gpt-5.4-mini', output_var: id },
});
const _estilo = (id: string, label: string, x: number, y: number) => ({
    id, type: 'estilo', position: { x, y }, data: { label, output_var: id },
});
const _edge = (id: string, source: string, target: string, label = '', sourceHandle = '') => ({
    id, source, target, label, sourceHandle,
});

export const FLOW_TEMPLATES: FlowTemplate[] = [
    {
        id: 'analise-completa',
        name: 'Análise Completa de Processo',
        description: 'Pipeline end-to-end que parte da leitura do processo até a minuta final de decisão, passando por extração de fatos, pesquisa jurídica e revisão.',
        steps: ['Triagem', 'Extração', 'Jurisprudência', 'Redação', 'Revisão'],
        color: '#3b82f6',
        tags: ['Completo', 'Decisão'],
        config: {
            nodes: [
                _start(),
                _agent('triagem', 'Triagem', 'gpt-5.3-chat',
                    'Identifique tipo de processo, partes, pedido principal e pontos controvertidos. Devolva em formato estruturado.',
                    240, 240),
                _extractor('extracao', 'Extrair Fatos',
                    'numero_processo:string:Número CNJ|partes:array:Lista de partes|pedido:string:Pedido principal|valor_causa:number:Valor em reais',
                    480, 240),
                _juris('juris', 'Jurisprudência', '{{Extrair Fatos}}', 720, 240),
                _agent('redacao', 'Redigir Minuta', 'claude-sonnet-4-6',
                    'Você é um magistrado. Com base na triagem ({{Triagem}}), nos fatos extraídos ({{Extrair Fatos}}) e na jurisprudência ({{Jurisprudência}}), redija minuta de decisão completa com relatório, fundamentação e dispositivo.',
                    960, 200),
                _agent('revisao', 'Revisão QA', 'gpt-5.3-chat',
                    'Audite a minuta abaixo: {{Redigir Minuta}}\n\nVerifique consistência factual, fundamentação legal e estrutura formal. Produza dashboard com ✅/⚠️/❌ por critério.',
                    1200, 280),
                _end('end', 1480),
            ],
            edges: [
                _edge('e1', 'start', 'triagem'),
                _edge('e2', 'triagem', 'extracao'),
                _edge('e3', 'extracao', 'juris'),
                _edge('e4', 'juris', 'redacao'),
                _edge('e5', 'redacao', 'revisao'),
                _edge('e6', 'revisao', 'end'),
            ],
        },
    },
    {
        id: 'triagem',
        name: 'Triagem Automática',
        description: 'Classifica automaticamente processos por matéria e tipo de provimento mais adequado.',
        steps: ['Leitura', 'Classificação', 'Priorização'],
        color: '#f59e0b',
        tags: ['Triagem', 'Automação'],
        config: {
            nodes: [
                _start(),
                _agent('leitura', 'Leitura do Processo', 'gpt-5.4-mini',
                    'Leia o processo e produza um resumo de 5-10 linhas identificando: partes, pedido principal, valor da causa e fase processual.',
                    260, 240),
                _switch('classificacao', 'Classificar Matéria', 'Civil|Penal|Tributário|Trabalhista|Família|Consumidor', 540, 240),
                _agent('priorizacao', 'Definir Prioridade', 'gpt-5.4-mini',
                    'Com base no resumo ({{Leitura do Processo}}) e na matéria ({{Classificar Matéria}}), classifique a urgência em: URGENTE / ALTA / NORMAL / BAIXA, justificando em 2 linhas.',
                    840, 240),
                _end('end', 1140),
            ],
            edges: [
                _edge('e1', 'start', 'leitura'),
                _edge('e2', 'leitura', 'classificacao'),
                _edge('e3a', 'classificacao', 'priorizacao', '', 'Civil'),
                _edge('e3b', 'classificacao', 'priorizacao', '', 'Penal'),
                _edge('e3c', 'classificacao', 'priorizacao', '', 'Tributário'),
                _edge('e3d', 'classificacao', 'priorizacao', '', 'Trabalhista'),
                _edge('e3e', 'classificacao', 'priorizacao', '', 'Família'),
                _edge('e3f', 'classificacao', 'priorizacao', '', 'Consumidor'),
                _edge('e4', 'priorizacao', 'end'),
            ],
        },
    },
    {
        id: 'revisao-camadas',
        name: 'Revisão em Múltiplas Camadas',
        description: 'Passa a minuta por três etapas de revisão paralelas: jurídica, formal e de coerência interna.',
        steps: ['Minuta', 'Jurídica', 'Formal', 'Coerência', 'Consolidação'],
        color: '#8B5CF6',
        tags: ['Revisão', 'Qualidade', 'Paralelo'],
        config: {
            nodes: [
                _start(),
                _agent('juridica', 'Revisão Jurídica', 'claude-sonnet-4-6',
                    'Audite a minuta abaixo focando em FUNDAMENTAÇÃO LEGAL: dispositivos citados estão corretos? Súmulas vinculantes foram observadas? Há jurisprudência consolidada ignorada? Aponte com ⚖️.',
                    340, 100),
                _agent('formal', 'Revisão Formal', 'gpt-5.4-mini',
                    'Audite a minuta abaixo focando em FORMA: ortografia, concordância, pontuação, estrutura (relatório/fundamentação/dispositivo), numeração de parágrafos. Aponte com 📐.',
                    340, 240),
                _agent('coerencia', 'Coerência Interna', 'gpt-5.3-chat',
                    'Audite a minuta abaixo focando em COERÊNCIA: o dispositivo responde ao pedido? Há contradição entre relatório e fundamentação? Há decisão extra/ultra petita? Aponte com 🎯.',
                    340, 380),
                _agent('consolidacao', 'Consolidação', 'claude-sonnet-4-6',
                    'Consolide os três pareceres em um relatório único:\n\n[Jurídica]: {{Revisão Jurídica}}\n[Formal]: {{Revisão Formal}}\n[Coerência]: {{Coerência Interna}}\n\nProduza um dashboard final com 3 colunas, pontos críticos em destaque, e recomendação final (APROVADO / APROVADO COM RESSALVAS / RETORNAR PARA AJUSTES).',
                    720, 240),
                _end('end', 1040),
            ],
            edges: [
                _edge('e1a', 'start', 'juridica'),
                _edge('e1b', 'start', 'formal'),
                _edge('e1c', 'start', 'coerencia'),
                _edge('e2a', 'juridica', 'consolidacao'),
                _edge('e2b', 'formal', 'consolidacao'),
                _edge('e2c', 'coerencia', 'consolidacao'),
                _edge('e3', 'consolidacao', 'end'),
            ],
        },
    },
    {
        id: 'pesquisa-sintese',
        name: 'Pesquisa e Síntese',
        description: 'Busca jurisprudência relevante e consolida em um memorando sintético com tese dominante e divergências.',
        steps: ['Tema', 'Jurisprudência', 'Síntese'],
        color: '#10b981',
        tags: ['Pesquisa', 'Jurisprudência'],
        config: {
            nodes: [
                _start(),
                _agent('tema', 'Identificar Tema', 'gpt-5.4-mini',
                    'Identifique o tema jurídico central da entrada e expresse em uma frase de busca otimizada para pesquisa em jurisprudência.',
                    260, 240),
                _juris('jurisprudencia', 'Buscar Jurisprudência', '{{Identificar Tema}}', 540, 240),
                _agent('sintese', 'Síntese Comparativa', 'claude-sonnet-4-6',
                    'Com base no tema ({{Identificar Tema}}) e nos acordãos pesquisados ({{Buscar Jurisprudência}}), produza memorando contendo:\n\n1. TESE DOMINANTE (com 3 julgados representativos)\n2. TESE MINORITÁRIA (se houver)\n3. EVOLUÇÃO RECENTE\n4. RECOMENDAÇÃO de aplicação ao caso',
                    840, 240),
                _end('end', 1140),
            ],
            edges: [
                _edge('e1', 'start', 'tema'),
                _edge('e2', 'tema', 'jurisprudencia'),
                _edge('e3', 'jurisprudencia', 'sintese'),
                _edge('e4', 'sintese', 'end'),
            ],
        },
    },
    {
        id: 'voto',
        name: 'Elaboração de Voto',
        description: 'Fluxo completo para elaboração de votos em câmaras e turmas, do relatório ao dispositivo, com aplicação do estilo do magistrado.',
        steps: ['Relatório', 'Fundamentação', 'Dispositivo', 'Estilo', 'Revisão'],
        color: '#ec4899',
        tags: ['Voto', 'Câmara'],
        config: {
            nodes: [
                _start(),
                _agent('relatorio', 'Relatório', 'gpt-5.3-chat',
                    'Produza o relatório do voto: histórico processual (1ª instância → recurso), pretensões das partes e ponto controvertido em julgamento. Use linguagem formal de acórdão.',
                    240, 240),
                _agent('fundamentacao', 'Fundamentação', 'claude-sonnet-4-6',
                    'Com base no relatório ({{Relatório}}), construa a fundamentação do voto: análise das teses, dispositivos legais aplicáveis, posicionamento do tribunal, citação de precedentes. Use parágrafos numerados.',
                    520, 240),
                _agent('dispositivo', 'Dispositivo', 'gpt-5.4-mini',
                    'Com base no relatório e fundamentação ({{Fundamentação}}), redija o dispositivo do voto: PROVIMENTO / NEGADO PROVIMENTO / CONHECIMENTO PARCIAL, com motivação sintética e indicação do que se modifica/mantém na decisão recorrida.',
                    800, 240),
                _estilo('estilo', 'Estilo do Magistrado', 1060, 240),
                _agent('revisao', 'Revisão Final', 'gpt-5.3-chat',
                    'Audite o voto final: {{Estilo do Magistrado}}\n\nVerifique consistência entre relatório/fundamentação/dispositivo, citações legais, e formatação. Dê ✅ se pronto ou liste correções necessárias.',
                    1320, 240),
                _end('end', 1600),
            ],
            edges: [
                _edge('e1', 'start', 'relatorio'),
                _edge('e2', 'relatorio', 'fundamentacao'),
                _edge('e3', 'fundamentacao', 'dispositivo'),
                _edge('e4', 'dispositivo', 'estilo'),
                _edge('e5', 'estilo', 'revisao'),
                _edge('e6', 'revisao', 'end'),
            ],
        },
    },
];

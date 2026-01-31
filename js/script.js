// Configuração e Estado
const STATE = {
    currentScreen: 'screen-visao',
    activeProcessId: '5001234'
};

// MOCK DATA PARA DEMONSTRAÇÃO
// MOCK DATA PARA DEMONSTRAÇÃO
const PROCESS_DATA = {
    '5001234': {
        title: 'Proc. 5001234-88.2025.8.13.0024',
        partes: 'Autor: Ana Pereira • Réu: Telefonia S.A.',
        docContent: `
            <!-- PROCESS SUMMARY DASHBOARD -->
            <div class="process-summary-card" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin-bottom: 40px;">
                <h2 style="color: var(--tjmg-brand); font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
                    <i class="fa-solid fa-file-waveform"></i> Resumo do Processo
                </h2>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div>
                        <h5 style="color: var(--text-secondary); text-transform: uppercase; font-size: 0.75rem; font-weight: 700;">Resumo das Peças</h5>
                        <ul style="list-style: none; margin-top: 8px; font-size: 0.9rem; color: var(--text-primary);">
                            <li style="margin-bottom: 6px;"><strong>Inicial:</strong> Alega interrupção indevida (5 dias) e pede danos morais (10k).</li>
                            <li style="margin-bottom: 6px;"><strong>Contestação:</strong> Nega falha, atribui a manutenção programada. Sem provas.</li>
                            <li><strong>Impugnação:</strong> Reitera revelia sobre documentos.</li>
                        </ul>
                    </div>
                    <div>
                        <h5 style="color: var(--text-secondary); text-transform: uppercase; font-size: 0.75rem; font-weight: 700;">Atos Processuais Recentes</h5>
                        <ul style="list-style: none; margin-top: 8px; font-size: 0.9rem; color: var(--text-primary);">
                            <li style="margin-bottom: 6px;"><span class="badge bg-neutral">10/01</span> Petição Inicial</li>
                            <li style="margin-bottom: 6px;"><span class="badge bg-neutral">15/02</span> Citação (AR Positivo)</li>
                            <li><span class="badge bg-neutral">20/02</span> Contestação Juntada</li>
                        </ul>
                    </div>
                </div>

                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 12px; display: flex; align-items: start; gap: 12px;">
                    <i class="fa-solid fa-lightbulb" style="color: #059669; margin-top: 2px;"></i>
                    <div>
                        <strong style="color: #047857; font-size: 0.95rem;">Sugestão de Próximo Ato</strong>
                        <p style="font-size: 0.9rem; color: #065f46; margin-top: 4px;">Julgamento Antecipado da Lide (Procedência Parcial). A matéria é de direito e fatos incontroversos.</p>
                    </div>
                </div>
            </div>

            <!-- DRAFTING AREA -->
            <p align="center" style="margin-bottom: 40px; font-size: 1.1em;"><strong>SENTENÇA</strong></p>
            <p style="margin-bottom: 1em;">Trata-se de ação proposta por <span class="var-field">ANA PEREIRA</span> em face de <span class="var-field">TELEFONIA S.A.</span>.</p>
            <p style="margin-bottom: 1em;">A parte autora alega interrupção indevida da linha <span class="var-field">(31) 99988-7766</span> em <span class="var-field">15/02/2025</span>.</p>
            <br>
            <p style="margin-bottom: 1em;"><strong>FUNDAMENTAÇÃO</strong></p>
            <p style="margin-bottom: 1em;">Aplica-se o CDC. A ré não comprovou a regularidade. Dano moral configurado <i>in re ipsa</i>.</p>
            <p style="margin-bottom: 1em;">Fixo a indenização em <span class="var-field">R$ 5.000,00</span> com juros de mora a partir da citação.</p>`,
        chatContext: `Identifiquei que a Contestação não apresenta B.O. Deseja adicionar um parágrafo sobre falha probatória?`
    },
    '5001001': {
        title: 'Proc. 5001001-22.2025.8.13.0024',
        partes: 'Autor: João Silva • Réu: Banco X',
        docContent: `
            <div class="process-summary-card" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin-bottom: 40px;">
                <h2 style="color: var(--tjmg-brand); font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;"><i class="fa-solid fa-file-waveform"></i> Resumo do Processo</h2>
                <div style="margin-bottom: 20px;">
                    <p><strong>Resumo:</strong> Negativação indevida. Autor alega conta encerrada.</p>
                    <p><strong>Atos:</strong> Inicial (05/01) -> Liminar Indeferida (07/01) -> Contestação Genérica.</p>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px; padding: 12px;">
                    <strong style="color: #b45309;">Sugestão:</strong> Audiência de Instrução Necessária (Dúvida sobre assinatura).
                </div>
            </div>
            
            <p align="center" style="margin-bottom: 40px; font-size: 1.1em;"><strong>SENTENÇA</strong></p>
            <p style="margin-bottom: 1em;">Dispensado o relatório (Art. 38, L 9099).</p>
            <p style="margin-bottom: 1em;">Trata-se de ação de inexigibilidade de débito c/c danos morais.</p>`,
        chatContext: `Este processo trata de negativação indevida. A Súmula 385 não se aplica pois não há pré-existências.`
    },
    '5001002': {
        title: 'Proc. 5001002-33.2025.8.13.0024',
        partes: 'Autor: Maria Souza • Réu: Construtora Y',
        docContent: `
             <div class="process-summary-card" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin-bottom: 40px;">
                <h2 style="color: var(--tjmg-brand); font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;"><i class="fa-solid fa-file-waveform"></i> Resumo do Processo</h2>
                <p><strong>Objeto:</strong> Vício construtivo (rachaduras).</p>
                <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px; margin-top: 10px;">
                    <strong style="color: #1e40af;">Sugestão:</strong> Saneador (Deferir Perícia).
                </div>
            </div>
            <p align="center" style="margin-bottom: 40px; font-size: 1.1em;"><strong>DESPACHO SANEADOR</strong></p>
            <p style="margin-bottom: 1em;">I - Defiro a prova pericial requerida.</p>`,
        chatContext: `Sugestão: Fixar prazo de 15 dias para quesitos e assistentes técnicos.`
    },
    '5001235': {
        title: 'Proc. 5001235-99.2025.8.13.0024',
        partes: 'Autor: Condomínio Z • Réu: Morador W',
        docContent: `
             <div class="process-summary-card" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin-bottom: 40px;">
                <h2 style="color: var(--tjmg-brand); font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;"><i class="fa-solid fa-file-waveform"></i> Resumo do Processo</h2>
                <p><strong>Objeto:</strong> Cobrança Condomínio (3 meses atraso).</p>
                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 12px; margin-top: 10px;">
                    <strong style="color: #047857;">Sugestão:</strong> Homologar Acordo (Minuta Juntada em ID 999).
                </div>
            </div>
            <p align="center" style="margin-bottom: 40px; font-size: 1.1em;"><strong>SENTENÇA</strong></p>
            <p style="margin-bottom: 1em;">Homologo por sentença o acordo...</p>`,
        chatContext: `O réu confessou a dívida em audiência. Julgamento antecipado possível.`
    }
};

// Funções de Interatividade Minutas
function loadProcess(id) {
    STATE.activeProcessId = id;
    const data = PROCESS_DATA[id];
    if (!data) return;

    // 1. Atualizar Lista Visual (Active State)
    document.querySelectorAll('.process-item').forEach(item => {
        item.classList.remove('active');
        item.querySelector('strong') ? item.querySelector('strong').style.color = 'var(--text-primary)' : null;
        if (item.getAttribute('onclick') && item.getAttribute('onclick').includes(id)) {
            item.classList.add('active');
            // Resetar visual do item ativo
            // Simulação simples: Na prática recriaria o DOM do item para adicionar o ícone de edição
        }
    });

    // 2. Atualizar Editor
    const editor = document.querySelector('.document-paper');
    if (editor) {
        editor.innerHTML = data.docContent;
        // Re-attach ghost text logic se necessário (simplificado aqui)
    }

    // 3. Atualizar Header Minutas (Simulado)
    const headerTitle = document.querySelector('#screen-minutas h3');
    // if(headerTitle) headerTitle.innerText = `Editando ${data.title}`;

    // 4. Atualizar Chat Context
    const chatMsg = document.querySelector('.chat-messages .msg.ai');
    if (chatMsg) {
        chatMsg.innerHTML = `<strong>Assistente:</strong> ${data.chatContext}`;
    }
}

// Menus Dropdown
function toggleMenu(menuId) {
    const menu = document.getElementById(menuId);
    if (!menu) return;

    // Fecha outros
    document.querySelectorAll('.dropdown-menu').forEach(m => {
        if (m.id !== menuId) m.classList.remove('show');
    });

    menu.classList.toggle('show');
}

// Fecha menus ao clicar fora
window.onclick = function (event) {
    if (!event.target.matches('.tool-btn') && !event.target.closest('.tool-btn')) {
        document.querySelectorAll('.dropdown-menu').forEach(m => m.classList.remove('show'));
    }
}

function selectAgent(name) {
    addMessage(`Agente <strong>${name}</strong> ativado.`, 'ai');
}

function selectLLM(name) {
    addMessage(`Modelo alterado para <strong>${name}</strong>.`, 'ai');
}

// Navegação

// Navegação
function goToScreen(screenId, navElement) {
    if (!screenId) return;

    // Atualiza estado
    STATE.currentScreen = screenId;

    // 1. Atualizar links da sidebar
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));

    // Identifica qual link ativar
    let targetNav;
    if (navElement) {
        targetNav = navElement;
    } else {
        // Mapeamento reverso simples
        if (screenId === 'screen-minutas') targetNav = document.getElementById('nav-minutas');
        else if (screenId === 'screen-acervo') targetNav = document.getElementById('nav-acervo');
        else if (screenId === 'screen-visao') document.querySelector('.nav-link[onclick*="screen-visao"]').classList.add('active'); // fallback
    }

    if (targetNav) targetNav.classList.add('active');

    // 2. Trocar a tela com animação suave
    document.querySelectorAll('.screen-section').forEach(screen => {
        screen.classList.remove('active');
        screen.style.display = 'none'; // Garante reset
    });

    const targetScreen = document.getElementById(screenId);
    if (targetScreen) {
        targetScreen.style.display = 'flex';
        // Pequeno delay para permitir o display flex aplicar antes da classe active (opcional para animações CSS mais complexas)
        requestAnimationFrame(() => {
            targetScreen.classList.add('active');
        });
    }
}

// Sub-abas (Tela de Apoio)
function openSubTab(tabId, btn) {
    // Esconde todos os containers
    document.querySelectorAll('#screen-apoio .view-container').forEach(el => {
        el.classList.remove('active');
    });

    // Reseta botões
    document.querySelectorAll('.module-tabs .tab-btn').forEach(b => b.classList.remove('active'));

    // Ativa alvo
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');

    if (btn) btn.classList.add('active');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- FUNCIONALIDADES ACERVO DIGITAL ---

// 1. Alternar Visualização (Lista vs Kanban)
function switchAcervoView(mode) {
    const listView = document.getElementById('acervo-list-view');
    const kanbanView = document.getElementById('acervo-kanban-view');
    const btnList = document.getElementById('btn-view-list');
    const btnKanban = document.getElementById('btn-view-kanban');

    if (mode === 'list') {
        listView.style.display = 'table';
        kanbanView.style.display = 'none';
        btnList.classList.add('active');
        btnKanban.classList.remove('active');
    } else {
        listView.style.display = 'none';
        kanbanView.style.display = 'flex';
        btnList.classList.remove('active');
        btnKanban.classList.add('active');
    }
}

// 2. Batch Toolbar (Ações em Massa)
function toggleBatchBar(force) {
    const bar = document.getElementById('batch-action-bar');
    const checkboxes = document.querySelectorAll('.row-check:checked');

    // Se force for false, fecha explicitamente
    if (force === false) {
        bar.classList.remove('show');
        document.querySelectorAll('.row-check').forEach(cb => cb.checked = false); // Limpa seleção
        return;
    }

    // Se houver algum checado, mostra
    if (checkboxes.length > 0) {
        bar.classList.add('show');
        document.querySelector('.batch-count').innerText = `${checkboxes.length} processo(s) selecionado(s)`;
    } else {
        bar.classList.remove('show');
    }
}

// 3. Quick Preview Drawer
function openDrawer(processId) {
    const drawer = document.getElementById('preview-drawer');
    const body = document.getElementById('drawer-body');

    // Simulação de Dados para Preview
    let content = '';
    if (processId === '5001234') {
        content = `
            <div style="margin-bottom: 20px;">
                <span class="badge bg-success" style="margin-bottom: 8px;">🟢 APTO PARA MINUTA</span>
                <h4 style="color: var(--tjmg-brand);">Proc. 5001234-88.2025...</h4>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Consumidor • Telefonia • Dano Moral</p>
            </div>
            
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
                <h5 style="font-size: 0.85rem; text-transform: uppercase;">Resumo IA</h5>
                <p style="font-size: 0.9rem; margin-top: 4px;">Ação de indenização por interrupção de serviço. Autor alega 5 dias sem sinal. Réu não contestou este ponto específico.</p>
            </div>

            <div style="margin-bottom: 16px;">
                 <h5 style="font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">Tags</h5>
                 <span class="badge" style="background:#e0f2fe; color:#0369a1;">Prioridade Legal</span>
                 <span class="badge" style="background:#f3e8ff; color:#7c3aed;">Cluster Telefonia</span>
            </div>
        `;
    } else {
        content = '<p>Carregando dados do processo...</p>';
    }

    body.innerHTML = content;
    drawer.classList.add('open');
}

function closeDrawer() {
    document.getElementById('preview-drawer').classList.remove('open');
}

// 4. SEI Suggestion Toggle
function toggleSeiSuggestion(id) {
    const el = document.getElementById(id);
    if (el) {
        el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }
}

// 5. Sidebar Toggle (Focus Mode)
function toggleSidebar() {
    const sb = document.getElementById('focus-sidebar');
    if (sb) {
        sb.classList.toggle('collapsed');
        // Logic to show/hide text vs icons could be added here
    }
}

// Inicialização e Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    console.log('TJMG Assistente 2.0 Iniciado');

    // Chat Button Listener
    const btnSend = document.querySelector('.btn-send');
    if (btnSend) {
        btnSend.addEventListener('click', () => {
            const input = document.querySelector('.chat-input-area input');
            if (input && input.value.trim() !== "") {
                addMessage(input.value, 'user');
                input.value = '';
                // Simulação de resposta da IA
                setTimeout(() => {
                    addMessage("Entendido. Atualizando a minuta conforme sua solicitação.", 'ai');
                }, 1000);
            }
        });

        // Permitir envio com Enter
        const input = document.querySelector('.chat-input-area input');
        if (input) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') btnSend.click();
            });
        }
    }

    // Ghost Text - Autocomplete Logic
    const editor = document.querySelector('.document-paper');
    if (editor) {
        editor.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                const ghost = document.getElementById('ghost-proposal');
                if (ghost) {
                    e.preventDefault();
                    const text = ghost.innerText.replace(' [TAB]', '');
                    const textNode = document.createTextNode(text);
                    ghost.parentNode.replaceChild(textNode, ghost);

                    const range = document.createRange();
                    const sel = window.getSelection();
                    range.setStartAfter(textNode);
                    range.collapse(true);
                    sel.removeAllRanges();
                    sel.addRange(range);

                    addMessage("Sugestão de dispositivo aceita.", 'ai');
                }
            }
        });
    }
});

function addMessage(text, type) {
    const chatMessages = document.querySelector('.chat-messages');
    if (!chatMessages) return;

    const div = document.createElement('div');
    div.classList.add('msg', type);
    div.innerHTML = type === 'ai' ? `<strong>Assistente:</strong> ${text}` : text;

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- EXECUTIVE DASHBOARD LOGIC ---

// 6. Map Layer Switching
function setMapLayer(layer) {
    // Buttons Active State
    document.querySelectorAll('.map-filter').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active'); // Assumes onclick passes event or we use 'this'

    // Map Layer Class
    const map = document.getElementById('mg-map-visual');
    map.className = `mg-map layer-${layer}`;

    // Update Data Mockup (Visual Only)
    // In a real app, this would re-render data-values based on the layer
}

// 7. Workforce Simulator
function updateSimulator(val) {
    document.getElementById('sim-val').innerText = val;

    // Simple mock math logic
    const impact = val * 8.5; // 8.5% improvement per judge
    document.getElementById('sim-impact').innerText = `+${impact.toFixed(1)}%`;
    document.getElementById('sim-impact').style.color = impact > 0 ? '#4ade80' : '#fff';

    // Date Estimate
    const dates = ["Nunca", "2 Anos", "18 Meses", "1 Ano", "6 Meses", "3 Meses"];
    const index = Math.min(Math.floor(val / 2), dates.length - 1);
    document.getElementById('sim-date').innerText = dates[index];
}



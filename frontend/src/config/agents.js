// Central agent definitions — single source of truth for all sidebar agents
import {
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib
} from 'react-icons/fa6';

const agents = [
    {
        id: 'gabinete-civel',
        icon: 'FaScaleBalanced',
        name: 'Gabinete Cível',
        desc: 'Assistente integral: triagem, gestão processual e minutas',
        color: '#4F46E5',
        promptModule: () => import('../prompts/gabineteCivel.js'),
    },
    {
        id: 'engenheiro-prompt',
        icon: 'FaPenNib',
        name: 'Engenheiro de Prompt',
        desc: 'Cria prompts otimizados via entrevista guiada',
        color: '#7C3AED',
        promptModule: () => import('../prompts/engenheiroPrompt.js'),
    },
    {
        id: 'gabinete-penal',
        icon: 'FaScaleBalanced',
        name: 'Gabinete Penal',
        desc: 'Gabinete Criminal: triagem de risco, sentenças e execução penal',
        color: '#DC2626',
        promptModule: () => import('../prompts/gabinetePenal.js'),
    },
    // Futuros agentes serão adicionados aqui pelo usuário
];

export default agents;

// Central agent definitions — single source of truth for all sidebar agents
import {
    FaScaleBalanced, FaFileLines, FaMagnifyingGlass,
    FaBookOpen, FaPenNib
} from 'react-icons/fa6';

const agents = [
    {
        id: 'gabinete-civel',
        icon: 'FaScaleBalanced',
        name: 'Gabinete',
        desc: 'Assistente integral: triagem, gestão processual e minutas',
        color: '#4F46E5',
        promptModule: () => import('../prompts/gabineteCivel.js'),
    },
    // Futuros agentes serão adicionados aqui pelo usuário
];

export default agents;

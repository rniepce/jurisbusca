// Central agent definitions — single source of truth for all sidebar agents

const agents = [
    {
        id: 'gabinete-1.0',
        icon: 'FaScaleBalanced',
        name: 'Gabinete 1.0',
        desc: 'Assistente integral: triagem, gestão e minutas com precedentes',
        color: '#10B981',
        engineId: 'v0',
        promptModule: () => import('../prompts/gabineteCivelV0.js'),
    },
    {
        id: 'gabinete-1.1',
        icon: 'FaPenNib',
        name: 'Gabinete 1.1',
        desc: 'Assessor jurídico RAG: raio-x, deliberação e minuta mimetizada',
        color: '#4285F4',
        engineId: 'v1',
        promptModule: () => import('../prompts/gabineteCivel.js'),
    },
    {
        id: 'gabinete-agentico',
        icon: 'FaBookOpen',
        name: 'Gabinete modo agêntico',
        desc: 'Pipeline autônomo: triagem → redação → auditoria (3 agentes)',
        color: '#D97706',
        engineId: 'v2',
        promptModule: () => import('../prompts/gabineteCivel.js'),
    },
    {
        id: 'auditor-qa',
        icon: 'FaClipboardCheck',
        name: 'Revisor (QA)',
        desc: 'Auditor de conformidade fática e eficiência em minutas',
        color: '#EF4444',
        engineId: 'v0',
        promptModule: () => import('../prompts/auditorQA.js'),
        autoAction: {
            label: '🔍 Revisar Minuta',
            requiresMinuta: true,
            requiresUploadedText: true,
        },
    }
];

export default agents;

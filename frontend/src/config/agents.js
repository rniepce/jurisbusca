// Central agent definitions — single source of truth for all sidebar agents

const agents = [
    {
        id: 'gabinete-civel',
        icon: 'FaScaleBalanced',
        name: 'Gabinete',
        desc: 'Assistente integral: triagem, gestão processual e minutas',
        color: '#4F46E5',
        promptModule: () => import('../prompts/gabineteCivel.js'),
    },
    {
        id: 'auditor-qa',
        icon: 'FaSearchCheck',
        name: 'Revisor (QA)',
        desc: 'Auditor de conformidade fática e eficiência em minutas',
        color: '#10B981',
        promptModule: () => import('../prompts/auditorQA.js'),
    }
];

export default agents;

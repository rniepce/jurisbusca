// Central agent definitions — single source of truth for all sidebar agents

const agents = [
    {
        id: 'gabinete-2.0',
        icon: 'FaScaleBalanced',
        name: 'Gabinete 2.0',
        desc: 'Assessor completo: raio-x → deliberação → minuta mimetizada + auditoria',
        color: '#10B981',
        engineId: 'v1',
        promptModule: () => import('../prompts/gabineteCivelV2.js'),
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

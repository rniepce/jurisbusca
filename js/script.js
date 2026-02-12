console.log("Assistente TJMG Loaded");

document.addEventListener('DOMContentLoaded', () => {
    const actionButtons = document.querySelectorAll('.action-card');

    actionButtons.forEach(button => {
        button.addEventListener('click', () => {
            console.log('Action Clicked:', button.innerText);
            // Here you would implement the navigation or logic for each action
        });
    });

    const chatInput = document.querySelector('.chat-input');
    const sendButton = document.querySelector('.send-btn');

    const handleSend = () => {
        const message = chatInput.value.trim();
        if (message) {
            console.log('Sending message:', message);
            chatInput.value = '';
            // Implement chat logic here
        }
    };

    sendButton.addEventListener('click', handleSend);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    });
});

const chatForm = document.getElementById('chat-form');
const chatLog = document.getElementById('chat-log');
const providerSelect = document.getElementById('provider');
const apiKeyInput = document.getElementById('api-key');
const modelInput = document.getElementById('model');
const baseUrlInput = document.getElementById('base-url');

const providers = {
    'openai': {
        'api-key': '',
        'model': 'text-davinci-003',
        'base-url': 'https://api.openai.com/v1/'
    },
    'anthropic': {
        'api-key': '',
        'model': 'claude-v1',
        'base-url': 'https://api.anthropic.com/v1/'
    },
    'openai-compatible': {
        'api-key': '',
        'model': 'text-davinci-003',
        'base-url': 'https://api.openai.com/v1/'
    },
    'google': {
        'api-key': '',
        'model': 'text-v1',
        'base-url': 'https://language.googleapis.com/v1/'
    },
    'groq': {
        'api-key': '',
        'model': 'groq-v1',
        'base-url': 'https://api.groq.com/v1/'
    },
    'lmstudio': {
        'api-key': '',
        'model': 'lmstudio-v1',
        'base-url': 'https://api.lmstudio.ai/v1/'
    },
    'ollama': {
        'api-key': '',
        'model': 'ollama-v1',
        'base-url': 'https://api.ollama.ai/v1/'
    }
};

providerSelect.addEventListener('change', (e) => {
    const provider = e.target.value;
    const providerConfig = providers[provider];
    apiKeyInput.value = providerConfig['api-key'];
    modelInput.value = providerConfig['model'];
    baseUrlInput.value = providerConfig['base-url'];
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const provider = providerSelect.value;
    const apiKey = apiKeyInput.value;
    const model = modelInput.value;
    const baseUrl = baseUrlInput.value;
    const messageInput = document.getElementById('message');
    const message = messageInput.value;
    // Make API call to the selected provider
    try {
        const endpoint = `${baseUrl}completions`;
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    'model': model,
                    'prompt': message,
                    'max_tokens': 2048
                })
            });
        if (!response.ok) {
            const errorData = await response.json();
            console.error(`Error: ${errorData.error}`);
            const chatLogHtml = `<p>Error: ${errorData.error}</p>`;
            chatLog.innerHTML += chatLogHtml;
        } else {
            const data = await response.json();
            const responseMessage = data.choices[0].text;
            const chatLogHtml = `<p>You: ${message}</p><p>AI: ${responseMessage}</p>`;
            chatLog.innerHTML += chatLogHtml;
        }
        messageInput.value = '';
    } catch (error) {
        console.error(error);
    }
});

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

chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const provider = providerSelect.value;
    const apiKey = apiKeyInput.value;
    const model = modelInput.value;
    const baseUrl = baseUrlInput.value;
    // Make API call to the selected provider
    console.log(`Making API call to ${provider} with API key ${apiKey}, model ${model}, and base URL ${baseUrl}`);
});

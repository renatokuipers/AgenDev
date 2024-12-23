import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ProviderConfig() {
  const [providers, setProviders] = useState([]);
  const [newProvider, setNewProvider] = useState({
    name: '',
    api_key: '',
    temperature: 0,
    max_tokens: 0,
    top_p: 0,
    context_length: 0,
  });

  const handleAddProvider = async () => {
    // TO DO: implement add provider
  };

  return (
    <div>
      <form>
        <label>
          Name:
          <input
            type="text"
            value={newProvider.name}
            onChange={(e) => setNewProvider({ ...newProvider, name: e.target.value })}
          />
        </label>
        <label>
          API Key:
          <input
            type="text"
            value={newProvider.api_key}
            onChange={(e) => setNewProvider({ ...newProvider, api_key: e.target.value })}
          />
        </label>
        <label>
          Temperature:
          <input
            type="number"
            value={newProvider.temperature}
            onChange={(e) => setNewProvider({ ...newProvider, temperature: e.target.value })}
          />
        </label>
        <label>
          Max Tokens:
          <input
            type="number"
            value={newProvider.max_tokens}
            onChange={(e) => setNewProvider({ ...newProvider, max_tokens: e.target.value })}
          />
        </label>
        <label>
          Top P:
          <input
            type="number"
            value={newProvider.top_p}
            onChange={(e) => setNewProvider({ ...newProvider, top_p: e.target.value })}
          />
        </label>
        <label>
          Context Length:
          <input
            type="number"
            value={newProvider.context_length}
            onChange={(e) => setNewProvider({ ...newProvider, context_length: e.target.value })}
          />
        </label>
        <button onClick={handleAddProvider}>Add Provider</button>
      </form>
      <ul>
        {providers.map((provider, index) => (
          <li key={index}>{provider.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default ProviderConfig;

import React, { useEffect, useState } from 'react';
import { Trash2, AlertTriangle, Zap, Code, Cpu, MessageSquare, ArrowLeftRight } from 'lucide-react';
import axios from 'axios';

const ICON_MAP = {
  cpu: <Cpu size={16} />,
  zap: <Zap size={16} />,
  'arrow-left-right': <ArrowLeftRight size={16} />,
  code: <Code size={16} />,
  // add any others here...
};

export const Settings = ({ config, setConfig, onClearChat, apiStatus }) => {
  const [modelOptions, setModelOptions] = useState([]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await axios.get('http://localhost:6145/api/ollama/models');
        const models = response.data.models || [];
  
        const withIcons = models
          .filter(modelName => modelName !== 'NAME')
          .map(modelName => {
            const cleanName = modelName.replace(':latest', '');
            return {
              value: cleanName,
              label: cleanName,
              description: '', // Add actual description if needed
              icon: ICON_MAP['cpu'] || <Cpu size={16} />,
            };
          });
         withIcons.push({
          value: "openai",
          label: "OpenAI",
          description: "",
          icon: ICON_MAP['cpu'] || <Cpu size={16} />,
        })
        setModelOptions(withIcons);
      } catch (error) {
        console.error('Failed to load models:', error);
      }
    };
  
    loadModels();
  }, []);
  

  return (
    <div className="flex-1 bg-gray-100 h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <MessageSquare className="mr-2" size={24} />
          Settings
        </h2>

        {/* Model Configuration Section */}
        <div className="mb-8 bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 pb-2 border-b border-gray-200">
            Model Configuration
          </h3>
          
          <div className="space-y-6">
            <div>
              <label className="block text-gray-700 mb-2 font-medium">Language Model</label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {modelOptions.map(option => (
                  <div 
                    key={option.value}
                    className={`
                      border ${config?.model === option.value 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200'} 
                      rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors
                    `}
                    onClick={() => setConfig({...config, model: option.value})}
                  >
                    <div className="flex items-start">
                      <div className={`rounded-full p-2 ${
                        config.model === option.value 
                          ? 'bg-blue-100 text-blue-600' 
                          : 'bg-gray-100 text-gray-500'
                      }`}>
                        {option.icon}
                      </div>
                      <div className="ml-3">
                        <h4 className="font-medium text-gray-900">{option.label}</h4>
                        <p className="text-sm text-gray-500">{option.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
             
            {/* Temperature Slider */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <label className="text-gray-700 font-medium">Temperature</label>
                <span className="text-blue-600 font-medium">{config.temperature}</span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-500">Precise</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={config.temperature}
                  onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <span className="text-sm text-gray-500">Creative</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">
                Lower values make responses more deterministic and focused. Higher values increase creativity and variability.
              </p>
            </div>
            
            {/* Top K Results */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <label className="text-gray-700 font-medium">Top K Results</label>
                <span className="text-blue-600 font-medium">{config.topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={config.topK}
                onChange={(e) => setConfig({...config, topK: parseInt(e.target.value)})}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <p className="mt-2 text-sm text-gray-500">
                Number of code snippets to retrieve from your knowledge bases.
              </p>
            </div>
            
            {/* Max Tokens */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <label className="text-gray-700 font-medium">Max Tokens</label>
                <span className="text-blue-600 font-medium">{config.maxTokens}</span>
              </div>
              <input
                type="range"
                min="100"
                max="4000"
                step="100"
                value={config.maxTokens}
                onChange={(e) => setConfig({...config, maxTokens: parseInt(e.target.value)})}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <p className="mt-2 text-sm text-gray-500">
                Maximum length of the generated response.
              </p>
            </div>

            {/* OpenAI API Key */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <label className="text-gray-700 font-medium">OpenAI API Key</label>
              </div>
              <input
                type="text"
                value={config.openaiKey || localStorage.getItem("openapikey")}
                onChange={(e) => {
                  localStorage.setItem("openapikey", e.target.value);
                  setConfig({...config, openaiKey: e.target.value})
                }}
                className="w-full h-15 pl-4 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>
          </div>
        </div>
        
        {/* Session Section */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 pb-2 border-b border-gray-200">
            Session
          </h3>
          
          <div className="space-y-4">
            {apiStatus && (
              <div className={`flex items-center p-3 rounded-lg ${
                apiStatus.status === 'ok' 
                  ? 'bg-green-100 text-green-800'
                  : 'bg-yellow-100 text-yellow-800'
              }`}>
                <div className={`rounded-full w-3 h-3 ${
                  apiStatus.status === 'ok' ? 'bg-green-500' : 'bg-yellow-500'
                } mr-3`}></div>
                <span className="text-sm font-medium">
                  API Status: {apiStatus.status === 'ok' ? 'Connected' : 'Limited Connectivity'}
                </span>
              </div>
            )}
            
            <div className="flex flex-col space-y-3">
              <button 
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors duration-200 max-w-xs"
                onClick={onClearChat}
              >
                <Trash2 size={18} />
                <span>Clear Chat History</span>
              </button>
              
              <div className="flex items-start space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertTriangle size={20} className="text-red-500 flex-shrink-0 mt-0.5" />
                <span className="text-sm text-red-600">
                  This will permanently delete your entire conversation history.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
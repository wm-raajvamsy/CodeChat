import React, { useState, useEffect, useRef } from 'react';
import { FilePicker } from './components/FilePicker';
import { ChatInterface } from './components/chatInterface';
import Sidebar from './components/Sidebar';
import KnowledgeBase from './components/KnowledgeBase';
import { Settings } from './components/Settings';
import { Login } from './components/Login';
import { processChat, processChatStreaming, cancelStream } from './services/chatService';
import {
  createKnowledgeBase,
  getAllKnowledgeBases,
  removeKnowledgeBase,
  getApiStatus,
  rebuildKnowledgeBase
} from './services/knowledgeBaseService';
import {
  login,
  register,
  loginWithGoogle,
  loginWithGitHub,
  logout,
  isAuthenticated,
  getCurrentUser
} from './services/authService';
import { isEmpty } from 'lodash';
import { AlertCircle, X, Send, Slash, ChevronLeft, ChevronRight, Book, Settings as SettingsIcon } from 'lucide-react';

const App = () => {
  // User state
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // UI state
  const [activeTab, setActiveTab] = useState('chat');
  const [configPanelExpanded, setConfigPanelExpanded] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);

  // Chat state
  const [messages, setMessages] = useState([]);
  const [streamController, setStreamController] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // Config state
  const [config, setConfig] = useState({
    temperature: 0.1,
    topK: 40,
    maxTokens: 1000,
    model: 'qwen2.5:14b-instruct',
    streamingEnabled: true
  });

  // Knowledge base state
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [creatingKnowledgeBase, setCreatingKnowledgeBase] = useState(false);
  const [selectedKnowledgeBases, setSelectedKnowledgeBases] = useState([]);

  // Load initial data
  useEffect(() => {
    // Check authentication
    // if (isAuthenticated()) {
    //   const currentUser = getCurrentUser();
    //   if (currentUser) {
    //     setUser(currentUser);
    setIsAuthenticated(true);
    //   }
    // }

    // Load chat history
    const savedMessages = localStorage.getItem('chatHistory');
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }

    // Load knowledge bases
    loadKnowledgeBases();

    // Load config
    const savedConfig = localStorage.getItem('config');
    if (savedConfig) {
      setConfig(JSON.parse(savedConfig));
    }

    // Check API status
    checkApiStatus();
  }, []);

  // Save chat history when messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chatHistory', JSON.stringify(messages));
    }
  }, [messages]);

  // Save config when it changes
  useEffect(() => {
    localStorage.setItem('config', JSON.stringify(config));
  }, [config]);

  // Cleanup streaming on unmount
  useEffect(() => {
    return () => {
      if (streamController) {
        cancelStream(streamController);
      }
    };
  }, [streamController]);

  const checkApiStatus = async () => {
    try {
      const status = await getApiStatus();
      setApiStatus(status);
      console.log('API Status:', status);
    } catch (err) {
      console.error('Error checking API status:', err);
      setError('API is unreachable. Some features may not work correctly.');
    }
  };

  const handleLogin = async (email, password) => {
    setError(null);
    try {
      const userData = await login(email, password);
      setUser(userData.user);
      setIsAuthenticated(true);
      loadKnowledgeBases();
      return true;
    } catch (err) {
      setError(err.message || 'Failed to login');
      return false;
    }
  };

  const handleRegister = async (userData) => {
    setError(null);
    try {
      const response = await register(userData);
      setUser(response.user);
      setIsAuthenticated(true);
      return true;
    } catch (err) {
      setError(err.message || 'Failed to register');
      return false;
    }
  };

  const handleGoogleLogin = async (idToken) => {
    setError(null);
    try {
      const userData = await loginWithGoogle(idToken);
      setUser(userData.user);
      setIsAuthenticated(true);
      loadKnowledgeBases();
      return true;
    } catch (err) {
      setError(err.message || 'Failed to login with Google');
      return false;
    }
  };

  const handleGitHubLogin = async (code) => {
    setError(null);
    try {
      const userData = await loginWithGitHub(code);
      setUser(userData.user);
      setIsAuthenticated(true);
      loadKnowledgeBases();
      return true;
    } catch (err) {
      setError(err.message || 'Failed to login with GitHub');
      return false;
    }
  };

  const handleLogout = () => {
    logout();
    setUser(null);
    setIsAuthenticated(false);
    setMessages([]);
    setSelectedKnowledgeBases([]);
  };

  const handleSendMessage = async (inputMessage) => {
    if (!inputMessage.trim()) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      sender: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);

    // If streaming is already in progress, cancel it
    if (isStreaming && streamController) {
      cancelStream(streamController);
    }

    // Parse knowledge base tags from message (#knowledge1, #knowledge2, etc.)
    const kbRegex = /#([a-zA-Z0-9_-]+)/g;
    const kbMatches = [...inputMessage.matchAll(kbRegex)];
    const kbTags = kbMatches.map(match => match[1]);

    // Get selected knowledge bases by tag
    const kbsToUse = kbTags.length > 0
      ? knowledgeBases.filter(kb => kbTags.includes(kb.id) || kbTags.includes(kb.name))
      : selectedKnowledgeBases.length > 0
        ? selectedKnowledgeBases
        : [];

    try {
      if (config.streamingEnabled) {
        // Set up streaming response
        setLoading(true);
        setIsStreaming(true);

        // Create a temporary message that will be updated during streaming
        const tempAssistantMessage = {
          id: Date.now() + 1,
          sender: 'assistant',
          content: '',
          timestamp: new Date().toISOString(),
          isStreaming: true
        };

        setMessages(prev => [...prev, tempAssistantMessage]);

        // Handle token callback (update message incrementally)
        const handleToken = (token) => {
          setMessages(prev => {
            const updated = [...prev];
            const lastMessage = updated[updated.length - 1];
            if (lastMessage.isStreaming) {
              lastMessage.content += token;
            }
            return updated;
          });
        };

        // Handle completion
        const handleComplete = (fullResponse) => {
          setMessages(prev => {
            const updated = [...prev];
            const lastMessage = updated[updated.length - 1];
            if (lastMessage.isStreaming) {
              lastMessage.content = fullResponse;
              lastMessage.isStreaming = false;
            }
            return updated;
          });
          setLoading(false);
          setIsStreaming(false);
          setStreamController(null);
        };

        // Handle error
        const handleError = (err) => {
          setError(err.message || 'Failed to get response');
          setMessages(prev => {
            const updated = [...prev];
            const lastMessage = updated[updated.length - 1];
            if (lastMessage.isStreaming) {
              lastMessage.content += "\n\n[Error: Failed to complete response]";
              lastMessage.isStreaming = false;
              lastMessage.error = true;
            }
            return updated;
          });
          setLoading(false);
          setIsStreaming(false);
          setStreamController(null);
        };

        // Start streaming
        const controller = await processChatStreaming(
          inputMessage,
          kbsToUse,
          config,
          handleToken,
          handleComplete,
          handleError
        );

        setStreamController(controller);
      } else {
        // Non-streaming response
        setLoading(true);

        // Process message
        const response = await processChat(
          inputMessage,
          kbsToUse,
          config
        );

        // Add assistant message to chat
        const assistantMessage = {
          id: Date.now() + 1,
          sender: 'assistant',
          content: response,
          timestamp: new Date().toISOString(),
        };

        setMessages(prev => [...prev, assistantMessage]);
        setLoading(false);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err.message || 'Failed to get response. Please try again.');

      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'system',
        content: 'Failed to get response. Please try again.',
        error: true,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
      setLoading(false);
      setIsStreaming(false);
    }
  };

  const handleCancelGeneration = () => {
    if (streamController) {
      cancelStream(streamController);
      setStreamController(null);
      setIsStreaming(false);
      setLoading(false);

      // Mark the current streaming message as cancelled
      setMessages(prev => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        if (lastMessage.isStreaming) {
          lastMessage.content += "\n\n[Response cancelled]";
          lastMessage.isStreaming = false;
          lastMessage.cancelled = true;
        }
        return updated;
      });
    }
  };

  const handleCreateKnowledgeBase = async (name, gitUrl, description = '') => {
    setCreatingKnowledgeBase(true);
    setError(null);

    try {
      const newKB = await createKnowledgeBase(name, gitUrl, description);
      setKnowledgeBases(prev => [...Object.values(prev), newKB]);
      return true;
    } catch (err) {
      console.error('Error creating knowledge base:', err);
      setError(err.message || 'Failed to create knowledge base. Please check the repository path and try again.');
      return false;
    } finally {
      setCreatingKnowledgeBase(false);
    }
  };

  const handleRemoveKnowledgeBase = async (kbId) => {
    try {
      await removeKnowledgeBase(kbId);
      setKnowledgeBases(prev => Object.values(prev).filter(kb => kb.id !== kbId));
      setSelectedKnowledgeBases(prev => Object.values(prev).filter(kb => kb.id !== kbId));
      return true;
    } catch (err) {
      console.error('Error removing knowledge base:', err);
      setError(err.message || 'Failed to remove knowledge base.');
      return false;
    }
  };

  const loadKnowledgeBases = async () => {
    if (!isAuthenticated) return;

    try {
      const kbs = await getAllKnowledgeBases();
      setKnowledgeBases(kbs);
    } catch (err) {
      console.error('Error loading knowledge bases:', err);
      setError('Failed to load knowledge bases.');
    }
  };

  const toggleKnowledgeBase = (kb) => {
    if (selectedKnowledgeBases.some(selected => selected.id === kb.id)) {
      setSelectedKnowledgeBases(prev => Object.values(prev).filter(selected => selected.id !== kb.id));
    } else {
      setSelectedKnowledgeBases(prev => [...Object.values(prev), kb]);
    }
  };

  const clearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      setMessages([]);
      localStorage.removeItem('chatHistory');
    }
  };

  // If not authenticated, show login screen
  if (!isAuthenticated) {
    return (
      <Login
        onLogin={handleLogin}
        onRegister={handleRegister}
        onGoogleLogin={handleGoogleLogin}
        onGitHubLogin={handleGitHubLogin}
        error={error}
      />
    );
  }

  return (
    <div className="flex h-screen bg-gray-100 text-gray-900">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        user={user}
        onLogout={handleLogout}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Error banner */}
        {error && (
          <div className="bg-red-500 text-white px-4 py-2 flex items-center justify-between">
            <div className="flex items-center">
              <AlertCircle size={20} className="mr-2" />
              <p>{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="p-1 hover:bg-red-600 rounded-full focus:outline-none"
            >
              <X size={18} />
            </button>
          </div>
        )}

        {/* Main content area */}
        <main className="flex-1 overflow-hidden flex">
          {/* Chat tab */}
          {activeTab === 'chat' && (
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Use ChatInterface component */}
              <ChatInterface 
                messages={messages}
                onSendMessage={handleSendMessage}
                isLoading={loading}
                selectedKnowledgeBases={selectedKnowledgeBases}
              />
            </div>
          )}

          {/* Knowledge base tab */}
          {activeTab === 'knowledge' && (
            <KnowledgeBase
              knowledgeBases={knowledgeBases}
              selectedKnowledgeBases={selectedKnowledgeBases}
              onToggleKnowledgeBase={toggleKnowledgeBase}
              onCreateKnowledgeBase={handleCreateKnowledgeBase}
              onRemoveKnowledgeBase={handleRemoveKnowledgeBase}
              onRebuildKnowledgeBase={rebuildKnowledgeBase}
              isCreating={creatingKnowledgeBase}
              error={error}
              onRefresh={loadKnowledgeBases}
            />
          )}

          {/* Settings tab */}
          {activeTab === 'settings' && (
            <Settings
              config={config}
              setConfig={setConfig}
              onClearChat={clearChat}
              apiStatus={apiStatus}
            />
          )}

          {/* Config panel */}
          <div
            className={`${configPanelExpanded ? 'w-72' : 'w-16'
              } relative border-l border-gray-700 bg-gray-900 transition-all duration-300 ease-in-out flex flex-col`}
          >
            {/* Toggle button */}
            <button
              onClick={() => setConfigPanelExpanded(!configPanelExpanded)}
              className="absolute -left-4 top-6 bg-blue-600 text-white rounded-full p-2 shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 z-20 transition-all duration-200"
            >
              {configPanelExpanded ? (
                <ChevronRight size={18} />
              ) : (
                <ChevronLeft size={18} />
              )}
            </button>

            {/* Header */}
            <div className="h-16 px-4 flex items-center border-b border-gray-700 bg-gray-800">
              {configPanelExpanded && (
                <h2 className="text-base font-semibold text-white p-4">
                  Configuration
                </h2>
              )}
            </div>

            {/* Body */}
            {configPanelExpanded ? (
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Streaming Toggle */}
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <div className="flex items-center justify-between">
                    <label htmlFor="streaming-toggle" className="flex items-center cursor-pointer">
                      <div className="relative">
                        <input
                          id="streaming-toggle"
                          type="checkbox"
                          checked={config.streamingEnabled}
                          onChange={(e) =>
                            setConfig({ ...config, streamingEnabled: e.target.checked })
                          }
                          className="sr-only"
                        />
                        <div className={`block w-10 h-6 rounded-full ${config.streamingEnabled ? 'bg-blue-600' : 'bg-gray-600'}`}></div>
                        <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${config.streamingEnabled ? 'transform translate-x-4' : ''}`}></div>
                      </div>
                      <span className="ml-3 text-sm font-medium text-gray-200">
                        Streaming responses
                      </span>
                    </label>
                  </div>
                  <p className="mt-2 text-xs text-gray-400">
                    See responses appear in real-time as they're generated.
                  </p>
                </div>

                {/* Knowledge Bases */}
                <div className="bg-gray-800 p-2 rounded-lg border border-gray-700">
                  <h3 className="text-sm font-semibold text-gray-100 mb-3 flex items-center">
                    <Book size={16} className="mr-2" />
                    Available Knowledge Bases
                  </h3>

                  {isEmpty(knowledgeBases) || knowledgeBases.length === 0 ? (
                    <div className="text-center py-6 bg-gray-850 rounded-lg">
                      <Book size={24} className="mx-auto text-gray-500 mb-3" />
                      <p className="text-sm text-gray-400 mb-2">
                        No knowledge bases created yet
                      </p>
                      {/* <button
                        onClick={() => setActiveTab('knowledge')}
                        className="mt-2 text-xs font-medium bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md transition-all duration-200"
                      >
                        Create your first knowledge base
                      </button> */}
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1 mt-3">
                      {Object.values(knowledgeBases).map((kb) => (
                        <div
                          key={kb.id}
                          className="flex items-center justify-between p-2 rounded-md hover:bg-gray-700 transition-all duration-200"
                        >
                          <label htmlFor={`kb-${kb.id}`} className="flex items-center truncate cursor-pointer w-full">
                            <div className="relative h-4 w-4 mr-2">
                              <input
                                type="checkbox"
                                id={`kb-${kb.id}`}
                                checked={selectedKnowledgeBases.some(
                                  (selected) => selected.id === kb.id
                                )}
                                onChange={() => toggleKnowledgeBase(kb)}
                                className="opacity-0 absolute h-4 w-4 cursor-pointer"
                              />
                              <div className={`border ${selectedKnowledgeBases.some(selected => selected.id === kb.id)
                                  ? 'bg-blue-600 border-blue-600'
                                  : 'border-gray-500'
                                } rounded h-4 w-4 flex flex-shrink-0 justify-center items-center`}>
                                {selectedKnowledgeBases.some(selected => selected.id === kb.id) && (
                                  <svg className="fill-current w-2 h-2 text-white pointer-events-none" viewBox="0 0 20 20">
                                    <path d="M0 11l2-2 5 5L18 3l2 2L7 18z" />
                                  </svg>
                                )}
                              </div>
                            </div>
                            <span className="text-sm text-gray-200 truncate">
                              {kb.name}
                            </span>
                          </label>
                          <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded-md">
                            #{kb.id.slice(0, 5)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center pt-8 space-y-6">
                <div className="tooltip relative">
                  <div className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg cursor-pointer transition-all duration-200">
                    <Book size={20} className="text-gray-400" />
                  </div>
                  <span className="tooltip-text absolute left-12 bg-gray-800 px-2 py-1 text-xs rounded hidden">
                    Knowledge Bases
                  </span>
                </div>
                <div className="tooltip relative">
                  <div className={`p-2 py-1 rounded-lg cursor-pointer transition-all duration-200 ${config.streamingEnabled ? 'bg-blue-600' : 'bg-gray-800 hover:bg-gray-700'}`}>
                    <span className="text-xs text-gray-200">
                      {config.streamingEnabled ? 'ON' : 'OFF'}
                    </span>
                  </div>
                  <span className="tooltip-text absolute left-12 bg-gray-800 px-2 py-1 text-xs rounded hidden">
                    Streaming
                  </span>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
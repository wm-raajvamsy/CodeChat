import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { FilePicker } from './FilePicker';
import { SendIcon, PaperclipIcon } from 'lucide-react';
import { TypingIndicator } from './TypingIndicator';

export const ChatInterface = ({
  messages,
  onSendMessage,
  isLoading,
  selectedKnowledgeBases
}) => {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleFileSelected = async (file) => {
    if (!selectedKnowledgeBases || selectedKnowledgeBases.length === 0) {
      setUploadError("Please select a knowledge base first");
      return;
    }

    const kbId = selectedKnowledgeBases[0].id;
    setIsUploading(true);
    setUploadError(null);

  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Chat messages area */}
      <div className="flex-grow overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
            <div className="bg-blue-100 p-6 rounded-full mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <h2 className="text-xl font-medium mb-2">Start a conversation</h2>
            <p className="max-w-md">Ask questions about your code repositories or use the knowledge bases to get specific answers.</p>
          </div>
        ) :
          messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}

        {isLoading && (
          <div className="ml-auto mr-auto">
            <TypingIndicator />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 p-4 bg-white shadow-lg">
        {uploadError && (
          <div className="mb-2 px-4 py-2 bg-red-50 border border-red-200 text-red-600 rounded-md text-sm">
            {uploadError}
          </div>
        )}

        {/* Knowledge base tags */}
        <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          {selectedKnowledgeBases.length > 0 ? (
            <>
              <span className="text-gray-500">Using:</span>
              {selectedKnowledgeBases.map(kb => (
                <span key={kb.id} className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded-md">
                  #{kb.name}
                </span>
              ))}
            </>
          ) : (
            <span className="text-gray-500">Using all knowledge bases</span>
          )}
        </div>

        {/* Message input form */}
        <form onSubmit={handleSubmit} className="flex items-center space-x-2">
          <FilePicker onFileSelected={handleFileSelected} />

          <div className="flex-grow relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isLoading ? "Waiting for response..." : "Ask about your code..."}
              disabled={isLoading || isUploading}
              ref={inputRef}
              className="w-full px-4 py-3 bg-gray-50 text-gray-900 rounded-lg border border-gray-300 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed"
            />
          </div>

          <button
            type="submit"
            disabled={!input.trim() || isLoading || isUploading}
            className="px-4 py-3 bg-indigo-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-700 transition-colors"
          >
            {isUploading ? "Uploading..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};
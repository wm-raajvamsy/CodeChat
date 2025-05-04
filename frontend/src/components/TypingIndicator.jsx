// components/TypingIndicator.jsx
import React from 'react';

export const TypingIndicator = () => {
  return (
    <div className="flex items-center space-x-3 p-4 bg-indigo-50 rounded-lg max-w-md animate-pulse">
      <div className="flex space-x-2">
        <span className="h-2.5 w-2.5 bg-indigo-500 rounded-full animate-bounce" />
        <span className="h-2.5 w-2.5 bg-indigo-500 rounded-full animate-bounce delay-100" />
        <span className="h-2.5 w-2.5 bg-indigo-500 rounded-full animate-bounce delay-200" />
      </div>
      <span className="text-indigo-700 text-sm font-medium">Thinking...</span>
    </div>
  );
};

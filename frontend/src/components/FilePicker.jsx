import React from 'react';
import { PaperclipIcon } from 'lucide-react';

export const FilePicker = ({ onFileSelected }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileSelected(file);
    }
  };

  return (
    <div className="relative">
      <label className="inline-flex items-center justify-center p-3 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded-lg cursor-pointer transition-colors shadow-sm">
        <input
          type="file"
          onChange={handleFileChange}
          accept=".py,.js,.jsx,.ts,.tsx,.html,.css,.json,.yaml,.yml,.md,.txt"
          className="hidden"
        />
        <PaperclipIcon size={20} className="text-indigo-600" />
      </label>
    </div>
  );
};
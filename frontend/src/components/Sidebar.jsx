import React, { useState } from 'react';
import { Menu, X, MessageCircle, Book, Settings, LogOut, ChevronLeft, ChevronRight } from 'lucide-react';

export default function Sidebar({ activeTab = 'chat', setActiveTab = () => {}, user = { name: 'Alex Smith', email: 'alex@example.com' }, onLogout = () => {} }) {
  const [collapsed, setCollapsed] = useState(false);
  
  const toggleSidebar = () => setCollapsed(!collapsed);
  
  return (
    <div className={`${collapsed ? 'w-16' : 'w-64'} bg-gray-900 text-gray-100 h-screen flex flex-col justify-between transition-all duration-300 ease-in-out`}>
      {/* Header */}
      <div className="border-b border-gray-700">
        <div className="flex items-center justify-between p-4">
          {!collapsed && (
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">CodeChat</h1>
              <p className="text-xs text-gray-400 mt-1">Query your repositories intelligently</p>
            </div>
          )}
          <button 
            onClick={toggleSidebar} 
            className="p-1 rounded-md hover:bg-gray-800 focus:outline-none"
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>
      </div>
      
      {/* Navigation */}
      <nav className="flex-grow p-4">
        <div className="space-y-2">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center ${collapsed ? 'justify-center' : 'justify-start px-3'} w-full py-2 rounded-lg transition-all duration-200 ${
              activeTab === 'chat' ? 'bg-blue-600 text-white' : 'hover:bg-gray-800'
            }`}
          >
            <MessageCircle size={collapsed ? 24 : 20} />
            {!collapsed && <span className="ml-3">Chat</span>}
          </button>
          
          <button
            onClick={() => setActiveTab('knowledge')}
            className={`flex items-center ${collapsed ? 'justify-center' : 'justify-start px-3'} w-full py-2 rounded-lg transition-all duration-200 ${
              activeTab === 'knowledge' ? 'bg-blue-600 text-white' : 'hover:bg-gray-800'
            }`}
          >
            <Book size={collapsed ? 24 : 20} />
            {!collapsed && <span className="ml-3">Knowledge Bases</span>}
          </button>
          
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex items-center ${collapsed ? 'justify-center' : 'justify-start px-3'} w-full py-2 rounded-lg transition-all duration-200 ${
              activeTab === 'settings' ? 'bg-blue-600 text-white' : 'hover:bg-gray-800'
            }`}
          >
            <Settings size={collapsed ? 24 : 20} />
            {!collapsed && <span className="ml-3">Settings</span>}
          </button>
        </div>
      </nav>
      
      {/* Footer with user info */}
      <div className="border-t border-gray-700 p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0 h-9 w-9 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center text-white font-medium">
            {user?.name?.charAt(0) || '?'}
          </div>
          
          {!collapsed && (
            <div className="ml-3 overflow-hidden">
              <p className="text-sm font-medium">{user?.name || 'User'}</p>
              <p className="text-xs text-gray-400 truncate">{user?.email || ''}</p>
            </div>
          )}
        </div>
        
        {!collapsed && (
          <button
            onClick={onLogout}
            className="mt-4 w-full flex items-center justify-center px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 transition-all duration-200"
          >
            <LogOut size={16} />
            <span className="ml-2">Logout</span>
          </button>
        )}
        
        {collapsed && (
          <button
            onClick={onLogout}
            className="mt-4 w-full flex items-center justify-center p-2 rounded-lg bg-red-600 hover:bg-red-700 transition-all duration-200"
          >
            <LogOut size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
import React, { useEffect, useState } from 'react';
import { formatDate } from '../utils/helpers';
import { isEmpty } from 'lodash';
import { RefreshCw, GitBranch, Clock, Database, Tag, ExternalLink, Trash2, AlertTriangle, Plus, Loader2 } from 'lucide-react';

export default function KnowledgeBase({
  knowledgeBases = [],
  selectedKnowledgeBases = [],
  onToggleKnowledgeBase = () => {},
  onCreateKnowledgeBase = () => {},
  onRemoveKnowledgeBase = () => {},
  onRebuildKnowledgeBase = () => {},
  isCreating = false,
  onRefresh = () => {},
  error = null
}) {
  const [kbName, setKbName] = useState('');
  const [gitUrl, setGitUrl] = useState('');
  const [showConfirmRemove, setShowConfirmRemove] = useState(null);
  const [rebuildingKbId, setRebuildingKbId] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (kbName && gitUrl) {
      onCreateKnowledgeBase(kbName, gitUrl);
      setKbName('');
      setGitUrl('');
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      onRefresh();
    }, 5000);
  
    return () => clearInterval(interval);
  }, [onRefresh]);

  const confirmRemove = (id) => setShowConfirmRemove(id);
  const handleRemove = (id) => {
    onRemoveKnowledgeBase(id);
    setShowConfirmRemove(null);
  };

  // Filter out any invalid or empty knowledge bases
  const validKnowledgeBases = Array.isArray(knowledgeBases) 
    ? knowledgeBases.filter(kb => kb && kb.id && kb.name)
    : [];

  return (
    <div className="w-full mx-auto p-6 px-36 flex-col flex-1 overflow-scroll">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Knowledge Bases</h2>
          <p className="text-gray-500 mt-1">
            Add repositories to create searchable knowledge bases for your code
          </p>
        </div>
        <button 
          onClick={onRefresh}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-gray-700 transition-colors"
        >
          <RefreshCw size={16} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-md p-6 mb-8">
        <h3 className="text-lg font-medium mb-4">Create New Knowledge Base</h3>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="kb-name" className="block text-sm font-medium text-gray-700 mb-1">Knowledge Base Name</label>
              <input
                id="kb-name"
                type="text"
                value={kbName}
                onChange={(e) => setKbName(e.target.value)}
                placeholder="My Project"
                disabled={isCreating}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
              />
            </div>
            <div>
              <label htmlFor="git-url" className="block text-sm font-medium text-gray-700 mb-1">Git Repository URL</label>
              <input
                id="git-url"
                type="text"
                value={gitUrl}
                onChange={(e) => setGitUrl(e.target.value)}
                placeholder="https://github.com/username/repo.git"
                disabled={isCreating}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
              />
            </div>
          </div>
          <div className="flex justify-end">
            <button 
              onClick={handleSubmit}
              disabled={isCreating || !kbName || !gitUrl}
              className="flex items-center gap-2 px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Plus size={16} />
              {isCreating ? 'Creating...' : 'Create Knowledge Base'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 mb-6 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="space-y-6">
        {isEmpty(validKnowledgeBases) ? (
          <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-8 text-center">
            <div className="inline-block p-4 bg-gray-100 rounded-full mb-4">
              <Database size={24} className="text-gray-500" />
            </div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">No knowledge bases found</h3>
            <p className="text-gray-500">Add a repository above to get started with your first knowledge base.</p>
          </div>
        ) : (
          validKnowledgeBases.map((kb) => {
            const isActive = selectedKnowledgeBases.some((s) => s.id === kb.id);
            const statusColor = kb.status === 'ready' ? 'bg-green-500' : 
                               kb.status === 'indexing' ? 'bg-yellow-500' : 'bg-gray-500';
            
            return (
              <div key={kb.id} className="bg-white rounded-xl shadow-md overflow-hidden">
                <div className="border-b border-gray-100 p-4 md:p-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <h3 className="text-xl font-semibold text-gray-800">{kb.name}</h3>
                    <div className="flex flex-wrap items-center gap-3">
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          checked={isActive}
                          onChange={() => onToggleKnowledgeBase(kb)}
                          className="sr-only peer"
                        />
                        <div className="relative w-11 h-6 bg-gray-200 peer-checked:bg-blue-600 rounded-full peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all"></div>
                        <span className="ml-2 text-sm font-medium text-gray-700">Active</span>
                      </label>
                      <button
                        onClick={async () => {
                          setRebuildingKbId(kb.id);
                          await onRebuildKnowledgeBase(kb.id);
                          setRebuildingKbId(null);
                        }}
                        disabled={rebuildingKbId === kb.id}
                        className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-lg disabled:bg-gray-50 disabled:text-gray-400 transition-colors"
                      >
                        <span className="flex items-center gap-1">
                          <RefreshCw size={14} className={rebuildingKbId === kb.id ? "animate-spin" : ""} />
                          {rebuildingKbId === kb.id ? 'Rebuilding...' : 'Rebuild'}
                        </span>
                      </button>
                      
                      {showConfirmRemove === kb.id ? (
                        <div className="flex items-center gap-2">
                          <button 
                            onClick={() => handleRemove(kb.id)} 
                            className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                          >
                            Confirm
                          </button>
                          <button 
                            onClick={() => setShowConfirmRemove(null)} 
                            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-lg transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button 
                          onClick={() => confirmRemove(kb.id)} 
                          className="p-1 text-gray-400 hover:text-red-600 rounded-lg transition-colors"
                        >
                          <Trash2 size={18} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                <div className="p-4 md:p-6 bg-gray-50">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-start gap-2">
                      <GitBranch size={18} className="text-gray-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-700">Repository</p>
                        {kb.git_url ? (
                          <a 
                            href={kb.git_url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
                          >
                            {kb.git_url}
                            <ExternalLink size={14} />
                          </a>
                        ) : (
                          <p className="text-sm text-gray-500">No repository URL</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-2">
                      <Clock size={18} className="text-gray-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-700">Created</p>
                        <p className="text-sm text-gray-600">{formatDate(kb.created)}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-2">
                      <div className={`w-4 h-4 rounded-full ${statusColor} mt-0.5 flex-shrink-0`}></div>
                      <div>
                        <p className="text-sm font-medium text-gray-700">Status</p>
                        <div className="flex items-center gap-2">
                          <p className="text-sm text-gray-600 capitalize">{kb.status}</p>
                          {kb.status === 'indexing' && <Loader2 size={14} className="animate-spin text-gray-500" />}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-2">
                      <Database size={18} className="text-gray-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-700">Chunks</p>
                        <p className="text-sm text-gray-600">{kb.chunkCount ?? 'N/A'}</p>
                      </div>
                    </div>
                  </div>

                  {/* Progress and Operation Status */}
                  {(kb.status === 'indexing' || kb.status === 'pending') && (
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-600">{kb.current_operation || 'Processing...'}</span>
                        <span className="text-gray-600">{kb.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${kb.progress}%` }}
                        ></div>
                      </div>

                      {/* Detailed Indexing State */}
                      {kb.indexing_details && (
                        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                          <h4 className="text-sm font-medium text-gray-700 mb-3">Indexing Details</h4>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <p className="text-xs text-gray-500">Current Phase</p>
                              <p className="text-sm font-medium text-gray-700">
                                {kb.indexing_details.phase_descriptions?.[kb.indexing_details.current_phase] || kb.indexing_details.current_phase}
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Files Processed</p>
                              <p className="text-sm font-medium text-gray-700">
                                {kb.indexing_details.processed_files} / {kb.indexing_details.total_files}
                              </p>
                            </div>
                            {kb.indexing_details.current_file && (
                              <div className="col-span-2">
                                <p className="text-xs text-gray-500">Current File</p>
                                <p className="text-sm font-medium text-gray-700 truncate">
                                  {kb.indexing_details.current_file}
                                </p>
                              </div>
                            )}
                            {kb.indexing_details.processed_files_list?.length > 0 && (
                              <div className="col-span-2">
                                <p className="text-xs text-gray-500">Recently Processed Files</p>
                                <div className="mt-1 space-y-1">
                                  {kb.indexing_details.processed_files_list.map((file, index) => (
                                    <p key={index} className="text-xs text-gray-600 truncate">
                                      {file}
                                    </p>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Error Message */}
                  {kb.status === 'error' && (
                    <div className="mt-4 p-3 bg-red-50 rounded-lg">
                      <div className="flex items-start gap-2">
                        <AlertTriangle size={18} className="text-red-500 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm font-medium text-red-700">Error</p>
                          <p className="text-sm text-red-600">{kb.error_message || 'An error occurred during processing'}</p>
                          {kb.indexing_details?.last_error && (
                            <p className="text-xs text-red-500 mt-1">{kb.indexing_details.last_error}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {kb.description && (
                    <div className="mt-4 text-sm text-gray-600">
                      <p>{kb.description}</p>
                    </div>
                  )}

                  <div className="mt-6 flex items-center gap-2 p-3 bg-blue-50 rounded-lg">
                    <Tag size={18} className="text-blue-600" />
                    <span className="font-mono font-medium text-blue-700">#{kb.name}</span>
                    <span className="text-sm text-blue-600">Use this tag in chat to reference this knowledge base</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
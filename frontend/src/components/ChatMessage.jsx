import React from 'react';
import { formatDate } from '../utils/helpers';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

export const ChatMessage = ({ message }) => {
  const { sender, content, timestamp, error } = message;

  const isUser = sender === 'user';
  const isAssistant = sender === 'assistant';

  const avatarClass = isUser
    ? 'bg-gradient-to-br from-violet-500 to-purple-600'
    : isAssistant
    ? 'bg-gradient-to-br from-indigo-500 to-blue-600'
    : 'bg-gradient-to-br from-gray-500 to-gray-700';

  const messageWrapperClass = isUser
    ? 'ml-auto max-w-3xl'
    : 'mr-auto';

  const bgClass = isUser
    ? 'bg-violet-50'
    : 'bg-white';

  const borderClass = error
    ? 'border-red-500'
    : isUser
    ? 'border-violet-400'
    : isAssistant
    ? 'border-indigo-400'
    : 'border-gray-400';

  return (
    <div className={`w-full flex ${messageWrapperClass} `}>
      <div className={`flex items-start gap-3 w-full py-3 px-2 rounded-2xl shadow-md border-l-4 ${bgClass} ${borderClass}`}>
        <div className={`h-9 w-9 rounded-full ${avatarClass} flex items-center justify-center text-white font-semibold shadow-md`}>
          {isUser ? 'U' : isAssistant ? 'A' : 'S'}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className="font-semibold text-sm">
              {isUser ? 'You' : isAssistant ? 'Assistant' : 'System'}
            </span>
            <span className="text-xs text-gray-500">
              {formatDate(timestamp)}
            </span>
          </div>

          <div className={`p-4 rounded-xl border ${borderClass} ${bgClass} text-sm leading-relaxed`}>
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={tomorrow}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-md mt-3 mb-3"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className="px-1 py-0.5 rounded bg-gray-200 font-mono text-sm" {...props}>
                      {children}
                    </code>
                  );
                },
                p: ({ children, ...props }) => <p className="mb-3 text-base" {...props}>{children}</p>,
                h1: ({ children }) => <h1 className="text-2xl font-bold mt-5 mb-3">{children}</h1>,
                h2: ({ children }) => <h2 className="text-xl font-bold mt-4 mb-2">{children}</h2>,
                h3: ({ children }) => <h3 className="text-lg font-semibold mt-3 mb-1">{children}</h3>,
                ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
                li: ({ children }) => <li>{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-600 my-3">
                    {children}
                  </blockquote>
                ),
                a: ({ href, children }) => (
                  <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto my-3">
                    <table className="min-w-full divide-y divide-gray-300 text-sm">{children}</table>
                  </div>
                ),
                thead: ({ children }) => <thead className="bg-gray-100">{children}</thead>,
                tbody: ({ children }) => <tbody className="divide-y divide-gray-200">{children}</tbody>,
                tr: ({ children }) => <tr>{children}</tr>,
                th: ({ children }) => (
                  <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide text-xs">{children}</th>
                ),
                td: ({ children }) => <td className="px-3 py-2 whitespace-nowrap">{children}</td>,
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
};

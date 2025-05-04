/**
 * Format a date string into a readable format
 * 
 * @param {string} dateString - ISO date string
 * @param {boolean} includeTime - Whether to include time in the output
 * @returns {string} - Formatted date string
 */
export const formatDate = (dateString, includeTime = false) => {
    if (!dateString) return 'Unknown';
    
    const date = new Date(dateString);
    
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }
    
    const options = {
      year: 'numeric', 
      month: 'short', 
      day: 'numeric'
    };
    
    if (includeTime) {
      options.hour = '2-digit';
      options.minute = '2-digit';
    }
    
    return date.toLocaleDateString(undefined, options);
  };
  
  /**
   * Calculate time elapsed since a date
   * 
   * @param {string} dateString - ISO date string
   * @returns {string} - Human readable time elapsed
   */
  export const timeAgo = (dateString) => {
    if (!dateString) return 'Unknown';
    
    const date = new Date(dateString);
    
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }
    
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);
    
    // Less than a minute
    if (seconds < 60) {
      return 'just now';
    }
    
    // Less than an hour
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
      return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    }
    
    // Less than a day
    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
      return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    }
    
    // Less than a month
    const days = Math.floor(hours / 24);
    if (days < 30) {
      return `${days} day${days !== 1 ? 's' : ''} ago`;
    }
    
    // Less than a year
    const months = Math.floor(days / 30);
    if (months < 12) {
      return `${months} month${months !== 1 ? 's' : ''} ago`;
    }
    
    // More than a year
    const years = Math.floor(months / 12);
    return `${years} year${years !== 1 ? 's' : ''} ago`;
  };
  
  /**
   * Truncate text to a specified length with ellipsis
   * 
   * @param {string} text - Input text to truncate
   * @param {number} maxLength - Maximum length before truncation
   * @returns {string} - Truncated text
   */
  export const truncateText = (text, maxLength = 100) => {
    if (!text || text.length <= maxLength) {
      return text;
    }
    
    return text.substring(0, maxLength) + '...';
  };
  
  /**
   * Generate a random color based on a string (e.g., for user avatars)
   * 
   * @param {string} str - Input string to generate color from
   * @returns {string} - CSS hex color value
   */
  export const stringToColor = (str) => {
    if (!str) return '#5a5a5a';
    
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    let color = '#';
    for (let i = 0; i < 3; i++) {
      const value = (hash >> (i * 8)) & 0xFF;
      color += ('00' + value.toString(16)).substr(-2);
    }
    
    return color;
  };
  
  /**
   * Extract first letter of each word in a string (for initials)
   * 
   * @param {string} name - Full name
   * @param {number} count - Number of initials to extract
   * @returns {string} - Initials
   */
  export const getInitials = (name, count = 2) => {
    if (!name) return '?';
    
    return name
      .split(' ')
      .map(part => part.charAt(0).toUpperCase())
      .filter(char => char.match(/[A-Z]/))
      .slice(0, count)
      .join('');
  };
  
  /**
   * Format file size in bytes to human-readable format
   * 
   * @param {number} bytes - File size in bytes
   * @param {number} decimals - Number of decimal places
   * @returns {string} - Formatted file size
   */
  export const formatFileSize = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
  };
  
  /**
   * Format code snippet with line numbers
   * 
   * @param {string} code - The code snippet
   * @param {number} startLine - Starting line number
   * @returns {string} - Formatted code with line numbers
   */
  export const formatCodeWithLineNumbers = (code, startLine = 1) => {
    if (!code) return '';
    
    const lines = code.split('\n');
    const maxLineNumberWidth = String(startLine + lines.length - 1).length;
    
    return lines
      .map((line, index) => {
        const lineNumber = String(startLine + index).padStart(maxLineNumberWidth, ' ');
        return `${lineNumber} | ${line}`;
      })
      .join('\n');
  };
  
  /**
   * Detect programming language from file extension
   * 
   * @param {string} fileName - Name of the file
   * @returns {string} - Programming language name
   */
  export const detectLanguage = (fileName) => {
    if (!fileName) return 'text';
    
    const extension = fileName.split('.').pop().toLowerCase();
    
    const languageMap = {
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'py': 'python',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'cs': 'csharp',
      'go': 'go',
      'rb': 'ruby',
      'php': 'php',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'json': 'json',
      'md': 'markdown',
      'sql': 'sql',
      'sh': 'bash',
      'yaml': 'yaml',
      'yml': 'yaml',
      'xml': 'xml',
      'swift': 'swift',
      'kt': 'kotlin',
      'rs': 'rust'
    };
    
    return languageMap[extension] || 'text';
  };
  
  /**
   * Parse markdown content to HTML
   * 
   * Simple markdown parser for basic formatting
   * In a real app, use a proper markdown library
   * 
   * @param {string} markdown - Markdown string
   * @returns {string} - HTML string
   */
  export const parseMarkdown = (markdown) => {
    if (!markdown) return '';
    
    // Replace code blocks
    let html = markdown.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Replace inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Replace headers
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Replace bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Replace links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Replace unordered lists
    html = html.replace(/^\s*-\s+(.*$)/gm, '<li>$1</li>');
    html = html.replace(/<li>(.*)<\/li>/g, '<ul><li>$1</li></ul>');
    
    // Replace paragraphs (must be last)
    html = html.replace(/^(?!<[a-z])/gm, '<p>');
    html = html.replace(/^<p>(.*)<\/[a-z0-9]+>$/gm, '<p>$1</p>');
    html = html.replace(/^<p>(.*)$/gm, '<p>$1</p>');
    
    return html;
  };
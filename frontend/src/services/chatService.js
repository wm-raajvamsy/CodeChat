// services/chatService.js
import axios from 'axios';
import { handleApiError, reportError } from './errorHandlingService';
import { searchCombineKnowledgeBase, searchKnowledgeBase } from './knowledgeBaseService';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:6146/api';
const OLLAMA_URL = process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434/api';
const OPENAI_URL = 'https://api.openai.com/v1/chat/completions';
// const OPENAI_API_KEY = process.env.OPENAI_API_KEY || localStorage.getItem('openaiKey');

/**
 * Send a user message (with optional context) to OpenAI and return the assistant's reply.
 * @param {string} userMessage
 * @param {string[]} contextSnippets
 * @returns {Promise<string>}
 */
export async function callOpenAI(userMessage, openaiKey, contextSnippets = []) {
  // Construct the prompt/messages based on whether we have context
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    // inject any prior context snippets as system messages, if you like:
    // ...contextSnippets.map((snippet) => ({ role: 'system', content: snippet })),
    { role: 'user', content: userMessage }
  ];

  console.log(`Sending prompt to OpenAI (model: gpt-4)...`);

  const response = await axios.post(
    OPENAI_URL,
    {
      model: 'gpt-4.1-nano-2025-04-14',
      messages: messages,
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 1024,
      stream: false
    },
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${openaiKey}`
      }
    }
  );

  // Basic validation of the OpenAI response
  if (
    !response.data ||
    !response.data.choices ||
    !response.data.choices.length ||
    !response.data.choices[0].message ||
    !response.data.choices[0].message.content
  ) {
    throw new Error('Invalid response from OpenAI API');
  }

  return response.data.choices[0]?.message?.content;
}


/**
 * Process a chat message using the knowledge base and Ollama
 * For non-streaming responses
 * 
 * @param {string} userMessage - The user's message
 * @param {Array} knowledgeBases - Array of knowledge bases to use for context
 * @param {Object} config - Configuration for the chat (temperature, model, etc.)
 * @returns {Promise<string>} - The assistant's response
 */
export const processChat = async (userMessage, knowledgeBases = [], config = {}) => {
  try {
    // Default configuration
    const defaultConfig = {
      temperature: 0.7,
      topK: 5,
      maxTokens: 1000,
      model: 'qwen2.5:14b-instruct',
    };

    // Merge with user config
    const finalConfig = { ...defaultConfig, ...config };

    // Format the prompt based on available knowledge bases
    let contextSnippets = [];

    // For each selected knowledge base, fetch relevant snippets
    if (knowledgeBases && knowledgeBases.length > 0) {
      console.log(`Retrieving context from ${knowledgeBases.length} knowledge bases...`);
      if (finalConfig.combineSearch) {
        const kb_ids = knowledgeBases.map(kb => kb.id);
        const names = knowledgeBases.map(kb => kb.name).join(",");
        // Search knowledge base for relevant snippets
        const snippets = await searchCombineKnowledgeBase(kb_ids, userMessage, finalConfig.topK);

          if (snippets && snippets.length > 0) {
            console.log(`Found ${snippets.length} relevant snippets from ${names}`);
            contextSnippets.push({
              kbName: names,
              snippets: snippets
            });
          }
      }
      else {
        for (const kb of knowledgeBases) {
          // Search knowledge base for relevant snippets
          const snippets = await searchKnowledgeBase(kb.id, userMessage, finalConfig.topK);

          if (snippets && snippets.length > 0) {
            console.log(`Found ${snippets.length} relevant snippets from ${kb.name}`);
            contextSnippets.push({
              kbName: kb.name,
              snippets: snippets
            });
          }
        }
      }
    }

    // Construct the prompt based on whether we have context
    const prompt = constructPrompt(userMessage, contextSnippets);
    if (finalConfig.model == "openai") {
      return callOpenAI(prompt, finalConfig.openaiKey);
    }
    // Call Ollama API with auth token
    console.log(`Sending prompt to ${finalConfig.model}...`);
    const response = await axios.post(`${OLLAMA_URL}/generate`, {
      model: finalConfig.model,
      prompt: prompt,
      stream: false,
      options: {
        temperature: finalConfig.temperature,
        top_p: 0.9,
        top_k: 40,
        max_tokens: finalConfig.maxTokens,
      }
    }, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      }
    });

    if (!response.data || !response.data.response) {
      throw new Error('Invalid response from Ollama API');
    }

    return response.data.response;
  } catch (error) {
    throw handleApiError(error, 'Failed to process chat message');
  }
};

/**
 * Process a chat message with streaming response
 * 
 * @param {string} userMessage - The user's message
 * @param {Array} knowledgeBases - Array of knowledge bases to use for context
 * @param {Object} config - Configuration for the chat
 * @param {function} onTokenCallback - Function to call for each new token
 * @param {function} onCompleteCallback - Function to call when stream is complete
 * @param {function} onErrorCallback - Function to call when an error occurs
 * @returns {AbortController} - Controller that can be used to abort the stream
 */
export const processChatStreaming = async (
  userMessage, 
  knowledgeBases = [], 
  config = {}, 
  onTokenCallback, 
  onCompleteCallback, 
  onErrorCallback
) => {
  // Create abort controller for cancellation
  const abortController = new AbortController();
  
  try {
    // Default configuration
    const defaultConfig = {
      temperature: 0.7,
      topK: 5,
      maxTokens: 1000,
      model: 'qwen2.5:14b-instruct',
    };

    // Merge with user config
    const finalConfig = { ...defaultConfig, ...config };

    // Format the prompt based on available knowledge bases
    let contextSnippets = [];

    // For each selected knowledge base, fetch relevant snippets
    if (knowledgeBases && knowledgeBases.length > 0) {
      console.log(`Retrieving context from ${knowledgeBases.length} knowledge bases...`);
      if (finalConfig.combineSearch) {
        const kb_ids = knowledgeBases.map(kb => kb.id);
        const names = knowledgeBases.map(kb => kb.name).join(",");
        // Search knowledge base for relevant snippets
        const snippets = await searchCombineKnowledgeBase(kb_ids, userMessage, finalConfig.topK);

          if (snippets && snippets.length > 0) {
            console.log(`Found ${snippets.length} relevant snippets from ${names}`);
            contextSnippets.push({
              kbName: names,
              snippets: snippets
            });
          }
      }
      else {
        for (const kb of knowledgeBases) {
          try {
            // Search knowledge base for relevant snippets
            const snippets = await searchKnowledgeBase(kb.id, userMessage, finalConfig.topK);

            if (snippets && snippets.length > 0) {
              console.log(`Found ${snippets.length} relevant snippets from ${kb.name}`);
              contextSnippets.push({
                kbName: kb.name,
                snippets: snippets
              });
            }
          } catch (error) {
            console.error(`Error retrieving snippets from KB ${kb.name}:`, error);
            // Continue with other knowledge bases if one fails
          }
        }
      }
    }

    // Construct the prompt based on whether we have context
    const prompt = constructPrompt(userMessage, contextSnippets);
    let response;
    // --- OPENAI STREAMING PATH ---
    if (finalConfig.model === 'openai') {
      response = await fetch(OPENAI_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${finalConfig.openaiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4.1-nano-2025-04-14',
          messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: prompt }
          ],
          stream: true,
          temperature: finalConfig.temperature,
          top_p: 0.9,
          max_tokens: finalConfig.maxTokens
        }),
        signal: abortController.signal
      });

    }
    else {
      // Start streaming request
      response = await fetch(`${OLLAMA_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: JSON.stringify({
          model: finalConfig.model,
          prompt: prompt,
          stream: true,
          options: {
            temperature: finalConfig.temperature,
            top_p: 0.9,
            top_k: 40,
            max_tokens: finalConfig.maxTokens,
          }
        }),
        signal: abortController.signal
      });
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('ReadableStream not supported in this browser.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let fullResponse = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Decode the stream chunk
        const chunk = decoder.decode(value, { stream: true });
        
        try {
          // Ollama sends newline-delimited JSON objects
          const lines = chunk.replaceAll("data: ","").split('\n').filter(line => !line.includes("[DONE]") && line.trim());
          for (const line of lines) {
            const parsedChunk = JSON.parse(line);
            let token = "";
            if(finalConfig.model === "openai" && (!response.data ||
              !response.data.choices ||
              !response.data.choices.length ||
              !response.data.choices[0].message ||
              !response.data.choices[0].message.content)){
              token = parsedChunk.choices[0]?.delta?.content
            }
            if (finalConfig.model !== "openai" && parsedChunk.response) {
              token = parsedChunk.response;
            }
            if(token && token != ""){
              fullResponse += token;
              
              // Call the token callback
              if (onTokenCallback && typeof onTokenCallback === 'function') {
                onTokenCallback(token);
              }
            }
          }
        } catch (parseError) {
          console.error('Error parsing streaming chunk:', parseError);
          // Continue processing despite parse errors
        }
      }
      
      // Stream completed successfully
      if (onCompleteCallback && typeof onCompleteCallback === 'function') {
        onCompleteCallback(fullResponse);
      }
      
      return fullResponse;
    } catch (streamError) {
      // Check if this was an abort error
      if (streamError.name === 'AbortError') {
        console.log('Stream was aborted');
      } else {
        if (onErrorCallback && typeof onErrorCallback === 'function') {
          onErrorCallback(streamError);
        }
        throw streamError;
      }
    }
  } catch (error) {
    const enhancedError = handleApiError(error, 'Failed to process streaming chat');
    
    if (onErrorCallback && typeof onErrorCallback === 'function') {
      onErrorCallback(enhancedError);
    }
    
    reportError(enhancedError, { messageType: 'streaming', userMessageLength: userMessage.length });
    throw enhancedError;
  }
  
  return abortController;
};

/**
 * Construct prompt with code context for the LLM
 * 
 * @param {string} userMessage - The user's message
 * @param {Array} contextData - Array of knowledge base context data
 * @returns {string} - Formatted prompt for the LLM
 */
export const constructPrompt = (userMessage, contextData = []) => {
  if (contextData.length === 0) {
    // No knowledge base context available
    return `You are a helpful AI assistant specializing in programming and software development. 
Please answer the following question:

${userMessage}`;
  }

  // Build context from knowledge base snippets
  let contextBlocks = [];

  contextData.forEach(kb => {
    const snippetsText = kb.snippets.map((snippet, idx) => {
      let text = `[SNIPPET ${idx + 1}] ${snippet.file_path}`;

      if (snippet.chunk_type && snippet.name) {
        text += ` (${snippet.chunk_type} ${snippet.name})`;
      }

      if (snippet.metadata && snippet.metadata.dependencies && snippet.metadata.dependencies.length > 0) {
        text += `\nDependencies: ${snippet.metadata.dependencies.join(', ')}`;
      }

      text += `\n\n${snippet.content}\n`;
      return text;
    }).join('\n\n');

    contextBlocks.push(`--- FROM KNOWLEDGE BASE: ${kb.kbName} ---\n${snippetsText}`);
  });

  return `You are a helpful programming assistant that answers code questions based on repository snippets.

CODE CONTEXT:
${contextBlocks.join('\n\n')}

Based on the above code snippets from the repository, please answer the following question:
${userMessage}

Provide a clear and concise answer, referencing specific parts of the code when needed.
If the answer requires code examples, always format code blocks properly using markdown syntax.
If you're unsure about something, acknowledge the uncertainty rather than making assumptions.`;
};

/**
 * Cancel an ongoing stream
 * 
 * @param {AbortController} controller - The abort controller to cancel
 */
export const cancelStream = (controller) => {
  if (controller && typeof controller.abort === 'function') {
    controller.abort();
  }
};
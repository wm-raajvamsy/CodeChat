// services/authService.js
import axios from 'axios';
import { handleApiError } from './errorHandlingService';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:6146/api';

/**
 * Register a new user account
 * 
 * @param {Object} userData - User data including name, email, password
 * @returns {Promise<Object>} - User data and token
 */
export const register = async (userData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/register`, userData);
    
    // Store auth token
    if (response.data.token) {
      localStorage.setItem('authToken', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to register account');
  }
};

/**
 * Login a user
 * 
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise<Object>} - User data and token
 */
export const login = async (email, password) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/login`, { email, password });
    
    // Store auth token
    if (response.data.token) {
      localStorage.setItem('authToken', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to login');
  }
};

/**
 * Login with Google OAuth
 * 
 * @param {string} idToken - Google ID token
 * @returns {Promise<Object>} - User data and token
 */
export const loginWithGoogle = async (idToken) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/google`, { idToken });
    
    // Store auth token
    if (response.data.token) {
      localStorage.setItem('authToken', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to login with Google');
  }
};

/**
 * Login with GitHub OAuth
 * 
 * @param {string} code - GitHub authorization code
 * @returns {Promise<Object>} - User data and token
 */
export const loginWithGitHub = async (code) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/github`, { code });
    
    // Store auth token
    if (response.data.token) {
      localStorage.setItem('authToken', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to login with GitHub');
  }
};

/**
 * Logout the current user
 */
export const logout = () => {
  localStorage.removeItem('authToken');
  localStorage.removeItem('user');
  
  // Optional: Call logout endpoint to invalidate token on server
  axios.post(`${API_BASE_URL}/auth/logout`, {}, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    }
  }).catch(error => {
    console.warn('Error during logout:', error);
  });
};

/**
 * Check if the user is authenticated
 * 
 * @returns {boolean} - True if authenticated
 */
export const isAuthenticated = () => {
  return localStorage.getItem('authToken') !== null;
};

/**
 * Get current user data
 * 
 * @returns {Object|null} - User data or null if not authenticated
 */
export const getCurrentUser = () => {
  const userJson = localStorage.getItem('user');
  return userJson ? JSON.parse(userJson) : null;
};

/**
 * Check if authentication token is valid
 * 
 * @returns {Promise<boolean>} - True if token is valid
 */
export const validateToken = async () => {
  try {
    const token = localStorage.getItem('authToken');
    if (!token) {
      return false;
    }
    
    const response = await axios.get(`${API_BASE_URL}/auth/validate`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    return response.data.valid === true;
  } catch (error) {
    console.warn('Token validation failed:', error);
    
    // Clear invalid tokens
    if (error.response && (error.response.status === 401 || error.response.status === 403)) {
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
    }
    
    return false;
  }
};

/**
 * Reset password request
 * 
 * @param {string} email - User email
 * @returns {Promise<Object>} - Response status
 */
export const requestPasswordReset = async (email) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/reset-password`, { email });
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to request password reset');
  }
};

/**
 * Set new password with reset token
 * 
 * @param {string} token - Password reset token
 * @param {string} newPassword - New password
 * @returns {Promise<Object>} - Response status
 */
export const resetPassword = async (token, newPassword) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/auth/reset-password/confirm`, {
      token,
      password: newPassword
    });
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to reset password');
  }
};

/**
 * Update user profile
 * 
 * @param {Object} userData - User data to update
 * @returns {Promise<Object>} - Updated user data
 */
export const updateProfile = async (userData) => {
  try {
    const response = await axios.put(
      `${API_BASE_URL}/auth/profile`,
      userData,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    
    // Update stored user data
    localStorage.setItem('user', JSON.stringify(response.data.user));
    
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to update profile');
  }
};

/**
 * Change user password
 * 
 * @param {string} currentPassword - Current password
 * @param {string} newPassword - New password
 * @returns {Promise<Object>} - Response status
 */
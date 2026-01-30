
// Central Configuration for API URL
// Priority:
// 1. Environment Variable (VITE_API_URL)
// 2. Hardcoded Fallback (for local development without .env or fallback in production)

const API_URL = import.meta.env.VITE_API_URL || 'https://learnify-api-ohc0.onrender.com';

export default API_URL;

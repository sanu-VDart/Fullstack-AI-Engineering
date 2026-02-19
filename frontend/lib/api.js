/**
 * API Service Layer for Turbofan Engine Predictive Maintenance
 * Connects Next.js frontend to FastAPI backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Fetch API health status
 */
export async function fetchHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
}

/**
 * Fetch list of all engine units
 */
export async function fetchEngines() {
    const res = await fetch(`${API_BASE}/engines`);
    if (!res.ok) throw new Error('Failed to fetch engines');
    return res.json();
}

/**
 * Predict RUL for a specific engine unit
 * @param {number} unitId - Engine unit ID
 */
export async function predictRUL(unitId) {
    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unit_id: unitId })
    });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Prediction failed');
    }
    return res.json();
}

/**
 * Get maintenance recommendations for an engine unit
 * @param {number} unitId - Engine unit ID
 */
export async function getMaintenanceRecommendation(unitId) {
    const res = await fetch(`${API_BASE}/maintenance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unit_id: unitId })
    });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to get recommendations');
    }
    return res.json();
}

/**
 * Chat with the AI assistant
 * @param {string} message - User message
 * @param {string|null} conversationId - Optional conversation ID for context
 */
export async function chat(message, conversationId = null) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            conversation_id: conversationId
        })
    });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Chat request failed');
    }
    return res.json();
}

/**
 * Analyze sensor data for an engine unit
 * @param {number} unitId - Engine unit ID
 * @param {number|null} cycle - Optional specific cycle to analyze
 */
export async function analyzeSensors(unitId, cycle = null) {
    const body = { unit_id: unitId };
    if (cycle) body.cycle = cycle;

    const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Analysis failed');
    }
    return res.json();
}

/**
 * Trigger model training
 * @param {string} datasetId - Dataset ID (default: FD001)
 */
export async function triggerTraining(datasetId = 'FD001') {
    const res = await fetch(`${API_BASE}/train?dataset_id=${datasetId}`, {
        method: 'POST'
    });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Training failed');
    }
    return res.json();
}

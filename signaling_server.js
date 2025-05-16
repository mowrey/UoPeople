// signaling_server.js (Group Chat + Link Blocking + AI Chat + Inactivity + Comment Generation API)
const WebSocket = require('ws');
const http = require('http'); // Import http module
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");
const { v4: uuidv4 } = require('uuid');

// --- Configuration ---
const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MAX_HISTORY_LENGTH = 100000; // Note: This is a high number of messages. Consider token limits for very long conversations.
const AI_MODEL_NAME = "gemini-1.5-flash-latest";
const INACTIVITY_TIMEOUT = 60 * 60 * 1000;
const ACTIVITY_CHECK_INTERVAL = 30 * 1000;
const LINK_REPLACEMENT = "****";
const COMMENT_API_ENDPOINT = "/api/generate-comment"; // Define API path

// --- Validate API Key ---
if (!GEMINI_API_KEY) { console.error("FATAL ERROR: GEMINI_API_KEY environment variable is not set."); process.exit(1); }

// --- Initialize AI Client ---
let genAI, aiModel;
try {
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    aiModel = genAI.getGenerativeModel({ model: AI_MODEL_NAME });
    console.log(`Initialized Google AI: ${AI_MODEL_NAME}`);
} catch (error) { console.error("FATAL ERROR: AI client init failed.", error); process.exit(1); }

const generationSafetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

// --- Server State ---
let peers = {};
let rooms = {};
let roomHistories = {};

// --- Helper Functions (Keep existing ones) ---
function generateId() { return Math.random().toString(36).substring(2, 10); }
function generateMessageId() { return uuidv4(); } // Ensure uuid is installed: npm install uuid
function safeSend(ws, data) { try { if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(data)); return true; } else { return false; } } catch (error) { console.error("safeSend Error:", error); return false; } }
function broadcastToRoom(topic, messageData, senderId) { if (!rooms[topic]) return; rooms[topic].forEach(peerId => { if (peerId !== senderId && peers[peerId]) { safeSend(peers[peerId].ws, messageData); } }); }
function notifyRoomUpdate(topic, reason = "update") { if (!rooms[topic]) return; const currentPeers = Array.from(rooms[topic]); let genericReason = reason; if (reason.startsWith('peer_joined:')) genericReason = 'peer_joined'; if (reason.startsWith('peer_left:')) genericReason = 'peer_left'; const updateMessage = { type: 'room_update', topic: topic, peers: currentPeers, reason: genericReason }; currentPeers.forEach(peerId => { if (peers[peerId]) { safeSend(peers[peerId].ws, updateMessage); } }); console.log(`Room update: ${topic}. Reason: ${genericReason}. Peers: ${currentPeers.length}`); }
function cleanupPeer(peerId, reasonCode = 1000, reasonMsg = "Cleanup called") { console.log(`Cleaning up peer: ${peerId}. Reason: ${reasonMsg}`); const peerData = peers[peerId]; if (!peerData) { return; } const topic = peerData.topic; if (topic && rooms[topic]) { const wasInRoom = rooms[topic].delete(peerId); if (wasInRoom) { console.log(`Removed ${peerId} from room ${topic}. Size: ${rooms[topic].size}`); if (rooms[topic].size === 0) { console.log(`Deleting empty room: ${topic}`); delete rooms[topic]; delete roomHistories[topic]; } else { notifyRoomUpdate(topic, `peer_left`); } } } delete peers[peerId]; console.log(`Peer ${peerId} removed. Total: ${Object.keys(peers).length}.`); if (peerData.ws && peerData.ws.readyState !== WebSocket.CLOSED && peerData.ws.readyState !== WebSocket.CLOSING) { try { peerData.ws.close(reasonCode, reasonMsg); } catch (e) { /* ignore close errors */ } } }
function blockLinks(text) { if (!text) return ""; const urlRegex = /(?:https?:\/\/|www\.)[^\s/$.?#].[^\s]*|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(\/[^\s]*)?/gi; const blockedText = text.replace(urlRegex, LINK_REPLACEMENT); if (blockedText !== text) { console.log(`Blocked link(s) in message.`); } return blockedText; }

// --- NEW AI Function for Single Comment Generation (for API endpoint) ---
async function generateSingleComment(postContext) {
    if (!postContext || postContext.trim() === "") {
        return "[No context provided for comment generation.]";
    }
    console.log(`Requesting AI comment generation for context: "${postContext.substring(0, 50)}..."`);

    const prompt = `Write a short, realistic, and relevant comment (like one you'd see on a blog or social media) reacting to the following post content snippet: "${postContext}". Keep the comment under 25 words. Be supportive, curious, or offer a brief related thought. Do not use hashtags. Do not introduce yourself. Do not ask a question.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.8, topP: 0.95, maxOutputTokens: 60 },
            safetySettings: generationSafetySettings
        });

        if (result.response) {
            const text = result.response.text()?.trim();
            if (!text) {
                console.warn("AI Gen Comment: Empty text received.");
                return "[AI had no comment.]";
            }
            console.log(`AI Comment Gen Success: ${text.substring(0, 80)}...`);
            return text;
        } else {
            const blockReason = result?.response?.promptFeedback?.blockReason || result?.response?.candidates?.[0]?.finishReason;
            console.error(`AI Comment Gen Err: Blocked? ${blockReason}`);
            return `[AI response blocked. Reason: ${blockReason || 'Filter'}.]`;
        }
    } catch (error) {
        console.error(`AI Comment Gen API error:`, error);
        return `[AI encountered an error generating comment.]`;
    }
}

// --- AI Function for Chat Response (Implemented) ---
async function getAIResponseForChat(topic) {
    if (!topic || !roomHistories[topic] || roomHistories[topic].length === 0) {
        console.warn(`AI Chat: No history or invalid topic for ${topic}`);
        return "[AI cannot respond without chat context. Please send a message first.]";
    }

    console.log(`Requesting AI chat response for topic: "${topic}" with ${roomHistories[topic].length} history entries.`);

    // Prepare the history for the AI.
    // Roles should be 'user' (for human messages) and 'model' (for AI messages).
    const currentChatHistory = roomHistories[topic].map(entry => ({
        role: entry.role, // This should be "user" or "model"
        parts: entry.parts  // This should be [{ text: "message content" }]
    }));

    // System instruction to guide the AI's persona and behavior in the chat.
    const systemInstruction = {
        role: "user", // Using "user" role for system instructions if model role isn't specifically for system prompts
        parts: [{ text: "You are an AI assistant named 'Stimuli Network AI' participating in this group chat. Your goal is to be helpful, engaging, and contribute meaningfully to the conversation based on the preceding messages. Keep your responses relatively concise and conversational. Do not use markdown formatting in your responses." }]
    };
    
    const promptContents = [
        systemInstruction,
        ...currentChatHistory
    ];

    try {
        const result = await aiModel.generateContent({
            contents: promptContents,
            generationConfig: {
                temperature: 0.7,       // A balanced temperature for chat
                topP: 0.95,
                maxOutputTokens: 300,   // Max length for a chat response
            },
            safetySettings: generationSafetySettings
        });

        if (result.response) {
            const text = result.response.text()?.trim();
            if (!text) {
                console.warn(`AI Chat Gen (${topic}): Empty text received from AI.`);
                return "[AI had no specific comment this time.]";
            }
            console.log(`AI Chat Gen Success (${topic}): ${text.substring(0, 80)}...`);
            return text;
        } else {
            const blockReason = result?.response?.promptFeedback?.blockReason || result?.response?.candidates?.[0]?.finishReason;
            console.error(`AI Chat Gen Err (${topic}): Response was problematic. Block Reason: ${blockReason}`);
            let userMessage = "[AI response was filtered";
            if (blockReason && blockReason !== "OTHER" && blockReason !== "SAFETY") { // Avoid overly generic reasons
                 userMessage += ` (Reason: ${blockReason})`;
            } else if (blockReason === "SAFETY") {
                 userMessage += ` due to safety settings`;
            }
            userMessage += ".]";
            return userMessage;
        }
    } catch (error) {
        console.error(`AI Chat Gen API error for topic (${topic}):`, error);
        return `[AI encountered an error processing the request. Please try again later.]`;
    }
}


// --- Create HTTP Server ---
const server = http.createServer(async (req, res) => {
    // --- CORS Handling ---
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204); res.end(); return;
    }

    // --- API Endpoint Handling for Comment Generation ---
    if (req.method === 'POST' && req.url === COMMENT_API_ENDPOINT) {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
            try {
                const requestData = JSON.parse(body);
                const postContext = requestData.context;

                if (!postContext) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Missing "context" in request body.' }));
                    return;
                }
                const generatedComment = await generateSingleComment(postContext);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ comment: generatedComment }));
            } catch (error) {
                console.error("Error processing API request for comment generation:", error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Internal server error processing request.' }));
            }
        });
    }
    // --- Default Handling for other HTTP requests ---
    else {
        if (req.url !== '/favicon.ico') {
             console.log(`Ignoring HTTP ${req.method} request for ${req.url}`);
        }
        res.writeHead(404);
        res.end();
    }
});

// --- WebSocket Server Setup (Attached to HTTP Server) ---
const wss = new WebSocket.Server({ noServer: true });

server.on('upgrade', (request, socket, head) => {
    console.log('Handling WebSocket upgrade request...');
    wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
    });
});


// --- WebSocket Connection Handling ---
wss.on('connection', (ws) => {
    const peerId = generateId();
    peers[peerId] = { ws: ws, topic: null, isAlive: true, lastActivity: Date.now() };
    console.log(`Peer connected via WebSocket: ${peerId}`);
    safeSend(ws, { type: 'your_id', id: peerId });

    ws.isAlive = true; // Used for heartbeat
    ws.on('pong', () => { if(peers[peerId]) peers[peerId].isAlive = true; });

    ws.on('message', async (message) => {
        let data;
        try { data = JSON.parse(message.toString()); }
        catch (e) { console.error(`Msg parse error from ${peerId}:`, e); return; }

        if (!peers[peerId]) { console.warn(`Message from already cleaned up peer ${peerId}. Ignoring.`); return; }

        peers[peerId].isAlive = true;
        peers[peerId].lastActivity = Date.now();

        const currentTopic = peers[peerId]?.topic;

        switch (data.type) {
            case 'join':
                const newTopic = data.topic?.trim();
                if (!newTopic) { safeSend(ws, {type: 'error', message: 'Topic cannot be empty.'}); return; }
                const oldTopic = peers[peerId]?.topic;
                if (oldTopic && oldTopic !== newTopic && rooms[oldTopic]) {
                    if (rooms[oldTopic].delete(peerId)) {
                        if (rooms[oldTopic].size === 0) { delete rooms[oldTopic]; delete roomHistories[oldTopic]; }
                        else { notifyRoomUpdate(oldTopic, `peer_left`); }
                    }
                }
                peers[peerId].topic = newTopic; // Assign topic to peer
                if (!rooms[newTopic]) { rooms[newTopic] = new Set(); roomHistories[newTopic] = []; console.log(`Creating room: ${newTopic}`); }
                rooms[newTopic].add(peerId);
                console.log(`Peer ${peerId} joined room ${newTopic}. Size: ${rooms[newTopic].size}`);
                notifyRoomUpdate(newTopic, `peer_joined:${peerId}`);
                // Send chat history to newly joined peer (optional, can be large)
                // if (roomHistories[newTopic] && roomHistories[newTopic].length > 0) {
                // safeSend(ws, { type: 'chat_history', topic: newTopic, history: roomHistories[newTopic] });
                // }
                break;

            case 'chat_message':
                if (!currentTopic || !rooms[currentTopic] || !data.message) return;
                let receivedChatMsg = data.message.substring(0, 1500); // Limit message length
                const messageWithBlockedLinks = blockLinks(receivedChatMsg);
                const messageId = generateMessageId();
                const historyEntry = { msgId: messageId, senderId: peerId, role: "user", parts: [{ text: messageWithBlockedLinks }], timestamp: Date.now() };
                
                if (!roomHistories[currentTopic]) roomHistories[currentTopic] = [];
                roomHistories[currentTopic].push(historyEntry);
                while (roomHistories[currentTopic].length > MAX_HISTORY_LENGTH) roomHistories[currentTopic].shift();
                
                const messagePayload = { type: 'chat_message', topic: currentTopic, messageId: messageId, senderId: peerId, message: messageWithBlockedLinks };
                broadcastToRoom(currentTopic, messagePayload, peerId); // Broadcast to others
                // Do NOT send the message back to the sender here, client-side already displays it.
                break;

            case 'ask_ai':
                if (!currentTopic || !rooms[currentTopic]) { safeSend(ws, { type: 'error', message: 'Must be in a room to ask AI.' }); return; }
                console.log(`AI request received for topic: ${currentTopic} from peer ${peerId}`);
                
                // Notify room that AI is thinking
                broadcastToRoom(currentTopic, { type: 'system_message', topic: currentTopic, message: `AI is thinking...` }, null);
                
                const aiChatResponse = await getAIResponseForChat(currentTopic);
                
                // Add AI's response to history
                if (aiChatResponse && roomHistories[currentTopic]) {
                    // We check if it's not an error/placeholder before adding to history as "model"
                    const isActualAIReply = !aiChatResponse.startsWith("[AI");
                    if (isActualAIReply) {
                        const aiHistoryEntry = {
                            msgId: generateMessageId(),
                            senderId: 'AI', // Consistent senderId for AI
                            role: "model",  // Gemini API uses "model" for AI responses
                            parts: [{ text: aiChatResponse }],
                            timestamp: Date.now()
                        };
                        roomHistories[currentTopic].push(aiHistoryEntry);
                        while (roomHistories[currentTopic].length > MAX_HISTORY_LENGTH) {
                            roomHistories[currentTopic].shift();
                        }
                    }
                }
                
                const aiMessagePayload = { type: 'ai_message', topic: currentTopic, message: aiChatResponse, senderId: 'AI' };
                broadcastToRoom(currentTopic, aiMessagePayload, null); // Broadcast AI message to everyone in the room
                break;

            case 'leave':
                console.log(`Peer ${peerId} requested leaving room ${currentTopic}.`);
                cleanupPeer(peerId, 1000, "User requested leave");
                break;

            default:
                console.log(`Unknown message type received from ${peerId}: ${data.type}`);
                safeSend(ws, { type: 'error', message: `Unknown command type: ${data.type}` });
                break;
        }
    });

    ws.on('close', (code, reason) => {
        const reasonString = reason.toString();
        console.log(`Peer WS closed: ${peerId} (Code: ${code}, Reason: ${reasonString || 'N/A'})`);
        cleanupPeer(peerId, code, `WebSocket closed: ${reasonString || 'N/A'}`);
    });

    ws.on('error', (error) => {
        console.error(`WebSocket error for peer ${peerId}:`, error);
        cleanupPeer(peerId, 1011, "WebSocket error occurred"); // 1011: Server error
    });
});


// --- Heartbeat & Inactivity Intervals (Unchanged) ---
const heartbeatInterval = setInterval(() => {
    wss.clients.forEach((ws) => {
        const peerEntry = Object.entries(peers).find(([/*id*/, data]) => data?.ws === ws);
        if (!peerEntry) { ws.terminate(); return; } // Should not happen if cleanup is correct
        const peerId = peerEntry[0]; const peerData = peerEntry[1];

        if (!peerData) { // Should also not happen
            console.warn(`Heartbeat: Peer data for WS client not found. Terminating.`);
            ws.terminate();
            return;
        }

        if (peerData.isAlive === false) {
            console.log(`Heartbeat timeout for peer ${peerId}. Cleaning up.`);
            cleanupPeer(peerId, 1001, "Heartbeat timeout"); // 1001: Going away
            return;
        }
        peerData.isAlive = false; // Expect a pong to set it back to true
        try {
            if (ws.readyState === WebSocket.OPEN) {
                ws.ping(() => {}); // Send ping
            } else {
                // If not open, it might be closing or closed, cleanup will handle or has handled
                console.log(`Heartbeat: WS for peer ${peerId} not open (state: ${ws.readyState}). Skipping ping.`);
            }
        } catch (e) {
            console.error(`Error pinging peer ${peerId}:`, e);
            cleanupPeer(peerId, 1011, "Heartbeat ping error");
        }
    });
}, 30000); // 30 seconds

const inactivityInterval = setInterval(() => {
    const now = Date.now();
    Object.keys(peers).forEach(peerId => {
        const peerData = peers[peerId];
        if (peerData && peerData.lastActivity) {
            const idleTime = now - peerData.lastActivity;
            if (idleTime > INACTIVITY_TIMEOUT) {
                console.warn(`Peer ${peerId} inactive for ${idleTime / 1000}s. Disconnecting.`);
                safeSend(peerData.ws, { type: 'disconnect_inactive', topic: peerData.topic, timeout: INACTIVITY_TIMEOUT / 1000 });
                // Give a small delay for the message to be sent before forceful close
                setTimeout(() => cleanupPeer(peerId, 1001, "Inactivity timeout"), 200);
            }
        } else if (peerData && !peerData.lastActivity) {
            // Initialize lastActivity if it's somehow missing for an active peer
            peerData.lastActivity = now;
        }
    });
}, ACTIVITY_CHECK_INTERVAL);


// --- Start the HTTP Server ---
server.listen(PORT, () => {
    console.log(`HTTP and WebSocket Server started on port ${PORT}`);
    console.log(`Comment Generation API endpoint available at POST ${COMMENT_API_ENDPOINT}`);
});

// Cleanup intervals on server shutdown
process.on('SIGINT', () => { // Handle Ctrl+C
    console.log("SIGINT received. Shutting down gracefully...");
    server.close(() => {
        console.log("HTTP server closed.");
        wss.close(() => {
            console.log("WebSocket server closed.");
            clearInterval(heartbeatInterval);
            clearInterval(inactivityInterval);
            console.log("Intervals cleared. Exiting.");
            process.exit(0);
        });
    });
});

server.on('close', () => { // This event is for the HTTP server itself
    console.log("HTTP server is shutting down. Clearing intervals if not already handled by SIGINT.");
    clearInterval(heartbeatInterval);
    clearInterval(inactivityInterval);
    // wss.close() might be called here too, but SIGINT handler is more robust for graceful shutdown
});

console.log("Server setup complete. Listening for connections...");

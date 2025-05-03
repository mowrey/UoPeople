// signaling_server.js (Group Chat + Link Blocking + AI Chat + Inactivity + Comment Generation API)
const WebSocket = require('ws');
const http = require('http'); // Import http module
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");
const { v4: uuidv4 } = require('uuid');

// --- Configuration ---
const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MAX_HISTORY_LENGTH = 100000;
const AI_MODEL_NAME = "gemini-1.5-flash-latest"; // Use a fast model for comments
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
function generateMessageId() { return uuidv4(); }
function safeSend(ws, data) { try { if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(data)); return true; } else { return false; } } catch (error) { console.error("safeSend Error:", error); return false; } }
function broadcastToRoom(topic, messageData, senderId) { if (!rooms[topic]) return; rooms[topic].forEach(peerId => { if (peerId !== senderId && peers[peerId]) { safeSend(peers[peerId].ws, messageData); } }); }
function notifyRoomUpdate(topic, reason = "update") { if (!rooms[topic]) return; const currentPeers = Array.from(rooms[topic]); let genericReason = reason; if (reason.startsWith('peer_joined:')) genericReason = 'peer_joined'; if (reason.startsWith('peer_left:')) genericReason = 'peer_left'; const updateMessage = { type: 'room_update', topic: topic, peers: currentPeers, reason: genericReason }; currentPeers.forEach(peerId => { if (peers[peerId]) { safeSend(peers[peerId].ws, updateMessage); } }); console.log(`Room update: ${topic}. Reason: ${genericReason}. Peers: ${currentPeers.length}`); }
function cleanupPeer(peerId, reasonCode = 1000, reasonMsg = "Cleanup called") { console.log(`Cleaning up peer: ${peerId}. Reason: ${reasonMsg}`); const peerData = peers[peerId]; if (!peerData) { return; } const topic = peerData.topic; if (topic && rooms[topic]) { const wasInRoom = rooms[topic].delete(peerId); if (wasInRoom) { console.log(`Removed ${peerId} from room ${topic}. Size: ${rooms[topic].size}`); if (rooms[topic].size === 0) { console.log(`Deleting empty room: ${topic}`); delete rooms[topic]; delete roomHistories[topic]; } else { notifyRoomUpdate(topic, `peer_left`); } } } delete peers[peerId]; console.log(`Peer ${peerId} removed. Total: ${Object.keys(peers).length}.`); if (peerData.ws && peerData.ws.readyState !== WebSocket.CLOSED && peerData.ws.readyState !== WebSocket.CLOSING) { try { peerData.ws.close(reasonCode, reasonMsg); } catch (e) { /* ignore close errors */ } } }
function blockLinks(text) { if (!text) return ""; const urlRegex = /(?:https?:\/\/|www\.)[^\s/$.?#].[^\s]*|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(\/[^\s]*)?/gi; const blockedText = text.replace(urlRegex, LINK_REPLACEMENT); if (blockedText !== text) { console.log(`Blocked link(s) in message.`); } return blockedText; }

// --- NEW AI Function for Single Comment Generation ---
async function generateSingleComment(postContext) {
    if (!postContext || postContext.trim() === "") {
        return "[No context provided for comment generation.]";
    }
    console.log(`Requesting AI comment generation for context: "${postContext.substring(0, 50)}..."`);

    const prompt = `Write a short, realistic, and relevant comment (like one you'd see on a blog or social media) reacting to the following post content snippet: "${postContext}". Keep the comment under 25 words. Be supportive, curious, or offer a brief related thought. Do not use hashtags. Do not introduce yourself. Do not ask a question.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.8, topP: 0.95, maxOutputTokens: 60 }, // Slightly higher temp for variety
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


// --- Create HTTP Server ---
const server = http.createServer(async (req, res) => {
    // --- CORS Handling ---
    // Set CORS headers for all responses to allow frontend access
    res.setHeader('Access-Control-Allow-Origin', '*'); // Allow requests from any origin (adjust for production)
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Handle CORS preflight requests (OPTIONS method)
    if (req.method === 'OPTIONS') {
        res.writeHead(204); // No Content
        res.end();
        return;
    }

    // --- API Endpoint Handling ---
    if (req.method === 'POST' && req.url === COMMENT_API_ENDPOINT) {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); }); // Collect request body data
        req.on('end', async () => {
            try {
                const requestData = JSON.parse(body);
                const postContext = requestData.context; // Expecting { "context": "..." }

                if (!postContext) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Missing "context" in request body.' }));
                    return;
                }

                // Generate the comment using the new function
                const generatedComment = await generateSingleComment(postContext);

                // Send the response
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ comment: generatedComment }));

            } catch (error) {
                console.error("Error processing API request:", error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Internal server error processing request.' }));
            }
        });
    }
    // --- Default Handling for other HTTP requests ---
    else {
        // Ignore other HTTP requests (or handle favicon, etc. if needed)
        if (req.url !== '/favicon.ico') { // Avoid logging favicon requests
             console.log(`Ignoring HTTP ${req.method} request for ${req.url}`);
        }
        res.writeHead(404);
        res.end();
    }
});

// --- WebSocket Server Setup (Attached to HTTP Server) ---
const wss = new WebSocket.Server({ noServer: true }); // Important: Use noServer option

// Handle WebSocket upgrade requests via the HTTP server
server.on('upgrade', (request, socket, head) => {
    // You could add authentication/origin checks here if needed
    console.log('Handling WebSocket upgrade request...');
    wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request); // Emit connection event for wss listeners
    });
});


// --- WebSocket Connection Handling (largely unchanged) ---
wss.on('connection', (ws) => {
    const peerId = generateId();
    peers[peerId] = { ws: ws, topic: null, isAlive: true, lastActivity: Date.now() };
    console.log(`Peer connected via WebSocket: ${peerId}`);
    safeSend(ws, { type: 'your_id', id: peerId });

    ws.isAlive = true;
    ws.on('pong', () => { if(peers[peerId]) peers[peerId].isAlive = true; });

    ws.on('message', async (message) => {
        let data;
        try { data = JSON.parse(message.toString()); }
        catch (e) { console.error(`Msg parse error ${peerId}:`, e); return; }

        if (!peers[peerId]) { console.warn(`Msg from dead peer ${peerId}`); return; }

        peers[peerId].isAlive = true;
        peers[peerId].lastActivity = Date.now();

        const currentTopic = peers[peerId]?.topic;
        // console.log(`Received WS ${data.type} from ${peerId} in topic ${currentTopic || 'None'}`); // Less verbose log

        switch (data.type) {
            case 'join':
                const newTopic = data.topic?.trim();
                if (!newTopic) { safeSend(ws, {type: 'error', message: 'Topic invalid.'}); return; }
                const oldTopic = peers[peerId]?.topic;
                if (oldTopic && oldTopic !== newTopic && rooms[oldTopic]) {
                    if (rooms[oldTopic].delete(peerId)) {
                        if (rooms[oldTopic].size === 0) { delete rooms[oldTopic]; delete roomHistories[oldTopic]; }
                        else { notifyRoomUpdate(oldTopic, `peer_left`); }
                    }
                }
                if(peers[peerId]) { peers[peerId].topic = newTopic; }
                else { peers[peerId] = { ws: ws, topic: newTopic, isAlive: true, lastActivity: Date.now() }; console.warn(`Re-added missing peer ${peerId} during join.`); }
                if (!rooms[newTopic]) { rooms[newTopic] = new Set(); roomHistories[newTopic] = []; console.log(`Creating room: ${newTopic}`); }
                rooms[newTopic].add(peerId);
                console.log(`Peer ${peerId} joined room ${newTopic}. Size: ${rooms[newTopic].size}`);
                notifyRoomUpdate(newTopic, `peer_joined:${peerId}`);
                break;

            case 'chat_message':
                if (!currentTopic || !rooms[currentTopic] || !data.message) return;
                let receivedChatMsg = data.message.substring(0, 1500);
                const messageWithBlockedLinks = blockLinks(receivedChatMsg);
                const messageId = generateMessageId();
                const historyEntry = { msgId: messageId, senderId: peerId, role: "user", parts: [{ text: messageWithBlockedLinks }], timestamp: Date.now() };
                if (!roomHistories[currentTopic]) roomHistories[currentTopic] = [];
                roomHistories[currentTopic].push(historyEntry);
                while (roomHistories[currentTopic].length > MAX_HISTORY_LENGTH) roomHistories[currentTopic].shift();
                const messagePayload = { type: 'chat_message', topic: currentTopic, messageId: messageId, senderId: peerId, message: messageWithBlockedLinks };
                broadcastToRoom(currentTopic, messagePayload, peerId);
                break;

             case 'ask_ai': // Keep this for the chat functionality
                 if (!currentTopic || !rooms[currentTopic]) { safeSend(ws, { type: 'error', message: 'Must be in a room to ask AI.' }); return; }
                 console.log(`Chat AI request for: ${currentTopic} from ${peerId}`);
                 broadcastToRoom(currentTopic, { type: 'system_message', topic: currentTopic, message: `AI is thinking...` }, null);
                 // Using getAIResponseForChat - assuming you might want different logic/prompts
                 // If not, you could reuse generateSingleComment with history, but that's less typical for chat context
                 // Let's assume getAIResponseForChat exists or you adapt it
                 const aiChatResponse = await getAIResponseForChat(currentTopic); // Rename or adapt this function
                 const aiMessagePayload = { type: 'ai_message', topic: currentTopic, message: aiChatResponse, senderId: 'AI' };
                 broadcastToRoom(currentTopic, aiMessagePayload, null);
                 break;

             case 'leave':
                 console.log(`Peer ${peerId} requested leaving room ${currentTopic}.`);
                 cleanupPeer(peerId, 1000, "User left");
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
        cleanupPeer(peerId, 1011, "WebSocket error");
    });
});

// Placeholder for the chat-specific AI function (adapt from original if needed)
async function getAIResponseForChat(topic) {
    // This should contain the logic from your original 'getAIResponse'
    // using roomHistories[topic] to generate a context-aware chat response.
    // For now, returning a placeholder:
    console.warn(`getAIResponseForChat function needs implementation based on original server logic.`);
    return "[Chat AI response placeholder]";
}


// --- Heartbeat & Inactivity Intervals (Unchanged) ---
const heartbeatInterval = setInterval(() => {
    wss.clients.forEach((ws) => {
        const peerEntry = Object.entries(peers).find(([id, data]) => data?.ws === ws);
        if (!peerEntry) { ws.terminate(); return; }
        const peerId = peerEntry[0]; const peerData = peerEntry[1];
        if (peerData.isAlive === false) { cleanupPeer(peerId, 1001, "Heartbeat timeout"); return; }
        peerData.isAlive = false;
        try { if (ws.readyState === WebSocket.OPEN) ws.ping(() => {}); else cleanupPeer(peerId, 1001, "Heartbeat non-open"); }
        catch (e) { console.error(`Error pinging ${peerId}:`, e); cleanupPeer(peerId, 1011, "Heartbeat ping error"); }
    });
}, 30000);

const inactivityInterval = setInterval(() => {
    const now = Date.now();
    Object.keys(peers).forEach(peerId => {
        const peerData = peers[peerId];
        if (peerData && peerData.lastActivity) {
            const idleTime = now - peerData.lastActivity;
            if (idleTime > INACTIVITY_TIMEOUT) {
                console.warn(`Peer ${peerId} inactive. Disconnecting.`);
                safeSend(peerData.ws, { type: 'disconnect_inactive', topic: peerData.topic, timeout: INACTIVITY_TIMEOUT / 1000 });
                setTimeout(() => cleanupPeer(peerId, 1001, "Inactivity timeout"), 100);
            }
        } else if (peerData && !peerData.lastActivity) {
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
server.on('close', () => {
    console.log("HTTP server shutting down. Clearing intervals.");
    clearInterval(heartbeatInterval);
    clearInterval(inactivityInterval);
    wss.close(); // Close WebSocket server too
});

console.log("Server setup complete.");

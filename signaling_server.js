// signaling_server.js (Group Chat + Link Blocking + AI Chat + Inactivity)
const WebSocket = require('ws');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");
const { v4: uuidv4 } = require('uuid');

// --- Configuration ---
const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MAX_HISTORY_LENGTH = 100000; // Max messages stored per room history
const AI_MODEL_NAME = "gemini-1.5-flash-latest";
const INACTIVITY_TIMEOUT = 60 * 60 * 1000; // 1 hour (in milliseconds)
const ACTIVITY_CHECK_INTERVAL = 30 * 1000; // How often to check inactivity
const LINK_REPLACEMENT = "****"; // What to replace detected URLs with

// --- Validate API Key ---
if (!GEMINI_API_KEY) { console.error("FATAL ERROR: GEMINI_API_KEY environment variable is not set."); process.exit(1); }

// --- Initialize AI Client ---
let genAI, aiModel;
try {
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    aiModel = genAI.getGenerativeModel({ model: AI_MODEL_NAME });
    console.log(`Initialized Google AI: ${AI_MODEL_NAME}`);
} catch (error) { console.error("FATAL ERROR: AI client init failed.", error); process.exit(1); }

// Safety settings specifically for the AI's *generated* responses
const generationSafetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

// --- WebSocket Server Setup ---
const wss = new WebSocket.Server({ port: PORT });
console.log(`Chat Relay Server started on port ${PORT}`);

// --- Server State ---
let peers = {};         // Stores data for connected clients { peerId: { ws, topic, isAlive, lastActivity }, ... }
let rooms = {};         // Tracks peers in each room { topic: Set<peerId>, ... }
let roomHistories = {}; // Stores recent messages per room { topic: [{ msgId, senderId, role, parts, timestamp }], ... }

// --- Helper Functions ---
function generateId() { return Math.random().toString(36).substring(2, 10); }
function generateMessageId() { return uuidv4(); }
function safeSend(ws, data) { try { if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(data)); return true; } else { return false; } } catch (error) { console.error("safeSend Error:", error); return false; } }
function broadcastToRoom(topic, messageData, senderId) { if (!rooms[topic]) return; rooms[topic].forEach(peerId => { if (peerId !== senderId && peers[peerId]) { safeSend(peers[peerId].ws, messageData); } }); }
function notifyRoomUpdate(topic, reason = "update") { if (!rooms[topic]) return; const currentPeers = Array.from(rooms[topic]); let genericReason = reason; if (reason.startsWith('peer_joined:')) genericReason = 'peer_joined'; if (reason.startsWith('peer_left:')) genericReason = 'peer_left'; const updateMessage = { type: 'room_update', topic: topic, peers: currentPeers, reason: genericReason }; currentPeers.forEach(peerId => { if (peers[peerId]) { safeSend(peers[peerId].ws, updateMessage); } }); console.log(`Room update: ${topic}. Reason: ${genericReason}. Peers: ${currentPeers.length}`); }

// Cleans up resources associated with a peer connection
function cleanupPeer(peerId, reasonCode = 1000, reasonMsg = "Cleanup called") {
    console.log(`Cleaning up peer: ${peerId}. Reason: ${reasonMsg}`);
    const peerData = peers[peerId]; if (!peerData) { return; }
    const topic = peerData.topic;

    // Remove peer from their room
    if (topic && rooms[topic]) {
        const wasInRoom = rooms[topic].delete(peerId);
        if (wasInRoom) {
             console.log(`Removed ${peerId} from room ${topic}. Size: ${rooms[topic].size}`);
             if (rooms[topic].size === 0) { // If room becomes empty, delete its data
                 console.log(`Deleting empty room: ${topic}`);
                 delete rooms[topic]; delete roomHistories[topic];
             } else {
                 notifyRoomUpdate(topic, `peer_left`); // Notify remaining peers
             }
        }
    }
    delete peers[peerId]; // Remove from the main peer list
    console.log(`Peer ${peerId} removed. Total: ${Object.keys(peers).length}.`);
    if (peerData.ws && peerData.ws.readyState !== WebSocket.CLOSED && peerData.ws.readyState !== WebSocket.CLOSING) {
        peerData.ws.close(reasonCode, reasonMsg); // Attempt to formally close WS
    }
}

// Replaces detected URLs in a string
function blockLinks(text) {
    if (!text) return "";
    // Regex to find common URL patterns (http/https, www, domain.tld)
    const urlRegex = /(?:https?:\/\/|www\.)[^\s/$.?#].[^\s]*|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(\/[^\s]*)?/gi;
    const blockedText = text.replace(urlRegex, LINK_REPLACEMENT);
    if (blockedText !== text) {
        console.log(`Blocked link(s) in message. Original: "${text.substring(0,50)}...", Blocked: "${blockedText.substring(0,50)}..."`);
    }
    return blockedText;
}

// Generates an AI response using the room's history
async function getAIResponse(topic) {
    if (!roomHistories[topic] || roomHistories[topic].length === 0) return "The conversation hasn't started yet.";
    console.log(`Requesting AI response for: ${topic}`);

    const history = roomHistories[topic].filter(m => m.role === 'user' || m.role === 'model');
    if (history.length === 0) return "[Not enough conversation history yet.]";

    // Format history for the AI model, merging consecutive messages from the same role
    let formattedHistory = []; let lastRole = null; let currentParts = [];
    history.forEach((entry, index) => {
        const sanitizedParts = entry.parts.map(part => ({ text: (typeof part.text === 'string' ? part.text.substring(0, 1000) : "[invalid]") })); // Sanitize/truncate parts
        if (entry.role === lastRole) {
            currentParts.push(...sanitizedParts);
        } else {
            if (currentParts.length > 0 && lastRole) formattedHistory.push({ role: lastRole, parts: currentParts });
            currentParts = sanitizedParts; lastRole = entry.role;
        }
        if (index === history.length - 1 && currentParts.length > 0) formattedHistory.push({ role: lastRole, parts: currentParts });
    });

    // Add a system prompt to guide the AI
    const systemPrompt = `Observe the chat about "${topic}". Briefly comment on recent messages, offer insight, or ask a question. Concise (1-2 sentences).`;
    formattedHistory.unshift({ role: "user", parts: [{ text: systemPrompt }] });
    if (formattedHistory.length > 1 && formattedHistory[0].role === 'model') formattedHistory.shift(); // Prefer user start for context

    if (formattedHistory.length === 0 || (formattedHistory.length === 1 && formattedHistory[0].role === 'user')) {
        return "[Not enough conversation yet for AI comment.]";
    }

    try {
        const result = await aiModel.generateContent({
            contents: formattedHistory,
            generationConfig: { temperature: 0.75, topP: 0.95, maxOutputTokens: 150 },
            safetySettings: generationSafetySettings
        });

        if (result.response) {
            const text = result.response.text();
            if (!text || text.trim().length === 0) return "[AI had no comment.]";

            console.log(`AI Resp ${topic}: ${text.substring(0, 80)}...`);
            if (!roomHistories[topic]) roomHistories[topic] = [];
            roomHistories[topic].push({ role: "model", parts: [{ text: text }], msgId: generateMessageId(), senderId: 'AI', timestamp: Date.now() });
            while (roomHistories[topic].length > MAX_HISTORY_LENGTH) roomHistories[topic].shift();
            return text;
        } else {
            const blockReason = result?.response?.promptFeedback?.blockReason || result?.response?.candidates?.[0]?.finishReason;
            console.error(`AI Gen Err ${topic}: Blocked? ${blockReason}`);
            return `[AI response blocked. Reason: ${blockReason || 'Filter'}.]`;
        }
    } catch (error) {
        console.error(`AI API error ${topic}:`, error);
        return `[AI encountered an error.]`;
    }
 }

// --- WebSocket Connection Handling ---
wss.on('connection', (ws) => {
    const peerId = generateId();
    peers[peerId] = { ws: ws, topic: null, isAlive: true, lastActivity: Date.now() };
    console.log(`Peer connected: ${peerId}`);
    safeSend(ws, { type: 'your_id', id: peerId });

    // Heartbeat mechanism: flag client as alive when pong received
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
        console.log(`Received ${data.type} from ${peerId} in topic ${currentTopic || 'None'}`);

        switch (data.type) {
            case 'join':
                const newTopic = data.topic?.trim();
                if (!newTopic) { safeSend(ws, {type: 'error', message: 'Topic invalid.'}); return; }
                const oldTopic = peers[peerId]?.topic;

                // Handle topic switching: leave old room first
                if (oldTopic && oldTopic !== newTopic && rooms[oldTopic]) {
                    console.log(`Peer ${peerId} switching from ${oldTopic} to ${newTopic}`);
                    if (rooms[oldTopic].delete(peerId)) {
                        if (rooms[oldTopic].size === 0) {
                            console.log(`Deleting empty room: ${oldTopic}`);
                            delete rooms[oldTopic]; delete roomHistories[oldTopic];
                        } else {
                            notifyRoomUpdate(oldTopic, `peer_left`);
                        }
                    }
                }

                // Add peer to the new room
                if(peers[peerId]) { peers[peerId].topic = newTopic; }
                else { peers[peerId] = { ws: ws, topic: newTopic, isAlive: true, lastActivity: Date.now() }; console.warn(`Re-added missing peer ${peerId} during join.`); } // Handle rare race condition
                if (!rooms[newTopic]) { // Initialize room if needed
                    rooms[newTopic] = new Set(); roomHistories[newTopic] = [];
                    console.log(`Creating room: ${newTopic}`);
                }
                rooms[newTopic].add(peerId);
                console.log(`Peer ${peerId} joined room ${newTopic}. Size: ${rooms[newTopic].size}`);
                notifyRoomUpdate(newTopic, `peer_joined:${peerId}`);
                break;

            case 'chat_message':
                if (!currentTopic || !rooms[currentTopic] || !data.message) return;
                let receivedChatMsg = data.message.substring(0, 1500); // Limit length

                // Block links in the message
                const messageWithBlockedLinks = blockLinks(receivedChatMsg);
                const messageId = generateMessageId();

                // Store the link-blocked version in history
                const historyEntry = { msgId: messageId, senderId: peerId, role: "user", parts: [{ text: messageWithBlockedLinks }], timestamp: Date.now() };
                if (!roomHistories[currentTopic]) roomHistories[currentTopic] = [];
                roomHistories[currentTopic].push(historyEntry);
                while (roomHistories[currentTopic].length > MAX_HISTORY_LENGTH) roomHistories[currentTopic].shift();

                // Broadcast the link-blocked message
                const messagePayload = { type: 'chat_message', topic: currentTopic, messageId: messageId, senderId: peerId, message: messageWithBlockedLinks };
                broadcastToRoom(currentTopic, messagePayload, peerId);
                break;

             case 'ask_ai':
                 if (!currentTopic || !rooms[currentTopic]) { safeSend(ws, { type: 'error', message: 'Must be in a room to ask AI.' }); return; }

                 console.log(`AI request for: ${currentTopic} from ${peerId}`);
                 broadcastToRoom(currentTopic, { type: 'system_message', topic: currentTopic, message: `AI's response...` }, null); // Notify room

                 const aiResponse = await getAIResponse(currentTopic); // Get AI response

                 const aiMessagePayload = { type: 'ai_message', topic: currentTopic, message: aiResponse, senderId: 'AI' };
                 broadcastToRoom(currentTopic, aiMessagePayload, null); // Broadcast AI response
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
        cleanupPeer(peerId, 1011, "WebSocket error"); // 1011: Internal Server Error
    });
});

// --- Heartbeat & Inactivity Intervals ---

// Heartbeat: Pings clients periodically, cleans up unresponsive ones
const heartbeatInterval = setInterval(() => {
    wss.clients.forEach((ws) => {
        const peerEntry = Object.entries(peers).find(([id, data]) => data?.ws === ws);
        if (!peerEntry) { console.warn("Found zombie WebSocket client, terminating."); ws.terminate(); return; } // Safety check

        const peerId = peerEntry[0];
        const peerData = peerEntry[1];

        if (peerData.isAlive === false) { // Didn't respond to the last ping
            console.log(`Heartbeat failed for peer ${peerId}. Cleaning up.`);
            cleanupPeer(peerId, 1001, "Heartbeat timeout"); // 1001: Going Away
            return;
        }

        peerData.isAlive = false; // Assume dead until pong arrives
        try {
            if (ws.readyState === WebSocket.OPEN) ws.ping(() => {});
            else cleanupPeer(peerId, 1001, "Heartbeat non-open");
        } catch (e) {
            console.error(`Error sending ping to ${peerId}:`, e);
            cleanupPeer(peerId, 1011, "Heartbeat ping error");
        }
    });
}, 30000); // Interval: 30 seconds

// Inactivity Check: Disconnects peers idle for longer than INACTIVITY_TIMEOUT
const inactivityInterval = setInterval(() => {
    const now = Date.now();
    Object.keys(peers).forEach(peerId => {
        const peerData = peers[peerId];
        if (peerData && peerData.lastActivity) {
            const idleTime = now - peerData.lastActivity;
            if (idleTime > INACTIVITY_TIMEOUT) {
                console.warn(`Peer ${peerId} inactive for ${Math.round(idleTime / 1000)}s. Disconnecting.`);
                safeSend(peerData.ws, { type: 'disconnect_inactive', topic: peerData.topic, timeout: INACTIVITY_TIMEOUT / 1000 });
                setTimeout(() => cleanupPeer(peerId, 1001, "Inactivity timeout"), 100); // Delay cleanup slightly to allow message send
            }
        } else if (peerData && !peerData.lastActivity) {
            peerData.lastActivity = now; // Initialize if somehow missing
        }
    });
}, ACTIVITY_CHECK_INTERVAL);

// Cleanup intervals on server shutdown
wss.on('close', () => {
    console.log("WebSocket server shutting down. Clearing intervals.");
    clearInterval(heartbeatInterval);
    clearInterval(inactivityInterval);
});

console.log("Server setup complete with Link Blocking, AI Chat, and Inactivity Timeout.");

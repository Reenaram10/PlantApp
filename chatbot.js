const API_BASE_URL = 'http://127.0.0.1:5001';

// Initialize chatbot when document loads
document.addEventListener('DOMContentLoaded', () => {
    const chatbotHTML = `
        <div id="chatbot-container">
            <div id="chatbot-header">
                <h3>Plant Assistant</h3>
                <div id="user-status"></div>
                <button id="logout-btn" onclick="handleLogout()" class="logout-button">Logout</button>
            </div>
            <div id="chatbot-messages"></div>
            <div id="chatbot-input-area">
                <input type="text" id="chatbot-input" placeholder="Ask about plants..." disabled>
                <button id="chatbot-send" disabled>Send</button>
            </div>
        </div>
        <div id="orders-section">
            <h3>Your Orders</h3>
            <div id="orders-list"></div>
        </div>
    `;

    const chatbotContainer = document.getElementById('chatbot-wrapper');
    if (chatbotContainer) {
        chatbotContainer.innerHTML = chatbotHTML;
        initializeChatbot();
    } else {
        console.error('Chatbot wrapper not found!');
    }
});

// ---------- USER & LOCATION ----------
function getCurrentUser() {
    try {
        const userStr = localStorage.getItem('user');
        return userStr ? JSON.parse(userStr) : null;
    } catch (error) {
        console.error('Error getting user:', error);
        return null;
    }
}

function getUserLocation() {
    try {
        const locationStr = localStorage.getItem('user_location');
        return locationStr ? JSON.parse(locationStr) : null;
    } catch (error) {
        console.error('Error getting location:', error);
        return null;
    }
}

async function requestAndStoreLocation() {
    return new Promise((resolve) => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    const location = `${pos.coords.latitude},${pos.coords.longitude}`;
                    localStorage.setItem('user_location', JSON.stringify(location));
                    resolve(location);
                },
                (error) => {
                    console.error('Geolocation error:', error);
                    resolve('');
                }
            );
        } else {
            resolve('');
        }
    });
}

// ---------- CHATBOT INITIALIZATION ----------
async function initializeChatbot() {
    const user = getCurrentUser();
    const location = getUserLocation();
    const inputField = document.getElementById('chatbot-input');
    const sendButton = document.getElementById('chatbot-send');
    const userStatus = document.getElementById('user-status');
    const logoutBtn = document.getElementById('logout-btn');

    if (user && user.id) {
        inputField.disabled = false;
        sendButton.disabled = false;
        logoutBtn.style.display = 'block';
        userStatus.innerHTML = `<span class="logged-in">Logged in as: ${user.username}</span>`;
        
        if (!location) {
            await requestAndStoreLocation();
        }
        
        addMessage('Bot', `Welcome back, ${user.username}! How can I help you with plants today?`);
        loadUserOrders();
    } else {
        inputField.disabled = true;
        sendButton.disabled = true;
        logoutBtn.style.display = 'none';
        userStatus.innerHTML = '<span class="logged-out">Please login to chat</span>';
        addMessage('Bot', 'Please login to start the conversation.');
    }

    sendButton.onclick = handleSend;
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
}

// ---------- SEND & PROCESS MESSAGES ----------
async function handleSend() {
    const user = getCurrentUser();
    if (!user || !user.id) {
        addMessage('Bot', 'Please login to continue the conversation.');
        return;
    }

    const input = document.getElementById('chatbot-input');
    const msg = input.value.trim();
    if (!msg) return;
    
    addMessage('You', msg);
    input.value = '';

    try {
        const location = getUserLocation() || await requestAndStoreLocation();
        await sendToBot(msg, location, user.id);
    } catch (error) {
        console.error('Error:', error);
        addMessage('Bot', 'Sorry, I encountered an error. Please try again.');
    }
}

// ---------- SEND TO BOT ----------
async function sendToBot(msg, location, userId) {
    addMessage('Bot', '*Thinking...*');
    
    try {
        const res = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, location, user_id: userId })
        });

        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        // Remove thinking message
        const messagesDiv = document.getElementById('chatbot-messages');
        messagesDiv.lastChild.remove();

        if (data.status === 'success') {
            if (data.type === 'image') {
                addMessageWithImage('Bot', data);
            } else {
                handleBotResponse(data);
            }
        } else {
            throw new Error(data.message || 'Unknown error');
        }
    } catch (error) {
        console.error('Error:', error);
        const messagesDiv = document.getElementById('chatbot-messages');
        if (messagesDiv.lastChild) messagesDiv.lastChild.remove();
        addMessage('Bot', 'Sorry, I encountered an error. Please try again.');
    }
}

// ---------- HANDLE BOT RESPONSE ----------
function handleBotResponse(data) {
    if (!data.reply) return;

    if (data.reply.includes('PLACE_ORDER:')) {
        const [orderCommand, ...restOfReply] = data.reply.split('\n');
        const plantId = orderCommand.split(':')[1].trim();
        handleOrder(plantId);
        addMessage('Bot', restOfReply.join('\n'));
    } else if (data.reply.includes('DELETE_ORDER:')) {
        const [deleteCommand, ...restOfReply] = data.reply.split('\n');
        const plantId = deleteCommand.split(':')[1].trim();
        handleOrderDeletion(plantId);
        addMessage('Bot', restOfReply.join('\n'));
    } else {
        addMessage('Bot', data.reply);
    }
}

// ---------- ADD MESSAGE (TEXT) ----------
function addMessage(sender, text) {
    const messages = document.getElementById('chatbot-messages');
    const div = document.createElement('div');
    div.className = `message ${sender.toLowerCase()}-message`;
    div.innerHTML = sender === 'Bot' ? `<strong>${sender}:</strong> ${marked.parse(text)}` : `<strong>${sender}:</strong> ${text}`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

// ---------- ADD MESSAGE (IMAGE) ----------
function addMessageWithImage(sender, data) {
    const messages = document.getElementById('chatbot-messages');
    const div = document.createElement('div');
    div.className = `message ${sender.toLowerCase()}-message`;
    
    div.innerHTML = `
        <strong>${sender}:</strong> ${data.reply}
        <div class="image-container">
            <img src="${data.image_url}" alt="Plant image" class="chat-image">
            <div class="image-credit">${data.image_credit}</div>
        </div>
    `;
    
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

// ---------- ORDER HANDLING ----------
async function handleOrder(plantId) {
    const user = getCurrentUser();
    if (!user || !user.id) {
        addMessage('Bot', 'Please login to place orders.');
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/order`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: user.id, plant_id: parseInt(plantId) })
        });

        const data = await res.json();
        addMessage('Bot', data.status === 'success' ? `✅ ${data.message}` : `❌ Error: ${data.message}`);
        if (data.status === 'success') loadUserOrders();
    } catch (error) {
        console.error('Order error:', error);
        addMessage('Bot', 'Sorry, there was an error placing your order.');
    }
}

async function handleOrderDeletion(plantId) {
    const user = getCurrentUser();
    if (!user || !user.id) {
        addMessage('Bot', 'Please login to manage orders.');
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/order/${user.id}/${plantId}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await res.json();
        addMessage('Bot', data.status === 'success' ? '✅ Order cancelled successfully' : `❌ Error: ${data.message}`);
        if (data.status === 'success') loadUserOrders();
    } catch (error) {
        console.error('Delete error:', error);
        addMessage('Bot', 'Sorry, there was an error cancelling your order.');
    }
}

// ---------- LOAD USER ORDERS ----------
async function loadUserOrders() {
    const user = getCurrentUser();
    if (!user || !user.id) return;

    try {
        const res = await fetch(`${API_BASE_URL}/api/orders/${user.id}`);
        const data = await res.json();
        
        const ordersList = document.getElementById('orders-list');
        if (data.status === 'success' && data.orders.length > 0) {
            ordersList.innerHTML = data.orders.map(order => `
                <div class="order-item">
                    <h4>${order.plant_name}</h4>
                    <p>Price: ₹${order.price}</p>
                    <p>Ordered: ${order.order_date}</p>
                    <button onclick="handleOrderDeletion(${order.plant_id})" class="delete-btn">Cancel Order</button>
                </div>
            `).join('');
        } else {
            ordersList.innerHTML = '<p>No orders found</p>';
        }
    } catch (error) {
        console.error('Error loading orders:', error);
        document.getElementById('orders-list').innerHTML = '<p>Error loading orders</p>';
    }
}

// ---------- LOGOUT ----------
async function handleLogout() {
    try {
        const response = await fetch(`${API_BASE_URL}/logout`, { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') clearUserCache();
    } catch (error) {
        console.error('Logout error:', error);
    }
}

function clearUserCache() {
    localStorage.removeItem('user');
    localStorage.removeItem('user_location');

    const inputField = document.getElementById('chatbot-input');
    const sendButton = document.getElementById('chatbot-send');
    const userStatus = document.getElementById('user-status');
    const ordersList = document.getElementById('orders-list');
    const messages = document.getElementById('chatbot-messages');
    const logoutBtn = document.getElementById('logout-btn');

    inputField.disabled = true;
    sendButton.disabled = true;
    logoutBtn.style.display = 'none';
    userStatus.innerHTML = '<span class="logged-out">Please login to chat</span>';
    messages.innerHTML = '';
    ordersList.innerHTML = '';

    addMessage('Bot', 'Please login to start the conversation.');
    window.location.href = '/login.html';
}

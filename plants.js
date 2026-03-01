const API_BASE_URL = "http://127.0.0.1:5001";

// ------------------ Bot Message Helper ------------------
function addMessage(sender, text) {
    const messages = document.getElementById('chatbot-messages');
    if (!messages) return;
    const div = document.createElement('div');
    div.className = `message ${sender.toLowerCase()}-message`;
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

// ------------------ User Helper ------------------
function getCurrentUser() {
    try {
        const userStr = localStorage.getItem('user');
        return userStr ? JSON.parse(userStr) : null;
    } catch (error) {
        console.error('Error getting user:', error);
        return null;
    }
}

// ------------------ Fetch & Display Plants ------------------
async function fetchPlants() {
    const container = document.getElementById("plantsList");
    if (!container) return;
    container.innerHTML = "<p style='color:#006400;'>🌱 Loading plants...</p>";

    try {
        const response = await fetch(`${API_BASE_URL}/plants`);
        const data = await response.json();

        if (data.status === "success" && Array.isArray(data.plants)) {
            displayPlants(data.plants);
        } else {
            container.innerHTML = `<p style='color:red;'>❌ ${data.message || "No plants found."}</p>`;
        }
    } catch (error) {
        console.error("Error fetching plants:", error);
        container.innerHTML = "<p style='color:red;'>❌ Unable to load plants. Check server.</p>";
    }
}

function displayPlants(plants) {
    const container = document.getElementById("plantsList");
    if (!container) return;
    container.innerHTML = "";

    const user = getCurrentUser();

    plants.forEach(plant => {
        const plantCard = document.createElement("div");
        plantCard.className = "plant-card";

        let addButton = "";
        if (user && user.username) {
            addButton = `<button onclick="addToCart('${plant.plant_name}')">Add to Cart</button>`;
        }

        plantCard.innerHTML = `
            <h3>${plant.plant_name}</h3>
            <p><strong>Description:</strong> ${plant.description}</p>
            <p><strong>Price:</strong> ₹${plant.price}</p>
            ${addButton}
        `;

        container.appendChild(plantCard);
    });
}

// ------------------ Cart Management ------------------
// ------------------ Cart Management ------------------
async function loadUserCart() {
    const user = getCurrentUser();
    const container = document.getElementById("cart-list");
    if (!user || !user.username || !container) return;

    try {
        const res = await fetch(`${API_BASE_URL}/api/cart?user_id=${user.username}`);
        const data = await res.json();

        if (data.status === "success" && data.cart.length > 0) {
            container.innerHTML = data.cart.map(item => `
                <div class="cart-item">
                    <span>${item.plant_name}</span>
                    <span>₹${item.price}</span>
                    <button onclick="removeFromCart('${item.plant_name}')">❌</button>
                </div>
            `).join('');
        } else {
            container.innerHTML = "<p>🛒 Your cart is empty</p>";
        }
    } catch (err) {
        console.error(err);
        container.innerHTML = "<p style='color:red;'>❌ Failed to load cart</p>";
    }
}

async function addToCart(plantName) {
    const user = getCurrentUser();
    if (!user || !user.username) return addMessage("Bot", "Please login to add items.");

    plantName = plantName.replace(/[^\w\s]/gi, '').trim();

    try {
        const res = await fetch(`${API_BASE_URL}/api/cart`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: user.username, plant_name: plantName })
        });

        const data = await res.json();
        if (data.status === "success") {
            addMessage("Bot", `✅ ${plantName} added to cart!`);
            await loadUserCart(); // reload cart immediately
        } else {
            addMessage("Bot", `❌ ${data.message}`);
        }
    } catch (err) {
        console.error(err);
        addMessage("Bot", "❌ Error adding to cart.");
    }
}

async function removeFromCart(plantName) {
    const user = getCurrentUser();
    if (!user || !user.username) return;

    try {
        const res = await fetch(`${API_BASE_URL}/api/cart`, {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: user.username, plant_name: plantName })
        });

        const data = await res.json();
        if (data.status === "success") {
            addMessage("Bot", `❌ ${plantName} removed from cart.`);
            await loadUserCart(); // refresh after removing
        } else {
            addMessage("Bot", `❌ ${data.message || "Failed to remove item."}`);
        }
    } catch (err) {
        console.error(err);
        addMessage("Bot", "❌ Error removing item.");
    }
}


// ------------------ Chatbot ------------------
async function sendUserText(userMessage) {
    const user = getCurrentUser();
    if (!user || !user.username) return addMessage("Bot", "Please login to chat.");

    addMessage("You", userMessage);

    try {
        const res = await fetch(`${API_BASE_URL}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage, user_id: user.username })
        });

        const data = await res.json();
        addMessage("Bot", data.status === "success" ? data.reply : `❌ ${data.message}`);
    } catch (err) {
        console.error(err);
        addMessage("Bot", "❌ Error sending message.");
    }
}

// ------------------ Voice Recording ------------------
document.addEventListener("DOMContentLoaded", () => {
    fetchPlants();
    loadUserCart();

    const recordButton = document.getElementById("record-btn");
    if (recordButton) {
        let mediaRecorder, audioChunks = [];
        recordButton.addEventListener("click", async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.onstart = () => addMessage("Bot", "🎙 Listening...");
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    const formData = new FormData();
                    formData.append("audio", audioBlob, "voice.wav");

                    try {
                        const res = await fetch(`${API_BASE_URL}/api/voice`, { method: "POST", body: formData });
                        const data = await res.json();

                        if (data.error) addMessage("Bot", `❌ ${data.error}`);
                        else {
                            addMessage("You", `🗣 ${data.transcribed_text}`);
                            addMessage("Bot", data.reply);
                        }
                    } catch (err) {
                        addMessage("Bot", "❌ Could not connect to server.");
                    }
                };

                mediaRecorder.start();
                setTimeout(() => { mediaRecorder.stop(); stream.getTracks().forEach(track => track.stop()); }, 5000);

            } catch (err) {
                addMessage("Bot", "⚠ Microphone access denied.");
            }
        });
    }
});
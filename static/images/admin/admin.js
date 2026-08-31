const API_BASE = "http://127.0.0.1:8080";
let currentSection = 'plant-master';
const fixImageUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    if (url.startsWith('/static')) return url; // Let Live Server serve local static files
    return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCategories();
    loadInventory();
    loadSuppliers();
});

// Navigation
function showSection(sectionId, event) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    document.getElementById(sectionId).classList.add('active');
    event.currentTarget.classList.add('active');

    currentSection = sectionId;
    const titles = {
        'plant-master': 'Plant Master',
        'varieties': 'Varieties Tab',
        'inventory': 'Inventory Management',
        'supplier-master': 'Supplier Master',
        'category-master': 'Category Master',
        'customer-master': 'Customer Master',
        'sales-master': 'Sales Master'
    };
    const subtitles = {
        'plant-master': 'Manage your core plant catalog and stock levels.',
        'varieties': 'Generate and add plant varieties using AI-powered suggestions.',
        'inventory': 'Track inward supplier stock and outward returns/wastage.',
        'supplier-master': 'Maintain global directory of your business suppliers.',
        'category-master': 'Create and manage plant categories.',
        'customer-master': 'View registered customers and their activity.',
        'sales-master': 'Track customer orders, update delivery status, and view invoices.'
    };

    document.getElementById('section-title').textContent = titles[sectionId];
    document.getElementById('section-subtitle').textContent = subtitles[sectionId];

    if (sectionId === 'plant-master') loadInventory();
    if (sectionId === 'supplier-master') loadSuppliers();
    if (sectionId === 'category-master') loadCategories();
    if (sectionId === 'customer-master') loadCustomers();
    if (sectionId === 'sales-master') loadOrders();
    if (sectionId === 'purchase-master') loadPurchases();
    if (sectionId === 'inventory') {
        loadSuppliers();
        loadInventory();
        loadTransactions();
    }
}

// Modal Controls
function openModal(modalId) {
    document.getElementById(modalId).style.display = 'flex';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
    if (modalId === 'add-plant-modal') document.getElementById('plant-form').reset();
    if (modalId === 'add-supplier-modal') document.getElementById('supplier-form').reset();
}

let allCategories = [];

// --- PLANT MASTER LOGIC ---
async function openPlantModal() {
    document.getElementById('plant-form').reset();
    document.getElementById('edit-plant-id').value = '';
    document.getElementById('m-stock-group').style.display = 'block';
    document.getElementById('plant-modal-title').textContent = 'Add New Plant';

    // Render category checkboxes
    renderCategoryCheckboxes();
    openModal('add-plant-modal');
}

async function editPlant(id) {
    try {
        const res = await fetch(`${API_BASE}/api/admin/get-plant/${id}`);
        const data = await res.json();
        if (data.status === 'success') {
            const plant = data.plant;
            document.getElementById('edit-plant-id').value = plant.plant_id;
            document.getElementById('m-plant-name').value = plant.plant_name;
            document.getElementById('m-plant-desc').value = plant.description;
            document.getElementById('m-plant-price').value = plant.price;
            document.getElementById('m-stock-group').style.display = 'none'; // Stock updated via inventory
            document.getElementById('m-plant-stock').value = plant.stock;
            document.getElementById('plant-modal-title').textContent = 'Edit Plant Details';

            const catIds = plant.categories.map(c => c.category_id);
            renderCategoryCheckboxes(catIds);
            openModal('add-plant-modal');
        }
    } catch (err) {
        showToast("Failed to fetch plant details", "error");
    }
}

function renderCategoryCheckboxes(selectedIds = []) {
    const container = document.getElementById('m-plant-categories');
    container.innerHTML = '';
    if (allCategories.length === 0) {
        container.innerHTML = '<span style="color:var(--gray-700); font-size: 0.9rem;">No categories available. Please create them first.</span>';
        return;
    }
    allCategories.forEach(c => {
        const isChecked = selectedIds.includes(c.category_id) ? 'checked' : '';
        container.insertAdjacentHTML('beforeend', `
            <label style="display:flex; align-items:center; gap:5px; font-size: 0.9rem;">
                <input type="checkbox" class="plant-cat-checkbox" value="${c.category_id}" ${isChecked}>
                ${c.category_name}
            </label>
        `);
    });
}

document.getElementById('plant-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = document.getElementById('edit-plant-id').value;
    const selectedCategories = Array.from(document.querySelectorAll('.plant-cat-checkbox:checked')).map(cb => parseInt(cb.value));

    const payload = {
        plant_name: document.getElementById('m-plant-name').value,
        description: document.getElementById('m-plant-desc').value,
        price: document.getElementById('m-plant-price').value,
        categories: selectedCategories
    };

    let url = `${API_BASE}/api/admin/add-plant`;
    let method = 'POST';

    if (id) {
        url = `${API_BASE}/api/admin/update-plant/${id}`;
        method = 'PUT';
    } else {
        payload.stock = document.getElementById('m-plant-stock').value;
    }

    try {
        const res = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(data.message, "success");
            closeModal('add-plant-modal');
            loadInventory();
        } else {
            showToast(data.message, "error");
        }
    } catch (err) {
        showToast("Error saving plant", "error");
    }
});

let searchTimeout;
function debounceSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => searchPlants(), 300);
}

async function searchPlants() {
    const query = document.getElementById('plant-search').value.trim();
    const dropdown = document.getElementById('search-dropdown');

    if (query.length < 2) {
        dropdown.style.display = 'none';
        if (query.length === 0) loadInventory();
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/admin/search-plants?q=${query}`);
        const data = await res.json();

        if (data.status === 'success') {
            displayInventory(data.plants);
        }
    } catch (err) {
        showToast("Error searching plants", "error");
    }
}

async function loadInventory() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/inventory`);
        const data = await res.json();
        if (data.status === 'success') {
            displayInventory(data.plants);
        }
    } catch (err) {
        showToast("Failed to load inventory", "error");
    }
}

function displayInventory(plants) {
    const tbody = document.getElementById('inventoryBody');
    tbody.innerHTML = '';

    plants.forEach(p => {
        const stockBadge = p.stock <= 5 ? 'badge-danger' : (p.stock < 20 ? 'badge-warning' : 'badge-success');
        const row = `
            <tr>
                <td>#${p.plant_id}</td>
                <td style="width: 50px; text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px; overflow: hidden; background: #f0f7f0; display: flex; align-items: center; justify-content: center; border: 1px solid var(--gray-200);">
                        ${p.image_url ?
                `<img src="${fixImageUrl(p.image_url)}" style="width: 100%; height: 100%; object-fit: cover;">` :
                `<span style="font-size: 20px;">🌱</span>`
            }
                    </div>
                </td>
                <td>
                    <strong>${p.plant_name}</strong>
                    <div style="font-size: 0.8rem; color: var(--gray-700);">${p.categories || 'No Category'}</div>
                </td>
                <td>$${parseFloat(p.price).toFixed(2)}</td>
                <td><span class="badge ${stockBadge}">${p.stock} units</span></td>
                <td>${p.stock > 0 ? '<span class="badge badge-success">In Stock</span>' : '<span class="badge badge-danger">Out of Stock</span>'}</td>
                <td>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button onclick="viewPlantImage('${p.plant_name}', '${fixImageUrl(p.image_url || '')}')" class="btn" style="padding: 5px 10px; background:#27ae60; color:#fff;" title="View Image"><i class="fas fa-image"></i></button>
                        <button onclick="editPlant(${p.plant_id})" class="btn btn-secondary" style="padding: 5px 10px;" title="Edit Details"><i class="fas fa-edit"></i></button>
                        <button onclick="openChangeImageModal(${p.plant_id}, '${p.plant_name}')" class="btn btn-primary" style="padding: 5px 10px; background:#8e44ad;" title="Change Image"><i class="fas fa-camera"></i></button>
                        <button onclick="deletePlant(${p.plant_id}, '${p.plant_name}')" class="btn btn-danger" style="padding: 5px 10px;" title="Delete"><i class="fas fa-trash"></i></button>
                    </div>
                </td>
            </tr>
        `;
        tbody.insertAdjacentHTML('beforeend', row);
    });
}

async function deletePlant(id, name) {
    if (!confirm(`Are you sure you want to PERMANENTLY remove ${name}? This will delete all varieties and order history.`)) return;

    try {
        const res = await fetch(`${API_BASE}/api/admin/remove-plant/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(data.message, "success");
            loadInventory();
        } else {
            showToast(data.message, "error");
        }
    } catch (err) {
        showToast("Server error deleting plant", "error");
    }
}

function openChangeImageModal(plantId, plantName) {
    document.getElementById('ci-plant-id').value = plantId;
    document.getElementById('ci-plant-name').textContent = plantName;
    document.getElementById('ci-file-input').value = '';
    document.getElementById('ci-url-input').value = '';
    document.getElementById('ci-preview').src = '';
    document.getElementById('ci-preview').style.display = 'none';
    openModal('change-image-modal');
}

function viewPlantImage(name, imageUrl) {
    document.getElementById('vpi-name').textContent = name;
    const img = document.getElementById('vpi-img');
    const noImg = document.getElementById('vpi-no-img');
    if (imageUrl) {
        img.src = imageUrl;
        img.style.display = 'block';
        noImg.style.display = 'none';
    } else {
        img.style.display = 'none';
        noImg.style.display = 'block';
    }
    openModal('view-plant-image-modal');
}

document.getElementById('ci-file-input')?.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            const img = document.getElementById('ci-preview');
            img.src = e.target.result;
            img.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

async function submitPlantImage() {
    const plantId = document.getElementById('ci-plant-id').value;
    const fileInput = document.getElementById('ci-file-input');
    const urlInput = document.getElementById('ci-url-input').value.trim();

    const formData = new FormData();
    if (fileInput.files[0]) {
        formData.append('image', fileInput.files[0]);
    } else if (urlInput) {
        formData.append('image_url', urlInput);
    } else {
        showToast('Please select a file or provide an image URL.', 'error');
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/admin/update-plant-image/${plantId}`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(data.message, 'success');
            closeModal('change-image-modal');
            loadInventory();
        } else {
            showToast(data.message, 'error');
        }
    } catch (err) {
        showToast('Server error updating image', 'error');
    }
}

// --- SUPPLIER MASTER LOGIC ---

async function loadSuppliers() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/suppliers`);
        const data = await res.json();
        if (data.status === 'success') {
            const suppliers = data.suppliers;

            // Update table
            const tbody = document.getElementById('supplierBody');
            tbody.innerHTML = '';
            suppliers.forEach(s => {
                tbody.insertAdjacentHTML('beforeend', `
            <tr>
                        <td>#${s.supplier_id}</td>
                        <td><strong>${s.name}</strong></td>
                        <td>${s.contact_number || '-'}</td>
                        <td>${s.email || '-'}</td>
                        <td>${s.address || '-'}</td>
                        <td>
                            <button onclick="editSupplier(${s.supplier_id})" class="btn btn-secondary" style="padding:5px 10px;"><i class="fas fa-edit"></i></button>
                        </td>
                    </tr>
            `);
            });

            // Update Inventory Dropdown
            const select = document.getElementById('inv-supplier-id');
            select.innerHTML = '<option value="">Select Supplier...</option>';
            suppliers.forEach(s => {
                select.insertAdjacentHTML('beforeend', `<option value="${s.supplier_id}">${s.name}</option>`);
            });
        }
    } catch (err) {
        showToast("Failed to load suppliers", "error");
    }
}

document.getElementById('supplier-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        name: document.getElementById('s-name').value,
        contact_number: document.getElementById('s-phone').value,
        email: document.getElementById('s-email').value,
        address: document.getElementById('s-address').value
    };

    try {
        const res = await fetch(`${API_BASE}/api/admin/suppliers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast("Supplier saved!", "success");
            closeModal('add-supplier-modal');
            loadSuppliers();
        }
    } catch (err) {
        showToast("Error saving supplier", "error");
    }
});

// --- CATEGORY MASTER LOGIC ---

async function loadCategories() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/categories`);
        const data = await res.json();
        if (data.status === 'success') {
            const categories = data.categories;
            allCategories = categories;
            const tbody = document.getElementById('categoryBody');
            tbody.innerHTML = '';
            categories.forEach(c => {
                tbody.insertAdjacentHTML('beforeend', `
            <tr>
                        <td>#${c.category_id}</td>
                        <td><strong>${c.category_name}</strong></td>
                        <td>
                            <button onclick="viewCategoryPlants(${c.category_id}, '${c.category_name}')" class="btn btn-secondary" style="padding:5px 10px; margin-right:5px;" title="View Plants"><i class="fas fa-eye"></i> View Plants</button>
                            <button onclick="deleteCategory(${c.category_id}, '${c.category_name}')" class="btn btn-danger" style="padding:5px 10px;" title="Delete Category"><i class="fas fa-trash"></i></button>
                        </td>
                    </tr>
            `);
            });
        }
    } catch (err) {
        showToast("Failed to load categories", "error");
    }
}

document.getElementById('category-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        category_name: document.getElementById('c-name').value
    };

    try {
        const res = await fetch(`${API_BASE}/api/admin/categories`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast("Category saved!", "success");
            closeModal('add-category-modal');
            loadCategories();
        }
    } catch (err) {
        showToast("Error saving category", "error");
    }
});

async function deleteCategory(id, name) {
    if (!confirm(`Are you sure you want to delete category '${name}' ? `)) return;
    try {
        const res = await fetch(`${API_BASE}/api/admin/categories/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(data.message, "success");
            loadCategories();
        } else {
            showToast(data.message, "error");
        }
    } catch (err) {
        showToast("Server error deleting category", "error");
    }
}

async function viewCategoryPlants(categoryId, categoryName) {
    document.getElementById('vcp-category-name').textContent = categoryName;
    const tbody = document.getElementById('vcp-plants-body');
    const loading = document.getElementById('vcp-loading');
    const empty = document.getElementById('vcp-empty');

    tbody.innerHTML = '';
    loading.style.display = 'block';
    empty.style.display = 'none';
    openModal('view-category-plants-modal');

    try {
        const res = await fetch(`${API_BASE}/api/admin/categories/${categoryId}/plants`);
        const data = await res.json();

        loading.style.display = 'none';

        if (data.status === 'success') {
            if (data.plants.length === 0) {
                empty.style.display = 'block';
            } else {
                data.plants.forEach(p => {
                    tbody.insertAdjacentHTML('beforeend', `
                        <tr>
                            <td>#${p.plant_id}</td>
                            <td><strong>${p.plant_name}</strong></td>
                            <td>${p.stock}</td>
                            <td>$${p.price.toFixed(2)}</td>
                            <td><button onclick="viewPlantImage('${p.plant_name}', '${p.image_url || ''}')" class="btn" style="padding: 5px 10px; background:#27ae60; color:#fff;" title="View Image"><i class="fas fa-image"></i></button></td>
                        </tr>
                    `);
                });
            }
        } else {
            showToast(data.message || "Failed to load plants", "error");
        }
    } catch (err) {
        loading.style.display = 'none';
        showToast("Error connecting to server", "error");
    }
}

// --- CUSTOMER MASTER LOGIC ---

async function loadCustomers() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/customers`);
        const data = await res.json();
        if (data.status === 'success') {
            const customers = data.customers;
            const tbody = document.getElementById('customerBody');
            tbody.innerHTML = '';
            customers.forEach(c => {
                const statusBadge = c.is_online ? '<span class="badge badge-success">Online</span>' : '<span class="badge" style="background:var(--gray-300); color:var(--gray-700)">Offline</span>';
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td>#${c.id}</td>
                        <td><strong>${c.email}</strong></td>
                        <td>${c.last_login || 'Never'}</td>
                        <td>${statusBadge}</td>
                        <td>${c.purchases || 0}</td>
                        <td>$${(c.amount_purchased || 0).toFixed(2)}</td>
                    </tr>
                `);
            });
        }
    } catch (err) {
        showToast("Failed to load customers", "error");
    }
}

// --- INVENTORY MANAGEMENT ---

async function searchPlantsForInventory(type) {
    const queryInput = type === 'ADD' ? 'inv-plant-search' : 'inv-plant-search-rem';
    const dropdownId = type === 'ADD' ? 'inv-search-dropdown-add' : 'inv-search-dropdown-rem';
    const query = document.getElementById(queryInput).value;
    const dropdown = document.getElementById(dropdownId);

    if (query.length < 2) { dropdown.style.display = 'none'; return; }

    const res = await fetch(`${API_BASE}/api/admin/search-plants?q=${query}`);
    const data = await res.json();

    dropdown.innerHTML = '';
    dropdown.style.display = 'block';

    data.plants.forEach(p => {
        const item = document.createElement('div');
        item.className = 'search-item';
        item.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px;">
                <img src="${p.image_url ? fixImageUrl(p.image_url) : 'https://via.placeholder.com/40'}" style="width:30px; height:30px; border-radius:4px; object-fit:cover;">
                <span>${p.plant_name}</span> <small>Stock: ${p.stock}</small>
            </div>
        `;
        item.onclick = () => {
            const suffix = type === 'ADD' ? 'add' : 'rem';
            document.getElementById(queryInput).value = p.plant_name;
            document.getElementById('inv-plant-id-' + suffix).value = p.plant_id;

            // Show Preview
            const previewDiv = document.getElementById('inv-preview-' + suffix);
            const previewImg = document.getElementById('inv-img-' + suffix);
            const previewName = document.getElementById('inv-name-' + suffix);

            previewDiv.style.display = 'flex';
            previewImg.src = p.image_url ? fixImageUrl(p.image_url) : 'https://via.placeholder.com/60';
            previewName.textContent = p.plant_name;
            previewName.dataset.id = p.plant_id;

            dropdown.style.display = 'none';
        };
        dropdown.appendChild(item);
    });
}

function triggerChangeImageFromInventory(suffix) {
    const suffixLower = suffix.toLowerCase();
    const plantId = document.getElementById('inv-name-' + suffixLower).dataset.id;
    const plantName = document.getElementById('inv-name-' + suffixLower).textContent;
    openChangeImageModal(plantId, plantName);
}

document.getElementById('add-stock-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const source = document.getElementById('inv-source-add').value;
    const supplierId = source === 'New Supplies' ? document.getElementById('inv-supplier-id').value : null;

    const payload = {
        plant_id: document.getElementById('inv-plant-id-add').value,
        supplier_id: supplierId,
        quantity: document.getElementById('inv-qty-add').value,
        notes: document.getElementById('inv-notes-add').value,
        bill_date: document.getElementById('inv-bill-date-add').value,
        bill_no: document.getElementById('inv-bill-no-add').value
    };

    if (source === 'New Supplies' && !supplierId) { showToast("Please select a supplier", "error"); return; }
    if (source === 'Customer Return' && !payload.notes) payload.notes = 'Customer Return';
    if (!payload.plant_id) { showToast("Please select a plant from search", "error"); return; }

    const res = await fetch(`${API_BASE}/api/admin/inventory/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === 'success') {
        showToast(data.message, "success");
        e.target.reset();
        loadInventory();
        loadTransactions();
    }
});

document.getElementById('remove-stock-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        plant_id: document.getElementById('inv-plant-id-rem').value,
        quantity: document.getElementById('inv-qty-rem').value,
        notes: document.getElementById('inv-notes-rem').value,
        bill_date: document.getElementById('inv-bill-date-rem').value,
        bill_no: document.getElementById('inv-bill-no-rem').value
    };

    if (!payload.plant_id) { showToast("Please select a plant from search", "error"); return; }

    const res = await fetch(`${API_BASE}/api/admin/inventory/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === 'success') {
        showToast(data.message, "success");
        e.target.reset();
        loadInventory();
        loadTransactions();
    }
});

async function loadTransactions() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/inventory/transactions`);
        const data = await res.json();
        if (data.status === 'success') {
            const tbody = document.getElementById('transactionsBody');
            tbody.innerHTML = '';
            data.transactions.forEach(t => {
                const typeBadge = t.type === 'ADD' ? '<span class="badge badge-success">IN</span>' : '<span class="badge badge-danger">OUT</span>';
                const billInfo = t.bill_no ? `${t.bill_no} (${t.bill_date || 'No Date'})` : '-';
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td>${t.date}</td>
                        <td><strong>${t.plant_name}</strong></td>
                        <td>${typeBadge}</td>
                        <td>${t.quantity}</td>
                        <td>${t.supplier_name}</td>
                        <td>${billInfo}</td>
                        <td><small>${t.notes}</small></td>
                    </tr>
                `);
            });
        }
    } catch (err) {
        console.error("Failed to load transactions", err);
    }
}

// --- VARIETIES TAB ---

async function searchPlantsForVariety() {
    const query = document.getElementById('variety-plant-search').value;
    const dropdown = document.getElementById('variety-search-dropdown');
    if (query.length < 2) { dropdown.style.display = 'none'; return; }

    const res = await fetch(`${API_BASE}/api/admin/search-plants?q=${query}`);
    const data = await res.json();

    dropdown.innerHTML = '';
    dropdown.style.display = 'block';
    data.plants.forEach(p => {
        const item = document.createElement('div');
        item.className = 'search-item';
        item.innerHTML = `<span>${p.plant_name}</span>`;
        item.onclick = async () => {
            document.getElementById('vp-id').value = p.plant_id;
            document.getElementById('vp-name').textContent = p.plant_name;
            document.getElementById('vp-desc').textContent = p.description || 'No description';
            document.getElementById('vp-stock').textContent = `Stock: ${p.stock}`;
            document.getElementById('selected-plant-info').style.display = 'block';
            document.getElementById('varieties-results').style.display = 'none';
            dropdown.style.display = 'none';
            document.getElementById('variety-plant-search').value = p.plant_name;
            await loadExistingVarieties(p.plant_id);
        };
        dropdown.appendChild(item);
    });
}

async function loadExistingVarieties(plantId) {
    const container = document.getElementById('existing-varieties-container');
    const list = document.getElementById('existing-varieties-list');
    try {
        const res = await fetch(`${API_BASE}/api/admin/varieties/existing/${plantId}`);
        const data = await res.json();

        if (data.status === 'success' && data.varieties.length > 0) {
            list.innerHTML = data.varieties.map(v =>
                `<span class="badge" style="background:var(--gray-200); color:var(--gray-800); border:1px solid var(--gray-300);">${v.variety_name}</span>`
            ).join('');
            container.style.display = 'block';
        } else {
            list.innerHTML = '<span style="font-size:0.85rem; color:var(--gray-500);">No varieties exist for this plant yet.</span>';
            container.style.display = 'block';
        }
    } catch (err) {
        console.error("Error loading existing varieties", err);
        container.style.display = 'none';
    }
}

async function generateVarieties() {
    const name = document.getElementById('vp-name').textContent;
    const list = document.getElementById('varieties-list');
    list.innerHTML = '<p>AI is thinking...</p>';
    document.getElementById('varieties-results').style.display = 'block';

    try {
        const res = await fetch(`${API_BASE}/api/admin/suggest-varieties`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plant_name: name })
        });
        const data = await res.json();

        list.innerHTML = '';
        data.varieties.forEach(v => {
            list.insertAdjacentHTML('beforeend', `
                <label style="display:flex; gap:10px; background:white; padding:10px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
                    <input type="checkbox" value="${v}" checked class="v-check"> ${v}
                </label>
            `);
        });
    } catch (err) {
        showToast("AI generation failed", "error");
    }
}

async function saveSelectedVarieties() {
    const plantId = document.getElementById('vp-id').value;
    const plantName = document.getElementById('vp-name').textContent;
    const selected = Array.from(document.querySelectorAll('.v-check:checked')).map(c => c.value);

    if (selected.length === 0) return;

    try {
        const res = await fetch(`${API_BASE}/api/admin/add-varieties`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plant_id: parseInt(plantId),
                plant_name: plantName,
                selected_varieties: selected
            })
        });
        const data = await res.json();
        showToast(`Added ${data.added} new varieties!`, "success");
        document.getElementById('varieties-results').style.display = 'none';

        // Refresh existing varieties list to show the new ones immediately
        await loadExistingVarieties(plantId);
    } catch (err) {
        showToast("Error saving varieties", "error");
    }
}

// --- UTILS ---

function showToast(message, type = "success") {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// --- SALES MASTER LOGIC ---

async function loadOrders() {
    try {
        console.log("Fetching orders from:", `${API_BASE}/api/admin/orders`);
        const res = await fetch(`${API_BASE}/api/admin/orders`);
        const data = await res.json();
        const tbody = document.getElementById('ordersBody');
        if (!tbody) { console.error("ordersBody element not found!"); return; }
        tbody.innerHTML = '';

        if (data.status === 'success') {
            if (!data.orders || data.orders.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No orders found.</td></tr>';
                return;
            }

            data.orders.forEach(o => {
                try {
                    let badgeColor = '#666';
                    if (o.order_status === 'Processing') badgeColor = '#f39c12';
                    else if (o.order_status === 'Shipped') badgeColor = '#3498db';
                    else if (o.order_status === 'Delivered') badgeColor = '#27ae60';
                    else if (o.order_status === 'Cancelled') badgeColor = '#e74c3c';

                    const amount = typeof o.total_amount === 'number' ? o.total_amount : 0;
                    const paid = typeof o.amount_paid === 'number' ? o.amount_paid : 0;
                    const balance = typeof o.balance === 'number' ? o.balance : 0;
                    const dateDisplay = (o.order_date && o.order_date !== 'N/A') ? o.order_date.split(' ')[0] : 'N/A';

                    tbody.innerHTML += `
                        <tr>
                            <td><b>${o.order_group_id || 'N/A'}</b></td>
                            <td>${o.customer_name || 'Guest'}</td>
                            <td>${dateDisplay}</td>
                            <td>₹${amount.toFixed(2)}</td>
                            <td>₹${paid.toFixed(2)}</td>
                            <td style="color:${balance > 0 ? 'var(--danger)' : 'var(--success)'}; font-weight:600;">₹${balance.toFixed(2)}</td>
                            <td>${o.payment_method || 'N/A'}</td>
                            <td>
                                <span style="background:${badgeColor}; color:white; padding:4px 8px; border-radius:12px; font-size:0.8rem;">
                                    ${o.order_status || 'Unknown'}
                                </span>
                            </td>
                            <td>
                                <div style="display: flex; gap: 8px;">
                                    <button class="btn btn-secondary" onclick="viewOrder('${o.order_group_id}')" style="padding: 4px 8px;">
                                        <i class="fas fa-eye"></i> View
                                    </button>
                                    <button class="btn btn-secondary" onclick="openSalesPayment('${o.order_group_id}', ${balance})" ${balance <= 0 ? 'disabled' : ''} style="padding: 4px 8px;">
                                        <i class="fas fa-money-bill-wave"></i> Pay
                                    </button>
                                </div>
                            </td>
                        </tr>
                    `;
                } catch (loopErr) {
                    console.error("Error rendering order row:", loopErr, o);
                }
            });
        } else {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:red;">Error: ${data.message || 'Unknown error'}</td></tr>`;
        }
    } catch (e) {
        console.error('Error fetching orders:', e);
        const tbody = document.getElementById('ordersBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:red;">Failed to connect to server.</td></tr>';
    }
}

async function viewOrder(groupId) {
    if (!groupId || groupId === 'N/A') {
        showToast("Invalid Order ID", "error");
        return;
    }
    try {
        console.log("Viewing order:", groupId);
        const res = await fetch(`${API_BASE}/api/admin/orders/${groupId}`);
        const data = await res.json();

        if (data.status === 'success') {
            const sum = data.summary;
            document.getElementById('v-order-id').textContent = sum.order_group_id;
            document.getElementById('v-customer').textContent = sum.customer_name;
            document.getElementById('v-email').textContent = sum.customer_email;
            document.getElementById('v-payment').textContent = sum.payment_method;
            document.getElementById('v-address').textContent = sum.shipping_address;

            document.getElementById('edit-order-group-id').value = sum.order_group_id;
            document.getElementById('edit-order-status').value = sum.order_status;
            document.getElementById('edit-tracking-no').value = sum.tracking_number;
            document.getElementById('edit-delivery-date').value = sum.delivery_date || '';
            document.getElementById('edit-order-notes').value = sum.notes;

            const tbody = document.getElementById('v-order-items');
            tbody.innerHTML = '';
            data.items.forEach(i => {
                tbody.innerHTML += `
                    <tr>
                        <td>${i.plant_name}</td>
                        <td>x${i.quantity}</td>
                        <td>₹${i.total_amount.toFixed(2)}</td>
                    </tr>
                `;
            });

            openModal('order-modal');
        } else {
            showToast(data.message || "Order not found", "error");
        }
    } catch (e) {
        console.error("View order error:", e);
        showToast("Failed to fetch order details", "error");
    }
}

async function updateOrderStatusAction() {
    const groupId = document.getElementById('edit-order-group-id').value;
    const payload = {
        order_status: document.getElementById('edit-order-status').value,
        tracking_number: document.getElementById('edit-tracking-no').value,
        delivery_date: document.getElementById('edit-delivery-date').value,
        notes: document.getElementById('edit-order-notes').value
    };

    try {
        const res = await fetch(`${API_BASE}/api/admin/orders/${groupId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(data.message, "success");
            closeModal('order-modal');
            loadOrders();
        } else {
            showToast(data.message, "error");
        }
    } catch (e) {
        showToast("Failed to update order", "error");
    }
}

// --- PURCHASE MASTER LOGIC (TABBED) ---

function switchPurchaseTab(tabId, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    document.querySelectorAll('.purchase-tab-content').forEach(c => c.classList.remove('active'));
    const content = document.getElementById(`p-tab-${tabId}`);
    if (content) content.classList.add('active');

    if (tabId === 'bill-list') loadPurchases();
    if (tabId === 'balance-bills') loadBalanceBills();
    if (tabId === 'add-bill') preparePurchaseForm();
    if (tabId === 'make-payment') preparePaymentForm();
}

async function loadPurchases() {
    const tbody = document.getElementById('purchasesBody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Loading purchases...</td></tr>';

    try {
        const res = await fetch(`${API_BASE}/api/admin/purchases`);
        const data = await res.json();
        if (data.status === 'success') {
            tbody.innerHTML = '';
            window.allPurchases = data.purchases;
            data.purchases.forEach(p => {
                const statusClass = p.status.toLowerCase();
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td style="font-weight:600;">${p.bill_no}</td>
                        <td>${p.supplier_name}</td>
                        <td>${p.bill_date}</td>
                        <td>₹${p.total_amount.toFixed(2)}</td>
                        <td>₹${p.amount_paid.toFixed(2)}</td>
                        <td style="color:${p.balance > 0 ? 'var(--danger)' : 'var(--success)'}; font-weight:600;">
                            ₹${p.balance.toFixed(2)}
                        </td>
                        <td><span class="status-badge status-${statusClass}">${p.status}</span></td>
                        <td>
                            <button class="btn btn-secondary" onclick="openPaymentForBill(${p.purchase_id})" ${p.balance <= 0 ? 'disabled' : ''} style="padding: 4px 8px;">
                                <i class="fas fa-money-bill-wave"></i> Pay
                            </button>
                        </td>
                    </tr>
                `);
            });
        }
    } catch (err) {
        showToast("Failed to load purchases", "error");
    }
}

async function loadBalanceBills() {
    const tbody = document.getElementById('balanceBillsBody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Loading balance bills...</td></tr>';

    try {
        const res = await fetch(`${API_BASE}/api/admin/purchases`);
        const data = await res.json();
        if (data.status === 'success') {
            tbody.innerHTML = '';
            window.allPurchases = data.purchases;
            const balanceBills = data.purchases.filter(p => p.balance > 0);
            if (balanceBills.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">No balance bills found.</td></tr>';
                return;
            }
            balanceBills.forEach(p => {
                const statusClass = p.status.toLowerCase();
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td style="font-weight:600;">${p.bill_no}</td>
                        <td>${p.supplier_name}</td>
                        <td>${p.bill_date}</td>
                        <td>₹${p.total_amount.toFixed(2)}</td>
                        <td>₹${p.amount_paid.toFixed(2)}</td>
                        <td style="color:var(--danger); font-weight:600;">
                            ₹${p.balance.toFixed(2)}
                        </td>
                        <td><span class="status-badge status-${statusClass}">${p.status}</span></td>
                        <td>
                            <button class="btn btn-secondary" onclick="openPaymentForBill(${p.purchase_id})" style="padding: 4px 8px;">
                                <i class="fas fa-money-bill-wave"></i> Pay
                            </button>
                        </td>
                    </tr>
                `);
            });
        }
    } catch (err) {
        showToast("Failed to load balance bills", "error");
    }
}

async function preparePurchaseForm() {
    const supplierSelect = document.getElementById('purchase-supplier');
    const itemsContainer = document.getElementById('purchase-items-container');
    const pForm = document.getElementById('purchaseForm');

    if (pForm) pForm.reset();
    if (document.getElementById('purchase-date')) document.getElementById('purchase-date').valueAsDate = new Date();
    if (itemsContainer) itemsContainer.innerHTML = '';

    try {
        const [supRes, plantRes] = await Promise.all([
            fetch(`${API_BASE}/api/admin/suppliers`),
            fetch(`${API_BASE}/api/admin/inventory`)
        ]);
        const supData = await supRes.json();
        const plantData = await plantRes.json();

        if (supplierSelect) {
            supplierSelect.innerHTML = '<option value="">Select Supplier</option>';
            supData.suppliers.forEach(s => {
                supplierSelect.insertAdjacentHTML('beforeend', `<option value="${s.supplier_id}">${s.supplier_name}</option>`);
            });
        }

        window.allPlantsForPurchase = plantData.plants;
        addPurchaseRow();
    } catch (err) {
        showToast("Failed to load selectors", "error");
    }
}

function openPaymentForBill(id) {
    const btn = document.getElementById('record-payment-tab-btn');
    switchPurchaseTab('make-payment', btn);
    setTimeout(() => {
        const selector = document.getElementById('payment-purchase-id');
        if (selector) {
            selector.value = id;
            updatePaymentBalance();
        }
    }, 100);
}

async function preparePaymentForm() {
    const selector = document.getElementById('payment-purchase-id');
    if (!selector) return;
    selector.innerHTML = '<option value="">Select Bill (Unpaid)</option>';

    if (!window.allPurchases) {
        const res = await fetch(`${API_BASE}/api/admin/purchases`);
        const data = await res.json();
        window.allPurchases = data.purchases;
    }

    const unpaid = (window.allPurchases || []).filter(p => p.balance > 0);
    unpaid.forEach(p => {
        selector.insertAdjacentHTML('beforeend', `<option value="${p.purchase_id}">${p.bill_no} (${p.supplier_name}) - ₹${p.balance}</option>`);
    });

    const payForm = document.getElementById('paymentForm');
    if (payForm) payForm.reset();
    if (document.getElementById('payment-balance-display')) document.getElementById('payment-balance-display').value = '';
}

function updatePaymentBalance() {
    const selector = document.getElementById('payment-purchase-id');
    if (!selector) return;
    const id = parseInt(selector.value);
    const bill = (window.allPurchases || []).find(p => p.purchase_id === id);
    if (bill) {
        document.getElementById('payment-balance-display').value = `₹${bill.balance.toFixed(2)}`;
        document.getElementById('payment-amount').value = bill.balance.toFixed(2);
        document.getElementById('payment-amount').max = bill.balance;
    } else {
        document.getElementById('payment-balance-display').value = '';
    }
}

function addPurchaseRow() {
    const container = document.getElementById('purchase-items-container');
    if (!container) return;
    const plantOptions = (window.allPlantsForPurchase || []).map(p => `<option value="${p.plant_id}">${p.plant_name}</option>`).join('');

    const row = `
        <div class="purchase-item-row" style="display: grid; grid-template-columns: 2fr 1fr 1.5fr auto; gap: 0.5rem; margin-bottom: 0.5rem; align-items: end;">
            <div class="form-group" style="margin:0;">
                <select class="form-control plant-selector" required>
                    <option value="">Select Plant</option>
                    ${plantOptions}
                </select>
            </div>
            <div class="form-group" style="margin:0;">
                <input type="number" class="form-control item-qty" required min="1" placeholder="Qty">
            </div>
            <div class="form-group" style="margin:0;">
                <input type="number" class="form-control item-price" required step="0.01" placeholder="Unit Price">
            </div>
            <button type="button" class="btn btn-danger" style="padding: 8px;" onclick="removePurchaseRow(this)">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', row);
}

function removePurchaseRow(btn) {
    const rows = document.querySelectorAll('.purchase-item-row');
    if (rows.length > 1) {
        btn.parentElement.remove();
    } else {
        showToast("At least one item is required", "warning");
    }
}

document.getElementById('paymentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const purchaseId = document.getElementById('payment-purchase-id').value;
    const amount = document.getElementById('payment-amount').value;

    if (!purchaseId || !amount) {
        showToast("Please select a bill and enter an amount", "warning");
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/admin/purchases/record-payment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                purchase_id: parseInt(purchaseId),
                amount: parseFloat(amount)
            })
        });
        const data = await res.json();
        if (data.status === 'success') {
            showToast("Payment recorded successfully", "success");
            e.target.reset();
            document.getElementById('payment-balance-display').value = '';

            // Re-fetch Data
            const resPurchases = await fetch(`${API_BASE}/api/admin/purchases`);
            const pData = await resPurchases.json();
            window.allPurchases = pData.purchases;

            // Update UI
            if (document.getElementById('p-tab-bill-list')?.classList.contains('active')) loadPurchases();
            if (document.getElementById('p-tab-balance-bills')?.classList.contains('active')) loadBalanceBills();
            if (document.getElementById('p-tab-make-payment')?.classList.contains('active')) preparePaymentForm();

            // also trigger UI update for lists if active
            loadPurchases();
        } else {
            showToast(data.message || "Failed to record payment", "error");
        }
    } catch (err) {
        showToast("Server error processing payment", "error");
    }
});

// --- SALES MASTER LOGIC (PAYMENT) ---

function openSalesPayment(groupId, balance) {
    document.getElementById('sp-order-group-id').value = groupId;
    document.getElementById('sp-balance-display').value = `₹${balance.toFixed(2)}`;
    document.getElementById('sp-amount').value = balance.toFixed(2);
    document.getElementById('sp-amount').max = balance;

    openModal('sales-payment-modal');
}

document.getElementById('salesPaymentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const groupId = document.getElementById('sp-order-group-id').value;
    const amount = document.getElementById('sp-amount').value;

    if (!groupId || !amount) {
        showToast("Missing details", "error");
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/admin/orders/record-payment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                order_group_id: groupId,
                amount: parseFloat(amount)
            })
        });
        const data = await res.json();

        if (data.status === 'success') {
            showToast("Sales payment recorded perfectly!", "success");
            closeModal('sales-payment-modal');
            e.target.reset();
            loadOrders(); // Refresh table
        } else {
            showToast(data.message || "Failed to record payment", "error");
        }
    } catch (err) {
        showToast("Server error recording sales payment", "error");
    }
});

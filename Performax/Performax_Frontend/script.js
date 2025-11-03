const baseURL = "http://127.0.0.1:8000";
const token = localStorage.getItem("token");

// --- UTILS ---
function getAuthHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

// --- AUTH FUNCTIONS (index.html) ---

function showRegister() {
    document.getElementById("login-section").classList.add("hidden");
    document.getElementById("register-section").classList.remove("hidden");
    document.getElementById("form-title").textContent = "Performax Register";
}

function showLogin() {
    document.getElementById("register-section").classList.add("hidden");
    document.getElementById("login-section").classList.remove("hidden");
    document.getElementById("form-title").textContent = "Performax Login";
}

async function login() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const messageEl = document.getElementById("message");
    messageEl.textContent = "Logging in...";
    messageEl.style.color = "#555";

    try {
        const res = await fetch(`${baseURL}/token`, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ username, password }),
        });
        const data = await res.json(); // This 'data' object has the role
        if (!res.ok) throw new Error(data.detail);

        localStorage.setItem("token", data.access_token);
        
        const userRole = data.role; 
        
        // This is for debugging
        console.log("LOGIN SUCCESSFUL. User Role:", userRole);

        messageEl.textContent = "✅ Login successful! Redirecting...";
        messageEl.style.color = "green";

        // This logic checks the role and redirects
        if (userRole === 'admin' || userRole === 'faculty') {
            window.location.href = "admin_dashboard.html";
        } else {
            window.location.href = "dashboard.html";
        }
    } catch (err) {
        messageEl.textContent = `❌ ${err.message || "Login failed"}`;
        messageEl.style.color = "red";
    }
}

async function register() {
    const messageEl = document.getElementById("message");
    messageEl.textContent = "Public registration is disabled. Please contact an administrator to create an account.";
    messageEl.style.color = "orange";
}


// --- COMMON DASHBOARD FUNCTIONS ---
function checkAuth() {
    if (!token) {
        window.location.href = "index.html";
    }
}

function logout() {
    localStorage.removeItem("token");
    window.location.href = "index.html";
}

// --- STUDENT DASHBOARD (dashboard.html) ---
async function fetchStudentData() {
    if (!token) return;
    try {
        const [meRes, summaryRes] = await Promise.all([
            fetch(`${baseURL}/me`, { headers: getAuthHeaders() }),
            fetch(`${baseURL}/dashboard/summary`, { headers: getAuthHeaders() })
        ]);
        if (meRes.status === 401 || summaryRes.status === 401) return logout();

        const userData = await meRes.json();
        const summaryData = await summaryRes.json();
        
        const displayName = userData.full_name || userData.username;
        document.getElementById("welcome-message").textContent = `Hi, ${displayName} 👋`;
        document.getElementById("user-name-welcome").textContent = displayName;
        document.getElementById("avg-grade").textContent = `Average Grade: ${summaryData.my_average ? summaryData.my_average.toFixed(2) : 'N/A'}`;
        
        // --- THIS IS THE UPDATED FALLBACK LIST ---
        const defaultSubjects = ['HCI101', 'Networking', 'Information Management', 'Intermediate Programming'];
        const subjects = summaryData.my_subjects.length > 0 ? summaryData.my_subjects : defaultSubjects;
        
        populatePredictionForm(subjects);
    } catch (err) {
        console.error("Fetch Student Data Error:", err);
    }
}

function populatePredictionForm(subjects) {
    const form = document.getElementById('grade-inputs');
    if (!form) return;
    form.innerHTML = '';
    subjects.forEach(subject => {
        form.innerHTML += `
            <div>
                <label for="grade-${subject}">${subject} Grade (0-100):</label>
                <input type="number" id="grade-${subject}" class="form-input" min="0" max="100" value="75" required>
            </div>
        `;
    });
}

async function runPrediction() {
    const predictionForm = document.getElementById('predictionForm');
    const grades = {};
    const gradeInputs = document.getElementById('grade-inputs').querySelectorAll('input[type="number"]');

    gradeInputs.forEach(input => {
        const subject = input.id.replace('grade-', '');
        grades[subject] = parseFloat(input.value);
    });

    const payload = {
        grades: grades,
        attendance_pct: parseFloat(predictionForm.querySelector('#attendance').value),
        study_hours_per_week: parseFloat(predictionForm.querySelector('#study-hours').value)
    };

    const resultsDiv = document.getElementById('prediction-results');
    resultsDiv.innerHTML = '<p>Running prediction...</p>';
    
    try {
        const [predictRes, recommendRes] = await Promise.all([
            fetch(`${baseURL}/predict`, { method: "POST", headers: getAuthHeaders(), body: JSON.stringify(payload) }),
            fetch(`${baseURL}/recommend`, { method: "POST", headers: getAuthHeaders(), body: JSON.stringify(payload) })
        ]);

        if (predictRes.status === 401) return logout();

        const predictionData = await predictRes.json();
        const recommendationData = await recommendRes.json();
        displayPredictionResults(predictionData, recommendationData);

    } catch (err) {
        resultsDiv.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
    }
}

function displayPredictionResults(prediction, recommendation) {
    document.getElementById('prediction-results').innerHTML = `
        <h4>Risk Assessment: <span class="risk-${prediction.overall_risk}">${prediction.overall_risk.toUpperCase()}</span></h4>
        <p>This assessment predicts your overall academic standing.</p>
        <hr style="margin: 10px 0;"><h5>Subject Pass Probabilities:</h5>
        <ul>${Object.entries(prediction.subject_probabilities).map(([subj, prob]) => `<li>${subj}: <strong>${(prob * 100).toFixed(1)}%</strong> ${prob < 0.6 ? '<span class="risk-high">(At-Risk)</span>' : ''}</li>`).join('')}</ul>
        <hr style="margin: 10px 0;"><h5>Recommended Academic Track:</h5>
        <div class="track-recommendation">
            <strong>${recommendation.recommended_tracks[0].track}</strong> 
            <span style="font-weight:bold;">Compatibility: ${(recommendation.recommended_tracks[0].score * 100).toFixed(1)}%</span>
            <p style="font-size: 0.9em; color: #555; margin-top: 5px;">${recommendation.recommended_tracks[0].description}</p>
        </div>`;
}

async function sendChat() {
    const chatInput = document.getElementById("chatInput");
    const chatBox = document.getElementById("chatBox");
    if (!chatInput.value.trim()) return;

    const userMessage = chatInput.value;
    chatBox.innerHTML += `<p class="chat-user"><strong>You:</strong> ${userMessage}</p>`;
    chatInput.value = '';

    try {
        const res = await fetch(`${baseURL}/chat`, {
            method: "POST", headers: getAuthHeaders(), body: JSON.stringify({ message: userMessage })
        });
        const data = await res.json();
        chatBox.innerHTML += `<p class="chat-bot"><strong>Tutor:</strong> ${data.reply}</p>`;
    } catch (err) {
        chatBox.innerHTML += `<p class="chat-bot" style="color:red;"><strong>Tutor:</strong> Error: ${err.message}</p>`;
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- ADMIN PAGES ---

// Admin Dashboard (admin_dashboard.html)
async function fetchAdminData() {
    if (!token) return;
    try {
        const [meRes, analyticsRes] = await Promise.all([
            fetch(`${baseURL}/me`, { headers: getAuthHeaders() }),
            fetch(`${baseURL}/analytics/summary`, { headers: getAuthHeaders() })
        ]);

        if (meRes.status === 401 || analyticsRes.status === 401) return logout();
        
        const userData = await meRes.json();
        const analyticsData = await analyticsRes.json();

        // Welcome message
        const displayName = userData.full_name || userData.username;
        if(document.getElementById("welcome-message")) {
            document.getElementById("welcome-message").textContent = `Welcome back, ${displayName}!`;
        }

        // High risk count
        let highRiskCount = 0;
        if (analyticsData.pass_fail_counts) {
            Object.values(analyticsData.pass_fail_counts).forEach(subject => {
                if (subject.Fail && subject.Fail > (subject.Pass || 0)) {
                    highRiskCount++;
                }
            });
        }
        if(document.getElementById('high-risk-count')) {
            document.getElementById('high-risk-count').textContent = highRiskCount;
        }

    } catch (err) {
        console.error("Fetch Admin Data Error:", err);
    }
}


// User Management (user_management.html)
async function loadUsers() {
    try {
        const res = await fetch(`${baseURL}/users`, { headers: getAuthHeaders() });
        if (res.status === 401) return logout();
        const users = await res.json();

        const tableBody = document.getElementById('user-table-body');
        if (!tableBody) return;
        tableBody.innerHTML = '';

        users.forEach(user => {
            tableBody.innerHTML += `
                <tr>
                    <td>${user.id}</td>
                    <td>${user.full_name || 'N/A'}</td>
                    <td>${user.username}</td>
                    <td><span class="role-tag role-${user.role}">${user.role}</span></td>
                    <td class="actions">
                        <button class="edit-btn" onclick='openUserModal(${JSON.stringify(user)})'><i class="fas fa-edit"></i></button>
                        <button class="delete-btn" onclick="deleteUser(${user.id}, '${user.username}')"><i class="fas fa-trash"></i></button>
                    </td>
                </tr>
            `;
        });
    } catch (err) {
        console.error("Failed to load users:", err);
    }
}

function openUserModal(user = null) {
    const modal = document.getElementById('user-modal');
    const form = document.getElementById('user-form');
    form.reset();
    
    document.getElementById('modal-title').textContent = user ? 'Edit User' : 'Add New User';
    document.getElementById('modal-submit-btn').textContent = user ? 'Update User' : 'Create User';
    document.getElementById('user-id').value = user ? user.id : '';
    
    if (user) {
        document.getElementById('full_name').value = user.full_name || '';
        document.getElementById('username').value = user.username;
        document.getElementById('role').value = user.role;
        document.getElementById('username').disabled = true;
    } else {
        document.getElementById('username').disabled = false;
    }
    
    modal.style.display = 'block';
}

function closeUserModal() {
    document.getElementById('user-modal').style.display = 'none';
}

async function handleUserFormSubmit(event) {
    event.preventDefault();
    const userId = document.getElementById('user-id').value;
    const isEditing = !!userId;

    const payload = {
        full_name: document.getElementById('full_name').value,
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
        role: document.getElementById('role').value,
    };
    if (isEditing && !payload.password) {
        delete payload.password; // Don't send empty password on update
    }

    const url = isEditing ? `${baseURL}/users/${userId}` : `${baseURL}/users`;
    const method = isEditing ? 'PUT' : 'POST';

    try {
        const res = await fetch(url, { method, headers: getAuthHeaders(), body: JSON.stringify(payload) });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail);
        }
        closeUserModal();
        loadUsers();
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function deleteUser(userId, username) {
    if (confirm(`Are you sure you want to delete user: ${username}?`)) {
        try {
            const res = await fetch(`${baseURL}/users/${userId}`, { method: 'DELETE', headers: getAuthHeaders() });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail);
            }
            loadUsers();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }
}

// Reports Page (reports.html)
async function loadAnalytics() {
    try {
        const res = await fetch(`${baseURL}/analytics/summary`, { headers: getAuthHeaders() });
        if (res.status === 401) return logout();
        const data = await res.json();
        
        renderPassFailChart(data.pass_fail_counts);
        renderTrackEnrollmentChart(data.track_enrollment);
        renderGradeDistChart(data.grade_distribution);
    } catch (err) {
        console.error("Failed to load analytics:", err);
    }
}

function renderPassFailChart(passFailData) {
    const ctx = document.getElementById('passFailChart')?.getContext('2d');
    if (!ctx) return;
    const subjects = Object.keys(passFailData);
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: subjects,
            datasets: [
                { label: 'Pass', data: subjects.map(s => passFailData[s].Pass || 0), backgroundColor: 'rgba(75, 192, 192, 0.6)' },
                { label: 'Fail', data: subjects.map(s => passFailData[s].Fail || 0), backgroundColor: 'rgba(255, 99, 132, 0.6)' }
            ]
        },
        options: { scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } }, responsive: true }
    });
}

function renderTrackEnrollmentChart(trackData) {
    const ctx = document.getElementById('trackEnrollmentChart')?.getContext('2d');
    if (!ctx) return;
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: trackData.map(t => t.track),
            datasets: [{ data: trackData.map(t => t.count), backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'] }]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
    });
}

function renderGradeDistChart(gradeData) {
    const ctx = document.getElementById('gradeDistChart')?.getContext('2d');
    if (!ctx) return;
    const colors = ['rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)', 'rgba(255, 206, 86, 0.6)', 'rgba(75, 192, 192, 0.6)'];
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(gradeData),
            datasets: [{
                label: 'Average Grade',
                data: Object.values(gradeData).map(grades => grades.length ? grades.reduce((a, b) => a + b, 0) / grades.length : 0),
                backgroundColor: colors
            }]
        },
        options: { scales: { y: { beginAtZero: true, max: 100 } }, responsive: true }
    });
}


// --- ROUTER / PAGE LOAD ---
document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname;
    
    if (path.endsWith('dashboard.html')) {
        checkAuth();
        fetchStudentData();
    } else if (path.endsWith('admin_dashboard.html')) {
        checkAuth();
        fetchAdminData(); // This will fetch admin-specific data
    } else if (path.endsWith('user_management.html')) {
        checkAuth();
        loadUsers();
    } else if (path.endsWith('reports.html')) {
        checkAuth();
        loadAnalytics();
    }
});
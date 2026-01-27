# Deployment Guide: Tutorial Platform

This guide provides step-by-step instructions for hosting the Tutorial Platform on a server (e.g., VPS, Cloud instance) and how to manage the code via GitHub.

## Prerequisites
- **Server:** A minimal Ubuntu/Linux server (e.g., AWS EC2, DigitalOcean Droplet).
- **Software:** Node.js (v18+), npm, Git.
- **Tools:** `pm2` (Process Manager) for keeping the backend alive.

---

## Part 1: Preparing Your Code (Local Machine)

Before deploying, you must ensure your Frontend knows where to find your Backend on the real server.

### 1. Update API Endpoint
Currently, your frontend points to `http://192.168.5.138:5004` (Network IP). For production, this must match your **Server's Public IP** or **Domain Name**.

**Files to Change:**
- `frontend/src/components/Sidebar.jsx`
- `frontend/src/pages/TutorialPage.jsx`
- `frontend/src/pages/QuestionsPage.jsx`
- `frontend/src/pages/ResourcesPage.jsx`
- `frontend/src/pages/ExercisesPage.jsx` (and any other page fetching data)

**Action:**
Replace `http://192.168.5.138:5004` with:
- `http://YOUR_SERVER_IP:5004` (if using IP)
- `https://api.yourdomain.com` (if using a Domain/SSL)

### 2. Push to GitHub
If you haven't already, push your code to a GitHub repository.

```bash
# Initialize git if not done
git init

# Add remote (create a repo on GitHub first)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Commit and Push
git add .
git commit -m "Ready for deployment"
git push -u origin main
```

---

## Part 2: Server Setup (On the Remote Server)

Login to your server via SSH:
```bash
ssh user@YOUR_SERVER_IP
```

### 1. Install Node.js & NPM
```bash
sudo apt update
sudo apt install curl -y
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

### 2. Install PM2 (Process Manager)
PM2 keeps your backend running 24/7, even if the server restarts.
```bash
sudo npm install -g pm2
```

### 3. Clone Your Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

## Part 3: Running the Backend

### 1. Install Dependencies
```bash
cd backend
npm install
```

### 2. Configure Environment (.env)
Create a `.env` file if it's not tracked by Git.
```bash
nano .env
# Add: PORT=5004
```

### 3. Start Backend with PM2
```bash
pm2 start server.js --name "tutorial-backend"
pm2 save
pm2 startup
```
*Your backend is now running in the background on port 5004.*

---

## Part 4: Running the Frontend

### 1. Install Dependencies & Build
```bash
cd ../frontend
npm install
npm run build
```
This creates a `dist` folder containing the optimized production files.

### 2. Serve the Frontend
You have two options: Simple or Professional (Nginx).

#### Option A: Simple (Using 'serve')
Good for testing or simple usage.
```bash
sudo npm install -g serve
pm2 start serve --name "tutorial-frontend" -- -s dist -l 5173
```
*Access site at: http://YOUR_SERVER_IP:5173*

#### Option B: Professional (Using Nginx Reverse Proxy)
Recommended for real hosting on Port 80.

1. Install Nginx: `sudo apt install nginx -y`
2. Create config: `sudo nano /etc/nginx/sites-available/tutorial`

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP_OR_DOMAIN;

    location / {
        root /path/to/your/repo/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:5004;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```
3. Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tutorial /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```
*Access site at: http://YOUR_SERVER_IP*

---

## Summary of Commands for updates

When you change code on your local machine:
1. **Local:** `git push`
2. **Server:** `cd` to repo -> `git pull`
3. **Server Backend:** `pm2 restart tutorial-backend`
---

## Option 2: Free Cloud Hosting (GitHub Pages + Render)

**Important:** GitHub Pages only hosts **Static Sites** (Frontend). It cannot run your Node.js Backend.
To publish your full app for free, you must split it:
1.  **Backend:** Host on **Render.com** (Free Tier).
2.  **Frontend:** Host on **GitHub Pages** (Free).

### Step 1: Deploy Backend to Render
1.  Push your code to GitHub.
2.  Sign up at [render.com](https://render.com).
3.  Click **"New +"** -> **"Web Service"**.
4.  Connect your GitHub repo.
5.  Settings:
    *   **Root Directory:** `backend`
    *   **Build Command:** `npm install`
    *   **Start Command:** `node server.js`
6.  Click **Deploy**. Render will give you a URL (e.g., `https://my-api.onrender.com`).

### Step 2: Update Frontend API URL
Now that your backend is live, tell your frontend to talk to it.
1.  Open `frontend/src/components/Sidebar.jsx` (and other pages).
2.  Replace `http://192.168.5.138:5004` with your new Render URL: `https://my-api.onrender.com`.
3.  Commit and Push these changes to GitHub.

### Step 3: Deploy Frontend to GitHub Pages
1.  Open `frontend/package.json` and add:
    ```json
    "homepage": "https://Satviksalat.github.io/TutorialPlatform",
    ```
2.  Install the deployer tool:
    ```bash
    cd frontend
    npm install gh-pages --save-dev
    ```
3.  Add scripts to `frontend/package.json`:
    ```json
    "scripts": {
      "predeploy": "npm run build",
      "deploy": "gh-pages -d dist"
    }
    ```
4.  Deploy:
    ```bash
    npm run deploy
    ```

**Result:**
*   Your Website: `https://Satviksalat.github.io/TutorialPlatform`
*   Your API: `https://my-api.onrender.com`

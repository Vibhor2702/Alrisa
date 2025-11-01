# Deploying Alrisa to Cloudflare

## Deploy to Cloudflare Pages (Static) + Cloudflare Workers (API)

Since your app has a Python backend, we need to use Cloudflare Workers for the API. Here's how:

### Step 1: Install Wrangler (Cloudflare CLI)

```powershell
npm install -g wrangler
```

### Step 2: Login to Cloudflare

```powershell
wrangler login
```

This will open your browser for authentication.

### Step 3: Create Cloudflare Pages Project

```powershell
cd c:\Users\versu\OneDrive\Desktop\data
wrangler pages project create alrisa-automl
```

### Step 4: Deploy

```powershell
wrangler pages deploy .
```

### IMPORTANT: Python Backend Issue

⚠️ **Cloudflare Workers doesn't fully support Flask/Python yet with all ML libraries.**

### Better Solution: Hybrid Deployment

1. **Frontend on Cloudflare Pages** (Free, fast, global CDN)
2. **Backend on Render.com** (Free tier, 750 hours/month)

This gives you:
- ✅ Lightning-fast frontend on Cloudflare's CDN
- ✅ Working Python backend with all ML libraries
- ✅ Both are FREE
- ✅ Backend wakes up quickly (~30 seconds)

### Quick Hybrid Setup:

#### Part 1: Deploy Backend to Render

```powershell
# Just push to GitHub, then:
# 1. Go to https://render.com
# 2. Click "New +" -> "Web Service"
# 3. Connect your GitHub repo
# 4. Render auto-configures everything
# 5. Copy your backend URL (e.g., https://alrisa.onrender.com)
```

#### Part 2: Update Frontend for Backend URL

I'll update the code to point to your Render backend URL.

#### Part 3: Deploy Frontend to Cloudflare

```powershell
wrangler pages deploy ./static --project-name=alrisa-automl
```

### OR: Simplest Option - Use Render for Everything

Just use Render for both frontend and backend (still FREE):

```powershell
# Push to GitHub
git add .
git commit -m "Ready for deployment"
git push

# Then go to render.com and connect your repo
# That's it! Everything works.
```

**Which approach do you prefer?**
1. Hybrid (Cloudflare Pages + Render Backend) - Fastest
2. All Render - Simplest, one-click deploy
3. Try Cloudflare Workers Python (experimental, may not work with all libraries)

Let me know and I'll configure it for you!

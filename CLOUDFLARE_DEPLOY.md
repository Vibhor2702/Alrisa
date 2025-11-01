# Cloudflare Workers Deployment Guide

## Free Deployment with Cloudflare Workers

Cloudflare Workers is **completely FREE** for up to 100,000 requests/day with no sleep time!

### Prerequisites
1. A Cloudflare account (free): https://dash.cloudflare.com/sign-up
2. Node.js installed: https://nodejs.org/

### Step-by-Step Deployment

#### 1. Install Wrangler CLI
```powershell
npm install -g wrangler
```

#### 2. Login to Cloudflare
```powershell
wrangler login
```
This will open a browser for authentication.

#### 3. Deploy Your App
```powershell
cd c:\Users\versu\OneDrive\Desktop\data
wrangler deploy
```

That's it! Your app will be live at: `https://alrisa-automl.your-subdomain.workers.dev`

### Alternative: Cloudflare Pages (Frontend) + Free Backend Options

Since ML processing is heavy, here are 100% free alternatives:

#### Option 1: Vercel (Recommended)
- **Free Forever**: https://vercel.com
- No sleep time
- Perfect for Python apps
```powershell
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

#### Option 2: Render (Free Tier)
- **Free**: 750 hours/month (enough for most usage)
- Sleeps after 15 min inactivity (wakes in ~1 min)
- Steps:
  1. Go to https://render.com
  2. Connect GitHub
  3. Deploy (auto-configured)

#### Option 3: Fly.io (Free Tier)
- **Free**: 3 shared-cpu-1x VMs
- No sleep time
```powershell
# Install Fly CLI
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Deploy
fly launch
fly deploy
```

#### Option 4: PythonAnywhere (Free)
- **Free**: 1 web app
- No sleep
- Steps:
  1. Sign up: https://www.pythonanywhere.com
  2. Upload files
  3. Configure WSGI

### Comparison

| Platform | Cost | Sleep Time | Best For |
|----------|------|------------|----------|
| **Vercel** | Free | Never | Best overall choice |
| **Fly.io** | Free | Never | Great performance |
| Render | Free | 15min inactive | Good balance |
| Railway | $5/mo after trial | Never | If you can pay |
| PythonAnywhere | Free | Never | Simple setup |

### Recommended: Deploy to Vercel Now

1. Install Vercel:
```powershell
npm install -g vercel
```

2. Create `vercel.json` (I'll create this for you)

3. Deploy:
```powershell
vercel
```

Would you like me to set up the configuration for **Vercel** or **Fly.io**? Both are completely free with no sleep time!

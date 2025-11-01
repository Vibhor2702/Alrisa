# Deployment Guide for Alrisa

## Quick Deployment (Recommended: Railway)

### Step 1: Prepare Your Repository
1. Make sure all your code is committed to GitHub
2. Push to your repository

### Step 2: Deploy to Railway
1. Go to https://railway.app
2. Sign up/Login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `Alrisa` repository
6. Railway will automatically:
   - Detect it's a Python app
   - Install dependencies from requirements.txt
   - Start the app using the Procfile
7. Wait 2-3 minutes for deployment
8. Click "Generate Domain" to get your public URL
9. Done! Your app is live! 🚀

## Alternative: Deploy to Render

### Step 1: Prepare Your Repository
Same as above

### Step 2: Deploy to Render
1. Go to https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: alrisa-automl (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Wait 3-5 minutes for deployment
8. Your app will be live at the provided URL! 🎉

## Cloudflare Pages (Static Frontend Only)

⚠️ **Note**: Cloudflare Pages doesn't support Python backends directly. You have two options:

### Option A: Deploy Backend Separately
1. Deploy the Flask backend to Railway/Render (see above)
2. Get your backend URL (e.g., `https://your-app.railway.app`)
3. Update `static/script.js`:
   - Replace `/upload` with `https://your-app.railway.app/upload`
   - Replace `/train` with `https://your-app.railway.app/train`
   - Replace `/chat` with `https://your-app.railway.app/chat`
4. Deploy to Cloudflare Pages:
   - Go to Cloudflare Dashboard
   - Click "Pages" → "Create a project"
   - Connect your repository
   - Build settings: (leave empty)
   - Deploy

### Option B: Use Cloudflare Workers (Requires Rewrite)
This would require converting the Flask app to JavaScript/TypeScript for Cloudflare Workers. Not recommended unless you're familiar with Workers.

## Testing Locally First

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open browser to http://localhost:8080
```

## Environment Variables

No environment variables needed! The app works out of the box.

## Post-Deployment Checklist

- ✅ App loads without errors
- ✅ File upload works
- ✅ EDA displays correctly
- ✅ Model training completes
- ✅ Results show up
- ✅ Code download works
- ✅ Chat assistant responds

## Troubleshooting

### "Application Error" or "502 Bad Gateway"
- Check the logs in your hosting platform dashboard
- Verify requirements.txt has all dependencies
- Make sure Python version is compatible (3.11 recommended)

### Slow Performance
- Free tiers have limited resources
- Consider upgrading to paid tier for better performance
- PyCaret operations are CPU-intensive

### File Upload Fails
- Check file size limits (16MB max by default)
- Verify CSV format is correct
- Check platform storage limits

## Estimated Costs

- **Railway**: Free tier (500 hours/month), then $5/month
- **Render**: Free tier (750 hours/month), then $7/month
- **Cloudflare Pages**: Free (with separate backend)

## Support

If you encounter issues:
1. Check the hosting platform's logs
2. Review the README.md
3. Open an issue on GitHub

Happy deploying! 🚀

@echo off
echo ========================================
echo   Alrisa - Cloudflare Deployment
echo ========================================
echo.
echo IMPORTANT: Cloudflare Pages doesn't support Python backends.
echo.
echo Choose your deployment option:
echo.
echo 1. Deploy to Render.com (Recommended - FREE, works perfectly)
echo 2. Deploy to Vercel (Also FREE, great for Python)
echo 3. Deploy to Fly.io (FREE, no sleep time)
echo.
echo For Cloudflare, you would need to:
echo - Deploy backend to Render/Vercel/Fly.io
echo - Deploy static frontend to Cloudflare Pages
echo.
echo ========================================
echo Press any key to open deployment guide...
pause >nul
start https://render.com

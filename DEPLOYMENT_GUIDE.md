# Options Trading System - Deployment Guide

## Quick Deployment Options

### Option 1: Streamlit Cloud + Railway (Easiest & Free Tier Available)

#### Frontend (Streamlit Cloud - FREE)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Set:
   - Repository: your-repo
   - Branch: main
   - Main file path: frontend/app.py
6. Advanced settings → Add environment variable:
   ```
   API_URL = https://your-backend.railway.app
   ```
7. Deploy! Your app will be at: `https://yourapp.streamlit.app`

#### Backend (Railway - $5/month)
1. Visit [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repository
4. Set root directory: `/backend`
5. Add environment variables:
   ```
   PORT = 8000
   DATABASE_URL = (auto-generated)
   REDIS_URL = (auto-generated)
   ```
6. Deploy! URL will be: `https://your-api.railway.app`

### Option 2: Render.com (All-in-One - $7/month)
1. Push code to GitHub
2. Visit [render.com](https://render.com)
3. New → Blueprint
4. Connect GitHub repo
5. Use the `deploy/render.yaml` file
6. Everything deploys automatically!

### Option 3: DigitalOcean App Platform ($5-12/month)
```bash
# Install doctl CLI
brew install doctl  # Mac
# or
snap install doctl  # Linux

# Deploy
doctl apps create --spec .do/app.yaml
```

### Option 4: Self-Hosted VPS ($5-20/month)

#### Setup VPS (DigitalOcean, Linode, etc.)
```bash
# 1. SSH into server
ssh root@your-server-ip

# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Clone repository
git clone https://github.com/yourusername/OptionsTrader.git
cd OptionsTrader

# 4. Create .env file
cp .env.example .env
nano .env  # Edit with your credentials

# 5. Run with Docker Compose
docker-compose up -d

# 6. Setup Nginx (optional, for domain)
apt update && apt install nginx certbot python3-certbot-nginx
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/trading
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site and SSL
ln -s /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/
certbot --nginx -d yourdomain.com
systemctl reload nginx
```

### Option 5: Heroku (Simple but $$)

#### Create Procfile
```procfile
web: cd backend && uvicorn api.endpoints:app --host 0.0.0.0 --port $PORT
worker: cd frontend && streamlit run app.py --server.port $PORT
```

```bash
# Deploy
heroku create your-trading-app
heroku addons:create heroku-postgresql:mini
heroku addons:create heroku-redis:mini
git push heroku main
```

## Production Checklist

### Security
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/SSL
- [ ] Set up firewall rules
- [ ] Use strong passwords
- [ ] Enable 2FA where possible
- [ ] Regular security updates

### Performance
- [ ] Enable caching (Redis)
- [ ] Use CDN for static assets
- [ ] Optimize database queries
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure auto-scaling

### Reliability
- [ ] Set up automated backups
- [ ] Configure health checks
- [ ] Implement error logging (Sentry)
- [ ] Set up uptime monitoring
- [ ] Create disaster recovery plan

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy to Railway
        uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}

      - name: Deploy to Streamlit
        run: |
          # Streamlit auto-deploys from GitHub
          echo "Frontend will auto-deploy"
```

## Monitoring Setup

### Using UptimeRobot (Free)
1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. Add monitors:
   - API Health: `https://your-api.com/health`
   - Dashboard: `https://your-dashboard.com`
3. Set up alerts (email, SMS, Slack)

### Application Monitoring
```python
# Add to your backend
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Cost Comparison

| Platform | Backend | Frontend | Database | Redis | Total/month |
|----------|---------|----------|----------|-------|-------------|
| Streamlit + Railway | $5 | FREE | Included | Included | $5 |
| Render | $7 | Included | Included | Included | $7 |
| DigitalOcean VPS | $6 | Included | Included | Included | $6 |
| AWS (t3.micro) | $8 | $5 | $15 | $10 | $38 |
| Heroku | $7 | $7 | $9 | $15 | $38 |
| Google Cloud Run | ~$5 | ~$5 | $10 | $10 | ~$30 |

## Recommended: Start with Streamlit Cloud + Railway
1. **Fastest to deploy** (under 10 minutes)
2. **Free frontend hosting**
3. **Automatic SSL/HTTPS**
4. **Built-in monitoring**
5. **Easy to scale later**

## Environment Variables Template
```env
# Backend (.env)
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key

# Frontend (.streamlit/secrets.toml)
[api]
url = "https://your-backend.railway.app"
key = "your-api-key"

[database]
url = "postgresql://user:pass@host/db"
```

## Next Steps
1. Choose a deployment option based on your needs
2. Set up GitHub repository if not already done
3. Configure environment variables
4. Deploy backend first, get URL
5. Deploy frontend with backend URL
6. Set up monitoring
7. Configure custom domain (optional)

## Support & Troubleshooting
- Check logs: `docker-compose logs -f`
- Test health endpoint: `curl https://your-api.com/health`
- Verify environment variables are set
- Ensure ports are not blocked by firewall
- Check service status: `systemctl status your-service`

Remember: Start simple with Streamlit Cloud + Railway, then migrate to more complex solutions as you grow!
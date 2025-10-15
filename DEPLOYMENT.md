# Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step-by-Step Instructions

#### 1. Prepare Your Repository

```bash
git init
git add .
git commit -m "Initial commit: AI Tools & Frameworks Dashboard"
```

Create a new repository on GitHub and push:

```bash
git remote add origin https://github.com/yourusername/ai-tools-dashboard.git
git branch -M main
git push -u origin main
```

#### 2. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

#### 3. Handle Model Files

**Option A: Include in Repository (if < 100MB)**
```bash
python setup_models.py
git add models/*.pkl models/*.h5
git commit -m "Add trained models"
git push
```

**Option B: Train on Deployment**
- Models will need to be trained on first run
- Add setup commands in Streamlit Cloud settings
- May increase first-load time

#### 4. Configure Settings

In Streamlit Cloud app settings:
- **Python version**: 3.9 or higher
- **Advanced settings** > Add packages:
  ```
  python -m spacy download en_core_web_sm
  ```

### Environment Variables

No environment variables required for basic deployment.

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

RUN python setup_models.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t ai-tools-dashboard .
docker run -p 8501:8501 ai-tools-dashboard
```

Access at `http://localhost:8501`

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

---

## Heroku Deployment

### Setup

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

3. Update `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

### Deploy

```bash
heroku login
heroku create ai-tools-dashboard
git push heroku main
heroku open
```

---

## AWS EC2 Deployment

### Instance Setup

1. Launch EC2 instance (t2.medium recommended)
2. Configure security group (allow port 8501)
3. SSH into instance

### Installation

```bash
sudo apt update
sudo apt install -y python3-pip
git clone https://github.com/yourusername/ai-tools-dashboard.git
cd ai-tools-dashboard
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 setup_models.py
```

### Run with Screen

```bash
screen -S streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Detach with `Ctrl+A, D`

### Production with Nginx

Install Nginx:
```bash
sudo apt install nginx
```

Configure reverse proxy in `/etc/nginx/sites-available/streamlit`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## Performance Optimization

### 1. Model Caching

Models are already cached with `@st.cache_resource` decorator.

### 2. Reduce Memory Usage

In `setup_models.py`, reduce MNIST training:
```python
epochs=5  # instead of 10
batch_size=256  # instead of 128
```

### 3. Lazy Loading

Models load only when needed, not at startup.

### 4. CDN for Static Assets

Use Streamlit's built-in asset optimization.

---

## Monitoring

### Streamlit Cloud
- Built-in metrics dashboard
- Error tracking
- Usage analytics

### Custom Monitoring

Add to `app.py`:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction made: {prediction}")
```

---

## Troubleshooting

### Issue: Models Not Found
**Solution**: Ensure models directory exists and contains `.pkl` and `.h5` files

### Issue: spaCy Model Missing
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: Out of Memory
**Solution**:
- Use smaller model architectures
- Reduce batch sizes
- Use swap space on Linux

### Issue: Canvas Not Working
**Solution**: Ensure `streamlit-drawable-canvas` is installed

### Issue: Slow SHAP Analysis
**Solution**: Reduce sample size in SHAP calculations

---

## Security Considerations

1. **No Sensitive Data**: Don't include API keys or credentials
2. **Input Validation**: Already implemented in the app
3. **Rate Limiting**: Use Streamlit Cloud's built-in limits
4. **HTTPS**: Enable in deployment platform settings
5. **Updates**: Regularly update dependencies

---

## Scaling

### Horizontal Scaling
- Use Streamlit Cloud's auto-scaling
- Or deploy multiple containers with load balancer

### Vertical Scaling
- Increase instance size for better performance
- Recommended: 2GB+ RAM

### Database Integration
- Add PostgreSQL for user data
- Cache predictions in Redis
- Store analytics in time-series DB

---

## Cost Estimates

### Streamlit Cloud (Free Tier)
- Cost: $0
- Limits: Public apps, community support
- Best for: Demos, portfolios

### Streamlit Cloud (Teams)
- Cost: $250/month per user
- Features: Private apps, priority support
- Best for: Team projects

### AWS EC2 (t2.medium)
- Cost: ~$35/month
- Performance: Good for production
- Best for: Custom deployments

### Docker + Cloud Run
- Cost: Pay per use (~$5-20/month)
- Scales to zero
- Best for: Variable traffic

---

## Support

For deployment issues:
- Streamlit Community: https://discuss.streamlit.io
- GitHub Issues: Create issue in your repo
- Team Aimtech7: Contact via assignment portal

---

**Last Updated**: October 2025
**Version**: 1.0.0

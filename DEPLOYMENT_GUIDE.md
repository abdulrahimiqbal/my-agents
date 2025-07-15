# 🚀 PhysicsGPT Deployment Guide

## 📋 Overview

PhysicsGPT is a sophisticated 10-agent physics research system that can be deployed in multiple ways. This guide covers all deployment options from local development to production cloud hosting.

## 🎯 Deployment Options

### 1. 🖥️ Local Development

**Quick Start:**
```bash
# Activate environment
source physics-env/bin/activate

# Install UI dependencies
pip install streamlit plotly

# Run Streamlit UI
streamlit run streamlit_physics_app.py

# Or run CLI version
python physics_crew_system.py "Your physics question"
```

**Access:** http://localhost:8501

### 2. ☁️ Streamlit Cloud (Recommended for Demos)

**Steps:**
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `streamlit_physics_app.py`
5. Add secrets in Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "your-key-here"
   TAVILY_API_KEY = "your-key-here"
   LANGCHAIN_API_KEY = "your-key-here"
   ```

**Pros:** Free, easy sharing, auto-deploys
**Cons:** Limited resources, public by default

### 3. 🚂 Railway (Recommended for Production)

**Steps:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Environment Variables:**
```bash
railway variables set OPENAI_API_KEY=your-key
railway variables set TAVILY_API_KEY=your-key
railway variables set LANGCHAIN_API_KEY=your-key
```

**Pros:** Production-ready, custom domains, auto-scaling
**Cost:** ~$5-20/month depending on usage

### 4. 🐳 Docker Deployment

**Local Docker:**
```bash
# Build image
docker build -t physics-gpt .

# Run container
docker run -p 8501:8501 --env-file .env physics-gpt
```

**Docker Compose:**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

**Production Docker (AWS/GCP/Azure):**
- Push to container registry
- Deploy to ECS/Cloud Run/Container Instances
- Set environment variables in cloud console

### 5. 🌐 Other Cloud Options

#### Render
```bash
# Connect GitHub repo
# Set build command: pip install -r requirements.txt
# Set start command: streamlit run streamlit_physics_app.py --server.port=$PORT
```

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_physics_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key
PHYSICS_AGENT_MODEL=gpt-4o-mini         # Model to use

# Optional
TAVILY_API_KEY=tvly-...                 # For web search
LANGCHAIN_API_KEY=lsv2_...              # For tracing
LANGCHAIN_TRACING_V2=true               # Enable tracing
LANGCHAIN_PROJECT=physics-gpt           # Project name
```

### Resource Requirements
- **Memory:** 2GB+ (4GB recommended for 10-agent analysis)
- **CPU:** 2+ cores recommended
- **Storage:** 1GB minimum
- **Network:** Stable internet for API calls

## 🎨 UI Features

### Streamlit Interface
- **Modern Design:** Gradient headers, responsive layout
- **Agent Selection:** Choose 5 core, custom, or all 10 agents
- **Real-time Progress:** Live updates during analysis
- **Analysis History:** Track previous queries
- **Download Results:** Export analysis as text files
- **Example Queries:** Quick-start templates

### CLI Interface
- **Direct Queries:** `python physics_crew_system.py "question"`
- **Interactive Mode:** Menu-driven interface
- **Agent Selection:** Custom agent combinations
- **Demo Mode:** Full 10-agent showcase

## 📊 Monitoring & Scaling

### Health Checks
```bash
# Streamlit health endpoint
curl http://localhost:8501/_stcore/health

# Custom health check
curl http://localhost:8501/health
```

### Performance Optimization
- **Agent Selection:** Use 5 core agents for faster responses
- **Caching:** Streamlit caches results automatically
- **Memory Management:** Monitor memory usage with 10-agent runs
- **API Rate Limits:** Implement request throttling if needed

### Scaling Considerations
- **Horizontal Scaling:** Deploy multiple instances behind load balancer
- **Vertical Scaling:** Increase memory/CPU for complex analyses
- **Database:** Add persistent storage for analysis history
- **CDN:** Use CDN for static assets in production

## 🔒 Security

### API Key Management
- Never commit API keys to version control
- Use environment variables or secret management
- Rotate keys regularly
- Monitor API usage and costs

### Access Control
- Enable authentication in production
- Use HTTPS/TLS encryption
- Implement rate limiting
- Monitor for abuse

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. API Key Errors:**
```bash
# Check environment variables
echo $OPENAI_API_KEY
```

**3. Memory Issues:**
```bash
# Reduce agent count or increase memory
# Use 5 core agents instead of all 10
```

**4. Port Conflicts:**
```bash
# Use different port
streamlit run streamlit_physics_app.py --server.port=8502
```

### Logs and Debugging
```bash
# Streamlit logs
streamlit run streamlit_physics_app.py --logger.level=debug

# Docker logs
docker logs container-name

# Railway logs
railway logs
```

## 📈 Cost Estimation

### API Costs (Monthly)
- **Light Usage (10 queries/day):** ~$5-15
- **Medium Usage (50 queries/day):** ~$25-75
- **Heavy Usage (200 queries/day):** ~$100-300

### Hosting Costs (Monthly)
- **Streamlit Cloud:** Free
- **Railway:** $5-20
- **Render:** $7-25
- **AWS/GCP/Azure:** $10-50+

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured
- [ ] Health checks implemented
- [ ] Monitoring setup
- [ ] Backup strategy
- [ ] Error handling
- [ ] Rate limiting
- [ ] HTTPS enabled
- [ ] Domain configured
- [ ] Performance tested

## 📞 Support

For deployment issues:
1. Check logs for error messages
2. Verify environment variables
3. Test API connectivity
4. Monitor resource usage
5. Review configuration files

## 🎯 Recommended Setup

**For Demos/Testing:** Streamlit Cloud
**For Production:** Railway + Custom Domain
**For Enterprise:** Docker on AWS/GCP/Azure

Choose based on your needs, budget, and technical requirements!
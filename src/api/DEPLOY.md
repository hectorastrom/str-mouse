# scribble API Deployment Guide

Deploy the scribble API to an EC2 instance at `api.hectorastrom.com`.

## Architecture

```
hectorastrom.com/blog/scribble  -->  api.hectorastrom.com/predict
       (Next.js page)                   (FastAPI on EC2)
              |                                |
              v                                v
         Cloudflare                       Cloudflare
         (SSL + CDN)                    (SSL + Proxy)
```

- **Frontend**: Next.js component on your blog, sends JSON velocity data
- **Backend**: Stateless FastAPI server on EC2, returns predictions without storing data
- **Cloudflare**: Handles SSL certificates and DNS for both domains

## Key Concepts

### What is Nginx Reverse Proxy?

Your FastAPI server runs on `localhost:8000` inside the EC2 instance. Nginx sits in front of it and:

1. **Listens on port 80/443** - The standard HTTP/HTTPS ports that browsers expect
2. **Forwards requests** to your FastAPI app on port 8000
3. **Handles SSL termination** - Decrypts HTTPS traffic from Cloudflare
4. **Adds security headers** - Protects against common attacks
5. **Serves as a buffer** - Can handle slow clients without blocking your Python app

```
Internet → Cloudflare → EC2:443 (nginx) → localhost:8000 (FastAPI)
```

Without nginx, you'd have to run FastAPI directly on port 443, which requires root privileges and lacks the performance/security benefits.

### Do I Need a New SSL Certificate?

**No.** Cloudflare provides free SSL certificates automatically for any subdomain you add. When you create the `api` subdomain in Cloudflare, it's automatically covered by Cloudflare's Universal SSL.

The connection works like this:
- **Browser → Cloudflare**: Encrypted with Cloudflare's SSL cert (automatic)
- **Cloudflare → EC2**: Can be encrypted (Full SSL) or unencrypted (Flexible SSL)

We'll use **Full (Strict)** mode with a Cloudflare Origin Certificate on the EC2 server for end-to-end encryption.

## Prerequisites

- AWS account with EC2 access
- Domain `hectorastrom.com` on Cloudflare
- SSH key pair for EC2 access

## EC2 Setup

### 1. Launch Instance

Recommended specs for cost-effective inference:
- **Instance type**: `t3.small` (2 vCPU, 2GB RAM)
- **AMI**: Ubuntu 22.04 LTS
- **Storage**: 20GB gp3
- **Security Group**:
  - SSH (22) from your IP
  - Allow inbound 443 (and optionally 80) only from Cloudflare IP ranges

### 2. Connect and Install Dependencies

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# System setup
sudo apt update && sudo apt upgrade -y
sudo reboot
sudo apt-get install python3.11-dev linux-headers-$(uname -r)
sudo apt-get install build-essential

# Install Python 3.11+ and uv
sudo apt install -y python3.11 python3.11-venv git
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install nginx
sudo apt install -y nginx
```

### 3. Clone Repository and Setup

```bash
mkdir projects
cd projects
git clone https://github.com/hectorastrom/scribble.git
cd scribble

# Install dependencies with uv
uv sync
```

### 4. Install a Cloudflare Origin Certificate (required for Full (strict))

Cloudflare terminates TLS for visitors at the edge (Browser ⇄ Cloudflare). **Full (strict)** also requires TLS on the second leg (Cloudflare ⇄ EC2). A **Cloudflare Origin Certificate** is a TLS certificate trusted by Cloudflare (not by public browsers) that secures that origin connection.

1. Cloudflare Dashboard → your domain → **SSL/TLS** → **Origin Server**
2. Click **Create Certificate**
3. Hostnames: include `api.hectorastrom.com` (and any other API subdomains you plan to use)
4. Keep defaults (RSA 2048, long validity)
5. Copy the **Origin Certificate** and **Private Key**

On your EC2 instance:

```bash
# Create certificate directory
sudo mkdir -p /etc/ssl/cloudflare

# Create certificate file
sudo nano /etc/ssl/cloudflare/cert.pem
# Paste the Origin Certificate (full PEM block), then save/exit

# Create private key file
sudo nano /etc/ssl/cloudflare/key.pem
# Paste the Private Key (full PEM block), then save/exit

# Lock down permissions
sudo chown -R root:root /etc/ssl/cloudflare
sudo chmod 644 /etc/ssl/cloudflare/cert.pem
sudo chmod 600 /etc/ssl/cloudflare/key.pem
```

### 5. Configure Nginx reverse proxy (TLS termination + proxy to FastAPI)

Create the Nginx site file:

```bash
sudo nano /etc/nginx/sites-available/api.hectorastrom.com
```

Add:

```nginx
server {
    listen 80;
    server_name api.hectorastrom.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.hectorastrom.com;

    # TLS certificate for Cloudflare -> origin (Origin Certificate)
    ssl_certificate     /etc/ssl/cloudflare/cert.pem;
    ssl_certificate_key /etc/ssl/cloudflare/key.pem;

    # TLS settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;

    # Reverse proxy to FastAPI (listening on localhost only)
    location / {
        proxy_pass http://127.0.0.1:8000;

        proxy_http_version 1.1;
        proxy_set_header Host $host;

        # Cloudflare sets CF-Connecting-IP with the real client IP.
        # Without additional real_ip config, $remote_addr will be a Cloudflare IP.
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 60s;
        proxy_connect_timeout 60s;
    }
}
```

Enable the site and reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/api.hectorastrom.com /etc/nginx/sites-enabled/

sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t

sudo systemctl reload nginx
```

### 6. Create a systemd service for FastAPI (uvicorn)

Create the unit file:

```bash
sudo nano /etc/systemd/system/scribble-api.service
```

Add:

```ini
[Unit]
Description=scribble API Server
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/projects/scribble
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable scribble-api
sudo systemctl start scribble-api

sudo systemctl status scribble-api
sudo journalctl -u scribble-api -f
```

## Cloudflare DNS setup

### Add the `api` subdomain

1. Cloudflare Dashboard → select `hectorastrom.com`
2. **DNS** → **Add record**
3. Create:

   * **Type**: `A`
   * **Name**: `api`
   * **IPv4 address**: your EC2 public IP
   * **Proxy status**: **Proxied** (orange cloud)
   * **TTL**: Auto
4. Save

### Set SSL mode to Full (strict)

1. **SSL/TLS** → **Overview**
2. Set mode to **Full (strict)**

This encrypts both legs:

* Browser ⇄ Cloudflare (Cloudflare Universal SSL)
* Cloudflare ⇄ EC2 (Origin Certificate installed above)

## Verify deployment

After DNS propagates:

```bash
# Check health and available models
curl https://api.hectorastrom.com/health

# List available model variants
curl https://api.hectorastrom.com/models

# Test prediction (uses default model - highest class count)
curl -X POST https://api.hectorastrom.com/predict \
  -H "Content-Type: application/json" \
  -d '{"velocities":[{"vx":1,"vy":0},{"vx":2,"vy":1},{"vx":1,"vy":2}]}'

# Test prediction with specific model variant
curl -X POST https://api.hectorastrom.com/predict \
  -H "Content-Type: application/json" \
  -d '{"velocities":[{"vx":1,"vy":0},{"vx":2,"vy":1},{"vx":1,"vy":2}], "num_classes": 53}'
```

Place `widget.html` in your `public/` directory as `public/scribble-widget.html`.

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "available_classes": [53, 63]
}
```

### `GET /`

Root endpoint with API info.

**Response:**
```json
{
  "name": "scribble API",
  "version": "1.1.0",
  "docs": "/docs",
  "health": "/health",
  "models": "/models"
}
```

### `GET /models`

List available model variants. Use these values in the `num_classes` field of `/predict` requests.

**Response:**
```json
{
  "available": [53, 63],
  "default": 63
}
```

### `POST /predict`

Predict character from mouse velocity data.

**Request:**
```json
{
  "velocities": [
    {"vx": 1.5, "vy": 0.0},
    {"vx": 2.0, "vy": 1.0}
  ],
  "num_classes": 63
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `velocities` | array | Yes | List of velocity points `{vx, vy}` |
| `num_classes` | int | No | Model variant to use. If omitted, uses default (highest available). Get valid values from `GET /models`. |

**Response:**
```json
{
  "predicted_char": "A",
  "confidence": 94.5,
  "inference_ms": 2.3,
  "image_base64": "iVBORw0KGgo...",
  "image_data_url": "data:image/png;base64,iVBORw0KGgo...",
  "num_classes": 63
}
```

**Error Response (invalid num_classes):**
```json
{
  "detail": "No model available for 70 classes. Available: [53, 63]"
}
```
Status code: 400

## Monitoring & Maintenance

### View Logs

```bash
sudo journalctl -u scribble-api -f
```

### Restart Service

```bash
sudo systemctl restart scribble-api
```

### Update Code

```bash
cd ~/projects/scribble
git pull
sudo systemctl restart scribble-api
```

Note: On restart, all `best_finetune-N-class` checkpoints are automatically discovered and loaded. Add new model variants by placing them in `checkpoints/mouse_finetune/best_finetune-N-class/best.ckpt`.

## Cost Estimate

- **t3.small** (2 vCPU, 2GB): ~$15/month
- **t3.medium** (2 vCPU, 4GB): ~$30/month
- **Cloudflare**: Free tier is sufficient
- **Data transfer**: Minimal (small JSON payloads)
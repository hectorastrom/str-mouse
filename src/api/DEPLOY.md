# str(mouse) API Deployment Guide

Deploy the str(mouse) API to an EC2 instance at `api.hectorastrom.com`.

## Architecture

```
hectorastrom.com/blog/strmouse  -->  api.hectorastrom.com/predict
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
- **Instance type**: `t3.small` or `t3.medium` (2 vCPU, 2-4GB RAM)
- **AMI**: Ubuntu 22.04 LTS
- **Storage**: 20GB gp3
- **Security Group**:
  - SSH (22) from your IP
  - HTTP (80) from anywhere (Cloudflare will redirect to HTTPS)
  - HTTPS (443) from anywhere

### 2. Connect and Install Dependencies

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and uv
sudo apt install -y python3.11 python3.11-venv git
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Install nginx
sudo apt install -y nginx
```

### 3. Clone Repository and Setup

```bash
cd ~
git clone https://github.com/hectorastrom/strmouse.git
cd strmouse

# Install dependencies with uv
uv sync

# Download model checkpoint from S3
mkdir -p checkpoints/mouse_finetune/best_finetune
aws s3 cp s3://hectorastrom-str-mouse/checkpoints/mouse_finetune/best_finetune/best.ckpt \
    checkpoints/mouse_finetune/best_finetune/best.ckpt
```

### 4. Setup Cloudflare Origin Certificate (for Full SSL)

This creates a certificate that Cloudflare trusts, enabling encrypted traffic between Cloudflare and your server.

1. Go to Cloudflare Dashboard → your domain → **SSL/TLS** → **Origin Server**
2. Click **Create Certificate**
3. Keep defaults (RSA 2048, 15 years validity)
4. Copy the **Origin Certificate** and **Private Key**

On your EC2 instance:

```bash
# Create certificate directory
sudo mkdir -p /etc/ssl/cloudflare

# Paste the Origin Certificate
sudo nano /etc/ssl/cloudflare/cert.pem
# (paste certificate, save)

# Paste the Private Key
sudo nano /etc/ssl/cloudflare/key.pem
# (paste key, save)

# Secure the key file
sudo chmod 600 /etc/ssl/cloudflare/key.pem
```

### 5. Configure Nginx Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/api.hectorastrom.com
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name api.hectorastrom.com;
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.hectorastrom.com;

    # Cloudflare Origin Certificate
    ssl_certificate /etc/ssl/cloudflare/cert.pem;
    ssl_certificate_key /etc/ssl/cloudflare/key.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Proxy to FastAPI
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 60s;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/api.hectorastrom.com /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # Remove default site
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

### 6. Create Systemd Service

```bash
sudo nano /etc/systemd/system/strmouse-api.service
```

Add:

```ini
[Unit]
Description=str(mouse) API Server
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/strmouse
Environment="PATH=/home/ubuntu/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/.cargo/bin/uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable strmouse-api
sudo systemctl start strmouse-api

# Check status
sudo systemctl status strmouse-api

# View logs
sudo journalctl -u strmouse-api -f
```

## Cloudflare DNS Setup

### Adding the Subdomain

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → select `hectorastrom.com`
2. Click **DNS** in the sidebar
3. Click **Add record**
4. Fill in:
   - **Type**: `A`
   - **Name**: `api` (this creates `api.hectorastrom.com`)
   - **IPv4 address**: Your EC2 instance's public IP (e.g., `54.123.45.67`)
   - **Proxy status**: **Proxied** (orange cloud ON) - this enables Cloudflare's SSL and protection
   - **TTL**: Auto
5. Click **Save**

### Configure SSL Mode

1. Go to **SSL/TLS** in the sidebar
2. Under **Overview**, select **Full (strict)**
   - This ensures traffic is encrypted end-to-end
   - Requires the Origin Certificate we set up earlier

### Recommended Cloudflare Settings

In **SSL/TLS** → **Edge Certificates**:
- **Always Use HTTPS**: ON
- **Minimum TLS Version**: TLS 1.2

In **Security** → **Settings**:
- **Security Level**: Medium (or higher if you get attacks)

## Verify Deployment

Wait 1-2 minutes for DNS to propagate, then:

```bash
# Health check
curl https://api.hectorastrom.com/health

# Test prediction
curl -X POST https://api.hectorastrom.com/predict \
  -H "Content-Type: application/json" \
  -d '{"velocities": [{"vx": 1, "vy": 0}, {"vx": 2, "vy": 1}, {"vx": 1, "vy": 2}]}'
```

## Embedding in Next.js

The widget needs to be converted to a React component for Next.js. Create this component:

### `components/StrMouseWidget.tsx`

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = "https://api.hectorastrom.com";

interface PredictResponse {
  predicted_char: string;
  confidence: number;
  inference_ms: number;
  image_data_url: string;
}

export default function StrMouseWidget() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [output, setOutput] = useState("");
  const [confidence, setConfidence] = useState<string>("--");
  const [inferenceMs, setInferenceMs] = useState<string>("--");
  const [previewSrc, setPreviewSrc] = useState<string>("");
  const [error, setError] = useState<string>("");

  // Recording state refs (to avoid stale closures in event handlers)
  const recordingActive = useRef(false);
  const velocities = useRef<{ vx: number; vy: number }[]>([]);
  const currentPos = useRef<{ x: number; y: number } | null>(null);
  const lastLoggedPos = useRef<{ x: number; y: number } | null>(null);
  const lastMoveTime = useRef(0);
  const lastDrawPos = useRef<{ x: number; y: number } | null>(null);
  const samplingTimer = useRef<NodeJS.Timeout | null>(null);

  const sampleIntervalMs = 10;
  const inactivityMs = 250;
  const minSamples = 3;

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawLine = useCallback(
    (from: { x: number; y: number }, to: { x: number; y: number }) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    },
    []
  );

  const stopRecording = useCallback(async () => {
    recordingActive.current = false;
    if (samplingTimer.current) {
      clearInterval(samplingTimer.current);
      samplingTimer.current = null;
    }

    const vels = velocities.current;
    if (vels.length === 0) return;

    clearCanvas();

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ velocities: vels }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: PredictResponse = await response.json();

      setOutput((prev) => prev + data.predicted_char);
      setConfidence(`${data.confidence.toFixed(1)}%`);
      setInferenceMs(`${data.inference_ms.toFixed(1)}ms`);
      setPreviewSrc(data.image_data_url);
      setError("");
    } catch (err) {
      console.error("Prediction failed:", err);
      setError(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
    }

    velocities.current = [];
    lastDrawPos.current = null;
  }, [clearCanvas]);

  const sampleVelocity = useCallback(() => {
    if (!recordingActive.current || !currentPos.current || !lastLoggedPos.current)
      return;

    const vx = currentPos.current.x - lastLoggedPos.current.x;
    const vy = currentPos.current.y - lastLoggedPos.current.y;
    lastLoggedPos.current = { ...currentPos.current };
    velocities.current.push({ vx, vy });

    if (
      performance.now() - lastMoveTime.current >= inactivityMs &&
      velocities.current.length >= minSamples
    ) {
      stopRecording();
    }
  }, [stopRecording]);

  const startRecording = useCallback(
    (pos: { x: number; y: number }) => {
      recordingActive.current = true;
      velocities.current = [];
      currentPos.current = { ...pos };
      lastLoggedPos.current = { ...pos };
      lastMoveTime.current = performance.now();

      if (samplingTimer.current) clearInterval(samplingTimer.current);
      samplingTimer.current = setInterval(sampleVelocity, sampleIntervalMs);
    },
    [sampleVelocity]
  );

  const getCanvasPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      let clientX: number, clientY: number;
      if ("touches" in e && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else if ("clientX" in e) {
        clientX = e.clientX;
        clientY = e.clientY;
      } else {
        return { x: 0, y: 0 };
      }

      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
      };
    },
    []
  );

  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const pos = getCanvasPos(e);

      if (!recordingActive.current) {
        startRecording(pos);
      }

      currentPos.current = pos;
      lastMoveTime.current = performance.now();

      if (lastDrawPos.current) {
        drawLine(lastDrawPos.current, pos);
      }
      lastDrawPos.current = { ...pos };
    },
    [getCanvasPos, startRecording, drawLine]
  );

  const handleLeave = useCallback(() => {
    lastDrawPos.current = null;
  }, []);

  const handleClear = useCallback(() => {
    setOutput("");
    setConfidence("--");
    setInferenceMs("--");
    setPreviewSrc("");
    setError("");
  }, []);

  useEffect(() => {
    clearCanvas();
    return () => {
      if (samplingTimer.current) clearInterval(samplingTimer.current);
    };
  }, [clearCanvas]);

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between bg-white border border-gray-300 rounded-t-lg p-3">
        <div
          className="font-mono text-sm bg-gray-50 border border-gray-200 rounded px-3 py-2 min-w-[200px] max-w-[400px] min-h-[24px] break-words text-gray-800"
        >
          {output || <span className="text-gray-400">Start drawing...</span>}
        </div>
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 border border-gray-300 rounded bg-white overflow-hidden">
            {previewSrc && (
              <img src={previewSrc} alt="preview" className="w-full h-full" />
            )}
          </div>
          <div className="font-mono text-xs text-right text-gray-500 min-w-[100px]">
            <div>conf: {confidence}</div>
            <div>time: {inferenceMs}</div>
          </div>
        </div>
      </div>

      <div className="border border-t-0 border-gray-300 rounded-b-lg bg-white p-3">
        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          className="w-full h-auto block bg-white border border-gray-300 rounded-md cursor-crosshair touch-none"
          onMouseMove={handleMove}
          onMouseLeave={handleLeave}
          onTouchMove={handleMove}
          onTouchEnd={handleLeave}
        />
        <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
          <span>Move mouse over canvas to draw. Pause briefly between characters.</span>
          <button
            onClick={handleClear}
            className="bg-gray-100 border border-gray-300 rounded px-2 py-1 text-gray-600 hover:bg-gray-200"
          >
            Clear Output
          </button>
        </div>
        {error && <div className="text-red-600 text-xs mt-2">{error}</div>}
      </div>
    </div>
  );
}
```

### Usage in a Blog Post Page

```tsx
// app/blog/strmouse/page.tsx
import StrMouseWidget from "@/components/StrMouseWidget";

export default function StrMouseBlogPost() {
  return (
    <article className="prose lg:prose-xl mx-auto py-8">
      <h1>str(mouse): Decoding Characters from Mouse Movement</h1>

      <p>
        Try it yourself! Draw characters with your mouse and watch the model
        decode them in real-time.
      </p>

      <div className="not-prose my-8">
        <StrMouseWidget />
      </div>

      <p>
        The model runs on a lightweight CNN that processes your mouse velocity
        data and classifies it into one of 53 characters...
      </p>
    </article>
  );
}
```

### Alternative: Iframe Embed

If you prefer not to add a React component, you can host `widget.html` as a static file and embed it:

```tsx
// In your Next.js page
export default function StrMouseBlogPost() {
  return (
    <article>
      <h1>str(mouse)</h1>
      <iframe
        src="/strmouse-widget.html"
        width="100%"
        height="500"
        style={{ border: "none", borderRadius: "8px" }}
      />
    </article>
  );
}
```

Place `widget.html` in your `public/` directory as `public/strmouse-widget.html`.

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "healthy", "model_loaded": true}
```

### `GET /`

Root endpoint with API info.

**Response:**
```json
{"name": "str(mouse) API", "version": "1.0.0", "docs": "/docs", "health": "/health"}
```

### `POST /predict`

Predict character from mouse velocity data.

**Request:**
```json
{
  "velocities": [
    {"vx": 1.5, "vy": 0.0},
    {"vx": 2.0, "vy": 1.0}
  ]
}
```

**Response:**
```json
{
  "predicted_char": "A",
  "confidence": 94.5,
  "inference_ms": 2.3,
  "image_base64": "iVBORw0KGgo...",
  "image_data_url": "data:image/png;base64,iVBORw0KGgo..."
}
```

## Monitoring & Maintenance

### View Logs

```bash
sudo journalctl -u strmouse-api -f
```

### Restart Service

```bash
sudo systemctl restart strmouse-api
```

### Update Code

```bash
cd ~/strmouse
git pull
sudo systemctl restart strmouse-api
```

## Cost Estimate

- **t3.small** (2 vCPU, 2GB): ~$15/month
- **t3.medium** (2 vCPU, 4GB): ~$30/month
- **Cloudflare**: Free tier is sufficient
- **Data transfer**: Minimal (small JSON payloads)

Use Spot Instances or Reserved Instances for additional savings.

## Troubleshooting

### API returns 502 Bad Gateway

Check if the service is running:
```bash
sudo systemctl status strmouse-api
sudo journalctl -u strmouse-api --since "5 minutes ago"
```

### CORS errors in browser console

Ensure your domain is in `ALLOWED_ORIGINS` in `main.py`:
```python
ALLOWED_ORIGINS = [
    "https://hectorastrom.com",
    "https://www.hectorastrom.com",
    # ... add any other domains
]
```
Restart the service after changes.

### SSL errors / "Origin Certificate" issues

1. Verify Cloudflare SSL mode is set to **Full (strict)**
2. Check that origin certificate files exist and have correct permissions:
   ```bash
   ls -la /etc/ssl/cloudflare/
   sudo nginx -t
   ```

### DNS not resolving

1. Check Cloudflare DNS dashboard for the A record
2. Verify proxy status is enabled (orange cloud)
3. Wait a few minutes for propagation
4. Test with: `dig api.hectorastrom.com`

### Model loading fails

Verify checkpoint exists:
```bash
ls -la ~/strmouse/checkpoints/mouse_finetune/best_finetune/best.ckpt
```

Set custom path via environment variable in the systemd service:
```ini
Environment="STROKNET_CKPT_PATH=/path/to/your/checkpoint.ckpt"
```

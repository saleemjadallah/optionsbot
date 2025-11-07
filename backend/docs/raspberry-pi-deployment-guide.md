# Raspberry Pi Deployment Guide for Options Trading Bot

## Hardware Requirements

### Recommended Setup
- **Raspberry Pi 4 Model B (4GB or 8GB RAM)** - $75-95
- **64GB+ SSD with USB 3.0 adapter** - $30-50 (more reliable than SD cards)
- **Active cooling case with fan** - $15-25
- **High-quality power supply (USB-C, 5V/3A)** - $10-15
- **Ethernet cable** (preferred over WiFi for stability)

### Optional but Recommended
- **UPS (Uninterruptible Power Supply)** - $50-100
- **GPIO expansion board** for status LEDs
- **Small HDMI monitor** for initial setup

**Total Cost: $180-285**

## Operating System Setup

### 1. Install Raspberry Pi OS (64-bit)

```bash
# Download Raspberry Pi Imager
# https://www.raspberrypi.org/software/

# Flash to SSD/SD card with these settings:
# - OS: Raspberry Pi OS Lite (64-bit) - no desktop needed
# - Enable SSH
# - Set username: trader
# - Set strong password
# - Configure WiFi (backup to ethernet)
```

### 2. Initial System Configuration

```bash
# SSH into Pi
ssh trader@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose \
    htop \
    vim \
    curl \
    unzip \
    build-essential \
    libffi-dev \
    libssl-dev

# Add user to docker group
sudo usermod -aG docker trader
newgrp docker

# Configure timezone
sudo timedatectl set-timezone America/New_York

# Configure automatic updates
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 3. Performance Optimization

```bash
# Edit boot config for better performance
sudo nano /boot/config.txt

# Add these lines for optimization:
# GPU Memory Split (since no desktop)
gpu_mem=16

# CPU Governor for consistent performance
force_turbo=1

# USB Boot priority (if using SSD)
program_usb_boot_mode=1

# Increase swap for compilation
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Reboot
sudo reboot
```

## Docker Configuration for Trading Bot

### 1. Create Project Structure

```bash
# Create project directory
mkdir -p ~/options-trader
cd ~/options-trader

# Create directory structure
mkdir -p {config,logs,data,backups}

# Create environment file
cat > .env << 'EOF'
# Tastyworks Credentials
TW_USERNAME=your_username
TW_PASSWORD=your_password
TW_ACCOUNT=your_account_number
TW_SANDBOX=false

# Trading Configuration
INITIAL_CAPITAL=10000
RISK_LEVEL=low
MAX_DAILY_LOSS=0.02
MAX_POSITION_SIZE=0.15

# System Configuration
LOG_LEVEL=INFO
DATABASE_PATH=/data/trading.db
BACKUP_INTERVAL=3600

# Timezone
TZ=America/New_York
EOF

# Secure the environment file
chmod 600 .env
```

### 2. Simplified Docker Compose for Pi

```yaml
# docker-compose.yml
version: '3.8'

services:
  trader:
    build: .
    container_name: options_trader
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
      - /etc/localtime:/etc/localtime:ro
    ports:
      - "8080:8080"  # Health check endpoint
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: trader_dashboard
    restart: unless-stopped
    depends_on:
      - trader
    ports:
      - "8050:8050"
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs:ro
    environment:
      - DATABASE_PATH=/app/data/trading.db
    logging:
      driver: "json-file"
      options:
        max-size: "5m"
        max-file: "3"

  watchtower:
    image: containrrr/watchtower
    container_name: watchtower
    restart: unless-stopped
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - WATCHTOWER_CLEANUP=true
      - WATCHTOWER_POLL_INTERVAL=3600
    command: options_trader trader_dashboard
```

### 3. Optimized Dockerfile for Raspberry Pi

```dockerfile
# Dockerfile
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader
RUN chown -R trader:trader /app
USER trader

# Health check endpoint
EXPOSE 8080

# Start application
CMD ["python", "main.py"]
```

### 4. Lightweight Dashboard Dockerfile

```dockerfile
# Dockerfile.dashboard
FROM python:3.11-slim-bullseye

WORKDIR /app

# Install only dashboard dependencies
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY monitoring/ ./monitoring/
COPY config/ ./config/

RUN groupadd -r trader && useradd -r -g trader trader
RUN chown -R trader:trader /app
USER trader

EXPOSE 8050

CMD ["python", "monitoring/dashboard.py"]
```

## System Monitoring and Maintenance

### 1. System Monitoring Script

```bash
# Create monitoring script
cat > ~/monitor.sh << 'EOF'
#!/bin/bash

# System health monitoring for Raspberry Pi

LOG_FILE="/home/trader/system_health.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# CPU Temperature
CPU_TEMP=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\' -f1)

# Memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')

# Disk usage
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

# Docker container status
TRADER_STATUS=$(docker ps --filter "name=options_trader" --format "{{.Status}}")
DASHBOARD_STATUS=$(docker ps --filter "name=trader_dashboard" --format "{{.Status}}")

# Log system stats
echo "$DATE - CPU: ${CPU_TEMP}°C, RAM: ${MEM_USAGE}%, Disk: ${DISK_USAGE}%, Trader: $TRADER_STATUS, Dashboard: $DASHBOARD_STATUS" >> $LOG_FILE

# Alert if temperature too high
if (( $(echo "$CPU_TEMP > 70" | bc -l) )); then
    echo "$DATE - WARNING: High CPU temperature: ${CPU_TEMP}°C" >> $LOG_FILE
    # Optional: Send notification
fi

# Alert if memory usage too high
if (( $(echo "$MEM_USAGE > 90" | bc -l) )); then
    echo "$DATE - WARNING: High memory usage: ${MEM_USAGE}%" >> $LOG_FILE
fi

# Keep only last 1000 lines
tail -n 1000 $LOG_FILE > ${LOG_FILE}.tmp && mv ${LOG_FILE}.tmp $LOG_FILE
EOF

chmod +x ~/monitor.sh
```

### 2. Cron Jobs for Automation

```bash
# Add cron jobs
crontab -e

# Add these lines:
# System monitoring every 5 minutes
*/5 * * * * /home/trader/monitor.sh

# Daily backup at 2 AM
0 2 * * * /home/trader/backup.sh

# Weekly system update at 3 AM Sunday
0 3 * * 0 sudo apt update && sudo apt upgrade -y

# Daily log rotation at 1 AM
0 1 * * * find /home/trader/options-trader/logs -name "*.log" -mtime +7 -delete

# Restart containers weekly to clear memory leaks
0 4 * * 1 cd /home/trader/options-trader && docker-compose restart
```

### 3. Backup Script

```bash
# Create backup script
cat > ~/backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/home/trader/backups"
DATE=$(date +%Y%m%d_%H%M%S)
TRADER_DIR="/home/trader/options-trader"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup trading data
tar -czf "$BACKUP_DIR/trading_data_$DATE.tar.gz" \
    -C "$TRADER_DIR" \
    data/ \
    config/ \
    .env

# Backup to external storage if mounted
if mountpoint -q /mnt/usb; then
    cp "$BACKUP_DIR/trading_data_$DATE.tar.gz" /mnt/usb/
fi

# Keep only last 14 days of backups
find $BACKUP_DIR -name "trading_data_*.tar.gz" -mtime +14 -delete

echo "$(date): Backup completed - trading_data_$DATE.tar.gz" >> $BACKUP_DIR/backup.log
EOF

chmod +x ~/backup.sh
```

## Security Configuration

### 1. Firewall Setup

```bash
# Install and configure UFW
sudo apt install -y ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if needed)
sudo ufw allow 22/tcp

# Allow dashboard access from local network only
sudo ufw allow from 192.168.0.0/16 to any port 8050

# Allow health checks
sudo ufw allow 8080/tcp

# Enable firewall
sudo ufw enable
```

### 2. SSH Hardening

```bash
# Create SSH key pair (on your local machine)
ssh-keygen -t ed25519 -C "trading-bot-key"

# Copy public key to Pi
ssh-copy-id -i ~/.ssh/id_ed25519.pub trader@raspberrypi.local

# Harden SSH config on Pi
sudo nano /etc/ssh/sshd_config

# Add/modify these settings:
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
Port 2222  # Change from default 22
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2

# Restart SSH
sudo systemctl restart ssh

# Update firewall for new SSH port
sudo ufw delete allow 22/tcp
sudo ufw allow 2222/tcp
```

### 3. Fail2Ban Setup

```bash
# Install fail2ban
sudo apt install -y fail2ban

# Configure for SSH protection
sudo nano /etc/fail2ban/jail.local

# Add configuration:
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Deployment Process

### 1. Deploy Trading Bot

```bash
# Clone your repository
cd ~/options-trader
git clone https://github.com/yourusername/options-trading-bot.git .

# Set up environment
cp .env.example .env
nano .env  # Add your credentials

# Build and start containers
docker-compose build
docker-compose up -d

# Check logs
docker-compose logs -f trader
```

### 2. Verify Deployment

```bash
# Check container status
docker-compose ps

# Check system resources
htop

# Check temperature
vcgencmd measure_temp

# Test dashboard access
curl http://localhost:8050

# Check trading bot health
curl http://localhost:8080/health
```

### 3. Remote Access Setup

```bash
# Install ngrok for secure tunneling (optional)
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-arm64.zip
unzip ngrok-stable-linux-arm64.zip
sudo mv ngrok /usr/local/bin/

# Create ngrok config
mkdir -p ~/.ngrok2
cat > ~/.ngrok2/ngrok.yml << 'EOF'
authtoken: your_ngrok_token
tunnels:
  dashboard:
    addr: 8050
    proto: http
    auth: "username:password"
  ssh:
    addr: 2222
    proto: tcp
EOF

# Start tunnels
ngrok start --all --config ~/.ngrok2/ngrok.yml
```

## Troubleshooting Guide

### Common Issues

```bash
# Container won't start
docker-compose logs trader

# High CPU temperature
# Check cooling, reduce CPU frequency
echo "arm_freq=1200" | sudo tee -a /boot/config.txt

# Out of memory
# Check swap configuration
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Increase CONF_SWAPSIZE

# Network connectivity issues
# Check network configuration
ip route show
sudo systemctl status networking

# SD card corruption (if using SD card)
# Always use SSD for production
sudo fsck /dev/mmcblk0p1
```

### Performance Optimization

```bash
# Monitor performance
# Install performance monitoring
pip install psutil

# Add to your monitoring script
cat >> ~/monitor.sh << 'EOF'

# Network stats
RX_BYTES=$(cat /sys/class/net/eth0/statistics/rx_bytes)
TX_BYTES=$(cat /sys/class/net/eth0/statistics/tx_bytes)

# Load average
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}')

echo "$DATE - Network RX: $RX_BYTES, TX: $TX_BYTES, Load: $LOAD_AVG" >> $LOG_FILE
EOF
```

## Maintenance Schedule

### Daily
- [ ] Check system health logs
- [ ] Verify containers are running
- [ ] Monitor trading performance
- [ ] Check temperature and resource usage

### Weekly
- [ ] Review backup logs
- [ ] Update system packages
- [ ] Restart containers (automated)
- [ ] Check disk space

### Monthly
- [ ] Review security logs
- [ ] Update trading bot code
- [ ] Test backup restoration
- [ ] Performance optimization review

## Emergency Procedures

### 1. Emergency Shutdown

```bash
# Create emergency stop script
cat > ~/emergency_stop.sh << 'EOF'
#!/bin/bash
echo "EMERGENCY STOP INITIATED" | sudo tee -a /var/log/emergency.log
docker-compose -f /home/trader/options-trader/docker-compose.yml stop
sudo shutdown -h now
EOF

chmod +x ~/emergency_stop.sh
```

### 2. Remote Kill Switch

```bash
# Create file that bot checks for emergency stop
# Bot should check for this file every iteration
touch ~/.emergency_stop

# In your trading bot code:
# if os.path.exists(os.path.expanduser('~/.emergency_stop')):
#     await emergency_shutdown()
```

## Cost Analysis

### Hardware (One-time)
- Raspberry Pi 4 (4GB): $75
- SSD + Adapter: $40
- Case with cooling: $20
- Power supply: $15
- **Total: $150**

### Operating Costs
- Electricity (5W × 24h × 365d × $0.12/kWh): $5.26/year
- Internet: Existing connection
- **Total: ~$5/year**

### Comparison with Cloud VPS
- VPS ($10/month): $120/year
- **Raspberry Pi Savings: $115/year after first year**

This setup provides professional-grade 24/7 trading operations at minimal cost with full control over your infrastructure.

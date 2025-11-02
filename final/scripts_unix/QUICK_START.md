# Quick Start - macOS/Linux

## 🚀 One-Time Setup

```bash
# 1. Make scripts executable
chmod +x scripts_unix/*.sh

# 2. Create virtual environment (if not exists)
python3.12 -m venv venv312

# 3. Activate and install dependencies
source venv312/bin/activate
pip install -r requirements.txt
```

## ▶️ Start Everything

```bash
./scripts_unix/run_all.sh
```

**Opens:**
- Backend API: http://localhost:8000
- Gradio UI: http://localhost:7860 (auto-opens in browser)

## ⏹️ Stop Everything

```bash
./scripts_unix/stop_all.sh
```

Or press `Ctrl+C` in the terminal running `run_all.sh`

---

## 📋 Individual Services

### Backend Only
```bash
./scripts_unix/start_backend.sh
```

### Gradio UI Only
```bash
# Backend must be running first!
./scripts_unix/start_gradio.sh
```

### Streamlit UI Only
```bash
# Backend must be running first!
./scripts_unix/start_streamlit.sh
```

---

## 🔍 Check What's Running

```bash
# Check all ports
lsof -i :8000,7860,8501

# Check backend
lsof -i :8000

# Check Gradio
lsof -i :7860

# Check Streamlit
lsof -i :8501
```

---

## 📝 View Logs

When using `run_all.sh`:
```bash
# Backend logs
tail -f backend.log

# Gradio logs
tail -f gradio.log
```

---

## ⚠️ Troubleshooting

### Port already in use?
```bash
./scripts_unix/stop_all.sh
```

### Permission denied?
```bash
chmod +x scripts_unix/*.sh
```

### Backend not responding?
```bash
# Check if running
lsof -i :8000

# View logs
cat backend.log

# Restart
./scripts_unix/stop_all.sh
./scripts_unix/start_backend.sh
```

---

## 💡 Tips

- **Run in background:** Add `&` to any command
  ```bash
  ./scripts_unix/start_backend.sh &
  ```

- **View all processes:**
  ```bash
  ps aux | grep python
  ```

- **Kill specific process:**
  ```bash
  kill -9 <PID>
  ```

- **Check Python version:**
  ```bash
  source venv312/bin/activate
  python --version  # Should be 3.12.x
  ```

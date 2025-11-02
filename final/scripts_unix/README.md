# Unix Scripts (macOS/Linux)

Shell script alternatives for the Windows batch files.

## 📁 Files

- `run_all.sh` - Start both backend and Gradio UI
- `start_backend.sh` - Start backend API only
- `start_gradio.sh` - Start Gradio UI only
- `start_streamlit.sh` - Start Streamlit UI only
- `stop_all.sh` - Stop all services

## 🚀 Usage

### First Time Setup

Make scripts executable:
```bash
chmod +x scripts_unix/*.sh
```

### Start All Services

```bash
./scripts_unix/run_all.sh
```

This will:
1. Kill any existing processes on ports 7860 and 8000
2. Activate venv312 (Python 3.12)
3. Start backend API on http://localhost:8000
4. Start Gradio UI on http://localhost:7860
5. Open browser automatically (macOS/Linux with xdg-open)

### Start Individual Services

**Backend only:**
```bash
./scripts_unix/start_backend.sh
```

**Gradio UI only:**
```bash
./scripts_unix/start_gradio.sh
```

**Streamlit UI only:**
```bash
./scripts_unix/start_streamlit.sh
```

### Stop All Services

```bash
./scripts_unix/stop_all.sh
```

## 📝 Notes

### macOS
- Uses `lsof` to check ports
- Automatically opens browser with `open` command
- Requires Python 3.12 in venv312

### Linux
- Uses `lsof` to check ports
- Attempts to open browser with `xdg-open` (if available)
- Requires Python 3.12 in venv312

### Logs

When using `run_all.sh`, logs are saved to:
- `backend.log` - Backend API logs
- `gradio.log` - Gradio UI logs

### Process Management

The scripts automatically:
- Deactivate conda environments
- Activate venv312
- Check for port conflicts
- Kill existing processes
- Track PIDs for cleanup

### Stopping Services

**Option 1:** Use the stop script
```bash
./scripts_unix/stop_all.sh
```

**Option 2:** Press Ctrl+C in the terminal running `run_all.sh`

**Option 3:** Manually kill by port
```bash
# Kill backend (port 8000)
kill -9 $(lsof -ti:8000)

# Kill Gradio (port 7860)
kill -9 $(lsof -ti:7860)

# Kill Streamlit (port 8501)
kill -9 $(lsof -ti:8501)
```

## 🔧 Troubleshooting

### "Permission denied" error
Make scripts executable:
```bash
chmod +x scripts_unix/*.sh
```

### "lsof: command not found"
Install lsof:
- **macOS:** `brew install lsof` (usually pre-installed)
- **Ubuntu/Debian:** `sudo apt-get install lsof`
- **Fedora/RHEL:** `sudo yum install lsof`

### "No virtual environment found"
Create venv312:
```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

### Port already in use
Stop existing services:
```bash
./scripts_unix/stop_all.sh
```

### Backend not starting
Check logs:
```bash
cat backend.log
```

### Gradio not starting
Check logs:
```bash
cat gradio.log
```

## 🆚 Differences from Windows Scripts

| Feature | Windows (.bat/.ps1) | Unix (.sh) |
|---------|---------------------|------------|
| **Port checking** | `netstat` | `lsof` |
| **Process killing** | `Stop-Process` | `kill -9` |
| **Environment activation** | `venv312\Scripts\Activate.ps1` | `source venv312/bin/activate` |
| **Browser opening** | `Start-Process` | `open` (macOS) / `xdg-open` (Linux) |
| **Background processes** | Separate PowerShell windows | `&` background jobs |
| **Logging** | Separate windows | Log files |

## ✅ Compatibility

- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, Debian, Fedora, Arch, etc.)
- ✅ WSL (Windows Subsystem for Linux)
- ❌ Windows (use .bat or .ps1 scripts instead)

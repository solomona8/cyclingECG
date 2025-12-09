# cyclingECG Quick Start Guide

## The Problem You Were Having

You were trying to run commands from your home directory (`~`) instead of the project directory. This caused:
- `requirements.txt` not found error
- `uvicorn` command not found (dependencies weren't installed)

## Solution: Easy Setup Scripts

### First Time Setup

1. **Navigate to the project directory:**
   ```bash
   cd /Users/aaronsolomon/Documents/LocalCode/cyclingECG
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

   This will:
   - Create/verify the virtual environment
   - Activate it
   - Install all dependencies from requirements.txt

### Starting the Server

After setup, start the server with:
```bash
./start_server.sh
```

The server will be available at:
- Main API: http://0.0.0.0:8000
- API Documentation: http://0.0.0.0:8000/docs

### Manual Method (Alternative)

If you prefer to do it manually:

```bash
# 1. Navigate to project
cd /Users/aaronsolomon/Documents/LocalCode/cyclingECG

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Common Issues

**Issue**: `bash: ./setup.sh: Permission denied`
**Solution**: Make scripts executable:
```bash
chmod +x setup.sh start_server.sh
```

**Issue**: `uvicorn: command not found`
**Solution**: You forgot to activate the virtual environment or install dependencies. Run `./setup.sh`

**Issue**: `requirements.txt not found`
**Solution**: You're in the wrong directory. Navigate to `/Users/aaronsolomon/Documents/LocalCode/cyclingECG` first

### Development Tips

- Always run commands from the project directory `/Users/aaronsolomon/Documents/LocalCode/cyclingECG`
- Keep your virtual environment activated when working
- Use `--reload` flag with uvicorn for development (already included in start_server.sh)

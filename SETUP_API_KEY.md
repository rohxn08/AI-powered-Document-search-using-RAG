# 🔑 Setting Up Your OpenAI API Key

There are multiple ways to configure your OpenAI API key for this application. Choose the method that works best for you.

## 📋 Method 1: Environment Variable (Recommended for Local Development)

### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-api-key-here
```

### Linux/Mac:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Permanent setup** (add to your shell profile `~/.bashrc`, `~/.zshrc`, etc.):
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 📋 Method 2: .env File (Local Development)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. The application will automatically load this when you run `streamlit run app.py`

**Note**: You may need to install `python-dotenv` and update the app to load .env files, or use the Streamlit UI method below.

## 📋 Method 3: Streamlit UI (Easiest for Testing)

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar, find the **"OpenAI API Key"** input field

3. Paste your API key (it will be masked for security)

4. Click **"Initialize System"**

The key will be stored in the session for the current run.

## 📋 Method 4: Docker Environment Variable

### Using Docker Compose:
Edit `docker-compose.yml` and add your key:
```yaml
environment:
  - OPENAI_API_KEY=your-api-key-here
```

Or set it when running:
```bash
OPENAI_API_KEY="your-api-key-here" docker-compose up
```

### Using Docker Run:
```bash
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -e OPENAI_API_KEY="your-api-key-here" \
  ai-doc-search-rag
```

## 🔐 Getting Your API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an OpenAI account
3. Click **"Create new secret key"**
4. Give it a name (e.g., "RAG Document Search")
5. Copy the key immediately (you won't be able to see it again!)
6. Paste it using one of the methods above

## ✅ Verify It's Working

1. Start the application
2. Check the sidebar - you should see:
   - ✅ "OpenAI API key found in environment" (if using env var)
   - ✅ "API key configured" (when key is valid)
3. Try initializing the system and uploading a document

## 🛡️ Security Best Practices

- **Never commit your API key** to version control
- **Never share your API key** publicly
- Use environment variables or .env files (which are in .gitignore)
- Regularly rotate your API keys
- Monitor your API usage at https://platform.openai.com/usage

## ❓ Troubleshooting

**"API key required" error:**
- Make sure your key starts with `sk-`
- Check for extra spaces when copying
- Verify the key is valid at https://platform.openai.com/api-keys

**Key not working in Docker:**
- Ensure you're passing the environment variable correctly
- Check docker-compose.yml has the key in the environment section
- Restart the container after adding the key

**Key not persisting:**
- For permanent setup, use environment variables in your shell profile
- For Docker, use docker-compose.yml or a .env file


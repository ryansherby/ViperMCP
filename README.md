# 🚀 ViperMCP: A Model Context Protocol for Viper Server

> **Mixture-of-Experts VQA, streaming-ready, and MCP-native.**

[![Made with FastMCP](https://img.shields.io/badge/MCP-FastMCP-4B9CE2)](#-setup)
[![ViperGPT Inspired](https://img.shields.io/badge/Inspiration-ViperGPT-6f42c1)](https://viper.cs.columbia.edu/)
[![GPU Ready](https://img.shields.io/badge/GPU-Enabled-0aa344)](#-installation)
[![License](https://img.shields.io/badge/License-Commons-blue.svg)](#-contributions)

ViperMCP is a **mixture-of-experts (MoE) visual question‑answering** (VQA) server that exposes **streamable MCP tools** for:

* 🔎 **Visual grounding**
* 🧩 **Compositional image QA**
* 🌐 **External knowledge‑dependent image QA**

It’s built on the shoulders of 🐍 **[ViperGPT](https://viper.cs.columbia.edu/)** and delivered as a **[FastMCP](https://gofastmcp.com/getting-started/welcome)** HTTP server, so it **just works** with all FastMCP client tooling.

---

## ✨ Highlights

* ⚡ **MCP-native** JSON‑RPC 2.0 endpoint (`/mcp/`) with streaming
* 🧠 **MoE routing** across classic and modern VLMs/LLMs
* 🧰 **Two tools** out of the box: `viper_query` (text) & `viper_task` (crops/masks)
* 🐳 **One‑command Docker** or **pure‑Python** install
* 🔐 **Secure key handling** via env var or secret mount

---

## ⚙️ Setup

### 🔑 OpenAI API Key

An **OpenAI API key** is required. Provide it via **one** of the following:

* `OPENAI_API_KEY` (environment variable)
* `OPENAI_API_KEY_PATH` (path to a file containing the key)
* `?apiKey=...` **HTTP query parameter** (for quick local testing)

### 🌐 Ngrok (Optional)

Use **[ngrok](https://ngrok.com/)** to expose your local server:

```bash
pip install ngrok
ngrok http 8000
```

Use the ngrok URL anywhere you see `http://0.0.0.0:8000` below.

---

## 🛠️ Installation

### 🐳 Option A: Dockerized FastMCP Server (GPU‑ready)

1. Save your key to `api.key`, then run:

```bash
docker run -i --rm \
  --mount type=bind,source=/path/to/api.key,target=/run/secrets/openai_api.key,readonly \
  -e OPENAI_API_KEY_PATH=/run/secrets/openai_api.key \
  -p 8000:8000 \
  rsherby/vipermcp:latest
```

This starts a CUDA‑enabled container serving MCP at:

```
http://0.0.0.0:8000/mcp/
```

> 💡 Prefer building from source? Use the included `docker-compose.yaml`. By default it reads `api.key` from the project root. If your platform injects env vars, you can also set `OPENAI_API_KEY` directly.

---

### 🐍 Option B: Pure FastMCP Server (dev‑friendly)

```bash
git clone --recurse-submodules https://github.com/ryansherby/ViperMCP.git
cd ViperMCP
bash download-models.sh

# Store your key for local dev
echo YOUR_OPENAI_API_KEY > api.key

# (recommended) activate a virtualenv / conda env
pip install -r requirements.txt
pip install -e .

# run the server
python run_server.py
```

Your server should be live at:

```
http://0.0.0.0:8000/mcp/
```

To use OpenAI‑backed models via query param:

```
http://0.0.0.0:8000/mcp?apiKey=sk-proj-XXXXXXXXXXXXXXXXXXXX
```

---

## 🧪 Usage

### 🤝 FastMCP Client Example

Pass images as **base64** (shown) or as **URLs**:

```python
async with client:
    await client.ping()

    tools = await client.list_tools()  # optional

    query = await client.call_tool(
        "viper_query",
        {"query": "how many muffins can each kid have for it to be fair?"},
        {"image": f"data:image/png;base64,{image_base64_string}"},
    )

    task = await client.call_tool(
        "viper_task",
        {"task": "return a mask of all the people in the image"},
        {"image": f"data:image/png;base64,{image_base64_string}"},
    )
```

### 🧵 OpenAI API (MCP Integration)

> ℹ️ The OpenAI MCP integration currently accepts **image URLs** (not raw base64). Send the URL as `type: "input_text"`.

```python
response = client.responses.create(
    model="gpt-4o",
    tools=[
        {
            "type": "mcp",
            "server_label": "ViperMCP",
            "server_url": f"{server_url}/mcp/",
            "require_approval": "never",
        },
    ],
    input=[
        {"role": "system", "content": "Forward any queries or tasks relating to an image directly to the ViperMCP server."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "based on this image, how many muffins can each kid have for it to be fair?"},
                {"type": "input_text", "text": img_url},
            ],
        },
    ],
)
```

---

## 🌐 Endpoints

### 🔓 HTTP GET Endpoints

```
GET /health      => 'OK' (200)
GET /device      => {"device": "cuda"|"mps"|"cpu"}
GET /mcp?apiKey= => 'Query parameters set successfully.'
```

### 🧠 MCP Client Endpoints (JSON‑RPC 2.0)

```
POST /mcp/
```

### 🔨 MCP Client Functions

```
viper_query(query, image) -> str
# Returns a text answer to your query.

viper_task(task, image) -> list[Image]
# Returns a list of images (e.g., masks) satisfying the task.
```

---

## 🧩 Models (Default MoE Pool)

* 🐊 Grounding DINO
* ✂️ Segment Anything (SAM)
* 🤖 GPT‑4o‑mini (LLM)
* 👀 GPT‑4o‑mini (VLM)
* 🧠 GPT‑4.1
* 🔭 X‑VLM
* 🌊 MiDaS (depth)
* 🐝 BERT

> 🧭 The MoE router picks from these based on the tool & prompt.

---

## ⚠️ Security & Production Notes

This package may **generate and execute code** on the host. We include basic injection guards, but you **must** harden for production. A recommended architecture separates concerns:

```
MCP Server (Query + Image)
  => Client Server (Generate Code Request)
    => Backend Server (Generates Code)
      => Client Server (Executes Wrapper Functions)
        => Backend Server (Executes Underlying Functions)
          => Client Server (Return Result)
            => MCP Server (Respond)
```

* 🧱 Isolate codegen & execution.
* 🔒 Lock down secrets & file access.
* 🧪 Add unit/integration tests around wrappers.

---

## 📚 Citations

Huge thanks to the **ViperGPT** team:

```
@article{surismenon2023vipergpt,
    title={ViperGPT: Visual Inference via Python Execution for Reasoning},
    author={D'idac Sur'is and Sachit Menon and Carl Vondrick},
    journal={arXiv preprint arXiv:2303.08128},
    year={2023}
}
```

---

## 🤝 Contributions

PRs welcome! Please:

1. ✅ Ensure all tests in `/tests` pass
2. 🧪 Add coverage for new features
3. 📦 Keep docs & examples up to date

---

## 🧭 Quick Commands Cheat‑Sheet

```bash
# Run with Docker (mount key file)
docker run -i --rm \
  --mount type=bind,source=$(pwd)/api.key,target=/run/secrets/openai_api.key,readonly \
  -e OPENAI_API_KEY_PATH=/run/secrets/openai_api.key \
  -p 8000:8000 rsherby/vipermcp:latest

# From source (after setup)
python run_server.py

# Hit health
curl http://0.0.0.0:8000/health

# List device
curl http://0.0.0.0:8000/device

# Use query param key (local only)
curl "http://0.0.0.0:8000/mcp?apiKey=sk-proj-XXXX..."
```

---

### 💬 Questions?

Open an issue or start a discussion. We ❤️ feedback and ambitious ideas!

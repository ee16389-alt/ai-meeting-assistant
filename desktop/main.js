const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { ensureOllama, ensureModel } = require("./ollama");
const http = require("http");

const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const SUMMARY_ENGINE = (process.env.SUMMARY_ENGINE || "llama_cpp").trim().toLowerCase();
const DEFAULT_GGUF = "qwen2.5-1.5b-instruct-q5_k_m.gguf";

let backendProcess = null;

function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(url + "/health", (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          retry();
        }
      });
      req.on("error", retry);
    };
    const retry = () => {
      if (Date.now() - start > timeoutMs) {
        reject(new Error("Backend timeout"));
        return;
      }
      setTimeout(tryOnce, 500);
    };
    tryOnce();
  });
}

function startBackend() {
  const isProd = app.isPackaged;
  if (isProd) {
    const backendPath = path.join(process.resourcesPath, "backend", "ai_meeting_backend");
    if (!fs.existsSync(backendPath)) {
      dialog.showErrorBox(
        "安裝包缺少後端",
        `找不到後端執行檔：${backendPath}\n請重新打包，並確認 desktop/backend/ai_meeting_backend 存在。`
      );
      app.quit();
      return;
    }
    const packagedModelPath = path.join(process.resourcesPath, "models", "llm", DEFAULT_GGUF);
    const modelPath = (process.env.LLM_MODEL_PATH || packagedModelPath).trim();
    if (!fs.existsSync(modelPath)) {
      dialog.showErrorBox(
        "安裝包缺少摘要模型",
        `找不到 GGUF 模型檔：${modelPath}\n請重新打包，並確認 models/llm/${DEFAULT_GGUF} 存在。`
      );
      app.quit();
      return;
    }
    backendProcess = spawn(backendPath, [], {
      stdio: "inherit",
      env: {
        ...process.env,
        COGNITION_BACKEND: SUMMARY_ENGINE === "ollama" ? "ollama" : "llama_cpp",
        LLM_MODEL_PATH: modelPath,
      },
    });
  } else {
    // Dev: use system python to run app.py
    const projectRoot = path.resolve(__dirname, "..");
    backendProcess = spawn("python3", ["app.py"], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PORT: String(BACKEND_PORT),
        COGNITION_BACKEND: SUMMARY_ENGINE === "ollama" ? "ollama" : "llama_cpp",
      },
      stdio: "inherit",
    });
  }
}

async function createWindow() {
  if (SUMMARY_ENGINE !== "ollama") {
    startBackend();
    try {
      await waitForServer(BACKEND_URL);
    } catch (e) {
      dialog.showErrorBox("後端啟動失敗", "無法連線到後端服務，請稍後再試。");
    }

    const win = new BrowserWindow({
      width: 1280,
      height: 800,
      backgroundColor: "#ffffff",
      webPreferences: {
        contextIsolation: true,
      },
    });
    await win.loadURL(BACKEND_URL);
    return;
  }

  const ollamaReady = await ensureOllama();
  const progressWin = new BrowserWindow({
    width: 520,
    height: 220,
    resizable: false,
    minimizable: false,
    maximizable: false,
    modal: true,
    show: false,
    title: "下載模型中",
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  const progressHtml = `
    <html>
    <body style="font-family: -apple-system, system-ui, sans-serif; background: #fff; margin: 0;">
      <div style="padding: 20px;">
        <h3 style="margin: 0 0 12px 0; color: #111827;">正在下載語言模型...</h3>
        <div style="height: 10px; background: #f3f4f6; border-radius: 999px; overflow: hidden;">
          <div id="bar" style="height: 100%; width: 0%; background: #f36b21; transition: width 0.2s;"></div>
        </div>
        <div id="status" style="margin-top: 12px; font-size: 12px; color: #6b7280;">準備下載</div>
      </div>
      <script>
        const { ipcRenderer } = require('electron');
        ipcRenderer.on('progress', (_, payload) => {
          if (payload && payload.percent !== null) {
            document.getElementById('bar').style.width = payload.percent + '%';
          }
          if (payload && payload.text) {
            document.getElementById('status').textContent = payload.text.trim().slice(0, 200);
          }
        });
      </script>
    </body>
    </html>
  `;

  progressWin.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(progressHtml));
  progressWin.once("ready-to-show", () => progressWin.show());

  let modelOk = false;
  if (ollamaReady) {
    modelOk = await ensureModel((text) => {
      const match = text.match(/(\\d+)%/);
      const percent = match ? parseInt(match[1], 10) : null;
      progressWin.webContents.send("progress", { percent, text });
      if (percent !== null) {
        progressWin.setProgressBar(percent / 100);
      }
    });
  } else {
    progressWin.webContents.send("progress", {
      percent: null,
      text: "已略過 Ollama 安裝，將先啟動 App（摘要功能暫時不可用）",
    });
  }
  progressWin.setProgressBar(-1);
  progressWin.close();
  if (ollamaReady && !modelOk) {
    dialog.showErrorBox("模型下載失敗", "無法下載語言模型，請稍後再試。");
  }
  startBackend();
  try {
    await waitForServer(BACKEND_URL);
  } catch (e) {
    dialog.showErrorBox("後端啟動失敗", "無法連線到後端服務，請稍後再試。");
  }

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: "#ffffff",
    webPreferences: {
      contextIsolation: true,
    },
  });

  await win.loadURL(BACKEND_URL);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (backendProcess) {
    backendProcess.kill();
  }
  app.quit();
});

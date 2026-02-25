const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { ensureOllama, ensureModel } = require("./ollama");
const http = require("http");

const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

let backendProcess = null;

function hasBundledModels() {
  if (!app.isPackaged) return false;

  const ggufDir = path.join(process.resourcesPath, "models", "llm");
  const sherpaDir = path.join(process.resourcesPath, "models", "sherpa-onnx");

  try {
    const hasGguf =
      fs.existsSync(ggufDir) &&
      fs.readdirSync(ggufDir).some((name) => name.toLowerCase().endsWith(".gguf"));
    const hasSherpa = fs.existsSync(sherpaDir);
    return hasGguf && hasSherpa;
  } catch (_) {
    return false;
  }
}

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
    const backendName = process.platform === "win32" ? "ai_meeting_backend.exe" : "ai_meeting_backend";
    const backendPath = path.join(process.resourcesPath, "backend", backendName);
    backendProcess = spawn(backendPath, [], { stdio: "inherit" });
  } else {
    // Dev: use system python to run app.py
    const projectRoot = path.resolve(__dirname, "..");
    backendProcess = spawn("python3", ["app.py"], {
      cwd: projectRoot,
      env: { ...process.env, PORT: String(BACKEND_PORT) },
      stdio: "inherit",
    });
  }
}

async function createWindow() {
  const bundledMode = hasBundledModels();

  if (!bundledMode) {
    const ollamaReady = await ensureOllama();
    if (!ollamaReady) {
      app.quit();
      return;
    }

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

    const sendProgress = (payload) => {
      if (progressWin.isDestroyed()) return;
      if (progressWin.webContents.isDestroyed()) return;
      progressWin.webContents.send("progress", payload);
    };

    const setProgressBarSafe = (value) => {
      if (progressWin.isDestroyed()) return;
      progressWin.setProgressBar(value);
    };

    const modelOk = await ensureModel((text) => {
      const match = text.match(/(\\d+)%/);
      const percent = match ? parseInt(match[1], 10) : null;
      sendProgress({ percent, text });
      if (percent !== null) {
        setProgressBarSafe(percent / 100);
      }
    });
    setProgressBarSafe(-1);
    if (!progressWin.isDestroyed()) {
      progressWin.close();
    }
    if (!modelOk) {
      dialog.showErrorBox("模型下載失敗", "無法下載語言模型，請稍後再試。");
    }
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

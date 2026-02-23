const { app, BrowserWindow, dialog, shell } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { ensureOllama, ensureModel } = require("./ollama");
const http = require("http");
const modelPackConfig = require("./model_pack_config.json");

const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const SUMMARY_ENGINE = (process.env.SUMMARY_ENGINE || "llama_cpp").trim().toLowerCase();
const DEFAULT_GGUF = modelPackConfig.ggufFilename;
const MODEL_VERSION_LABEL = modelPackConfig.versionLabel;
const MODEL_INSTALLER_FILENAME = modelPackConfig.installerFilename;
const MODEL_PACK_DOWNLOAD_URL = "https://github.com/ee16389-alt/ai-meeting-assistant/releases";
const BACKEND_BIN_NAME = process.platform === "win32" ? "ai_meeting_backend.exe" : "ai_meeting_backend";

let backendProcess = null;

function getManagedModelDir() {
  if (process.platform === "win32") {
    const localAppData = process.env.LOCALAPPDATA || path.join(app.getPath("home"), "AppData", "Local");
    return path.join(localAppData, "AI Meeting Assistant", "models", "llm");
  }
  return path.join(app.getPath("userData"), "models", "llm");
}

function findModelInSelectedDirectory(selectedDir) {
  const direct = path.join(selectedDir, DEFAULT_GGUF);
  if (fs.existsSync(direct)) {
    return direct;
  }
  const llmSubdir = path.join(selectedDir, "models", "llm", DEFAULT_GGUF);
  if (fs.existsSync(llmSubdir)) {
    return llmSubdir;
  }
  return null;
}

function promptForModelPath(candidates) {
  while (true) {
    const choice = dialog.showMessageBoxSync({
      type: "warning",
      buttons: ["選擇模型資料夾", "開啟模型包下載頁", "結束"],
      defaultId: 0,
      cancelId: 2,
      noLink: true,
      title: "缺少摘要模型",
      message: "找不到內建摘要模型",
      detail:
        `支援模型版本：${MODEL_VERSION_LABEL}\n` +
        `預期檔名：${DEFAULT_GGUF}\n` +
        `建議安裝模型包：${MODEL_INSTALLER_FILENAME}\n\n` +
        "已檢查路徑：\n" +
        candidates.map((p) => `- ${p}`).join("\n"),
    });

    if (choice === 0) {
      const picked = dialog.showOpenDialogSync({
        title: "選擇模型資料夾（包含 GGUF 檔）",
        properties: ["openDirectory"],
      });
      if (!picked || picked.length === 0) {
        continue;
      }
      const found = findModelInSelectedDirectory(picked[0]);
      if (found) {
        return found;
      }
      dialog.showErrorBox(
        "模型版本不符或檔名錯誤",
        `在所選資料夾中找不到 ${DEFAULT_GGUF}。\n請安裝對應版本模型包（${MODEL_VERSION_LABEL}）。`
      );
      continue;
    }

    if (choice === 1) {
      shell.openExternal(MODEL_PACK_DOWNLOAD_URL);
      continue;
    }

    return null;
  }
}

function resolveProductionModelPath() {
  const envModelPath = (process.env.LLM_MODEL_PATH || "").trim();
  const packagedModelPath = path.join(process.resourcesPath, "models", "llm", DEFAULT_GGUF);
  const managedModelPath = path.join(getManagedModelDir(), DEFAULT_GGUF);
  const candidates = [envModelPath, packagedModelPath, managedModelPath].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return promptForModelPath(candidates);
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
    const backendPath = path.join(process.resourcesPath, "backend", BACKEND_BIN_NAME);
    if (!fs.existsSync(backendPath)) {
      dialog.showErrorBox(
        "安裝包缺少後端",
        `找不到後端執行檔：${backendPath}\n請重新打包，並確認 desktop/backend/ai_meeting_backend 存在。`
      );
      app.quit();
      return;
    }
    const modelPath = resolveProductionModelPath();
    if (!modelPath || !fs.existsSync(modelPath)) {
      if (!modelPath) {
        app.quit();
        return;
      }
      dialog.showErrorBox("安裝包缺少摘要模型", `找不到 GGUF 模型檔：${modelPath}`);
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

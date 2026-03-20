const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const https = require("https");
const http = require("http");
const { spawn, execFile } = require("child_process");

// 禁用 Chromium 內建的語音與翻譯功能，防止出現上方橫幅與權限衝突
app.commandLine.appendSwitch("disable-speech-api");
app.commandLine.appendSwitch("disable-speech-synthesis-api");
app.commandLine.appendSwitch("disable-features", "Translate,LiveCaption");

const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

let backendProcess = null;
let backendRecentLogs = [];

// ── 工具函式 ───────────────────────────────────────────

function appendBackendLog(stream, chunk) {
  const lines = (chunk ? chunk.toString() : "").split(/\r?\n/).filter(Boolean);
  lines.forEach((l) => console.log(`[backend:${stream}] ${l}`));
  backendRecentLogs.push(...lines.map((l) => `[${stream}] ${l}`));
  if (backendRecentLogs.length > 40) backendRecentLogs = backendRecentLogs.slice(-40);
}

function loadModelPackConfig() {
  const candidates = [
    app.isPackaged
      ? path.join(process.resourcesPath, "model_pack_config.json")
      : null,
    path.join(__dirname, "model_pack_config.json"),
  ].filter(Boolean);
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf8"));
    } catch (_) {}
  }
  return {};
}

// ── 模型路徑管理 ───────────────────────────────────────

function modelsBaseDir() {
  // 打包版：AppData/Roaming/AI Meeting Assistant/models
  // 開發版：專案根目錄/desktop/models
  if (app.isPackaged) {
    return path.join(app.getPath("userData"), "models");
  }
  return path.join(__dirname, "models");
}

function hasBundledModels() {
  if (!app.isPackaged) return false;
  const ggufDir = path.join(process.resourcesPath, "models", "llm");
  const sherpaDir = path.join(process.resourcesPath, "models", "sherpa-onnx");
  try {
    const hasGguf =
      fs.existsSync(ggufDir) &&
      fs.readdirSync(ggufDir).some((n) => n.toLowerCase().endsWith(".gguf"));
    return hasGguf && fs.existsSync(sherpaDir);
  } catch (_) {
    return false;
  }
}

function cleanStaleModels() {
  const cfg = loadModelPackConfig();
  if (!cfg.ggufFilename) return;
  const ggufDir = path.join(modelsBaseDir(), "llm");
  if (!fs.existsSync(ggufDir)) return;
  try {
    for (const file of fs.readdirSync(ggufDir)) {
      if (file.toLowerCase().endsWith(".gguf") && file !== cfg.ggufFilename) {
        fs.unlinkSync(path.join(ggufDir, file));
        console.log(`[models] 刪除舊模型: ${file}`);
      }
    }
  } catch (e) {
    console.warn("[models] 清理舊模型失敗:", e.message);
  }
}

function _sherpaEncoderPath(sherpaDir) {
  const int8 = path.join(sherpaDir, "encoder.int8.onnx");
  const fp32 = path.join(sherpaDir, "encoder.onnx");
  return fs.existsSync(int8) ? int8 : fs.existsSync(fp32) ? fp32 : null;
}

function hasDownloadedModels() {
  const cfg = loadModelPackConfig();
  const base = modelsBaseDir();
  const ggufPath = path.join(base, "llm", cfg.ggufFilename || "");
  const sherpaDirName = cfg.sherpaModelDirName || "sherpa-onnx-streaming-paraformer-bilingual-zh-en";
  const sherpaDir = path.join(base, "sherpa-onnx", sherpaDirName);
  return (
    cfg.ggufFilename &&
    fs.existsSync(ggufPath) &&
    !!_sherpaEncoderPath(sherpaDir)
  );
}

function ensureBackendModelCompatPath() {
  if (!app.isPackaged) return;
  const srcModelsDir = path.join(process.resourcesPath, "models");
  const backendInternalDir = path.join(process.resourcesPath, "backend", "_internal");
  const compatModelsDir = path.join(backendInternalDir, "models");
  try {
    if (!fs.existsSync(srcModelsDir) || !fs.existsSync(backendInternalDir)) return;
    if (fs.existsSync(compatModelsDir)) return;
    fs.mkdirSync(backendInternalDir, { recursive: true });
    try {
      fs.symlinkSync(srcModelsDir, compatModelsDir, "junction");
    } catch (_) {
      fs.cpSync(srcModelsDir, compatModelsDir, { recursive: true });
    }
  } catch (e) {
    console.error("[backend] model compat path error:", e);
  }
}

// ── 模型大小估計（用於進度顯示）──
const SHERPA_TOTAL_MB = 220;
const GGUF_TOTAL_MB = 2500; // Phi-4-mini Q4_K_M ≈ 2.5 GB

// ── 下載邏輯 ───────────────────────────────────────────

function downloadFile(url, destPath, onProgress) {
  return new Promise((resolve, reject) => {
    const follow = (currentUrl, redirectCount) => {
      if (redirectCount > 10) return reject(new Error("Too many redirects: " + currentUrl));
      const mod = currentUrl.startsWith("https://") ? https : http;
      const req = mod.get(currentUrl, { headers: { "User-Agent": "Mozilla/5.0" } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          const next = res.headers.location.startsWith("http")
            ? res.headers.location
            : new URL(res.headers.location, currentUrl).href;
          res.resume();
          return follow(next, redirectCount + 1);
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${currentUrl}`));
        }

        const total = parseInt(res.headers["content-length"] || "0", 10);
        let downloaded = 0;
        const tmp = destPath + ".tmp";
        const out = fs.createWriteStream(tmp);

        res.on("data", (chunk) => {
          downloaded += chunk.length;
          if (total > 0) onProgress({ downloaded, total, percent: Math.floor((downloaded / total) * 100) });
          else onProgress({ downloaded, total: 0, percent: -1 });
        });
        res.pipe(out);
        out.on("finish", () => {
          out.close(() => {
            try { fs.renameSync(tmp, destPath); } catch (_) {}
            resolve();
          });
        });
        out.on("error", (e) => { try { fs.unlinkSync(tmp); } catch (_) {} reject(e); });
        res.on("error", reject);
      });
      req.on("error", reject);
      // 僅在完全無資料流動超過 10 分鐘時 timeout（大型模型下載需要充裕時間）
      req.setTimeout(600000, () => { req.destroy(); reject(new Error("Request timeout")); });
    };
    follow(url, 0);
  });
}

async function downloadSherpaModel(archiveUrl, destDir, sendProgress, progressBase, progressRange) {
  fs.mkdirSync(destDir, { recursive: true });
  const isTarBz2 = archiveUrl.includes(".tar.bz2") || archiveUrl.includes(".tgz") || archiveUrl.includes(".tar.gz");
  const archivePath = path.join(destDir, isTarBz2 ? "sherpa-onnx-model.tar.bz2" : "sherpa-onnx-model.zip");
  sendProgress({ stage: "sherpa", percent: progressBase, text: `下載語音辨識模型（約 ${SHERPA_TOTAL_MB} MB）...` });
  await downloadFile(archiveUrl, archivePath, ({ downloaded, total }) => {
    const mb = (downloaded / 1024 / 1024).toFixed(0);
    const totalMb = total > 0 ? `/ ${(total / 1024 / 1024).toFixed(0)} MB` : "";
    const pct = progressBase + (total > 0 ? Math.floor((downloaded / total) * progressRange) : 0);
    sendProgress({
      stage: "sherpa",
      percent: Math.min(progressBase + progressRange - 1, pct),
      text: `下載語音辨識模型... ${mb} MB ${totalMb}`,
    });
  });
  sendProgress({ stage: "sherpa", percent: progressBase + progressRange - 2, text: "解壓縮語音辨識模型..." });
  if (isTarBz2) {
    await extractTar(archivePath, destDir);
  } else {
    await extractZip(archivePath, destDir);
  }
  try { fs.unlinkSync(archivePath); } catch (_) {}
}

function extractTar(archivePath, destDir) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(destDir, { recursive: true });
    // tar 內建於 Windows 10+
    const proc = spawn("tar", ["-xf", archivePath, "-C", destDir], { windowsHide: true });
    proc.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`tar extraction failed with code ${code}`));
    });
    proc.on("error", reject);
  });
}

function extractZip(zipPath, destDir) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(destDir, { recursive: true });
    const ps = spawn("powershell", [
      "-NoProfile", "-NonInteractive", "-Command",
      `Expand-Archive -Path '${zipPath.replace(/'/g, "''")}' -DestinationPath '${destDir.replace(/'/g, "''")}' -Force`,
    ], { windowsHide: true });
    ps.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Expand-Archive failed with code ${code}`));
    });
    ps.on("error", reject);
  });
}

async function ensureModels(sendProgress) {
  const cfg = loadModelPackConfig();
  if (!cfg.ggufFilename || !cfg.ggufDownloadUrl) {
    throw new Error("model_pack_config.json 缺少必要欄位 (gguf)");
  }

  const base = modelsBaseDir();
  const ggufDir = path.join(base, "llm");
  const ggufPath = path.join(ggufDir, cfg.ggufFilename);

  // ── 下載 GGUF ──────────────────────────────────────
  if (!fs.existsSync(ggufPath)) {
    fs.mkdirSync(ggufDir, { recursive: true });
    sendProgress({ stage: "gguf", percent: 0, text: `下載語言模型（約 ${GGUF_TOTAL_MB} MB）...` });
    await downloadFile(cfg.ggufDownloadUrl, ggufPath, ({ percent, downloaded, total }) => {
      const mb = (downloaded / 1024 / 1024).toFixed(0);
      const totalMb = total > 0 ? `/ ${(total / 1024 / 1024).toFixed(0)} MB` : "";
      sendProgress({
        stage: "gguf",
        percent: percent >= 0 ? Math.floor(percent * 0.6) : -1,
        text: `下載語言模型... ${mb} MB ${totalMb}`,
      });
    });
    sendProgress({ stage: "gguf", percent: 60, text: "語言模型下載完成" });
  } else {
    sendProgress({ stage: "gguf", percent: 60, text: "語言模型已存在，跳過下載" });
  }

  // ── 下載 Sherpa-ONNX 模型 ──────────────────────────
  const sherpaDirName = cfg.sherpaModelDirName || "sherpa-onnx-streaming-paraformer-bilingual-zh-en";
  const sherpaBase = path.join(base, "sherpa-onnx");
  const sherpaFinal = path.join(sherpaBase, sherpaDirName);

  if (_sherpaEncoderPath(sherpaFinal)) {
    sendProgress({ stage: "sherpa", percent: 97, text: "語音辨識模型已存在，跳過下載" });
  } else if (cfg.sherpaZipDownloadUrl) {
    sendProgress({ stage: "sherpa", percent: 60, text: "準備下載語音辨識模型（約 1 GB）..." });
    await downloadSherpaModel(cfg.sherpaZipDownloadUrl, sherpaBase, sendProgress, 60, 35);
    // 解壓後確保目錄名稱正確（tar 可能解出不同層級）
    if (!_sherpaEncoderPath(sherpaFinal)) {
      const dirs = fs.readdirSync(sherpaBase, { withFileTypes: true })
        .filter(d => d.isDirectory() && _sherpaEncoderPath(path.join(sherpaBase, d.name)));
      if (dirs.length > 0) {
        fs.renameSync(path.join(sherpaBase, dirs[0].name), sherpaFinal);
      }
    }
  } else {
    sendProgress({ stage: "sherpa", percent: 97, text: "語音辨識模型將於首次啟動時載入..." });
  }

  sendProgress({ stage: "done", percent: 100, text: "模型準備完成，正在啟動..." });
  return { ggufPath, sherpaModelDir: _sherpaEncoderPath(sherpaFinal) ? sherpaFinal : null };
}

// ── 後端啟動 ───────────────────────────────────────────

function waitForServer() {
  // 無 timeout 限制，持續等到後端就緒或進程結束
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      if (backendProcess && backendProcess.exitCode !== null) {
        let msg = `後端程式已意外終止 (代碼: ${backendProcess.exitCode})。`;
        if (backendRecentLogs.length) {
          msg += `\n\n最近日誌：\n${backendRecentLogs.slice(-10).join("\n")}`;
        }
        reject(new Error(msg));
        return;
      }

      const req = http.get(BACKEND_URL + "/health", (res) => {
        let body = "";
        res.on("data", (c) => (body += c));
        res.on("end", () => {
          try {
            const json = JSON.parse(body);
            if (json.ok && json.models_ready) { resolve(); return; }
            if (json.ok && json.stt_error && !json.stt_initializing) {
              reject(new Error(`STT 初始化失敗：${json.stt_error}`));
              return;
            }
          } catch (_) {}
          retry();
        });
      });
      req.on("error", retry);
      req.setTimeout(10000, () => { req.destroy(); retry(); });
    };
    const retry = () => setTimeout(tryOnce, 1500);
    tryOnce();
  });
}

function startBackend(modelEnv = {}) {
  backendRecentLogs = [];
  const exportRoot = path.join(app.getPath("documents"), "AI Meeting Assistant");
  const env = {
    ...process.env,
    PORT: String(BACKEND_PORT),
    AMA_EXPORT_ROOT: exportRoot,
    AMA_DISABLE_OLLAMA_FALLBACK: "1",
    ...modelEnv,
  };

  if (app.isPackaged) {
    const backendExe = path.join(process.resourcesPath, "backend", "ai_meeting_backend.exe");
    backendProcess = spawn(backendExe, [], {
      cwd: path.dirname(backendExe),
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
  } else {
    backendProcess = spawn("python3", ["app.py"], {
      cwd: path.resolve(__dirname, ".."),
      env,
      stdio: "inherit",
    });
  }

  backendProcess.stdout?.on("data", (c) => appendBackendLog("stdout", c));
  backendProcess.stderr?.on("data", (c) => appendBackendLog("stderr", c));
  backendProcess.on("exit", (code) => console.log(`[backend] exited code=${code}`));
}

// ── 進度視窗 ───────────────────────────────────────────

function createProgressWindow() {
  const win = new BrowserWindow({
    width: 520, height: 260,
    resizable: false, minimizable: false, maximizable: false,
    show: false, title: "AI 會議助理 — 初始化",
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  });

  win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(`<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,system-ui,sans-serif; background:#fff8f3; padding:28px; }
h3 { font-size:1rem; color:#111827; margin-bottom:6px; }
p  { font-size:0.8rem; color:#6b7280; margin-bottom:20px; }
.track { height:8px; background:#fed7aa; border-radius:99px; overflow:hidden; }
.bar   { height:100%; background:#f36b21; border-radius:99px; width:0%; transition:width 0.3s; }
.status { margin-top:12px; font-size:0.78rem; color:#92400e; min-height:1.2em; }
</style></head>
<body>
  <h3>AI 會議助理 — 首次啟動</h3>
  <p>正在下載 AI 模型（語言模型 ~1.8 GB + 語音辨識 ~220 MB），下載完成後即可離線使用。</p>
  <div class="track"><div id="bar" class="bar"></div></div>
  <div id="status" class="status">準備中...</div>
  <script>
    const { ipcRenderer } = require('electron');
    ipcRenderer.on('progress', (_, p) => {
      if (p.percent >= 0) document.getElementById('bar').style.width = p.percent + '%';
      if (p.text) document.getElementById('status').textContent = p.text;
    });
  </script>
</body></html>`));

  win.once("ready-to-show", () => win.show());
  return win;
}

// ── 主流程 ─────────────────────────────────────────────

async function createWindow() {
  let modelEnv = {};

  if (hasBundledModels()) {
    // 已打包模型（未來可能的完整包版本）
    process.env.AMA_DISABLE_OLLAMA_FALLBACK = "1";
    ensureBackendModelCompatPath();
  } else {
    // 清理版本不符的舊 GGUF，確保重新下載正確版本
    cleanStaleModels();
    // 首次下載 or 已下載過
    if (!hasDownloadedModels()) {
      const progressWin = createProgressWindow();
      const send = (payload) => {
        if (!progressWin.isDestroyed()) {
          progressWin.webContents.send("progress", payload);
          if (payload.percent >= 0) progressWin.setProgressBar(payload.percent / 100);
        }
      };

      try {
        const { ggufPath, sherpaModelDir } = await ensureModels(send);
        modelEnv = {
          AMA_GGUF_PATH: ggufPath,
          HF_HOME: path.join(app.getPath("userData"), "hf_cache"),
          ...(sherpaModelDir ? { AMA_SHERPA_DIR: sherpaModelDir } : {}),
        };
      } catch (err) {
        if (!progressWin.isDestroyed()) progressWin.close();
        dialog.showErrorBox(
          "模型下載失敗",
          `無法下載 AI 模型：\n\n${err.message}\n\n請確認網路連線後重新啟動。`
        );
        app.quit();
        return;
      }

      progressWin.setProgressBar(-1);
      if (!progressWin.isDestroyed()) progressWin.close();
    } else {
      // 已下載過，直接讀路徑
      const cfg = loadModelPackConfig();
      const base = modelsBaseDir();
      const sherpaDirName = cfg.sherpaModelDirName || "sherpa-onnx-streaming-paraformer-bilingual-zh-en";
      const sherpaDir = path.join(base, "sherpa-onnx", sherpaDirName);
      modelEnv = {
        AMA_GGUF_PATH: path.join(base, "llm", cfg.ggufFilename),
        HF_HOME: path.join(app.getPath("userData"), "hf_cache"),
        ...(_sherpaEncoderPath(sherpaDir) ? { AMA_SHERPA_DIR: sherpaDir } : {}),
      };
    }
  }

  // 主視窗（先顯示載入畫面）
  const mainWin = new BrowserWindow({
    width: 1280, height: 800,
    backgroundColor: "#fff8f3",
    show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });

  mainWin.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(`<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:linear-gradient(135deg,#fff8f3,#f8fafc,#f3f7f2);display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:-apple-system,sans-serif;color:#1f2937;}
.s{width:36px;height:36px;border:3px solid #fed7aa;border-top-color:#f36b21;border-radius:50%;animation:spin .8s linear infinite;margin-bottom:14px;}
@keyframes spin{to{transform:rotate(360deg)}}
.t{font-size:1.1rem;font-weight:600;margin-bottom:6px;}
.h{font-size:.8rem;color:#94a3b8;}
</style></head>
<body><div class="s"></div><div class="t">AI 會議助理</div><div class="h">正在載入 AI 模型，請稍候...</div></body></html>`));

  mainWin.once("ready-to-show", () => mainWin.show());

  startBackend(modelEnv);

  try {
    await waitForServer();
    await mainWin.loadURL(BACKEND_URL);
  } catch (e) {
    let msg = `後端服務無法就緒。\n\n${e.message}`;
    if (backendRecentLogs.length) msg += `\n\n最近日誌：\n${backendRecentLogs.slice(-6).join("\n")}`;
    dialog.showErrorBox("啟動失敗", msg);
    app.quit();
  }
}

app.whenReady().then(createWindow);

// 強制終止後端子程序（不等待 LLM 推理完成）
function killBackend() {
  if (!backendProcess) return;
  const proc = backendProcess;
  backendProcess = null;
  try { proc.kill("SIGTERM"); } catch (_) {}
  // 500ms 後強制 SIGKILL，確保 LLM 推理中也能立即退出
  setTimeout(() => {
    try { proc.kill("SIGKILL"); } catch (_) {}
  }, 500);
}

app.on("before-quit", killBackend);

app.on("window-all-closed", () => {
  killBackend();
  app.quit();
});

const { spawnSync, spawn } = require("child_process");
const https = require("https");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { dialog, shell } = require("electron");

const OLLAMA_PKG_URL = "https://ollama.com/download/Ollama.pkg";
const MODEL = "qwen2.5:1.5b";
const BUNDLED_MODEL_DIR = "ollama-models";

function hasOllama() {
  const result = spawnSync("which", ["ollama"]);
  return result.status === 0;
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: ${response.statusCode}`));
        return;
      }
      response.pipe(file);
      file.on("finish", () => file.close(resolve));
    }).on("error", reject);
  });
}

async function installOllamaMac() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ollama-"));
  const pkgPath = path.join(tmpDir, "Ollama.pkg");
  await downloadFile(OLLAMA_PKG_URL, pkgPath);

  const script = `do shell script "installer -pkg '${pkgPath}' -target /" with administrator privileges`;
  const result = spawnSync("osascript", ["-e", script], { stdio: "inherit" });
  return result.status === 0;
}

async function ensureOllama() {
  if (hasOllama()) return true;

  const choice = dialog.showMessageBoxSync({
    type: "warning",
    buttons: ["自動安裝", "略過（先啟動）", "開啟下載頁"],
    defaultId: 0,
    cancelId: 1,
    message: "未偵測到 Ollama，是否自動安裝？",
    detail:
      "若你不想現在安裝，可先略過啟動。\n之後可手動安裝：\n1) brew install --cask ollama\n2) open -a Ollama",
  });

  if (choice === 2) {
    shell.openExternal("https://ollama.com/download");
    return false;
  }
  if (choice === 1) {
    return false;
  }

  try {
    const ok = await installOllamaMac();
    if (!ok) {
      const next = dialog.showMessageBoxSync({
        type: "error",
        buttons: ["重試自動安裝", "開啟下載頁", "略過（先啟動）"],
        defaultId: 0,
        cancelId: 2,
        message: "Ollama 安裝失敗",
        detail:
          "請改用手動安裝：\n1) brew install --cask ollama\n2) open -a Ollama\n\n你也可以先略過安裝進入 App（摘要功能暫時不可用）。",
      });
      if (next === 0) return ensureOllama();
      if (next === 1) shell.openExternal("https://ollama.com/download");
      return false;
    }
    return hasOllama();
  } catch (e) {
    const next = dialog.showMessageBoxSync({
      type: "error",
      buttons: ["重試自動安裝", "開啟下載頁", "略過（先啟動）"],
      defaultId: 0,
      cancelId: 2,
      message: "Ollama 安裝失敗",
      detail:
        "系統安裝流程未完成。\n請改用手動安裝：\n1) brew install --cask ollama\n2) open -a Ollama\n\n你也可以先略過安裝進入 App（摘要功能暫時不可用）。",
    });
    if (next === 0) return ensureOllama();
    if (next === 1) shell.openExternal("https://ollama.com/download");
    return false;
  }
}

function ensureModel(onProgress) {
  const result = spawnSync("ollama", ["list"], { encoding: "utf8" });
  if (result.status !== 0) return Promise.resolve(false);
  if (result.stdout && result.stdout.includes(MODEL)) return Promise.resolve(true);

  if (importBundledModel(onProgress)) {
    const check = spawnSync("ollama", ["list"], { encoding: "utf8" });
    if (check.status === 0 && check.stdout && check.stdout.includes(MODEL)) {
      return Promise.resolve(true);
    }
  }

  return new Promise((resolve) => {
    const pull = spawn("ollama", ["pull", MODEL]);
    pull.stdout.on("data", (data) => {
      const text = data.toString();
      if (onProgress) onProgress(text);
    });
    pull.stderr.on("data", (data) => {
      const text = data.toString();
      if (onProgress) onProgress(text);
    });
    pull.on("close", (code) => resolve(code === 0));
  });
}

function modelStoreRoot() {
  if (process.env.OLLAMA_MODELS && process.env.OLLAMA_MODELS.trim()) {
    return process.env.OLLAMA_MODELS.trim();
  }
  return path.join(os.homedir(), ".ollama", "models");
}

function importBundledModel(onProgress) {
  try {
    const bundledRoot = path.join(process.resourcesPath, BUNDLED_MODEL_DIR);
    if (!fs.existsSync(bundledRoot)) return false;

    const targetRoot = modelStoreRoot();
    fs.mkdirSync(targetRoot, { recursive: true });
    if (onProgress) onProgress("正在匯入內建模型檔案...");

    const srcManifests = path.join(bundledRoot, "manifests");
    const srcBlobs = path.join(bundledRoot, "blobs");
    if (fs.existsSync(srcManifests)) {
      fs.cpSync(srcManifests, path.join(targetRoot, "manifests"), { recursive: true, force: true });
    }
    if (fs.existsSync(srcBlobs)) {
      fs.cpSync(srcBlobs, path.join(targetRoot, "blobs"), { recursive: true, force: true });
    }
    return true;
  } catch (e) {
    if (onProgress) onProgress(`內建模型匯入失敗，改用下載：${String(e)}`);
    return false;
  }
}

module.exports = { ensureOllama, ensureModel };

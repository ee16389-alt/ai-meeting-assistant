const { spawnSync, spawn } = require("child_process");
const https = require("https");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { dialog, shell } = require("electron");

const OLLAMA_PKG_URL = "https://ollama.com/download/Ollama.pkg";
const MODEL = "qwen2.5:1.5b";

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
  if (hasOllama()) return;

  const choice = dialog.showMessageBoxSync({
    type: "warning",
    buttons: ["自動安裝", "取消"],
    defaultId: 0,
    message: "未偵測到 Ollama，是否自動安裝？",
  });

  if (choice !== 0) {
    shell.openExternal("https://ollama.com/download");
    return;
  }

  try {
    const ok = await installOllamaMac();
    if (!ok) {
      dialog.showErrorBox("安裝失敗", "Ollama 安裝失敗，請手動安裝。");
      shell.openExternal("https://ollama.com/download");
    }
  } catch (e) {
    dialog.showErrorBox("安裝失敗", "Ollama 安裝失敗，請手動安裝。");
    shell.openExternal("https://ollama.com/download");
  }
}

function ensureModel(onProgress) {
  const result = spawnSync("ollama", ["list"], { encoding: "utf8" });
  if (result.status !== 0) return Promise.resolve(false);
  if (result.stdout && result.stdout.includes(MODEL)) return Promise.resolve(true);

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

module.exports = { ensureOllama, ensureModel };

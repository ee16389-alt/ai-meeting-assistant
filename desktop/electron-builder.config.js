const fs = require("fs");
const path = require("path");

function isTrue(value) {
  const v = String(value || "").trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

const bundleGguf = isTrue(process.env.BUNDLE_GGUF);
const bundleOllamaModels = isTrue(process.env.BUNDLE_OLLAMA_MODELS);
const artifactSuffix = String(process.env.BUILD_ARTIFACT_SUFFIX || "");

const extraResources = [
  {
    from: "backend",
    to: "backend",
    filter: ["**/*"],
  },
];

if (bundleGguf) {
  extraResources.push({
    from: "models/llm",
    to: "models/llm",
    filter: ["*.gguf"],
  });
}

if (bundleOllamaModels) {
  extraResources.push({
    from: "ollama-models",
    to: "ollama-models",
    filter: ["**/*"],
  });
}

const vcRedistPath = path.join(__dirname, "prereqs", "vc_redist.x64.exe");
if (fs.existsSync(vcRedistPath)) {
  extraResources.push({
    from: "prereqs/vc_redist.x64.exe",
    to: "prereqs/vc_redist.x64.exe",
  });
}

const vbcablePath = path.join(__dirname, "prereqs", "VBCABLE_Driver_Pack.zip");
if (fs.existsSync(vbcablePath)) {
  extraResources.push({
    from: "prereqs/VBCABLE_Driver_Pack.zip",
    to: "prereqs/VBCABLE_Driver_Pack.zip",
  });
}

module.exports = {
  appId: "com.minashih.ai-meeting-assistant",
  productName: "AI Meeting Assistant",
  artifactName: `\${productName}-\${version}-\${arch}${artifactSuffix}.\${ext}`,
  compression: "store",
  files: [
    "**/*",
    "!backend/**",
    "!models/**",
    "!ollama-models/**",
    "!dist/**",
  ],
  extraResources,
  mac: {
    category: "public.app-category.productivity",
    target: ["dmg"],
  },
  win: {
    target: ["nsis"],
  },
  nsis: {
    include: "build/installer.nsh",
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    perMachine: false,
    createDesktopShortcut: "always",
    createStartMenuShortcut: true,
    shortcutName: "AI Meeting Assistant",
    uninstallDisplayName: "AI Meeting Assistant",
    deleteAppDataOnUninstall: false,
  },
};

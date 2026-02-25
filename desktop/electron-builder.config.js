function isTrue(value) {
  const v = String(value || "").trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

const bundleGguf = isTrue(process.env.BUNDLE_GGUF);
const bundleSherpaModels = isTrue(process.env.BUNDLE_SHERPA_MODELS);
const bundleOllamaModels = isTrue(process.env.BUNDLE_OLLAMA_MODELS);

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

if (bundleSherpaModels) {
  extraResources.push({
    from: "models/sherpa-onnx",
    to: "models/sherpa-onnx",
    filter: ["**/*"],
  });
}

if (bundleOllamaModels) {
  extraResources.push({
    from: "ollama-models",
    to: "ollama-models",
    filter: ["**/*"],
  });
}

module.exports = {
  appId: "com.minashih.ai-meeting-assistant",
  productName: "AI Meeting Assistant",
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
};

# Windows 安裝說明

## 一般安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 若出現 WinError 0xc000001d

這個錯誤代表 `llama-cpp-python` 編譯時啟用了 CPU 不支援的指令集（如 AVX-512）。
請以下列方式重新安裝，強制使用 AVX2 並關閉 AVX-512：

```bat
set CMAKE_ARGS=-DLLAMA_AVX2=on -DLLAMA_AVX512=off
pip install llama-cpp-python --force-reinstall
```

或使用預編譯的 AVX2 wheel（無需本機編譯，速度較快）：

```bat
pip install llama-cpp-python ^
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu ^
  --force-reinstall
```

安裝完成後重新啟動後端即可。

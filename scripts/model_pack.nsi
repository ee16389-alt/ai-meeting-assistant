; AI Meeting Assistant Model Pack Installer
; 安裝路徑：%APPDATA%\AI Meeting Assistant\models\

!define PRODUCT_NAME "AI Meeting Assistant Model Pack"
!define MODELS_BASE "$APPDATA\AI Meeting Assistant\models"

; 讀取版本標籤（由 makensis 命令列 /D 傳入）
!ifndef VERSION_LABEL
  !define VERSION_LABEL "model-pack"
!endif

OutFile "AI-Meeting-Assistant-Model-Pack-${VERSION_LABEL}-Setup.exe"
InstallDir "${MODELS_BASE}"
RequestExecutionLevel user
ShowInstDetails show

; 不壓縮大型二進制檔案，避免 NSIS mmap 限制
SetCompress off

Name "${PRODUCT_NAME}"

Page instfiles

Section "Install Models" SEC01
  ; ── GGUF 語言模型 ──────────────────────────────────
  SetOutPath "${MODELS_BASE}\llm"
  File /r "models\llm\*.gguf"

  ; ── Sherpa-ONNX 語音辨識模型 ──────────────────────
  SetOutPath "${MODELS_BASE}\sherpa-onnx\sherpa-onnx-streaming-paraformer-bilingual-zh-en"
  File /r "models\sherpa-onnx\sherpa-onnx-streaming-paraformer-bilingual-zh-en\*.*"

  ; ── 寫入解除安裝資訊 ──────────────────────────────
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AIMeetingAssistantModels" \
    "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AIMeetingAssistantModels" \
    "UninstallString" '"$INSTDIR\uninstall_models.exe"'
  WriteUninstaller "$INSTDIR\uninstall_models.exe"

  MessageBox MB_OK "AI 模型安裝完成！請開啟 AI Meeting Assistant 使用。"
SectionEnd

Section "Uninstall"
  RMDir /r "${MODELS_BASE}\llm"
  RMDir /r "${MODELS_BASE}\sherpa-onnx"
  Delete "$INSTDIR\uninstall_models.exe"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AIMeetingAssistantModels"
SectionEnd

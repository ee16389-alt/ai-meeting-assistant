!include "LogicLib.nsh"

Var VcRedistExitCode

!macro customInstall
  ; Auto-install Microsoft VC++ Runtime if missing (required by native libs, e.g. llama-cpp)
  StrCpy $0 0
  SetRegView 64
  ClearErrors
  ReadRegDWORD $1 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  ${IfNot} ${Errors}
    ${If} $1 = 1
      StrCpy $0 1
    ${EndIf}
  ${EndIf}
  ${If} $0 = 1
    DetailPrint "Microsoft Visual C++ Runtime already installed. Skipping."
  ${Else}
    IfFileExists "$INSTDIR\resources\prereqs\vc_redist.x64.exe" has_vcredist missing_vcredist
has_vcredist:
    DetailPrint "Installing Microsoft Visual C++ Runtime (x64)..."
    ExecWait '"$INSTDIR\resources\prereqs\vc_redist.x64.exe" /install /quiet /norestart' $VcRedistExitCode
    ${If} $VcRedistExitCode = 0
    ${OrIf} $VcRedistExitCode = 1638
    ${OrIf} $VcRedistExitCode = 3010
    ${OrIf} $VcRedistExitCode = 1641
      DetailPrint "VC++ Runtime install result code: $VcRedistExitCode"
    ${Else}
      MessageBox MB_ICONEXCLAMATION|MB_OK "Microsoft Visual C++ Runtime 安裝失敗 (code=$VcRedistExitCode)。程式可能無法啟動。"
    ${EndIf}
    Goto done_vcredist
missing_vcredist:
    MessageBox MB_ICONEXCLAMATION|MB_OK "找不到 vc_redist.x64.exe，將略過 VC++ Runtime 安裝。程式可能無法啟動。"
    done_vcredist:
  ${EndIf}
!macroend

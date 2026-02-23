Unicode true
ManifestDPIAware true
RequestExecutionLevel user

!include "MUI2.nsh"

!ifndef MODEL_FILE
  !error "MODEL_FILE define is required"
!endif
!ifndef MODEL_FILENAME
  !error "MODEL_FILENAME define is required"
!endif
!ifndef MODEL_VERSION_LABEL
  !error "MODEL_VERSION_LABEL define is required"
!endif
!ifndef OUTPUT_EXE
  !error "OUTPUT_EXE define is required"
!endif

Name "AI Meeting Assistant Model Pack (${MODEL_VERSION_LABEL})"
OutFile "${OUTPUT_EXE}"
InstallDir "$LOCALAPPDATA\AI Meeting Assistant\models\llm"
InstallDirRegKey HKCU "Software\AI Meeting Assistant" "ModelDir"

!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "TradChinese"

Section "InstallModel"
  SetOutPath "$INSTDIR"
  File /oname=${MODEL_FILENAME} "${MODEL_FILE}"
  WriteRegStr HKCU "Software\AI Meeting Assistant" "ModelDir" "$INSTDIR"
SectionEnd

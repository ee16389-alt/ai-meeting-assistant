[Setup]
AppName=AI Meeting Assistant Model Pack
AppVersion={#VERSION_LABEL}
AppPublisher=minashih
DefaultDirName={userappdata}\AI Meeting Assistant\models
DisableDirPage=yes
OutputDir=.
OutputBaseFilename=AI-Meeting-Assistant-Model-Pack-{#VERSION_LABEL}-Setup
Compression=none
SolidCompression=no
PrivilegesRequired=lowest
ShowLanguageDialog=no

[Messages]
FinishedLabel=AI 模型安裝完成！請開啟 AI Meeting Assistant 使用。

[Files]
Source: "models\llm\*.gguf"; DestDir: "{userappdata}\AI Meeting Assistant\models\llm"; Flags: ignoreversion
Source: "models\sherpa-onnx\sherpa-onnx-streaming-paraformer-bilingual-zh-en\*"; DestDir: "{userappdata}\AI Meeting Assistant\models\sherpa-onnx\sherpa-onnx-streaming-paraformer-bilingual-zh-en"; Flags: ignoreversion recursesubdirs

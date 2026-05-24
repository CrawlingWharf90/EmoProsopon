[Setup]
AppName=EmoProsopon
AppVersion=1.0.0
; Install into the user's local app data (no admin rights required)
DefaultDirName={localappdata}\EmoProsopon
DefaultGroupName=EmoProsopon
OutputDir=.\Output
OutputBaseFilename=EmoProsopon_Installer_v1.0
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
; Tells Windows to refresh the environment variables after install
ChangesEnvironment=yes
; We can theme this later for a sleek, dark aesthetic
WizardStyle=modern

[Files]
; Copy everything from the root directory, but ignore git, vscode, and the installers folder
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: ".git\*, installers\*, .vscode\*, __pycache__\*, *.pyc"

[Registry]
; Safely append the installation directory to the User PATH so 'eop' works globally
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
// A safety check to ensure we don't add the path twice if they reinstall
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
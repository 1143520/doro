@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

:: Store root directory path
set "ROOT_DIR=%CD%"

:: Initialize check
echo [INFO] Starting GitHub Auto Upload Tool...
echo [INFO] Current directory: %CD%
pause

title GitHub Auto Upload Tool

:: Check Git installation
echo [INFO] Checking Git installation...
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not installed! Please install Git first.
    pause
    goto :eof
)

:: Create config folder
if not exist ".git-upload-config" (
    echo [INFO] Creating config folder...
    mkdir ".git-upload-config" 2>nul
    if errorlevel 1 (
        echo [ERROR] Failed to create config folder
        pause
        goto :eof
    )
)

:: Create .gitignore
if not exist ".git-upload-config\.gitignore" (
    echo [INFO] Creating .gitignore file...
    (
        echo .git-upload-config/
    ) > ".git-upload-config\.gitignore"
)

:: Remove old .gitignore
if exist ".gitignore" (
    echo [INFO] Removing old .gitignore file...
    del /f /q ".gitignore" >nul 2>&1
)

:: Configure git exclude file
echo [INFO] Configuring git exclude file...
git config core.excludesFile ".git-upload-config\.gitignore" >nul 2>&1

:: Check proxy config
if exist ".git-upload-config\proxy_config.txt" (
    set /p PROXY_PORT=<.git-upload-config\proxy_config.txt
    echo [INFO] Current proxy port: !PROXY_PORT!
    set /p "CHANGE_PROXY=Change proxy port? (Y/N, default=N): "
    if /i "!CHANGE_PROXY!"=="Y" (
        set /p "PROXY_PORT=Enter new proxy port (default 7890): "
        if "!PROXY_PORT!"=="" set "PROXY_PORT=7890"
        echo !PROXY_PORT!> .git-upload-config\proxy_config.txt
        echo [INFO] New proxy port saved
    ) else (
        echo [INFO] Using saved proxy port: !PROXY_PORT!
    )
) else (
    echo [INFO] First time proxy setup...
    set /p "PROXY_PORT=Enter proxy port (default 7890): "
    if "!PROXY_PORT!"=="" set "PROXY_PORT=7890"
    echo !PROXY_PORT!> .git-upload-config\proxy_config.txt
    echo [INFO] Proxy port saved
)

:: Set proxy with timeout settings
echo [INFO] Setting up Git proxy...
git config --global http.proxy "http://127.0.0.1:!PROXY_PORT!" 2>nul
git config --global https.proxy "http://127.0.0.1:!PROXY_PORT!" 2>nul
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 10
git config --global http.postBuffer 524288000

:: Repository config
if not exist ".git-upload-config\repo_config.txt" (
    echo [INFO] First time repository setup...
    :input_repo
    set /p "REPO_URL=Enter GitHub repository URL: "
    if "!REPO_URL!"=="" (
        echo [ERROR] Repository URL cannot be empty
        goto input_repo
    )
    echo !REPO_URL!> .git-upload-config\repo_config.txt
    echo [INFO] Repository URL saved: !REPO_URL!
) else (
    set /p REPO_URL=<.git-upload-config\repo_config.txt
    if "!REPO_URL!"=="" (
        echo [ERROR] Repository URL is empty in config
        del .git-upload-config\repo_config.txt
        goto :eof
    )
    echo [INFO] Using saved repository: !REPO_URL!
)

:: Git initialization
if not exist .git (
    echo [INFO] Initializing git repository...
    git init
    if errorlevel 1 (
        echo [ERROR] Failed to initialize git repository
        goto :eof
    )
)

:: Configure remote
echo [INFO] Configuring remote...
git remote remove origin 2>nul
git remote add origin !REPO_URL!
if errorlevel 1 (
    echo [ERROR] Failed to add remote origin
    goto :eof
)

:: Initial git setup
git config --global core.autocrlf true
git config --global core.safecrlf false

:: Initial branch setup
git checkout -b main 2>nul
if errorlevel 1 (
    git checkout main 2>nul
)

goto show_folder_menu

:show_folder_menu
cls
echo ====================================
echo    GitHub Auto Upload Tool
echo ====================================
echo Current path: %CD%
echo.
echo [0] Upload all files
echo [B] Back to parent directory
echo.
echo === Folders ===

:: List folders first
set itemCount=0
for /f "tokens=*" %%F in ('dir /b /ad') do (
    if "%%F" neq ".git-upload-config" (
        if "%%F" neq ".git" (
            if not exist "%%F\.git" (
                set /a itemCount+=1
                set "item_!itemCount!=%%F"
                set "type_!itemCount!=folder"
                echo [!itemCount!] %%F
            )
        )
    )
)

echo.
echo === Files ===

:: Then list files
for /f "tokens=*" %%F in ('dir /b /a-d') do (
    if "%%F" neq ".git-upload-config" (
        set /a itemCount+=1
        set "item_!itemCount!=%%F"
        set "type_!itemCount!=file"
        echo [!itemCount!] %%F
    )
)

echo.
echo Total items: %itemCount%
echo Enter numbers separated by space to select multiple items
set /p "choice=Select number(s) (0=all, B=back): "

:: Check for back command
if /i "!choice!"=="B" (
    cd ..
    goto show_folder_menu
)

:: Simple validation
if "!choice!"=="0" goto process_upload
if "!choice!"=="" goto show_folder_menu

:: Process selection
for %%i in (!choice!) do (
    if defined item_%%i (
        set "selected=!item_%%i!"
        if "!type_%%i!"=="folder" (
            echo [INFO] Entering directory: !selected!
            cd "!selected!"
            goto show_folder_menu
        )
    )
)

:process_upload
:: Store current directory
set "CURRENT_DIR=%CD%"

:: Change to root directory for git operations
cd /d "%ROOT_DIR%"

if "!choice!"=="0" (
    echo [INFO] Adding all files...
    :: If in subdirectory, only add files from that directory
    if /i not "%CURRENT_DIR%"=="%ROOT_DIR%" (
        set "REL_PATH=!CURRENT_DIR:%ROOT_DIR%\=!"
        :: Skip .git-upload-config and git repositories
        for /f "tokens=*" %%F in ('dir /b "!CURRENT_DIR!"') do (
            if "%%F" neq ".git-upload-config" (
                if "%%F" neq ".git" (
                    if not exist "!CURRENT_DIR!\%%F\.git" (
                        git add "!REL_PATH!\%%F"
                        echo [INFO] Added: !REL_PATH!\%%F
                    ) else (
                        echo [SKIP] Git repository: !REL_PATH!\%%F
                    )
                )
            )
        )
    ) else (
        :: In root directory, add all files
        for /f "tokens=*" %%F in ('dir /b') do (
            if "%%F" neq ".git-upload-config" (
                if "%%F" neq ".git" (
                    if not exist "%%F\.git" (
                        git add "%%F"
                        echo [INFO] Added: %%F
                    ) else (
                        echo [SKIP] Git repository: %%F
                    )
                )
            ) else (
                echo [SKIP] Config folder: %%F
            )
        )
    )
) else (
    echo [INFO] Adding selected files...
    :: Process multiple selections
    for %%i in (!choice!) do (
        if defined item_%%i (
            if "!type_%%i!"=="file" (
                set "selected=!item_%%i!"
                if /i not "%CURRENT_DIR%"=="%ROOT_DIR%" (
                    set "REL_PATH=!CURRENT_DIR:%ROOT_DIR%\=!"
                    git add "!REL_PATH!\!selected!"
                    echo [INFO] Added: !REL_PATH!\!selected!
                ) else (
                    git add "!selected!"
                    echo [INFO] Added: !selected!
                )
            )
        )
    )
)

:: Commit changes
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set COMMIT_DATE=!datetime:~0,4!-!datetime:~4,2!-!datetime:~6,2!
set COMMIT_TIME=!datetime:~8,2!:!datetime:~10,2!

:: Check if there are changes to commit
git diff --cached --quiet
if errorlevel 1 (
    echo [INFO] Committing changes...
    git commit -m "[BAT] !COMMIT_DATE! !COMMIT_TIME!"
    if errorlevel 1 (
        echo [ERROR] Failed to commit changes
        cd /d "!CURRENT_DIR!"
        pause
        goto show_folder_menu
    )

    echo [INFO] Pushing to GitHub...
    git push -u origin main --force
    if errorlevel 1 (
        echo [ERROR] Failed to push to GitHub
        echo [INFO] Retrying with different settings...
        
        :: Try with different proxy settings
        git config --global http.proxy "socks5://127.0.0.1:!PROXY_PORT!" 2>nul
        git config --global https.proxy "socks5://127.0.0.1:!PROXY_PORT!" 2>nul
        
        git push -u origin main --force
        if errorlevel 1 (
            echo [ERROR] Push failed again
            echo [INFO] Restoring original proxy settings...
            git config --global http.proxy "http://127.0.0.1:!PROXY_PORT!" 2>nul
            git config --global https.proxy "http://127.0.0.1:!PROXY_PORT!" 2>nul
            cd /d "!CURRENT_DIR!"
            pause
            goto show_folder_menu
        )
    )
) else (
    echo [WARNING] No changes to commit
    cd /d "!CURRENT_DIR!"
    pause
    goto show_folder_menu
)

:: Return to original directory
cd /d "!CURRENT_DIR!"

echo.
echo ====================================
echo [SUCCESS] Upload completed!
echo ====================================
echo Repository: !REPO_URL!
echo.
pause
goto show_folder_menu

:exit_script
echo [INFO] Cleaning up proxy settings...
git config --global --unset http.proxy
git config --global --unset https.proxy
echo [INFO] Cleanup completed
exit /b

:eof
goto exit_script 
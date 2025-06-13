@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Set window title
title GitHub Auto Delete Tool

:: Default values
set "COMMIT_MESSAGE=Auto commit"
set "REMOTE_BRANCH=main"

:: Configure git for Chinese display
git config --global core.quotepath false
git config --global i18n.logoutputencoding utf-8
git config --global i18n.commitencoding utf-8
git config --global gui.encoding utf-8

:list_files
cls
:: Reset all counters and arrays
set i=1
set "dir_count=0"
set "file_count=0"
set "md_count=0"
for /l %%n in (1,1,100) do (
    set "file_%%n="
)

:: Show header with current time
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "Sec=%dt:~12,2%"

echo.
echo ════════════════════════════════════════
echo         GitHub Auto Delete Tool
echo ════════════════════════════════════════
echo  Time: %HH%:%Min%:%Sec%  Date: %YYYY%-%MM%-%DD%
echo ────────────────────────────────────────

:: Fetch latest changes from remote
echo [*] Fetching latest changes from remote...
git fetch origin %REMOTE_BRANCH%

echo.
echo Files in repository:
echo ────────────────────────────────────────

:: Temporary files for sorting
set "temp_dir=%TEMP%\dirs.txt"
set "temp_md=%TEMP%\md_files.txt"
set "temp_other=%TEMP%\other_files.txt"
set "temp_special=%TEMP%\special_files.txt"

:: Clear temporary files
type nul > "%temp_dir%"
type nul > "%temp_md%"
type nul > "%temp_other%"
type nul > "%temp_special%"

:: Sort files by type
for /f "delims=" %%a in ('git ls-files') do (
    set "filepath=%%a"
    if "!filepath:~-1!"=="/" (
        echo %%a>> "%temp_dir%"
        set /a dir_count+=1
    ) else if "!filepath!"=="README.md" (
        echo %%a>> "%temp_special%"
    ) else if "!filepath:~-3!"==".md" (
        echo %%a>> "%temp_md%"
        set /a md_count+=1
    ) else (
        echo %%a>> "%temp_other%"
        set /a file_count+=1
    )
)

:: Display special files first (README.md)
for /f "delims=" %%a in ('type "%temp_special%"') do (
    echo  !i!. [SPECIAL] %%a
    set "file_!i!=%%a"
    set /a i+=1
)
if exist "%temp_special%" (
    if %i% gtr 1 echo.
)

:: Display directories
if %dir_count% gtr 0 (
    echo [Directories]
    echo ────────────────────────────────────────
    for /f "delims=" %%a in ('type "%temp_dir%"') do (
        echo  !i!. [DIR] %%a
        set "file_!i!=%%a"
        set /a i+=1
    )
    echo.
)

:: Display markdown files
if %md_count% gtr 0 (
    echo [Markdown Files]
    echo ────────────────────────────────────────
    for /f "delims=" %%a in ('type "%temp_md%"') do (
        echo  !i!. [MD] %%a
        set "file_!i!=%%a"
        set /a i+=1
    )
    echo.
)

:: Display other files
if %file_count% gtr 0 (
    echo [Other Files]
    echo ────────────────────────────────────────
    for /f "delims=" %%a in ('type "%temp_other%"') do (
        echo  !i!. %%a
        set "file_!i!=%%a"
        set /a i+=1
    )
)

:: Clean up temp files
del "%temp_dir%" 2>nul
del "%temp_md%" 2>nul
del "%temp_other%" 2>nul
del "%temp_special%" 2>nul

set /a total_files=i-1
echo ════════════════════════════════════════

:: Show file statistics
echo  Total: %total_files% files ^( MD: %md_count%, Other: %file_count% ^)
echo ────────────────────────────────────────

:: Prompt user to input numbers
echo.
echo  Enter numbers separated by spaces (e.g., "1 3 5")
set /p FILE_NUMS="  Enter the numbers of files to delete (1-%total_files%, or 'q' to quit): "

:: Check for quit
if /i "%FILE_NUMS%"=="q" exit /b

:: Initialize deletion list
set "files_to_delete="
set "invalid_numbers=0"

:: Validate each number and build deletion list
for %%n in (%FILE_NUMS%) do (
    set "num=%%n"
    if !num! LSS 1 (
        set /a invalid_numbers+=1
    ) else if !num! GTR %total_files% (
        set /a invalid_numbers+=1
    ) else (
        if defined files_to_delete (
            set "files_to_delete=!files_to_delete!,!file_%%n!"
        ) else (
            set "files_to_delete=!file_%%n!"
        )
    )
)

:: Check if any numbers were invalid
if !invalid_numbers! GTR 0 (
    echo [!] Invalid number^(s^) detected
    pause
    goto list_files
)

:: Show selected files and confirm
echo.
echo Selected files for deletion:
echo ────────────────────────────────────────
for %%f in (%files_to_delete%) do (
    echo  - %%f
)
echo.
set "CONFIRM="
set /p "CONFIRM=Are you sure you want to delete these files? (Y/N, default=Y): "
if /i "!CONFIRM!"=="N" (
    echo [!] Operation cancelled
    pause
    goto list_files
)

:: Remove files from git
echo.
echo [*] Removing files...
for %%f in (%files_to_delete%) do (
    git rm --cached "%%f"
)

:: Commit changes
echo [*] Committing changes...
git commit -m "%COMMIT_MESSAGE% - Delete multiple files"

:: Push to remote repository
echo [*] Pushing to remote repository...
git push origin %REMOTE_BRANCH%

echo.
echo [√] Files have been deleted from remote repository
echo ════════════════════════════════════════
pause
goto list_files 
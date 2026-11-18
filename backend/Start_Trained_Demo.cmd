@echo off
setlocal
cd /d "%~dp0.."

if not defined PATHSNAP_TRIP_ROOT (
    echo Set PATHSNAP_TRIP_ROOT to the extracted trip folder.
    exit /b 2
)
if not exist "%PATHSNAP_TRIP_ROOT%\." (
    echo The configured trip folder does not exist.
    exit /b 2
)

if not defined PATHSNAP_MODEL set "PATHSNAP_MODEL=%CD%\artifacts\person2\hgb_full_train.joblib"
if not exist "%PATHSNAP_MODEL%" (
    echo The trained model artifact does not exist.
    exit /b 2
)

set "PATHSNAP_HOST=127.0.0.1"
if not "%~1"=="" set "PATHSNAP_HOST=%~1"
python -B -m uvicorn backend.main:app --host "%PATHSNAP_HOST%" --port 8000 --access-log
exit /b %ERRORLEVEL%

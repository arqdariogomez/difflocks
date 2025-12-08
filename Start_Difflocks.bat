@echo off
TITLE DiffLocks Studio Launcher
COLOR 0A

echo ========================================================
echo        RUNNING DIFFLOCKS STUDIO (DOCKER EDITION)
echo ========================================================
echo.

REM 1. Verificar si Docker está corriendo
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    echo [ERROR] Docker Desktop no esta corriendo.
    echo Por favor abre Docker Desktop y vuelve a intentarlo.
    echo.
    pause
    exit
)

REM 2. Arrancar el contenedor
echo [INFO] Iniciando servidor... (Esto puede tardar un poco si hay actualizaciones)
docker compose up -d

echo.
echo [EXITO] Servidor corriendo en segundo plano.
echo.
echo --------------------------------------------------------
echo    ABRE ESTO EN TU NAVEGADOR: http://localhost:7860
echo --------------------------------------------------------
echo.
echo Presiona cualquier tecla para ver los logs en vivo (o cierra esta ventana si no los necesitas)...
pause >nul

REM 3. Mostrar logs
docker compose logs -f
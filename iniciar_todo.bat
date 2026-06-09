@echo off
echo ==========================================
echo    INICIANDO VARROA DETECTOR
echo ==========================================

echo [1/2] Arrancando Inteligencia Artificial...
IF NOT EXIST ".venv\" (
    echo [!] Detectado ordenador nuevo. Creando entorno virtual e instalando librerias...
    echo [!] Esto puede tardar unos minutos la primera vez.
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r AppWeb\api-python\requirements.txt
)

start "API Python" cmd /k "call .venv\Scripts\activate.bat && cd AppWeb\api-python && python app.py"

echo [2/2] Arrancando Servidor Web y Base de Datos...
cd AppWeb\basico
start "Web Java" cmd /k "java -jar target\basico-0.0.1-SNAPSHOT.jar"

cd ..\..

echo.
echo ==========================================
echo TODO LISTO! Ya puedes abrir tu navegador en:
echo http://localhost:8099
echo ==========================================
pause
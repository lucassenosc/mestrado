@echo off
echo ====================================================
echo INICIANDO BATCH NOTURNO DE EXPERIMENTOS
echo ====================================================
echo.

echo [1/2] RODANDO standart_wisard_copy...
REM Chamando diretamente o Python de dentro do venv!
venv\Scripts\python.exe standart_wisard_copy.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] A standart_wisard falhou! Continuando para o proximo...
) else (
    echo [SUCESSO] standart_wisard finalizada!
)

echo.
echo [2/2] RODANDO bloom_wisard_final_copy...
REM Chamando diretamente o Python de dentro do venv!
venv\Scripts\python.exe bloom_wisard_final_copy.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] A bloom_wisard_final falhou!
) else (
    echo [SUCESSO] bloom_wisard_final finalizada!
)

echo.
echo ====================================================
echo TODOS OS EXPERIMENTOS CONCLUIDOS!
echo ====================================================
pause
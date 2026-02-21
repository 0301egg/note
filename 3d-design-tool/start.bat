@echo off
chcp 65001 >nul
echo ========================================
echo   3Dデザインツール 起動スクリプト
echo ========================================
echo.

REM 仮想環境がなければ作成
if not exist "venv\" (
    echo [1/3] 仮想環境を作成中...
    python -m venv venv
    if errorlevel 1 (
        echo エラー: Pythonが見つかりません。
        echo https://www.python.org/downloads/ からインストールしてください。
        pause
        exit /b 1
    )
)

REM 仮想環境を有効化
call venv\Scripts\activate.bat

REM ライブラリのインストール（初回のみ時間がかかります）
echo [2/3] ライブラリを確認中...
pip install -r requirements.txt -q

REM .envが無ければ作成を促す
if not exist ".env" (
    echo.
    echo [注意] .env ファイルが見つかりません。
    echo .env.example をコピーして .env を作成し、
    echo ANTHROPIC_API_KEY にAPIキーを設定してください。
    echo.
    copy .env.example .env >nul
    echo .env ファイルを作成しました。メモ帳で開きます...
    notepad .env
    echo.
    echo APIキーを設定・保存したら、このウィンドウで何かキーを押してください。
    pause
)

REM アプリ起動
echo [3/3] アプリを起動中...
echo.
echo ブラウザで以下のURLを開いてください:
echo   http://localhost:5000
echo.
echo 終了するには Ctrl+C を押してください。
echo ----------------------------------------
python app.py
pause

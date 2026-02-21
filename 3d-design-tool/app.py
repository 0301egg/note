"""
3Dデザインツール - Flask + Ollama バックエンド
テキストから3DモデルのSTLファイルを生成する
Bambu Lab mini 対応
"""

import base64
import io
import json
import os
import re

import ollama
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from stl_generator import generate_stl, get_supported_shapes

load_dotenv()

app = Flask(__name__)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Claude API に渡すシステムプロンプト
SYSTEM_PROMPT = """あなたは3Dモデル設計の専門家です。
ユーザーの日本語の説明から3Dオブジェクトのパラメータを正確に抽出してください。

以下のJSON形式のみで回答してください（他のテキストは一切含めないこと）:
{
  "shape": "形状タイプ",
  "dimensions": {
    // 形状に応じたパラメータ（単位はすべてmm）
  },
  "description": "日本語での形状説明",
  "print_notes": "Bambu Lab 3Dプリントの注意事項（省略可）"
}

## サポートする形状と必須パラメータ:

- **box** (直方体): width (幅mm), depth (奥行きmm), height (高さmm)
- **sphere** (球体): radius (半径mm)
- **cylinder** (円柱): radius (半径mm), height (高さmm)
- **cone** (円錐): radius (底面半径mm), height (高さmm)
- **torus** (トーラス/ドーナツ): major_radius (外径mm), minor_radius (管径mm)
- **capsule** (カプセル): radius (半径mm), height (高さmm)
- **pyramid** (ピラミッド): base (底面一辺mm), height (高さmm)

## 寸法の推論ルール:
- ユーザーが「cm」単位で指定した場合は自動的に「mm」に変換（例: 5cm → 50mm）
- 寸法が未指定の場合は文脈から合理的なサイズを推測する（一般的な用途を考慮）
- 「小さい」→ ~30mm、「中くらい」→ ~60mm、「大きい」→ ~100mm を目安とする
- Bambu Lab mini の造形サイズは 180×180×180mm なので、その範囲内に収めること

## 回答形式の厳守:
- 必ずJSON形式のみで回答する
- コードブロック (``` ) は使わない
- 説明文は含めない
- JSONのみ出力する"""


def extract_json_from_response(text: str) -> dict:
    """LLMレスポンスからJSONを安全に抽出する"""
    # まず直接パースを試みる
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # コードブロックからJSONを抽出
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # テキスト中の最初のJSONオブジェクトを抽出
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSONの抽出に失敗しました。レスポンス: {text[:200]}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """テキストからSTLを生成するメインAPI"""
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "テキストを入力してください"}), 400

    if len(text) > 1000:
        return jsonify({"error": "テキストは1000文字以内で入力してください"}), 400

    # Ollama でテキスト解析
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            return jsonify({"error": f"モデル '{OLLAMA_MODEL}' が見つかりません。'ollama pull {OLLAMA_MODEL}' を実行してください"}), 503
        return jsonify({"error": f"Ollamaエラー: {str(e)}"}), 500
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            return jsonify({"error": f"Ollamaに接続できません。Ollamaが起動しているか確認してください ({OLLAMA_HOST})"}), 503
        return jsonify({"error": f"API呼び出しエラー: {str(e)}"}), 500

    response_text = response.message.content

    if not response_text:
        return jsonify({"error": "モデルからの回答を取得できませんでした"}), 500

    # JSONパース
    try:
        shape_params = extract_json_from_response(response_text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # 必須フィールドの検証
    if "shape" not in shape_params:
        return jsonify({"error": "形状タイプが特定できませんでした"}), 422

    if "dimensions" not in shape_params or not isinstance(shape_params["dimensions"], dict):
        shape_params["dimensions"] = {}

    # STL生成
    try:
        stl_bytes = generate_stl(shape_params)
    except Exception as e:
        return jsonify({"error": f"3Dモデルの生成に失敗しました: {str(e)}"}), 500

    if not stl_bytes:
        return jsonify({"error": "STLデータの生成に失敗しました"}), 500

    # レスポンス
    return jsonify(
        {
            "success": True,
            "shape_params": shape_params,
            "stl_base64": base64.b64encode(stl_bytes).decode("utf-8"),
            "file_size_kb": round(len(stl_bytes) / 1024, 1),
        }
    )


@app.route("/api/download", methods=["POST"])
def download():
    """STLファイルをダウンロードする"""
    data = request.get_json()
    stl_base64 = (data or {}).get("stl_base64", "")
    filename = (data or {}).get("filename", "model.stl")

    if not stl_base64:
        return jsonify({"error": "STLデータがありません"}), 400

    # ファイル名のサニタイズ
    filename = re.sub(r"[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF._-]", "_", filename)
    if not filename.endswith(".stl"):
        filename += ".stl"

    try:
        stl_bytes = base64.b64decode(stl_base64)
    except Exception:
        return jsonify({"error": "STLデータのデコードに失敗しました"}), 400

    return send_file(
        io.BytesIO(stl_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/shapes", methods=["GET"])
def shapes():
    """サポートする形状の一覧を返す"""
    return jsonify({"shapes": get_supported_shapes()})


@app.route("/api/health", methods=["GET"])
def health():
    """ヘルスチェック"""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        client.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return jsonify(
        {
            "status": "ok",
            "ollama_connected": ollama_ok,
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"3Dデザインツール起動中... http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

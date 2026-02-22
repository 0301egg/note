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
import traceback

import ollama
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from stl_generator import generate_stl, get_supported_shapes

load_dotenv()

app = Flask(__name__)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# LLM に渡すシステムプロンプト
SYSTEM_PROMPT = """あなたは3Dモデル設計の専門家です。ユーザーの説明から3Dオブジェクトのパラメータを正確に抽出してください。
JSONのみ出力してください（コードブロック・説明文は一切不要）。

## 単一形状の場合:
{"shape": "形状タイプ", "dimensions": {...}, "description": "説明"}

## 複合形状の場合（複数パーツの組み合わせ）:
{"shape": "compound", "components": [{"shape": "...", "dimensions": {...}, "position": [x,y,z], "rotation": [rx,ry,rz], "operation": "union"}, ...], "description": "説明"}
- operation: "union"(合体), "subtract"(切り抜き), "intersect"(交差)
- position: コンポーネント中心のXYZ座標(mm)。Z=0が底面。
- rotation: XYZ回転角度(度)。省略時は[0,0,0]

## 利用可能な形状と主要パラメータ（単位はmm）:

### 基本形状:
- box: width, depth, height
- rounded_box: width, depth, height, radius(角丸半径)
- sphere: radius
- hemisphere: radius（半球・ドーム）
- ellipsoid: radius_x, radius_y, radius_z（楕円体）
- cylinder: radius, height
- cone: radius, height
- frustum: bottom_radius, top_radius, height（切頭円錐）
- torus: major_radius, minor_radius（ドーナツ）
- capsule: radius, height
- pyramid: base, height

### 複雑形状:
- star: points(頂点数), outer_radius, inner_radius, height（星形柱）
- arrow: total_length, shaft_width, head_width, head_length, thickness（矢印）
- cross: arm_length, arm_width, height（十字形）
- gear: teeth(歯数), module(モジュール,通常1～3), thickness, bore_radius(穴半径)
- spring: coil_radius, wire_radius, num_coils(巻数), pitch(ピッチ)
- l_bracket: arm1_length, arm2_length, thickness, depth（L字金具）
- t_bracket: top_length, stem_length, arm_width, depth（T字金具）
- pipe: outer_radius, inner_radius, height（パイプ）
- hexagonal_prism: radius, height（六角柱）
- wedge: radius, angle(度), height（扇形柱）

## 複合形状の例（四隅に穴あきベースプレート）:
{"shape":"compound","components":[{"shape":"box","dimensions":{"width":50,"depth":50,"height":8},"position":[0,0,0],"operation":"union"},{"shape":"cylinder","dimensions":{"radius":3,"height":12},"position":[20,20,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3,"height":12},"position":[-20,20,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3,"height":12},"position":[20,-20,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3,"height":12},"position":[-20,-20,-2],"operation":"subtract"}],"description":"四隅に取り付け穴のあるベースプレート"}

## 複合形状の例（台座付き円柱）:
{"shape":"compound","components":[{"shape":"cylinder","dimensions":{"radius":20,"height":5},"position":[0,0,0],"operation":"union"},{"shape":"cylinder","dimensions":{"radius":8,"height":40},"position":[0,0,5],"operation":"union"}],"description":"台座付き円柱"}

## 寸法ルール:
- cm指定はmmに変換（5cm→50mm）
- 未指定は文脈から推測（「小さい」≈30mm、「中」≈60mm、「大きい」≈100mm）
- Bambu Lab mini最大造形サイズ: 180×180×180mm
- subtract操作では形状を貫通させるため、少し大きめ・長めにすること
- 複雑なものはcompoundを積極的に使用すること"""


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
        traceback.print_exc()  # サーバーターミナルにフルのトレースバックを出力
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

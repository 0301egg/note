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

# ── LLM システムプロンプト ──
# 短く・密度高く → LLM推論速度と精度の両方を改善
SYSTEM_PROMPT = """3DモデルAI。入力から最適な形状JSONのみ出力（説明・コードブロック不要）。

【出力形式】
単一: {"shape":"形状名","dimensions":{...},"description":"説明"}
複合: {"shape":"compound","components":[{"shape":"名前","dimensions":{...},"position":[x,y,z],"operation":"union|subtract"}],"description":"説明"}
※subtract時は形状を2mm大きく・長くして完全に貫通させること

【形状リスト（単位:mm）】
box: width,depth,height | rounded_box: width,depth,height,radius
cylinder: radius,height | sphere: radius | hemisphere: radius
ellipsoid: radius_x,radius_y,radius_z | cone: radius,height
frustum: bottom_radius,top_radius,height | torus: major_radius,minor_radius
capsule: radius,height | pyramid: base,height
pipe: outer_radius,inner_radius,height | hexagonal_prism: radius,height
vase: height,bottom_radius,max_radius,top_radius,wall_thickness
bowl: outer_radius,depth,wall_thickness
egg: radius_xy,height
split_egg: egg_radius_xy,egg_height,wall_thickness,sphere_radius
twisted_prism: sides,radius,height,twist_angle
wavy_plate: width,depth,base_height,amplitude,wave_count
gear: teeth,module,thickness,bore_radius | spring: coil_radius,wire_radius,num_coils,pitch
l_bracket: arm1_length,arm2_length,thickness,depth | t_bracket: top_length,stem_length,arm_width,depth
star: points,outer_radius,inner_radius,height | arrow: total_length,shaft_width,head_width,head_length,thickness
cross: arm_length,arm_width,height | wedge: radius,angle,height
※box/cylinderにwall_thickness追加→上部開口の中空容器

【必須ルール】
・直径→半径に変換（直径60mm→radius=30、直径5cm→radius=25）
・cm→mm変換（5cm=50mm）
・未指定サイズ: 小30mm 中60mm 大100mm、最大180mm
・花瓶/ボトル/コップ→vase（wall_thickness必須）
・ボウル/皿/器→bowl（wall_thickness必須）
・入れ物/容器/箱→box+wall_thickness または cylinder+wall_thickness
・卵(中実)→egg / 卵+割れる+球体→split_egg(M卵: egg_radius_xy=21.5,egg_height=55,wall_thickness=2)
・穴あき/複数パーツ/組み合わせ→compound

【出力例】
直径5cmの円柱形の花瓶(高さ8cm):
{"shape":"vase","dimensions":{"height":80,"bottom_radius":22,"max_radius":35,"top_radius":15,"wall_thickness":3},"description":"円柱形花瓶"}

中央に丸穴のプレート(100×60×8mm、穴の直径20mm):
{"shape":"compound","components":[{"shape":"box","dimensions":{"width":100,"depth":60,"height":8},"position":[0,0,0],"operation":"union"},{"shape":"cylinder","dimensions":{"radius":10,"height":12},"position":[0,0,-2],"operation":"subtract"}],"description":"穴あきプレート"}"""


def extract_json_from_response(text: str) -> dict:
    """LLMレスポンスからJSONを安全に抽出する（複数フォールバック付き）"""
    text = text.strip()

    # 1. 直接パース
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. コードブロック内を抽出
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 最初の {...} ブロックを抽出（ネスト対応）
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False; continue
        if ch == "\\" and in_string:
            escaped = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass

    raise ValueError(f"JSONの抽出に失敗しました。レスポンス: {text[:200]}")


# パラメータの物理的上限（プリンタ最大サイズ）
_MAX_DIM = 180.0
_RADIUS_KEYS = {
    "radius", "radius_x", "radius_y", "radius_z",
    "outer_radius", "inner_radius", "bottom_radius", "top_radius", "max_radius",
    "major_radius", "minor_radius", "coil_radius", "wire_radius", "bore_radius",
    "egg_radius_xy", "radius_xy", "sphere_radius",
}
_LENGTH_KEYS = {
    "width", "depth", "height", "base", "arm1_length", "arm2_length",
    "arm_length", "top_length", "stem_length", "thickness", "pitch",
    "arm_width", "egg_height",
}


def sanitize_dimensions(dims: dict) -> dict:
    """
    LLMが出力したパラメータの自動補正:
    ・半径が 90mm 超 → 最大 90mm にクランプ（直径を半径と誤った場合への対処）
    ・全長が 180mm 超 → クランプ
    ・負値 → 絶対値
    """
    out = {}
    for k, v in dims.items():
        if not isinstance(v, (int, float)):
            out[k] = v
            continue
        v = float(v)
        if v < 0:
            v = abs(v)
        if k in _RADIUS_KEYS:
            # 半径が 90mm 超は「直径を半径に誤った」可能性が高い → ÷2
            if v > 90:
                v = v / 2.0
            v = min(v, 90.0)
        elif k in _LENGTH_KEYS:
            v = min(v, _MAX_DIM)
        out[k] = round(v, 2)
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    """テキストを解析して3Dモデルをプレビュー用に生成する（STLファイルは作成しない）"""
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
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": 500,   # 出力トークン上限（暴走防止・速度改善）
            },
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

    # パラメータ自動補正（範囲オーバー・直径誤記など）
    shape_params["dimensions"] = sanitize_dimensions(shape_params["dimensions"])

    # ビューワー用メッシュ生成（STLとして返却するが、ファイルとしては保存しない）
    try:
        stl_bytes = generate_stl(shape_params)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"3Dモデルの生成に失敗しました: {str(e)}"}), 500

    if not stl_bytes:
        return jsonify({"error": "3Dモデルの生成に失敗しました"}), 500

    return jsonify(
        {
            "success": True,
            "shape_params": shape_params,
            "stl_base64": base64.b64encode(stl_bytes).decode("utf-8"),
        }
    )


@app.route("/api/export", methods=["POST"])
def export_stl():
    """保存済みの shape_params からSTLファイルを生成してダウンロードさせる"""
    data = request.get_json()
    shape_params = (data or {}).get("shape_params")
    filename = (data or {}).get("filename", "model.stl")

    if not shape_params or "shape" not in shape_params:
        return jsonify({"error": "形状パラメータがありません"}), 400

    # ファイル名のサニタイズ
    filename = re.sub(r"[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF._-]", "_", filename)
    if not filename.endswith(".stl"):
        filename += ".stl"

    try:
        stl_bytes = generate_stl(shape_params)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"STLの生成に失敗しました: {str(e)}"}), 500

    return send_file(
        io.BytesIO(stl_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/render", methods=["POST"])
def render_shape():
    """shape_params から直接3Dモデルを生成（LLMなし・リアルタイム調整用）"""
    data = request.get_json()
    shape_params = (data or {}).get("shape_params")

    if not shape_params or "shape" not in shape_params:
        return jsonify({"error": "形状パラメータがありません"}), 400

    if "dimensions" not in shape_params or not isinstance(shape_params.get("dimensions"), dict):
        shape_params["dimensions"] = {}

    try:
        stl_bytes = generate_stl(shape_params)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"3Dモデルの生成に失敗しました: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "stl_base64": base64.b64encode(stl_bytes).decode("utf-8"),
    })


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

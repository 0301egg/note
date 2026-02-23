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
SYSTEM_PROMPT = """あなたは3Dモデル設計の専門家です。ユーザーの説明を読み、最も適切な3D形状JSONを出力してください。
必ずJSONのみを出力してください（説明文・コードブロック不要）。

=== 出力形式 ===

【単一形状】
{"shape":"形状名","dimensions":{パラメータ},"description":"説明"}

【複合形状】穴・突起・複数パーツがある場合は必ずこちらを使用
{"shape":"compound","components":[{"shape":"形状名","dimensions":{...},"position":[x,y,z],"operation":"union"},...],"description":"説明"}
  operation: "union"=合体, "subtract"=切り抜き(貫通穴など)
  position: [x,y,z] 各コンポーネントの中心座標(mm)、Z=0が底面
  subtract時は形状を2mm程度大きく・長くして完全に貫通させること

=== 利用可能な形状 ===

■ 基本プリミティブ
  box             : width, depth, height
  rounded_box     : width, depth, height, radius(角丸半径)
  cylinder        : radius, height
  sphere          : radius
  hemisphere      : radius (半球・ドーム)
  ellipsoid       : radius_x, radius_y, radius_z (楕円体)
  cone            : radius, height
  frustum         : bottom_radius, top_radius, height (台形円錐)
  torus           : major_radius, minor_radius (ドーナツ)
  capsule         : radius, height
  pyramid         : base, height
  pipe            : outer_radius, inner_radius, height (中空パイプ)
  hexagonal_prism : radius, height (六角柱)

■ 有機的・装飾的形状 ← 容器・インテリア系ならこちらを使う
  vase            : height, bottom_radius, max_radius, top_radius, wall_thickness
                    (花瓶/ボトル: 回転体、中空、くびれ付き)
  bowl            : outer_radius, depth, wall_thickness
                    (ボウル/皿: 丸底、中空)
  twisted_prism   : sides(辺数3〜8), radius, height, twist_angle(ねじれ度数)
                    (ねじれた角柱: 装飾的なオブジェクト)
  wavy_plate      : width, depth, base_height, amplitude(波高), wave_count(波数)
                    (波打ったプレート: コースター・壁パネルなど)
  egg             : radius_xy(最大半径mm), height(高さmm)
                    (卵形・中実。単体の卵形オブジェクト)
  split_egg       : egg_radius_xy, egg_height, wall_thickness, sphere_radius(0=自動)
                    (水平2分割の中空卵カプセル + 内部球体を3パーツ並列出力)
                    ※「卵を半分に割る・カプセル・中に球/ボール」→ 必ず split_egg

■ 機械・構造系形状
  gear            : teeth(歯数12〜32), module(1〜3), thickness, bore_radius(軸穴)
  spring          : coil_radius, wire_radius, num_coils, pitch
  l_bracket       : arm1_length, arm2_length, thickness, depth (L字金具)
  t_bracket       : top_length, stem_length, arm_width, depth (T字金具)
  star            : points(頂点数5〜8), outer_radius, inner_radius, height
  arrow           : total_length, shaft_width, head_width, head_length, thickness
  cross           : arm_length, arm_width, height (十字形)
  wedge           : radius, angle(度), height (扇形柱)

=== 形状選択ルール ===
- 花瓶・コップ・ボトル・カップ → vase
- ボウル・皿・受け皿 → bowl
- ねじれた柱・螺旋柱・装飾柱 → twisted_prism (sides=4〜8, twist_angle=90〜270)
- 波板・コースター・模様付き → wavy_plate
- 卵形のみ(中実) → egg
- 卵 + 割れる/分割/カプセル/中に球・ボールが入る → split_egg (必須)
  └ 日本M卵サイズ: egg_radius_xy=21.5, egg_height=55, wall_thickness=2, sphere_radius=0(自動)
- 穴あき・溝・組み合わせ形状 → compound
- 機械部品・ブラケット・ギア → それぞれ専用形状
- 単純な基本形状 → box/cylinder/sphere など

=== split_egg 出力例（卵Mサイズ・分割・球体入り） ===
{"shape":"split_egg","dimensions":{"egg_radius_xy":21.5,"egg_height":55,"wall_thickness":2,"sphere_radius":0},"description":"卵Mサイズ 水平分割カプセル + 内部球体"}

=== 複合形状の例 ===
スマホスタンド:
{"shape":"compound","components":[{"shape":"box","dimensions":{"width":80,"depth":60,"height":5},"position":[0,0,0],"operation":"union"},{"shape":"box","dimensions":{"width":80,"depth":8,"height":60},"position":[0,26,5],"operation":"union"},{"shape":"box","dimensions":{"width":80,"depth":40,"height":3},"position":[0,7,5],"rotation":[20,0,0],"operation":"union"}],"description":"スマホスタンド"}

四隅ボルト穴プレート:
{"shape":"compound","components":[{"shape":"rounded_box","dimensions":{"width":60,"depth":40,"height":8,"radius":3},"position":[0,0,0],"operation":"union"},{"shape":"cylinder","dimensions":{"radius":3.5,"height":12},"position":[22,14,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3.5,"height":12},"position":[-22,14,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3.5,"height":12},"position":[22,-14,-2],"operation":"subtract"},{"shape":"cylinder","dimensions":{"radius":3.5,"height":12},"position":[-22,-14,-2],"operation":"subtract"}],"description":"四隅ボルト穴付きプレート"}

=== 寸法ルール ===
- cm→mm変換（5cm=50mm）
- 未指定: 小≈30mm, 中≈60mm, 大≈100mm
- 最大サイズ: 180×180×180mm"""


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
            options={"temperature": 0.1},
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

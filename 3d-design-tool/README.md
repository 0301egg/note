# 3Dデザインツール

テキストで説明するだけで3DモデルのSTLファイルを生成するツールです。
**Bambu Lab mini** で直接印刷できるSTL形式に対応しています。

## 機能

- **テキストから3D生成**: 日本語で形状・サイズを説明するとClaude AIが解析してSTLを自動生成
- **リアルタイム3Dプレビュー**: Three.jsによるブラウザ内3Dビューワー（回転・ズーム対応）
- **STLダウンロード**: Bambu Studio に読み込み可能な標準バイナリSTL形式で出力
- **対応形状**: 直方体・球体・円柱・円錐・トーラス・カプセル・ピラミッド・中空円柱

## 動作例

```
入力: 「高さ80mm、直径50mmの円柱形のペン立てを作って」
  ↓ Claude AI が解析
出力: cylinder.stl（半径25mm、高さ80mm）→ Bambu Lab で印刷
```

## セットアップ

```bash
# 1. リポジトリをクローン
git clone <repo-url>
cd 3d-design-tool

# 2. 依存ライブラリをインストール
pip install -r requirements.txt

# 3. APIキーを設定
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定

# 4. 起動
python app.py
# → http://localhost:5000 をブラウザで開く
```

## 使い方

1. テキストエリアに3Dオブジェクトを日本語で説明する
   - 例: 「幅60mm、奥行き40mm、高さ20mmの箱」
   - 例: 「半径35mm、管径8mmのドーナツ形」
2. **STLを生成する** ボタンをクリック（または Ctrl+Enter）
3. 右のビューワーで3Dプレビューを確認
4. **STLファイルをダウンロード** → Bambu Studio でスライス → 印刷

## Bambu Lab mini 対応サイズ

最大造形サイズ: **180 × 180 × 180 mm**
生成されるSTLはこのサイズ以内で設計してください。

## 技術スタック

| 役割 | 技術 |
|------|------|
| AI テキスト解析 | Claude API (`claude-opus-4-6`) |
| 3D形状生成 | [trimesh](https://trimsh.org/) |
| Web バックエンド | Flask |
| 3D プレビュー | Three.js r160 |
| STL 形式 | バイナリSTL（Bambu Lab / PrusaSlicer 対応） |

## ライセンス

MIT

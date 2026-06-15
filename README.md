# GLM-OCR Server

`zai-org/GLM-OCR` を使ったローカルOCRサーバーです。  
FastAPI + シンプルなWeb UIで、画像/PDFをページ単位でOCRできます。

![GLM-OCR UI](image.jpg)

## 主な機能

- GLM-OCR推論（`text` / `table` / `formula` / `extract_json`）
- PDF入力（`pypdfium2` でページを画像化してOCR）
- 進捗表示（`事前処理中` → `i/nページをOCR中` / `i/n pages, k/m regions`）
- 実行中断（中断API + UIボタン）
- 改行後処理モード（`none` / `paragraph` / `compact`）
- ページ表示切替（ドロップダウン + `ALL`合算表示）
- 表示中結果のコピー（`raw` はコピー対象外）
- モデル/キャッシュはプロジェクト配下に保存
- **レイアウト検出モード**（`use_layout=true`）
  - PaddleOCR（PP-DocLayoutV3）によるブロック検出
  - PaddleOCR 未インストール時はカラム分割フォールバック
  - 縦書き・段組みを含む4種の読み順をサポート
  - ブロック単位のOCR結果・レイアウトプレビュー画像を返却
- **クロップ領域指定**（`crop_region`）で特定ページ・座標のみを抽出
- OAuth（Auth0）によるログイン保護（オプション）

## 動作環境

- Python 3.10+
- Windows または Linux/macOS
- CUDA使用時は対応GPU + ドライバ
- レイアウト検出には `paddleocr`（オプション）

## クイックスタート

### 配布・移行時のコピー対象

オンプレサーバーへそのまま移す場合は、以下をコピーしてください。

- [app](app)
- [run.bat](run.bat)
- [README.md](README.md)
- [.env](.env)
- [models](models)

補足:

- `models` をコピーしておくと、モデルの初回再取得を避けやすくなります。
- `.venv` はコピーしても動く場合がありますが、環境差分を避けたいなら先方で作り直すのが安全です。
- `run.bat` は起動時に `.venv` を自動作成し、必要な依存関係とモデルキャッシュを使います。

### Windows

```bat
run.bat
```

### Linux / macOS

```bash
chmod +x run.sh
./run.sh
```

起動後:

- UI: `http://localhost:8000/`
- API Docs: `http://localhost:8000/docs`

## `.env` で設定

プロジェクト直下に `.env` を置くと、`run.sh` / `run.bat` 起動時に読み込みます。

例:

```env
HOST=0.0.0.0
PORT=9000
TORCH_CHANNEL=cu126
```

主な設定値:

- `HOST`: バインドアドレス（既定 `0.0.0.0`）
- `PORT`: 起動ポート（既定 `8000`）
- `TORCH_CHANNEL`: PyTorch配布チャネル（例 `cu126`, `cpu`）
- `AUTH_DISABLED`: 認証を一時的に外すときに `1` を設定（`0` または未設定で認証有効）
- `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `AUTH0_SECRET`, `AUTH0_REDIRECT_URI`: Auth0 ログイン設定

> OAuth 設定も含めて、環境変数は **プロジェクトルートの `.env` に一本化**してください。

認証の切り替えは、`.env` の `AUTH_DISABLED` だけでできます。

- `AUTH_DISABLED=1` で認証を無効化
- `AUTH_DISABLED=0` または未設定で認証を有効化

## モデル保存先

モデルキャッシュはプロジェクト内に保存されます。

- `models/hf_cache`
- `models/hf_home`

環境変数 `GLM_MODEL_CACHE` で明示可能です。

## API

### `GET /api/status`

実行環境状態（CUDA可否・既定デバイス・モデル情報・OAuth有効状態）を返します。

```json
{
  "cuda_available": true,
  "device_default": "cuda",
  "model": "zai-org/GLM-OCR",
  "model_cache_dir": "...",
  "oauth_enabled": false,
  "oauth_base_path": null
}
```

### `POST /api/analyze`

マルチパートフォームでOCR実行。

フォーム項目:

| フィールド | 既定値 | 説明 |
|---|---|---|
| `file` | (必須) | 画像 / PDF |
| `device` | `auto` | `auto` / `cuda` / `cpu` |
| `dpi` | `220` | PDFレンダリングDPI（36〜600） |
| `task` | `text` | `text` / `table` / `formula` / `extract_json` |
| `linebreak_mode` | `none` | `none` / `paragraph` / `compact` |
| `schema` | — | `task=extract_json` のとき必須 |
| `max_new_tokens` | `1024` | 生成トークン上限（1〜32768） |
| `temperature` | `0.0` | サンプリング温度 |
| `use_layout` | `false` | レイアウト検出モードを有効化 |
| `layout_backend` | `ppdoclayoutv3` | `ppdoclayoutv3` / `none` |
| `reading_order` | `auto` | `auto` / `ltr_ttb` / `rtl_ttb` / `vertical_rl` |
| `region_padding` | `12` | ブロックbboxへの余白px（0〜256） |
| `max_regions` | `200` | 1ページあたりの最大ブロック数（1〜1000） |
| `region_parallelism` | `1` | 並列処理ブロック数（1〜8） |
| `crop_region` | — | 切り出し範囲のJSON。例: `{"x1":0,"y1":0,"x2":800,"y2":400,"page":1}` |
| `request_id` | (自動採番) | 進捗・中断操作に使うID |

#### レスポンス（`use_layout=false` 時）

```json
{
  "request_id": "...",
  "device": "cuda",
  "task": "text",
  "state": "done",
  "page_count": 2,
  "results": [
    {
      "page": 1,
      "text": "...",
      "raw": "...",
      "json": null,
      "truncated": false
    }
  ]
}
```

#### レスポンス（`use_layout=true` 時）

`results` の各ページには `blocks`・`reading_order`・`layout_preview_base64` が追加されます。

```json
{
  "page": 1,
  "text": "（ブロック結合テキスト）",
  "raw": "（ブロック結合rawテキスト）",
  "json": null,
  "blocks": [
    {
      "id": "b1",
      "type": "text",
      "bbox": {"x1": 10, "y1": 20, "x2": 300, "y2": 80},
      "text": "...",
      "raw": "...",
      "truncated": false
    }
  ],
  "reading_order": "ltr_ttb",
  "layout_preview_base64": "（PNG base64）"
}
```

### `GET /api/progress/{request_id}`

進捗状態を取得します。

```json
{
  "request_id": "...",
  "state": "ocr",
  "message": "1/3ページをOCR中",
  "current_page": 1,
  "total_pages": 3,
  "current_region": 0,
  "total_regions": 0,
  "updated_at": 1712345678.0
}
```

`state` の値: `preprocessing` / `ocr` / `done` / `error` / `canceled` / `cancel_requested`

### `POST /api/cancel/{request_id}`

中断要求を送信します。  
生成中はトークン生成ステップ単位で停止判定します。

## ライセンス

- このプロジェクト: `LICENSE`（MIT）
- サードパーティ情報: `THIRD_PARTY_NOTICES.md`

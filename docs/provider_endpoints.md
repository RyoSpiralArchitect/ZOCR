# LLM/VLM プロバイダー設定ガイド

下流のテキスト LLM と補助用ビジョン LLM (vLM) で別々のプロバイダーを即座に切り替えられるよう、環境変数と YAML のひな形をまとめました。API キーはファイルに直接書かず、環境変数を参照する想定です。

## 1. `.env` で最小限の切り替え
下流 LLM と補助 vLM にそれぞれプレフィックス付きの環境変数を割り当てます。キーは OS の環境変数にだけ設定してください。

```bash
# 下流 LLM
ZOCR_LLM_PROVIDER=azure_openai        # local_hf|aws_bedrock|azure_openai|gemini|anthropic
ZOCR_LLM_MODEL=gpt-4o-mini            # または modelId/deployment 名
ZOCR_LLM_ENDPOINT=https://<res>.openai.azure.com/
ZOCR_LLM_API_KEY=$AZURE_OPENAI_API_KEY
ZOCR_LLM_REGION=us-east-1             # AWS/Gemini の場合に利用
ZOCR_LLM_LOCAL_PATH=/models/text-llm  # local_hf の場合のみ

# 補助 vLM（画像説明やレイアウト要約用）
ZOCR_VLM_PROVIDER=gemini
ZOCR_VLM_MODEL=gemini-1.5-pro-latest
ZOCR_VLM_ENDPOINT=https://generativelanguage.googleapis.com
ZOCR_VLM_API_KEY=$GEMINI_API_KEY
ZOCR_VLM_REGION=us-east1             # Bedrock なら AWS リージョン
ZOCR_VLM_LOCAL_PATH=/models/vision   # local_hf の場合のみ
```

### プロバイダー別の書き分けメモ
- **local_hf**: `ZOCR_*_LOCAL_PATH` にモデルディレクトリを記載するだけで利用できます（API キー不要）。
- **aws_bedrock**: `AWS_PROFILE` または `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` を OS 側にエクスポートし、`ZOCR_*_REGION` にリージョンを指定します。
- **azure_openai**: `ZOCR_*_ENDPOINT` にリソース URL、`ZOCR_*_MODEL` にデプロイ名、`ZOCR_*_API_KEY` に API キーをセットします（`https://<resource>.openai.azure.com/` 形式）。
- **gemini**: `ZOCR_*_ENDPOINT` と `ZOCR_*_API_KEY` を指定し、必要ならリージョンも `ZOCR_*_REGION` に設定してください（`us-east1` など）。
- **anthropic**: `ZOCR_*_API_KEY` を指定し、`ZOCR_*_MODEL` に Claude ファミリーのモデル名を入れます（エンドポイントは省略可）。

> ワンポイント: **LLM と vLM を別プロバイダーにする**（例: LLM は Azure OpenAI、vLM は Gemini）と、料金・能力の最適化がしやすくなります。

## 2. YAML ひな形で永続化
`samples/llm_vlm_endpoints.example.yaml` をコピーして、CI/ローカルで共通のプロファイルを共有できます。環境変数名だけを書き、秘密情報はファイルに含めません。

```bash
cp samples/llm_vlm_endpoints.example.yaml llm_vlm_endpoints.yaml
```

このファイルをロードするユーティリティを追加する際は、`downstream_llm` と `aux_vlm` の両セクションから `provider`/`model`/`endpoint`/`api_key_env` を読み出すだけで済む構造になっています。最低限チェックするとよい項目は次の通りです。

1. **provider**: `local_hf|aws_bedrock|azure_openai|gemini|anthropic` のいずれか。
2. **model**: Bedrock なら `modelId`、Azure ならデプロイ名、その他はモデル名を指定。
3. **endpoint**: Azure/Gemini では必須。Anthropic は省略可。Bedrock は SDK の既定を使う場合は空でも可。
4. **api_key_env**: ファイルにキーを書かず、環境変数名だけ記載。ローカル HF なら未使用。
5. **region**: Bedrock/Gemini/Anthropic で必要な場合にセット。
6. **local_path**: `provider: local_hf` のときだけ解釈し、他のプロバイダーでは無視するようにしてください。

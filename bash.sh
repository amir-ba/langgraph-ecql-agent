#!/bin/bash
set -e -E

if [ $# -lt 2 ]; then
  echo "Usage: $0 <GEMINI_API_KEY> <CONTENT>"
  exit 1
fi

GEMINI_API_KEY="$1"
CONTENT="$2"
MODEL_ID="gemini-flash-latest"
GENERATE_CONTENT_API="streamGenerateContent"

cat << EOF > request.json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "$CONTENT"
        }
      ]
    }
  ],
  "generationConfig": {
    "thinkingConfig": {
      "thinkingLevel": "HIGH"
    }
  },
  "tools": [
    {
      "googleSearch": {}
    }
  ]
}
EOF

curl \
-X POST \
-H "Content-Type: application/json" \
"https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:${GENERATE_CONTENT_API}?key=${GEMINI_API_KEY}" -d '@request.json'
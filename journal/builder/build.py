import re
import sys
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension


def build_markdown_to_html_with_cards_and_nl2br(input_md_file, output_html_file):
    # Markdownファイルを読み込む
    with open(input_md_file, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # MarkdownをHTMLに変換し、シンタックスハイライトと改行保持を有効にする
    html_content = markdown.markdown(
        md_content,
        extensions=[CodeHiliteExtension(
            linenums=True), 'nl2br'],  # nl2brオプションを追加
        output_format="html5",
    )

    # 見出し1（<h1>）ごとに分割してカードにする
    sections = re.split(r'(<h1>.*?</h1>)', html_content)
    card_content = ""
    for section in sections:
        if section.startswith('<h1>'):
            # 見出し1が出たとき、新しいカードセクションを開始
            card_content += f'<div class="card">{section}'
        else:
            # 見出し1以外の部分をカード内に追加
            card_content += section + '</div>'

    # カード形式で出力ファイルにHTMLを書き込む
    with open(output_html_file, 'w', encoding='utf-8') as html_file:
        html_file.write(f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>試行錯誤ジャーナル</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            padding: 20px;
        }}
        h1 {{
            font-size: 1.8em;
            color: #333;
        }}
        .codehilite {{
            background: #f8f8f8;
            border: 1px solid #ccc;
            padding: 10px;
            overflow: auto;
        }}
        pre code {{
            display: block;
            padding: 10px;
            font-size: 0.95em;
            line-height: 1.5;
        }}
        /* Pygments用のスタイル */
        .codehilite .hll {{ background-color: #ffffcc }}
        .codehilite .c {{ color: #3D7B7B; font-style: italic }}
        .codehilite .k {{ color: #008000; font-weight: bold }}
        .codehilite .o {{ color: #666666 }}
        /* 他のPygmentsスタイルもここに追加 */
    </style>
</head>
<body>
    {card_content}
</body>
</html>
        """)

    print(f"{input_md_file} を {output_html_file} に変換しました（見出しごとにカード分け、改行を保持）")


if __name__ == "__main__":
    try:
        build_markdown_to_html_with_cards_and_nl2br(
            sys.argv[1], "dist/"+sys.argv[2])
    except Exception:
        print("何らかのエラーが発生しています")

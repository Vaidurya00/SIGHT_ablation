import requests

# 设置 API 密钥和自定义搜索引擎 ID
GOOGLE_API_KEY = "AIzaSyBJCVaR6GF1dq83dQ5_BLL0Q3DreLXei3U"
GOOGLE_CSE_ID = "c4d9d910716004113"

# query 可替换为希望搜索的内容
query = "小米公司是哪一年成立的？"

url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"
response = requests.get(url)

if response.status_code == 200:
    results = response.json()
    items = results.get("items", [])
    for item in items:
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Snippet: {item['snippet']}\n")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
import requests

API_KEY = 'your_api_key'
CX = 'your_cx'

def format_links_as_see_more(links):
    prefix = "Xem thêm:"
    formatted_links = [f"{idx}. {link}" for idx, link in enumerate(links, 1)]
    return f"{prefix}\n" + "\n".join(formatted_links)

def get_google_search_results(query, api_key=API_KEY, cx=CX):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data:
            return format_links_as_see_more([item['link'] for item in data['items'][:3]])
    else:
        print(f"Failed to retrieve search results. Status code: {response.status_code}")
    return []
if __name__ == "__main__":
    # Example usage
    query = 'So sánh Nghị định 123/2020/NĐ-CP với Nghị định 119/2018/NĐ-CP trước đây '

    search_results = get_google_search_results(query)
    print(search_results)

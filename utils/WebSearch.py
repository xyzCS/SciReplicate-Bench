import requests
from utils.utils import llm
from bs4 import BeautifulSoup
import re
import requests  # For making HTTP requests to APIs and websites
import requests
import os
from bs4 import BeautifulSoup

def retrieve_content(url, max_tokens=50000):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            text = soup.get_text(separator=' ', strip=True)
            characters = max_tokens * 4  # Approximate conversion
            text = text[:characters]
            return text
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return None

def get_search_results(search_items):
    # Generate a summary of search results for the given search term
    results_list = []
    for idx, item in enumerate(search_items, start=1):
        url = item.get('link')
        web_content = retrieve_content(url)
        results_list.append(web_content)
    return results_list

class WebSearch:
    def __init__(self, GPT_model='gpt-4o-mini', effort="medium"):
        self.GPT_model = GPT_model
        self.max_tokens = 2048
        self.effort = effort
    
    def searchGoogle(self, search_item, api_key=None, cse_id=None, search_depth=3, site_filter=None):
        service_url = 'https://www.googleapis.com/customsearch/v1'

        params = {
            'q': search_item,
            'key': os.environ["GoogleSearch_API_KEY"],
            'cx': os.environ["GoogleSearch_CSEID"],
            'num': search_depth
        }

        try:
            response = requests.get(service_url, params=params)
            response.raise_for_status()
            results = response.json()

            # Check if 'items' exists in the results
            if 'items' in results:
                if site_filter is not None:
                    
                    # Filter results to include only those with site_filter in the link
                    filtered_results = [result for result in results['items'] if site_filter not in result['link']]

                    if filtered_results:
                        return filtered_results
                    else:
                        print(f"No results with {site_filter} found.")
                        return []
                else:
                    if 'items' in results:
                        return results['items']
                    else:
                        print("No search results found.")
                        return []

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the search: {e}")
            return []

    def retrieve_from_web_gpt(self, latex_code, query, effort='medium', model=None, tokenizer=None):
        prompt = f"""
I will provide a web content and a search query, you should retrieve information related to the query from the web content. Here is the latex snippet and the query:

[Web Content]

{latex_code}

[query]

{query}

Please return the relevant information from the web content and follow the following requirements:
1. Identify and retrieve information from the web content that can answer the given query. Only retrieve most important and relevant information from the web content that directly addresses the query. Make the response concise and to the point.
2. If multiple relevant pieces of information are found, return them as bullet points, separated by '*', for example:
    * [Relevant content]
    * [Relevant content]
    Ensure each bullet point contains at least one complete sentence.
3. If no relevant information is found, return "No relevant information found."

Return your answer:
"""
        answer = llm(prompt, model=self.GPT_model, effort=effort, generate_code=False)
        return answer, len(prompt.split()), len(answer.split())

    def WebsearchGoogle(self, input_query, site_filter=None):
        search_items = self.searchGoogle(search_item=input_query, site_filter=site_filter)
        results = get_search_results(search_items)
        output = ""
        num = 1
        max_chunk_length = 20000
        input_token = 0
        output_token = 0
        for result in results:
            retrieved_combined = ""
            if result and len(result) > max_chunk_length:
                # Split the result into chunks
                chunks = [result[i:i + max_chunk_length] for i in range(0, len(result), max_chunk_length)]
                for chunk in chunks:
                    retrieved_chunk, input, ouput = self.retrieve_from_web_gpt(chunk, input_query)
                    input_token += input
                    output_token += ouput
                    if not retrieved_chunk.startswith("No relevant information found."):
                        retrieved_combined += retrieved_chunk + "\n"
            else:
                chunk = result
                retrieved_chunk, input, ouput = self.retrieve_from_web_gpt(chunk, input_query)
                input_token += input
                output_token += ouput
                if not retrieved_chunk.startswith("No relevant information found."):
                        retrieved_combined += retrieved_chunk + "\n"
            if retrieved_combined and not retrieved_combined.startswith("No relevant information found."):
                output += f"{retrieved_combined.strip()}\n"
                num += 1
        
        return output, input_token, output_token

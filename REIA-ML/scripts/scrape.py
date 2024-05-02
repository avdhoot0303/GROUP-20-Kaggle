import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

def scrape_faqs_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            faq_elements = soup.find_all('div', class_='faq')  # Adjust this based on your HTML structure
            faqs = []
            for faq_element in faq_elements:
                question = faq_element.find('h2').text.strip()
                answer = faq_element.find('p').text.strip()
                faqs.append({"question": question, "answer": answer})
            return faqs
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while scraping {url}: {str(e)}")
        return None

def main():
    urls = [
        "https://www.tesla.com/support/energy/solar-panels/faqs/frequently-asked-questions"
        # Add more URLs as needed
    ]

    all_faqs = []
    for url in tqdm(urls, desc="Scraping FAQs"):
        faqs = scrape_faqs_from_url(url)
        if faqs:
            all_faqs.extend(faqs)

    # Save the FAQs to a JSON file
    with open('faq_data.json', 'w') as json_file:
        json.dump(all_faqs, json_file, indent=4)

    print("FAQs scraped and saved successfully.")

if __name__ == "__main__":
    main()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException, NoSuchElementException
import time
import csv

def setup_driver():
    chrome_options = Options()
    chrome_options.headless = False
    service = Service('./chromedriver-win64/chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def load_and_fetch_title_url(driver, url, nArticles=None):
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='listContent']/ul")))
    time.sleep(5)

    while True:
        try:
            load_more_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//button[text()='Load more']"))
            )
            if load_more_button.is_displayed():
                try:
                    load_more_button.click()
                    print("Clicked the 'Load more' button.")
                    time.sleep(5)
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", load_more_button)
                    print("Clicked the 'Load more' button using JavaScript.")
                    time.sleep(5)
            else:
                print("No more 'Load more' buttons displayed.")
                break
        except (NoSuchElementException, TimeoutException) as e:
            print(f"No more 'Load more' buttons found or an error occurred: {str(e)}")
            break

    articles = driver.find_elements(By.XPATH, "//div[@class='listContent']/ul/li")
    print(f"Total articles found: {len(articles)}")

    if nArticles is None:
        nArticles = len(articles)
    
    titles = []
    links = []
    for i in range(1, nArticles + 1):
        xpath_title = f'//div[@class="listContent"]/ul/li[{i}]//h2[@class="article-head"]'
        title_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_title)))
        title_text = title_element.text
        titles.append(title_text)
        print(f"Article {i} Title: {title_text}")

        xpath_href = f'//div[@class="listContent"]/ul/li[{i}]//a[@class="article-link selfServiceArticleHeaderDetail"]'
        link_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_href)))
        article_url = link_element.get_attribute('href')
        links.append(article_url)

    return links, titles

def fetch_articles_answers(driver, links, titles):
    data = []
    for i in range(len(links)):
        retries = 3
        while retries > 0:
            try:
                driver.get(links[i])
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, '//span/div[@dir="ltr"]')))
                article_text = driver.find_element(By.XPATH, '//span/div[@dir="ltr"]').text
                data.append([titles[i], article_text])
                print(f"Processed Article {i+1}")
                break
            except TimeoutException:
                retries -= 1
                if retries == 0:
                    print(f"Failed to process Article {i+1} after retries.")
                    data.append([titles[i], "Failed to retrieve content after retries."])
                else:
                    print(f"Retrying Article {i+1}... ({3 - retries}/3)")

    return data

def main():
    url = 'https://help-qa.pge.com/s/topic/0TO8M000000GxHCWA0/bill-faqs?language=en_US'
    driver = setup_driver()
    try:
        urls, titles = load_and_fetch_title_url(driver, url)
        faq_data = fetch_articles_answers(driver, urls, titles)
        with open('FAQ.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Title', 'Content']) 
            writer.writerows(faq_data) 
            print(f"Saved {len(faq_data)} articles to FAQ.csv")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()

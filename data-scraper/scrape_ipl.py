from bs4 import BeautifulSoup
import requests
import time
from pathlib import Path

html_content = "".join(open("ipl-page.html", "r").readlines())

# Assuming 'html_content' contains the HTML source of the page
soup = BeautifulSoup(html_content, "html.parser")

articles = soup.find_all("a", {"data-type": "match-reports"})
article_links = [x["href"] for x in articles]

for article in article_links:

    article_name = Path(article).name

    # Skip if the data is already scraped for a particular match
    if Path(f"data/{article_name}.txt").exists():
        continue

    # Sleep for 2 seconds
    time.sleep(2)

    # Get the match report
    report = requests.get(article).text

    article_soup = BeautifulSoup(report, "html.parser")

    relevant_section = article_soup.find_all(
        "div", {"class": "vn-blogDetCntInr ann-width"}
    )[0]

    paras = relevant_section.find_all("p")

    # The first element contains the whole article within which the same article is nested, hence skip this one
    para_texts = [x.text for x in paras][1:]

    article_text = "\n".join(para_texts)

    with open(f"data/{article_name}.txt", "w") as f:
        f.writelines(article_text)

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
    # Sleep for 2 seconds
    time.sleep(2)

    # Get the match report
    report = requests.get(article).text

    article_soup = BeautifulSoup(report, "html.parser")

    relevant_section = article_soup.find_all(
        "div", {"class": "vn-blogDetCntInr ann-width"}
    )[0]

    paras = relevant_section.find_all("p")

    para_texts = [x.text for x in paras]

    article_text = "\n".join(para_texts)

    article_name = Path(article).name

    with open(f"data/{article_name}.txt", "w") as f:
        f.writelines(article_text)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

def get_booking_rating(url: str) -> float:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        xpath = '//div[@data-testid="review-score-component"]/div[1]'
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, xpath)))

        # Find element AFTER waiting
        rating_element = driver.find_element(By.XPATH, xpath)
        rating_text = rating_element.text.strip()

        match = re.search(r"(\d+(?:[.,]\d+)?)", rating_text)
        if match:
            return float(match.group(1).replace(',', '.'))
        else:
            raise ValueError(f"No float found in: {rating_text}")

    except Exception as e:
        print(f"[Booking Scraper Error] {e}")
        return None

    finally:
        driver.quit()


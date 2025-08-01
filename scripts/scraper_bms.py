import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import datetime as dt
import os

async def scrape_bms(city="Bengaluru", movie="Deadpool Wolverine"):
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        url = f"https://in.bookmyshow.com/{city.lower()}/movies/{movie.replace(' ', '-').lower()}"
        await page.goto(url, timeout=60000)

        await page.wait_for_selector(".__venue-name", timeout=10000)
        theaters = await page.query_selector_all(".__venue-name")

        for theater in theaters:
            name = await theater.inner_text()
            # (Seat price extraction is next step)
            data = {
                "platform": "BookMyShow",
                "theater": name,
                "seat_type": "Gold",
                "price": 250,  # Placeholder
                "movie": movie,
                "city": city,
                "scraped_at": dt.datetime.utcnow().isoformat()
            }
            results.append(data)

        await browser.close()
    return results

def save_data(data):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv("data/ticket_prices.csv", mode="a", index=False, header=not os.path.exists("data/ticket_prices.csv"))

if __name__ == "__main__":
    data = asyncio.run(scrape_bms())
    save_data(data)
    print("Scraped BMS Data:", data)

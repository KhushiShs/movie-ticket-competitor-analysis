# scripts/scraper_paytm.py
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import datetime as dt
import os

async def scrape_paytm(city="Bengaluru", movie="Deadpool Wolverine"):
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # set True later
        page = await browser.new_page()

        # NOTE: Paytm Movies URL pattern can vary – adjust after you open it once.
        # Open Paytm Movies, search for the movie+city, copy the resulting URL and hardcode it.
        # Placeholder generic URL (likely needs tweaking):
        url = f"https://paytm.com/movies/{city.lower()}/{movie.replace(' ', '-')}-tickets"
        await page.goto(url, timeout=60000)

        # Adjust selectors after inspecting the page (these are placeholders to get you started)
        await page.wait_for_selector("div._3RZq", timeout=15000)  # theater container (example)
        theaters = await page.query_selector_all("div._3RZq")

        for th in theaters:
            try:
                name_el = await th.query_selector("h3")  # adjust selector
                name = (await name_el.inner_text()).strip() if name_el else "Unknown Theater"
            except:
                name = "Unknown Theater"

            data = {
                "platform": "Paytm",
                "theater": name,
                "seat_type": "Gold",   # placeholder
                "price": 240,          # placeholder – fill real price in Step 4.2
                "movie": movie,
                "city": city,
                "scraped_at": dt.datetime.utcnow().isoformat()
            }
            print("PAYTM:", data)
            results.append(data)

        await browser.close()
    return results

def save_data(data, csv_path="data/ticket_prices.csv"):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

if __name__ == "__main__":
    data = asyncio.run(scrape_paytm())
    save_data(data)
    print("Scraped Paytm Data:", len(data))

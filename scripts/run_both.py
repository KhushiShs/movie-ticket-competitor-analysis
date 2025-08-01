import subprocess
import sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    py = sys.executable
    run([py, "scripts/scraper_bms.py"])
    run([py, "scripts/scraper_paytm.py"])
    print("Done. Data appended to data/ticket_prices.csv")

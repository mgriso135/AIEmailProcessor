import schedule
import time
from email_processor import get_email_data

def job():
    print("Running email fetcher and processor...")
    get_email_data()
    print("Finished running job")

schedule.every(1).minutes.do(job)


while True:
    schedule.run_pending()
    time.sleep(1)
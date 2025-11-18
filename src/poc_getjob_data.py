import os
import logging
import pandas as pd
from dotenv import load_dotenv
from db_youtube_transcripts.database import get_connection_youtube_transcripts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

JOB_ID = "77df278f-e38c-4c28-a198-ef54e67ee35e"
OUTPUT_CSV = "job_videos_data.csv"

try:
    logger.info("Connecting to database...")
    conn = get_connection_youtube_transcripts()
    logger.info("✓ Database connection successful!")

    cursor = conn.cursor()

    # PostgreSQL uses %s for parameters, not ?
    logger.info(f"Querying job_videos for job_id: {JOB_ID}")
    cursor.execute("SELECT * FROM job_videos WHERE job_id = %s", (JOB_ID,))

    rows = cursor.fetchall()

    # Get column names
    colnames = [desc[0] for desc in cursor.description]

    logger.info(f"✓ Query successful! Found {len(rows)} rows")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=colnames)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✓ Data saved to {OUTPUT_CSV}")

    # Display DataFrame info
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Display first few rows
    print("\n" + "=" * 80)
    print(df.to_string())
    print("=" * 80)

    cursor.close()
    conn.close()

except Exception as e:
    logger.error(f"❌ Error: {type(e).__name__}: {str(e)}")
    import traceback

    logger.error(traceback.format_exc())

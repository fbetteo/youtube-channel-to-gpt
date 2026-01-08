"""
Schema migration for API Keys table.
Run this script to add the api_keys table to the database.
"""

from dotenv import load_dotenv
import os

load_dotenv()
import psycopg2

connection = psycopg2.connect(
    database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
    host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
    user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
    password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
    port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
)
connection.autocommit = True


with connection.cursor() as c:

    # Create API_KEYS table
    api_keys_query = """
    CREATE TABLE IF NOT EXISTS api_keys (
        key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES user_credits(user_id) ON DELETE CASCADE,
        
        -- API Key storage (hashed, never store plaintext)
        key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 hash of the API key
        key_prefix VARCHAR(16) NOT NULL,       -- First 16 chars for identification (e.g., "yt_live_AbC1DeF2")
        
        -- Metadata
        name VARCHAR(100) NOT NULL,            -- User-friendly name (e.g., "Production Key")
        
        -- Status and limits
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        rate_limit_tier VARCHAR(20) NOT NULL DEFAULT 'standard' 
            CHECK (rate_limit_tier IN ('free', 'standard', 'premium')),
        
        -- Usage tracking
        last_used_at TIMESTAMP NULL,
        total_requests INTEGER NOT NULL DEFAULT 0,
        total_credits_used INTEGER NOT NULL DEFAULT 0,
        
        -- Timestamps
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMP NULL  -- NULL means never expires
    );
    """

    # Create indexes for performance
    indexes_query = """
    -- Index for looking up keys by hash (primary lookup)
    CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
    
    -- Index for listing user's keys
    CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
    
    -- Index for finding active keys
    CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active) WHERE is_active = TRUE;
    """

    # Add trigger for updated_at
    trigger_query = """
    DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
    CREATE TRIGGER update_api_keys_updated_at 
        BEFORE UPDATE ON api_keys 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        # Execute api_keys table creation
        print("Creating api_keys table...")
        c.execute(api_keys_query)
        print("✓ API Keys table created successfully")

        # Execute indexes creation
        print("Creating indexes...")
        c.execute(indexes_query)
        print("✓ Indexes created successfully")

        # Execute trigger creation
        print("Creating trigger...")
        c.execute(trigger_query)
        print("✓ Trigger created successfully")

        print("\n🎉 API Keys schema created successfully!")

    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        raise

print("API Keys schema creation completed.")
connection.close()

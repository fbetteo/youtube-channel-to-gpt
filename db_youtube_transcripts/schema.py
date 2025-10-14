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

    # Drop existing job_videos table if it exists with wrong schema
    c.execute("DROP TABLE IF EXISTS job_videos CASCADE;")

    # Create JOBS table
    jobs_query = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id UUID PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES user_credits(user_id),
        status VARCHAR(50) NOT NULL DEFAULT 'initializing',
        
        -- Source information (channel or playlist)
        source_type VARCHAR(20) NOT NULL CHECK (source_type IN ('channel', 'playlist')),
        source_id TEXT NOT NULL,
        source_name TEXT NOT NULL,
        channel_name TEXT NULL,  -- For channels
        playlist_id TEXT NULL,   -- For playlists
        
        -- Progress tracking
        total_videos INTEGER NOT NULL DEFAULT 0,
        processed_count INTEGER NOT NULL DEFAULT 0,
        completed INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        
        -- Credit management
        credits_reserved INTEGER NOT NULL DEFAULT 0,
        credits_used INTEGER NOT NULL DEFAULT 0,
        reservation_id UUID NULL,
        
        -- Formatting options
        include_timestamps BOOLEAN NOT NULL DEFAULT false,
        include_video_title BOOLEAN NOT NULL DEFAULT true,
        include_video_id BOOLEAN NOT NULL DEFAULT true,
        include_video_url BOOLEAN NOT NULL DEFAULT true,
        include_view_count BOOLEAN NOT NULL DEFAULT false,
        concatenate_all BOOLEAN NOT NULL DEFAULT false,
        
        -- Lambda processing
        lambda_dispatched_count INTEGER DEFAULT 0,
        lambda_dispatch_time TIMESTAMP NULL,
        prefetch_completed BOOLEAN DEFAULT false,
        
        -- Metadata storage
        videos_metadata JSONB NULL,
        formatting_options JSONB NULL,
        error_message TEXT NULL,
        
        -- Timing information
        start_time TIMESTAMP NOT NULL DEFAULT NOW(),
        end_time TIMESTAMP NULL,
        duration FLOAT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        
        -- Concurrency control
        version INTEGER NOT NULL DEFAULT 1
    );
    """

    # Create JOB_VIDEOS table
    job_videos_query = """
    CREATE TABLE IF NOT EXISTS job_videos (
        job_id UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
        video_id TEXT NOT NULL,
        
        -- Video metadata
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        description TEXT NULL,
        published_at TIMESTAMP NULL,
        channel_id TEXT NULL,
        channel_title TEXT NULL,
        
        -- Duration information
        duration_iso TEXT NULL,        -- ISO 8601 format (PT4M13S)
        duration_seconds INTEGER NULL,
        duration_category VARCHAR(20) CHECK (duration_category IN ('short', 'medium', 'long')),
        
        -- View statistics
        view_count BIGINT NULL,
        like_count INTEGER NULL,
        comment_count INTEGER NULL,
        
        -- Content information
        language VARCHAR(10) NULL,     -- Language code (en, es, etc.)
        default_language VARCHAR(10) NULL,
        category_id INTEGER NULL,
        tags TEXT[] NULL,              -- Array of tags
        
        -- Processing status
        status VARCHAR(30) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
        transcript_language VARCHAR(10) NULL,  -- Language of extracted transcript
        
        -- File information
        file_path TEXT NULL,           -- Local file path (if stored locally)
        s3_key TEXT NULL,             -- S3 object key (if stored in S3)
        file_size INTEGER NULL,        -- File size in bytes
        
        -- Processing details
        processed_at TIMESTAMP NULL,
        error_message TEXT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0,
        
        -- Timestamps
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        
        -- Composite primary key allows same video in different jobs
        PRIMARY KEY (job_id, video_id)
    );
    """

    # Create indexes for performance
    indexes_query = """
    -- Jobs table indexes
    CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_source_type ON jobs(source_type);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
    CREATE INDEX IF NOT EXISTS idx_jobs_reservation_id ON jobs(reservation_id) WHERE reservation_id IS NOT NULL;
    
    -- Job_videos table indexes
    CREATE INDEX IF NOT EXISTS idx_job_videos_job_id ON job_videos(job_id);
    CREATE INDEX IF NOT EXISTS idx_job_videos_status ON job_videos(status);
    CREATE INDEX IF NOT EXISTS idx_job_videos_duration_category ON job_videos(duration_category);
    CREATE INDEX IF NOT EXISTS idx_job_videos_processed_at ON job_videos(processed_at);
    CREATE INDEX IF NOT EXISTS idx_job_videos_language ON job_videos(language);
    """

    # Add triggers for updated_at timestamps
    triggers_query = """
    -- Function to update updated_at timestamp
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    -- Triggers for updated_at
    DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs;
    CREATE TRIGGER update_jobs_updated_at 
        BEFORE UPDATE ON jobs 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
    DROP TRIGGER IF EXISTS update_job_videos_updated_at ON job_videos;
    CREATE TRIGGER update_job_videos_updated_at 
        BEFORE UPDATE ON job_videos 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        # Execute jobs table creation
        print("Creating jobs table...")
        c.execute(jobs_query)
        print("✓ Jobs table created successfully")

        # Execute job_videos table creation
        print("Creating job_videos table...")
        c.execute(job_videos_query)
        print("✓ Job_videos table created successfully")

        # Execute indexes creation
        print("Creating indexes...")
        c.execute(indexes_query)
        print("✓ Indexes created successfully")

        # Execute triggers creation
        print("Creating triggers...")
        c.execute(triggers_query)
        print("✓ Triggers created successfully")

        print("\n🎉 Database schema created successfully!")

    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        raise

print("Schema creation completed.")
connection.close()

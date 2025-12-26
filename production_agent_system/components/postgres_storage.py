import os
import json
import logging
import asyncpg
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class UserProfile(BaseModel):
    """
    Pydantic model representing the user profile structure to be extracted/stored.
    """
    preferences: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    expertise: List[str] = Field(default_factory=list)
    environment: List[str] = Field(default_factory=list)
    personal_info: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"  # Allow extra fields if the schema evolves

class PostgresStorage:
    """
    Production-ready PostgreSQL storage component for managing user profiles and context.
    Uses asyncpg for high-performance asynchronous database operations.
    """

    def __init__(self):
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.database = os.getenv("POSTGRES_DB", "agents_db")
        self.pool: Optional[asyncpg.Pool] = None
        self._table_name = "user_profiles"

    async def initialize(self):
        """
        Initialize the database connection pool and ensure the schema exists.
        Should be called on application startup.
        """
        try:
            if not self.pool:
                self.pool = await asyncpg.create_pool(
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    min_size=1,
                    max_size=10
                )
                logger.info("PostgreSQL connection pool created.")
            
            await self._create_tables()
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL storage: {e}")
            raise

    async def _create_tables(self):
        """
        Create necessary tables if they don't exist.
        """
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            user_id TEXT PRIMARY KEY,
            profile_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        async with self.pool.acquire() as connection:
            await connection.execute(create_table_query)
            logger.info(f"Table '{self._table_name}' checked/created.")

    async def upsert_profile(self, user_id: str, new_profile: UserProfile) -> bool:
        """
        Insert or update a user profile.
        Merges new data with existing data to prevent overwriting.
        """
        if not self.pool:
            await self.initialize()

        try:
            # 1. Get existing profile
            existing_profile = await self.get_profile(user_id)
            
            if existing_profile:
                # 2. Merge logic
                merged_data = {}
                for field in new_profile.model_fields:
                    existing_val = getattr(existing_profile, field)
                    new_val = getattr(new_profile, field)
                    
                    if isinstance(existing_val, list) and isinstance(new_val, list):
                        # Merge and deduplicate
                        # Filter out empty strings just in case
                        merged_set = set(filter(None, existing_val)) | set(filter(None, new_val))
                        merged_data[field] = list(merged_set)
                    else:
                        merged_data[field] = new_val if new_val else existing_val
                
                final_profile = UserProfile(**merged_data)
                logger.info(f"Merged profile for {user_id}: {final_profile}")
            else:
                final_profile = new_profile

            # 3. Save merged profile
            query = f"""
            INSERT INTO {self._table_name} (user_id, profile_data, updated_at)
            VALUES ($1, $2, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                profile_data = EXCLUDED.profile_data,
                updated_at = CURRENT_TIMESTAMP;
            """
            
            # Convert Pydantic model to dict, then to JSON string
            profile_json = final_profile.model_dump_json()
            
            async with self.pool.acquire() as connection:
                await connection.execute(query, user_id, profile_json)
            logger.info(f"Profile for user '{user_id}' upserted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error upserting profile for user '{user_id}': {e}")
            return False

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile by ID.
        """
        if not self.pool:
            await self.initialize()

        query = f"SELECT profile_data FROM {self._table_name} WHERE user_id = $1;"
        
        try:
            async with self.pool.acquire() as connection:
                record = await connection.fetchrow(query, user_id)
            
            if record:
                # record['profile_data'] is already a string containing JSON in asyncpg if not decoded, 
                # but asyncpg usually decodes JSONB to python objects automatically if configured, 
                # or returns a string. Let's handle the string case.
                data = record['profile_data']
                if isinstance(data, str):
                    data = json.loads(data)
                return UserProfile(**data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving profile for user '{user_id}': {e}")
            return None

    async def get_all_profiles(self) -> Dict[str, UserProfile]:
        """
        Retrieve all user profiles.
        """
        if not self.pool:
            await self.initialize()

        query = f"SELECT user_id, profile_data FROM {self._table_name};"
        
        try:
            async with self.pool.acquire() as connection:
                records = await connection.fetch(query)
            
            results = {}
            for record in records:
                user_id = record['user_id']
                data = record['profile_data']
                if isinstance(data, str):
                    data = json.loads(data)
                results[user_id] = UserProfile(**data)
            return results
        except Exception as e:
            logger.error(f"Error retrieving all profiles: {e}")
            return {}

    async def close(self):
        """
        Close the database connection pool.
        """
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed.")

# Singleton instance for easy import
postgres_storage = PostgresStorage()

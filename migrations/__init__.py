"""SQL schema migrations for UploadM8.

**Single source of truth:** ``migrations.runtime_migrations.run_migrations(db_pool)``.

The API calls it from ``app.py`` ``lifespan`` after ``asyncpg`` pool creation.
Do not maintain a second copy of the migration list elsewhere.
"""

import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import select
from app.models import ImportJob

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ImportJob.id, ImportJob.status, ImportJob.error))
        jobs = result.all()
        for j in jobs:
            print(f"Job {j.id}: status={j.status}, error={j.error}")

if __name__ == "__main__":
    asyncio.run(main())

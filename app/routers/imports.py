"""Import job endpoints — create and poll background import jobs."""
import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import CurrentUser, get_current_user
from app.models import ImportJob
from app.schemas.api import ImportJobListResponse, ImportJobStatus, ImportRequest
from app.services.import_job import run_import

router = APIRouter(prefix="/api/import", tags=["import"])
log = structlog.get_logger()


def _job_to_status(job: ImportJob) -> ImportJobStatus:
    return ImportJobStatus(
        job_id=job.id,
        status=job.status,
        total_games=job.total_games,
        processed_games=job.processed_games,
        error=job.error,
        finished_at=job.finished_at,
    )


@router.post("", response_model=ImportJobStatus, status_code=202)
async def create_import_job(
    req: ImportRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = ImportJob(
        user_id=user.user_id,
        platform=req.platform,
        username=req.username,
        period_days=req.period_days,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    asyncio.create_task(run_import(job.id))
    log.info("import_job_created", job_id=job.id, platform=req.platform, username=req.username)
    return _job_to_status(job)


@router.get("/{job_id}", response_model=ImportJobStatus)
async def get_import_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ImportJob).where(ImportJob.id == job_id, ImportJob.user_id == user.user_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found")
    return _job_to_status(job)


@router.get("", response_model=ImportJobListResponse)
async def list_import_jobs(
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ImportJob)
        .where(ImportJob.user_id == user.user_id)
        .order_by(ImportJob.created_at.desc())
    )
    jobs = result.scalars().all()
    return ImportJobListResponse(jobs=[_job_to_status(j) for j in jobs])

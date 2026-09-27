from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import TimeResource
from app.schemas import TimeResourceCreate, TimeResourceUpdate, TimeResourceOut

router = APIRouter(prefix="/time-resources", tags=["Zeitressourcen"])


@router.get("/", response_model=list[TimeResourceOut])
async def list_time_resources(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TimeResource).offset(skip).limit(limit))
    return result.scalars().all()


@router.post("/", response_model=TimeResourceOut, status_code=status.HTTP_201_CREATED)
async def create_time_resource(payload: TimeResourceCreate, db: AsyncSession = Depends(get_db)):
    tr = TimeResource(**payload.model_dump())
    db.add(tr)
    await db.flush()
    await db.refresh(tr)
    await db.commit()
    await db.refresh(tr)
    return tr


@router.get("/{tr_id}", response_model=TimeResourceOut)
async def get_time_resource(tr_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TimeResource).where(TimeResource.id == tr_id))
    tr = result.scalar_one_or_none()
    if not tr:
        raise HTTPException(status_code=404, detail="Zeitressource nicht gefunden")
    return tr


@router.patch("/{tr_id}", response_model=TimeResourceOut)
async def update_time_resource(tr_id: int, payload: TimeResourceUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TimeResource).where(TimeResource.id == tr_id))
    tr = result.scalar_one_or_none()
    if not tr:
        raise HTTPException(status_code=404, detail="Zeitressource nicht gefunden")
    for key, val in payload.model_dump(exclude_unset=True).items():
        setattr(tr, key, val)
    await db.commit()
    await db.refresh(tr)
    return tr


@router.delete("/{tr_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_time_resource(tr_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TimeResource).where(TimeResource.id == tr_id))
    tr = result.scalar_one_or_none()
    if not tr:
        raise HTTPException(status_code=404, detail="Zeitressource nicht gefunden")
    await db.execute(delete(TimeResource).where(TimeResource.id == tr_id))
    await db.commit()

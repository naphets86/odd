from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import Allocation
from app.schemas import AllocationCreate, AllocationUpdate, AllocationOut

router = APIRouter(prefix="/allocations", tags=["Zuteilungen"])


@router.get("/", response_model=list[AllocationOut])
async def list_allocations(
    project_id: int | None = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    q = select(Allocation).offset(skip).limit(limit)
    if project_id is not None:
        q = q.where(Allocation.project_id == project_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/", response_model=AllocationOut, status_code=status.HTTP_201_CREATED)
async def create_allocation(payload: AllocationCreate, db: AsyncSession = Depends(get_db)):
    allocation = Allocation(**payload.model_dump())
    db.add(allocation)
    await db.flush()
    await db.refresh(allocation)
    await db.commit()
    await db.refresh(allocation)
    return allocation


@router.get("/{allocation_id}", response_model=AllocationOut)
async def get_allocation(allocation_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Allocation).where(Allocation.id == allocation_id))
    allocation = result.scalar_one_or_none()
    if not allocation:
        raise HTTPException(status_code=404, detail="Zuteilung nicht gefunden")
    return allocation


@router.patch("/{allocation_id}", response_model=AllocationOut)
async def update_allocation(allocation_id: int, payload: AllocationUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Allocation).where(Allocation.id == allocation_id))
    allocation = result.scalar_one_or_none()
    if not allocation:
        raise HTTPException(status_code=404, detail="Zuteilung nicht gefunden")
    for key, val in payload.model_dump(exclude_unset=True).items():
        setattr(allocation, key, val)
    await db.commit()
    await db.refresh(allocation)
    return allocation


@router.delete("/{allocation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_allocation(allocation_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Allocation).where(Allocation.id == allocation_id))
    allocation = result.scalar_one_or_none()
    if not allocation:
        raise HTTPException(status_code=404, detail="Zuteilung nicht gefunden")
    await db.execute(delete(Allocation).where(Allocation.id == allocation_id))
    await db.commit()

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import Material
from app.schemas import MaterialCreate, MaterialUpdate, MaterialOut

router = APIRouter(prefix="/materials", tags=["Material"])


@router.get("/", response_model=list[MaterialOut])
async def list_materials(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Material).offset(skip).limit(limit))
    return result.scalars().all()


@router.post("/", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
async def create_material(payload: MaterialCreate, db: AsyncSession = Depends(get_db)):
    material = Material(**payload.model_dump())
    db.add(material)
    await db.flush()
    await db.refresh(material)
    await db.commit()
    await db.refresh(material)
    return material


@router.get("/{material_id}", response_model=MaterialOut)
async def get_material(material_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Material).where(Material.id == material_id))
    material = result.scalar_one_or_none()
    if not material:
        raise HTTPException(status_code=404, detail="Material nicht gefunden")
    return material


@router.patch("/{material_id}", response_model=MaterialOut)
async def update_material(material_id: int, payload: MaterialUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Material).where(Material.id == material_id))
    material = result.scalar_one_or_none()
    if not material:
        raise HTTPException(status_code=404, detail="Material nicht gefunden")
    for key, val in payload.model_dump(exclude_unset=True).items():
        setattr(material, key, val)
    await db.commit()
    await db.refresh(material)
    return material


@router.delete("/{material_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_material(material_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Material).where(Material.id == material_id))
    material = result.scalar_one_or_none()
    if not material:
        raise HTTPException(status_code=404, detail="Material nicht gefunden")
    await db.execute(delete(Material).where(Material.id == material_id))
    await db.commit()

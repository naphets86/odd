from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.database import get_db
from app.models import Person
from app.schemas import PersonCreate, PersonUpdate, PersonOut

router = APIRouter(prefix="/persons", tags=["Personen"])


@router.get("/", response_model=list[PersonOut])
async def list_persons(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Person).offset(skip).limit(limit))
    return result.scalars().all()


@router.post("/", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def create_person(payload: PersonCreate, db: AsyncSession = Depends(get_db)):
    person = Person(**payload.model_dump())
    db.add(person)
    await db.flush()
    person_id = person.id
    await db.commit()
    # Re-fetch the person to ensure all attributes are loaded
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one()
    return person


@router.get("/{person_id}", response_model=PersonOut)
async def get_person(person_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person nicht gefunden")
    return person


@router.patch("/{person_id}", response_model=PersonOut)
async def update_person(person_id: int, payload: PersonUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person nicht gefunden")
    for key, val in payload.model_dump(exclude_unset=True).items():
        setattr(person, key, val)
    await db.commit()
    await db.refresh(person)
    return person


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(person_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person nicht gefunden")
    await db.execute(delete(Person).where(Person.id == person_id))
    await db.commit()

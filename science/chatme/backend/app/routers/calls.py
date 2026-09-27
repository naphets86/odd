from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Call
from app.schemas.schemas import CallOut
from typing import List

router = APIRouter(prefix="/api/calls", tags=["calls"])


@router.get("", response_model=List[CallOut])
async def list_calls(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Call).where(
            or_(Call.caller_id == current_user.id, Call.callee_id == current_user.id)
        ).order_by(Call.started_at.desc()).limit(100)
    )
    return result.scalars().all()

"""The site location, reported by the dashboard from the browser's geolocation."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session

from ..db import get_session
from ..deps import get_settings
from ..location import current_site_location, update_site_location
from ..schemas import DashboardLocation

router = APIRouter(prefix="/api", tags=["location"])


class LocationReport(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy: float | None = Field(None, ge=0, description="Error radius in meters")


@router.get("/location", response_model=DashboardLocation)
def get_location(session: Session = Depends(get_session), settings=Depends(get_settings)):
    name, coords = current_site_location(settings, session)
    return DashboardLocation(name=name, coords=coords or "")


@router.put("/location", response_model=DashboardLocation)
def report_location(report: LocationReport, settings=Depends(get_settings)):
    """Set where the system is now. New events are stamped with this location."""
    site = update_site_location(settings, report.latitude, report.longitude, report.accuracy)
    name, coords = current_site_location(settings)
    return DashboardLocation(name=site.name or name, coords=coords or "")

"""The site's current location: reported by the dashboard's browser, named via OpenStreetMap.

Which location an event or the dashboard header uses, in order:
1. the camera's own location, if it was started with one,
2. the last location reported by this station's dashboard browser (stored in the
   database, one row per station),
3. FALL_SITE_LOCATION / FALL_SITE_COORDS.
"""

import json
import logging
import math
import urllib.parse
import urllib.request

from sqlmodel import Session, select

from .db import SiteLocation, get_engine, utc_now

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
# Nominatim's usage policy requires an identifying User-Agent.
USER_AGENT = "Oxy-Worker-Fall-Detection/0.1 (+https://github.com/EvanGruhlkey/Worker-Fall-Detection)"
# Reports closer than this to the stored point keep the stored name, which
# avoids a lookup on every page load (Nominatim allows 1 request per second).
REUSE_NAME_WITHIN_METERS = 500


def format_coords(latitude, longitude):
    return f"{latitude:.4f}, {longitude:.4f}"


def distance_meters(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)."""
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def reverse_geocode(latitude, longitude, timeout=5):
    """'City, State, Country' for a point, or None if the lookup fails."""
    query = urllib.parse.urlencode(
        {"format": "jsonv2", "lat": latitude, "lon": longitude, "zoom": 10, "addressdetails": 1}
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            address = json.load(response).get("address", {})
    except Exception as error:
        log.warning("Reverse geocoding failed: %s", error)
        return None

    city = next(
        (address[key] for key in ("city", "town", "village", "hamlet", "municipality", "county")
         if address.get(key)),
        None,
    )
    parts = [part for part in (city, address.get("state"), address.get("country")) if part]
    return ", ".join(parts) or None


def _station_site(session, settings):
    return session.exec(
        select(SiteLocation).where(SiteLocation.station == settings.station_name)
    ).first()


def update_site_location(settings, latitude, longitude, accuracy=None):
    """Store a location reported by the browser; returns the stored row."""
    with Session(get_engine()) as session:
        site = _station_site(session, settings) or SiteLocation(station=settings.station_name)

        name = None
        moved = site.latitude is None or distance_meters(
            site.latitude, site.longitude, latitude, longitude
        ) > REUSE_NAME_WITHIN_METERS
        # Reuse a real place name; a coordinates fallback is looked up again.
        named = site.name and site.name != format_coords(site.latitude, site.longitude)
        if not moved and named:
            name = site.name
        elif settings.reverse_geocode:
            name = reverse_geocode(latitude, longitude)

        site.latitude = latitude
        site.longitude = longitude
        site.accuracy = accuracy
        # Without a place name, show the coordinates rather than nothing.
        site.name = name or format_coords(latitude, longitude)
        site.updated_at = utc_now()
        session.add(site)
        session.commit()
        session.refresh(site)
        return site


def current_site_location(settings, session=None):
    """(name, coords) for the site: browser-reported if known, else settings."""
    if session is None:
        with Session(get_engine()) as own_session:
            return current_site_location(settings, own_session)
    site = _station_site(session, settings)
    if site and site.latitude is not None:
        return site.name, format_coords(site.latitude, site.longitude)
    return settings.site_location, settings.site_coords

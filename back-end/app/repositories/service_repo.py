import difflib
from typing import List, Optional

from sqlmodel import Session

from app.core.cache import service_catalog_cache
from app.core.config import settings
from app.models.domain import Service
from app.repositories.base import BaseRepository

_CATALOG_CACHE_KEY = "all_services"


class ServiceRepository(BaseRepository[Service]):
    """
    Data access layer for spa `Service` catalog queries and fuzzy service matching.
    Uses Tier-1 TTL caching (`service_catalog_cache`) so repeated booking and
    modification turns do not hit PostgreSQL on every turn.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, Service)

    def list_all(self) -> List[Service]:
        cached_services = service_catalog_cache.get(_CATALOG_CACHE_KEY)
        if cached_services is not None:
            return cached_services

        db_services = super().list_all()
        # Snapshot into detached SQLModel instances so they can be safely read across sessions
        detached_snapshots = [
            Service(
                id=svc.id,
                name=svc.name,
                description=svc.description,
                price=svc.price,
                duration_minutes=svc.duration_minutes,
                category_id=svc.category_id,
            )
            for svc in db_services
        ]
        if detached_snapshots:
            service_catalog_cache.set(_CATALOG_CACHE_KEY, detached_snapshots)
        return detached_snapshots

    @staticmethod
    def invalidate_catalog_cache() -> None:
        """Clear the cached service catalog (call when services are added or modified)."""
        service_catalog_cache.clear()

    def match_service(
        self,
        requested_service_name: str,
        cutoff: Optional[float] = None,
    ) -> tuple[Optional[Service], tuple[str, ...]]:
        """Return the catalog match, or the choices from the same read when it misses."""
        services = self.list_all()
        lower_to_service = {service.name.lower(): service for service in services}
        effective_cutoff = cutoff if cutoff is not None else settings.FUZZY_MATCH_CUTOFF

        closest_matches = difflib.get_close_matches(
            requested_service_name.strip().lower(),
            list(lower_to_service.keys()),
            n=1,
            cutoff=effective_cutoff,
        )
        if not closest_matches:
            return None, tuple(service.name for service in services)

        return lower_to_service[closest_matches[0]], ()

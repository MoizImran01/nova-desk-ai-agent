from typing import Generic, List, Optional, Type, TypeVar
import uuid

from sqlmodel import SQLModel, Session, select

ModelType = TypeVar("ModelType", bound=SQLModel)


class BaseRepository(Generic[ModelType]):
    """
    Generic base repository encapsulating common SQLModel data access patterns.
    IMPORTANT: Repository methods use `session.flush()` rather than `session.commit()`
    so that multiple repository calls can participate in a single atomic transaction
    (Unit of Work).
    """

    def __init__(self, session: Session, model: Type[ModelType]) -> None:
        self.session = session
        self.model = model

    def get_by_id(self, entity_id: uuid.UUID | str) -> Optional[ModelType]:
        try:
            parsed_id = uuid.UUID(str(entity_id)) if not isinstance(entity_id, uuid.UUID) else entity_id
        except (ValueError, TypeError):
            return None
        return self.session.get(self.model, parsed_id)

    def list_all(self) -> List[ModelType]:
        statement = select(self.model)
        return list(self.session.exec(statement).all())

    def add(self, entity: ModelType) -> ModelType:
        self.session.add(entity)
        self.session.flush()
        return entity

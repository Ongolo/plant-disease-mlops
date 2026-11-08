import os
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String, create_engine,
                        inspect, text)
from sqlalchemy.orm import declarative_base, sessionmaker

# Adresse de la base : PostgreSQL dans Docker, SQLite par defaut en local
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./usage.db")
# Format : "alice:token1,bob:token2,ui:token3"
API_TOKENS = os.getenv("API_TOKENS", "")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class User(Base):
    """Un utilisateur autorise a appeler l'API, identifie par son token."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    token = Column(String, unique=True, nullable=False, index=True)


class PredictionLog(Base):
    """Une ligne par prediction : sert au suivi d'usage (et plus tard au data drift)."""
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    model_version = Column(String)
    predicted_class = Column(String)
    confidence = Column(Float)
    latency_ms = Column(Float)
    # Statistiques de l'image recue (utilisees pour le data drift a l'etape 15)
    width = Column(Integer)
    height = Column(Integer)
    brightness = Column(Float)
    contrast = Column(Float)
    mean_r = Column(Float)
    mean_g = Column(Float)
    mean_b = Column(Float)


def _add_missing_columns() -> None:
    """Petite migration : ajoute la colonne model_version si la table existait deja sans elle."""
    cols = [c["name"] for c in inspect(engine).get_columns("prediction_logs")]
    if "model_version" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE prediction_logs ADD COLUMN model_version VARCHAR"))


def init_db() -> None:
    """Cree les tables si besoin, puis cree/met a jour les utilisateurs de API_TOKENS."""
    Base.metadata.create_all(engine)
    _add_missing_columns()
    pairs = [p.split(":", 1) for p in API_TOKENS.split(",") if ":" in p]
    with SessionLocal() as db:
        for name, token in pairs:
            name, token = name.strip(), token.strip()
            user = db.query(User).filter_by(name=name).first()
            if user is None:
                db.add(User(name=name, token=token))
            elif user.token != token:
                user.token = token  # le token a change dans .env
        db.commit()


def get_db():
    """Fournit une session de base de donnees a chaque requete, puis la ferme."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
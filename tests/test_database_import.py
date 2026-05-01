def test_database_imports():
    from app.database import Base, get_db, engine
    assert Base is not None
    assert get_db is not None
    assert engine is not None
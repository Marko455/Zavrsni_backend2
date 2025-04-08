# app/models.py
from sqlalchemy import Column, Integer, String, Float
from app.database import Base
from sqlalchemy.inspection import inspect
from sqlalchemy import ForeignKey, DateTime

#-------------------------------------------------------------Tablica nekretnina --------------------------------------------
class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    location = Column(String(255))
    price = Column(Float)
    description = Column(String(1000))
    image = Column(String(500))
    seller = Column(String(500), nullable=False)
    contact = Column(String(500))
    property_type = Column(String(500))
    parking_space = Column(Integer)
    sq_ft = Column(Float)
    rooms = Column(Integer)
    bathrooms = Column(Integer)
    bedrooms = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)

#-------------------------------------------------------------Tablica prodanih nekretnina --------------------------------------------
class BoughtProperty(Base):
    __tablename__ = "bought_properties"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    location = Column(String(255))
    price = Column(Float)
    description = Column(String(1000))
    seller = Column(String(500), nullable=False)
    contact = Column(String(500))
    property_type = Column(String(500))
    parking_space = Column(Integer)
    sq_ft = Column(Float)
    rooms = Column(Integer)
    bathrooms = Column(Integer)
    bedrooms = Column(Integer)

#-------------------------------------------------------------Tablica buyers --------------------------------------------
class Buyer(Base):
    __tablename__ = "buyers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

#-------------------------------------------------------------Tablica sellers --------------------------------------------
class Seller(Base):
    __tablename__ = "sellers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

#-------------------------------------------------------------Tablica agent --------------------------------------------
class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    contact = Column(String(100), unique=True, nullable=False)
    field = Column(String(50), nullable=False)


#-------------------------------------------------------------Tablica agent --------------------------------------------
class ScheduleTour(Base):
    __tablename__ = "schedule_tour"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    visit_date_time = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey("buyers.id"), nullable=True)
    user_email = Column(String(100), unique=True, nullable=False)


#------------------------------------------------------- Schema -----------------------------------------

def generate_schema_description():
    schema_description = "Database Schema:\n"
    schema_name = "baza_podataka"
    
    schema_description += f"Schema Name: {schema_name}\n\n"
    schema_description += "Tables and Columns:\n\n"
    
    # Lista modela za koje treba schema
    models = [Property, 
              #BoughtProperty, 
              Buyer, Seller, Agent]
    
    for model in models:
        table_name = f"{schema_name}.{model.__tablename__.lower()}"
        schema_description += f"Table: {table_name}\n"
        mapper = inspect(model)
        for column in mapper.columns:
            column_name = column.name.lower()
            column_type = str(column.type)
            if column.primary_key:
                column_type += ", primary key"
            schema_description += f"  - {column_name} ({column_type})\n"
        schema_description += "\n"
    
    return schema_description

schema = generate_schema_description()
print("Generirana Schema Description:\n", schema)


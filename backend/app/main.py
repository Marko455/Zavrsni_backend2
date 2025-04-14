# app/main.py

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app import models
from pydantic import BaseModel
from passlib.context import CryptContext
from llama_index.llms.ollama import Ollama
from sqlalchemy import func
from sqlalchemy.sql import text
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import and_
from app.models import schema
from app.models import Property
from datetime import datetime

app = FastAPI()

# ------------------------ CORS ---------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ Llama model i embeddings model ---------------------------
llm = Ollama(model="llama3.2:3b", request_timeout=150.0)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ Hashiranje lozinke ---------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


Base.metadata.create_all(bind=engine)
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

# ------------------------ Pydantic modeli ---------------------------
class SellerCreate(BaseModel):
    name: str
    email: str
    password: str

class SellerLogin(BaseModel):
    email: str
    password: str

class BuyerCreate(BaseModel):
    name: str
    email: str
    password: str

class BuyerLogin(BaseModel):
    email: str
    password: str

class AgentCreate(BaseModel):
    name: str
    contact: str
    field: str

class PropertyCreate(BaseModel):
    title: str
    location: str
    price: float
    description: str
    image: str
    seller: str
    contact: str
    property_type: str
    parking_space: int
    sq_ft: float
    rooms: int
    bathrooms: int
    bedrooms: int
    latitude: float
    longitude: float

class PropertyBuy(BaseModel):
    title: str
    location: str
    price: float
    description: str
    seller: str
    contact: str
    property_type: str
    parking_space: int
    sq_ft: float
    rooms: int
    bathrooms: int
    bedrooms: int

class PropertySell(BaseModel):
    title: str
    location: str
    price: float
    description: str
    seller: str
    contact: str
    property_type: str
    parking_space: int
    sq_ft: float
    rooms: int
    bathrooms: int
    bedrooms: int
    latitude: float
    longitude: float

class ScheduleTourCreate(BaseModel):
    property_id: int
    visit_date_time: datetime
    user_id: int
    user_email: str

class QuestionRequest(BaseModel):
    question: str
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "asc"

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# -------------------------------------------------------- Pomoćne funkcije -------------------------------------------------

# Embedding tekst
def generate_embedding(text: str):
    embedding = embedding_model.encode(text, convert_to_numpy=True)  # Generiranje embedding-a koristenjem SentenceTransformer
    return embedding


def initialize_faiss_index(db: Session):
    properties = db.query(models.Property).all()
    for prop in properties:
        try:
            embedding = generate_embedding(prop.description)
            index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([prop.id], dtype=np.int64))
        except Exception as e:
            print(f"Failed to add property {prop.id} to FAISS: {e}")


#  FAISS indeksiranje
with Session(engine) as db:
    initialize_faiss_index(db)

# ------------------------ Rute ---------------------------

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with MySQL!"}

# ------------------------ RAG Ruta ---------------------------

@app.post("/ask/")
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """
    Accepts a question, retrieves the most relevant properties from the FAISS index, 
    and uses Llama to generate an answer based on the property data.
    """
    question = request.question

    try:
        question_embedding = generate_embedding(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    k = 5 
    try:
        distances, indices = index.search(np.array([question_embedding], dtype=np.float32), k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying FAISS index: {str(e)}")

    properties = []
    for idx in indices[0]:
        if idx != -1:
            property = db.query(Property).filter(Property.id == idx).first()
            if property:
                properties.append({
                    "title": property.title,
                    "location": property.location,
                    "price": property.price,
                    "description": property.description,
                    "type": property.property_type,
                    "rooms": property.rooms,
                    "bathrooms": property.bathrooms,
                    "bedrooms": property.bedrooms,
                })

    if not properties:
        return {"answer": "No relevant properties found."}

    context = "\n".join([
        f"Property {i+1}:\n"
        f"  Title: {prop['title']}\n"
        f"  Location: {prop['location']}\n"
        f"  Price: ${prop['price']:,.2f}\n"
        f"  Description: {prop['description']}\n"
        f"  Type: {prop['type']}\n"
        f"  Rooms: {prop['rooms']}\n"
        f"  Bathrooms: {prop['bathrooms']}\n"
        f"  Bedrooms: {prop['bedrooms']}\n"
        for i, prop in enumerate(properties)
    ])
    prompt = f"""
        You are an expert real estate assistant. Answer the following question using the provided property data.

    Question: {question}

    Property Data:{context}

    Answer:
    """

    try:
        response = llm.complete(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Llama: {str(e)}")

    return {"answer": response}



# ------------------------ Ponovno FAAS indeksiranje testiranje ---------------------------
@app.post("/reindex/")
def reindex_properties(db: Session = Depends(get_db)):
    initialize_faiss_index(db)
    return {"message": "Reindexed all properties in FAISS."}
print(f"Number of indexed properties: {index.ntotal}")

# ------------------------ Svi selleri  ---------------------------
@app.get("/sellers/")
def get_sellers(db: Session = Depends(get_db)):
    sellers = db.query(models.Seller).all()
    return sellers

# ------------------------ Ruta kreirananja novog sellera ---------------------------
@app.post("/sellers/")
def create_seller(seller: SellerCreate, db: Session = Depends(get_db)):
    db_seller = db.query(models.Seller).filter(models.Seller.email == seller.email).first()
    if db_seller:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(seller.password)
    db_seller = models.Seller(name=seller.name, email=seller.email, password=hashed_password)
    db.add(db_seller)
    db.commit()
    db.refresh(db_seller)
    return db_seller

# ------------------------ Ruta prijave sellera ---------------------------
@app.post("/login_seller/")
def login_seller(seller: SellerLogin, db: Session = Depends(get_db)):
    db_seller = db.query(models.Seller).filter(models.Seller.email == seller.email).first()
    if not db_seller or not verify_password(seller.password, db_seller.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {"message": "Login successful", "id": db_seller.id, "name": db_seller.name, "email": db_seller.email, "role": "seller"}

# ------------------------ Svi buyeri  ---------------------------
@app.get("/buyers/")
def get_buyers(db: Session = Depends(get_db)):
    buyers = db.query(models.Buyer).all()
    return buyers

# ------------------------ Ruta kreirananja novog kupca ---------------------------
@app.post("/buyers/")
def create_buyer(buyer: BuyerCreate, db: Session = Depends(get_db)):
    db_buyer = db.query(models.Buyer).filter(models.Buyer.email == buyer.email).first()
    if db_buyer:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(buyer.password)
    db_buyer = models.Buyer(name=buyer.name, email=buyer.email, password=hashed_password)
    db.add(db_buyer)
    db.commit()
    db.refresh(db_buyer)
    return db_buyer

# ------------------------ Ruta prijave buyera ---------------------------
@app.post("/login_buyer/")
def login_buyer(buyer: BuyerLogin, db: Session = Depends(get_db)):
    db_buyer = db.query(models.Buyer).filter(models.Buyer.email == buyer.email).first()
    if not db_buyer or not verify_password(buyer.password, db_buyer.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {"message": "Login successful", "id": db_buyer.id, "name": db_buyer.name, "email": db_buyer.email, "role": "buyer"} #"role": "buyer" je novo


# ------------------------ Ruta kreirananja novog agenta ---------------------------
@app.post("/agents/")
def create_agent(agent: AgentCreate, db: Session = Depends(get_db)):
    db_agent = db.query(models.Agent).filter(models.Agent.contact == agent.contact).first()
    if db_agent:
        raise HTTPException(status_code=400, detail="Contact already registered")

    db_agent = models.Agent(name=agent.name, contact=agent.contact, field=agent.field)
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    return db_agent

# ------------------------ Ruta svih agenata ---------------------------
@app.get("/agents/")
def get_agents(db: Session = Depends(get_db)):
    agents = db.query(models.Agent).all()
    return agents

# ------------------------ Ruta svih nekretnina ---------------------------
@app.get("/properties/")
def get_properties(db: Session = Depends(get_db)):
    properties = db.query(models.Property).all()
    return properties

@app.get("/properties/{property_id}")
def get_property(property_id: int, db: Session = Depends(get_db)):
    property = db.query(models.Property).filter(models.Property.id == property_id).first()
    if property is None:
        raise HTTPException(status_code=404, detail="Property not found")
    return property

# ------------------------ Ruta slanja nove nekretnine --------------------------- 
@app.post("/properties/")
def create_property(property: PropertyCreate, db: Session = Depends(get_db)):

    db_property = models.Property(
        title=property.title,
        location=property.location,
        price=property.price,
        description=property.description,
        image=property.image,
        seller=property.seller,
        contact=property.contact,
        property_type=property.property_type,
        parking_space=property.parking_space,
        sq_ft=property.sq_ft,
        rooms=property.rooms,
        bathrooms=property.bathrooms,
        bedrooms=property.bedrooms,
        latitude=property.latitude,
        longitude=property.longitude
    )

    db.add(db_property)
    db.commit()
    db.refresh(db_property)
    return db_property

# ------------------------ Ruta kupovanja nekretnine --------------------------- 
@app.post("/buy_property/")
def buy_property(property: PropertyBuy, db: Session = Depends(get_db)):

    db_bought_property = models.BoughtProperty(
        title=property.title,
        location=property.location,
        price=property.price,
        description=property.description,
        seller=property.seller,
        contact=property.contact,
        property_type=property.property_type,
        parking_space=property.parking_space,
        sq_ft=property.sq_ft,
        rooms=property.rooms,
        bathrooms=property.bathrooms,
        bedrooms=property.bedrooms,
    )

    db.add(db_bought_property)
    db.commit()

    embedding = generate_embedding(db_bought_property.description)
    index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([db_bought_property.id], dtype=np.int64))

    db.refresh(db_bought_property)
    return db_bought_property

# ------------------------ Ruta azuriranja nekretnina ---------------------------
@app.put("/properties/{property_id}")
def update_property(property_id: int, property: PropertyCreate, db: Session = Depends(get_db)):
    db_property = db.query(models.Property).filter(models.Property.id == property_id).first()
    if db_property is None:
        raise HTTPException(status_code=404, detail="Property not found")

    for key, value in property.dict().items():
        setattr(db_property, key, value)

    db.commit()
    db.refresh(db_property)

    embedding = generate_embedding(db_property.description)
    index.remove_ids(np.array([property_id], dtype=np.int64))
    index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([property_id], dtype=np.int64))

    return db_property



#------------------------- Rute spremanja razgledavanja nekretnina ----------------------
@app.post("/schedule_tour/")
def scheduleTour(tour: ScheduleTourCreate, db: Session = Depends(get_db)):

    scheduleTour = models.ScheduleTour(
        property_id=tour.property_id,
        visit_date_time=tour.visit_date_time,
        user_id=tour.user_id,
        user_email=tour.user_email,
    )

    db.add(scheduleTour)
    db.commit()

    db.refresh(scheduleTour)
    return scheduleTour
#--------------------------------------------------------------------------------------------------------------------
class SQLQueryRequest(BaseModel):
    question: str
    sort_by: Optional[str]
    sort_order: Optional[str]
    

class SQLQueryResponse(BaseModel):
    sql_query: str
    results: List[dict]

# Generiraj SQL upit sa Llamom

def generate_sql_with_llama(question: str) -> str:
    """
    Use Llama to generate an SQL query from a natural language question.
    """
    prompt = f"""
    You are a database assistant. Convert the following question into a valid MySQL SQL query.
    
    Only return the SQL query and do not add any of your additional explanation or comments.
    
    Question: {question}
    
    SQL Query:
    """
    try:
        response = llm.complete(prompt)
        sql_query = response.strip()
        return sql_query
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL query: {str(e)}")


# Provjeri točnost SQL upita dali je točan i da ne sadržava određene naredbe
def validate_sql_query(sql_query: str) -> bool:
    """
    Validate the generated SQL query to prevent harmful operations.
    Disallow specific commands for safety.
    """
    disallowed_commands = ["DROP", "DELETE","JOIN","LIMIT","LEFT JOIN","RIGHT JOIN","FULL JOIN","UNION","CROSS JOIN"]
    sql_command = sql_query.strip().split()[0].upper()
    print(sql_query)
    return sql_command not in disallowed_commands


# Izvršavanje SQL upita
def execute_sql_query(sql_query: str, db: Session) -> list:
    """
    Execute a validated SQL query against the database.
    """
    try:
        result = db.execute(text(sql_query))
        rows = result.fetchall()
        columns = result.keys()  # Get column names
        print("Raw SQL result from DB: {rows}")  

        # Convert tuples to dictionaries
        result_dicts = [dict(zip(columns, row)) for row in rows]
        print("Converted result: {result_dicts}")

        return result_dicts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL query: {str(e)}")


# API ruta koja koristi funkcije generiranja i izvršavanja SQL upita
@app.post("/sql_query/")
def generate_and_execute_sql(
    request: SQLQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts a natural language question, generates an SQL query using Llama,
    validates the query, and executes it on the database.
    Allows sorting preferences before executing the query.
    """
    question = request.question
    sort_by = request.sort_by if hasattr(request, "sort_by") else None  # Opcionalni stupac po kojem će sortirati
    sort_order = request.sort_order if hasattr(request, "sort_order") else "asc"  # Default: rastući poredak

    print("Korisnicko pitanje:", question)
    print("Izbor sortiranja - Stupac:", sort_by, "Poredak:", sort_order)

    # 1. Generiranje SQL upita koristeći Llamu ("specialized in real estate data" je NOVO, NEMOJ ZABORAVITI!!!)
    prompt = f"""
    You are an expert SQL assistant specialized in real estate data. Generate an SQL query based on the user's question and the following database schema:
    {schema}

    Ensure that the query includes all relevant columns from the table, does not use JOIN commands, and applies sorting if specified.
    Use the format 'baza_podataka.table_name'.
    
    Sort column: {sort_by if sort_by else "None"}
    Sort order: {sort_order}

    Question: {question}

    SQL Query:
    """
    try:
        print("Prompt:" ,prompt)
        response = llm.complete(prompt)
        print(response)
        sql_query = response.text.strip()  # Izvlačenje SQL upita iz generiranog odgovora Llama modela (brisanje teksta i objašnjavanja)
        print("SQL upit generiran Llamom:", sql_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL query: {str(e)}")

    # 2. Provjeravanj da SQL upit počinje sa SELECT naredbom zbog zaštite baze podataka
    if not validate_sql_query(sql_query):
        raise HTTPException(status_code=400, detail="Invalid SQL query generated. Only SELECT queries are allowed.")

    print("SQL query prolazi validaciju: {sql_query}")

    print("SQL upit prije dodavanja sortiranja: ", sql_query)

    # 3. Add sorting to the SQL query if sorting preferences are provided
    
    print("SQL upit nakon dodavanja sortiranja: ", sql_query)

    # 4. Izvršavanje SQL upita
    try:
        print("Zavrsni oblik SQL upita koji ce se izvrsiti:", sql_query)
        print("Pozivanje execute_sql_query() funkcije...")
        result_dicts = execute_sql_query(sql_query, db)
        print("Funkcija execute_sql_query() vraca: {result_dicts}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing SQL query: {str(e)}")

    # 5. Generiranje ljudski čitljiv odgovor od rezultata SQL upita
    if not result_dicts:
        return {
            "answer": "I'm sorry, but I couldn't find any properties matching your query.",
            "sql_query": sql_query,
            "results": []
        }

    # Definiranje najbitnijih polja za bolje generiranje odgovora
    relevant_fields = {
        "location": "located in",
        "price": "priced at",
        "bedrooms": "with",
        "bathrooms": "offering",
        "title": "titled",
        "description": "described as",
        "property_type": "property type is",
        "parking_space": "with parking space for",
        "sq_ft": "covering",
        "rooms": "featuring"
    }

    # Generiranje "razgovornih" rečenica
    sentences = []
    for row in result_dicts:
        details = []
        for key, value in row.items():
            key_lower = key.lower()
            if key_lower in relevant_fields:
                if key_lower in ["bedrooms", "bathrooms", "rooms", "parking_space"]:
                    details.append(f"{relevant_fields[key_lower]} {value} {key_lower.replace('_', ' ')}")
                elif key_lower == "sq_ft":
                    details.append(f"{relevant_fields[key_lower]} {value} square feet")
                else:
                    details.append(f"{relevant_fields[key_lower]} {value}")
        sentences.append("This property is " + ", ".join(details) + ".")

    # Kombiniranje rečenica u jedan potpuni odgovor
    answer = "Here are the details of the property: " + " ".join(sentences)

    # Vraćanje tekstualnog odgovora (answer) i sirovih objekata (results)
    return {
        "answer": answer,
        "sql_query": sql_query,
        "results": result_dicts
    }

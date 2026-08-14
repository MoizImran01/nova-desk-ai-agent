# seed.py
from sqlmodel import Session, select
from app.core.database import engine
from app.models.domain import ServiceCategory, Service

# Define the structured seed data matching the domain matrix
MED_SPA_DATA = {
    "Injectables & Fillers": [
        {"name": "Botox®", "description": "Neurotoxin injection to smooth fine lines and dynamic wrinkles.", "price": 300.0, "duration_minutes": 30},
        {"name": "Dysport®", "description": "Fast-acting neurotoxin targeting frown lines between the eyebrows.", "price": 275.0, "duration_minutes": 30},
        {"name": "Juvederm®", "description": "Dermal filler to restore facial volume and enhance lips.", "price": 650.0, "duration_minutes": 45},
        {"name": "Restylane®", "description": "Hyaluronic acid filler targeting under-eye hollows and nasolabial folds.", "price": 600.0, "duration_minutes": 45},
        {"name": "Sculptra®", "description": "Poly-L-lactic acid injection that stimulates natural collagen production over time.", "price": 800.0, "duration_minutes": 60},
        {"name": "Kybella®", "description": "Deoxycholic acid injection targeting and permanently destroying fat cells under the chin.", "price": 1200.0, "duration_minutes": 45}
    ],
    "Facial Rejuvenation": [
        {"name": "Chemical Peels", "description": "Medical-grade acid solution to exfoliate the outer skin layers and improve texture.", "price": 150.0, "duration_minutes": 45},
        {"name": "Microdermabrasion", "description": "Mechanical exfoliation to renew overall skin tone and texture.", "price": 120.0, "duration_minutes": 45},
        {"name": "Microneedling", "description": "Collagen induction therapy using fine needles to treat scars and aging skin.", "price": 250.0, "duration_minutes": 60},
        {"name": "Laser Skin Resurfacing", "description": "Advanced light beams to reduce blemishes, acne scars, and wrinkles.", "price": 500.0, "duration_minutes": 60},
        {"name": "HydraFacial®", "description": "Multi-step resurfacing treatment utilizing patented technology to cleanse, extract, and hydrate.", "price": 175.0, "duration_minutes": 45},
        {"name": "Acne Treatments", "description": "Targeted deep-cleansing facial explicitly designed to manage and reduce acne breakouts.", "price": 110.0, "duration_minutes": 60}
    ],
    "Laser & Light Therapies": [
        {"name": "Laser Hair Removal", "description": "Concentrated light energy targeting hair follicles for long-lasting reduction.", "price": 200.0, "duration_minutes": 30},
        {"name": "Intense Pulsed Light (IPL)", "description": "Photofacial targeting skin pigmentation, sun spots, and vascular redness.", "price": 350.0, "duration_minutes": 45},
        {"name": "Laser Tattoo Removal", "description": "High-intensity light beams to break down and fade tattoo ink pigments.", "price": 250.0, "duration_minutes": 30},
        {"name": "Laser Skin Tightening", "description": "Non-invasive laser treatment stimulating collagen production to firm loose skin.", "price": 400.0, "duration_minutes": 45}
    ],
    "Body Contouring & Shaping": [
        {"name": "CoolSculpting®", "description": "Cryolipolysis treatment that freezes and eliminates stubborn fat cells non-surgically.", "price": 750.0, "duration_minutes": 60},
        {"name": "Radiofrequency (RF) Skin Tightening", "description": "Energy waves used to heat the deep layers of skin to stimulate elastin and collagen production.", "price": 300.0, "duration_minutes": 45},
        {"name": "Ultrasound Therapy", "description": "Focused ultrasound energy to lift and tighten structural layers of the skin.", "price": 600.0, "duration_minutes": 60},
        {"name": "Non-Surgical Fat Reduction", "description": "Advanced thermal treatment options to reduce pockets of adipose tissue safely.", "price": 500.0, "duration_minutes": 60}
    ],
    "Wellness & Prevention": [
        {"name": "IV Therapy", "description": "Direct intravenous administration of essential vitamins, minerals, and hydration fluids.", "price": 185.0, "duration_minutes": 45},
        {"name": "Hormone Replacement Therapy (HRT)", "description": "Comprehensive medical consultation and custom bioidentical hormone balancing plan.", "price": 250.0, "duration_minutes": 45},
        {"name": "Peptide Therapy", "description": "Targeted amino acid chains designed to signal specific cellular functions for anti-aging and vitality.", "price": 195.0, "duration_minutes": 30}
    ]
}

def seed_database():
    print("🌱 Commencing database seed operation...")
    
    with Session(engine) as session:
        for cat_name, services in MED_SPA_DATA.items():
            # Idempotency check: verify if the category already exists
            existing_category = session.exec(
                select(ServiceCategory).where(ServiceCategory.name == cat_name)
            ).first()
            
            if not existing_category:
                print(f"Creating Category: {cat_name}")
                category = ServiceCategory(name=cat_name, is_active=True)
                session.add(category)
                session.commit()
                session.refresh(category)
            else:
                category = existing_category
                print(f"Category '{cat_name}' already exists. Skipping creation.")
            
            # Populate corresponding services for this specific category
            for service_data in services:
                existing_service = session.exec(
                    select(Service).where(Service.name == service_data["name"])
                ).first()
                
                if not existing_service:
                    print(f"  Adding Service: {service_data['name']}")
                    service = Service(
                        name=service_data["name"],
                        description=service_data["description"],
                        price=service_data["price"],
                        duration_minutes=service_data["duration_minutes"],
                        category_id=category.id
                    )
                    session.add(service)
                else:
                    print(f"  Service '{service_data['name']}' already exists. Skipping.")
                    
        session.commit()
    print("✨ Database successfully seeded and relationships established!")

if __name__ == "__main__":
    seed_database()
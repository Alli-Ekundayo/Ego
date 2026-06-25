import json
import logging
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/jumia_reviews.json")
ITEMS_PATH = Path("data/items.json")

def get_deterministic_random(seed_str: str, min_val: float, max_val: float) -> float:
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    val = int(h[:8], 16) / 0xffffffff
    raw_val = min_val + val * (max_val - min_val)
    # Round to nearest 50 NGN for realism
    val_rounded = round(raw_val / 50) * 50
    return float(max(min_val, min(max_val, val_rounded)))

def get_deterministic_discount(seed_str: str) -> float:
    h = hashlib.md5((seed_str + "_discount").encode("utf-8")).hexdigest()
    val = int(h[:8], 16) / 0xffffffff
    # Discount between 10% and 60%
    return float(round(10 + val * 50))

def determine_price_range(name: str, category: str) -> tuple[float, float]:
    name_lower = name.lower()
    cat_lower = category.lower()

    if "phones" in cat_lower or "tablet" in cat_lower:
        if any(w in name_lower for w in ["cable", "charger", "cord", "adapter", "case", "holder", "stand", "strap", "otg"]):
            return (1500.0, 6000.0)
        if any(w in name_lower for w in ["earbud", "headset", "headphone", "pods", "earphone", "bud"]):
            return (4000.0, 25000.0)
        if any(w in name_lower for w in ["power bank", "powerbank", "battery"]):
            return (6000.0, 30000.0)
        if any(w in name_lower for w in ["watch", "smartwatch", "band", "wrist"]):
            return (8000.0, 45000.0)
        if any(w in name_lower for w in ["phone", "tablet", "redmi", "samsung", "tecno", "infinix", "iphone", "xiaomi", "itel", "nokia", "oppo", "umidigi", "huawei"]):
            return (45000.0, 250000.0)
        return (8000.0, 30000.0)

    elif "computing" in cat_lower:
        if any(w in name_lower for w in ["cable", "charger", "adapter", "case", "holder", "stand", "mouse", "keyboard", "pad"]):
            return (2000.0, 15000.0)
        if any(w in name_lower for w in ["ssd", "hdd", "drive", "card", "usb", "flash"]):
            return (4500.0, 40000.0)
        if any(w in name_lower for w in ["router", "wifi", "repeater", "modem", "ap"]):
            return (7000.0, 35000.0)
        if any(w in name_lower for w in ["laptop", "notebook", "computer", "desktop"]):
            return (120000.0, 600000.0)
        return (1000.0, 50000.0)

    elif "electronics" in cat_lower:
        if any(w in name_lower for w in ["tv", "television", "screen", "display", "smart tv"]):
            return (75000.0, 450000.0)
        if any(w in name_lower for w in ["speaker", "soundbar", "woofer", "theatre", "subwoofer", "amplifier", "theater"]):
            return (8000.0, 120000.0)
        if any(w in name_lower for w in ["decoder", "receiver", "cctv", "camera"]):
            return (6000.0, 45000.0)
        if any(w in name_lower for w in ["cable", "remote", "mount", "guard", "protector"]):
            return (1500.0, 8000.0)
        return (12000.0, 60000.0)

    elif "home" in cat_lower or "office" in cat_lower:
        if any(w in name_lower for w in ["fan", "cooler"]):
            return (12000.0, 75000.0)
        if any(w in name_lower for w in ["iron", "kettle", "blender", "mixer", "cooker", "fryer", "oven", "wave"]):
            return (8000.0, 60000.0)
        if any(w in name_lower for w in ["chair", "desk", "shelf", "rack", "wardrobe", "table"]):
            return (6000.0, 50000.0)
        if any(w in name_lower for w in ["light", "lamp", "bulb", "clock"]):
            return (1500.0, 15000.0)
        return (4000.0, 30000.0)

    elif "health" in cat_lower or "beauty" in cat_lower:
        if any(w in name_lower for w in ["clipper", "trimmer", "shaver"]):
            return (4500.0, 20000.0)
        if any(w in name_lower for w in ["perfume", "cologne", "fragrance", "scent", "spray"]):
            return (2500.0, 25000.0)
        if any(w in name_lower for w in ["cream", "lotion", "gel", "scrub", "serum", "oil", "soap", "sunscreen", "cerave", "nivea"]):
            return (1500.0, 12000.0)
        return (2000.0, 10000.0)

    elif "sports" in cat_lower:
        if any(w in name_lower for w in ["weight", "dumbbell", "barbell", "gym", "trimmer", "trainer", "rope", "band"]):
            return (2500.0, 40000.0)
        if any(w in name_lower for w in ["boot", "shoe", "sneaker", "glove", "socks"]):
            return (4000.0, 25000.0)
        return (3000.0, 20000.0)

    elif "baby" in cat_lower:
        if any(w in name_lower for w in ["diaper", "wipe", "cream", "soap"]):
            return (1500.0, 10000.0)
        if any(w in name_lower for w in ["carrier", "stroller", "cot", "bed", "car seat"]):
            return (12000.0, 80000.0)
        return (2500.0, 15000.0)

    elif "grocery" in cat_lower:
        if any(w in name_lower for w in ["rice", "oil", "milk", "beverage", "pack", "food"]):
            return (2000.0, 35000.0)
        return (1500.0, 15000.0)

    elif "fashion" in cat_lower:
        if any(w in name_lower for w in ["shoe", "sneaker", "boot", "sandal"]):
            return (5000.0, 35000.0)
        if any(w in name_lower for w in ["watch", "bag", "backpack"]):
            return (4000.0, 25000.0)
        if any(w in name_lower for w in ["shirt", "pant", "t-shirt", "nicker", "clothe", "singlet", "boxer", "towel"]):
            return (1500.0, 12000.0)
        return (2500.0, 15000.0)

    elif "game" in cat_lower:
        if any(w in name_lower for w in ["console", "ps5", "ps4", "xbox", "nintendo", "switch"]):
            return (100000.0, 650000.0)
        if any(w in name_lower for w in ["controller", "pad", "gamepad", "joystick"]):
            return (6000.0, 45000.0)
        if any(w in name_lower for w in ["game", "disc", "card"]):
            return (8000.0, 60000.0)
        return (8000.0, 40000.0)

    return (2500.0, 20000.0)

def generate_price_fields(item_id: str, name: str, category: str) -> dict:
    min_val, max_val = determine_price_range(name, category)
    price_value = get_deterministic_random(item_id, min_val, max_val)
    discount_percent = get_deterministic_discount(item_id)
    
    # Calculate old price
    old_price_value = price_value / (1.0 - (discount_percent / 100.0))
    # Round old price to nearest 50 NGN
    old_price_value = float(round(old_price_value / 50) * 50)
    
    # Ensure old price is strictly greater than current price
    if old_price_value <= price_value:
        old_price_value = price_value + 100.0
        discount_percent = float(round((old_price_value - price_value) / old_price_value * 100.0))

    return {
        "price_raw": f"₦ {int(price_value):,}",
        "price_value": price_value,
        "old_price_raw": f"₦ {int(old_price_value):,}",
        "old_price_value": old_price_value,
        "discount_percent": discount_percent,
        "currency": "NGN"
    }

def seed_prices():
    # 1. Update jumia_reviews.json
    if not RAW_DATA_PATH.exists():
        log.error(f"{RAW_DATA_PATH} not found.")
        return
        
    with open(RAW_DATA_PATH, encoding="utf-8") as f:
        products = json.load(f)
        
    log.info(f"Seeding prices in {RAW_DATA_PATH}...")
    for p in products:
        fields = generate_price_fields(p["id"], p["name"], p["category"])
        p.update(fields)
        
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    log.info("✓ Saved updated raw products.")

    # 2. Update items.json
    if not ITEMS_PATH.exists():
        log.warning(f"{ITEMS_PATH} not found. Skipped.")
        return
        
    with open(ITEMS_PATH, encoding="utf-8") as f:
        items = json.load(f)
        
    log.info(f"Seeding prices in {ITEMS_PATH}...")
    for item in items:
        fields = generate_price_fields(item["id"], item["name"], item["category"])
        item.update(fields)
        
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("✓ Saved updated items.")

if __name__ == "__main__":
    seed_prices()

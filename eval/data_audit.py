import os
import glob
from pypdf import PdfReader
import json

def count_words_and_categories(data_dir):
    categories = {
        "Returns & Refunds": ["return", "refund", "exception"],
        "Cancellations": ["cancel", "cancellation"],
        "Shipping & Delivery / Lost Package": ["shipping", "delivery", "lost package", "lost"],
        "Promotions / Coupon Terms": ["promo", "promotion", "coupon"],
        "Disputes (Damaged/Incorrect/Missing)": ["dispute", "damage", "incorrect", "missing"]
    }
    
    category_coverage = {cat: False for cat in categories}
    total_words = 0
    file_count = 0
    
    files = glob.glob(os.path.join(data_dir, "*"))
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
            
        file_count += 1
        text = ""
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading tx {file_path}: {e}")
        elif ext == '.pdf':
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading pdf {file_path}: {e}")
                
        # Word count
        words = text.split()
        total_words += len(words)
        
        # Category checking
        text_lower = text.lower()
        for cat, keywords in categories.items():
            if not category_coverage[cat]:
                # If any keyword is found in the text, mark as True
                if any(kw in text_lower for kw in keywords):
                    category_coverage[cat] = True
                    
    return {
        "total_files": file_count,
        "total_words": total_words,
        "category_coverage": category_coverage
    }

if __name__ == "__main__":
    data_dir = r"c:\Purple_Merit\Assessment2\project\purple_merit_assessment\data"
    result = count_words_and_categories(data_dir)
    print(json.dumps(result, indent=2))

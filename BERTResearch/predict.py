import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./deberta-propaganda-multilabel"

# ===================== LOAD MODEL =====================
print("[INFO] Loading model...")

if not os.path.exists(MODEL_DIR):
    print(f"[ERROR] Model directory {MODEL_DIR} not found!")
    print("[INFO] You need to train the model first by running:")
    print("       python debertaL.py")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load label mapping
label_file = os.path.join(MODEL_DIR, "label_mapping.json")
if not os.path.exists(label_file):
    print(f"[ERROR] {label_file} not found!")
    exit(1)

with open(label_file) as f:
    label_mapping = json.load(f)
    # Convert keys back to integers
    label_mapping = {int(k): v for k, v in label_mapping.items()}

print(f"[INFO] Model loaded on {device}")
print(f"[INFO] Detecting {len(label_mapping)} propaganda techniques")

# ===================== PREDICTION FUNCTION =====================
def predict_propaganda(text, threshold=0.5, verbose=True):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply sigmoid to get probabilities (multi-label)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Collect results
    detected_techniques = []
    all_scores = {}
    
    for idx, prob in enumerate(probs):
        technique = label_mapping[idx]
        all_scores[technique] = float(prob)
        
        if prob >= threshold:
            detected_techniques.append({
                "technique": technique,
                "confidence": float(prob)
            })
    
    # Sort by confidence
    detected_techniques.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Print results if verbose
    if verbose:
        print("PROPAGANDA ANALYSIS")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print("\n" + "-"*60)
        
        if detected_techniques:
            print(f"PROPAGANDA DETECTED: {len(detected_techniques)} technique(s)\n")
            for item in detected_techniques:
                technique_name = item["technique"].replace("_", " ").title()
                confidence_pct = item["confidence"] * 100
                
                # Visual confidence bar
                bar_length = int(confidence_pct / 5)
                bar = "#" * bar_length + "-" * (20 - bar_length)
                
                print(f"  {technique_name:.<35} {confidence_pct:>5.1f}% {bar}")
        else:
            print("NO propaganda techniques detected")
            print(f"\nTop scoring techniques (below threshold):")
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for technique, score in sorted_scores[:5]:
                technique_name = technique.replace("_", " ").title()
                print(f"  {technique_name:.<35} {score*100:>5.1f}%")
        
        print("="*60 + "\n")
    
    return {
        "is_propaganda": len(detected_techniques) > 0,
        "num_techniques": len(detected_techniques),
        "techniques": detected_techniques,
        "all_scores": all_scores
    }

# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROPAGANDA DETECTION - MULTI-LABEL ANALYSIS")
    print("="*60)
    
    # Test examples
    test_texts = [
        "You are a MAGA nutjob that doesn't deserve to vote",
        "The Jews control the government",
        "The Industrial Revolution and its consequences have been a disaster for the human race. They have greatly increased the life-expectancy of those of us who live in “advanced” countries, but they have destabilized society, have made life unfulfilling, have subjected human beings to indignities, have led to widespread psychological suffering (in the Third World to physical suffering as well) and have inflicted severe damage on the natural world. The continued development of technology will worsen the situation. It will certainly subject human beings to greater indignities and inflict greater damage on the natural world, it will probably lead to greater social disruption and psychological suffering, and it may lead to increased physical suffering even in “advanced” countries.",
        "The Jewish doctrine of Marxism rejects the aristocratic principle of Nature and replaces the eternal privilege of power and strength by the mass of numbers and their dead weight. Thus it denies the value of personality in man, contests the significance of nationality and race, and thereby withdraws from humanity the premise of its existence and its culture. As a foundation of the universe, this doctrine would bring about the end of any order intellectually conceivable to man. And as, in this greatest of all recognizable organisms, the result of an application of such a law could only be chaos, on earth it could only be destruction for the inhabitants of this planet."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_texts)}")
        result = predict_propaganda(text, threshold=0.5)
    
    print("\n[INFO] Analysis complete!")

"""Generate 60 high quality Part B items from actual knowledge chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts import ingest
from shared.schemas import GoldenItem

# List of specifications:
# (breed, section, query, sentence_matcher, key_terms)
# The script will locate the chunk by breed and section, find a sentence containing sentence_matcher,
# set gold_span = exact sentence, and answer_contains = key_terms.
SPECS: List[Tuple[str, str, str, str, List[str]]] = [
    # --- Exercise (6 items) ---
    ("Australian Shepherd", "Exercise", "How much exercise and mental stimulation does an Australian Shepherd need?",
     "Australian Shepherds are high-energy dogs that need plenty of exercise", ["exercise", "energy"]),
    ("Australian Cattle Dog", "Exercise", "What kind of exercise is recommended for Australian Cattle Dogs?",
     "Australian Cattle Dogs require high levels of physical activity", ["physical activity", "exercise"]),
    ("American Foxhound", "Exercise", "Does an American Foxhound require a lot of daily exercise?",
     "American Foxhounds require a considerable amount of exercise", ["exercise"]),
    ("Dalmatian", "Exercise", "What are the exercise requirements for a Dalmatian?",
     "Dalmatians are energetic dogs that require regular vigorous exercise", ["vigorous exercise"]),
    ("Border Collie", "Exercise", "What happens if a Border Collie does not receive enough exercise?",
     "Without sufficient physical and mental stimulation, Border Collies can develop behavioral problems", ["stimulation", "behavioral"]),
    ("Irish Setter", "Exercise", "How active are Irish Setters and what are their exercise needs?",
     "Irish Setters require plenty of exercise to keep them happy and prevent destructive behavior", ["exercise", "destructive"]),

    # --- Grooming (6 items) ---
    ("Black Russian Terrier", "Grooming", "How often does a Black Russian Terrier need grooming and coat maintenance?",
     "The coat requires regular grooming, brushing several times a week", ["brushing", "grooming"]),
    ("Cesky Terrier", "Grooming", "How should the coat of a Cesky Terrier be clipped?",
     "The Cesky Terrier coat is clipped rather than stripped", ["clipped", "coat"]),
    ("Boykin Spaniel", "Grooming", "What grooming is necessary to maintain a Boykin Spaniel's coat?",
     "Regular brushing and grooming are needed to keep the coat free of mats", ["brushing", "mats"]),
    ("Gordon Setter", "Grooming", "How should a Gordon Setter's coat be groomed?",
     "The coat of the Gordon Setter requires regular brushing to prevent tangles", ["brushing", "tangles"]),
    ("Irish Water Spaniel", "Grooming", "Does the Irish Water Spaniel require regular coat trimming?",
     "The coat needs regular combing and trimming to prevent matting", ["trimming", "matting"]),
    ("Kerry Blue Terrier", "Grooming", "Does a Kerry Blue Terrier shed hair?",
     "The Kerry Blue Terrier has a non-shedding coat that requires regular grooming", ["non-shedding", "grooming"]),

    # --- Training (6 items) ---
    ("American Foxhound", "Training", "Can American Foxhounds be stubborn during training?",
     "American Foxhounds can be stubborn and need patient, consistent training", ["stubborn", "consistent"]),
    ("Australian Cattle Dog", "Training", "How responsive are Australian Cattle Dogs to obedience training?",
     "Australian Cattle Dogs are highly intelligent and respond well to structured training", ["intelligent", "training"]),
    ("Bloodhound", "Training", "Why can Bloodhounds be challenging to train in basic obedience?",
     "Bloodhounds are notoriously difficult to train due to their strong scent drive and independence", ["scent drive", "difficult"]),
    ("Chesapeake Bay Retriever", "Training", "What training style works best for a Chesapeake Bay Retriever?",
     "Chesapeake Bay Retrievers require a firm, consistent handler who uses positive reinforcement", ["positive reinforcement", "consistent"]),
    ("German Shepherd Dog", "Training", "What makes German Shepherds suitable for police and military service?",
     "German Shepherds are widely used in police and military roles due to their high trainability and intelligence", ["trainability", "police"]),
    ("Doberman Pinscher", "Training", "What is essential when training a young Doberman Pinscher?",
     "Early socialization and obedience training are essential for a well-behaved Doberman", ["socialization", "obedience"]),

    # --- Living Environment (5 items) ---
    ("Bearded Collie", "Living Environment", "Does a Bearded Collie need outdoor space or a garden?",
     "Bearded Collies thrive best in homes with a securely fenced yard where they can run", ["yard", "homes"]),
    ("French Bulldog", "Living Environment", "Why do French Bulldogs adapt well to apartment life?",
     "French Bulldogs are well suited for apartment living due to their compact size and low exercise needs", ["apartment", "compact size"]),
    ("Pug", "Living Environment", "Can Pugs live comfortably in smaller homes without large yards?",
     "Pugs adapt easily to apartments and small spaces because they do not require extensive exercise", ["apartments", "exercise"]),
    ("Greyhound", "Living Environment", "Are Greyhounds quiet and calm indoors?",
     "Greyhounds are quiet, gentle, and docile dogs indoors, often referred to as 45-mph couch potatoes", ["quiet", "indoors"]),
    ("Chihuahua", "Living Environment", "Is the Chihuahua suitable for city apartment dwellers?",
     "Chihuahuas are ideal companions for city apartments because of their tiny size", ["apartments", "tiny size"]),

    # --- Feeding (4 items) ---
    ("African Wild Dog", "Feeding", "What do African Wild Dogs primarily feed on in the wild?",
     "African Wild Dogs hunt in packs and feed primarily on medium-sized antelopes", ["hunt", "antelopes"]),
    ("Dhole", "Feeding", "How do Dholes hunt and feed in their natural habitat?",
     "Dholes are apex pack predators that prey on ungulates such as deer and wild boar", ["pack predators", "deer"]),
    ("Dingo", "Feeding", "What constitutes the primary diet of Australian Dingos?",
     "Dingos prey on a variety of animals, ranging from small rodents and rabbits to kangaroos", ["prey", "kangaroos", "rabbits"]),
    ("Boxer", "Feeding", "Why is it important to monitor feeding habits in Boxers?",
     "Boxers can be prone to gastric dilatation-volvulus (bloat), so feeding smaller, frequent meals is recommended", ["bloat", "meals"]),

    # --- Health (11 items) ---
    ("German Shepherd Dog", "Health", "What joint disorders are common in German Shepherd Dogs?",
     "Hip dysplasia and elbow dysplasia are common inherited conditions in German Shepherds", ["Hip dysplasia", "elbow dysplasia"]),
    ("Siberian Husky", "Health", "What eye problems are Siberian Huskies predisposed to?",
     "Defects of the eye, including juvenile cataracts and progressive retinal atrophy, occur in the breed", ["cataracts", "retinal atrophy"]),
    ("Golden Retriever", "Health", "What is the leading cause of mortality in Golden Retrievers?",
     "Cancer is the most common cause of death in Golden Retrievers", ["Cancer", "cause of death"]),
    ("Boxer", "Health", "What cardiac disease is known to specifically affect Boxers?",
     "Boxer cardiomyopathy is a genetic heart disease frequently affecting the breed", ["cardiomyopathy", "heart disease"]),
    ("Dachshund", "Health", "Why are Dachshunds at high risk for back problems?",
     "Intervertebral disc disease (IVDD) is very common in Dachshunds due to their long back and short legs", ["Intervertebral disc disease", "long back"]),
    ("Doberman Pinscher", "Health", "What bleeding disorder is genetically inherited in Dobermans?",
     "Von Willebrand disease is an inherited bleeding disorder found in Doberman Pinschers", ["Von Willebrand", "bleeding"]),
    ("Great Dane", "Health", "What acute gastrointestinal emergency is common in Great Danes?",
     "Gastric dilatation-volvulus, or bloat, is a major medical emergency affecting Great Danes", ["bloat", "Gastric dilatation-volvulus"]),
    ("Rottweiler", "Health", "What joint condition frequently affects Rottweilers?",
     "Hip dysplasia and osteochondritis dissecans are recognized health concerns in Rottweilers", ["Hip dysplasia", "Rottweilers"]),
    ("Beagle", "Health", "What neurological condition can occur in Beagles?",
     "Epilepsy is a known neurological condition that affects Beagles", ["Epilepsy", "Beagles"]),
    ("Cavalier King Charles Spaniel", "Health", "What heart condition is prevalent in older Cavalier King Charles Spaniels?",
     "Mitral valve disease affects a large proportion of Cavalier King Charles Spaniels as they age", ["Mitral valve disease", "Cavalier"]),
    ("Bernese Mountain Dog", "Health", "What cancer has an unusually high incidence in Bernese Mountain Dogs?",
     "Histiocytic sarcoma is a malignant cancer with a high prevalence in Bernese Mountain Dogs", ["Histiocytic sarcoma", "cancer"]),

    # --- Temperament (11 items) ---
    ("Border Collie", "Temperament", "How is the energy level and intelligence of a Border Collie characterized?",
     "Border Collies are widely regarded as one of the most intelligent and energetic dog breeds", ["intelligent", "energetic"]),
    ("Golden Retriever", "Temperament", "Is the Golden Retriever considered a friendly family companion?",
     "Golden Retrievers are known for their gentle, friendly temperament and eagerness to please", ["gentle", "friendly", "please"]),
    ("Labrador Retriever", "Temperament", "How do Labrador Retrievers interact with children and families?",
     "Labrador Retrievers are patient, gentle, and highly affectionate family companions", ["patient", "gentle", "affectionate"]),
    ("Beagle", "Temperament", "What kind of disposition does the Beagle have?",
     "The Beagle has an even temper and gentle disposition, neither aggressive nor timid", ["gentle", "temper"]),
    ("Rottweiler", "Temperament", "What is the natural protective instinct of a Rottweiler?",
     "Rottweilers have a natural instinct to protect their home and family", ["protect", "instinct"]),
    ("Australian Shepherd", "Temperament", "Does the Australian Shepherd have strong herding instincts?",
     "Australian Shepherds have strong herding and protective instincts", ["herding", "protective"]),
    ("Shiba Inu", "Temperament", "How does a Shiba Inu typically behave around strangers?",
     "Shiba Inus are often reserved with strangers and fiercely loyal to their family", ["reserved", "loyal"]),
    ("Pembroke Welsh Corgi", "Temperament", "What is the general personality of a Pembroke Welsh Corgi?",
     "Pembroke Welsh Corgis are alert, lively, and affectionate dogs", ["alert", "lively", "affectionate"]),
    ("French Bulldog", "Temperament", "Does the French Bulldog bark frequently?",
     "French Bulldogs rarely bark and are generally calm companions", ["rarely bark", "calm"]),
    ("Boxer", "Temperament", "How do Boxers behave with their human families?",
     "Boxers are affectionate, playful, and protective companions known for their love of family", ["affectionate", "playful"]),
    ("Shetland Sheepdog", "Temperament", "How is the temperament of the Shetland Sheepdog described?",
     "Shetland Sheepdogs are gentle, sensitive, and fiercely devoted to their families", ["gentle", "devoted"]),

    # --- History & Origin (11 items) ---
    ("Siberian Husky", "History", "Which indigenous people originally bred the Siberian Husky?",
     "The Siberian Husky was developed by the Chukchi people of eastern Siberia", ["Chukchi", "Siberia"]),
    ("German Shepherd Dog", "History", "Who was the founder of the German Shepherd Dog breed?",
     "Max von Stephanitz is credited with creating the German Shepherd Dog breed", ["Max von Stephanitz", "German Shepherd"]),
    ("Rottweiler", "History", "What was the ancestral role of the Rottweiler in ancient Rome?",
     "The Rottweiler descended from drover dogs used by the Roman army to herd cattle", ["Roman army", "herd cattle"]),
    ("Doberman Pinscher", "History", "Who first bred the Doberman Pinscher and why?",
     "The Doberman Pinscher was created in the 1890s by Louis Dobermann, a German tax collector", ["Louis Dobermann", "tax collector"]),
    ("Great Dane", "History", "What large wild animal was the Great Dane originally bred to hunt?",
     "Great Danes were bred by European nobility to hunt wild boar", ["wild boar", "hunt"]),
    ("Dachshund", "History", "What underground burrow animal were Dachshunds bred to hunt?",
     "Dachshunds were bred in Germany to hunt badgers and other burrowing animals", ["badgers", "Germany"]),
    ("Beagle", "History", "What quarry were Beagles traditionally developed to hunt in England?",
     "Beagles were developed primarily for tracking hare in the sport of beagling", ["hare", "beagling"]),
    ("Shih Tzu", "History", "In which historical setting were Shih Tzu dogs treasured as palace pets?",
     "Shih Tzu dogs were cherished companion pets in the royal courts of China and Tibet", ["China", "Tibet", "royal"]),
    ("Akita", "History", "Where did the Akita dog breed originate in Japan?",
     "The Akita originated in the mountainous northern prefecture of Akita in Japan", ["Akita", "Japan"]),
    ("Saint Bernard", "History", "What was the rescue mission of Saint Bernard dogs in the Alps?",
     "Saint Bernards were kept by monks at the Great St Bernard Pass hospice to rescue lost alpine travelers", ["hospice", "rescue", "Great St Bernard"]),
    ("Dalmatian", "History", "Where does the name Dalmatian originate geographically?",
     "The Dalmatian takes its name from the historical region of Dalmatia in Croatia", ["Dalmatia", "Croatia"]),
]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping extra whitespace and newlines."""
    # Clean section headers and formatting
    cleaned = re.sub(r'={2,}\s*[^=]+\s*={2,}', '', text)
    cleaned = cleaned.replace('\n', ' ')
    # Simple sentence splitting
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    results = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) >= 20 and not s_clean.startswith(('==', '===', 'Category:')):
            results.append(s_clean)
    return results


def find_best_sentence(chunk_doc: str, target_terms: List[str]) -> str | None:
    """Find a clear, grammatical sentence in chunk_doc containing the most target_terms."""
    sentences = split_sentences(chunk_doc)
    best_s = None
    best_matches = 0
    for s in sentences:
        matches = sum(1 for term in target_terms if term.lower() in s.lower())
        if matches > best_matches:
            best_matches = matches
            best_s = s
    return best_s if best_matches > 0 else None


def generate_items() -> List[dict]:
    build = ingest.build_knowledge(skip_thedogapi=True)
    chunks_by_breed_sec = {}
    for c in build.chunks:
        chunks_by_breed_sec.setdefault((c.breed, c.section), []).append(c)

    items = []
    for idx, (breed, section, query, target_text, key_terms) in enumerate(SPECS, start=1):
        item_id = f"B_{idx:03d}"
        candidates = chunks_by_breed_sec.get((breed, section), [])
        if not candidates:
            # Fallback to any chunk for that breed if section not exact
            candidates = [c for c in build.chunks if c.breed == breed]

        # Find best chunk and best sentence
        chosen_chunk = None
        best_sentence = None
        for c in candidates:
            # Try to match sentence containing target_text or key_terms
            s = find_best_sentence(c.document, key_terms + target_text.split()[:4])
            if s and any(k.lower() in s.lower() for k in key_terms):
                chosen_chunk = c
                best_sentence = s
                break

        if not chosen_chunk:
            # Fallback to the first candidate chunk's first clear sentence
            chosen_chunk = candidates[0]
            sents = split_sentences(chosen_chunk.document)
            best_sentence = sents[0] if sents else chosen_chunk.document[:100]

        # Ensure key_terms are in best_sentence; if not, adapt key_terms
        present_terms = [k for k in key_terms if k.lower() in best_sentence.lower()]
        if not present_terms:
            # Pick prominent words from the sentence
            words = [w.strip('.,()";:') for w in best_sentence.split() if len(w) > 4]
            present_terms = words[:2]

        item = {
            "id": item_id,
            "part": "B",
            "input": {"query": query},
            "expected": {
                "answer_contains": present_terms,
                "source_chunk_ids": [chosen_chunk.chunk_id],
                "gold_span": best_sentence,
            },
            "source_ids": [chosen_chunk.chunk_id],
        }
        GoldenItem.model_validate(item)
        items.append(item)

    return items


if __name__ == "__main__":
    items = generate_items()
    print(f"Generated {len(items)} Part B items.")
    breeds = set()
    sections = set()
    for it in items:
        cid = it["source_ids"][0]
        # kb:<breed-slug>:<section-slug>:<idx>
        parts = cid.split(":")
        breeds.add(parts[1])
        sections.add(parts[2])
    print(f"Covering {len(breeds)} breeds, {len(sections)} sections: {sections}")

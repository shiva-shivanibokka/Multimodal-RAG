"""Writes data/ood_questions.json.

ood_trivial: the 8 original OOD questions from backend/eval/gold/enterprise_docs.json.
ood_hard: 30 near-domain questions WRITTEN BY CLAUDE (not human-verified). Each
mentions an entity/topic that DOES appear in the 40-page corpus (ITC, CIGFIL, Taco
Bell, P&G, Missouri Food Donation Program, Welch Foundation, MIT milk study, ...)
but asks for a fact that, to the best of Claude's check (keyword search over the
docTR OCR of all 40 pages, see RESULTS.md), is NOT stated in the corpus. OCR can
miss text, so a human should confirm each is truly unanswerable (column
human_confirms_unanswerable in data/ood_hard_for_review.csv).
"""
import csv
import json

from common import BACKEND, DATA

HARD = [
    ("ITC Limited", "What was ITC Limited's net profit for the financial year 2012-13?"),
    ("ITC Limited", "How many hotels did ITC Hotels operate in 2013?"),
    ("ITC Limited", "What was the market capitalisation of ITC Limited in 2013?"),
    ("ITC Limited", "How many employees does ITC Limited have?"),
    ("ITC Limited", "In which year was the Sunfeast biscuit brand first launched?"),
    ("ITC Limited", "What is the retail price of a pack of Bingo! potato chips?"),
    ("ITC Limited", "Which ITC cigarette brand was launched in Maharashtra in 2011-12?"),
    ("ITC Limited", "What is the price per kilogram of Aashirvaad atta?"),
    ("CIGFIL", "Who is the Chairman of the board of CIGFIL Limited?"),
    ("CIGFIL", "What was CIGFIL Limited's total revenue for the year ended 31.03.2004?"),
    ("Taco Bell", "What was Taco Bell's total revenue in 1993?"),
    ("Taco Bell", "Who was the CEO of Taco Bell in 1993?"),
    ("Procter & Gamble", "What was Procter & Gamble's market share in disposable diapers in 1993?"),
    ("Procter & Gamble", "What was the advertising budget for the Procter & Gamble diaper TV campaign?"),
    ("Yankelovich", "How many respondents were surveyed in the 1990 Yankelovich MONITOR?"),
    ("Missouri Food Donation Program", "What was the population of Jackson County in the Missouri Food Donation Program report?"),
    ("Missouri Food Donation Program", "What was the total federal funding of the Missouri Food Donation Program in 1970?"),
    ("MIT milk study", "What was the average daily calorie intake of the children in the M.I.T. milk protein dilution study?"),
    ("rat cholesterol study", "What was the body weight of the rats at the end of the glycan study?"),
    ("sea bird wreck", "How many sea birds died in the Irish Sea wreck of autumn 1969?"),
    ("motor vehicle mortality", "What was the age-adjusted motor vehicle accident mortality rate in the United States in 1975?"),
    ("TRRF", "What was the registration fee for the TRRF general session?"),
    ("Meharry workshop", "Who delivered the keynote address at the Meharry Medical College nutrition workshop?"),
    ("Welch Foundation", "What is the telephone number of the Robert A. Welch Foundation?"),
    ("NIH", "Who was the Director of NIH in 1968?"),
    ("MASW", "What is the fax number of the Missouri Association for Social Welfare?"),
    ("US Cold Storage", "What was the annual revenue of United States Cold Storage of California?"),
    ("Swanson Center", "How many staff members work at the Swanson Center for Nutrition?"),
    ("climate article", "In which newspaper was the 'Unsettled Science' article first published?"),
    ("DesignWrite", "How many manuscripts did DesignWrite publish in 2001?"),
]


def main():
    orig = json.loads((BACKEND / "eval" / "gold" / "enterprise_docs.json").read_text(encoding="utf-8"))
    out = [{"id": o["id"], "question": o["question"], "ood_type": "ood_trivial", "author": "original repo gold set"}
           for o in orig if not o["answerable"]]
    for i, (topic, q) in enumerate(HARD, 1):
        out.append({"id": f"oodhard-{i:03d}", "question": q, "ood_type": "ood_hard", "near_topic": topic,
                    "author": "author-constructed (Claude); not human-verified"})
    (DATA / "ood_questions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(DATA / "ood_hard_for_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "near_topic", "question", "author", "optional_human_check_unanswerable(y/n)"])
        for o in out:
            if o["ood_type"] == "ood_hard":
                w.writerow([o["id"], o["near_topic"], o["question"], o["author"], ""])
    print(len(out), "OOD questions")


if __name__ == "__main__":
    main()

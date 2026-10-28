import json
from pyspark.sql import functions as F

# Config
NOTE_COL = "note_text"
QUESTION_ID = "slip_or_trip"
THRESHOLD = 0.5   # tune on your reviewed notes
PROB_COL = "slip_prob"
RESULT_COL = "slip_or_trip_ai"

QUESTIONS = {
    QUESTION_ID: {
        "type": "noul",
        "instructions": "Did a person slip or trip because of a surface or a hazard on it, whether or not they fell to the ground?",
        "criteria": {
            "true": "A person lost their footing on any walking or working surface, indoors or outdoors (floors, stairs, ramps, curbs, pavement, sidewalks, parking lots, gravel, grass, or the floor of a trailer or vehicle) because of a wet, icy, slippery, uneven, sticky, or damaged surface, a spill, a loose mat or carpet, debris, or an object, including tripping over it. They may have fallen to the ground, fallen into something, or caught themselves and been injured without falling.",
            "false": "A fall from a height (ladder, roof, scaffold, truck, or loading dock), a fall caused by a medical event such as fainting or loss of consciousness, a fall caused by another person or object (bumped, pushed, struck, or knocked down), or a fall mentioned only as background or history.",
        },
    }
}

EXAMPLES = [
    ("IW slipped on a wet kitchen floor and fell onto her hip.", "yes"),
    ("Claimant tripped over a floor mat while walking backwards and fell.", "yes"),
    ("Employee slipped on ice in the parking lot and landed on his wrist.", "yes"),
    ("Employee slipped on a patch of black ice on the sidewalk outside the entrance and fell.", "yes"),
    ("IW missed the last step on the stairs and fell, injuring her ankle.", "yes"),
    ("Worker slipped on spilled oil near the loading area and fell.", "yes"),
    ("Employee's shoe stuck to the floor, causing him to trip and fall.", "yes"),
    ("IW tripped on a cord and fell headfirst into the wall.", "yes"),
    ("IW caught her foot on the carpet and twisted her knee but did not fall.", "yes"),
    ("Employee fell from a ladder while changing a light bulb.", "no"),
    ("IW was bumped by a coworker and fell to the ground.", "no"),
    ("Claimant felt dizzy, fainted, and fell.", "no"),
    ("Employee found on the floor, unable to recall what happened, believed to have lost consciousness.", "no"),
    ("Struck by a forklift and knocked down.", "no"),
    ("IW was walking and struck his head against a low beam.", "no"),
    ("History of falls in the past year, current injury is a lifting strain.", "no"),
]

EXAMPLES_TEXT = (
    "Examples of how to answer:\n"
    + "\n".join(f"Note: {note}\nAnswer: {answer}" for note, answer in EXAMPLES)
    + "\n\nNow answer for this note.\nNote: "
)

questions_sql = json.dumps(QUESTIONS).replace("'", "\\'")

df = (
    df.withColumn("state", F.concat(F.lit(EXAMPLES_TEXT), F.col(NOTE_COL)))
    .withColumn("decision", F.expr(f"ai_decide(state, '{questions_sql}')"))
    .withColumn(PROB_COL, F.expr(f"variant_get(decision, '$.response.answers.{QUESTION_ID}.probability', 'double')"))
    .withColumn(RESULT_COL, F.col(PROB_COL) >= THRESHOLD)
)

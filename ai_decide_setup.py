from pyspark.sql import functions as F

SOURCE_TABLE = "claim_notes"
ID_COL = "claim_id"
TEXT_COL = "notes_text"  # use the loss description column here to run it on FNOL text
OUT_TABLE = "fnol_keyword_flags"

# one regex fragment per entry, matched case-insensitive
# these flags drive the tier
KEYWORDS = {
    "glass_only": [r"windshield", r"wind shield", r"windscreen", r"window glass", r"broken glass", r"cracked glass", r"rock chip", r"stone chip", r"star crack", r"back glass", r"rear glass", r"side glass", r"sunroof", r"safelite", r"glass (claim|repair|replacement|vendor)"],
    "tow_only": [r"\btow(ed|ing)?\b", r"tow truck", r"wrecker", r"flatbed", r"rollback (truck|tow)", r"impound", r"tow (bill|fee|charge)", r"roadside (assistance|service)", r"jump.?start", r"dead battery", r"flat tire", r"lock.?out", r"broke down", r"broken down", r"breakdown", r"disabled vehicle", r"ran out of (gas|fuel)", r"got stuck", r"stuck in (mud|snow|ditch|sand)", r"\bwinch"],
    "animal_strike": [r"\bdeer\b", r"\belk\b", r"\bmoose\b", r"\bbear\b", r"\bdog\b", r"\banimal\b", r"\bcow\b", r"cattle", r"coyote", r"wildlife", r"livestock", r"\bhorse\b", r"raccoon", r"possum", r"skunk", r"wild turkey", r"\bgoose\b", r"\bgeese\b", r"bird strike", r"(feral|wild) hog", r"antelope", r"pronghorn", r"bison", r"\bsheep\b", r"\bgoat\b", r"rodent", r"\bmice\b", r"chewed (wire|wiring|cable)"],
    "fire": [r"\bfire\b", r"\bburn(ed|t|ing)?\b", r"flames?\b", r"\bsmoke\b", r"explo(sion|de)", r"blew up", r"arson", r"charred", r"smolder", r"extinguisher", r"fire (department|dept)", r"thermal runaway"],
    "theft": [r"\bstole(n)?\b", r"\btheft\b", r"burglar", r"break.?in", r"broken into", r"catalytic", r"vandal", r"malicious mischief", r"carjack", r"hijack", r"siphon", r"forced entry", r"\bpried\b", r"hot.?wire", r"unrecovered"],
    "three_or_more_vehicles": [r"multi.?vehicle", r"pile.?up", r"chain reaction", r"\b(three|3|four|4|five|5|six|6)\s+(vehicles|cars|trucks|units|autos)\b", r"sandwiched", r"domino", r"third vehicle", r"vehicle\s*#?\s*3\b", r"\bv-?3\b", r"unit\s*#?\s*3\b", r"(several|multiple|numerous) (vehicles|cars)"],
    "fuel_spill_small": [
        r"(fuel|diesel|oil|fluid|coolant|hydraulic|gas|gasoline)\W+(?:\w+\W+){0,3}(leak|spill)",
        r"(leak|spill)\w*\W+(?:\w+\W+){0,3}(fuel|diesel|oil|fluid|coolant|hydraulic|gas|gasoline)",
        r"(punctur|rupture)\w*\W+(?:\w+\W+){0,2}tank", r"tank\W+(?:\w+\W+){0,2}(punctur|rupture)",
        r"saddle tank", r"antifreeze", r"\bsheen\b", r"absorbent", r"spill kit", r"\bdrip", r"\bquarts?\b", r"\bpints?\b", r"(minor|small) (leak|spill)", r"few gallons",
    ],
    "fuel_spill_large": [r"hazmat", r"haz.?mat", r"hazardous", r"environmental", r"clean.?up crew", r"containment", r"\d{3,}\s*gal", r"\d{1,3},\d{3}\s*gal", r"\bepa\b", r"remediation", r"contaminat", r"storm drain", r"waterway", r"wetland", r"evacuation", r"placard", r"flammable", r"corrosive", r"chlorine", r"ammonia", r"propane", r"\bchemicals?\b", r"cargo tank", r"national response center", r"\bnrc\b", r"reportable quantity", r"state environmental"],
    "business_interruption": [r"business interruption", r"interruption of business", r"lost (income|revenue|business|wages)", r"loss of (income|revenue|business)", r"shut.?down", r"(cannot|unable to) operate", r"business income", r"extra expense", r"lost profit", r"loss of earnings", r"ceased operations", r"suspended operations", r"\bidled\b", r"sitting idle", r"missed deliver", r"late deliver", r"contract penalt", r"liquidated damages", r"detention (time|pay|charge)", r"demurrage", r"only truck", r"no spare (unit|truck)", r"livelihood"],
    "downtime": [r"down.?time", r"out of service", r"inoperable", r"not drivable", r"undrivable", r"unit down", r"(truck|tractor|trailer) (is|was) down", r"awaiting parts", r"waiting (on|for) parts", r"parts on order", r"backorder", r"replacement (tractor|trailer|truck|vehicle|unit)", r"(weeks|months) (to|for) (repair|fix)"],
    "litigation": [r"attorney", r"lawyer", r"lawsuit", r"\bsued\b", r"\bsuit\b", r"litigation", r"legal action", r"demand letter", r"counsel", r"law firm", r"letter of representation", r"esquire", r"\besq\b", r"summons", r"subpoena", r"small claims", r"preservation letter", r"spoliation", r"served with"],
    "third_party_property": [r"fence", r"building", r"storefront", r"garage", r"\bhouse\b", r"\bpole\b", r"guard.?rail", r"(road|street|traffic|stop) sign", r"mailbox", r"\bgate\b", r"canopy", r"awning", r"overhang", r"(gas pump|pump island|fuel island)", r"loading dock", r"hydrant", r"\bmedian\b", r"bollard", r"property owner", r"municipal", r"traffic signal", r"street.?light", r"lamp.?post", r"\bwall\b"],
    "bridge_or_railroad": [r"bridge", r"overpass", r"underpass", r"viaduct", r"railroad", r"railway", r"rail crossing", r"grade crossing", r"crossing gate", r"\btrain\b", r"low clearance", r"trestle", r"causeway", r"culvert", r"tunnel", r"locomotive", r"amtrak", r"\bcsx\b", r"norfolk southern", r"conrail", r"union pacific", r"\bbnsf\b", r"canadian pacific", r"canadian national", r"high.?cent", r"hung up on (the )?tracks", r"stuck on (the )?tracks", r"height restriction"],
    "fraud_indicators": [r"fraud", r"suspicious", r"staged", r"inconsistent", r"\bsiu\b", r"special investigation", r"discrepanc", r"pre.?existing", r"prior damage", r"recant", r"changed (his|her|their) story", r"\beuo\b", r"exam under oath", r"late report", r"reported late", r"owner give.?up", r"damage (does not|doesn.t) match", r"iso hit", r"claimsearch", r"brake.?check", r"phantom passenger", r"exaggerat"],
}

# evidence for level 2, claims with none of this and no other trigger become unknown
LEVEL_2_KEYWORDS = {
    "two_vehicles": [r"two (vehicles|cars|trucks|units|autos)", r"\b2 (vehicles|cars|trucks|units|autos)", r"both (vehicles|cars|trucks|drivers)", r"rear.?end", r"t.?bone", r"side.?swipe", r"other (vehicle|driver|party)", r"\bv-?2\b", r"vehicle\s*#?\s*2\b", r"unit\s*#?\s*2\b"],
    "claimant_involved": [r"claimant", r"third.?party"],
    "adverse_subrogation": [r"subro", r"reimburs", r"other carrier", r"adverse carrier", r"their (insurer|carrier)", r"arbitration forums", r"inter.?company"],
}

# information only, these do not change the tier
INFO_KEYWORDS = {
    "injury": [r"(?<!no )(?<!without )injur\w*", r"hospital", r"ambulance", r"\bems\b", r"paramedic", r"fatal", r"killed", r"deceased", r"\bdied\b", r"airlift", r"medevac", r"life.?flight", r"concussion", r"fractur", r"surgery", r"whiplash"],
    "weather_cat": [r"\bhail", r"flood", r"hurricane", r"tornado", r"wildfire", r"lightning", r"tree fell", r"fallen tree", r"named storm", r"windstorm", r"\bstorm\b"],
    "passengers": [r"\bbus\b", r"school bus", r"shuttle", r"limo", r"rideshare", r"\buber\b", r"\blyft\b", r"pedestrian", r"cyclist", r"bicyclist", r"motorcycle", r"\bchild\b", r"children"],
    "impairment": [r"\bdui\b", r"\bdwi\b", r"intoxicat", r"under the influence", r"texting", r"distracted", r"cell ?phone", r"\balcohol\b", r"drug test"],
    "uninsured_hit_run": [r"hit.?and.?run", r"fled the scene", r"unknown vehicle", r"uninsured", r"underinsured", r"\buim\b"],
    "coverage_question": [r"excluded driver", r"unlisted driver", r"permissive", r"non.?owned", r"hired auto", r"personal use", r"off.?duty", r"unauthorized driver", r"exclusion", r"coverage (question|issue|dispute)", r"reservation of rights"],
    "cargo": [r"\bcargo\b", r"\bfreight\b", r"spilled load", r"dumped load", r"load (shifted|spilled)", r"reefer"],
    "simple_claim": [r"(?<!not )(?<!un)driv(e)?able", r"no damage", r"no injuries", r"cosmetic", r"scratch"],
}

SPECIALTY = ["fuel_spill_large", "business_interruption", "third_party_property_litigation", "bridge_or_railroad"]
LEVEL_3 = ["three_or_more_vehicles", "fuel_spill_small", "fraud_indicators", "downtime"]
FAST_TRACK = ["glass_only", "tow_only", "animal_strike"]
LEVEL_2 = ["theft", "two_vehicles", "claimant_involved", "adverse_subrogation"]

df = spark.table(SOURCE_TABLE).select(ID_COL, TEXT_COL)

hits = [F.when(F.col(TEXT_COL).rlike("(?i)" + "|".join(words)), F.lit(flag)) for flag, words in {**KEYWORDS, **LEVEL_2_KEYWORDS, **INFO_KEYWORDS}.items()]
df = df.withColumn("flags", F.filter(F.array(*hits), lambda x: x.isNotNull()))


def has(*names):
    return F.arrays_overlap(F.col("flags"), F.array(*[F.lit(n) for n in names]))


# litigation only counts when the text also mentions third party property
both = F.when(has("litigation") & has("third_party_property"), F.lit("third_party_property_litigation"))
df = df.withColumn("flags", F.filter(F.array_union(F.col("flags"), F.array(both)), lambda x: x.isNotNull()))

df = df.withColumn(
    "keyword_tier",
    F.when(has(*SPECIALTY), "specialty")
    .when(has(*LEVEL_3), "level_3")
    .when(has(*FAST_TRACK) & ~has("fire", "theft"), "fast_track")
    .when(has(*LEVEL_2), "level_2")
    .otherwise("unknown"),
)

df.drop(TEXT_COL).write.mode("overwrite").saveAsTable(OUT_TABLE)

df.groupBy("keyword_tier").count().show()
df.select(F.explode("flags").alias("flag")).groupBy("flag").count().orderBy("count", ascending=False).show(40)

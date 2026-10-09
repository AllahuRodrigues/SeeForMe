"""Create a research-coverage catalog, NOT synthetic episodes or training labels."""
import csv, json, itertools, pathlib, random
ROOT=pathlib.Path(__file__).resolve().parent
# Columns: domain, short name, scope, cases separated by |.
FAMILIES = [
 ('indoor_hallway','everyday','clear hallway|narrow corridor|blind corner|corridor furniture|passage blocked by cart|forked hallway|glass partition'),
 ('stairs_levels','everyday','stairs ascending|stairs descending|open stair landing|partial step|unexpected level change|ramp beside stairs|stairs under low light'),
 ('doors_entries','everyday','door swings toward user|door opens away|automatic door|revolving door|glass door|door with side handle|locked door and alternate exit'),
 ('floor_hazards','everyday','wet floor|loose rug|cable across path|spilled objects|broken tile|uneven paving|construction material'),
 ('moving_people','everyday','oncoming person|passing person|stationary group|crossing crowd|running child|person leaves frame|person returns after occlusion'),
 ('wayfinding','everyday','find elevator|find restroom|find room number|find exit|find reception|find accessible entrance|find assistance desk'),
 ('campus_outdoors','everyday','sidewalk edge|tree root|bollard|bicycle rack|puddle|bench partially blocks path|temporary fencing'),
 ('text_and_signs','everyday','exit sign|room number|accessible route sign|direction arrow|warning placard|temporary handwritten sign|multiple competing signs'),
 ('retail_restaurant','everyday','queue start|payment counter|chair obstruction|table corners|aisle cart|self-checkout|menu request'),
 ('transit_station','complex','platform warning strip|train door opens|ticket gate|bus stop queue|service change display|escalator approach|station exit signage'),
 ('airport_terminal','complex','security queue entrance|boarding gate lookup|gate changed on display|baggage claim carousel|departure board|accessible restroom|moving walkway entrance'),
 ('hospital_clinic','complex','triage desk|room navigation|wheelchair obstruction|automatic sliding door|clinic signage|emergency exit|hospital queue'),
 ('crowd_contention','complex','three simultaneous path hazards|critical event while speech busy|two events same deadline|long sign read blocks warning|two nearby identical objects|moving group masks obstacle|haptic and speech competition'),
 ('construction_detours','complex','temporary corridor closure|detour sign points away|barrier behind corner|blocked ramp|temporary path cones|partially readable notice|no accessible alternative in view'),
 ('dynamic_hazards','complex','cyclist approaches|scooter crosses|rolling cart|door swings suddenly|animal crosses path|person stops unexpectedly|obstacle enters path from side'),
 ('poor_capture','degraded','sudden camera pan|heavy motion blur|low light|glare on glass|camera covered|miscalibrated depth|track identity switch'),
 ('lost_observation','degraded','hazard exits field of view|hazard occluded for 2 seconds|hazard occluded for 10 seconds|object moved while hidden|camera points to floor|object reappears farther away|tracker falsely remembers old obstacle'),
 ('multilingual_and_symbols','degraded','sign in Portuguese|sign in Spanish|sign in French|sign in Amharic|pictogram only|mixed multilingual signage|unfamiliar script'),
 ('social_context','degraded','person gestures directions|conversation blocks cue|guide dog passes|mobility cane in view|child points at object|user asks repeated question|ambient alarm with unknown cause'),
 ('street_edge','ood_only','crosswalk approach|curb at signal|bus passing|turning vehicle|blocked sidewalk by street|traffic signal partly hidden|unfamiliar intersection'),
 ('accident_aftermath','ood_only','car crash aftermath visible|broken glass on pavement|fallen street sign|debris from incident|smoke with uncertain source|emergency vehicle arrival|crowd after collision'),
 ('extreme_rare','ood_only','building damage|power failure|flooding in corridor|evacuation alarm|major crowd surge|fire smoke|unrecognized emergency'),
]
rows=[]
for domain,scope,cases in FAMILIES:
 for i,c in enumerate(cases.split('|'),1):
  rows.append({'scenario_id':f'{domain}-{i:02d}','domain':domain,'case':c,'scope':scope,
    'annotation_status':'not_collected','scenario_status':'catalog_only',
    'recommended_role':'OOD/safe-failure evaluation only' if scope=='ood_only' else 'candidate training/evaluation',
    'human_review_required':True})
SIGNS = {
 'directional':['EXIT','ENTRANCE','THIS WAY','LEFT ARROW','RIGHT ARROW','YOU ARE HERE','MAP'],
 'vertical_access':['STAIRS','ELEVATOR','LIFT','RAMP','ESCALATOR','ACCESSIBLE ROUTE','NO STEP ACCESS'],
 'floor_hazard':['CAUTION WET FLOOR','WATCH YOUR STEP','UNEVEN SURFACE','SLIPPERY WHEN WET','LOW CLEARANCE','FALL HAZARD'],
 'restriction':['DO NOT ENTER','AUTHORIZED PERSONNEL','ROAD CLOSED','TEMPORARY CLOSURE','NO PEDESTRIANS','STAFF ONLY'],
 'emergency':['EMERGENCY EXIT','FIRE EXIT','EVACUATION ROUTE','FIRST AID','AED','EMERGENCY ASSEMBLY'],
 'airport':['GATES','GATE A12','DEPARTURES','ARRIVALS','BAGGAGE CLAIM','SECURITY CHECK','PASSPORT CONTROL','TRANSFER','BOARDING','GATE CHANGE'],
 'transit':['PLATFORM','TRACK 2','BUS STOP','TICKET OFFICE','TRAIN DELAYED','SERVICE CHANGE','MIND THE GAP'],
 'personal_services':['RESTROOM','ACCESSIBLE TOILET','INFORMATION','HELP DESK','RECEPTION','CUSTOMER SERVICE','ROOM 204'],
 'street':['STOP','YIELD','PEDESTRIAN CROSSING','NO CROSSING','DETOUR','SCHOOL ZONE','CONSTRUCTION AHEAD'],
 'medical':['EMERGENCY','TRIAGE','PHARMACY','RADIOLOGY','REGISTRATION','WARD','EXIT TO STREET'],
 'retail':['CHECKOUT','QUEUE HERE','PICKUP','CUSTOMER SERVICE','AISLE 5','SALE 50% OFF','CLOSED'],
 'negative_text':['ADVERTISEMENT','BRAND LOGO','DECORATIVE POSTER','MENU SPECIAL','EVENT POSTER','NONACTIONABLE WALL TEXT'],
 'symbol_only':['ACCESSIBILITY PICTOGRAM','EXIT PICTOGRAM','ELEVATOR PICTOGRAM','RESTROOM PICTOGRAM','HAZARD ICON','DIRECTIONAL ARROW'],
}
modifiers={
'light':['bright','dim','night','glare','backlit'],
'camera':['stable','handheld_pan','motion_blur','partial_occlusion','camera_tilt'],
'density':['empty','sparse','moderate','crowded','burst'],
'geometry':['near','far','left','right','center','occluded'],
'speech':['free','busy_short','busy_long','budget_empty','tts_delay'],
'perception':['nominal','depth_noisy','track_loss','ocr_error','unknown_object','camera_unavailable'],
'user_goal':['travel_forward','find_destination','read_sign','seek_assistance'],
'reaction_time':['short','medium','long'],
'language':['english','portuguese','spanish','french','other','icon_only'],
}
sources=[
 ('EgoBlind','https://github.com/doc-doc/EgoBlind','egocentric video and BLV-verified questions','relevance/human-needs; independent QA eval','registration/download terms; no scheduler action labels'),
 ('GuideDog','https://huggingface.co/datasets/kjunh/GuideDog','22,084 egocentric image-description pairs','accessibility guidance text; gold eval; silver exploratory train','gated CC BY-NC 4.0; static images; no deadlines'),
 ('Ego4D','https://ego4d-data.org/docs/start-here/','large egocentric video','diversity, negative scenes, transfer','license/credentials required; varied tasks; no action labels'),
 ('JRDB','https://jrdb.erc.monash.edu.au/dataset/','crowds, tracking, 2D/3D boxes','tracking and multi-human scenes','robot vantage, not BLV egocentric'),
 ('Aria Everyday Activities','https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset','wearable RGB, IMU, 3D pose','geometric robustness and memory','license/download; mostly indoor daily activities'),
 ('TextOCR','https://textvqa.org/textocr/','scene text images and word annotations','OCR for signs','static, not signage relevance'),
 ('HierText','https://github.com/google-research-datasets/hiertext','scene text words/lines/paragraphs','layout and OCR generalization','static, not timing'),
 ('Mapillary Vistas','https://openaccess.thecvf.com/content_iccv_2017/html/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.html','street image segmentation','outdoor sidewalk and traffic visual parsing','not first-person blind-user walking'),
 ('Mapillary traffic signs','https://help.mapillary.com/hc/en-us/articles/4407521157138-Downloading-map-data-via-the-Mapillary-web-app','road sign map data','traffic sign appearance / icon ontology','provider API terms; not all airports/indoor signage'),
 ('DADA-2000','https://github.com/JWFangit/LOTVS-DADA','accident dashcam video','out-of-distribution emergencies only','driving viewpoint; never pedestrian safe-route ground truth'),
 ('Waymo Open Motion','https://github.com/waymo-research/waymo-open-dataset','labeled traffic trajectories','offline motion perturbation study','noncommercial conditions; driving viewpoint'),
 ('User-approved campus recordings','local://approved-campus-recordings','recorded egocentric human walks','main independent event annotation corpus','consent/access/site policies; no public redistribution'),
]
(ROOT/'scenario_catalog.json').write_text(json.dumps({'kind':'coverage_catalog_not_training_examples','version':'1.0','scenarios':rows,'modifiers':modifiers,'sampling_note':'Do not materialize full cartesian product; use pairwise/stratified sampling. Keep OOD only separate from deployment claims.'},indent=2))
(ROOT/'sign_taxonomy.json').write_text(json.dumps({'version':'1.0','categories':SIGNS,'disclaimer':'Open vocabulary OCR/classification targets, not verified human communication labels.'},indent=2))
with (ROOT/'source_registry.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['dataset','official_url','data_type','use','limitations']);w.writerows(sources)
print(f'{len(rows)} catalog-only scenarios across {len(FAMILIES)} domains; {sum(map(len,SIGNS.values()))} sign examples across {len(SIGNS)} types; {len(sources)} sources')

# Voice commands through the real recogniser

Generated 2026-11-12 20:54 by `python scripts/eval_voice.py` (41.6 min; load average 38.6 at the start and 91.8 at the end on 10 cores). Vosk `vosk-model-small-en-us-0.15` (offline, 16 kHz), fed through the live demo's code path: `VoskBackend(source=...)` -> `VoiceListener` (hands-free: the wake phrase is required, a bare "SeeForMe" opens a 5 s window for one command) -> `parse_command`.

Speech: macOS `say` voices Samantha (en_US), Daniel (en_GB), Karen (en_AU), Moira (en_IE), clean and pink 20 dB (pink noise relative to the speech level, over the silence around it too). Phrases: the 62 hands-free command vectors of `tests/data/voice_commands.json`, 30 natural variants written for this eval (`NATURAL` in the script), and as negatives 20 sentences full of "see", "for", "me" (4 near-misses such as "Let's see, for me it's fine") plus the 3 ignore cases of the vectors. A command is right when intent and argument both match; any command from a negative, even a bare wake, is a false wake.

Recogniser modes (`VoskBackend(recogniser=...)`, `seeforme/live/voice.py`):

- **open**: the model's whole vocabulary (what the demo used before).
- **grammar**: `KaldiRecognizer` with a phrase list of the command vocabulary plus `[unk]` (`grammar_phrases()`, built from the parser's patterns and the FIND targets, not from these test phrases; Vosk makes a bigram model of it).
- **hybrid**: the grammar recogniser hears the words; an utterance in which it heard the wake phrase or `[unk]` (a word outside the vocabulary) is decoded again by the open recogniser, which decides whether the wake phrase was said (the grammar, a model of commands that all start with it, forces chatter into "see for me") and whose reading is used for a command with a word the grammar did not know (a FIND target outside the list).
- **snap**: the other way round: the open recogniser; an utterance that does not read as a command ("see for me mood") is decoded again with the grammar, whose reading snaps the near-miss to the command words when it is a command and adds no wake phrase the open reading lacked.

Paired design: each recording is decoded once by `VoskBackend(recogniser="hybrid", second_pass="always")`, which records the grammar's and the open vocabulary's reading of every end-pointed utterance; each mode is scored on those two readings through its own selection (`hybrid_text`, `snap_text`) and a fresh `VoiceListener`, so the modes differ only in how they use the same decodes. Check: the open mode run alone (its own end-pointing) on Samantha's clean speech gave the same command as the paired open reading on 115 of 115 recordings (right alone 102, paired 102).

| mode | commands right | shared vectors | natural variants | false wakes (all) | false wakes, plain chatter | false wakes, near-misses | decode CPU / audio s |
|---|---|---|---|---|---|---|---|
| open | 81.8% (602/736; 79-84%) | 78.6% (390/496; 75-82%) | 88.3% (212/240; 84-92%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.224 |
| grammar | 90.1% (663/736; 88-92%) | 94.6% (469/496; 92-96%) | 80.8% (194/240; 75-85%) | 26.6% (49/184; 21-33%) | 18.0% (23/128; 12-26%) | 81.2% (26/32; 65-91%) | 0.02 |
| hybrid | 87.5% (644/736; 85-90%) | 91.5% (454/496; 89-94%) | 79.2% (190/240; 74-84%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.214 |
| snap (default) | 89.9% (662/736; 88-92%) | 88.7% (440/496; 86-91%) | 92.5% (222/240; 88-95%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.231 |

Commands right by voice and by condition:

| mode | Samantha | Daniel | Karen | Moira | clean | pink 20 dB |
|---|---|---|---|---|---|---|
| open | 84% | 79% | 76% | 89% | 85% | 79% |
| grammar | 89% | 92% | 89% | 91% | 91% | 89% |
| hybrid | 89% | 90% | 80% | 91% | 89% | 86% |
| snap | 91% | 90% | 83% | 97% | 92% | 88% |

**Choice: snap** (now `DEFAULT_RECOGNISER`): the most commands right among the modes with no more false wakes than the open vocabulary, ties to the cheaper decoder. against open: +8.1 points of commands, +0.0 points of false wakes, decode 0.231 vs 0.224 CPU s per audio s; against grammar: -0.2 points of commands, -18.4 points of false wakes, decode 0.231 vs 0.02 CPU s per audio s; against hybrid: +2.4 points of commands, +0.0 points of false wakes, decode 0.231 vs 0.214 CPU s per audio s.

In-sample: the rule chose the mode on these same recordings and none of them is held out (a check on voices the choice was not made on is under Held-out voices at the end), so every figure above is one the choice was made on (and the hybrid's wake rule was revised after a first look at these negatives, see Limits). Paired on the same command recordings, snap is right where open is wrong 60 times and the reverse 0 times (sign test p = 1.7e-18); snap is right where grammar is wrong 53 times and the reverse 54 times (sign test p = 1); snap is right where hybrid is wrong 53 times and the reverse 35 times (sign test p = 0.069). Against grammar and hybrid the difference is within the noise of this set (p > 0.05): no evidence either way on new speech. Against open it is clear. Split by phrase into two fixed halves, the most commands right go to snap and grammar (among the modes with no more false wakes than the open vocabulary: snap and snap).

Negatives whose exact words already wake the parser (any recogniser that hears them right wakes): "Let's see, for me it's fine."; "The sea for me is too cold in winter.".

Parser alone on the texts (no audio): shared vectors 62/62, natural variants 30/30.

## open

- 134 of the 134 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "let's see for me it's fine"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| see for me scene mode | ['mode', 'scene'] | ['ask', 'seen mode'] | see for me seen mode | 6 |
| c for me where is the elevator | ['find', 'elevator'] | ['ask', 'whereas the elevator'] | see for me whereas the elevator | 5 |
| seeforme find door | ['find', 'door'] | None | see form find door | 4 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 4 |
| SeeForMe, where is the nearest exit | ['find', 'exit'] | ['ask', 'whereas the nearest exit'] | see for me whereas the nearest exit | 4 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 3 |
| see for me pause | ['pause', None] | ['ask', 'cause'] | see for me cause | 3 |
| see for me wait | ['pause', None] | ['ask', 'wage'] | see for me wage | 3 |
| see for me find room oh four | ['find', 'room 04'] | ['find', 'room oh for'] | see for me find room oh for | 3 |
| SeeForMe ready. Say 'SeeForMe, help' for commands. | ['help', None] | ['ask', 'held for commands'] | see for me ready say see for me held for commands | 3 |
| See for me, describe the room | ['describe', None] | ['ask', 'described the room'] | see for me described the room | 3 |
| See for me, stop looking for the door | ['stop_find', None] | ['find', 'door'] | see for me start looking for the door | 3 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| see for me what's around me | ['describe', None] | ['ask', 'once around me'] | see for me once around me | 2 |
| see for me what's around me | ['describe', None] | ['ask', 'walks around me'] | see for me walks around me | 2 |

## grammar

- 73 of the 73 wrong commands were parsed right as text: recognition errors.
- False wakes: "I saw four meals on the table." heard as "i see for me is on the table"; "Is this coffee for me or for you?" heard as "is there car see for me a for you"; "Is this coffee for me or for you?" heard as "is this car see for me a for you"; "Is this coffee for me or for you?" heard as "is this car see for me auto you"; "Is this coffee for me or for you?" heard as "is this car see for me for for you"; "Is this coffee for me or for you?" heard as "is this car see for me on for you"; "Is this coffee for me or for you?" heard as "is this for see for me auto you"; "Let me see, four men walked past the door." heard as "knife me see for me what path the door"; "Let me see, four men walked past the door." heard as "left me see for me walk path to the door"; "Let me see, four men walked past the door." heard as "left me see for me wall the door"; "Let me see, four men walked past the door." heard as "left me see for me what path to the door"; "Let me see, four men walked past the door." heard as "lift me see for me what path the door"; "Let me see, four men walked past the door." heard as "light me see for me walk path the door"; "Let me see, four men walked past the door." heard as "look me see for me walk path the door"; "Let me see, four men walked past the door." heard as "look me see for me walk the path the door"; "Let's see, for me it's fine." heard as "left see for me is find"; "Let's see, for me it's fine." heard as "less see for me is find"; "Let's see, for me it's fine." heard as "lift see for me a sign"; "Let's see, for me it's fine." heard as "lift see for me is find"; "Let's see, for me it's fine." heard as "light see for me a find".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| See for me, give me more details | ['verbosity', 'detailed'] | ['ask', 'get me more details'] | see for me get me more details | 8 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'i looking for the elevator'] | see for me i looking for the elevator | 7 |
| See for me, hold on a second | ['pause', None] | ['ask', 'hold on is there can'] | see for me hold on is there can | 6 |
| See for me, what do you see now | ['describe', None] | ['ask', 'what do you see know'] | see for me what do you see know | 5 |
| See for me, say that again please | ['repeat', None] | ['ask', 'say that again keys'] | see for me say that again keys | 5 |
| seeforme find door | ['find', 'door'] | None | see for find door | 4 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 4 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'flip find a chair'] | see for me flip find a chair | 4 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 un for'] | see for me find room two hundred un for | 3 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 2 |
| see four me look for a seat | ['find', 'seat'] | ['ask', 'look there a seat'] | see for me look there a seat | 2 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| see for me what do you see | ['describe', None] | ['ask', 'where do you see'] | see for me where do you see | 2 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'keys find a chair'] | see for me keys find a chair | 2 |
| See for me, what do you see now | ['describe', None] | ['ask', 'where do you see know'] | see for me where do you see know | 2 |

## hybrid

- 92 of the 92 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "left see for me is find"; "Let's see, for me it's fine." heard as "less see for me is find"; "Let's see, for me it's fine." heard as "lift see for me a sign"; "Let's see, for me it's fine." heard as "lift see for me is find"; "Let's see, for me it's fine." heard as "light see for me a find"; "The sea for me is too cold in winter." heard as "the see for me is to cow the window"; "The sea for me is too cold in winter." heard as "the see for me is to hold in window"; "The sea for me is too cold in winter." heard as "the see for me is to pole the window"; "The sea for me is too cold in winter." heard as "the see for me is to the window".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| See for me, give me more details | ['verbosity', 'detailed'] | ['ask', 'get me more details'] | see for me get me more details | 8 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'i looking for the elevator'] | see for me i looking for the elevator | 7 |
| See for me, hold on a second | ['pause', None] | ['ask', 'hold on is there can'] | see for me hold on is there can | 6 |
| See for me, what do you see now | ['describe', None] | ['ask', 'what do you see know'] | see for me what do you see know | 5 |
| See for me, say that again please | ['repeat', None] | ['ask', 'say that again keys'] | see for me say that again keys | 5 |
| seeforme find door | ['find', 'door'] | None | see for find door | 4 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 4 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'flip find a chair'] | see for me flip find a chair | 4 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 un for'] | see for me find room two hundred un for | 3 |
| seeforme find door | ['find', 'door'] | None | see form find door | 2 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 2 |
| see four me look for a seat | ['find', 'seat'] | ['ask', 'look there a seat'] | see for me look there a seat | 2 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| see for me what do you see | ['describe', None] | ['ask', 'where do you see'] | see for me where do you see | 2 |
| see for me normal detail | ['verbosity', 'balanced'] | None | see them a normal detail | 2 |

## snap

- 74 of the 74 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "let's see for me it's fine"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| seeforme find door | ['find', 'door'] | None | see form find door | 4 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 3 |
| see for me find room oh four | ['find', 'room 04'] | ['find', 'room oh for'] | see for me find room oh for | 3 |
| See for me, stop looking for the door | ['stop_find', None] | ['find', 'door'] | see for me start looking for the door | 3 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 2 |
| see for me normal detail | ['verbosity', 'balanced'] | None | see them a normal detail | 2 |
| see for me is the path clear | ['ask', 'is the path clear'] | ['ask', 'as the path clear'] | see for me as the path clear | 2 |
| see for me what's on my left | ['ask', "what's on my left"] | ['ask', 'walks on my left'] | see for me walks on my left | 2 |
| see for me find room oh four | ['find', 'room 04'] | ['find', 'bruno for'] | see for me find bruno for | 2 |
| See for me, take me to room two oh five | ['find', 'room 205'] | ['find', 'rome 205'] | see for me take me to rome two oh five | 2 |
| See for me, take me to room two oh five | ['find', 'room 205'] | ['find', 'room to or 5'] | see for me take me to room to or five | 2 |
| See for me, find the bus stop | ['find', 'bus stop'] | ['find', 'best stop'] | see for me find the best stop | 2 |
| seeforme find door | ['find', 'door'] | None | si va va door | 1 |
| sea for me find a chair | ['find', 'chair'] | ['find', 'share'] | see for me find a share | 1 |

## Limits

Synthesised voices are clear, steady and free of accent drift, reverberation and street noise; the pink noise is a mild stand-in for a room. A user's own voice outdoors will do worse in every mode, probably more for the open vocabulary. The phrases are short commands; long questions ("is there a bench nearby") lean on the open recogniser in hybrid mode. The grammar was written from the parser's patterns before the eval ran. One design change followed a first look at these negatives (grammar alone woke on 11 of 46 utterances of two voices; the open reading of the same 11 woke on 4, all near-misses whose words contain the wake phrase): the hybrid now lets the open recogniser decide whether the wake phrase was said. The snap mode and its rule existed before that look. No threshold or phrase list was fitted to the negatives. Decode CPU is process time on a shared machine; the open vocabulary's share is what the live demo pays per second of speech.

## Held-out voices

Held out: the default mode (snap) was chosen on Samantha, Daniel, Karen, Moira. This check uses Rishi (en_IN), Tessa (en_ZA), Fred (en_US), voices the choice was not made on; nothing was chosen or tuned on them. The phrases, the conditions (clean and pink 20 dB) and the code path are the same as above, so the voices are held out and the phrases are not. Generated 2026-11-12 20:51 (31.7 min) by `OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_voice.py --voices Rishi,Tessa,Fred --open-check '' --out /private/tmp/claude-501/-Users-allahurodrigues-Desktop-Desktop-My-Desk-PROJECT-Rodrigues-portfolio/1ed32f53-0651-4db2-a84d-91be38c2e8c0/scratchpad/voice_r3/eval/heldout --work /private/tmp/claude-501/-Users-allahurodrigues-Desktop-Desktop-My-Desk-PROJECT-Rodrigues-portfolio/1ed32f53-0651-4db2-a84d-91be38c2e8c0/scratchpad/voice_r3/eval/heldout_wavs`, then added here by `.venv/bin/python scripts/eval_voice.py --report-only --attach-held-out <out>/voice_eval.json`.

| mode | commands right | shared vectors | natural variants | false wakes (all) | false wakes, near-misses |
|---|---|---|---|---|---|
| open | 72.3% (399/552; 68-76%) | 69.1% (257/372; 64-74%) | 78.9% (142/180; 72-84%) | 8.0% (11/138; 4-14%) | 45.8% (11/24; 28-65%) |
| grammar | 85.9% (474/552; 83-88%) | 91.9% (342/372; 89-94%) | 73.3% (132/180; 66-79%) | 24.6% (34/138; 18-32%) | 79.2% (19/24; 60-91%) |
| hybrid | 85.3% (471/552; 82-88%) | 91.1% (339/372; 88-94%) | 73.3% (132/180; 66-79%) | 8.0% (11/138; 4-14%) | 45.8% (11/24; 28-65%) |
| snap (default) | 86.1% (475/552; 83-89%) | 84.4% (314/372; 80-88%) | 89.4% (161/180; 84-93%) | 8.0% (11/138; 4-14%) | 45.8% (11/24; 28-65%) |

| mode | Rishi | Tessa | Fred | clean | pink 20 dB |
|---|---|---|---|---|---|
| open | 75% | 81% | 61% | 74% | 70% |
| grammar | 85% | 92% | 80% | 87% | 85% |
| hybrid | 85% | 91% | 80% | 86% | 85% |
| snap | 85% | 94% | 79% | 87% | 85% |

Paired on the same command recordings, snap is right where open is wrong 76 times and the reverse 0 times (sign test p = 2.6e-23); snap is right where grammar is wrong 43 times and the reverse 42 times (sign test p = 1); snap is right where hybrid is wrong 43 times and the reverse 39 times (sign test p = 0.74). On the negatives, snap stays silent where open wakes 0 times and the reverse 0 times (p = 1); snap stays silent where grammar wakes 23 times and the reverse 0 times (p = 2.4e-07); snap stays silent where hybrid wakes 0 times and the reverse 0 times (p = 1).

Against the 4 selection voices above, snap gets -3.8 points of commands right and -0.2 points of false wakes on the held-out voices; the selection voices' rate of commands right (90%) is above the held-out 95% interval, and their false-wake rate (8%) is within it. Commands right on the held-out voices against the selection voices, per mode: open -9.5, grammar -4.2, hybrid -2.2, snap -3.8 points. The selection rule applied to the held-out voices alone would pick snap (reported only; the default was not chosen again).

Note added by hand after this section was generated (2026-11-12): `hybrid_text` later gained the same `may_refine` guard as snap (a question is not turned into a command). Re-scoring the stored held-out readings with that parser changes 2 recordings, both hybrid, both Fred with pink noise ("see for me normal detail" heard as "see for me normal more detail", "see for me switch camera" heard as "see for me is lit camera"): hybrid gets 85.0% (469/552; 82-88%), shared vectors 337/372, Fred 79%, pink 20 dB 84%, and snap against hybrid is 43 to 37 (p = 0.58). Snap, open and grammar are unchanged, as is every figure in the main sections above (re-scoring all 920 in-sample recordings changes no outcome). A new held-out run regenerates this section without the note.

# Voice commands through the real recogniser

Generated 2026-11-12 11:13 by `python scripts/eval_voice.py` (22.8 min; load average 38.6 at the start and 23.5 at the end on 10 cores). Vosk `vosk-model-small-en-us-0.15` (offline, 16 kHz), fed through the live demo's code path: `VoskBackend(source=...)` -> `VoiceListener` (hands-free: the wake phrase is required, a bare "SeeForMe" opens a 5 s window for one command) -> `parse_command`.

Speech: macOS `say` voices Samantha (en_US), Daniel (en_GB), Karen (en_AU), Moira (en_IE), clean and pink 20 dB (pink noise relative to the speech level, over the silence around it too). Phrases: the 62 hands-free command vectors of `tests/data/voice_commands.json`, 30 natural variants written for this eval (`NATURAL` in the script), and as negatives 20 sentences full of "see", "for", "me" (4 near-misses such as "Let's see, for me it's fine") plus the 3 ignore cases of the vectors. A command is right when intent and argument both match; any command from a negative, even a bare wake, is a false wake.

Recogniser modes (`VoskBackend(recogniser=...)`, `seeforme/live/voice.py`):

- **open**: the model's whole vocabulary (what the demo used before).
- **grammar**: `KaldiRecognizer` with a phrase list of the command vocabulary plus `[unk]` (`grammar_phrases()`, built from the parser's patterns and the FIND targets, not from these test phrases; Vosk makes a bigram model of it).
- **hybrid**: the grammar recogniser hears the words; an utterance in which it heard the wake phrase or `[unk]` (a word outside the vocabulary) is decoded again by the open recogniser, which decides whether the wake phrase was said (the grammar, a model of commands that all start with it, forces chatter into "see for me") and whose reading is used for a command with a word the grammar did not know (a FIND target outside the list).
- **snap**: the other way round: the open recogniser; an utterance that does not read as a command ("see for me mood") is decoded again with the grammar, whose reading snaps the near-miss to the command words when it is a command and adds no wake phrase the open reading lacked.

Paired design: each recording is decoded once by `VoskBackend(recogniser="hybrid", second_pass="always")`, which records the grammar's and the open vocabulary's reading of every end-pointed utterance; each mode is scored on those two readings through its own selection (`hybrid_text`, `snap_text`) and a fresh `VoiceListener`, so the modes differ only in how they use the same decodes. Check: the open mode run alone (its own end-pointing) on Samantha's clean speech gave the same command as the paired open reading on 115 of 115 recordings (right alone 102, paired 102).

| mode | commands right | shared vectors | natural variants | false wakes (all) | false wakes, plain chatter | false wakes, near-misses | decode CPU / audio s |
|---|---|---|---|---|---|---|---|
| open | 81.8% (602/736; 79-84%) | 78.6% (390/496; 75-82%) | 88.3% (212/240; 84-92%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.188 |
| grammar | 90.8% (668/736; 88-93%) | 95.6% (474/496; 93-97%) | 80.8% (194/240; 75-85%) | 25.0% (46/184; 19-32%) | 18.8% (24/128; 13-26%) | 68.8% (22/32; 51-82%) | 0.016 |
| hybrid | 88.2% (649/736; 86-90%) | 92.5% (459/496; 90-94%) | 79.2% (190/240; 74-84%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.182 |
| snap (default) | 89.9% (662/736; 88-92%) | 88.7% (440/496; 86-91%) | 92.5% (222/240; 88-95%) | 8.2% (15/184; 5-13%) | 0.0% (0/128; 0-3%) | 46.9% (15/32; 31-64%) | 0.193 |

Commands right by voice and by condition:

| mode | Samantha | Daniel | Karen | Moira | clean | pink 20 dB |
|---|---|---|---|---|---|---|
| open | 84% | 79% | 76% | 89% | 85% | 79% |
| grammar | 90% | 92% | 90% | 91% | 91% | 90% |
| hybrid | 90% | 90% | 82% | 91% | 89% | 87% |
| snap | 91% | 90% | 83% | 97% | 92% | 88% |

**Choice: snap** (now `DEFAULT_RECOGNISER`): the most commands right among the modes with no more false wakes than the open vocabulary, ties to the cheaper decoder. against open: +8.1 points of commands, +0.0 points of false wakes, decode 0.193 vs 0.188 CPU s per audio s; against grammar: -0.9 points of commands, -16.8 points of false wakes, decode 0.193 vs 0.016 CPU s per audio s; against hybrid: +1.7 points of commands, +0.0 points of false wakes, decode 0.193 vs 0.182 CPU s per audio s.

In-sample: the rule chose the mode on these same recordings and there is no held-out set, so every figure above is one the choice was made on (and the hybrid's wake rule was revised after a first look at these negatives, see Limits). Paired on the same command recordings, snap is right where open is wrong 60 times and the reverse 0 times (sign test p = 1.7e-18); snap is right where grammar is wrong 50 times and the reverse 56 times (sign test p = 0.63); snap is right where hybrid is wrong 50 times and the reverse 37 times (sign test p = 0.2). Against grammar and hybrid the difference is within the noise of this set (p > 0.05): no evidence either way on new speech. Against open it is clear. Split by phrase into two fixed halves, the most commands right go to snap and grammar (among the modes with no more false wakes than the open vocabulary: snap and snap).

Negatives whose exact words already wake the parser (any recogniser that hears them right wakes): "Let's see, for me it's fine."; "The sea for me is too cold in winter.".

Parser alone on the texts (no audio): shared vectors 62/62, natural variants 30/30.

## open

- 134 of the 134 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "let's see for me it's fine"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter / "; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter / ".

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

- 68 of the 68 wrong commands were parsed right as text: recognition errors.
- False wakes: "I can see the station from here." heard as "i can see the see for me hair"; "I saw four meals on the table." heard as "i see for me is on the table"; "Is this coffee for me or for you?" heard as "is there car see for me where for you"; "Is this coffee for me or for you?" heard as "is this car see for me auto you"; "Is this coffee for me or for you?" heard as "is this car see for me for for you"; "Is this coffee for me or for you?" heard as "is this car see for me for you"; "Is this coffee for me or for you?" heard as "is this car see for me on for you"; "Is this coffee for me or for you?" heard as "is this car see for me where for you"; "Is this coffee for me or for you?" heard as "is this for see for me auto you"; "Let me see, four men walked past the door." heard as "knife me see for me what path the door"; "Let me see, four men walked past the door." heard as "left me see for me walk path to the door"; "Let me see, four men walked past the door." heard as "left me see for me wall the door"; "Let me see, four men walked past the door." heard as "left me see for me what path to the door"; "Let me see, four men walked past the door." heard as "lift me see for me what path the door"; "Let me see, four men walked past the door." heard as "light me see for me walk path the door"; "Let me see, four men walked past the door." heard as "look me see for me walk path the door"; "Let me see, four men walked past the door." heard as "look me see for me walk the path the door"; "Let's see, for me it's fine." heard as "left see for me is find"; "Let's see, for me it's fine." heard as "less see for me is find"; "Let's see, for me it's fine." heard as "lift see for me is find".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| See for me, what do you see now | ['describe', None] | ['ask', 'what do you see mouse'] | see for me what do you see mouse | 7 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'flip find a chair'] | see for me flip find a chair | 5 |
| See for me, give me more details | ['verbosity', 'detailed'] | ['ask', 'glove me more details'] | see for me glove me more details | 5 |
| seeforme find door | ['find', 'door'] | None | see for find door | 4 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 4 |
| See for me, say that again please | ['repeat', None] | ['ask', 'say that again keys'] | see for me say that again keys | 4 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 un for'] | see for me find room two hundred un for | 3 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'on looking for the elevator'] | see for me on looking for the elevator | 3 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 2 |
| see four me look for a seat | ['find', 'seat'] | ['ask', 'look there a seat'] | see for me look there a seat | 2 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'keys find a chair'] | see for me keys find a chair | 2 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'am looking for the elevator'] | see for me am looking for the elevator | 2 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'i looking for the elevator'] | see for me i looking for the elevator | 2 |
| See for me, hold on a second | ['pause', None] | ['ask', 'hold on is there find'] | see for me hold on is there find | 2 |

## hybrid

- 87 of the 87 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "left see for me is find"; "Let's see, for me it's fine." heard as "less see for me is find"; "Let's see, for me it's fine." heard as "lift see for me is find"; "Let's see, for me it's fine." heard as "light see for me find"; "The sea for me is too cold in winter." heard as "the see for me is to / chair"; "The sea for me is too cold in winter." heard as "the see for me is to / window"; "The sea for me is too cold in winter." heard as "the see for me is to cow the window"; "The sea for me is too cold in winter." heard as "the see for me is to hold in window"; "The sea for me is too cold in winter." heard as "the see for me is to the window".

Most frequent errors (all voices and conditions):

| said | wanted | got | heard | times |
|---|---|---|---|---|
| See for me, what do you see now | ['describe', None] | ['ask', 'what do you see mouse'] | see for me what do you see mouse | 6 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'flip find a chair'] | see for me flip find a chair | 5 |
| See for me, give me more details | ['verbosity', 'detailed'] | ['ask', 'glove me more details'] | see for me glove me more details | 5 |
| seeforme find door | ['find', 'door'] | None | see for find door | 4 |
| see for me unmute | ['unmute', None] | ['ask', 'on mute'] | see for me on mute | 4 |
| See for me, say that again please | ['repeat', None] | ['ask', 'say that again keys'] | see for me say that again keys | 4 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 un for'] | see for me find room two hundred un for | 3 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'on looking for the elevator'] | see for me on looking for the elevator | 3 |
| seeforme find door | ['find', 'door'] | None | see form find door | 2 |
| seeforme find door | ['find', 'door'] | None | see phone find door | 2 |
| see four me look for a seat | ['find', 'seat'] | ['ask', 'look there a seat'] | see for me look there a seat | 2 |
| see for me find room 204 | ['find', 'room 204'] | ['find', 'room 200 for'] | see for me find room two hundred for | 2 |
| see for me normal detail | ['verbosity', 'balanced'] | None | see them a normal detail | 2 |
| See for me, please find a chair | ['find', 'chair'] | ['ask', 'keys find a chair'] | see for me keys find a chair | 2 |
| See for me, I'm looking for the elevator | ['find', 'elevator'] | ['ask', 'am looking for the elevator'] | see for me am looking for the elevator | 2 |

## snap

- 74 of the 74 wrong commands were parsed right as text: recognition errors.
- False wakes: "Let's see, for me it's fine." heard as "let's see for me it's fine"; "Let's see, for me it's fine." heard as "light see for me find"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me as too cold in winter / "; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter"; "The sea for me is too cold in winter." heard as "the see for me is too cold in winter / ".

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

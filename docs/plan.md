# Plan

## Overview

- Weeks 7-9 build the system, so a full MVP journey works end to end by the end of Week 9. Weeks 10-11 run the experiments. Week 12 fixes what the experiments show. Week 13 is the report, the slides and a buffer.
- Rule for the schedule: **every week ends with something we can run and show**, so a delay is visible after one week, not at the end.
- Dates beyond Week 7 are left out on purpose, because holidays or breaks may shift the calendar. Week numbers are the course weeks.

## Roles breakdown


| Role | Owns |
|---|---|
| **A: Data and RAG** | Ingestion script, knowledge and listings collections, chunking, embeddings, `search_knowledge`, `suggest_names` |
| **B: Vision** | Breed classifier, confidence calibration, `identify_breed`, `describe_pet` |
| **C: Agent and MCP** | MCP server, agent loop, state, guards, `get_breed_traits`, `recommend_breeds` |
| **D: Evaluation** | Golden dataset (Parts A-F), metrics script, experiment runs |
| **E: Integration** | Chat UI, GitHub repo and README, Milestone 2 document, demo, final report |

The person who builds a step does not write its golden questions (rule from the Golden dataset section). D coordinates this and asks a teammate from another role to write each part.

## Week by week

| Week | Goal | Tasks | Owner | End of week: what we can show |
|---|---|---|---|---|
| **7** | Foundations | Upload Milestone 2 document (**E**). Download and clean data, write the ingestion script, build the knowledge collection with section chunks (**A**). Fine-tune the first ConvNeXt-Tiny on Stanford Dogs (**B**). MCP server skeleton and agent loop talking to Qwen3-4B in LM Studio (**C**). Write golden Parts A-C (**D**) | All | Ingestion script runs on the real data. MCP Inspector can call one dummy tool. First classifier accuracy number |
| **8** | Tools for steps 1-3 | `search_knowledge` basic RAG with breed filter (**A**). Temperature scaling and confidence threshold tuned on Part A, `identify_breed` as a tool (**B**). `get_breed_traits` and `recommend_breeds` with the AKC rules, vet pattern check (**C**). Golden Part D (fit profiles) (**D**). Chat UI shell with photo upload (**E**) | All | A chat that identifies a breed from a photo, answers a breed question with citations, and recommends 3 breeds |
| **9** | Step 4 and full MVP | Listings collection, `suggest_names` (**A**). `describe_pet` with Qwen2.5-VL-3B (**B**). Citation guard, low-confidence reply, limits, retry logic (**C**). Golden Parts E and F (**D**). Wire everything in the UI (**E**) | All | **MVP checkpoint: the full journey from the diagram works end to end.** Golden dataset complete |
| **10** | Experiments: RAG and vision | RAG: section chunks vs 90-word windows; nomic vs bge-small vs BM25; basic vs corrective vs agentic RAG, on Part B and C (**A**, **D**). Vision: ConvNeXt-Tiny vs EfficientNet-B0 vs ViT-B/16 vs CLIP zero-shot on Part A; Qwen2.5-VL-3B vs SmolVLM vs colour heuristic (**B**). Collect the small set of photos from other sources to check the leakage risk (**B**) | A, B, D | Results tables for retrieval and vision |
| **11** | Experiments: agent | SLM comparison: Qwen3-4B vs Llama-3.2-3B vs Phi-4-mini vs a ~1.7B model on Parts B, C, F (**C**, **D**). SLM tool calling vs rule-based router (**C**). Short tool descriptions vs descriptions with an example call (**C**). Human rating of names, 2 blind teammates (**D**, **E**) | C, D, E | Results tables for the agent and for names. Final model choice |
| **12** | Fix and freeze | Fix what the experiments showed (thresholds, prompts, flaky tools). Freeze the final configuration. One last run of the full golden set. Error analysis of the worst failures (**All**) | All | **Code freeze.** Final numbers on the golden dataset |
| **13** | Report and rehearsal | Final report (results, limits, what we would do next), clean README and repo, slides, demo rehearsal with a backup recording (**E** leads, all contribute) | E, all | Slides and report ready. Buffer for anything that slipped |
| **14** | Final presentation | Present the architecture, the golden results, the experiment findings and a live demo | All | Progress final presentation |

## Experiments checklist

All experiments use the same golden questions and change one thing at a time.

| # | Experiment | Compared | Metric (golden part) | Week |
|---|---|---|---|---|
| 1 | Chunking | Section chunks vs 90-word windows | Hit rate at 1 and 3 (B) | 10 |
| 2 | Embeddings | nomic-embed-text-v1.5 vs bge-small-en-v1.5 vs BM25 | Hit rate at 1 and 3 (B) | 10 |
| 3 | RAG approach | Basic vs corrective vs agentic | Correct facts, valid citations, correct "I don't know" (B, C), SLM calls and seconds | 10 |
| 4 | Breed classifier | ConvNeXt-Tiny vs EfficientNet-B0 vs ViT-B/16 vs CLIP zero-shot | Top-1, top-3, "not sure" rate (A) | 10 |
| 5 | Pet description | Qwen2.5-VL-3B vs SmolVLM vs classifier + colour heuristic | Name rating (E), speed | 10-11 |
| 6 | Agent SLM | Qwen3-4B vs Llama-3.2-3B vs Phi-4-mini vs ~1.7B | Invalid tool calls, first-tool accuracy, journeys completed (F), declines (C), seconds | 11 |
| 7 | Tool choice | SLM tool calling vs rule-based router | Journeys completed (F) | 11 |
| 8 | Tool descriptions | Short vs with example call | Wrong-tool rate (F) | 11 |

## Risks and how we keep the deadline

| Risk | What we do |
|---|---|
| The MVP is not ready at the end of Week 9 | Cut scope in this order: (1) skip the name rating by humans and keep only the real-name hit rate, (2) use the colour heuristic instead of the VLM, (3) use the rule-based router instead of the SLM for tool choice. The fallbacks from Model choices apply |
| Fine-tuning does not fit on our laptops | Train on a free cloud GPU, or train only a linear head on the pre-trained backbone, as planned in Model choices |
| Experiments take longer than a week | Run the models in the background while writing the report. Experiments 5 and 8 are the first to be dropped |
| One person is stuck or unavailable | Weekly 15-minute sync; the owner of a step says on Monday if it is at risk. Pull requests are reviewed within 2 days |
| Last-minute demo failure | Record a backup video of the full journey in Week 13 |

## Working agreements

- **Tool names and schemas are fixed after Week 7**, so the Data, Vision and Agent parts can be built in parallel (the contract from the MCP server section).
- **Every change that touches results is logged** with the golden-set version it was run on, so the numbers in the report are reproducible.

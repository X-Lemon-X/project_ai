# Inbox

Drop any source material here: papers, lecture slides, PDFs, datasheets, forum
threads saved as HTML, screenshots, meeting notes, exported chats.

Files in this folder are **ignored by git** (see `.gitignore`). Only the
knowledge extracted from them is committed, in `docs/`.

## Workflow

1. Put the file in `inbox/`. A descriptive name helps, e.g.
   `2026-10-26_isaac-sim-replicator-docs.pdf`.
2. Extract what matters for the project into the knowledge base:
   - facts about tools, sensors, APIs → `docs/knowledge/`
   - ideas that change what we build → a new or updated file in `docs/proposals/`
   - unresolved points → `docs/open-questions.md`
3. Add one row to `docs/sources.md`: file name, what it is, where its content went.
   If the source is public, put its URL there so others can get it.
4. Leave the file here or delete it. Nothing in `docs/` should depend on the
   original being present.

When asking Claude to process the inbox, say e.g. *"process the inbox"*. It
follows the same steps.

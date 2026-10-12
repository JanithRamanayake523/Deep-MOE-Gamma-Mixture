# Reviewer and Researcher Comments

This folder keeps the project's work logs and the feedback from the three people on the team.

## Folder layout

```
comments/
├── README.md              ← this guide
├── TRACKER.md             ← master list of every comment and its status
├── work logs/             ← researcher's work logs (what was done, when, and why)
├── expert/                ← expert's reviews and suggestions
└── critic/                ← critic's reviews and listed mistakes
```

## Who writes where

| Folder | Who | What goes in it |
|---|---|---|
| `work logs/` | Researcher (the master's student) | Dated logs of the work done, decisions made, and open questions |
| `expert/` | Expert | Reviews, suggested changes, and recommendations |
| `critic/` | Critic | Critiques and the mistakes they found |

## Workflow

1. **Save** each file unedited in the folder of the person who wrote it. Name it with the round, the author, and the date, for example `round1_expert_2026-10-12.md`.
2. **Log** each comment in `TRACKER.md` with an ID such as `R1-E-03` (round 1, expert, comment 3). Use `E` for the expert, `C` for the critic, and `W` for the researcher's work log.
3. **Respond** in the researcher's notes or in `TRACKER.md`, and refer to comment IDs.
4. **Update** the status in `TRACKER.md`.

## Status values

| Status | Meaning |
|---|---|
| Open | Received, not yet addressed |
| In progress | Being worked on |
| Addressed | Change made in the manuscript or code |
| Rebutted | Disagreed; reason given in the response |
| Deferred | Valid, but out of scope for this round (state why) |

## Tips

- Keep the original files unchanged so the record stays accurate.
- Name the file and section that changed whenever a comment leads to an edit.
- Commit after each round so the history shows when each comment was addressed.

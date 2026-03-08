# Pac-Man Multi-Agent AI — Project Archive

Curated files and documentation from a competitive multi-agent Pac-Man AI project
built for a class tournament (CSE 140). This repo is not a fully runnable codebase —
it's an archive of selected source files, development artifacts, and media that
document the work we did and how we did it.

---

## Project Overview

The goal was to build a two-agent Pac-Man team to compete in a class tournament
against peer and baseline teams. Rather than committing to a static offense/defense
split, our initial design philosophy was a **hybrid role-switching approach** —
both agents capable of dynamically swapping roles based on game state, so the team
would always maintain both offensive pressure and defensive coverage simultaneously.

The final team combined a trained **Approximate Q-Learning offensive agent** with a
**heuristic-based defensive agent**, and outperformed the majority of baseline and
peer teams despite a tournament meta that heavily favored dual-offense strategies.

---

## The Agents

### Offensive Agent
The offensive agent pairs **Approximate Q-Learning** with **A\* pathfinding**. A\*
handles navigation when the path is clear; Q-learning takes over when enemies are
nearby, handling the harder problem of dodging ghosts while collecting food and
capsules.

Key features used in the Q-learning model:
`distanceToFood`, `distanceToGhost`, `foodDensity`, `scaredGhostProximity`,
`powerCapsuleDistance`, `ghostScaredTime`, `avoidDeadEnds`, `stop`

The reward structure penalized death heavily (-5), rewarded food collection (+1),
and imposed a small per-step penalty (-0.01) to encourage efficient movement.

**Training** was a significant engineering effort in its own right. Early attempts
saw weight values explode (e.g. `foodDistance` jumping from `1.0` to `10000.0`
within a few episodes). We solved this by implementing weight sanitization, feature
normalization, and a **three-phase decay schedule** for the learning parameters α
and ε — moving from high exploration in phase one, toward convergence in phase two,
and fine-tuning in phase three. Training was run across randomized maps to ensure
generalization, ultimately achieving a **100% win rate against the baseline team**
on any map.

To scale training, we deployed **multiple parallel instances on Oracle Cloud
Infrastructure (OCI)**, dramatically reducing the time needed to run enough episodes
for policy convergence.

### Defensive Agent
The defensive agent is heuristic and reflex-based — simpler in code complexity but
carefully tuned. It prioritizes chasing visible invaders, zones around power pellets
and high-value food when no invaders are visible, and avoids crossing into enemy
territory to prevent leaving our side exposed.

Early versions were too easily baited across the midline or bypassed via power
pellets. The final version significantly reduced enemy scores through better
positioning and power pellet control, though it remained vulnerable to fast hybrid
or dual-offense teams — a noted area for future improvement via RL.

---

## What's in This Repo *(PLACEHOLDER)*

| File / Folder | Description |
|---|---|
| `report.pdf` | Final project report submitted at end of term |
| `agents/` | Source files for offensive and defensive agents |
| `training/` | Training script and weight/parameter save logic |
| `weights/` | Saved weight files from training runs |
| `media/` | Gameplay gifs showing agents in action |

---

## Media

### Offensive Agent in Action
![Offensive agent gameplay](media/offense.gif)

### Defensive Agent in Action
![Defensive agent gameplay](media/defense.gif)

---

## Results

- **100% win rate** against the baseline team across all maps after full training
- Outperformed the majority of peer teams in the class tournament
- Achieved competitive results despite a meta that heavily favored dual-offense
  strategies, which our defensive agent was most vulnerable to

---

## Team

Built in collaboration with a partner for CSE 140 at UC Santa Cruz.
My primary contributions were the defensive agent, A\* pathfinding implementation,
training infrastructure, and performance tracking/visualization.
